# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple

import torch
import torch.distributed as dist
import torch_npu

from vllm.config import CUDAGraphMode
from vllm.distributed import (
    get_ep_group,
    get_dp_group,
    get_tp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.layers.utils import named_stream
from omni_npu.v1.utils import on_ascend950

logger = init_logger(__name__)

DEFAULT_MAX_DISPATCH_COMBINE_THRESHOLD = 64
GROUPED_FINALIZE_ROW_INDEX_STRIDE = 991


@dataclass
class PreparePermuteResult:
    hidden_states_sorted_by_experts: torch.Tensor
    expert_tokens: torch.Tensor
    dynamic_scale: Optional[torch.Tensor]
    avg_tokens_per_expert: Optional[torch.Tensor] = None


@dataclass
class All2AllPreparePermuteResult(PreparePermuteResult):
    input_splits: List[int] = field(default_factory=list)
    output_splits: List[int] = field(default_factory=list)
    expanded_x: Optional[torch.Tensor] = None
    expanded_row_idx: Optional[torch.Tensor] = None
    gathered_idxs_unsort: Optional[torch.Tensor] = None


@dataclass
class DispatchCombinePreparePermuteResult(PreparePermuteResult):
    tp_recv_counts: Optional[torch.Tensor] = None
    ep_recv_counts: Optional[torch.Tensor] = None
    expand_idx: Optional[torch.Tensor] = None


@dataclass
class AGRSPreparePermuteResult(PreparePermuteResult):
    expert_range: Optional[List[int]] = None
    expanded_row_idx: Optional[torch.Tensor] = None
    gathered_topk_ids: Optional[torch.Tensor] = None
    dtype: Optional[torch.dtype] = None
    row_idx_type: Optional[int] = 0
    shared_expert_gate_up: Optional[torch.Tensor] = None
    shared_expert_gate_up_proj_finished_event: Optional[torch.npu.Event] = None


@dataclass
class AGRSFinalizeParams:
    """Pre-computed params for grouped_matmul_finalize_routing."""
    expanded_row_idx: torch.Tensor
    row_index: torch.Tensor
    batch_size: int
    w2_scale: torch.Tensor
    w2_bias: Optional[torch.Tensor]


@dataclass
class AGRSFinalizeMetadata:
    gathered_topk_weights: torch.Tensor


class FusedMoEPreparePermuteAndUnpermuteFinalize(ABC):
    def __init__(self, layer):
        self.num_experts = layer.global_num_experts
        self.ep_size = get_ep_group().world_size
        self.ep_rank = get_ep_group().rank_in_group

    @abstractmethod
    def prepare_permute(
        self, layer: torch.nn.Module, x: torch.Tensor, topk_ids: torch.Tensor
    ) -> PreparePermuteResult:
        raise NotImplementedError

    def prepare_finalize_metadata(
        self,
        layer: torch.nn.Module,
        topk_weights: torch.Tensor,
        prepare_and_permute_result: PreparePermuteResult,
    ):
        return None

    @abstractmethod
    def unpermute_finalize(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        prepare_and_permute_result: PreparePermuteResult,
        finalize_params=None,
        finalize_metadata=None,
    ) -> torch.Tensor:
        raise NotImplementedError


class All2AllPrepPmtAndUnpmtFinal(FusedMoEPreparePermuteAndUnpermuteFinalize):
    def prepare_permute(
        self, layer: torch.nn.Module, x: torch.Tensor, topk_ids: torch.Tensor
    ) -> All2AllPreparePermuteResult:
        x = x.view(-1, x.shape[-1])
        topk_ids = topk_ids.int()
        max_num_deployed_expert = layer.w13_weight.shape[0] * self.ep_size
        quant_mode = 1 if layer.quant_config is not None else -1

        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = (
            torch_npu.npu_moe_init_routing_v2(
                x,
                expert_idx=topk_ids,
                scale=None,
                expert_num=max_num_deployed_expert,
                active_expert_range=[0, max_num_deployed_expert],
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_num=topk_ids.numel(),
                drop_pad_mode=0,
                row_idx_type=0,
                quant_mode=quant_mode,
            )
        )

        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=get_ep_group().device_group)
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        combine_tokens = combine_tokens.view(2, self.ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(
            gathered_tokens, expanded_x, output_splits, input_splits,
            group=get_ep_group().device_group,
        )

        if layer.quant_config is None:
            gathered_pertoken_scale = None
        else:
            gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
            dist.all_to_all_single(
                gathered_pertoken_scale, pertoken_scale, output_splits, input_splits, group=get_ep_group().device_group
            )

        hidden_states_sorted_by_experts, gathered_pertoken_scale, gathered_idxs_unsort, tokens_per_local_expert = (
            torch_npu.npu_moe_re_routing(
                gathered_tokens,
                tokens_per_expert_group.view(self.ep_size, -1),
                per_token_scales=gathered_pertoken_scale,
            )
        )

        return All2AllPreparePermuteResult(
            gathered_idxs_unsort=gathered_idxs_unsort,
            expanded_x=expanded_x,
            expanded_row_idx=expanded_row_idx,
            input_splits=input_splits,
            output_splits=output_splits,
            hidden_states_sorted_by_experts=hidden_states_sorted_by_experts,
            expert_tokens=tokens_per_local_expert,
            dynamic_scale=gathered_pertoken_scale,
        )

    def unpermute_finalize(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        all2all_prepare_permute_result: All2AllPreparePermuteResult,
        finalize_params=None,
        finalize_metadata=None,
    ) -> torch.Tensor:
        gathered_idxs_unsort = all2all_prepare_permute_result.gathered_idxs_unsort
        expanded_x = all2all_prepare_permute_result.expanded_x
        input_splits = all2all_prepare_permute_result.input_splits
        output_splits = all2all_prepare_permute_result.output_splits
        expanded_row_idx = all2all_prepare_permute_result.expanded_row_idx
        new_x = torch.index_select(
            hidden_states, 0, gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32)
        )
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=get_ep_group().device_group)
        return torch_npu.npu_moe_finalize_routing(
            gathered_tokens,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )


class AGRSPrepPmtAndUnpmtFinal(FusedMoEPreparePermuteAndUnpermuteFinalize):
    def __init__(self, layer):
        super().__init__(layer)
        device_name = torch_npu.npu.get_device_name(0)
        self.is_a2_device = device_name.startswith("Ascend910B")
        self.side_stream = named_stream("sub_stream")

    def prepare_permute(
        self, layer: torch.nn.Module, x: torch.Tensor, topk_ids: torch.Tensor
    ) -> AGRSPreparePermuteResult:
        x = x.view(-1, x.shape[-1])
        topk_ids = topk_ids.int()
        max_num_deployed_expert = layer.w13_weight.shape[0] * self.ep_size
        experts_start_idx = self.ep_rank * layer.w13_weight.shape[0]
        experts_end_idx = experts_start_idx + layer.w13_weight.shape[0]
        expert_range = [experts_start_idx, experts_end_idx]
        row_idx_type = 0
        shared_expert_gate_up = None
        shared_expert_gate_up_proj_finished_event = None
        if layer.quant_config is None:
            gathered_x = get_ep_group().all_gather(x, dim=0)
            gathered_topk_ids = get_ep_group().all_gather(topk_ids, dim=0)
            expanded_x, expanded_row_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
                gathered_x,
                gathered_topk_ids,
                active_num=gathered_topk_ids.numel(),
                expert_capacity=-1,
                expert_num=max_num_deployed_expert,
                drop_pad_mode=0,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                quant_mode=-1,
                active_expert_range=expert_range,
                row_idx_type=row_idx_type,
            )
            dynamic_scale = None
        else:
            attn_metadata = get_forward_context().attn_metadata
            is_prefill = attn_metadata is None or attn_metadata[next(iter(attn_metadata))].num_prefills > 0
            prefill_routing = model_extra_config.operator_opt_config.prefill_grouped_matmul_finalize_routing
            if self.is_a2_device and (not is_prefill or prefill_routing):
                row_idx_type = 1
            moe_quant_config = getattr(layer.quant_method, "moe_quant_config", None)
            if moe_quant_config and getattr(moe_quant_config, "use_hifloat8_w8a8", False):
                x_hif8 = torch_npu.npu_dtype_cast(x, torch_npu.hifloat8)
                x_quant = x_hif8.view(dtype=torch.int8)
                x_scale = None
            elif moe_quant_config and getattr(moe_quant_config, "use_mxfp8_w8a8", False):
                # OCP MXFP8: route bf16 first, quant the expanded output below.
                # The mxfp8 scale path through npu_moe_init_routing_v2 is broken
                # on current torch_npu (see kernel_example/mxfp8_moe_init_routing.py),
                # so we keep the pre-routing tensor in bf16 and call
                # npu_dynamic_mx_quant on `expanded_x` after routing.
                x_quant = x
                x_scale = None
            else:
                x_int8, x_scale = torch_npu.npu_dynamic_quant(x)
                x_quant = x_int8
            x_quant = get_ep_group().all_gather(x_quant, dim=0)
            if x_scale is not None:
                x_scale = get_ep_group().all_gather(x_scale, dim=0)
            gathered_topk_ids = get_ep_group().all_gather(topk_ids, dim=0)
            opt_cfg = model_extra_config.operator_opt_config
            if (opt_cfg.shared_expert_multi_stream
                    and opt_cfg.shared_expert_parallel_schedule == "with_routed_experts_cv"):
                ready_to_dispatch_event = torch.npu.Event()
                shared_expert_gate_up_proj_finished_event = torch.npu.Event()
                ready_to_dispatch_event.record()
                with torch.npu.stream(self.side_stream):
                    ready_to_dispatch_event.wait(self.side_stream)
                    if moe_quant_config and getattr(moe_quant_config, "use_mxfp8_w8a8", False):
                        # x_quant is still bf16 here (mxfp8 routing-scale is buggy
                        # so we route bf16 and quant after); quant once on the
                        # side stream for shared_experts.gate_up_proj.
                        shared_x_fp8, shared_x_scale = torch_npu.npu_dynamic_mx_quant(
                            x_quant, dst_type=torch.float8_e4m3fn,
                        )
                        shared_expert_gate_up = layer.shared_experts.gate_up_proj({
                            'x_mxfp8': shared_x_fp8,
                            'pertoken_scale': shared_x_scale,
                        })
                    elif moe_quant_config and getattr(moe_quant_config, "use_hifloat8_w8a8", False):
                        shared_expert_gate_up = layer.shared_experts.gate_up_proj({'x_hif8': x_quant})
                    else:
                        raise NotImplementedError(
                            "with_routed_experts_cv schedule only supports hifloat8 or "
                            "mxfp8 quant methods."
                        )
                    x_quant.record_stream(self.side_stream)
                    shared_expert_gate_up_proj_finished_event.record()

            expanded_x, expanded_row_idx, expert_tokens, dynamic_scale = torch_npu.npu_moe_init_routing_v2(
                x_quant,
                gathered_topk_ids,
                scale=x_scale,
                offset=None,
                active_num=gathered_topk_ids.numel(),
                expert_num=max_num_deployed_expert,
                expert_capacity=-1,
                drop_pad_mode=0,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=expert_range,
                quant_mode=-1,
                row_idx_type=row_idx_type,
            )
        moe_quant_config = getattr(layer.quant_method, "moe_quant_config", None)
        if moe_quant_config is not None and getattr(moe_quant_config, "use_mxfp8_w8a8", False):
            # npu_moe_init_routing_v2 mxfp8 scale handling is buggy on current
            # torch_npu, so we routed bf16 above and quant the expanded output
            # here.
            expanded_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
                expanded_x, dst_type=torch.float8_e4m3fn,
            )
        if moe_quant_config is not None and getattr(moe_quant_config, "use_hifloat8_w8a8", False):
            # init routing output dirty dynamic_scale even its input scale=None
            dynamic_scale = None
        return AGRSPreparePermuteResult(
            hidden_states_sorted_by_experts=expanded_x,
            expert_tokens=expert_tokens,
            dynamic_scale=dynamic_scale,
            avg_tokens_per_expert=None,
            expert_range=expert_range,
            expanded_row_idx=expanded_row_idx,
            gathered_topk_ids=gathered_topk_ids,
            dtype=x.dtype,
            row_idx_type=row_idx_type,
            shared_expert_gate_up=shared_expert_gate_up,
            shared_expert_gate_up_proj_finished_event=shared_expert_gate_up_proj_finished_event
        )

    def prepare_finalize_params(
        self,
        layer: torch.nn.Module,
        topk_ids: torch.Tensor,
        agrs_prepare_permute_result: AGRSPreparePermuteResult,
    ) -> Optional[AGRSFinalizeParams]:
        """Pre-compute finalize routing params on a separate stream."""
        if layer.quant_config is None:
            return None
        if agrs_prepare_permute_result.row_idx_type != 1:
            return None

        expanded_row_idx = agrs_prepare_permute_result.expanded_row_idx
        gathered_topk_ids = agrs_prepare_permute_result.gathered_topk_ids
        expert_tokens = agrs_prepare_permute_result.expert_tokens

        batch_size = gathered_topk_ids.shape[0]
        range1 = torch.arange(0, expanded_row_idx.shape[0], dtype=torch.int32, device="npu")
        range2 = range1 * GROUPED_FINALIZE_ROW_INDEX_STRIDE
        mask = (range1 >= torch.sum(expert_tokens)).to(torch.int32)
        expanded_row_idx += range2 * mask
        expanded_row_idx = expanded_row_idx % expanded_row_idx.shape[0]
        expanded_row_idx = torch.clamp(expanded_row_idx, min=0, max=expanded_row_idx.shape[0] - 1)
        last_dimension_size = gathered_topk_ids.shape[-1]
        last_dimension_size_tensor = gathered_topk_ids.new_empty(
            (), dtype=torch.int32
        ).fill_(last_dimension_size)
        row_index = expanded_row_idx // last_dimension_size_tensor
        row_index = row_index.to(torch.int64)

        is_w4a8 = hasattr(layer, "w2_weight_int4_scale")
        if is_w4a8:
            w2_scale = layer.w2_weight_int4_scale
            w2_bias = layer.w2_weight_bias
        else:
            w2_scale = layer.w2_weight_scale.to(torch.float)
            w2_bias = layer.w2_bias if hasattr(layer, "w2_bias") else None

        return AGRSFinalizeParams(
            expanded_row_idx=expanded_row_idx,
            row_index=row_index,
            batch_size=batch_size,
            w2_scale=w2_scale,
            w2_bias=w2_bias,
        )

    def prepare_finalize_metadata(
        self,
        layer: torch.nn.Module,
        topk_weights: torch.Tensor,
        agrs_prepare_permute_result: AGRSPreparePermuteResult,
    ) -> AGRSFinalizeMetadata:
        gathered_topk_weights = get_ep_group().all_gather(topk_weights, dim=0)
        return AGRSFinalizeMetadata(
            gathered_topk_weights=gathered_topk_weights,
        )

    def unpermute_finalize(
        self,
        layer: torch.nn.Module,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        agrs_prepare_permute_result: AGRSPreparePermuteResult,
        finalize_params: Optional[AGRSFinalizeParams] = None,
        finalize_metadata: Optional[AGRSFinalizeMetadata] = None,
    ) -> torch.Tensor:
        expanded_row_idx = agrs_prepare_permute_result.expanded_row_idx
        gathered_topk_ids = agrs_prepare_permute_result.gathered_topk_ids
        row_idx_type = agrs_prepare_permute_result.row_idx_type
        expert_tokens = agrs_prepare_permute_result.expert_tokens

        if finalize_metadata is None:
            finalize_metadata = self.prepare_finalize_metadata(
                layer, topk_weights, agrs_prepare_permute_result
            )
        gathered_topk_weights = finalize_metadata.gathered_topk_weights
        if layer.quant_config is not None:
            # TODO wjc: 1.hif8 gmmfr adaption 2. add share_experts_output.
            if row_idx_type == 1 and finalize_params is not None:
                x, pertoken_scale = hidden_states
                sorted_topk_weight = torch.index_select(
                    gathered_topk_weights.reshape(-1), 0, finalize_params.expanded_row_idx
                ).float()
                y = torch_npu.npu_grouped_matmul_finalize_routing(
                    x, layer.w2_weight, expert_tokens,
                    scale=finalize_params.w2_scale,
                    bias=finalize_params.w2_bias,
                    pertoken_scale=pertoken_scale,
                    shared_input=None,
                    logit=sorted_topk_weight,
                    row_index=finalize_params.row_index,
                    output_bs=finalize_params.batch_size,
                    shared_input_weight=1.0,
                    group_list_type=1,
                    shared_input_offset=0
                ).to(agrs_prepare_permute_result.dtype)
            elif row_idx_type == 1:
                x, pertoken_scale = hidden_states
                batch_size = gathered_topk_ids.shape[0]
                range1 = torch.arange(0, expanded_row_idx.shape[0], dtype=torch.int32, device="npu")
                range2 = range1 * GROUPED_FINALIZE_ROW_INDEX_STRIDE
                mask = (range1 >= torch.sum(expert_tokens)).to(torch.int32)
                expanded_row_idx += range2 * mask
                expanded_row_idx = expanded_row_idx % expanded_row_idx.shape[0]
                expanded_row_idx = torch.clamp(expanded_row_idx, min=0, max=expanded_row_idx.shape[0] - 1)
                sorted_topk_weight = torch.index_select(gathered_topk_weights.reshape(-1), 0, expanded_row_idx)
                if model_extra_config.operator_opt_config.prefill_grouped_matmul_finalize_routing:
                    sorted_topk_weight = sorted_topk_weight.float()
                row_index = expanded_row_idx // gathered_topk_ids.shape[-1]
                row_index = row_index.to(torch.int64)

                is_w4a8 = hasattr(layer, "w2_weight_int4_scale")
                if is_w4a8:
                    w2_scale = layer.w2_weight_int4_scale
                    w2_bias = layer.w2_weight_bias
                else:
                    w2_scale = layer.w2_weight_scale.to(torch.float)
                    w2_bias = layer.w2_bias if hasattr(layer, "w2_bias") else None
                y = torch_npu.npu_grouped_matmul_finalize_routing(
                    x,
                    layer.w2_weight,
                    expert_tokens,
                    scale=w2_scale,
                    bias=w2_bias,
                    pertoken_scale=pertoken_scale,
                    shared_input=None,
                    logit=sorted_topk_weight,
                    row_index=row_index,
                    output_bs=batch_size,
                    shared_input_weight=1.0,
                    group_list_type=1,
                    shared_input_offset=0
                ).to(agrs_prepare_permute_result.dtype)
            else:
                y = torch_npu.npu_moe_finalize_routing(
                    hidden_states.unsqueeze(0),
                    skip1=None,
                    skip2=None,
                    bias=None,
                    scales=gathered_topk_weights.float(),
                    expanded_src_to_dst_row=expanded_row_idx,
                    export_for_source_row=gathered_topk_ids,
                    drop_pad_mode=3,
                ).to(agrs_prepare_permute_result.dtype)
        else:
            y = torch_npu.npu_moe_finalize_routing(
                hidden_states.unsqueeze(0),
                skip1=None,
                skip2=None,
                bias=None,
                scales=gathered_topk_weights,
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=gathered_topk_ids,
                drop_pad_mode=3,
            )
        return get_ep_group().reduce_scatter(y, dim=0)


class DispatchCombinePrepPmtAndUnpmtFinal(FusedMoEPreparePermuteAndUnpermuteFinalize):
    def __init__(self, layer):
        super().__init__(layer)
        if on_ascend950():
            # TODO zhaoyi: fix get_hccl_comm_name error on A5
            logger.warning_once(
                "DispatchCombinePrepPmtAndUnpmtFinal's initialization is skipped "
                "due to get_hccl_comm_name error on Ascend950."
            )
        else:
            self.moe_all_to_all_group = get_ep_group().device_group
            self.moe_all_to_all_group_name = self.moe_all_to_all_group._get_backend(
                torch.device(current_platform.device_type)
            ).get_hccl_comm_name(get_ep_group().rank_in_group)

    def _get_mc2_mask(self, num_tokens: int) -> torch.Tensor | None:
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()), None) if attn_metadata else None
        if hasattr(attn_metadata, "decode") and attn_metadata.decode is not None:
            mc2_mask = getattr(attn_metadata.decode, "mc2_mask", None)
            if mc2_mask is not None:
                mc2_mask = mc2_mask[:num_tokens]
            return mc2_mask
        return None

    def prepare_permute(
        self, layer: torch.nn.Module, x: torch.Tensor, topk_ids: torch.Tensor
    ) -> DispatchCombinePreparePermuteResult:
        quant_mode = 2 if layer.quant_config is not None else 0
        output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=topk_ids,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=self.num_experts,
            global_bs=0,
            scales=None,
            quant_mode=quant_mode,
            group_ep=self.moe_all_to_all_group_name,
            ep_world_size=self.ep_size,
            ep_rank_id=self.ep_rank,
            x_active_mask=self._get_mc2_mask(topk_ids.shape[0]),
        )
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = output[0:6]
        return DispatchCombinePreparePermuteResult(
            hidden_states_sorted_by_experts=expand_x,
            expert_tokens=expert_token_nums.to(torch.int64),
            tp_recv_counts=tp_recv_counts,
            ep_recv_counts=ep_recv_counts,
            expand_idx=expand_idx,
            dynamic_scale=dynamic_scale,
        )

    def unpermute_finalize(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        dispatch_combine_prepare_permute_result: DispatchCombinePreparePermuteResult,
        finalize_params=None,
        finalize_metadata=None,
    ) -> torch.Tensor:
        return torch_npu.npu_moe_distribute_combine_v2(
            expand_x=hidden_states,
            expert_ids=topk_ids,
            assist_info_for_combine=dispatch_combine_prepare_permute_result.expand_idx,
            expert_scales=topk_weights.to(torch.float32),
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=self.num_experts,
            global_bs=0,
            ep_send_counts=dispatch_combine_prepare_permute_result.ep_recv_counts,
            group_ep=self.moe_all_to_all_group_name,
            ep_world_size=self.ep_size,
            ep_rank_id=self.ep_rank,
            tp_send_counts=dispatch_combine_prepare_permute_result.tp_recv_counts,
            x_active_mask=self._get_mc2_mask(topk_ids.shape[0]),
        )


class CommunicationStrategySelector:

    def __init__(self, moe: torch.nn.Module):
        device_name = torch_npu.npu.get_device_name(0)
        self.is_a2_device = device_name.startswith("Ascend910B")
        self.is_a5_device = device_name.startswith("Ascend950")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dp_size = get_dp_group().world_size

        self.max_dispatch_combine_threshold = int(os.getenv(
            'MAX_DISPATCH_COMBINE_THRESHOLD',
            DEFAULT_MAX_DISPATCH_COMBINE_THRESHOLD))
        if self.is_a2_device:
            assert self.max_dispatch_combine_threshold <= 256, \
            f"{self.max_dispatch_combine_threshold=} should be no larger than 256 on A2 devices."
        elif not self.is_a5_device:
            assert self.max_dispatch_combine_threshold <= 512, \
            f"{self.max_dispatch_combine_threshold=} should be no larger than 512 on A3 devices."

        self.moe = moe
        self.prepare_permute_and_unpermute_finalize_cls_dict = {
            "all2all": All2AllPrepPmtAndUnpmtFinal,
            "agrs": AGRSPrepPmtAndUnpmtFinal,
            "dispatch_combine": DispatchCombinePrepPmtAndUnpmtFinal,
        }
        self.prepare_permute_and_unpermute_finalize_dict = {}

    def _get_strategy_impl(self, strategy: str):
        strategy_impl = self.prepare_permute_and_unpermute_finalize_dict.get(strategy)
        if strategy_impl is None:
            strategy_impl_cls = self.prepare_permute_and_unpermute_finalize_cls_dict[strategy]
            strategy_impl = strategy_impl_cls(self.moe)
            self.prepare_permute_and_unpermute_finalize_dict[strategy] = strategy_impl
        return strategy_impl

    def select_communication_strategy(self, num_tokens: int):
        forward_ctx = get_forward_context()
        attn_metadata = forward_ctx.attn_metadata
        if attn_metadata is not None:
            attn_metadata = next(iter(attn_metadata.values()), None)

        if model_extra_config.parall_config.ena_seq_parallel:
            local_num_tokens = num_tokens
        else:
            # tokens per rank
            local_num_tokens = cdiv(num_tokens, self.tp_size)
        
        if self.is_a5_device or model_extra_config.operator_opt_config.enable_moe_agrs:
            # Only supports AGRS on A5 now
            strategy = "agrs"
        elif self.is_a2_device:
            if model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
                if local_num_tokens > self.max_dispatch_combine_threshold:
                    if model_extra_config.parall_config.ena_seq_parallel:
                        strategy = "agrs"
                    else:
                        strategy = "all2all"
                else:
                    strategy = "dispatch_combine"
            else:
                # TP or DP only
                if self.tp_size == 1 or self.dp_size == 1:
                    strategy = "agrs"
                else:
                    # TP + DP
                    if forward_ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                        strategy = "agrs"
                    else:
                        strategy = "all2all"
        else:
            # TP or DP only
            if self.dp_size == 1 or self.tp_size == 1:
                if local_num_tokens > self.max_dispatch_combine_threshold:
                    strategy = "all2all"
                else:
                    strategy = "dispatch_combine"
            else:
                # TP + DP
                if forward_ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                    strategy = "agrs"
                else:
                    strategy = "all2all"

        return strategy, self._get_strategy_impl(strategy)
