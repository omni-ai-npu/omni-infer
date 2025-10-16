# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from enum import IntEnum, auto
from typing import Any, Dict, Iterable, Optional, Tuple, Callable
from contextlib import nullcontext

import torch
from torch import nn
import torch.distributed as dist
import torchair as tng
import torch_npu
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
    get_pp_group,
    get_world_group,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import add_prefix
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder

from omni.adaptors.sglang.layers.moe.ep_moe.layer import FusedMoE
from omni.adaptors.sglang.layers.activation import SiluAndMul
from omni.adaptors.sglang.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)


# TODO: not really "replicated" currently
class ReplicatedDeepseekMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        # TODO: expecting a configuration for throw_dequant in __init__
        self.gate_up_proj.throw_dequant = True

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False
        self._x = None
        self.multi_stream = False

    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)

    def forward(
        self,
        x:Optional[torch.Tensor]=None,
        stage:Optional[str]="full",
        dependency:Optional[torch.Tensor]=None,
        multi_stream:Optional[bool]=None
    )->Optional[torch.Tensor]:

        if stage in ["full", "gate_up_proj"]:
            assert isinstance(x, torch.Tensor)
            self._x = x

        with self._stream_switch(dependency, multi_stream):
            if stage in ["full", "gate_up_proj"]:
                self._x, bias = self.gate_up_proj(self._x)
            if stage in ["full", "act_fn"]:
                self._x = self.act_fn(self._x, self.quant_symbol)
            if stage in ["full", "down_proj"]:
                self._x, bias = self.down_proj(self._x, skip_all_reduce=False)
                return self._x

    # ================ utils ==================

    def _stream_switch(self, dependency=None, multi_stream=None):

        if multi_stream is not None:
            self.multi_stream = multi_stream

        if not self.multi_stream:
            return nullcontext()

        if dependency is not None:
            tng.scope.npu_wait_tensor(self._get_tensor_x(), dependency)

        return tng.scope.npu_stream_switch('stream_shared_expert')

    def _get_tensor_x(self):
        if isinstance(self._x, torch.Tensor):
            return self._x
        elif isinstance(self._x, tuple) or isinstance(self._x, list):
            return self._x[0]
        elif isinstance(self._x, dict):
            return self._x['x_int8']
        raise ValueError(f"_get_tensor_x: unrecognized type {type(self._x)}")


class DeepseekMoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:

        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.world_size = get_world_group().world_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_moe_expert_parallel_world_size()
        self.top_k = config.num_experts_per_tok

        self.quant_symbol = True if quant_config else False

        self.num_fused_shared_experts = 0
        if not global_server_args_dict["disable_shared_experts_fusion"]:
            self.num_fused_shared_experts = config.n_shared_experts

        assert global_server_args_dict["moe_a2a_backend"].is_deepep()

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # ========================== init gate ==============================

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate")

        self.gate.quant_method.enable_weight_nz = False

        self.gate.e_score_correction_bias = None
        if getattr(config, "topk_method", "topk") == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float),
                requires_grad=False)

        # ====================== init FusedMoE ======================

        deepep_mode = global_server_args_dict["deepep_mode"]
        ep_num_redundant_experts = global_server_args_dict["ep_num_redundant_experts"]

        is_nextn = kwargs.get("is_nextn",False)
        if is_nextn:
            ep_num_redundant_experts = 0

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts + self.num_fused_shared_experts + ep_num_redundant_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            deepep_mode=deepep_mode, # moe_a2a_backend
        )

        moe_rs_group = get_pp_group().device_group
        moe_rs_group_rank = get_pp_group().rank_in_group
        self.moe_rs_group_name = moe_rs_group._get_backend(torch.device("npu")).get_hccl_comm_name(moe_rs_group_rank)

        # ======= init for _forward_decode_dispatch_combine =========

        # TODO: prefetch size not confirmed
        # w13: 50MB for default, 30MB for BF16
        # w2: 28MB when w8a8 and ep_size > 64, otherwise 0
        self.w13_prefetch_size = 30 * 1024 * 1024
        self.w2_prefetch_size = 0
        if self.quant_symbol and self.ep_size > 64:
            self.w2_prefetch_size = 28 * 1024 * 1024
        self.local_expert_num = self.experts.w13_weight.shape[0]

        # TODO: prefetch size not confirmed
        self.attn_prefetch_size = 96 * 1024 * 1024

        if self.quant_symbol:
            self.local_expert_num = self.experts.w13_weight.shape[0]

            self.in_scale_2 = torch.ones(
                (self.local_expert_num, config.moe_intermediate_size),
                dtype=torch.float32,
                device="npu")
            torch._dynamo.mark_static(self.in_scale_2) # call the mark_static to reduce memory usage

        # TODO: gmm_nz, not aligned with vLLM's tuning_config
        # model_extra_config.operator_opt_config.decode_gear_list[:1]
        self.tuning_config = None

        # ====================== init shared_experts ======================
        
        self.shared_experts = None
        if (config.n_shared_experts is not None
            and self.num_fused_shared_experts == 0):

            self.shared_experts = ReplicatedDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                tp_size=1, # disable tp for shared experts when enable deepep moe
                tp_rank=0,
            )

            # when shared_experts is fp8:
            if self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn:
                if (hasattr(self.shared_experts.gate_up_proj.quant_method, "quant_config")
                    and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name() == "moe_wna16"):
                    assert (
                        self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                        == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
                    )

        # ====================== misc ======================
        
        self.num_experts = config.n_routed_experts + ep_num_redundant_experts

        self.group = get_world_group().device_group
        self.global_rank = get_world_group().rank_in_group
        self.group_name = self.group._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank)

        self.experts_tp_size = 1 # TODO: not aligned with vLLM's: get_ep_group().world_size
        self.shared_expert_rank_num = 0 # route_share_on_same_card

        # TODO: not aligned with vLLM's: not using smooth_scale
        # smooth_scale, now dpsk use smooth_scale == 1
        epsilon = 1e-2
        self.smooth_scale = torch.nn.Parameter(
            torch.ones(
                size=(self.num_experts, config.hidden_size),
                dtype=torch.float32
            ) * (1 - epsilon) + epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        prefetch_list: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:

        # TODO: expecting a config for graph mode sccessible in __init__
        self.run_graph = forward_batch.can_run_graph

        if forward_batch.is_extend_in_batch:
            return self._forward_prefill_norm(hidden_states, residual, forward_batch)
        else:
            return self._forward_decode_dispatch_combine(hidden_states, residual, forward_batch, prefetch_list)

    def _forward_prefill_norm(self, hidden_states, residual, forward_batch) -> torch.Tensor:

        shared_output = None

        if hidden_states.shape[0] > 0 and not forward_batch.is_prefill_idle:

            if self.shared_experts is not None:
                shared_output = self.shared_experts(
                    x=hidden_states,
                    multi_stream=False)

            router_logits, _ = self.gate(hidden_states.float())

            topk_weights, topk_ids, _ = FusedMoE.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.experts.top_k,
                use_grouped_topk=True,
                renormalize=self.config.norm_topk_prob,
                topk_group=self.config.topk_group,
                num_expert_group=self.config.n_group,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.config.routed_scaling_factor)
            topk_ids = self.experts.apply_expert_load_balance(topk_ids=topk_ids)
        else:
            topk_ids = torch.randperm(256)[:hidden_states.size(0) * self.top_k].reshape(
                hidden_states.size(0), self.top_k
                ).npu()

            topk_weights = torch.empty(
                (hidden_states.size(0), self.top_k),
                dtype=torch.float32,
                device="npu")

        final_hidden_states_list = self.experts(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            forward_batch=forward_batch,
            comm_group=self.group,
            dynamic_scale=self.smooth_scale,
        )

        if len(final_hidden_states_list) != 3:
            raise RuntimeError("len(final_hidden_states_list) != 3")

        hidden_states = final_hidden_states_list[0]
        gathered_tokens = final_hidden_states_list[1]
        expanded_row_idx = final_hidden_states_list[2]

        # finalize-routing
        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens,
            skip1=shared_output,
            skip2=None,
            bias=None,
            scales=topk_weights.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )

        return hidden_states, residual

    def _forward_decode_dispatch_combine(self, hidden_states, residual, forward_batch, prefetch_list) -> torch.Tensor:

        # assert hidden_states.shape[0] > 0 and not forward_batch.is_prefill_idle

        # ====================== gate ======================

        router_logits, _ = self.gate(hidden_states.float())

        ena_multi_stream = self.run_graph

        # 2D->3D->2D conversion, to fuse add rms and cast into AddRmsNormCast.
        hidden_states = hidden_states.unsqueeze(1).squeeze(1)

        act_dtype = hidden_states.dtype # should be torch.bfloat16

        # ====================== multi_stream ======================

        # forward shared_experts.gate_up_proj
        self.shared_experts(
            x=hidden_states,
            stage="gate_up_proj",
            dependency=router_logits,
            multi_stream=ena_multi_stream)

        # expert weight prefetch
        if ena_multi_stream:
            wait_gate = self.shared_experts._get_tensor_x()
            if self.w13_prefetch_size > 0:
                torch_npu.npu_prefetch(self.experts.w13_weight, wait_gate, self.w13_prefetch_size)
            if self.w2_prefetch_size > 0:
                torch_npu.npu_prefetch(self.experts.w2_weight, wait_gate, self.w2_prefetch_size)

        # ========================== select_experts ==============================

        topk_weights, topk_ids, _ = FusedMoE.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.experts.top_k,
            use_grouped_topk=True,
            renormalize=self.config.norm_topk_prob,
            topk_group=self.config.topk_group,
            num_expert_group=self.config.n_group,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.config.routed_scaling_factor)

        topk_ids = self.experts.apply_expert_load_balance(topk_ids=topk_ids)
        # ====================== dispatch ======================

        # TODO: apply_expert_load_balance & best_topk not aligned with vLLM's

        dispatch_quant_mode = 2 # 0: non-quant; 1: static quant(not supported now); 2: dynamic quant

        (
            expand_x,
            dynamic_scale,
            expand_idx,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
        ) = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states,
            expert_ids=topk_ids,                                # [n * topk]
            expert_shard_type=0,                                # Set it to 0 for now
            shared_expert_rank_num=self.shared_expert_rank_num, # 0
            moe_expert_num=self.num_experts,                    # TODO: not aligned with vLLM's: local_expert_num * get_dp_group().world_size
            global_bs=0,                                        # 0 Default (all); all tokens can be set
            scales=None,                                        # Quantization coefficient
            quant_mode=dispatch_quant_mode,                     # 2-Dynamic quantization
            group_ep=self.group_name,                           # TODO: layer.moe_all_to_all_group_name
            ep_world_size=self.world_size,
            ep_rank_id=self.global_rank,
            group_tp=self.moe_rs_group_name,                    # self.experts.moe_rs_group_name
            tp_world_size=self.experts_tp_size,                 # disable tp for shared experts when enable deepep moe
            tp_rank_id=0,                                       # disable tp for shared experts when enable deepep moe
            x_active_mask=None,                                 # TODO: mc2_mask, not aligned with vLLM's
        )[:6]

        group_list = expert_token_nums.to(torch.int64)

        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            self.layer_id,
            [],
            num_tokens_per_rank=None,
            num_tokens_per_rdma_rank=None,
            num_tokens_per_expert=group_list,
        )

        # ====================== call experts part 1 ======================

        weight1_3 = self.experts.w13_weight
        weight2 = self.experts.w2_weight

        if self.quant_symbol:

            # TODO: must be setup when initializing self.experts
            # in AscendCompressedTensorsConfig.get_quant_method(...)
            assert hasattr(self.experts, "weight_num_bits")

            if self.experts.weight_num_bits == 8:
                weight_scale1_3 = self.experts.w13_weight_scale.squeeze(-1) # adapt shape
                weight_scale2 = self.experts.w2_weight_scale.squeeze(-1) # adapt shape and dtype
            elif self.experts.weight_num_bits == 4:
                weight_scale1_3 = self.experts.w13_weight_int4_scale
                weight_scale2 = self.experts.w2_weight_int4_scale
                weight_bias1_3 = self.experts.w13_weight_bias
                weight_bias2 = self.experts.w2_weight_bias
            else:
                raise NotImplementedError(f"Unsupported compress tensor type. num bits: {self.experts.weight_num_bits}")

            if dispatch_quant_mode != 0:
                pertoken_scale = dynamic_scale
            else:
                expand_x, pertoken_scale = torch_npu.npu_dynamic_quant(expand_x)

        # ====================== multi_stream ======================

        self.shared_experts(stage="act_fn", dependency=expand_x)

        # ====================== call experts part 2 ======================

        if self.quant_symbol:
            if self.experts.weight_num_bits == 8: # w8a8

                gate_up_proj = torch_npu.npu_grouped_matmul(
                    [expand_x],
                    [weight1_3],
                    bias=None,
                    group_list=group_list,
                    split_item=3,
                    output_dtype=torch.int32,
                    group_type=0,
                    group_list_type=1)[0]

                gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                    gate_up_proj,
                    weight_scale=weight_scale1_3,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=self.in_scale_2,
                    quant_offset=None,
                    group_index=group_list,
                    activate_left=True,
                    quant_mode=1) # 1: dynamic quant; 0: static quant(not supported now)

                hidden_states_experts = torch_npu.npu_grouped_matmul(
                    [gate_up_proj],
                    [weight2],
                    scale=[weight_scale2],
                    per_token_scale=[pertoken_scale],
                    bias=None,
                    group_list=group_list,
                    split_item=3,
                    output_dtype=act_dtype,
                    group_type=0,
                    group_list_type=1)[0]

            elif self.experts.weight_num_bits == 4:
                gate_up_proj = torch_npu.npu_grouped_matmul(
                    [expand_x],
                    [weight1_3],
                    bias=[weight_bias1_3],
                    scale=[weight_scale1_3],
                    offset=None,
                    antiquant_scale=None,
                    antiquant_offset=None,
                    per_token_scale=[pertoken_scale],
                    group_list=group_list,
                    activation_input=None,
                    activation_quant_scale=None,
                    activation_quant_offset=None,
                    split_item=3,
                    group_type=0,
                    group_list_type=1,
                    act_type=0,
                    tuning_config=self.tuning_config,
                    output_dtype=torch.bfloat16)[0]

                fake_scale = torch.ones(weight_bias1_3.shape, dtype=torch.float32, device="npu").view(-1,weight_bias1_3.shape[1])
                pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")

                gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                    gate_up_proj,
                    weight_scale=fake_scale,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=None,
                    quant_offset=None,
                    group_index=group_list,
                    activate_left=True,
                    quant_mode=1) # 1: dynamic quant; 0: static quant(not supported now)

                hidden_states_experts = torch_npu.npu_grouped_matmul(
                    [gate_up_proj],
                    [weight2],
                    scale=[weight_scale2],
                    per_token_scale=[pertoken_scale],
                    bias=[weight_bias2],
                    group_list=group_list,
                    split_item=3,
                    output_dtype=act_dtype,
                    group_type=0,
                    group_list_type=1,
                    tuning_config=self.tuning_config)[0]
            else:
                raise NotImplementedError(f"Unsupported compress tensor type. num bits: {self.experts.weight_num_bits}")
        else: # bf16
            gate_up_proj = torch_npu.npu_grouped_matmul(
                [expand_x], 
                [weight1_3], 
                bias=None, 
                group_list=group_list,
                split_item=3, 
                group_type=0,
                group_list_type=1)[0]

            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

            hidden_states_experts = torch_npu.npu_grouped_matmul(
                [gate_up_proj],
                [weight2],
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=act_dtype,
                group_type=0,
                group_list_type=1)[0]

        # ====================== multi_stream ======================

        shared_output = self.shared_experts(stage="down_proj", dependency=hidden_states_experts)
        assert shared_output is not None

        # prefetch weights for next attention
        if ena_multi_stream:
            if self.attn_prefetch_size > 0 and isinstance(prefetch_list, dict):
                for name, weight in prefetch_list.items():
                    torch_npu.npu_prefetch(
                        weight,
                        shared_output, # dependency
                        self.attn_prefetch_size)

        # ====================== combine ======================

        hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=hidden_states_experts,
            expert_ids=topk_ids,                                # [n * topk]
            assist_info_for_combine=expand_idx,
            expert_scales=topk_weights.to(torch.float32),       # weight [n * topk]
            expert_shard_type=0,
            shared_expert_x=None,                               # integrated "Add" of shared_output not suitable for multi_stream
            shared_expert_rank_num=self.shared_expert_rank_num,
            moe_expert_num=self.num_experts,                    # 0 redundancy 256, 1 redundancy expert 320
            global_bs=0,                                        # 0 Default (all); all tokens can be set
            ep_send_counts=ep_recv_counts,                      # dispatch's send_counts
            group_ep=self.group_name,                           # Unlike torch, it is obtained by name.
            ep_world_size=self.world_size,
            ep_rank_id=self.global_rank,
            tp_send_counts=tp_recv_counts,
            group_tp=self.moe_rs_group_name,
            tp_world_size=self.experts_tp_size,                 # disable tp for shared experts when enable deepep moe
            tp_rank_id=0,                                       # disable tp for shared experts when enable deepep moe
            x_active_mask=None,                                 # TODO: mc2_mask, not aligned with vLLM's
        )

        return hidden_states_route + shared_output, residual

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]
