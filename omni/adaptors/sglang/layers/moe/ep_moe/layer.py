from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch
import torch_npu
import torchair as tng
import torch.distributed as dist
from sglang.srt.distributed import get_moe_ep_group, get_world_group

from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.layers.moe.ep_moe.kernels import (
    moe_ep_deepgemm_preprocess,
    post_reorder_triton_kernel,
    silu_and_mul_masked_post_quant_fwd,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE as BaseFusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import dispose_tensor
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical

from torch.distributed import ProcessGroup

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutput


logger = logging.getLogger(__name__)


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:
    temp = x.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(temp, 23)
    mant = torch.bitwise_and(temp, 0x7FFFFF)
    is_ru = torch.logical_and(
        torch.logical_and((mant > 0), (exp != 0xFE)),
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),
    )
    exp = torch.where(is_ru, exp + 1, exp)
    new_x = exp.to(torch.uint8).view(torch.int)
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)


class EPMoE(BaseFusedMoE):
    """
    MoE Expert Parallel Impl
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        activation_alpha: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
        with_bias: bool = False,
    ):
        super().__init__(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_fused_shared_experts=num_fused_shared_experts,
            layer_id=layer_id,
            top_k=top_k,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            activation=activation,
            # apply_router_weight_on_input=apply_router_weight_on_input,
            routed_scaling_factor=routed_scaling_factor,
            activation_alpha=activation_alpha,
            swiglu_limit=swiglu_limit,
            with_bias=with_bias,
        )

        self.start_expert_id = self.moe_ep_rank * self.num_local_experts
        self.end_expert_id = self.start_expert_id + self.num_local_experts - 1

        self.intermediate_size = intermediate_size

        self.use_fp8_w8a8 = False
        self.use_block_quant = False
        self.block_shape = None
        self.activation_scheme = None

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8:
            return self.forward_deepgemm(hidden_states, topk_output)
        else:
            return super().forward(hidden_states, topk_output)

    def forward_deepgemm(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        self.w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv if self.use_block_quant else self.w2_weight_scale,
        )

        assert self.quant_method is not None
        assert self.activation == "silu"
        hidden_states_shape = hidden_states.shape
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        topk_weights, topk_ids, _ = topk_output

        if not self.use_block_quant:
            # Convert per-tensor quant to per-block quant by repeating scales for forward_deepgemm
            scale_block_size = 128
            w13_weight_scale_n = 2 * (
                (self.intermediate_size + scale_block_size - 1) // scale_block_size
            )
            w13_weight_scale_k = (
                hidden_states_shape[-1] + scale_block_size - 1
            ) // scale_block_size
            w13_weight_scale = (
                self.w13_weight_scale.unsqueeze(1)
                .repeat_interleave(w13_weight_scale_n, dim=1)
                .unsqueeze(2)
                .repeat_interleave(w13_weight_scale_k, dim=2)
            )
            self.w13_weight_fp8 = (
                self.w13_weight,
                w13_weight_scale,
            )
            w2_weight_scale_n = (
                hidden_states_shape[-1] + scale_block_size - 1
            ) // scale_block_size
            w2_weight_scale_k = (
                self.intermediate_size + scale_block_size - 1
            ) // scale_block_size
            w2_weight_scale = (
                self.w2_weight_scale.unsqueeze(1)
                .repeat_interleave(w2_weight_scale_n, dim=1)
                .unsqueeze(2)
                .repeat_interleave(w2_weight_scale_k, dim=2)
            )
            self.w2_weight_fp8 = (
                self.w2_weight,
                w2_weight_scale,
            )

        # PreReorder
        m_max, masked_m, expected_m, src2dst, gateup_input, gateup_input_scale = (
            moe_ep_deepgemm_preprocess(
                topk_ids,
                self.num_experts,
                hidden_states,
                self.top_k,
                self.start_expert_id,
                self.end_expert_id,
                self.block_shape,
            )
        )

        dispose_tensor(hidden_states)

        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            b, s_mn, s_k = gateup_input_scale.shape
            assert (
                s_mn % 4 == 0 and s_k % 4 == 0
            ), f"scales must be aligned to 4, but got ({b}, {s_mn}, {s_k})"

        # GroupGemm-0
        gateup_input_fp8 = (
            gateup_input,
            (
                _cast_to_e8m0_with_rounding_up(gateup_input_scale)
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else deep_gemm_wrapper.get_col_major_tma_aligned_tensor(
                    gateup_input_scale
                )
            ),
        )
        num_groups, m, k = gateup_input_fp8[0].size()
        n = self.w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            gateup_input_fp8,
            self.w13_weight_fp8,
            gateup_output,
            masked_m,
            expected_m,
            recipe=(1, 128, 128) if deep_gemm_wrapper.DEEPGEMM_BLACKWELL else None,
        )
        del gateup_input
        del gateup_input_fp8

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=hidden_states_device,
            dtype=self.fp8_dtype,
        )
        scale_block_size = 128
        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // scale_block_size,
            ),
            device=hidden_states_device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            masked_m,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del gateup_output

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_input_fp8 = (
            down_input,
            (
                down_input_scale
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else deep_gemm_wrapper.get_col_major_tma_aligned_tensor(
                    down_input_scale
                )
            ),
        )
        down_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            down_input_fp8,
            self.w2_weight_fp8,
            down_output,
            masked_m,
            expected_m,
            recipe=(1, 128, 128) if deep_gemm_wrapper.DEEPGEMM_BLACKWELL else None,
        )
        del down_input
        del down_input_fp8

        # PostReorder
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        post_reorder_triton_kernel[(hidden_states_shape[0],)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states_shape[1],
            m_max * self.start_expert_id,
            BLOCK_SIZE=512,
        )
        if self.routed_scaling_factor is not None:
            output *= self.routed_scaling_factor
        return output


class DeepEPMoE(EPMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.deepep_mode = deepep_mode

        # TODO: move to the beginning of the file
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.two_batch_overlap import MaybeTboDeepEPDispatcher

        self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
            group=get_tp_group().device_group,
            router_topk=self.top_k,
            permute_fusion=True,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
            async_finish=True,  # TODO
            return_recv_hook=True,
        )

        if self.deepep_mode.enable_low_latency():
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        self.w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            (
                self.w2_weight_scale_inv
                if self.use_block_quant
                else self.w2_weight_scale
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        dispatch_output = self.dispatch(
            hidden_states, topk_idx, topk_weights, forward_batch
        )
        hidden_states = self.combine(
            dispatch_output.hidden_states,
            dispatch_output.topk_idx,
            dispatch_output.topk_weights,
            forward_batch,
        )
        return hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self.deepep_dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )

    def moe_impl(self, dispatch_output: DispatchOutput):
        if dispatch_output.format.is_deepep_normal():
            assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
            return self.forward_deepgemm_contiguous(dispatch_output)
        elif dispatch_output.format.is_deepep_ll():
            assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
            return self.forward_deepgemm_masked(dispatch_output)
        else:
            raise ValueError(
                f"Dispatch output format {dispatch_output.format} is not supported"
            )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self.deepep_dispatcher.combine(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )


class FusedMoE(DeepEPMoE):

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        deepep_mode: DeepEPMode = DeepEPMode.NORMAL,
        **kwargs
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            deepep_mode=deepep_mode,
        )

        self.moe_layer_idx = None
        self.expert_mapping = None
        if global_server_args_dict["enable_omni_placement"]:
            self.moe_layer_idx = get_global_expert_location_metadata().get_moe_layer_idx(layer_id)
            self.expert_mapping = get_global_expert_location_metadata().expert_mapping_on_current_layer(self.moe_layer_idx)

        self.quant_scale = torch.nn.Parameter(
            torch.ones(
                size=(self.num_local_experts, self.w2_weight.size(-1)),
                dtype=torch.float,
            )
        )  # smooth scale, now dpsk use smooth_scale == 1

        assert self.quant_method is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids,
        forward_batch,
        dynamic_scale=None,
        comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ):
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        if forward_batch.is_extend_in_batch:
            final_hidden_states = self.quant_method.apply(
                layer=self,
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                scale=dynamic_scale,
                forward_batch=forward_batch,
                comm_group=comm_group
            )
        else:
            raise NotImplementedError("moe forward not support decode")
        return final_hidden_states

    @staticmethod
    def select_experts(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            use_grouped_topk: bool,
            renormalize: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None,
            e_score_correction_bias: Optional[torch.Tensor] = None,
            routed_scaling_factor: Optional[torch.Tensor] = None
        ):
        # DeekSeekv2 uses grouped_top_k
        # adapt: When num_expert_group=1, it degenerates to fused_topk.
        if use_grouped_topk:  # and num_expert_group != 1:
            # adapt end.
            if topk_group is None:
                raise ValueError(f"Unsupported topk_group is None")
            if num_expert_group is None:
                raise ValueError(f"Unsupported num_expert_group is None")

            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits.float(),
                k=top_k,  # topk is currently 8
                bias=e_score_correction_bias,  # float32
                k_group=topk_group,  # fix: 4
                group_count=num_expert_group,  # fix 8
                group_select_mode=1,  # 0: maximum in group; 1: topk2.sum(fix)
                renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                routed_scaling_factor=routed_scaling_factor,
                eps=float(1e-20))

            row_idx = torch.arange(
                topk_ids.numel(),
                device="npu",
                dtype=torch.int32
            ).view(
                -1, router_logits.shape[0]
            ).transpose(0, 1)
        elif custom_routing_function is None:
            topk_weights, topk_ids, row_idx = FusedMoE.fused_topk(
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
        else:
            topk_weights, topk_ids, row_idx = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids, row_idx

    @staticmethod
    def fused_topk(
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
    ):
        topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(gating_output, k=topk)

        if renormalize:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids, row_idx

    def apply_expert_load_balance(
            self,
            topk_ids: torch.Tensor,
            best_topk_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # omni placement
        if global_server_args_dict["enable_omni_placement"]:
            topk_ids = topk_ids_logical_to_physical(topk_ids, layer_idx_moe=self.moe_layer_idx, expert_mapping=self.expert_mapping)

        # Forced load balance
        #to do: get best_ep from model_extra_config
        import os
        best_ep=os.getenv("BEST_EP", 'False') == 'True'
        if best_ep:
            if best_topk_ids is None:
                t = (topk_ids.shape[0] * 8) // 256
                topk_ids = torch.arange(256, device='npu', dtype=torch.int32).unsqueeze(
                    0).repeat(t + 1, 1).view(-1, 8)[:topk_ids.shape[0]]
            elif global_server_args_dict["enable_torch_compile"]:
                topk_ids = tng.scope.npu_wait_tensor(best_topk_ids, topk_ids)
            else:
                topk_ids = best_topk_ids
        return topk_ids

def get_moe_impl_class():

    if global_server_args_dict["moe_a2a_backend"].is_deepep():
        return FusedMoE

    if get_moe_expert_parallel_world_size() > 1:
        return EPMoE

    return BaseFusedMoE

def moe_infer_fusion(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    scale: torch.Tensor,
    forward_batch,
    comm_group: Optional[ProcessGroup] = None
) -> torch.Tensor:
    world_size = get_world_group().world_size
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    max_num_deployed_expert = layer.w13_weight.shape[0] * get_moe_ep_group().world_size
    expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = (
        torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids.to(torch.int),
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            scale=scale,  # None: non-quant; tensor with shape [num_rows,]: quant
            expert_num=max_num_deployed_expert,
            expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
            expert_tokens_num_flag=True,
            active_expert_range=[0, max_num_deployed_expert],
            quant_mode=1,  # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
        )
    )
    tokens_per_expert_group = tokens_per_expert.new_empty(
        tokens_per_expert.shape[0]
    )
    dist.all_to_all_single(
        tokens_per_expert_group, tokens_per_expert, group=comm_group
    )
    combine_tokens = torch.stack(
        [tokens_per_expert_group, tokens_per_expert], dim=0
    )
    # view: EP, E // EP
    combine_tokens = combine_tokens.view(2, world_size, -1).sum(2)
    all_tokens = combine_tokens[0].sum()
    combine_tokens_cpu = combine_tokens.cpu().tolist()
    input_splits = combine_tokens_cpu[1]
    output_splits = combine_tokens_cpu[0]

    gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
    dist.all_to_all_single(
        gathered_tokens,
        expanded_x,
        output_splits,
        input_splits,
        group=comm_group)

    dynamic_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
    dist.all_to_all_single(
        dynamic_scale,
        pertoken_scale,
        output_splits,
        input_splits,
        group=comm_group)

    # reroute
    (
        hidden_states,
        dynamic_scale,
        topk_ids,
        expert_tokens,
    ) = torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(world_size, -1),
        per_token_scales=dynamic_scale,
    )
    expert_tokens = expert_tokens.to(torch.int64)

    get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
        layer.layer_id,
        [],
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        num_tokens_per_expert=expert_tokens,
    )

    hidden_size = hidden_states.size(-1)

    if dynamic_scale is not None and dynamic_scale.dim() > 1:
        dynamic_scale = dynamic_scale.reshape(-1)
        hidden_states = hidden_states.view(-1, hidden_size)

    # GroupGemm-0
    gateup_output = torch_npu.npu_grouped_matmul(
        [hidden_states],
        [layer.w13_weight],
        group_list=expert_tokens,
        split_item=3,
        group_type=0,
        scale=None,
        per_token_scale=None,
        output_dtype=torch.int32,
        tuning_config=[0],
        group_list_type=1,
    )[0]
    down_input, dynamic_scale = torch_npu.npu_dequant_swiglu_quant(
        gateup_output,
        weight_scale=layer.w13_weight_scale.squeeze(-1),
        activation_scale=dynamic_scale,
        quant_scale=layer.quant_scale,
        group_index=expert_tokens,
        activate_left=True,
        quant_mode=1,
    )

    del gateup_output

    if dynamic_scale is not None and dynamic_scale.dim() > 1:
        inter_size = down_input.size(-1)
        dynamic_scale = dynamic_scale.reshape(-1)
        down_input = down_input.view(-1, inter_size)

    # GroupGemm-1
    hidden_states = torch_npu.npu_grouped_matmul(
        [down_input],
        [layer.w2_weight],
        group_list=expert_tokens,
        split_item=3,
        group_type=0,
        scale=[layer.w2_weight_scale.squeeze(-1)],
        per_token_scale=[dynamic_scale],
        output_dtype=torch.bfloat16,
        tuning_config=[0],
        group_list_type=1,
    )[0]

    new_x = torch.index_select(hidden_states, 0, topk_ids.float().argsort().int())
    gathered_tokens = new_x.new_empty(*expanded_x.shape)
    dist.all_to_all_single(
        gathered_tokens, new_x, input_splits, output_splits, group=comm_group
    )

    return hidden_states, gathered_tokens, expanded_row_idx
