# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from collections.abc import Callable, Iterable
from typing import Optional, cast

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsMambaPrefixCaching,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context

from omni_npu.layers.fused_moe.layer import NPUSharedFusedMoE
from omni_npu.layers.mhc.cube_side_task_ops import (
    maybe_register_mhc_task,
    resolve_mhc_h_res,
)
from omni_npu.layers.mhc.npu_mhc import NPUmHC
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.distributed.parallel_state_ext import get_local_world_group
from omni_npu.v1.layers.attention.npu_pangu import NPUPanguSparseAttention
from omni_npu.v1.layers.logits_processor import NPULogitsProcessor
from omni_npu.v1.layers.vocab_parallel_embedding import NPUParallelLMHead, NPUVocabParallelEmbedding
from omni_npu.v1.utils import on_ascend910b, on_ascend950


logger = init_logger(__name__)

# Singleton marker used to signal "non-fused mhc_pre done, sinkhorn deferred
# to the next block's attention pre_epilog hook". Carried in the sk_event slot
# of (h_post, h_res, sk_event) so it's a single-slot signal that doesn't
# collide with None (sync) or torch.npu.Event (already launched).
_DEFERRED_SINKHORN = object()
try:
    import omni_custom_ops
except ImportError as e:
    logger.warning(f"Failed to import omni_custom_ops: {e}")


def check_ffn_act_fn(act_fn: str) -> None:
    """Validate FFN activation function.

    Note: current NPU fused kernels only support SiLU in this implementation.
    """
    if act_fn != "silu":
        raise ValueError(
            f"Unsupported activation: {act_fn}. Only silu is supported for now."
        )


def _normalize_rope_parameters(
    config: PretrainedConfig, *, max_position_embeddings: int
) -> None:
    """Normalize rope parameters in-place for compatibility.

    Some upstream configs may use `rope_type="default"`. For DeepSeek-style MLA,
    vLLM expects a concrete rope type; we map it to `deepseek_yarn` and fill
    commonly-required defaults.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if not isinstance(rope_params, dict):
        return

    if rope_params.get("rope_type") != "default":
        return

    # Mutate in-place on purpose: vLLM/hf_config is treated as a shared config.
    rope_params["rope_type"] = "deepseek_yarn"
    rope_params.setdefault("factor", 1.0)
    rope_params.setdefault("original_max_position_embeddings", max_position_embeddings)
    rope_params.setdefault("apply_yarn_scaling", False)


def _has_mla_config(config: PretrainedConfig) -> bool:
    """Whether the config contains required MLA fields used by this model."""
    return (
        hasattr(config, "qk_nope_head_dim")
        and hasattr(config, "qk_rope_head_dim")
        and hasattr(config, "v_head_dim")
        and hasattr(config, "kv_lora_rank")
    )


class OpenPanguV2MLP(nn.Module):
    """Dense FFN block used by Pangu layers (and shared experts)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        bias: bool = False,
        reduce_results: bool = True,
        disable_tp: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.disable_tp = disable_tp
        self.tp_size = get_tp_group().world_size
        self.moe_comm_strategy = model_extra_config.operator_opt_config.moe_comm_strategy
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
            return_bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=False,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
            return_bias=False,
        )

        check_ffn_act_fn(hidden_act)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # AllGather for sequence parallelism: when TP > 1 and not using naive backend,
        # each rank has only 1/TP of the sequence, so we need to gather the full sequence
        # before MLP computation
        if not self.disable_tp and self.tp_size > 1:
            if not self.moe_comm_strategy == "allreduce":
                x = get_tp_group().all_gather(x, dim=0)

        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        out = self.down_proj(act)

        # ReduceScatter or AllReduce for sequence parallelism:
        # - "allreduce" strategy: use AllReduce to fully gather across all ranks
        # - other strategies: use ReduceScatter to both reduce and split back to 1/TP sequence per rank
        if not self.disable_tp and self.tp_size > 1:
            if self.moe_comm_strategy == "allreduce":
                out = get_tp_group().all_reduce(out)
            else:
                out = get_tp_group().reduce_scatter(out, dim=0)

        return out


class OpenPanguV2MOE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.tp_size = get_tp_group().world_size
        self.tp_rank = get_tp_group().rank_in_group

        self.routed_scaling_factor = config.routed_scaling_factor
        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        ep_comm_obj = get_ep_group()
        ep_backend = self.ep_group._get_backend(
            torch.device(current_platform.device_type)
        )
        if not on_ascend950():
            self.ep_comm_name = ep_backend.get_hccl_comm_name(
                ep_comm_obj.rank_in_group
            )
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts

        self.on_ascend950 = on_ascend950()
        self.on_ascend910b = on_ascend910b()

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.moe_comm_strategy = model_extra_config.operator_opt_config.moe_comm_strategy
        check_ffn_act_fn(config.hidden_act)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
            params_dtype=torch.float32,
        )
        self.e_score_correction_bias = torch.nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        assert config.n_shared_experts is not None
        intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = OpenPanguV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            parallel_config=parallel_config,
            disable_tp=True,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

        self.use_moe_force_load_balance = (
            os.environ.get('USE_MOE_FORCE_LOAD_BALANCE', 'False').lower() == 'true'
        )
        
        self.n_redundant_experts = eplb_config.num_redundant_experts

        self.aux_load_balance_tensor = torch.arange(
            self.n_routed_experts,
            device=current_platform.device_type,
            dtype=torch.int32,
        ).unsqueeze(0)

        self.experts = NPUSharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )
        self.n_logical_experts = self.n_routed_experts
 
        if self.enable_eplb:
            self.n_physical_experts = self.n_logical_experts + (self.experts.num_of_redundant_experts * self.ep_size)
        else:
            self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        if self.enable_eplb:
            self.experts._expert_map = torch.arange(self.n_physical_experts, dtype=self.experts._expert_map.dtype, device=self.experts._expert_map.device) % self.n_local_physical_experts
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self._is_quant = self.experts.w13_weight.dtype == torch.int8

        self.gmm_fr_token_threshold = model_extra_config.operator_opt_config.gmm_fr_token_threshold
        
        self.moe_seq_split_length = model_extra_config.operator_opt_config.moe_seq_split_length

        self.side_stream = None
        self.fetch_stream = None

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 获取第0维长度（样本/序列维度）
        if isinstance(hidden_states, torch.Tensor):
            total_len = hidden_states.shape[0]
        elif isinstance(hidden_states, dict):
            # 取字典中第一个张量的第0维大小
            total_len = next(iter(hidden_states.values())).shape[0]
        else:
            raise TypeError(f"Unsupported type: {type(hidden_states)}")

        # 2. 长度不超过拆分阈值，直接单步计算
        if total_len <= self.moe_seq_split_length:
            return self._forward_single(hidden_states)

        # 3. 按第0维拆分成多个片段
        if isinstance(hidden_states, torch.Tensor):
            chunks = hidden_states.split(self.moe_seq_split_length, dim=0)
        else:
            # dict 情况：每个 value 都 split，然后 zip 组合
            chunks = [
                dict(zip(hidden_states.keys(), pieces))
                for pieces in zip(
                    *(v.split(self.moe_seq_split_length, dim=0) for v in hidden_states.values())
                )
            ]

        # 4. 逐片前向，并在第0维拼接结果
        outputs = [self._forward_single(c) for c in chunks]
        return torch.cat(outputs, dim=0)
    
    def _forward_single(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route to appropriate forward implementation based on the moe_comm_strategy:
        - "allreduce": MoE → AllReduce (expert parallel, local shared expert)
        - "allgather_reducescatter": AllGather → MoE → ReduceScatter (data parallel, local shared expert)
        - "all2allv": All-to-All mode for MoE
        - "dispatch_combine": Dispatch -> MoE -> Combine
        """
        if self.on_ascend950:
            return self._forward_fused_moe(
                hidden_states
            )
        elif self.tp_size > 1 and self.moe_comm_strategy == "allreduce":
            return self._forward_allgather(
                hidden_states,
                use_allreduce=True,
            )
        elif self.moe_comm_strategy == "all2allv":
            return self._forward_all2allv(
                hidden_states,
            )
        elif self.moe_comm_strategy == "dispatch_combine":
            return self._forward_dispatch_combine(
                hidden_states,
            )
        elif self.moe_comm_strategy == "allgather_reducescatter":
            return self._forward_allgather(
                hidden_states,
                use_allreduce=False,
            )
        else:
            return self._forward_fused_moe(
                hidden_states,
            )

    def _forward_fused_moe(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states.float())
        shared_output, final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_size)

    def _forward_allgather(
        self,
        hidden_states: torch.Tensor,
        use_allreduce: bool = True,
    ) -> torch.Tensor:
        """
        Unified MoE forward with configurable communication pattern.

        Args:
            hidden_states: Input tensor
            use_allreduce:
                - False: AllGather → MoE → ReduceScatter (data parallel, local shared expert)
                - True:  MoE → AllReduce (expert parallel, local shared expert)

        Returns:
            Output tensor with same parallel mode as input
        """

        if isinstance(hidden_states, dict):
            hidden_states_fp32 = hidden_states["hidden_states_fp32"]
            hidden_states = hidden_states["hidden_states_bf16"]
        else:
            assert isinstance(hidden_states, torch.Tensor)
            hidden_states_fp32 = None

        # Save original hidden_states for shared experts (will be computed after init_routing)
        shared_input = hidden_states
        use_side_stream = self.side_stream is not None

        # torch.npu.super_kernel_scope_begin(f"moe_{self.layer_idx}")
        with torch.npu.npugraph_ex.scope.limit_core_num(16,16):
            ENABLE_GMM_FR = hidden_states.shape[0] <= self.gmm_fr_token_threshold

            # Step 1: gating_topk - Select top-k experts
            router_logits, _ = self.gate(hidden_states_fp32) if hidden_states_fp32 is not None else self.gate(hidden_states.to(torch.float32))
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits.to(torch.float32),
                k=self.experts.top_k,
                bias=self.e_score_correction_bias,
                k_group=self.experts.topk_group,
                group_count=self.experts.num_expert_group,
                group_select_mode=1,
                renorm=0,
                norm_type=1, # softmax=0, sigmoid=1
                routed_scaling_factor=self.routed_scaling_factor,
            )

            # forced load balance
            if self.use_moe_force_load_balance: 
                force_k = (topk_ids.shape[0] * self.experts.top_k) // self.n_routed_experts
                topk_ids = self.aux_load_balance_tensor.repeat(force_k+1, 1) \
                            .view(-1, self.experts.top_k)[:topk_ids.shape[0]]


            if self._is_quant:
                hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                if not use_allreduce:
                    hidden_states_int8 = get_ep_group().all_gather(hidden_states_int8, dim=0)

                    topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
                    topk_all = get_ep_group().all_gather(topk_cat, dim=0)
                    topk_weights, topk_ids, pertoken_scale = torch.split(
                        topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
                    topk_ids = torch.round(topk_ids).to(torch.int32)
                    pertoken_scale = pertoken_scale.squeeze(-1)
            else:
                if not use_allreduce:
                    hidden_states = get_ep_group().all_gather(hidden_states, dim=0)
                    topk_ids = get_ep_group().all_gather(topk_ids, dim=0)
                    topk_weights = get_ep_group().all_gather(topk_weights, dim=0)

            # Step 2: init_routing - Initialize routing and get sorted tokens
            expert_range = [self.physical_expert_start, self.physical_expert_end]

            if self._is_quant:
                # Step 2: init_routing - Initialize routing and get sorted tokens
                sorted_tokens, expanded_x_idx, expert_tokens, expanded_activation_scale = torch_npu.npu_moe_init_routing_v2(
                    hidden_states_int8,
                    topk_ids,
                    scale=pertoken_scale,
                    offset=None,
                    active_num=topk_ids.numel(),
                    expert_capacity=-1,
                    expert_num=self.n_physical_experts,
                    drop_pad_mode=0,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    quant_mode=-1,
                    active_expert_range=expert_range,
                    row_idx_type=1 if ENABLE_GMM_FR else 0,
                )

                # Launch shared experts on side stream after init_routing
                # Code structure: trigger here, actual computation at the end
                if use_side_stream:
                    main_stream = torch.npu.current_stream()
                    shared_done_event = torch.npu.Event()
                    shared_pre_event = torch.npu.Event()
                    shared_pre_event.record()

                # Step 3: Int8 group_matmul - First matmul (gate_up projection) -> int32
                gate_up_proj_output = torch_npu.npu_grouped_matmul(
                    [sorted_tokens],
                    [self.experts.w13_weight],
                    bias=None,
                    group_list=expert_tokens,
                    split_item=3,
                    output_dtype=torch.int32,
                    group_type=0,
                    group_list_type=1,
                    act_type=0,
                )[0]

                # Step 4: swiglu - dequant + swiglu + requant
                quant_scale = torch.ones(
                    (expert_tokens.shape[0], self.experts.w13_weight_scale.shape[-1] // 2),
                    dtype=torch.float32,
                    device=sorted_tokens.device,
                )

                intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                    x=gate_up_proj_output,
                    weight_scale=self.experts.w13_weight_scale,
                    activation_scale=expanded_activation_scale,
                    bias=None,
                    quant_offset=None,
                    quant_scale=quant_scale,
                    group_index=expert_tokens,
                    activate_left=True,
                    quant_mode=1,
                )

                if not ENABLE_GMM_FR:
                    # Step 5: INT8 grouped matmul → bf16
                    down_proj_output = torch_npu.npu_grouped_matmul(
                        [intermediate_h],
                        [self.experts.w2_weight],
                        bias=None,
                        scale=[self.experts.w2_weight_scale.to(torch.bfloat16)],
                        per_token_scale=[pertoken_scale],
                        group_list=expert_tokens,
                        split_item=3,
                        output_dtype=torch.bfloat16,
                        group_type=0,
                        group_list_type=1,
                    )[0]

                    # Step 6: finalize_routing - Combine expert outputs
                    moe_output = torch_npu.npu_moe_finalize_routing(
                        down_proj_output.unsqueeze(1),
                        None, None, None,
                        topk_weights.to(torch.float32),
                        expanded_x_idx,
                        topk_ids,
                        drop_pad_mode=3,
                    )
                else:
                    # Step 5+6: Fused GMM2 + FinalizeRouting
                    range1 = torch.arange(0, expanded_x_idx.shape[0], dtype=torch.int32, device=hidden_states.device)
                    range2 = range1 * torch.empty((1), dtype=torch.int32, device=hidden_states.device).fill_(991)
                    mask = (range1 >= torch.sum(expert_tokens)).to(torch.int32)
                    expanded_x_idx += range2 * mask
                    expanded_x_idx = expanded_x_idx % expanded_x_idx.shape[0]
                    expanded_x_idx = torch.clamp(expanded_x_idx, min=0, max=expanded_x_idx.shape[0] - 1)
                    sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
                    sorted_topk_weight = sorted_topk_weight.to(torch.float32)
                    row_index = torch.floor(torch.div(expanded_x_idx, topk_ids.shape[-1])).to(torch.int64)
                    shared_input_fake = torch.zeros((hidden_states.shape[0], hidden_states.shape[1]), dtype=torch.bfloat16, device=hidden_states.device)

                    moe_output = torch_npu.npu_grouped_matmul_finalize_routing(
                        intermediate_h,
                        self.experts.w2_weight,
                        scale=self.experts.w2_weight_scale,
                        bias=None,
                        pertoken_scale=pertoken_scale,
                        group_list=expert_tokens,
                        shared_input=shared_input_fake,
                        logit=sorted_topk_weight,
                        row_index=row_index,
                        output_bs=hidden_states_int8.shape[0],
                        group_list_type=1,
                    )

                    # GMM_FR outputs float32, cast to bfloat16 to match downstream
                    moe_output = moe_output.to(torch.bfloat16)

            else:
                # Original BF16 path
                # Step 2: init_routing - Initialize routing and get sorted tokens
                with torch.npu.npugraph_ex.scope.limit_core_num(1,1):
                    sorted_tokens, expanded_x_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
                        hidden_states,
                        topk_ids,
                        offset=None,
                        active_num=topk_ids.numel(),
                        expert_capacity=-1,
                        expert_num=self.n_physical_experts,
                        drop_pad_mode=0,
                        expert_tokens_num_type=1,
                        expert_tokens_num_flag=True,
                        quant_mode=-1,
                        active_expert_range=expert_range,
                        row_idx_type=0,
                    )

                # Launch shared experts on side stream after init_routing
                # Code structure: trigger here, actual computation at the end
                if use_side_stream:
                    main_stream = torch.npu.current_stream()
                    shared_done_event = torch.npu.Event()
                    shared_pre_event = torch.npu.Event()
                    shared_pre_event.record()

                gate_up_proj_output = torch_npu.npu_grouped_matmul(
                    [sorted_tokens],
                    [self.experts.w13_weight],
                    bias=None,
                    group_list=expert_tokens,
                    split_item=3,
                    output_dtype=torch.bfloat16,
                    group_type=0,
                    group_list_type=1,
                )[0]
                gate_up_proj_output = torch_npu.npu_swiglu(gate_up_proj_output)
                down_proj_output = torch_npu.npu_grouped_matmul(
                    [gate_up_proj_output],
                    [self.experts.w2_weight],
                    bias=None,
                    group_list=expert_tokens,
                    split_item=3,
                    output_dtype=torch.bfloat16,
                    group_type=0,
                    group_list_type=1,
                )[0]

                # Step 6: finalize_routing - Combine expert outputs
                moe_output = torch_npu.npu_moe_finalize_routing(
                    down_proj_output.unsqueeze(1),
                    None, None, None,
                    topk_weights.to(torch.float32),
                    expanded_x_idx,
                    topk_ids,
                    drop_pad_mode=3,
                )

        # Step 7: Apply post-MoE communication
        if use_allreduce:
            routed_output = get_ep_group().all_reduce(moe_output)
        else:
            routed_output = get_ep_group().reduce_scatter(moe_output, dim=0)

        # Step 8: Compute shared experts (triggered after init_routing, executed here)
        # This code structure keeps shared expert computation separate from the main stream
        # (gating_topk -> allgather -> init_routing -> GMM -> finalize) for super kernel fusion
        with torch.npu.npugraph_ex.scope.limit_core_num(8,8):
            if use_side_stream:
                with torch.npu.stream(self.side_stream):
                    shared_input.record_stream(self.side_stream)
                    shared_pre_event.wait(self.side_stream)
                    shared_output = self.shared_experts(shared_input)
                    shared_done_event.record()
            else:
                shared_output = self.shared_experts(shared_input)

        # Combine MoE and shared expert outputs
        if use_side_stream:
            shared_done_event.wait(main_stream)
        result = routed_output + shared_output
        if use_side_stream:
            shared_output.record_stream(main_stream)
        return result

    def _get_mc2_mask(self, num_tokens: int) -> torch.Tensor | None:
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()), None) if attn_metadata else None
        if (
            hasattr(attn_metadata, "decode") and 
            attn_metadata.decode is not None and 
            attn_metadata.num_decode_tokens == attn_metadata.num_actual_tokens
        ):
            mc2_mask = getattr(attn_metadata.decode, "mc2_mask", None)
            if mc2_mask is not None:
                mc2_mask = mc2_mask[:num_tokens] if num_tokens <= mc2_mask.shape[0] else None
            return mc2_mask
        return None

    def _forward_dispatch_combine(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch + Combine mode using NPU fused operators.

        This mode uses npu_moe_distribute_dispatch_v2 and npu_moe_distribute_combine_v2
        to fuse communication and routing for maximum efficiency.

        flow:
            1. gating_topk - Select top-k experts
            2. npu_moe_distribute_dispatch_v2 - fused dispatch + routing
            3. grouped_matmul + swiglu + grouped_matmul - expert computation
            4. npu_moe_distribute_combine_v2 - fused combine + unpermutation
            5. Add shared expert output (overlapped with steps 1-4 via side stream)

        input:
            hidden_states: data parallel across all ep devices

        output:
            data parallel across all ep devices

        Note: NPU fusion operators have a batch size limit of 512. For larger batches,
        we split the input into smaller chunks.
        """
        
        if isinstance(hidden_states, dict):
            hidden_states_fp32 = hidden_states["hidden_states_fp32"]
            hidden_states = hidden_states["hidden_states_bf16"]
        else:
            assert isinstance(hidden_states, torch.Tensor)
            hidden_states_fp32 = None
        
        # Reuse cached HCCL comm name to avoid tracing non-Tensor torch.device
        ep_comm_name = self.ep_comm_name
        # Save original hidden_states for shared experts.
        # hidden_states is read-only throughout this function (gate/topk/dispatch_v2
        # all only read it), so it is safe to hand it to the side stream immediately.
        shared_input = hidden_states
        use_side_stream = self.side_stream is not None

        # Launch shared experts on side stream before gating_topk so that
        # shared_expert overlaps with the full routed-expert pipeline:
        # gating_topk -> dispatch_v2(comm) -> GMM -> combine_v2(comm).
        # Code structure: record the sync point here, submit side-stream work at the end
        # (matches the _forward_allgather pattern to stay outside any kernel-fusion scope).
        if use_side_stream:
            main_stream = torch.npu.current_stream()
            shared_done_event = torch.npu.Event()
            shared_pre_event = torch.npu.Event()
            shared_pre_event.record()

        # Step 1: gating_topk - Select top-k experts
        router_logits, _ = self.gate(hidden_states_fp32) if hidden_states_fp32 is not None else self.gate(hidden_states.to(torch.float32))
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=self.experts.top_k,
            bias=self.e_score_correction_bias,
            k_group=self.experts.topk_group,
            group_count=self.experts.num_expert_group,
            group_select_mode=1,
            renorm=0,
            norm_type=1,  # softmax=0, sigmoid=1
            routed_scaling_factor=self.routed_scaling_factor,
        )

        # forced load balance
        if self.use_moe_force_load_balance:
            force_k = (topk_ids.shape[0] * self.experts.top_k) // self.n_routed_experts
            topk_ids = self.aux_load_balance_tensor.repeat(force_k+1, 1) \
                        .view(-1, self.experts.top_k)[:topk_ids.shape[0]]

        elif self.enable_eplb:
            _, topk_ids, _ = self.experts.planner.plan(
                    layer_idx_moe=self.experts.moe_layer_idx,
                    tokens=hidden_states,
                    token_expert_ids=topk_ids,
                    token_expert_scores=topk_weights,
                    top_k=self.experts.top_k,
                    expert_mapping=self.experts.expert_mapping,
                )
        # Determine quantization mode
        # 0 - no quantization
        # 2 - w8a8 quantization
        quant_mode = 2 if self._is_quant else 0

        # NPU fusion operators have batch size limit of 512
        max_batch_size = 128
        num_tokens = hidden_states.shape[0]
        x_active_mask = self._get_mc2_mask(topk_ids.shape[0])

        if num_tokens <= max_batch_size:
            # Single batch processing
            moe_output = self._dispatch_combine_single_batch(
                hidden_states,
                topk_weights,
                topk_ids,
                quant_mode,
                ep_comm_name,
                x_active_mask,
            )
        else:
            # Multi-batch processing
            outputs = []
            for i in range(0, num_tokens, max_batch_size):
                end_idx = min(i + max_batch_size, num_tokens)
                batch_hidden = hidden_states[i:end_idx]
                batch_topk_weights = topk_weights[i:end_idx]
                batch_topk_ids = topk_ids[i:end_idx]
                batch_x_active_mask = x_active_mask[i:end_idx] if x_active_mask is not None else None

                batch_output = self._dispatch_combine_single_batch(
                    batch_hidden,
                    batch_topk_weights,
                    batch_topk_ids,
                    quant_mode,
                    ep_comm_name,
                    batch_x_active_mask,
                )
                outputs.append(batch_output)

            moe_output = torch.cat(outputs, dim=0)

        # Step 7: Compute shared expert output
        # Submit side-stream work here (after the routed-expert pipeline) so this block
        # stays outside any kernel-fusion scope, matching _forward_allgather convention.
        # On the NPU the side stream waits for shared_pre_event (recorded before gating_topk)
        # and therefore runs in parallel with the entire routed-expert pipeline above.
        if use_side_stream:
            with torch.npu.stream(self.side_stream):
                shared_input.record_stream(self.side_stream)
                shared_pre_event.wait(self.side_stream)
                shared_output = self.shared_experts(shared_input)
                shared_done_event.record()
        else:
            shared_output = self.shared_experts(shared_input)

        # Combine MoE and shared expert outputs
        if use_side_stream:
            shared_done_event.wait(main_stream)
        final_output = moe_output + shared_output
        if use_side_stream:
            shared_output.record_stream(main_stream)

        return final_output

    def _dispatch_combine_single_batch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_mode: int,
        ep_comm_name: str,
        x_active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single batch dispatch + combine processing."""
        # Step 2: npu_moe_distribute_dispatch_v2 - fused dispatch + routing
        dispatch_output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states,
            expert_ids=topk_ids,
            expert_shard_type=0,  # Expert sharding mode
            shared_expert_rank_num=0,  # Shared expert rank number (not used in this mode)
            moe_expert_num=self.n_physical_experts,
            global_bs=0,  # 0 means all tokens are valid
            scales=None,  # Quantization scales (only for int8 mode)
            quant_mode=quant_mode,
            group_ep=ep_comm_name,
            ep_world_size=self.ep_size,
            ep_rank_id=self.ep_rank,
            x_active_mask=x_active_mask, # (MC2 mask, set None for now)
            comm_alg = "fullmesh_v2",
        )
 
        # Unpack dispatch output
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts = dispatch_output[0:6]
 
        # Convert expert_token_nums to int64 for grouped_matmul
        expert_tokens = expert_token_nums.to(torch.int64)

        if self.enable_eplb:
            self.experts.planner.record_activation(self.experts.moe_layer_idx, expert_tokens, support_multi_stream=False)

        if self._is_quant:
            if dynamic_scale is None:
                raise ValueError("Quantized path requires dynamic per-token scale.")
            if dynamic_scale.dim() > 1:
                dynamic_scale = dynamic_scale.reshape(-1)
                expand_x = expand_x.view(-1, expand_x.shape[-1])

            gate_up_proj = torch_npu.npu_grouped_matmul(
                [expand_x],
                [self.experts.w13_weight],
                bias=None,
                scale=None,
                per_token_scale=None,
                group_list=expert_tokens,
                split_item=3,
                output_dtype=torch.int32,
                group_type=0,
                group_list_type=1,
                act_type=0,
            )[0]
            quant_scale = torch.ones(
                (expert_tokens.shape[0], self.experts.w13_weight_scale.shape[-1] // 2),
                dtype=torch.float32,
                device=current_platform.device_type,
            )
            dequant_swiglu_quant_kwargs: Dict[str, object] = {
                "x": gate_up_proj,
                "weight_scale": self.experts.w13_weight_scale,
                "activation_scale": dynamic_scale,
                "bias": None,
                "quant_offset": None,
                "quant_scale": quant_scale,
                "group_index": expert_tokens,
                "activate_left": True,
                "quant_mode": 1,
            }
            intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                **dequant_swiglu_quant_kwargs
            )
            down_proj_output = torch_npu.npu_grouped_matmul(
                [intermediate_h],
                [self.experts.w2_weight],
                bias=None,
                scale=[self.experts.w2_weight_scale.to(torch.bfloat16)],
                per_token_scale=[pertoken_scale],
                group_list=expert_tokens,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1,
            )[0]
        else:
            # Step 3: grouped_matmul - First matmul (gate_up projection)
            # Note: expand_x is already sorted and dispatched to local experts
            gate_up_proj_output = torch_npu.npu_grouped_matmul(
                [expand_x],
                [self.experts.w13_weight],
                bias=None,
                group_list=expert_tokens,
                split_item=3,
                output_dtype=hidden_states.dtype,
                group_type=0,
                group_list_type=1,
            )[0]

            # Step 4: swiglu - Apply activation
            gate_up_proj_output = torch_npu.npu_swiglu(
                gate_up_proj_output,
            )

            # Step 5: grouped_matmul - Second matmul (down projection)
            down_proj_output = torch_npu.npu_grouped_matmul(
                [gate_up_proj_output],
                [self.experts.w2_weight],
                bias=None,
                group_list=expert_tokens,
                split_item=3,
                output_dtype=hidden_states.dtype,
                group_type=0,
                group_list_type=1,
            )[0]

        # Step 6: npu_moe_distribute_combine_v2 - fused combine + unpermutation
        moe_output = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=down_proj_output,
            expert_ids=topk_ids,
            assist_info_for_combine=expand_idx,  # Assist info for combining outputs
            expert_scales=topk_weights.to(torch.float32),
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=self.n_physical_experts,
            global_bs=0,
            ep_send_counts=ep_recv_counts,  # Same as dispatch's receive counts
            group_ep=ep_comm_name,
            ep_world_size=self.ep_size,
            ep_rank_id=self.ep_rank,
            tp_send_counts=tp_recv_counts,
            x_active_mask=x_active_mask, # self._get_mc2_mask(topk_ids.shape[0]) (MC2 mask, set None for now)
        )

        return moe_output

    def _forward_all2allv(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        All-to-All mode.

        flow:
            1. gating_topk - Select top-k experts
            2. npu_moe_init_routing_v2 - Local routing initialization
            3. all_to_all_single - Exchange tokens_per_expert for splits calculation
            4. Calculate input/output splits
            5. all_to_all_single - Exchange tokens data
            6. npu_moe_re_routing - Re-sort tokens after all-to-all
            7. grouped_matmul + swiglu + grouped_matmul - Expert computation
            8. Prepare return data with index_select + argsort
            9. all_to_all_single - Reverse all-to-all for outputs
            10. npu_moe_finalize_routing - Finalize routing
            11. Add shared expert output

        input:
            hidden_states: data parallel across all ep devices

        communication:
            - 3 all_to_all_single operations:
              1. Exchange tokens_per_expert for splits
              2. Exchange input tokens (send to expert ranks)
              3. Exchange output tokens (return from expert ranks)

        output:
            data parallel across all ep devices
        """
        
        if isinstance(hidden_states, dict):
            hidden_states_fp32 = hidden_states["hidden_states_fp32"]
            hidden_states = hidden_states["hidden_states_bf16"]
        else:
            assert isinstance(hidden_states, torch.Tensor)
            hidden_states_fp32 = None
        
        # Step 1: gating_topk - Select top-k experts
        router_logits, _ = self.gate(hidden_states_fp32) if hidden_states_fp32 is not None else self.gate(hidden_states.to(torch.float32))
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=self.experts.top_k,
            bias=self.e_score_correction_bias,
            k_group=self.experts.topk_group,
            group_count=self.experts.num_expert_group,
            group_select_mode=1,
            renorm=0,
            norm_type=1,  # softmax=0, sigmoid=1
            routed_scaling_factor=self.routed_scaling_factor,
        )

        # forced load balance
        if self.use_moe_force_load_balance: 
            force_k = (topk_ids.shape[0] * self.experts.top_k) // self.n_routed_experts
            topk_ids = self.aux_load_balance_tensor.repeat(force_k+1, 1) \
                        .view(-1, self.experts.top_k)[:topk_ids.shape[0]]

        # Step 2: npu_moe_init_routing_v2 - Local routing initialization
        # This sorts tokens locally without communication
        topk_ids_int = topk_ids.int()

        # Expert range for all physical experts across all ranks
        # n_physical_experts is already the global physical expert count
        expert_range = [0, self.n_physical_experts]

        # Determine quantization mode
        quant_mode = -1 # no inner quantization

        if self._is_quant:
            hidden_states_int8, pertoken_scale_init = torch_npu.npu_dynamic_quant(hidden_states)
            hidden_states_view = hidden_states_int8.view(-1, hidden_states_int8.shape[-1])

            expanded_x, expanded_row_idx, expert_tokens, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
                hidden_states_view,
                expert_idx=topk_ids_int,
                scale=pertoken_scale_init,
                expert_num=self.n_physical_experts,
                active_expert_range=expert_range,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_num=topk_ids_int.numel(),
                drop_pad_mode=0,
                row_idx_type=0,
                quant_mode=quant_mode,
            )
        else:
            hidden_states_view = hidden_states.view(-1, hidden_states.shape[-1])
            
            expanded_x, expanded_row_idx, expert_tokens, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
                hidden_states_view,
                expert_idx=topk_ids_int,
                scale=None,
                expert_num=self.n_physical_experts,
                active_expert_range=expert_range,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_num=topk_ids_int.numel(),
                drop_pad_mode=0,
                row_idx_type=0,
                quant_mode=quant_mode,
            )

        # Step 3: all_to_all_single - Exchange tokens_per_expert
        # Shape: (total_experts,) -> (ep_size, n_routed_experts_per_rank)
        tokens_per_expert_group = expert_tokens.new_empty(expert_tokens.shape[0])
        dist.all_to_all_single(
            tokens_per_expert_group,
            expert_tokens,
            group=self.ep_group,
        )

        # Combine tensors, calculate send and receive counts
        combine_tokens = torch.stack([tokens_per_expert_group, expert_tokens], dim=0)
        # Sum over experts to get per-rank counts
        combine_tokens = combine_tokens.view(2, self.ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()

        # Convert to CPU lists for all_to_all_single
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # input_splits: tokens this rank sends to other ranks
        input_splits = combine_tokens_cpu[1]
        # output_splits: tokens this rank receives from other ranks
        output_splits = combine_tokens_cpu[0]

        # Step 4: all_to_all_single - Exchange tokens data
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(
            gathered_tokens,
            expanded_x,
            output_splits,
            input_splits,
            group=self.ep_group,
        )

        # Handle pertoken_scale if needed (not used in current implementation)
        gathered_pertoken_scale = None
        if pertoken_scale is not None:
            gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
            dist.all_to_all_single(
                gathered_pertoken_scale,
                pertoken_scale,
                output_splits,
                input_splits,
                group=self.ep_group,
            )

        # Step 5: npu_moe_re_routing - Re-sort tokens after all-to-all
        # Tokens are now grouped by destination rank, need to sort by expert
        hidden_states_sorted, gathered_pertoken_scale, gathered_idxs_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(
                gathered_tokens,
                tokens_per_expert_group.view(self.ep_size, -1),
                per_token_scales=gathered_pertoken_scale,
            )

        # Step 6: grouped_matmul + activation + grouped_matmul - Expert computation
        if self._is_quant:
            # INT8 path: gate_up projection -> int32 output
            gate_up_proj_output = torch_npu.npu_grouped_matmul(
                [hidden_states_sorted],
                [self.experts.w13_weight],
                bias=None,
                scale=None,
                per_token_scale=None,
                group_list=tokens_per_local_expert,
                split_item=3,
                output_dtype=torch.int32,
                group_type=0,
                group_list_type=1,
                act_type=0,
            )[0]

            # dequant + swiglu + requant
            quant_scale = torch.ones(
                (tokens_per_local_expert.shape[0], self.experts.w13_weight_scale.shape[-1] // 2),
                dtype=torch.float32,
                device=hidden_states_sorted.device,
            )
            intermediate_h, pertoken_scale_down = torch_npu.npu_dequant_swiglu_quant(
                x=gate_up_proj_output,
                weight_scale=self.experts.w13_weight_scale,
                activation_scale=gathered_pertoken_scale,
                bias=None,
                quant_offset=None,
                quant_scale=quant_scale,
                group_index=tokens_per_local_expert,
                activate_left=True,
                quant_mode=1,
            )

            # INT8 down projection -> bf16 output with weight and per-token scales
            down_proj_output = torch_npu.npu_grouped_matmul(
                [intermediate_h],
                [self.experts.w2_weight],
                bias=None,
                scale=[self.experts.w2_weight_scale.to(torch.bfloat16)],
                per_token_scale=[pertoken_scale_down],
                group_list=tokens_per_local_expert,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1,
            )[0]
        else:
            # BF16 path: gate_up projection
            gate_up_proj_output = torch_npu.npu_grouped_matmul(
                [hidden_states_sorted],
                [self.experts.w13_weight],
                bias=None,
                group_list=tokens_per_local_expert,
                split_item=3,
                output_dtype=hidden_states.dtype,
                group_type=0,
                group_list_type=1,
            )[0]

            # swiglu activation
            gate_up_proj_output = torch_npu.npu_swiglu(
                gate_up_proj_output,
            )

            # BF16 down projection
            down_proj_output = torch_npu.npu_grouped_matmul(
                [gate_up_proj_output],
                [self.experts.w2_weight],
                bias=None,
                group_list=tokens_per_local_expert,
                split_item=3,
                output_dtype=hidden_states.dtype,
                group_type=0,
                group_list_type=1,
            )[0]

        # Step 9: Prepare return data with index_select + argsort
        # Sort the output back to the order before all-to-all
        new_x = torch.index_select(
            down_proj_output,
            0,
            gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32),
        )

        # Step 10: all_to_all_single - Reverse all-to-all for outputs
        # Send outputs back to source ranks
        gathered_output_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(
            gathered_output_tokens,
            new_x,
            input_splits,
            output_splits,
            group=self.ep_group,
        )

        # Step 11: npu_moe_finalize_routing - Finalize routing
        moe_output = torch_npu.npu_moe_finalize_routing(
            gathered_output_tokens,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )

        # Step 12: Compute shared expert output
        # Shared experts work on the original input (data parallel mode)
        shared_output = self.shared_experts(hidden_states)

        # Combine MoE and shared expert outputs
        final_output = moe_output + shared_output

        return final_output


class OpenPanguV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig | None,
        prefix: str,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # Optional MHC configuration
        self.mhc_num_stream = getattr(config, "mhc_num_stream", 1)
        self.layer_idx = extract_layer_index(prefix)
        self.use_mhc = hasattr(config, "mhc_num_stream") and config.mhc_num_stream > 1 \
                        and self.layer_idx < config.num_hidden_layers
        self.enable_mhc_multistream = (
            self.use_mhc
            and model_extra_config.operator_opt_config.enable_mhc_multistream
        )

        self.use_mla = _has_mla_config(config)
        if not self.use_mla:
            raise ValueError(
                "PanguV2 expects MLA-related config fields "
                "(qk_nope_head_dim/qk_rope_head_dim/v_head_dim/kv_lora_rank)."
            )

        _normalize_rope_parameters(
            config,
            max_position_embeddings=max_position_embeddings,
        )

        # MHC modules (optional) — declared *before* self_attn/mlp so that
        # state_dict() traversal sees the params under attn_mhc_module/
        # mlp_mhc_module first. When enable_mhc_multistream=True we also
        # register the same NPUmHC instance as a child of self_attn /
        # mlp.experts (further down), and PyTorch dedupes by id; whichever
        # path appears first in the parent's _modules dict wins. The
        # checkpoint expects the attn_mhc_module/mlp_mhc_module prefix.
        if self.use_mhc:
            self.attn_mhc_module = NPUmHC(
                config=config,
                pre_only=False,
                prefix=f"{prefix}.attn_mhc_module",
            )

            self.mlp_mhc_module = NPUmHC(
                config=config,
                pre_only=False,
                prefix=f"{prefix}.mlp_mhc_module",
            )
        else:
            self.attn_mhc_module = None
            self.mlp_mhc_module = None

        self.self_attn = NPUPanguSparseAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=config.rope_theta,
            max_position_embeddings=max_position_embeddings,
            sliding_window_list=getattr(config, "sliding_window_list", []),
            swa_layers=getattr(config, "swa_layers", []),
            param_sink_number=config.param_sink_number,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (
            getattr(config, "n_routed_experts", None) is not None
            and self.layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = OpenPanguV2MOE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = OpenPanguV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                parallel_config=parallel_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.num_hidden_layers = config.num_hidden_layers
        self.first_k_dense_replace = getattr(
            config, "first_k_dense_replace", self.num_hidden_layers
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.pre_mlp_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_mlp_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Block post layernorm (optional)
        if getattr(config, 'block_post_layernorm_idx', None) and self.layer_idx in config.block_post_layernorm_idx:
            self.use_post_norm = True
            self.block_post_layernorm = RMSNorm(
                config.hidden_size * self.mhc_num_stream,
                eps=config.rms_norm_eps,
            )
        else:
            self.use_post_norm = False

        self.side_stream = None
        self.fetch_stream = None

        self.mlp_is_moe = hasattr(self.mlp, "experts")

        # Pre-compute task keys for the cube-side multistream optimization.
        # None means "run sinkhorn synchronously" — either because the
        # optimization is disabled, or because the call site is the dense
        # OpenPanguV2MLP whose down_proj isn't wrapped in a custom op (no cube-side
        # overlap available). The helpers in cube_side_task_ops short-circuit
        # on None, so forward() doesn't need to branch on the flag itself.
        self.attn_mhc_task_key = (
            self.self_attn.o_proj.prefix if self.enable_mhc_multistream else None
        )
        self.mlp_mhc_task_key = (
            self.mlp.experts.layer_name
            if self.enable_mhc_multistream and self.mlp_is_moe
            else None
        )

        # Wire MHC modules onto sub-modules so cube_side_task_ops can look them
        # up via forward_context.no_compile_layers[<layer>].mhc_module. Only
        # needed when the multistream optimization is enabled.
        if self.enable_mhc_multistream:
            self.self_attn.mhc_module = self.attn_mhc_module
            if self.mlp_is_moe:
                self.mlp.experts.mhc_module = self.mlp_mhc_module
        self.use_mhc_fusion_op = model_extra_config.operator_opt_config.use_mhc_fusion_op

    def set_side_stream(self, side_stream: torch.npu.Stream, fetch_stream: torch.npu.Stream = None) -> None:
        """Set the shared side/fetch streams for overlapping computations."""
        self.side_stream = side_stream
        self.fetch_stream = fetch_stream
        # Forward to MoE layer for shared expert overlap and prefetch
        if isinstance(self.mlp, OpenPanguV2MOE):
            self.mlp.side_stream = side_stream
            self.mlp.fetch_stream = fetch_stream
        # Forward to attention layer for future MLA prolog overlap
        self.self_attn.side_stream = side_stream

    def mhc_head(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.npu.Event | None,
    ]:
        use_side_stream = self.side_stream is not None
        residual = hidden_states.clone()
        sk_event: torch.npu.Event | None = None
        if self.use_mhc:
            hidden_states, h_post, h_res = self.attn_mhc_module.mhc_pre(
                hidden_states,
            )
            if use_side_stream:
                ev_pre = torch.npu.Event()
                ev_done = torch.npu.Event()
                ev_pre.record()
                with torch.npu.stream(self.side_stream):
                    ev_pre.wait(self.side_stream)
                    sk_h_res = self.attn_mhc_module.mhc_sinkhorn(h_res)
                    ev_done.record()
                    h_res.record_stream(self.side_stream)
                h_res = sk_h_res
                sk_event = ev_done
            else:
                h_res = self.attn_mhc_module.mhc_sinkhorn(h_res)
        else:
            h_post = None
            h_res = None
        hidden_states = self.input_layernorm(hidden_states)
        return hidden_states, residual, h_post, h_res, sk_event

    def _launch_split_post_res_sinkhorn(
        self,
        residual: torch.Tensor,
        pre_mhc_module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.npu.Event | None]:
        """Run mhc_pre split_post_res + sinkhorn, optionally on side_stream.

        Returns (h_post, h_res, sk_event). sk_event is None when no side stream.
        """
        use_side_stream = self.side_stream is not None
        if use_side_stream:
            ev_pre = torch.npu.Event()
            ev_done = torch.npu.Event()
            ev_pre.record()
            with torch.npu.stream(self.side_stream):
                ev_pre.wait(self.side_stream)
                h_post, h_res = torch.ops.custom.npu_ai_infra_mhc_pre_split_post_res(
                    residual,
                    pre_mhc_module.phi_weight_post_res,
                    pre_mhc_module.branch_alpha_post_res,
                    pre_mhc_module.branch_beta_post_res,
                    norm_eps=pre_mhc_module.norm_eps,
                )
                h_res = pre_mhc_module.mhc_sinkhorn(h_res)
                ev_done.record()
                residual.record_stream(self.side_stream)
            return h_post, h_res, ev_done

        h_post, h_res = torch.ops.custom.npu_ai_infra_mhc_pre_split_post_res(
            residual,
            pre_mhc_module.phi_weight_post_res,
            pre_mhc_module.branch_alpha_post_res,
            pre_mhc_module.branch_beta_post_res,
            norm_eps=pre_mhc_module.norm_eps,
        )
        h_res = pre_mhc_module.mhc_sinkhorn(h_res)
        return h_post, h_res, None

    def mhc_sandwich_norm_post_pre(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        h_post: torch.Tensor | None,
        h_res: torch.Tensor | None,
        post_norm_module,
        post_mhc_module,
        block_norm_module,
        pre_mhc_module,
        pre_norm_module,
        is_model_tail: Optional[bool] = False,
        return_h_in_f32: Optional[bool] = False,
        sk_event: Optional[torch.npu.Event] = None,
        defer_side_launch: bool = False,
    )-> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.npu.Event | None,
    ]:
        use_side_stream = self.side_stream is not None
        main_stream = torch.npu.current_stream() if use_side_stream else None
        new_sk_event: torch.npu.Event | None = None

        if self.use_mhc and self.use_mhc_fusion_op and hidden_states.shape[0] <= 256:
            # Wait for previous side-stream sinkhorn before fusion op consumes h_res.
            if use_side_stream and sk_event is not None:
                sk_event.wait(main_stream)


            if return_h_in_f32:
                hidden_states, residual, hidden_states_fp32 = torch.ops.custom.npu_ai_infra_mhc_sandwich_norm_post_preonly_v2(
                    hidden_states,
                    residual,
                    h_post,
                    h_res,
                    pre_mhc_module.phi_weight_pre,
                    pre_mhc_module.branch_alpha_pre,
                    pre_mhc_module.branch_beta_pre,
                    post_norm_module.weight_fp32,
                    pre_norm_module.weight_fp32,
                    gamma_2=block_norm_module.weight_fp32 if block_norm_module is not None else None,
                    norm_eps=pre_mhc_module.norm_eps,
                    hc_eps=pre_mhc_module.hc_eps,
                    return_h_in_f32=return_h_in_f32,
                )
                hidden_states = {"hidden_states_bf16": hidden_states, "hidden_states_fp32": hidden_states_fp32}
            else:
                hidden_states, residual, _ = torch.ops.custom.npu_ai_infra_mhc_sandwich_norm_post_preonly_v2(
                    hidden_states,
                    residual,
                    h_post,
                    h_res,
                    pre_mhc_module.phi_weight_pre,
                    pre_mhc_module.branch_alpha_pre,
                    pre_mhc_module.branch_beta_pre,
                    post_norm_module.weight_fp32,
                    pre_norm_module.weight_fp32,
                    gamma_2=block_norm_module.weight_fp32 if block_norm_module is not None else None,
                    norm_eps=pre_mhc_module.norm_eps,
                    hc_eps=pre_mhc_module.hc_eps,
                    return_h_in_f32=return_h_in_f32,
                )

            if use_side_stream and sk_event is not None:
                h_res.record_stream(main_stream)

            if not is_model_tail:
                if defer_side_launch and use_side_stream:
                    # Caller will launch split_post_res + sinkhorn from the next
                    # layer's attention pre-epilog hook so the side-stream work
                    # overlaps with v_up / o_proj / all_reduce inside _mla_epilog.
                    # Signal deferral via None values; sk_event stays None too.
                    h_post, h_res = None, None
                else:
                    h_post, h_res, new_sk_event = self._launch_split_post_res_sinkhorn(
                        residual, pre_mhc_module,
                    )
            else:
                h_post, h_res = None, None

        else:
            hidden_states = post_norm_module(hidden_states)
            if self.use_mhc:
                # Wait for previous side-stream sinkhorn (h_res used by mhc_post below).
                if use_side_stream and sk_event is not None:
                    sk_event.wait(main_stream)
                hidden_states = post_mhc_module.mhc_post(
                    hidden_states, h_post, residual, h_res,
                )
                if use_side_stream and sk_event is not None:
                    h_res.record_stream(main_stream)
            else:
                hidden_states = hidden_states + residual

            ###### Block Post LayerNorm (only when MHC is enabled) ######
            if self.use_mhc and block_norm_module is not None:
                hidden_states = block_norm_module(
                    hidden_states.view(-1, self.mhc_num_stream * self.hidden_size)
                )
                hidden_states = hidden_states.view(-1, self.mhc_num_stream, self.hidden_size)
            elif not self.use_mhc and hidden_states.dim() == 3:
                hidden_states = hidden_states.view(-1, self.hidden_size)

            residual = None if is_model_tail else hidden_states.clone()
            if self.use_mhc:
                hidden_states, h_post, h_res = pre_mhc_module.mhc_pre(
                    hidden_states,
                )
                if use_side_stream and not is_model_tail and defer_side_launch:
                    # Non-fused mhc_pre can't be split (the fused npu_mhc_pre
                    # op emits mixed hidden_states + h_post + h_res together,
                    # and the mixed hidden_states is needed on main for
                    # pre_norm_module below). Defer just the sinkhorn step to
                    # the next block's attention pre_epilog_callback so it
                    # overlaps with v_up / o_proj / all_reduce in _mla_epilog.
                    # h_post / h_res are valid; sk_event sentinel signals the
                    # next block to install the hook.
                    new_sk_event = _DEFERRED_SINKHORN
                elif use_side_stream and not is_model_tail:
                    ev_pre = torch.npu.Event()
                    ev_done = torch.npu.Event()
                    ev_pre.record()
                    with torch.npu.stream(self.side_stream):
                        ev_pre.wait(self.side_stream)
                        sk_h_res = pre_mhc_module.mhc_sinkhorn(h_res)
                        ev_done.record()
                        h_res.record_stream(self.side_stream)
                    h_res = sk_h_res
                    new_sk_event = ev_done
                else:
                    h_res = pre_mhc_module.mhc_sinkhorn(h_res)

            hidden_states = pre_norm_module(hidden_states)
            if return_h_in_f32:
                hidden_states = {"hidden_states_bf16": hidden_states, "hidden_states_fp32": hidden_states.to(torch.float32)}

        return hidden_states, residual, h_post, h_res, new_sk_event

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        h_post: torch.Tensor | None,
        h_res: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sk_event: Optional[torch.npu.Event] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.npu.Event | None,
    ]:

        # Clear any stale callback from a previous forward (e.g. installed but
        # never fired because that step was prefill instead of decode).
        if self.use_mhc and self.side_stream is not None:
            self.self_attn.pre_epilog_callback = None

        # If the previous layer's call B deferred its mhc_pre split_post_res +
        # sinkhorn, install a one-shot hook on self_attn that fires inside
        # _forward_decode after the attention core and before _mla_epilog.
        # This puts the side-stream work in parallel with v_up / o_proj /
        # all_reduce on the main stream instead of stalling the next K stream.
        deferred_results: list = []
        deferred = (
            self.use_mhc
            and self.side_stream is not None
            and h_post is None
            and h_res is None
            and residual is not None
        )
        if deferred:
            attn_mhc_module = self.attn_mhc_module
            captured_residual = residual

            def _pre_epilog_hook():
                deferred_results.append(
                    self._launch_split_post_res_sinkhorn(
                        captured_residual, attn_mhc_module,
                    )
                )

            self.self_attn.pre_epilog_callback = _pre_epilog_hook

        # Non-fused counterpart: previous layer's mhc_pre already produced
        # h_post / h_res, only the sinkhorn step is pending. Install a hook
        # that runs mhc_sinkhorn on side_stream so it overlaps with the
        # epilog (v_up / o_proj / all_reduce), then captures the result and
        # a done event for downstream waits.
        deferred_sinkhorn_results: list = []
        deferred_sinkhorn = (
            self.use_mhc
            and self.side_stream is not None
            and sk_event is _DEFERRED_SINKHORN
        )
        if deferred_sinkhorn:
            attn_mhc_module = self.attn_mhc_module
            captured_h_res = h_res
            side_stream = self.side_stream

            def _pre_epilog_sinkhorn_hook():
                ev_pre = torch.npu.Event()
                ev_pre.record()
                ev_done = torch.npu.Event()
                with torch.npu.stream(side_stream):
                    ev_pre.wait(side_stream)
                    sk_h_res = attn_mhc_module.mhc_sinkhorn(captured_h_res)
                    ev_done.record()
                    captured_h_res.record_stream(side_stream)
                deferred_sinkhorn_results.append((sk_h_res, ev_done))

            self.self_attn.pre_epilog_callback = _pre_epilog_sinkhorn_hook

        hidden_states = self.self_attn(
            hidden_states,
            cos, sin,
        )

        if deferred:
            if deferred_results:
                h_post, h_res, sk_event = deferred_results[0]
            else:
                # Hook didn't fire (prefill path took a non-decode branch).
                # Run synchronously now so call A has its inputs ready.
                self.self_attn.pre_epilog_callback = None
                h_post, h_res, sk_event = self._launch_split_post_res_sinkhorn(
                    residual, self.attn_mhc_module,
                )

        if deferred_sinkhorn:
            if deferred_sinkhorn_results:
                h_res, sk_event = deferred_sinkhorn_results[0]
            else:
                # Hook didn't fire — run sinkhorn synchronously so call A
                # gets a valid h_res. sk_event becomes None (sync done).
                self.self_attn.pre_epilog_callback = None
                h_res = self.attn_mhc_module.mhc_sinkhorn(h_res)
                sk_event = None

        hidden_states, residual, h_post, h_res, sk_event = self.mhc_sandwich_norm_post_pre(
            hidden_states,
            residual,
            h_post,
            h_res,
            self.post_attention_layernorm,
            self.attn_mhc_module,
            None,
            self.mlp_mhc_module,
            self.pre_mlp_layernorm,
            is_model_tail=False,
            return_h_in_f32=self.layer_idx >= self.first_k_dense_replace,
            sk_event=sk_event,
        )

        hidden_states = self.mlp(
            hidden_states,
        )

        nxt_mhc_pre, nxt_layernorm, is_model_tail = self._tail_refs

        hidden_states, residual, h_post, h_res, sk_event = self.mhc_sandwich_norm_post_pre(
            hidden_states,
            residual,
            h_post,
            h_res,
            self.post_mlp_layernorm,
            self.mlp_mhc_module,
            self.block_post_layernorm if self.use_post_norm else None,
            nxt_mhc_pre,
            nxt_layernorm,
            is_model_tail,
            return_h_in_f32=False,
            sk_event=sk_event,
            defer_side_launch=not is_model_tail,
        )

        return hidden_states, residual, h_post, h_res, sk_event

    def _forward_naive(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Original (pre-refactor) forward kept as a reference / fallback.
        """
        ###### Attention Block with MHC & SandwichNorm ######
        residual = hidden_states.clone()
        if self.use_mhc:
            hidden_states, h_post, h_res = self.attn_mhc_module.mhc_pre(
                hidden_states,
            )
            h_res = maybe_register_mhc_task(
                self.self_attn.prefix, self.attn_mhc_task_key, h_res,
            )

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos, sin
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_mhc:
            h_res = resolve_mhc_h_res(
                self.attn_mhc_module, self.attn_mhc_task_key, h_res,
            )
            hidden_states = self.attn_mhc_module.mhc_post(
                hidden_states,
                h_post,
                residual,
                h_res,
            )
        else:
            hidden_states = hidden_states + residual


        ###### FFN Block with MHC & SandwichNorm ######
        residual = hidden_states.clone()
        if self.use_mhc:
            hidden_states, h_post, h_res = self.mlp_mhc_module.mhc_pre(
                hidden_states,
            )
            h_res = maybe_register_mhc_task(
                self.mlp_mhc_task_key, self.mlp_mhc_task_key, h_res,
            )

        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
        )
        hidden_states = self.post_mlp_layernorm(hidden_states)

        if self.use_mhc:
            h_res = resolve_mhc_h_res(
                self.mlp_mhc_module, self.mlp_mhc_task_key, h_res,
            )
            hidden_states = self.mlp_mhc_module.mhc_post(
                hidden_states,
                h_post,
                residual,
                h_res,
            )
        else:
            hidden_states = hidden_states + residual

        ###### Block Post LayerNorm (only when MHC is enabled)
        if self.use_post_norm and self.use_mhc:
            hidden_states = self.block_post_layernorm(
                hidden_states.view(-1, self.mhc_num_stream * self.hidden_size)
            )
            hidden_states = hidden_states.view(-1, self.mhc_num_stream, self.hidden_size)

        # Flatten output if MHC was disabled
        elif not self.use_mhc and hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, self.hidden_size)

        return hidden_states


def _maybe_padding_and_slice(
    hidden_states: torch.Tensor,
    padding_only: bool = False,
):
    """
    Slice hidden_states evenly across tensor parallel ranks along sequence dimension.

    When tensor parallel is enabled (TP > 1), each rank processes
    only 1/TP of the sequence (tokens). This reduces communication overhead
    by only all-gathering when necessary (e.g., in Attention and MoE layers).

    If num_tokens is not divisible by tp_size, pads with zeros to make it divisible.

    Args:
        hidden_states: Full hidden states tensor [num_tokens, hidden_size]

    Returns:
        tuple: (sliced_hidden_states, original_num_tokens)
            - sliced_hidden_states: Tensor of shape [padded_num_tokens/tp_size, hidden_size]
            - original_num_tokens: Original number of tokens (without padding)
    """

    tp_size = get_tp_group().world_size
    if tp_size == 1:
        # No tensor parallelism, return as-is
        return hidden_states, hidden_states.shape[0]

    # Save original number of tokens
    original_num_tokens = hidden_states.shape[0]

    # Pad if necessary to make num_tokens divisible by tp_size
    num_tokens = original_num_tokens
    pad_size = (tp_size - (num_tokens % tp_size)) % tp_size
    if pad_size > 0:
        padding = torch.zeros(
            (pad_size, *hidden_states.shape[1:]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = torch.cat(
            [hidden_states, padding], dim=0,
        )
        num_tokens = hidden_states.shape[0]

    if not padding_only:
        tp_rank = get_tp_group().rank_in_group
        local_num_tokens = num_tokens // tp_size
        hidden_states = torch.narrow(
            hidden_states,
            dim=0,
            start=tp_rank * local_num_tokens,
            length=local_num_tokens,
        )

    return hidden_states, original_num_tokens


def _maybe_gather_and_unpadding(
    hidden_states: torch.Tensor,
    original_num_tokens: int,
) -> torch.Tensor:
    """
    AllGather hidden_states from all tensor parallel ranks along sequence dimension.
    Also removes any padding that was added during slicing.

    Reconstructs the full hidden states by gathering from all TP ranks.

    Args:
        hidden_states: Sliced hidden states [padded_num_tokens/tp_size, hidden_size]
        original_num_tokens: Original number of tokens (without padding)

    Returns:
        Full hidden states [original_num_tokens, hidden_size] gathered from all TP ranks
    """

    tp_size = get_tp_group().world_size
    if tp_size == 1:
        # No tensor parallelism, return as-is
        return hidden_states

    # AllGather the hidden_states along the sequence dimension (dim=0)
    hidden_states = get_tp_group().all_gather(hidden_states, dim=0)

    # Remove padding if present
    if hidden_states.shape[0] > original_num_tokens:
        hidden_states = hidden_states[:original_num_tokens]

    return hidden_states


@support_torch_compile
class OpenPanguV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config
        self.config = config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.hidden_size = config.hidden_size
        # Optional MHC configuration
        self.mhc_num_stream = getattr(config, "mhc_num_stream", 1)
        self.use_mhc = hasattr(config, "mhc_num_stream") and self.mhc_num_stream > 1

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = NPUVocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OpenPanguV2DecoderLayer(config, prefix, vllm_config),
            prefix=f"{prefix}.layers",
        )

        # Merge MHC module (optional)
        if self.use_mhc:
            self.merge_mhc_module = NPUmHC(
                config=config,
                pre_only=True,
                prefix=f"{prefix}.merge_mhc_module",
            )
        else:
            self.merge_mhc_module = None

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.dsa_layers = getattr(config, "dsa_layers", [])
        self.swa_layers = getattr(config, "swa_layers", [])
        if self.dsa_layers and len(self.dsa_layers) > 0:
            self.dsa_layer_name = self.layers[self.dsa_layers[0]].self_attn.attn.layer_name 
        else:
            self.dsa_layer_name = None
                                
        self.swa_layer_name = self.layers[self.swa_layers[0]].self_attn.attn.layer_name \
                                if self.swa_layers is not None else None
        self.cos_cached = self.layers[self.start_layer].self_attn.rotary_emb.cos_cached
        self.sin_cached = self.layers[self.start_layer].self_attn.rotary_emb.sin_cached
        self.attn_layer_name = self.layers[self.start_layer].self_attn.attn.layer_name
        self.need_tp_padding = not model_extra_config.operator_opt_config.moe_comm_strategy == "allreduce"
        self.param_sink_number = config.param_sink_number

        last_layer_idx = config.num_hidden_layers - 1
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i == last_layer_idx:
                layer._tail_refs = (
                    self.merge_mhc_module,
                    self.norm,
                    True,  # is_model_tail
                )
            else:
                nxt = self.layers[i + 1]
                layer._tail_refs = (
                    nxt.attn_mhc_module,
                    nxt.input_layernorm,
                    False,  # is_model_tail
                )

        use_multi_stream = model_extra_config.operator_opt_config.enable_multi_stream
        if use_multi_stream:
            self.side_stream = torch.npu.Stream()
            self.fetch_stream = torch.npu.Stream()
            for i in range(self.start_layer, self.end_layer):
                self.layers[i].set_side_stream(self.side_stream, self.fetch_stream)
            logger.info("Multi-stream enabled: side_stream and fetch_stream created")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        ### Add padding for sequence parallel (TP > 1 with non-naive backend)
        if self.need_tp_padding:
            hidden_states, original_num_tokens = _maybe_padding_and_slice(hidden_states)

        # Reshape for MHC if enabled
        if self.use_mhc:
            hidden_states = hidden_states.view(-1, 1, self.hidden_size) \
                                         .repeat(1, self.mhc_num_stream, 1)

        cos = self.cos_cached.index_select(dim=0, index=positions)
        sin = self.sin_cached.index_select(dim=0, index=positions)

        hidden_states, residual, h_post, h_res, sk_event = self.layers[0].mhc_head(hidden_states)
        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual, h_post, h_res, sk_event = self.layers[i](
                hidden_states,
                residual,
                h_post,
                h_res,
                cos, sin,
                sk_event,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # Unpad hidden_states: after all layers, gather and remove any padding that was added
        # for sequence parallelism to restore the original number of tokens
        if self.need_tp_padding:
            hidden_states = _maybe_gather_and_unpadding(hidden_states, original_num_tokens)

        return hidden_states


class OpenPanguV2ForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsLoRA,
    MixtureOfExperts,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.mome_state_shape(
            q_lora_rank=hf_config.q_lora_rank,
            kv_lora_rank=hf_config.kv_lora_rank,
            num_heads=hf_config.num_attention_heads,
            v_head_dim=hf_config.v_head_dim,
            kernel_size=getattr(hf_config, "router_sliding_window", 1),
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mome_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = OpenPanguV2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        parall_cfg = model_extra_config.parall_config
        if parall_cfg.ena_local_lmhead_parallel:
            local_lmhead = get_local_world_group().world_size > 1
            dp_lmhead = False
        elif parall_cfg.ena_dp_lmhead_parallel:
            local_lmhead = False
            dp_lmhead = get_dp_group().world_size > 1
        else:
            local_lmhead = False
            dp_lmhead = False
        if get_pp_group().is_last_rank:
            self.lm_head = NPUParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
                local_lmhead_parallel=local_lmhead,
                dp_parallel=dp_lmhead,
            )
            if config.tie_word_embeddings and not local_lmhead and not dp_lmhead:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = NPULogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_expert_groups = 1

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, OpenPanguV2DecoderLayer)
            if isinstance(layer.mlp, OpenPanguV2MOE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No MOE layer found in model.layers.")

        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.n_routed_experts = example_moe.n_routed_experts
        self.n_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
        )
        return logits


    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, OpenPanguV2MOE):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping: list[tuple[str, str, int]] = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # MLA packed weights: present on some checkpoints/configs.
        mla_params_mapping: list[tuple[str, str, int]] = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        stacked_params_mapping.extend(mla_params_mapping)

        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict: dict[str, torch.nn.Parameter] = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def _skip_weight(name: str) -> bool:
            if "rotary_emb.inv_freq" in name:
                return True
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                return True
            return get_spec_layer_idx_from_weight_name(self.config, name) is not None

        def _try_load_stacked(name: str, loaded_weight: torch.Tensor) -> str | None:
            """Try loading weights that are stacked/packed into a single param."""
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # Some checkpoints have expert weights; ignore if param doesn't exist.
                if "mlp.experts." in name and name not in params_dict:
                    continue

                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue

                if is_pp_missing_parameter(name_mapped, self):
                    continue

                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                return name_mapped
            return None

        def _try_load_expert(name: str, loaded_weight: torch.Tensor) -> str | None:
            """Try loading per-expert weights via fused MoE mapping."""
            saw_expert_key = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue

                saw_expert_key = True
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue

                param = params_dict[name_mapped]
                weight_loader = cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    return name_mapped

            # If it looks like an expert weight but couldn't be loaded, skip it.
            if saw_expert_key:
                return ""
            return None

        def _normalize_weight_name(name: str) -> str:
            """Normalize weight name for compatibility across different checkpoints."""
            # Map k_layernorm to kv_a_layernorm for backward compatibility
            if "k_layernorm" in name:
                return name.replace("k_layernorm", "kv_a_layernorm")
            return name

        for name, loaded_weight in weights:
            if _skip_weight(name):
                continue

            loaded_name = _try_load_stacked(name, loaded_weight)
            if loaded_name is not None:
                loaded_params.add(loaded_name)
                continue

            loaded_name = _try_load_expert(name, loaded_weight)
            if loaded_name == "":
                continue
            if loaded_name is not None:
                loaded_params.add(loaded_name)
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            if remapped_name is None:
                continue

            if is_pp_missing_parameter(remapped_name, self):
                continue

            remapped_name = _normalize_weight_name(remapped_name)
            if remapped_name not in params_dict: 
                print(f"Skip loading {remapped_name}.")
                continue

            param = params_dict[remapped_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(remapped_name)

            # MLA multi-stream split path: q_b_proj's weight_loader (installed
            # by NPUPanguSparseAttention._install_q_b_split_loaders) also
            # populates the q_b_nope_proj / q_b_pe_proj children's matching
            # params in-place. vLLM's "unloaded weights" check looks at
            # state_dict keys, so mark those synthetic names as loaded too
            # whenever the parent q_b_proj key was just consumed.
            if ".q_b_proj." in remapped_name:
                for synthetic in (
                    remapped_name.replace(".q_b_proj.", ".q_b_nope_proj."),
                    remapped_name.replace(".q_b_proj.", ".q_b_pe_proj."),
                ):
                    if synthetic in params_dict:
                        loaded_params.add(synthetic)

        return loaded_params


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> int | None:
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None