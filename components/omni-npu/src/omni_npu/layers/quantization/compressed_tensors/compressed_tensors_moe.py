# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Callable, Union, Dict
import os

import torch
import torch_npu

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported
from vllm.distributed import get_world_group
from vllm.config import get_current_vllm_config

from omni_npu.layers.prefetch import PrefetchManager
from omni_npu.layers.fused_moe.layer import NPUFusedMoE
from omni_npu.layers.fused_moe.fused_moe import fused_experts_tp
from omni_npu.layers.fused_moe.config import int4_w4a8_moe_quant_config
from omni_npu.layers.fused_moe.fused_moe_method_base import NPUFusedMoEMethodBase
from omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize import PreparePermuteResult
from omni_npu.layers.utils import named_stream
from omni_npu.model_config.config_loader.loader import model_extra_config

torch.npu.config.allow_internal_format = True
logger = init_logger(__name__)


class NPUCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod, NPUFusedMoEMethodBase):
    def __init__(
        self,
        weight_quant,
        parent,
        layer,
    ):
        super().__init__(weight_quant, parent, layer.moe_config)
        NPUFusedMoEMethodBase.__init__(self)
        self.init_eplb(layer)
        self.model_prefetch = PrefetchManager()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.sub_stream = named_stream("sub_stream")
        self.moe_final_stream = named_stream("moe_final_stream")
        self.agrs_overlap_stream = named_stream("moe_agrs_overlap_stream")
        self.enable_agrs_finalize_metadata_overlap = (
            model_extra_config.operator_opt_config.enable_agrs_finalize_metadata_overlap
        )
        self.enable_best_ep = os.getenv("BEST_EP", 'False').lower() == 'true'

    def init_eplb(self, layer):
        self.prefix = layer.layer_name
        self.enable_eplb = layer.enable_eplb and not self.prefix.startswith('mtp')
        self.n_routed_experts = layer.moe_config.num_experts
        self.vllm_config = get_current_vllm_config().model_config.hf_config
        self.num_of_redundant_experts = 0
        if self.enable_eplb:
            from omni_placement.omni_planner import OmniPlanner
            self.planner = OmniPlanner(config_file=None,
                            device="npu",
                            rank=get_world_group().rank_in_group,
                            world_size=get_world_group().world_size,
                            num_experts=self.n_routed_experts,
                            num_redundancy_shared_expert_rank=0)
            self.moe_layer_idx = OmniPlanner.get_deepseek_v3_moe_layer_idx(
                self.prefix, self.vllm_config.first_k_dense_replace,
            )
            self.expert_mapping = self.planner.expert_mapping_on_current_layer(self.moe_layer_idx)

            self.num_of_redundant_experts = self.planner.get_num_of_redundant_experts(
                moe_layer_idx=self.moe_layer_idx,
                num_expert_per_device_origin=layer.local_num_experts,
                rank_device=get_world_group().rank_in_group,
            )
            layer.planner = self.planner
            layer.expert_mapping = self.expert_mapping
            layer.moe_layer_idx = self.moe_layer_idx
            layer.num_of_redundant_experts = self.num_of_redundant_experts
            layer.local_num_experts += self.num_of_redundant_experts

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        params_dtype = torch.int8
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        # WEIGHT_OFFSETS
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(num_experts,
                        2 * intermediate_size_per_partition,
                        1,
                        dtype=torch.float32
                        if params_dtype == torch.float16 else torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(num_experts,
                        hidden_size,
                        1,
                        dtype=torch.float32
                        if params_dtype == torch.float16 else torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)
        
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts,
                            2 * intermediate_size_per_partition,
                            dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts,
                            hidden_size,
                            dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False)

        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)

        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.to(torch.float32).squeeze(-1), requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.to(torch.bfloat16).squeeze(-1), requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Support TP-only mode
        if not layer.moe_parallel_config.use_ep:
            return fused_experts_tp(
                layer=layer,
                x=x_slice,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )

        orig_num_tokens = hidden_states.shape[0]
        strategy, strategy_impl = self.select_communication_strategy(orig_num_tokens)
        is_sequence_parallel = getattr(layer, "is_sequence_parallel", False)
        is_need_slice = (
            not model_extra_config.parall_config.ena_seq_parallel
            and self.tp_size > 1 and not is_sequence_parallel
        )
        x_slice = hidden_states
        if is_need_slice:
            padded_num_tokens = -(orig_num_tokens // -self.tp_size) * self.tp_size
            local_num_tokens = padded_num_tokens // self.tp_size
            num_pads = padded_num_tokens - orig_num_tokens

            if num_pads > 0:
                x_slice = torch.nn.functional.pad(
                    x_slice, (0, 0, 0, num_pads), value=0
                )

            start = self.tp_rank * local_num_tokens
            end = (self.tp_rank + 1) * local_num_tokens
            x_slice = x_slice[start:end]

        if layer.gate is not None:
            router_logits, _ = layer.gate(x_slice)
        else:
            assert router_logits is not None, "Expected gate or router_logits must be provided."
            if is_need_slice:
                if num_pads > 0:
                    router_logits = torch.nn.functional.pad(
                        router_logits, (0, 0, 0, num_pads), value=0
                    )
                router_logits = router_logits[start:end]

        enable_prefetch = model_extra_config.operator_opt_config.enable_prefetch
        cur_stream = torch_npu.npu.current_stream()
        if enable_prefetch:
            self.sub_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.sub_stream):
                self.model_prefetch.prefetch("moe", router_logits, layer=layer)

        topk_weights, topk_ids = NPUFusedMoE.select_experts(
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )
        if self.enable_eplb:
            if self.enable_best_ep:
                topk_ids = self.planner.apply_best_load_balance(
                    topk_ids=topk_ids,
                    global_num_experts=global_num_experts,
                    moe_multi_stream_tune=model_extra_config.operator_opt_config.moe_multi_stream_tune
                )
            else:
                _, topk_ids, _ = self.planner.plan(
                    layer_idx_moe=self.moe_layer_idx,
                    tokens=x_slice,
                    token_expert_ids=topk_ids,
                    token_expert_scores=topk_weights,
                    top_k=top_k,
                    expert_mapping=self.expert_mapping,
                )
                layer.planner = self.planner
                layer.moe_layer_idx = self.moe_layer_idx

        prepare_permute_result = self.apply_prepare_permute(
            strategy_impl, layer, x_slice, topk_ids
        )
        finalize_metadata = None
        use_grouped_matmul_finalize_routing = (strategy == "agrs" and prepare_permute_result.row_idx_type == 1)
        overlap_experts = (
            strategy == "agrs"
            and self.enable_agrs_finalize_metadata_overlap
            and self.agrs_overlap_stream is not None
        )
        if overlap_experts:
            self.agrs_overlap_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.agrs_overlap_stream):
                output = self.apply_experts(
                    layer=layer,
                    prepare_permute_result=prepare_permute_result,
                    activation=activation,
                    use_grouped_matmul_finalize_routing=use_grouped_matmul_finalize_routing
                )
        if strategy == "agrs" and self.enable_agrs_finalize_metadata_overlap:
            if hasattr(self, "prepare_finalize_metadata"):
                def _prepare_finalize_metadata():
                    return self.prepare_finalize_metadata(
                        strategy_impl, layer, topk_weights, prepare_permute_result
                    )
            elif hasattr(strategy_impl, "prepare_finalize_metadata"):
                def _prepare_finalize_metadata():
                    return strategy_impl.prepare_finalize_metadata(
                        layer, topk_weights, prepare_permute_result
                    )
            else:
                _prepare_finalize_metadata = None
            if _prepare_finalize_metadata is not None:
                finalize_metadata = _prepare_finalize_metadata()

        if self.enable_eplb:
            self.planner.record_activation(
                self.moe_layer_idx, prepare_permute_result.expert_tokens,
                support_multi_stream=False,
            )

        finalize_params = None
        if use_grouped_matmul_finalize_routing and model_extra_config.operator_opt_config.moe_multi_stream_tune:
            self.moe_final_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.moe_final_stream):
                finalize_params = strategy_impl.prepare_finalize_params(
                    layer, topk_ids, prepare_permute_result
                )

        if not overlap_experts:
            output = self.apply_experts(
                layer=layer,
                prepare_permute_result=prepare_permute_result,
                activation=activation,
                use_grouped_matmul_finalize_routing=use_grouped_matmul_finalize_routing
            )

        if finalize_params is not None:
            cur_stream.wait_stream(self.moe_final_stream)

        shared_output = None
        if layer.shared_experts is not None:
            self.sub_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.sub_stream):
                if layer.shared_experts.gate_up_proj.tp_size > 1:
                    shared_output = layer.shared_experts(hidden_states)
                else:
                    shared_output = layer.shared_experts(x_slice)

        if enable_prefetch:
            self.sub_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.sub_stream):
                self.model_prefetch.prefetch("next_attn", shared_output, layer=layer)

        if overlap_experts:
            cur_stream.wait_stream(self.agrs_overlap_stream)
        routed_output = self.apply_unpermute_finalize(
            strategy_impl,
            layer,
            output,
            topk_ids,
            topk_weights,
            prepare_permute_result,
            finalize_params=finalize_params,
            finalize_metadata=finalize_metadata,
        )

        if layer.shared_experts is not None or enable_prefetch:
            cur_stream.wait_stream(self.sub_stream)
            
        use_custom_model_add = "omni_custom_models" in os.environ.get("VLLM_PLUGINS", "")
        share_expert_tp = (
            layer.shared_experts.gate_up_proj.tp_size
            if layer.shared_experts is not None else 1
        )
        if layer.shared_experts is not None:
            if share_expert_tp > 1:
                shared_output = tensor_model_parallel_all_reduce(shared_output)
            elif use_custom_model_add:
                routed_output = routed_output + shared_output

        if is_need_slice:
            routed_output = tensor_model_parallel_all_gather(routed_output, dim=0)[:orig_num_tokens]

        if layer.shared_experts is not None and share_expert_tp > 1 and use_custom_model_add:
            routed_output = routed_output + shared_output

        if shared_output is not None:
            return shared_output, routed_output
        return routed_output

    def apply_experts(
        self,
        layer: torch.nn.Module,
        prepare_permute_result: PreparePermuteResult,
        activation: str = "silu",
        use_grouped_matmul_finalize_routing: bool = False
    ) -> torch.Tensor:
        hidden_states = prepare_permute_result.hidden_states_sorted_by_experts
        expert_tokens = prepare_permute_result.expert_tokens
        avg_tokens_per_expert = prepare_permute_result.avg_tokens_per_expert or [0]
        pertoken_scale = prepare_permute_result.dynamic_scale

        if layer.quant_config is None:
            gate_up_proj = torch_npu.npu_grouped_matmul(
                [hidden_states],
                [layer.w13_weight],
                bias=None,
                group_list=expert_tokens,
                split_item=3,
                output_dtype=hidden_states.dtype,
                group_type=0,
                group_list_type=1,
            )[0]
            intermediate_hidden_states = torch_npu.npu_swiglu(gate_up_proj)
            return torch_npu.npu_grouped_matmul(
                [intermediate_hidden_states],
                [layer.w2_weight],
                bias=None,
                group_list=expert_tokens,
                split_item=3,
                output_dtype=hidden_states.dtype,
                group_type=0,
                group_list_type=1,
            )[0]

        if pertoken_scale is None:
            raise ValueError("Quantized path requires dynamic per-token scale.")
        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        gate_up_proj = torch_npu.npu_grouped_matmul(
            [hidden_states],
            [layer.w13_weight],
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
            (expert_tokens.shape[0], layer.w13_weight_scale.shape[-1] // 2),
            dtype=torch.float32,
            device=current_platform.device_type,
        )
        has_bias = getattr(getattr(self, "moe", None), "has_bias", False)
        dequant_swiglu_quant_kwargs: Dict[str, object] = {
            "x": gate_up_proj,
            "weight_scale": layer.w13_weight_scale,
            "activation_scale": pertoken_scale,
            "bias": getattr(layer, "w13_bias", None) if has_bias else None,
            "quant_offset": None,
            "quant_scale": quant_scale,
            "group_index": expert_tokens,
            "activate_left": True,
            "quant_mode": 1,
        }
        if activation == "swigluoai":
            dequant_swiglu_quant_kwargs.update({
                "swiglu_mode": 1,
                "clamp_limit": getattr(layer, "swiglu_limit", 7.0),
                "glu_alpha": getattr(layer, "glu_alpha", 1.702),
                "glu_bias": getattr(layer, "glu_bias", 1.0),
            })
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            **dequant_swiglu_quant_kwargs
        )
        if use_grouped_matmul_finalize_routing:
            return intermediate_h, pertoken_scale
        return torch_npu.npu_grouped_matmul(
            [intermediate_h],
            [layer.w2_weight],
            bias=[layer.w2_bias] if has_bias and hasattr(layer, "w2_bias") else None,
            scale=[layer.w2_weight_scale.to(torch.bfloat16)],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1,
            tuning_config=avg_tokens_per_expert,
        )[0]

    @property
    def supports_eplb(self) -> bool:
        return True


class NPUCompressedTensorsW4A8Int4MoEMethod(NPUCompressedTensorsW8A8Int8MoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(
        self,
        weight_quant,
        parent,
        layer,
    ):
        super().__init__(weight_quant, parent, layer)
        STORAGE_BITS_NPU = 8
        WEIGHT_BITS = 4
        self.warm_up = True
        self.n_total_experts = None
        self.pack_factor = STORAGE_BITS_NPU // WEIGHT_BITS
        self.smooth_scale = None

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({
            "is_transposed": False,
            "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value
        })
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size_per_partition // self.pack_factor,
                                                    hidden_size,
                                                    dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size // self.pack_factor,
                                                   intermediate_size_per_partition,
                                                   dtype=torch.int8),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_int4_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                                1,
                                                                2 * intermediate_size_per_partition,
                                                                dtype=torch.int64),
                                                    requires_grad=False)

        w2_weight_int4_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                                1,
                                                                hidden_size,
                                                                dtype=torch.int64),
                                                    requires_grad=False)

        w13_weight_bias = torch.nn.Parameter(torch.ones(num_experts,
                                                        2 * intermediate_size_per_partition,
                                                        dtype=torch.float32),
                                             requires_grad=False)

        w2_weight_bias = torch.nn.Parameter(torch.ones(num_experts,
                                                       hidden_size,
                                                       dtype=torch.float32),
                                            requires_grad=False)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        layer.register_parameter("w13_weight_int4_scale", w13_weight_int4_scale)
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        set_weight_attrs(w13_weight_int4_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        layer.register_parameter("w2_weight_int4_scale", w2_weight_int4_scale)
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        set_weight_attrs(w2_weight_int4_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False)

        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight, 29)            
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight, 29)

        layer.w13_weight.data = layer.w13_weight.data.view(torch.int32).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.view(torch.int32).contiguous()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ):
        return int4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_int4_scale,
            w2_scale=layer.w2_weight_int4_scale,
            w1_bias=layer.w13_weight_bias,
            w2_bias=layer.w2_weight_bias,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
        )

    def apply_experts(
        self,
        layer: torch.nn.Module,
        prepare_permute_result: PreparePermuteResult,
        activation: str = "silu",
        use_grouped_matmul_finalize_routing: bool = False
    ) -> torch.Tensor:
        hidden_states = prepare_permute_result.hidden_states_sorted_by_experts
        expert_tokens = prepare_permute_result.expert_tokens
        pertoken_scale = prepare_permute_result.dynamic_scale
        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        assert layer.moe_parallel_config.use_ep, "W4A8 only support ep"

        tuning_config = None

        gate_up_proj = torch_npu.npu_grouped_matmul(
            [hidden_states],
            [layer.w13_weight],
            bias=[layer.w13_weight_bias],
            scale=[layer.w13_weight_int4_scale],
            offset=None,
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            activation_input=None,
            activation_quant_scale=None,
            activation_quant_offset=None,
            split_item=3,
            group_type=0,
            group_list_type=1,
            act_type=0,
            tuning_config=tuning_config,
            output_dtype=torch.bfloat16)[0]

        scale_shape = layer.w13_weight_int4_scale.shape
        fake_scale = torch.ones(
            scale_shape, dtype=torch.float32, device="npu",
        ).view(-1, scale_shape[-1])
        pertoken_scale = torch.ones(pertoken_scale.shape, dtype=torch.float32, device="npu")
        gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            gate_up_proj,
            weight_scale=fake_scale,
            activation_scale=pertoken_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1)

        if use_grouped_matmul_finalize_routing:
            return gate_up_proj, pertoken_scale

        return torch_npu.npu_grouped_matmul(
            [gate_up_proj],
            [layer.w2_weight],
            scale=[layer.w2_weight_int4_scale],
            per_token_scale=[pertoken_scale],
            bias=[layer.w2_weight_bias],
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1,
            tuning_config=tuning_config)[0]
