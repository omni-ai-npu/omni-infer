# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import re
from typing import Dict, Optional

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from omni_npu.layers.fused_moe.config import hifloat8_moe_quant_config
from omni_npu.layers.fused_moe.fused_moe_method_base import NPUFusedMoEMethodBase
from omni_npu.layers.fused_moe.layer import NPUFusedMoE
from omni_npu.layers.mhc import cube_side_task_ops  # noqa: F401  registers cube-side ops
from omni_npu.layers.utils import named_stream
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.layers.linear import FlashCommLinearMethodBase

logger = init_logger(__name__)

HIFLOAT8 = "hifloat8"


def _is_layer_ignored(prefix: str, ignored_layers: list[str]) -> bool:
    for pattern in ignored_layers:
        if pattern.startswith("re:"):
            if re.match(pattern[3:], prefix):
                return True
        elif pattern == prefix or pattern in prefix:
            return True
    return False


@register_quantization_config(HIFLOAT8)
class Hifloat8Config(QuantizationConfig):
    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        ignored_layers = config.get("ignore", None) or None
        return cls(ignored_layers=ignored_layers)

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return HIFLOAT8

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "NPU hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method_custom(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from omni_npu.v1.layers.fused_mlp.layer import FusedMLP
        from omni_npu.v1.layers.linear import FlashCommLinearBase, UnquantizedFlashCommLinearMethod

        if isinstance(layer, FlashCommLinearBase):
            if self.ignored_layers and _is_layer_ignored(prefix, self.ignored_layers):
                return UnquantizedFlashCommLinearMethod()
            return Hifloat8FCLinearMethod(self)
        elif isinstance(layer, LinearBase):
            if self.ignored_layers and _is_layer_ignored(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Hifloat8LinearMethod(self)
        elif isinstance(layer, FusedMLP):
            return Hifloat8MlpMethod(self)
        elif isinstance(layer, FusedMoE):
            return Hifloat8MoEMethod(self, layer)
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        vllm_plugins = os.environ.get("VLLM_PLUGINS", "")
        custom_model_enabled = "omni_custom_models" in vllm_plugins
        if custom_model_enabled:
            return self.get_quant_method_custom(layer, prefix)
        
        raise NotImplementedError(
            "Hifloat8 quantization method is only implemented for custom models. "
            "Please set VLLM_PLUGINS environment variable to include "
            "\"omni_custom_models\" to enable it, or implement the method for non-custom models."
        )


class Hifloat8LinearMethod(LinearMethodBase):
    """Hifloat8 quantization method for vLLM LinearBase layers on NPU."""

    def __init__(self, quant_config: Hifloat8Config):
        self.quant_config = quant_config

    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((output_size_per_partition, 1), dtype=params_dtype),
            output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer):
        layer.weight = torch.nn.Parameter(
            layer.weight.data.t().contiguous(), requires_grad=False,
        )
        ws_data = layer.weight_scale.data.squeeze(-1).float()
        layer.weight_scale = torch.nn.Parameter(
            torch_npu.npu_trans_quant_param(ws_data), requires_grad=False,
        )

    def apply(self, layer, x, bias=None):
        layer_key = getattr(layer, "prefix", "") or ""
        if bias is not None:
            bias = bias.to(torch.float32)
        if isinstance(x, dict):
            x = x.get('x_hif8')
        else:
            x = torch_npu.npu_dtype_cast(x, torch_npu.hifloat8)

        # Cube-side overlap (opaque to Dynamo): if a task was registered
        # under layer_key, fire it on the cube-side stream now.
        x = torch.ops.vllm.cube_side_run(layer_key, x)

        y = torch_npu.npu_quant_matmul(
            x1=x, x2=layer.weight, scale=layer.weight_scale, pertoken_scale=None, bias=bias,
            x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8, output_dtype=layer.orig_dtype,
        )

        y = torch.ops.vllm.cube_side_wait(layer_key, y)
        return y


class Hifloat8FCLinearMethod(FlashCommLinearMethodBase):
    def __init__(self, quant_config: Hifloat8Config):
        self.quant_config = quant_config

    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=params_dtype),
            output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer):
        layer.weight = torch.nn.Parameter(
            layer.weight.data.t().contiguous(), requires_grad=False,
        )
        ws_data = layer.weight_scale.data.squeeze(-1).float()
        layer.weight_scale = torch.nn.Parameter(
            torch_npu.npu_trans_quant_param(ws_data), requires_grad=False,
        )

    def apply(self, layer, x, bias=None, x_transform=None, x_dim=0, throw_dequant=False):
        from omni_npu.v1.distributed.communication_op_ext import (
            layer_parallel_all_gather,
            layer_parallel_all2all_single,
        )

        if bias is not None:
            bias = bias.to(torch.float32)
        if isinstance(x, dict):
            x = x.get('x_hif8', None)
        else:
            x = torch_npu.npu_dtype_cast(x, torch_npu.hifloat8)

        if x_transform == "AllGather":
            x = layer_parallel_all_gather(x, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == "ALL2ALL":
            x = layer_parallel_all2all_single(x, layer.layer_name_inside_block, "x", x_dim)

        y = torch_npu.npu_quant_matmul(
            x1=x, x2=layer.weight, scale=layer.weight_scale, pertoken_scale=None, bias=bias,
            x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8, output_dtype=layer.orig_dtype,
        )
        return y


class Hifloat8MlpMethod:
    def __init__(self, quant_config: Hifloat8Config):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer):
        layer.gate_up_proj.weight_scale = torch.nn.Parameter(
            layer.gate_up_proj.weight_scale.data.float(), requires_grad=False,
        )

    def apply_quant(self, x, x_transform=None, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            x_scale = torch.ones((x.shape[0],), dtype=torch.float32, device=x.device)
            x = torch_npu.npu_dtype_cast(x, torch_npu.hifloat8)
        return x, x_scale

    def apply_part1_gate_up_on_stream(self, layer, x, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            gate_up, _ = layer.gate_up_proj(x, throw_dequant=False)
        return gate_up

    def apply_part2_activation_on_stream(self, layer, gate_up, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            x = {'x_hif8': gate_up, 'quant_type': 1}
            x = layer.act_fn(x, quant_symbol=True)
        return x

    def apply_part3_down_on_stream(self, layer, x, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            output, _ = layer.down_proj(x)
        return output

    def apply(self, layer, x, stream_label=None):
        x, x_scale = self.apply_quant(x, stream_label)
        x = {'x_hif8': x, 'pertoken_scale': x_scale}
        gate_up = self.apply_part1_gate_up_on_stream(layer, x, stream_label)
        x = self.apply_part2_activation_on_stream(layer, gate_up, stream_label)
        output = self.apply_part3_down_on_stream(layer, x, stream_label)
        return output


class Hifloat8MoEMethod(FusedMoEMethodBase, NPUFusedMoEMethodBase):

    def __init__(self, quant_config: Hifloat8Config, layer):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        NPUFusedMoEMethodBase.__init__(self)
        self.quant_config = quant_config
        self.moe = layer.moe_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.shared_experts_stream = named_stream("sub_stream")

        self.n_routed_experts = layer.moe_config.num_experts
        self.prefix = layer.layer_name
        self.vllm_config = get_current_vllm_config().model_config.hf_config
        self.num_of_redundant_experts = 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        params_dtype = torch.uint8
        num_experts = num_experts + self.num_of_redundant_experts

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

        # WEIGHT_OFFSETS
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, 1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, 2 * intermediate_size_per_partition, dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ):
        return hifloat8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False,
        )
        ws13 = layer.w13_weight_scale.data.squeeze(-1).float()
        layer.w13_weight_scale = torch.nn.Parameter(
            torch_npu.npu_trans_quant_param(ws13), requires_grad=False,
        )
        ws2 = layer.w2_weight_scale.data.squeeze(-1).float()
        layer.w2_weight_scale = torch.nn.Parameter(
            torch_npu.npu_trans_quant_param(ws2), requires_grad=False,
        )
        layer.ensure_moe_quant_config_init()

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
        custom_routing_function=None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ):
        orig_num_tokens = hidden_states.shape[0]
        strategy, strategy_impl = self.select_communication_strategy(orig_num_tokens)

        is_need_slice = self.tp_size > 1 and (strategy == "all2all" or strategy == "dispatch_combine")
        x_slice = hidden_states
        if is_need_slice:
            padded_num_tokens = -(orig_num_tokens // -self.tp_size) * self.tp_size
            local_num_tokens = padded_num_tokens // self.tp_size
            num_pads = padded_num_tokens - orig_num_tokens

            if num_pads > 0:
                x_slice = torch.nn.functional.pad(x_slice, (0, 0, 0, num_pads), value=0)

            start = self.tp_rank * local_num_tokens
            end = (self.tp_rank + 1) * local_num_tokens
            x_slice = x_slice[start:end]

        if layer.gate is not None:
            router_logits, _ = layer.gate(x_slice)
        else:
            assert router_logits is not None, "Expected gate or router_logits must be provided."
            if is_need_slice:
                if num_pads > 0:
                    router_logits = torch.nn.functional.pad(router_logits, (0, 0, 0, num_pads), value=0)
                router_logits = router_logits[start:end]

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

        prepare_permute_result = self.apply_prepare_permute(strategy_impl, layer, x_slice, topk_ids)

        use_grouped_matmul_finalize_routing = (strategy == "agrs" and prepare_permute_result.row_idx_type == 1)
        output = self.apply_experts(
            layer=layer,
            prepare_permute_result=prepare_permute_result,
            activation=activation,
            use_grouped_matmul_finalize_routing=use_grouped_matmul_finalize_routing,
        )

        shared_output = None
        multi_stream = model_extra_config.operator_opt_config.shared_expert_multi_stream
        schedule = model_extra_config.operator_opt_config.shared_expert_parallel_schedule
        if multi_stream and schedule == "with_routed_experts_cv":
            # Shared experts already ran on the side stream interleaved with the
            # routed-experts compute/vector path inside apply_experts; just unpack.
            output, shared_output = output  # output of self.apply_experts is a tuple
        elif multi_stream:
            # default with_finalize: launch shared experts on the side stream so
            # they overlap apply_unpermute_finalize below.
            if layer.shared_experts is not None:
                cur_stream = torch.npu.current_stream()
                self.shared_experts_stream.wait_stream(cur_stream)
                with torch.npu.stream(self.shared_experts_stream):
                    if layer.shared_experts.gate_up_proj.tp_size > 1:
                        shared_output = layer.shared_experts(hidden_states)
                    else:
                        shared_output = layer.shared_experts(x_slice)
        else:
            # Multi-stream disabled — run shared experts synchronously on the
            # main stream. Schedule is ignored.
            if layer.shared_experts is not None:
                if layer.shared_experts.gate_up_proj.tp_size > 1:
                    shared_output = layer.shared_experts(hidden_states)
                else:
                    shared_output = layer.shared_experts(x_slice)

        routed_output = self.apply_unpermute_finalize(
            strategy_impl, layer, output, topk_ids, topk_weights, prepare_permute_result,
        )

        if multi_stream and schedule == "with_finalize":
            if layer.shared_experts is not None:
                cur_stream.wait_stream(self.shared_experts_stream)
                if layer.shared_experts.gate_up_proj.tp_size > 1:
                    shared_output = tensor_model_parallel_all_reduce(shared_output)

        if is_need_slice:
            routed_output = tensor_model_parallel_all_gather(routed_output, dim=0)[:orig_num_tokens]

        if shared_output is not None:
            return shared_output, routed_output
        return routed_output

    def apply_experts(
        self,
        layer: torch.nn.Module,
        prepare_permute_result,
        activation: str = "silu",
        use_grouped_matmul_finalize_routing: bool = False,
    ) -> torch.Tensor:
        hidden_states = prepare_permute_result.hidden_states_sorted_by_experts
        expert_tokens = prepare_permute_result.expert_tokens
        avg_tokens_per_expert = prepare_permute_result.avg_tokens_per_expert or [0]
        pertoken_scale = prepare_permute_result.dynamic_scale
        if pertoken_scale is not None and pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        run_shared_with_cv = (
            model_extra_config.operator_opt_config.shared_expert_multi_stream
            and model_extra_config.operator_opt_config.shared_expert_parallel_schedule == "with_routed_experts_cv"
        )

        if run_shared_with_cv:
            shared_expert_gate_up_proj_finished_event = prepare_permute_result.shared_expert_gate_up_proj_finished_event
            shared_expert_gate_up_proj_finished_event.wait(torch.npu.current_stream())

            self.shared_experts_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(self.shared_experts_stream):
                shared_expert_gate_up = prepare_permute_result.shared_expert_gate_up
                shared_expert_act = layer.shared_experts.act_fn(shared_expert_gate_up)
                shared_expert_act = torch_npu.npu_dtype_cast(shared_expert_act, torch_npu.hifloat8)

        gate_up_proj = torch_npu.npu_grouped_matmul(
            [hidden_states], [layer.w13_weight],
            bias=None,
            scale=[layer.w13_weight_scale],
            per_token_scale=[pertoken_scale] if pertoken_scale is not None else None,
            group_list=expert_tokens,
            split_item=3, output_dtype=torch.bfloat16, group_type=0,
            x_dtype=torch_npu.hifloat8, weight_dtype=torch_npu.hifloat8,
            group_list_type=1,
        )[0]

        shared_expert_results = None
        if run_shared_with_cv:
            torch.npu.current_stream().wait_stream(self.shared_experts_stream)

            self.shared_experts_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(self.shared_experts_stream):
                shared_expert_results = layer.shared_experts.down_proj({'x_hif8': shared_expert_act})

        intermediate_hidden_states = torch_npu.npu_swiglu(gate_up_proj)
        intermediate_hidden_states = torch_npu.npu_dtype_cast(intermediate_hidden_states, torch_npu.hifloat8)
        pertoken_scale = None

        if run_shared_with_cv:
            torch.npu.current_stream().wait_stream(self.shared_experts_stream)
        if use_grouped_matmul_finalize_routing:
            return intermediate_hidden_states, pertoken_scale

        layer_key = getattr(layer, "layer_name", "") or ""
        intermediate_hidden_states = torch.ops.vllm.cube_side_run(
            layer_key, intermediate_hidden_states,
        )

        hidden_states_experts = torch_npu.npu_grouped_matmul(
            [intermediate_hidden_states], [layer.w2_weight],
            scale=[layer.w2_weight_scale],
            per_token_scale=[pertoken_scale] if pertoken_scale is not None else None,
            bias=None,
            group_list=expert_tokens,
            split_item=3, output_dtype=torch.bfloat16, group_type=0,
            x_dtype=torch_npu.hifloat8, weight_dtype=torch_npu.hifloat8,
            group_list_type=1, tuning_config=avg_tokens_per_expert,
        )[0]

        hidden_states_experts = torch.ops.vllm.cube_side_wait(
            layer_key, hidden_states_experts,
        )

        if shared_expert_results is not None:
            return hidden_states_experts, shared_expert_results
        return hidden_states_experts
