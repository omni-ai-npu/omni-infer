# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import re
from typing import Optional

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
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from omni_npu.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc, _quant_flags_to_group_shape
from omni_npu.layers.fused_moe.fused_moe_method_base import NPUFusedMoEMethodBase
from omni_npu.layers.fused_moe.layer import NPUFusedMoE
from omni_npu.layers.mhc import cube_side_task_ops  # noqa: F401  registers cube-side ops
from omni_npu.layers.utils import named_stream
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.layers.linear import FlashCommLinearMethodBase

logger = init_logger(__name__)

MXFP8 = "mxfp8"

# OCP MXFP8: one E8M0 scale per 32-element block along the last (input) dim.
MX_BLOCK_SIZE = 32

# npu_quant_matmul/grouped_matmul group scales into pairs of 32 => "block_size_64 with 2 sub-blocks".
_MX_SCALE_PAIR = 2

# Resolve torch_npu dtype symbols with graceful fallback if the runtime is
# missing them (older torch_npu releases).
_FLOAT8_E8M0FNU_DTYPE = getattr(torch_npu, "float8_e8m0fnu", None)


def _is_layer_ignored(prefix: str, ignored_layers: list[str]) -> bool:
    for pattern in ignored_layers:
        if pattern.startswith("re:"):
            if re.match(pattern[3:], prefix):
                return True
        elif pattern == prefix or pattern in prefix:
            return True
    return False


def _pack_mxfp8_weight(weight: torch.Tensor,
                       scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert on-disk mxfp8 layout into the layout that npu_quant_matmul wants.

    On-disk (from mxfp8_quant.py):
        weight: (O, I)       float8_e4m3fn
        scales: (O, I // 32) uint8 (E8M0 biased exponent)

    Kernel-ready (matmul):
        weight: (I, O)           float8_e4m3fn
        scale : (I // 64, O, 2)  e8m0 (uint8)
    """
    out_size = weight.shape[0]
    num_blocks = scales.shape[-1]
    assert num_blocks % _MX_SCALE_PAIR == 0, (
        f"num_blocks ({num_blocks}) must be even so scales fold into (K//64, N, 2)"
    )

    weight = weight.transpose(0, 1).contiguous()
    scale = scales.reshape(out_size, num_blocks // _MX_SCALE_PAIR,
                           _MX_SCALE_PAIR).transpose(0, 1).contiguous()
    return weight, scale


def _pack_mxfp8_expert_weight(weight: torch.Tensor,
                              scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MoE variant of :func:`_pack_mxfp8_weight` with a leading expert dim.

    On-disk:
        weight: (E, O, I)       float8_e4m3fn
        scales: (E, O, I // 32) uint8
    Kernel-ready (grouped matmul):
        weight: (E, I, O)           float8_e4m3fn
        scale : (E, I // 64, O, 2)  uint8
    """
    num_experts, out_size = weight.shape[0], weight.shape[1]
    num_blocks = scales.shape[-1]
    assert num_blocks % _MX_SCALE_PAIR == 0, (
        f"num_blocks ({num_blocks}) must be even so scales fold into (E, K//64, N, 2)"
    )

    weight = weight.transpose(1, 2).contiguous()
    scale = scales.reshape(num_experts, out_size, num_blocks // _MX_SCALE_PAIR,
                           _MX_SCALE_PAIR).transpose(1, 2).contiguous()
    return weight, scale


def mxfp8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
) -> FusedMoEQuantConfig:
    """FusedMoE quant-config descriptor for MXFP8 weights + activations."""
    quant_dtype = "mxfp8"
    a_shape, w_shape = _quant_flags_to_group_shape(quant_dtype, False, False, None)
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(quant_dtype, a_shape, a1_scale),
        _a2=FusedMoEQuantDesc(quant_dtype, a_shape, a2_scale),
        _w1=FusedMoEQuantDesc(quant_dtype, w_shape, w1_scale),
        _w2=FusedMoEQuantDesc(quant_dtype, w_shape, w2_scale),
    )


@register_quantization_config(MXFP8)
class Mxfp8Config(QuantizationConfig):
    """OCP MXFP8 config: FP8 E4M3 data + E8M0 block scales, block size 32."""

    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        ignored_layers = config.get("ignore", None) or None
        return cls(ignored_layers=ignored_layers)

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return MXFP8

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
            return Mxfp8FCLinearMethod(self)
        elif isinstance(layer, LinearBase):
            if self.ignored_layers and _is_layer_ignored(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Mxfp8LinearMethod(self)
        elif isinstance(layer, FusedMLP):
            return Mxfp8MlpMethod(self)
        elif isinstance(layer, FusedMoE):
            return Mxfp8MoEMethod(self, layer)
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        vllm_plugins = os.environ.get("VLLM_PLUGINS", "")
        custom_model_enabled = "omni_custom_models" in vllm_plugins
        if custom_model_enabled:
            return self.get_quant_method_custom(layer, prefix)

        raise NotImplementedError(
            "Mxfp8 quantization method is only implemented for custom models. "
            "Please set VLLM_PLUGINS environment variable to include "
            "\"omni_custom_models\" to enable it, or implement the method for "
            "non-custom models."
        )


def _create_mxfp8_linear_weights(layer, output_size_per_partition,
                                 input_size_per_partition, weight_loader):
    """Register ``weight`` and ``weight_scale`` for MXFP8 linear layers.

    Shapes follow the on-disk format so safetensor keys (``*.weight`` and
    ``*.weight_scale``) line up with the registered parameter names.

    weight:       (O, I)       float8_e4m3fn  — FP8 E4M3 data
    weight_scale: (O, I // 32) uint8          — E8M0 block exponents
    """
    assert input_size_per_partition % MX_BLOCK_SIZE == 0, (
        f"input_size_per_partition ({input_size_per_partition}) must be a "
        f"multiple of MXFP8 block size ({MX_BLOCK_SIZE})"
    )
    num_blocks = input_size_per_partition // MX_BLOCK_SIZE

    weight = ModelWeightParameter(
        data=torch.empty(output_size_per_partition, input_size_per_partition,
                         dtype=torch.float8_e4m3fn),
        input_dim=1, output_dim=0, weight_loader=weight_loader,
    )
    layer.register_parameter("weight", weight)

    weight_scale = ModelWeightParameter(
        data=torch.empty(output_size_per_partition, num_blocks, dtype=torch.uint8),
        input_dim=1, output_dim=0, weight_loader=weight_loader,
    )
    layer.register_parameter("weight_scale", weight_scale)


class Mxfp8LinearMethod(LinearMethodBase):
    """MXFP8 quantization method for vLLM LinearBase layers on NPU."""

    def __init__(self, quant_config: Mxfp8Config):
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        _create_mxfp8_linear_weights(layer, output_size_per_partition,
                                     input_size_per_partition, weight_loader)

    def process_weights_after_loading(self, layer):
        weight, scale = _pack_mxfp8_weight(layer.weight.data,
                                           layer.weight_scale.data)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

    def apply(self, layer, x, bias=None):
        layer_key = getattr(layer, "prefix", "") or ""
        if bias is not None:
            bias = bias.to(torch.float32)
        if isinstance(x, dict):
            x_fp8 = x.get('x_mxfp8')
            x_scale = x.get('pertoken_scale')
        else:
            x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

        # Cube-side overlap (opaque to Dynamo): if a task was registered
        # under layer_key, fire it on the cube-side stream now.
        x_fp8 = torch.ops.vllm.cube_side_run(layer_key, x_fp8)

        y = torch_npu.npu_quant_matmul(
            x_fp8, layer.weight, layer.weight_scale.view(torch.int8),
            pertoken_scale=x_scale.view(torch.int8),
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            group_sizes=[1, 1, 32],
            output_dtype=layer.orig_dtype, bias=bias,
        )

        y = torch.ops.vllm.cube_side_wait(layer_key, y)
        return y


class Mxfp8FCLinearMethod(FlashCommLinearMethodBase):
    """MXFP8 quantization for FlashComm LinearBase layers."""

    def __init__(self, quant_config: Mxfp8Config):
        self.quant_config = quant_config

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        _create_mxfp8_linear_weights(layer, output_size_per_partition,
                                     input_size_per_partition, weight_loader)

    def process_weights_after_loading(self, layer):
        weight, scale = _pack_mxfp8_weight(layer.weight.data,
                                           layer.weight_scale.data)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

    def apply(self, layer, x, bias=None, x_transform=None, x_dim=0, throw_dequant=False):
        from omni_npu.v1.distributed.communication_op_ext import (
            layer_parallel_all_gather,
            layer_parallel_all2all_single,
        )

        if bias is not None:
            bias = bias.to(torch.float32)
        if isinstance(x, dict):
            x_scale = x.get('pertoken_scale', None)
            x_fp8 = x.get('x_mxfp8', None)
        else:
            x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

        if x_transform == "AllGather":
            x_scale = layer_parallel_all_gather(x_scale, layer.layer_name_inside_block, "x", x_dim)
            x_fp8 = layer_parallel_all_gather(x_fp8, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == "ALL2ALL":
            x_scale = layer_parallel_all2all_single(x_scale, layer.layer_name_inside_block, "x", x_dim)
            x_fp8 = layer_parallel_all2all_single(x_fp8, layer.layer_name_inside_block, "x", x_dim)

        return torch_npu.npu_quant_matmul(
            x_fp8.view(torch.int8), layer.weight.view(torch.int8), layer.weight_scale.view(torch.int8),
            pertoken_scale=x_scale.view(torch.int8),
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            group_sizes=[1, 1, 32],
            x1_dtype=torch.float8_e4m3fn, x2_dtype=torch.float8_e4m3fn,
            output_dtype=layer.orig_dtype, bias=bias,
        )


class Mxfp8MlpMethod:
    """MXFP8 quantization glue for the fused-MLP stack (gate_up + silu + down)."""

    def __init__(self, quant_config: Mxfp8Config):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer):
        # gate_up_proj / down_proj are Mxfp8FCLinear submodules; their own
        # process_weights_after_loading has already repacked the weights.
        pass

    def apply_quant(self, x, x_transform=None, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        return x_fp8, x_scale

    def apply_part1_gate_up_on_stream(self, layer, x, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            gate_up, _ = layer.gate_up_proj(x, throw_dequant=False)
        return gate_up

    def apply_part2_activation_on_stream(self, layer, gate_up, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            x = {'x_mxfp8': gate_up, 'quant_type': 1}
            x = layer.act_fn(x, quant_symbol=True)
        return x

    def apply_part3_down_on_stream(self, layer, x, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            output, _ = layer.down_proj(x)
        return output

    def apply(self, layer, x, stream_label=None):
        x_fp8, x_scale = self.apply_quant(x, stream_label)
        x = {'x_mxfp8': x_fp8, 'pertoken_scale': x_scale}
        gate_up = self.apply_part1_gate_up_on_stream(layer, x, stream_label)
        x = self.apply_part2_activation_on_stream(layer, gate_up, stream_label)
        output = self.apply_part3_down_on_stream(layer, x, stream_label)
        return output


class Mxfp8MoEMethod(FusedMoEMethodBase, NPUFusedMoEMethodBase):

    def __init__(self, quant_config: Mxfp8Config, layer):
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
        assert hidden_size % MX_BLOCK_SIZE == 0, (
            f"hidden_size ({hidden_size}) must be a multiple of MXFP8 block size"
        )
        assert intermediate_size_per_partition % MX_BLOCK_SIZE == 0, (
            f"intermediate_size_per_partition ({intermediate_size_per_partition}) "
            f"must be a multiple of MXFP8 block size"
        )
        num_experts = num_experts + self.num_of_redundant_experts
        n13 = 2 * intermediate_size_per_partition
        h_blocks = hidden_size // MX_BLOCK_SIZE
        i_blocks = intermediate_size_per_partition // MX_BLOCK_SIZE

        # FP8 E4M3 weights — 2D per expert, standard FusedMoE shape
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, n13, hidden_size,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # E8M0 block scales (stored as uint8)
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, n13, h_blocks, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, i_blocks, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Activations are dynamically quantized each forward.
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, n13, dtype=torch.bfloat16),
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

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return mxfp8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_weight, w13_scale = _pack_mxfp8_expert_weight(
            layer.w13_weight.data, layer.w13_weight_scale.data,
        )
        w2_weight, w2_scale = _pack_mxfp8_expert_weight(
            layer.w2_weight.data, layer.w2_weight_scale.data,
        )
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
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

        if hidden_states.dim() > 2:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Upstream permute paths may hand us either an already-MXFP8-quantised
        # tensor + its E8M0 scale, or the raw bf16 permuted activations. In the
        # latter case we do the dynamic mxfp8 quant here.
        if pertoken_scale is None:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                hidden_states, dst_type=torch.float8_e4m3fn,
            )

        # npu_grouped_matmul wants per-token scale in (M, K//64, 2) layout;
        # npu_dynamic_mx_quant / dispatch produce it as (M, K//32).
        if pertoken_scale.dim() == 2:
            m, num_blocks = pertoken_scale.shape
            assert num_blocks % _MX_SCALE_PAIR == 0, (
                f"num_blocks ({num_blocks}) must be even to fold scales into pairs"
            )
            pertoken_scale = pertoken_scale.reshape(
                m, num_blocks // _MX_SCALE_PAIR, _MX_SCALE_PAIR,
            )

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
                shared_expert_act, shared_expert_pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                    shared_expert_act, dst_type=torch.float8_e4m3fn,
                )

        gate_up_proj = torch_npu.npu_grouped_matmul(
            [hidden_states], [layer.w13_weight],
            bias=None,
            scale=[layer.w13_weight_scale],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            split_item=3, output_dtype=torch.bfloat16, group_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=1,
        )[0]

        shared_expert_results = None
        if run_shared_with_cv:
            torch.npu.current_stream().wait_stream(self.shared_experts_stream)

            self.shared_experts_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(self.shared_experts_stream):
                shared_expert_results = layer.shared_experts.down_proj({
                    'x_mxfp8': shared_expert_act,
                    'pertoken_scale': shared_expert_pertoken_scale,
                })

        # SwiGLU + requantize to mxfp8 for the second GEMM.
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_swiglu_mx_quant(
            gate_up_proj,
            group_index=expert_tokens,
            activate_left=True,
            dst_type=torch.float8_e4m3fn
        )

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
            per_token_scale=[pertoken_scale],
            bias=None,
            group_list=expert_tokens,
            split_item=3, output_dtype=torch.bfloat16, group_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=1
        )[0]

        hidden_states_experts = torch.ops.vllm.cube_side_wait(
            layer_key, hidden_states_experts,
        )

        if shared_expert_results is not None:
            return hidden_states_experts, shared_expert_results
        return hidden_states_experts
