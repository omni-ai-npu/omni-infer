# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import re
from contextlib import nullcontext
from functools import partial
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
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from omni_npu.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc, _quant_flags_to_group_shape
from omni_npu.layers.fused_moe.fused_moe_method_base import NPUFusedMoEMethodBase
from omni_npu.layers.fused_moe.layer import NPUFusedMoE
from omni_npu.layers.fused_moe.shared_expert import activate_shared_expert_on_side_stream
from omni_npu.layers.mhc import cube_side_task_ops  # noqa: F401  registers cube-side ops
from omni_npu.layers.utils import named_stream
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.layers.linear import FlashCommLinearMethodBase
from omni_npu.v1.layers.fused_mlp.layer import FusedMLPMethodBase, UnquantizedFusedMLPMethod
from omni_npu.v1.utils import on_ascend950
from omni_npu.layers.fused_moe.fused_moe import fused_experts_tp
from omni_npu.layers.prefetch import PrefetchManager


logger = init_logger(__name__)

MXFP8 = "mxfp8"

# OCP MXFP8: one E8M0 scale per 32-element block along the last (input) dim.
MX_BLOCK_SIZE = 32

# npu_quant_matmul/grouped_matmul group scales into pairs of 32 => "block_size_64 with 2 sub-blocks".
_MX_SCALE_PAIR = 2

# Resolve torch_npu dtype symbols with graceful fallback if the runtime is
# missing them (older torch_npu releases).
_FLOAT8_E8M0FNU_DTYPE = getattr(torch_npu, "float8_e8m0fnu", None)


def _reshape_mxfp_scale_pairs(pertoken_scale: torch.Tensor) -> torch.Tensor:
    """Fold adjacent MXFP block scales into the pair layout used by NPU kernels."""
    num_tokens, num_blocks = pertoken_scale.shape
    if num_blocks % _MX_SCALE_PAIR != 0:
        raise ValueError(
            f"num_blocks ({num_blocks}) must be even to fold scales into pairs"
        )
    return pertoken_scale.reshape(
        num_tokens,
        num_blocks // _MX_SCALE_PAIR,
        _MX_SCALE_PAIR,
    )


def _prepare_mxfp_expert_inputs(prepare_permute_result, quantize):
    """Normalize routed expert inputs and their per-token MXFP scales."""
    hidden_states = prepare_permute_result.hidden_states_sorted_by_experts
    if hidden_states.dim() > 2:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    pertoken_scale = prepare_permute_result.dynamic_scale
    if pertoken_scale is None:
        hidden_states, pertoken_scale = quantize(hidden_states)
    if pertoken_scale.dim() == 2:
        pertoken_scale = _reshape_mxfp_scale_pairs(pertoken_scale)
    return hidden_states, prepare_permute_result.expert_tokens, pertoken_scale


def _activate_shared_expert_if_scheduled(
    stream,
    layer,
    prepare_permute_result,
    quantize,
):
    """Start the shared expert side stream when it overlaps routed experts."""
    operator_config = model_extra_config.operator_opt_config
    enabled = (
        operator_config.shared_expert_multi_stream
        and operator_config.shared_expert_parallel_schedule
        == "with_routed_experts_cv"
    )
    if not enabled:
        return False, None, None
    activation, scale = activate_shared_expert_on_side_stream(
        stream,
        layer,
        prepare_permute_result,
        quantize,
    )
    return True, activation, scale


def _apply_mxfp8_grouped_matmul(
    activation,
    weight,
    weight_scale,
    pertoken_scale,
    expert_tokens,
    group_list_type,
):
    return torch_npu.npu_grouped_matmul(
        [activation],
        [weight],
        bias=None,
        scale=[weight_scale],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        group_list_type=group_list_type,
    )[0]


def apply_mxfp_experts(
    method,
    layer,
    prepare_permute_result,
    quantize,
    grouped_matmul,
    swiglu_quant,
    use_grouped_matmul_finalize_routing,
    *,
    swiglu_scale_alg=None,
    reshape_swiglu_scale=False,
):
    """Run the shared MXFP expert pipeline with a format-specific GEMM."""
    moe_parallel_config = getattr(layer, "moe_parallel_config", None)
    group_list_type = int(getattr(moe_parallel_config, "use_ep", True))
    hidden_states, expert_tokens, pertoken_scale = _prepare_mxfp_expert_inputs(
        prepare_permute_result,
        quantize,
    )
    run_shared, shared_act, shared_scale = _activate_shared_expert_if_scheduled(
        method.shared_experts_stream,
        layer,
        prepare_permute_result,
        quantize,
    )

    gate_up_proj = grouped_matmul(
        hidden_states,
        layer.w13_weight,
        layer.w13_weight_scale,
        pertoken_scale,
        expert_tokens,
        group_list_type,
    )

    shared_results = None
    if run_shared:
        torch.npu.current_stream().wait_stream(method.shared_experts_stream)
        method.shared_experts_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(method.shared_experts_stream):
            shared_results = layer.shared_experts.down_proj(
                {"x_mxfp8": shared_act, "pertoken_scale": shared_scale}
            )

    swiglu_options = {}
    if swiglu_scale_alg is not None:
        swiglu_options["scale_alg"] = swiglu_scale_alg
    intermediate, pertoken_scale = swiglu_quant(
        gate_up_proj,
        group_index=None,
        activate_left=True,
        dst_type=torch.float8_e4m3fn,
        **swiglu_options,
    )
    if reshape_swiglu_scale and pertoken_scale.dim() == 2:
        pertoken_scale = _reshape_mxfp_scale_pairs(pertoken_scale)
    if run_shared:
        torch.npu.current_stream().wait_stream(method.shared_experts_stream)
    if use_grouped_matmul_finalize_routing:
        return intermediate, pertoken_scale

    layer_key = getattr(layer, "layer_name", "") or ""
    intermediate = torch.ops.vllm.cube_side_run(layer_key, intermediate)
    routed_results = grouped_matmul(
        intermediate,
        layer.w2_weight,
        layer.w2_weight_scale,
        pertoken_scale,
        expert_tokens,
        group_list_type,
    )
    routed_results = torch.ops.vllm.cube_side_wait(layer_key, routed_results)
    if shared_results is not None:
        return routed_results, shared_results
    return routed_results


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


def _unpack_mxfp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Restore a linear scale from kernel layout to checkpoint layout."""
    num_block_pairs, out_size, _ = scale.shape
    return scale.transpose(0, 1).reshape(
        out_size, num_block_pairs * _MX_SCALE_PAIR
    ).contiguous()


def _unpack_mxfp8_expert_scale(scale: torch.Tensor) -> torch.Tensor:
    """Restore an MoE scale from kernel layout to checkpoint layout."""
    num_experts, num_block_pairs, out_size, _ = scale.shape
    return scale.transpose(1, 2).reshape(
        num_experts, out_size, num_block_pairs * _MX_SCALE_PAIR,
    ).contiguous()


def _pack_mxfp8_scale(scale: torch.Tensor) -> torch.Tensor:
    out_size, num_blocks = scale.shape
    assert num_blocks % _MX_SCALE_PAIR == 0
    return scale.reshape(
        out_size, num_blocks // _MX_SCALE_PAIR, _MX_SCALE_PAIR
    ).transpose(0, 1).contiguous()


def _pack_mxfp8_expert_scale(scale: torch.Tensor) -> torch.Tensor:
    num_experts, out_size, num_blocks = scale.shape
    assert num_blocks % _MX_SCALE_PAIR == 0
    return scale.reshape(
        num_experts, out_size, num_blocks // _MX_SCALE_PAIR, _MX_SCALE_PAIR
    ).transpose(1, 2).contiguous()


def _mxfp8_linear_weight_loader(original_weight_loader,
                                 param: torch.nn.Parameter,
                                 loaded_weight: torch.Tensor,
                                 *args, **kwargs):
    """Reload a regular vLLM linear weight after MXFP8 kernel packing."""
    if not getattr(param, "is_mxfp8_packed", False):
        return original_weight_loader(param, loaded_weight, *args, **kwargs)

    packed_storage = param.data
    had_transposed_attr = hasattr(param, "is_weight_transposed")
    transposed_attr = getattr(param, "is_weight_transposed", None)
    param.data = packed_storage.transpose(0, 1)
    param.is_weight_transposed = False
    try:
        return original_weight_loader(param, loaded_weight, *args, **kwargs)
    finally:
        param.data = packed_storage
        if had_transposed_attr:
            param.is_weight_transposed = transposed_attr
        else:
            delattr(param, "is_weight_transposed")


def _mxfp8_scale_weight_loader(original_weight_loader, expert: bool,
                                param: torch.nn.Parameter,
                                loaded_weight: torch.Tensor, *args, **kwargs):
    """Reload a packed scale through its original checkpoint-layout loader."""
    if not getattr(param, "is_mxfp8_scale_packed", False):
        return original_weight_loader(param, loaded_weight, *args, **kwargs)

    packed_storage = param.data
    unpack = _unpack_mxfp8_expert_scale if expert else _unpack_mxfp8_scale
    pack = _pack_mxfp8_expert_scale if expert else _pack_mxfp8_scale
    param.data = unpack(packed_storage)
    try:
        result = original_weight_loader(param, loaded_weight, *args, **kwargs)
        repacked = pack(param.data)
        assert repacked.shape == packed_storage.shape
        packed_storage.copy_(repacked)
    finally:
        param.data = packed_storage
    return result


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
            "NPU hardware does not support \"get_min_capability\" feature.")

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
        elif isinstance(layer, RoutedExperts):
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
                                 input_size_per_partition, weight_loader,
                                 wrap_weight_loader=False):
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

    runtime_weight_loader = (
        partial(_mxfp8_linear_weight_loader, weight_loader)
        if wrap_weight_loader and weight_loader is not None else weight_loader
    )
    weight = ModelWeightParameter(
        data=torch.empty(output_size_per_partition, input_size_per_partition,
                         dtype=torch.float8_e4m3fn),
        input_dim=1, output_dim=0, weight_loader=runtime_weight_loader,
    )
    layer.register_parameter("weight", weight)

    scale_weight_loader = (
        partial(_mxfp8_scale_weight_loader, weight_loader, False)
        if weight_loader is not None else None
    )
    weight_scale = ModelWeightParameter(
        data=torch.empty(output_size_per_partition, num_blocks, dtype=torch.uint8),
        input_dim=1, output_dim=0, weight_loader=scale_weight_loader,
    )
    layer.register_parameter("weight_scale", weight_scale)


class _Mxfp8LinearMethodMixin:

    _wrap_weight_loader = False

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
                                     input_size_per_partition, weight_loader,
                                     wrap_weight_loader=self._wrap_weight_loader)

    def process_weights_after_loading(self, layer):
        if getattr(layer.weight, "is_mxfp8_packed", False):
            return
        weight, scale = _pack_mxfp8_weight(layer.weight.data,
                                           layer.weight_scale.data)
        layer.weight.data = weight
        layer.weight_scale.data = scale
        set_weight_attrs(layer.weight, {
            "is_weight_transposed": True,
            "is_mxfp8_packed": True,
        })
        set_weight_attrs(layer.weight_scale, {"is_mxfp8_scale_packed": True})


class Mxfp8LinearMethod(_Mxfp8LinearMethodMixin, LinearMethodBase):
    """MXFP8 quantization method for vLLM LinearBase layers on NPU."""

    _wrap_weight_loader = True

    def apply(self, layer, x, bias=None):
        layer_key = getattr(layer, "prefix", "") or ""
        if bias is not None:
            bias = bias.to(torch.float32)
        if isinstance(x, dict):
            x_fp8 = x.get('x_mxfp8')
            x_scale = x.get('pertoken_scale')
        else:
            x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=1)

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


class Mxfp8FCLinearMethod(_Mxfp8LinearMethodMixin, FlashCommLinearMethodBase):
    """MXFP8 quantization for FlashComm LinearBase layers."""

    input_quant_format = MXFP8

    def apply(self, layer, x, bias=None, x_transform=None, x_dim=0, throw_dequant=False):
        from omni_npu.v1.distributed.communication_op_ext import (
            layer_parallel_all_gather,
            layer_parallel_all2all_single,
        )

        layer_key = getattr(layer, "prefix", "") or ""
        if bias is not None:
            bias = bias.to(torch.float32)
        if isinstance(x, dict):
            x_scale = x.get('pertoken_scale', None)
            x_fp8 = x.get('x_mxfp8', None)
        else:
            x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=1)

        if x_transform == "AllGather":
            x_scale = layer_parallel_all_gather(x_scale, layer.layer_name_inside_block, "x", x_dim)
            x_fp8 = layer_parallel_all_gather(x_fp8, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == "ALL2ALL":
            x_scale = layer_parallel_all2all_single(x_scale, layer.layer_name_inside_block, "x", x_dim)
            x_fp8 = layer_parallel_all2all_single(x_fp8, layer.layer_name_inside_block, "x", x_dim)

        # FlashComm linears need the same Cube-side task hooks as regular
        # Mxfp8LinearMethod. In particular, attention o_proj is a
        # RowParallelFlashCommLinear, so without these hooks its registered
        # MHC Sinkhorn task is never overlapped with the quantized matmul and
        # falls back to synchronous execution in mhc_fetch.
        x_fp8 = torch.ops.vllm.cube_side_run(layer_key, x_fp8)

        y = torch_npu.npu_quant_matmul(
            x_fp8, layer.weight, layer.weight_scale,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            group_sizes=[1, 1, 32],
            output_dtype=layer.orig_dtype, bias=bias,
        )

        return torch.ops.vllm.cube_side_wait(layer_key, y)


class Mxfp8MlpMethod(FusedMLPMethodBase):
    """MXFP8 quantization glue for the fused-MLP stack (gate_up + silu + down)."""

    def __init__(self, quant_config: Mxfp8Config):
        self.quant_config = quant_config
        self.should_limit_core = (
            model_extra_config.task_config.graph_mode == "acl_graph"
            and on_ascend950()
        )

    def process_weights_after_loading(self, layer):
        # gate_up_proj / down_proj are Mxfp8FCLinear submodules; their own
        # process_weights_after_loading has already repacked the weights.
        pass

    def apply_quant(self, x, x_transform=None, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        core_limit_ctx = (
            torch.npu.npugraph_ex.scope.limit_core_num(0, 8)
            if self.should_limit_core
            else nullcontext()
        )

        with get_npu_execution_type(stream_label):
            with core_limit_ctx:
                x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=1)
        return x_fp8, x_scale

    def apply_part1_gate_up_on_stream(self, layer, x, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            gate_up, _ = layer.gate_up_proj(x, throw_dequant=False)
        return gate_up

    def apply_part2_activation_on_stream(self, layer, gate_up, stream_label=None):
        from omni_npu.v1.layers.utils import get_npu_execution_type

        core_limit_ctx = (
            torch.npu.npugraph_ex.scope.limit_core_num(0, 8)
            if self.should_limit_core
            else nullcontext()
        )

        with get_npu_execution_type(stream_label):
            with core_limit_ctx:
                # silu+quant
                x_fp8, x_scale = torch_npu.npu_swiglu_mx_quant(
                    gate_up,
                    activate_left=True,
                    dst_type=torch.float8_e4m3fn,
                    scale_alg=1
                )
            x = {"x_mxfp8": x_fp8, "pertoken_scale": x_scale}
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
        self.on_ascend950 = on_ascend950()
        self.model_prefetch = PrefetchManager()

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
        scale_weight_attrs = dict(extra_weight_attrs)
        scale_weight_loader = scale_weight_attrs.get("weight_loader")
        if scale_weight_loader is not None:
            scale_weight_attrs["weight_loader"] = partial(
                _mxfp8_scale_weight_loader, scale_weight_loader, True,
            )
        scale_weight_attrs["quant_method"] = FusedMoeWeightScaleSupported.CHANNEL.value
        set_weight_attrs(w13_weight_scale, scale_weight_attrs)
        set_weight_attrs(w2_weight_scale, scale_weight_attrs)

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
        if getattr(layer.w13_weight, "is_mxfp8_packed", False):
            return
        w13_weight, w13_scale = _pack_mxfp8_expert_weight(
            layer.w13_weight.data, layer.w13_weight_scale.data,
        )
        w2_weight, w2_scale = _pack_mxfp8_expert_weight(
            layer.w2_weight.data, layer.w2_weight_scale.data,
        )
        layer.w13_weight.data = w13_weight
        layer.w2_weight.data = w2_weight
        layer.w13_weight_scale.data = w13_scale
        layer.w2_weight_scale.data = w2_scale
        weight_attrs = {
            "is_weight_transposed": True,
            "is_mxfp8_packed": True,
        }
        set_weight_attrs(layer.w13_weight, weight_attrs)
        set_weight_attrs(layer.w2_weight, weight_attrs)
        scale_attrs = {"is_mxfp8_scale_packed": True}
        set_weight_attrs(layer.w13_weight_scale, scale_attrs)
        set_weight_attrs(layer.w2_weight_scale, scale_attrs)
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

        multi_stream = model_extra_config.operator_opt_config.shared_expert_multi_stream
        enable_prefetch = getattr(model_extra_config.operator_opt_config, "enable_prefetch", False)
        cur_stream = torch.npu.current_stream()
        if enable_prefetch:
            self.shared_experts_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.shared_experts_stream):
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

        moe_parallel_config = getattr(layer, "moe_parallel_config", None)
        use_ep = getattr(moe_parallel_config, "use_ep", True)
        if not use_ep:
            if self.on_ascend950:
                if multi_stream or enable_prefetch:
                    cur_stream.wait_stream(self.shared_experts_stream)
                routed_output = fused_experts_tp(
                    layer=layer,
                    x=x_slice,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                )
                if layer.shared_experts is not None:
                    shared_output = layer.shared_experts(x_slice)
                    return shared_output, routed_output + shared_output
                return routed_output
            return fused_experts_tp(
                layer=layer,
                x=x_slice,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
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

        if enable_prefetch:
            self.shared_experts_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.shared_experts_stream):
                self.model_prefetch.prefetch("next_attn", shared_output, layer=layer)

        routed_output = self.apply_unpermute_finalize(
            strategy_impl, 
            layer, 
            output, 
            topk_ids, 
            topk_weights, 
            prepare_permute_result,
        )

        if multi_stream or enable_prefetch:
            cur_stream.wait_stream(self.shared_experts_stream)

        use_custom_model_add = "omni_custom_models" in os.environ.get("VLLM_PLUGINS", "")
        if multi_stream and schedule == "with_finalize":
            if layer.shared_experts is not None:
                cur_stream.wait_stream(self.shared_experts_stream)
                if layer.shared_experts.gate_up_proj.tp_size > 1:
                    shared_output = tensor_model_parallel_all_reduce(shared_output)
                elif use_custom_model_add:
                    routed_output = routed_output + shared_output
        elif schedule == "with_finalize":
            if layer.shared_experts is not None:
                if layer.shared_experts.gate_up_proj.tp_size > 1:
                    shared_output = tensor_model_parallel_all_reduce(shared_output)
                elif use_custom_model_add:
                    routed_output = routed_output + shared_output

        if is_need_slice:
            routed_output = tensor_model_parallel_all_gather(routed_output, dim=0)[:orig_num_tokens]

        if shared_output is not None and layer.shared_experts.gate_up_proj.tp_size > 1 and use_custom_model_add: 
            return shared_output, routed_output + shared_output
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
        quantize = partial(
            torch_npu.npu_dynamic_mx_quant,
            dst_type=torch.float8_e4m3fn,
            scale_alg=1,
        )
        return apply_mxfp_experts(
            self,
            layer,
            prepare_permute_result,
            quantize,
            _apply_mxfp8_grouped_matmul,
            torch_npu.npu_swiglu_mx_quant,
            use_grouped_matmul_finalize_routing,
            swiglu_scale_alg=1,
        )
