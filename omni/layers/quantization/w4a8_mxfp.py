# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from functools import partial
from typing import Optional

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from omni_npu.layers.fused_moe.config import FusedMoEQuantConfig, FusedMoEQuantDesc, _quant_flags_to_group_shape
from omni_npu.layers.mhc import cube_side_task_ops  # noqa: F401
from omni_npu.layers.quantization.mxfp8 import (
    _is_layer_ignored,
    apply_mxfp_experts,
)
from omni_npu.layers.quantization.mxfp8 import Mxfp8Config, Mxfp8MoEMethod
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.layers.fused_mlp.layer import FusedMLPMethodBase
from omni_npu.v1.layers.linear import FlashCommLinearMethodBase


W4A8_MXFP = "w4a8_mxfp"
MXFP8 = "mxfp8"

# OCP MXFP8: one E8M0 scale per 32-element block along the last (input) dim.
MX_BLOCK_SIZE = 32

# npu_quant_matmul/grouped_matmul group scales into pairs of 32 => "block_size_64 with 2 sub-blocks".
_MX_SCALE_PAIR = 2
_W4A8_MXFP_PACKED_ATTR = "is_w4a8_mxfp_packed"

_FLOAT4_E2M1FN_X2_DTYPE = getattr(
    torch_npu, "float4_e2m1fn_x2", None
)

# Resolve torch_npu dtype symbols with graceful fallback if the runtime is
# missing them (older torch_npu releases).
_FLOAT8_E8M0FNU_DTYPE = getattr(torch_npu, "float8_e8m0fnu", None)


def _require_w4a8_mxfp_runtime() -> None:
    missing = []
    if _FLOAT4_E2M1FN_X2_DTYPE is None:
        missing.append("torch_npu.float4_e2m1fn_x2")
    if _FLOAT8_E8M0FNU_DTYPE is None:
        missing.append("torch_npu.float8_e8m0fnu")
    for op_name in (
        "npu_dynamic_mx_quant",
        "npu_format_cast",
        "npu_quant_matmul",
    ):
        if not hasattr(torch_npu, op_name):
            missing.append(f"torch_npu.{op_name}")
    if missing:
        raise RuntimeError(
            "W4A8 MXFP is unavailable because the current torch_npu runtime "
            f"is missing: {', '.join(missing)}"
        )


def _validate_group_size(input_size: int, group_size: int) -> None:
    if group_size <= 0:
        raise ValueError(
            f"group_size must be positive, but got {group_size}"
        )
    scale_pair_size = group_size * _MX_SCALE_PAIR
    if input_size % scale_pair_size != 0:
        raise ValueError(
            f"input_size ({input_size}) must be divisible by "
            f"2 * group_size ({scale_pair_size}) so E8M0 scales can be "
            "packed in pairs"
        )


def _pack_w4a8_mxfp_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint W4A8 MXFP tensors to the NPU kernel layout.

    Checkpoint:
        weight: (N, K / 2)       uint8, packed MXFP4 E2M1
        scale:  (N, K / G)       uint8, E8M0

    Kernel:
        weight: (K / 2, N)       format 29
        scale:  (K / (2G), N, 2) uint8, E8M0
    """
    output_size = weight.shape[0]
    num_blocks = scales.shape[-1]
    _validate_group_size(num_blocks * group_size, group_size)
    if weight.shape[-1] * 2 != num_blocks * group_size:
        raise ValueError(
            "Packed weight and weight_scale shapes are inconsistent: "
            f"weight={tuple(weight.shape)}, scale={tuple(scales.shape)}, "
            f"group_size={group_size}"
        )

    weight = torch_npu.npu_format_cast(
        weight,
        29,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=_FLOAT4_E2M1FN_X2_DTYPE,
    ).transpose(-1, -2)
    scales = (
        scales.reshape(
            output_size,
            num_blocks // _MX_SCALE_PAIR,
            _MX_SCALE_PAIR,
        )
        .transpose(0, 1)
    )
    return weight, scales


def _pack_w4a8_mxfp_expert_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert expert checkpoint tensors to the grouped-matmul layout."""
    num_experts, output_size, packed_input_size = weight.shape
    num_blocks = scales.shape[-1]
    _validate_group_size(num_blocks * group_size, group_size)
    if packed_input_size * 2 != num_blocks * group_size:
        raise ValueError(
            "Packed expert weight and weight_scale shapes are inconsistent: "
            f"weight={tuple(weight.shape)}, scale={tuple(scales.shape)}, "
            f"group_size={group_size}"
        )

    weight = torch_npu.npu_format_cast(
        weight,
        29,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=_FLOAT4_E2M1FN_X2_DTYPE,
    ).transpose(1, 2)
    scales = (
        scales.reshape(
            num_experts,
            output_size,
            num_blocks // _MX_SCALE_PAIR,
            _MX_SCALE_PAIR,
        )
        .transpose(1, 2)
    )
    return weight, scales


def w4a8_mxfp_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FusedMoEQuantConfig:
    """Describe MXFP8 activations and MXFP4 weights to the MoE runtime."""
    activation_dtype = "mxfp8"
    weight_dtype = "mxfp4"
    activation_shape, weight_shape = _quant_flags_to_group_shape(activation_dtype, False, False, None)
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(activation_dtype, activation_shape, None),
        _a2=FusedMoEQuantDesc(activation_dtype, activation_shape, None),
        _w1=FusedMoEQuantDesc(weight_dtype, weight_shape, w1_scale),
        _w2=FusedMoEQuantDesc(weight_dtype, weight_shape, w2_scale),
    )


@register_quantization_config(W4A8_MXFP)
class W4A8MXFPConfig(QuantizationConfig):
    """MXFP8 activations with packed MXFP4 weights and E8M0 scales.

    ``non_moe_quant_method="mxfp8"`` enables mixed checkpoints whose
    attention, dense/shared MLP weights are MXFP8 and routed-expert weights
    are W4A8 MXFP. The default keeps the original all-W4A8 behavior.
    """

    def __init__(
        self,
        group_size: int = MX_BLOCK_SIZE,
        ignored_layers: Optional[list[str]] = None,
        non_moe_quant_method: str = W4A8_MXFP,
    ):
        super().__init__()
        if group_size <= 0:
            raise ValueError(
                f"group_size must be positive, but got {group_size}"
            )
        if non_moe_quant_method not in (W4A8_MXFP, MXFP8):
            raise ValueError(
                "non_moe_quant_method must be either "
                f'"{W4A8_MXFP}" or "{MXFP8}", but got '
                f'"{non_moe_quant_method}"'
            )
        self.group_size = group_size
        self.ignored_layers = ignored_layers
        self.non_moe_quant_method = non_moe_quant_method
        self._mxfp8_config = Mxfp8Config(
            ignored_layers=ignored_layers,
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            group_size=config.get("group_size", MX_BLOCK_SIZE),
            ignored_layers=config.get("ignore", None) or None,
            non_moe_quant_method=config.get(
                "non_moe_quant_method",
                W4A8_MXFP,
            ),
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return W4A8_MXFP

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            'NPU hardware does not support "get_min_capability" feature.'
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method_custom(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from omni_npu.v1.layers.fused_mlp.layer import FusedMLP
        from omni_npu.v1.layers.linear import (
            FlashCommLinearBase,
            UnquantizedFlashCommLinearMethod,
        )

        ignored = (
            self.ignored_layers
            and _is_layer_ignored(prefix, self.ignored_layers)
        )
        # Mixed checkpoints can store attention, dense MLP, and shared-expert
        # weights as MXFP8 while keeping only routed experts in W4A8 MXFP.
        # RoutedExperts must be selected before delegating because Mxfp8Config
        # also has its own routed-expert implementation.
        if isinstance(layer, RoutedExperts):
            return W4A8MXFPMoEMethod(self, layer)
        if self.non_moe_quant_method == MXFP8:
            return self._mxfp8_config.get_quant_method_custom(layer, prefix)
        if isinstance(layer, FlashCommLinearBase):
            if ignored:
                return UnquantizedFlashCommLinearMethod()
            return W4A8MXFPFCLinearMethod(self)
        if isinstance(layer, LinearBase):
            if ignored:
                return UnquantizedLinearMethod()
            return W4A8MXFPLinearMethod(self)
        if isinstance(layer, FusedMLP):
            return W4A8MXFPMlpMethod(self)
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        plugins = os.environ.get("VLLM_PLUGINS", "")
        if "omni_custom_models" in plugins:
            return self.get_quant_method_custom(layer, prefix)
        raise NotImplementedError(
            "W4A8 MXFP is currently implemented for Omni custom models only. "
            "Set VLLM_PLUGINS to include \"omni_custom_models\"."
        )


def _create_w4a8_mxfp_linear_weights(
    layer: torch.nn.Module,
    output_size_per_partition: int,
    input_size_per_partition: int,
    weight_loader,
    group_size: int,
) -> None:
    _validate_group_size(input_size_per_partition, group_size)
    num_blocks = input_size_per_partition // group_size

    weight = ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition // 2,
            dtype=torch.uint8,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight", weight)

    weight_scale = ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            num_blocks,
            dtype=torch.uint8,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_scale", weight_scale)


def _get_quantized_input(
    x: torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    if isinstance(x, tuple):
        x_mxfp8, pertoken_scale = x
        return x_mxfp8, pertoken_scale, torch.bfloat16
    if isinstance(x, dict):
        return (
            x["x_mxfp8"],
            x["pertoken_scale"],
            torch.bfloat16,
        )
    x_mxfp8, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
        x,
        dst_type=torch.float8_e4m3fn,
    )
    return x_mxfp8, pertoken_scale, x.dtype


def _apply_w4a8_mxfp_linear(
    layer: torch.nn.Module,
    x_mxfp8: torch.Tensor,
    pertoken_scale: torch.Tensor,
    output_dtype: torch.dtype,
    group_size: int,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch_npu.npu_quant_matmul(
        x_mxfp8,
        layer.weight,
        layer.weight_scale,
        scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
        bias=bias,
        output_dtype=output_dtype,
        x2_dtype=_FLOAT4_E2M1FN_X2_DTYPE,
        group_sizes=[0, 0, group_size],
    )


def _apply_w4a8_mxfp_grouped_matmul(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    expert_tokens: torch.Tensor,
    group_list_type: int,
) -> torch.Tensor:
    return torch_npu.npu_grouped_matmul(
        [activation],
        [weight],
        bias=None,
        scale=None,
        antiquant_scale=[weight_scale],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=_FLOAT4_E2M1FN_X2_DTYPE,
        output_dtype=torch.bfloat16,
        per_token_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
    )[0]


def _run_w4a8_mxfp_experts(
    method,
    layer,
    prepare_permute_result,
    use_grouped_matmul_finalize_routing,
):
    quantize = partial(
        torch_npu.npu_dynamic_mx_quant,
        dst_type=torch.float8_e4m3fn,
    )
    return apply_mxfp_experts(
        method,
        layer,
        prepare_permute_result,
        quantize,
        _apply_w4a8_mxfp_grouped_matmul,
        torch_npu.npu_swiglu_mx_quant,
        use_grouped_matmul_finalize_routing,
        reshape_swiglu_scale=True,
    )


class _W4A8MXFPLinearMethodMixin:

    def __init__(self, quant_config: W4A8MXFPConfig):
        _require_w4a8_mxfp_runtime()
        self.quant_config = quant_config
        self.group_size = quant_config.group_size

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        _create_w4a8_mxfp_linear_weights(
            layer,
            output_size_per_partition,
            input_size_per_partition,
            extra_weight_attrs.get("weight_loader"),
            self.group_size,
        )

    def process_weights_after_loading(self, layer):
        if getattr(layer.weight, _W4A8_MXFP_PACKED_ATTR, False):
            return
        weight, scale = _pack_w4a8_mxfp_weight(
            layer.weight.data,
            layer.weight_scale.data,
            self.group_size,
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        set_weight_attrs(layer.weight, {_W4A8_MXFP_PACKED_ATTR: True})

    def _apply_on_cube_side(
        self, layer, x_mxfp8, pertoken_scale, output_dtype, bias
    ):
        layer_key = getattr(layer, "prefix", "") or ""
        x_mxfp8 = torch.ops.vllm.cube_side_run(layer_key, x_mxfp8)
        output = _apply_w4a8_mxfp_linear(
            layer,
            x_mxfp8,
            pertoken_scale,
            output_dtype,
            self.group_size,
            bias,
        )
        return torch.ops.vllm.cube_side_wait(layer_key, output)


class W4A8MXFPLinearMethod(_W4A8MXFPLinearMethodMixin, LinearMethodBase):
    """W4A8 MXFP quantization method for vLLM LinearBase layers on NPU."""

    def apply(self, layer, x, bias=None):
        x_mxfp8, pertoken_scale, output_dtype = _get_quantized_input(x)
        return self._apply_on_cube_side(
            layer, x_mxfp8, pertoken_scale, output_dtype, bias
        )


class W4A8MXFPFCLinearMethod(_W4A8MXFPLinearMethodMixin, FlashCommLinearMethodBase):
    """W4A8 MXFP quantization for FlashComm LinearBase layers."""

    def apply(
        self,
        layer,
        x,
        bias=None,
        x_transform=None,
        x_dim=0,
        throw_dequant=False,
    ):
        from omni_npu.v1.distributed.communication_op_ext import (
            layer_parallel_all_gather,
            layer_parallel_all2all_single,
        )

        x_mxfp8, pertoken_scale, output_dtype = _get_quantized_input(x)

        if x_transform == "AllGather":
            pertoken_scale = layer_parallel_all_gather(pertoken_scale, layer.layer_name_inside_block, "x", x_dim)
            x_mxfp8 = layer_parallel_all_gather(x_mxfp8, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == "ALL2ALL":
            pertoken_scale = layer_parallel_all2all_single(pertoken_scale, layer.layer_name_inside_block, "x", x_dim)
            x_mxfp8 = layer_parallel_all2all_single(x_mxfp8, layer.layer_name_inside_block, "x", x_dim)

        return self._apply_on_cube_side(
            layer, x_mxfp8, pertoken_scale, output_dtype, bias
        )


class W4A8MXFPMlpMethod(FusedMLPMethodBase):
    """W4A8 MXFP gate/up, SwiGLU requantization, and down projection."""

    def __init__(self, quant_config: W4A8MXFPConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer):
        # gate_up_proj / down_proj are Mxfp8FCLinear submodules; their own
        # process_weights_after_loading has already repacked the weights.
        pass

    def apply_quant(
        self,
        x: torch.Tensor,
        stream_label=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            return torch_npu.npu_dynamic_mx_quant(
                x, dst_type=torch.float8_e4m3fn, scale_alg=1
            )

    def apply_part1_gate_up_on_stream(
        self,
        layer,
        x,
        stream_label=None,
    ) -> torch.Tensor:
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            gate_up, _ = layer.gate_up_proj(x, throw_dequant=False)
        return gate_up

    def apply_part2_activation_on_stream(
        self,
        layer,
        gate_up: torch.Tensor,
        stream_label=None,
    ) -> dict[str, torch.Tensor]:
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            x_mxfp8, pertoken_scale = torch_npu.npu_swiglu_mx_quant(
                gate_up,
                activate_left=True,
                dst_type=torch.float8_e4m3fn,
                scale_alg=1,
            )
            x = {
                "x_mxfp8": x_mxfp8,
                "pertoken_scale": pertoken_scale,
            }
        return x

    def apply_part3_down_on_stream(
        self,
        layer,
        x,
        stream_label=None,
    ) -> torch.Tensor:
        from omni_npu.v1.layers.utils import get_npu_execution_type

        with get_npu_execution_type(stream_label):
            output, _ = layer.down_proj(x)
        return output

    def apply(
        self,
        layer,
        x: torch.Tensor | dict[str, torch.Tensor],
        stream_label=None,
    ) -> torch.Tensor:
        if not isinstance(x, dict):
            x_mxfp8, pertoken_scale = self.apply_quant(x, stream_label=stream_label)
            x = {"x_mxfp8": x_mxfp8, "pertoken_scale": pertoken_scale}
        gate_up = self.apply_part1_gate_up_on_stream(layer, x, stream_label)
        x = self.apply_part2_activation_on_stream(layer, gate_up, stream_label)
        output = self.apply_part3_down_on_stream(layer, x, stream_label)
        return output


class W4A8MXFPMoEMethod(Mxfp8MoEMethod):
    """W4A8 MXFP grouped-expert implementation."""

    def __init__(self, quant_config: W4A8MXFPConfig, layer):
        _require_w4a8_mxfp_runtime()
        super().__init__(quant_config, layer)
        self.group_size = quant_config.group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        _validate_group_size(hidden_size, self.group_size)
        _validate_group_size(
            intermediate_size_per_partition,
            self.group_size,
        )
        num_experts += self.num_of_redundant_experts
        gate_up_size = 2 * intermediate_size_per_partition
        hidden_blocks = hidden_size // self.group_size
        intermediate_blocks = intermediate_size_per_partition // self.group_size

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                gate_up_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                gate_up_size,
                hidden_blocks,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_blocks,
                dtype=torch.uint8,
            ),
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
            raise NotImplementedError(
                "W4A8 MXFP MoE does not support expert bias"
            )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return w4a8_mxfp_moe_quant_config(
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )

    def process_weights_after_loading(
        self,
        layer: torch.nn.Module,
    ) -> None:
        if getattr(layer.w13_weight, _W4A8_MXFP_PACKED_ATTR, False):
            return
        w13_weight, w13_scale = _pack_w4a8_mxfp_expert_weight(
            layer.w13_weight.data,
            layer.w13_weight_scale.data,
            self.group_size,
        )
        w2_weight, w2_scale = _pack_w4a8_mxfp_expert_weight(
            layer.w2_weight.data,
            layer.w2_weight_scale.data,
            self.group_size,
        )
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
        packed_attrs = {_W4A8_MXFP_PACKED_ATTR: True}
        set_weight_attrs(layer.w13_weight, packed_attrs)
        set_weight_attrs(layer.w2_weight, packed_attrs)
        layer.ensure_moe_quant_config_init()

    def apply_experts(
        self,
        layer: torch.nn.Module,
        prepare_permute_result,
        activation: str = "silu",
        use_grouped_matmul_finalize_routing: bool = False,
    ) -> torch.Tensor:
        return _run_w4a8_mxfp_experts(
            self,
            layer,
            prepare_permute_result,
            use_grouped_matmul_finalize_routing,
        )
