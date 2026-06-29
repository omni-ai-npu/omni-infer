# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import List, Optional, Tuple, Dict, Union

import torch
import torch_npu
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.distributed import GroupCoordinator
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm.model_executor.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from omni_npu.v1.distributed.communication_op_ext import layer_parallel_all2all_single, layer_parallel_all_gather
from omni_npu.v1.layers.fused_mlp.layer import FusedMLPMethodBase
from omni_npu.v1.layers.linear import (
    FlashCommLinearMethodBase,
    ShardedLinearMethodBase,
)
from omni_npu.v1.layers.utils import get_npu_execution_type
from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)


class W8A8Int8FCLinearMethod(FlashCommLinearMethodBase):
    """FlashComm Linear method for NPU W8A8.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: CompressedTensorsConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_dtype = torch.int8

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        logger.debug("NpuW8A8LinearMethod params_dtype=%s", params_dtype)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(sum(output_partition_sizes), dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        weight_offset = ChannelQuantScaleParameter(
            data=torch.empty(sum(output_partition_sizes), dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_offset", weight_offset)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_data = torch_npu.npu_format_cast(
            layer.weight.data.t().contiguous(),
            torch_npu.Format.FRACTAL_NZ,
        )
        layer.weight = torch.nn.Parameter(weight_data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.view(-1), requires_grad=False
        )

        # NOTE(Daylight): mlaprolog need q_b_proj weight scale in float dtype
        if model_extra_config.operator_opt_config.enable_mlaprolog and "q_b_proj" in layer.prefix:
            layer.weight_scale.data = layer.weight_scale.data.float()

        if hasattr(layer, 'weight_offset'):
            layer.weight_offset = torch.nn.Parameter(
                layer.weight_offset.data.view(-1).float(), requires_grad=False
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        x_transform: Optional[str] = None,
        x_dim: Optional[int] = 0,
        throw_dequant: Optional[bool] = False,
    ) -> torch.Tensor:
        if isinstance(x, Dict):
            x_scale = x.get('pertoken_scale', None)
            x = x.get('x_int8', None)
        else:
            x, x_scale = torch_npu.npu_dynamic_quant(x)
        # TODO: scale_parallel is not supported yet.
        if x_transform == "AllGather":
            x_scale = layer_parallel_all_gather(
                x_scale, layer.layer_name_inside_block, "x", x_dim
            )
            x = layer_parallel_all_gather(x, layer.layer_name_inside_block, "x", x_dim)
        elif x_transform == "ALL2ALL":
            x_scale = layer_parallel_all2all_single(
                x_scale, layer.layer_name_inside_block, "x", x_dim
            )
            x = layer_parallel_all2all_single(
                x, layer.layer_name_inside_block, "x", x_dim
            )
        if throw_dequant and bias is None:
            y = torch_npu.npu_quant_matmul(
                x1=x,
                x2=layer.weight,
                scale=layer.weight_scale,
                bias=None,
                output_dtype=torch.int32,
            )
            return y, x_scale
        else:
            y = torch_npu.npu_quant_matmul(
                x1=x,
                x2=layer.weight,
                scale=layer.weight_scale,
                pertoken_scale=x_scale,
                bias=bias,
                output_dtype=layer.orig_dtype,
            )
            return y


class W8A8Int8MlpMethod(FusedMLPMethodBase):
    """Apply dequant_swiglu_quant fused kernel.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.gate_up_proj.weight_scale = torch.nn.Parameter(
            layer.gate_up_proj.weight_scale.data.float(), requires_grad=False
        )

    def apply_quant(
        self,
        x: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ):
        with get_npu_execution_type(stream_label):
            x, x_scale = torch_npu.npu_dynamic_quant(x)
        return x, x_scale

    def apply_part1_gate_up_on_stream(
        self,
        layer: torch.nn.Module,
        x: Dict[str, torch.Tensor],
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        with get_npu_execution_type(stream_label):
            gate_up, _ = layer.gate_up_proj(x, throw_dequant=True)

        return gate_up

    def apply_part2_activation_on_stream(
        self,
        layer: torch.nn.Module,
        gate_up: Tuple[torch.Tensor, torch.Tensor],
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with get_npu_execution_type(stream_label):
            x = {
                'x_int8': gate_up[0],
                'pertoken_scale': gate_up[1],
                'out_scale': layer.gate_up_proj.weight_scale
            }
            x = layer.act_fn(x, quant_symbol=True)

        return x

    def apply_part3_down_on_stream(
        self,
        layer: torch.nn.Module,
        x: Dict[str, torch.Tensor],
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        with get_npu_execution_type(stream_label):
            output, _ = layer.down_proj(x)

        return output

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        x, x_scale = self.apply_quant(x, stream_label)

        x = {'x_int8': x, 'pertoken_scale': x_scale}
        gate_up = self.apply_part1_gate_up_on_stream(
            layer, x, stream_label
        )

        x = self.apply_part2_activation_on_stream(
            layer, gate_up, stream_label
        )

        output = self.apply_part3_down_on_stream(
            layer, x, stream_label
        )
        return output


class W8A8Int8ShardedLinearMethod(ShardedLinearMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        in_size: int,
        out_size: int,
        params_dtype: torch.dtype,
        shard_group: GroupCoordinator,
        bias: bool = True,
        weight_nz: bool = True,
    ):
        self.shape = torch.Size([in_size, out_size])
        self.shard_group = (shard_group.rank_in_group,
                            shard_group.world_size,
                            shard_group.device_group)
        self.weight_nz = weight_nz
        self.full_weight: torch.Tensor = None

        base = ShardedLinearMethodBase
        base.register(layer, "weight", [0], torch.int8, self._mat_loader) # virtual
        base.register(layer, "weight_scale", [out_size], params_dtype, self._vec_loader)
        if bias:
            base.register(layer, "bias", [out_size], params_dtype, self._vec_loader)
        else:
            layer.bias = None

    def _vec_loader(self, weight: Parameter, loaded: torch.Tensor):
        weight.data.copy_(loaded.flatten())

    def _mat_loader(self, weight: Parameter, loaded: torch.Tensor):
        base = ShardedLinearMethodBase
        assert loaded.dtype == weight.data.dtype
        loaded = loaded.to(device=weight.data.device)
        loaded = loaded.transpose(0, 1).contiguous()
        assert loaded.shape == self.shape
        weight.gather_fn = base.shard_weight(
            loaded, self.shard_group, self.weight_nz)

    def prefetch(self, layer: torch.nn.Module):
        self.full_weight = layer.weight.gather_fn()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | dict,
        return_bias: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.full_weight is not None, f"error: apply before prefetch"
        if isinstance(x, dict):
            x, x_scale = x["x_int8"], x["pertoken_scale"]
        else:
            x, x_scale = torch_npu.npu_dynamic_quant(x)
        bias = None if return_bias else layer.bias
        y = torch_npu.npu_quant_matmul(
            x1=x,
            x2=self.full_weight,
            scale=layer.weight_scale,
            pertoken_scale=x_scale,
            bias=bias,
            output_dtype=torch.bfloat16,
        )
        del self.full_weight
        self.full_weight = None
        return (y, layer.bias) if return_bias else y
