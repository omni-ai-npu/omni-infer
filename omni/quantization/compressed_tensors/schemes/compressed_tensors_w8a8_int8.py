# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Callable, Any, Dict, List, Optional, Union

import torch
from torch.nn import Parameter
import torch_npu

from compressed_tensors.quantization import QuantizationStrategy
from vllm.distributed import get_tp_group
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import CompressedTensorsScheme
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from omni.models.config_loader.loader import model_extra_config


class AscendCompressedTensorsW8A8Int8LinearMethod(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(self, strategy: str, is_static_input_scheme: bool,
                 input_symmetric: bool, layer_name: str = ""):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric
        self.layer_name = layer_name

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(self, layer: torch.nn.Module, output_partition_sizes: List[int],
                    input_size_per_partition: int, params_dtype: torch.dtype, weight_loader: Callable,
                    **kwargs):
        self.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty( sum(output_partition_sizes),
            input_size_per_partition, dtype=torch.int8),
            input_dim=1, output_dim=0, weight_loader=weight_loader)

        layer.register_parameter("weight", weight)

        if self.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16),
                weight_loader=weight_loader)
            weight_offset = None
        else:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16),
                output_dim=0,
                weight_loader=weight_loader)
            weight_offset = ChannelQuantScaleParameter(
                data=torch.zeros((sum(output_partition_sizes), 1),
                                dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16),
                output_dim=0,
                weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, {"is_2_dims": True})
        layer.register_parameter("weight_offset", weight_offset)
        self.params_dtype = params_dtype
        layer.orig_dtype = self.params_dtype

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        weight_scale = layer.weight_scale
        if getattr(layer, 'throw_dequant', False):
            weight_scale = weight_scale.to(torch.float32)
        weight_offset = layer.weight_offset
        if (model_extra_config.operator_opt_config.shared_experts_to_gmm and
            model_extra_config.parall_config.redundancy_shared_expert_num > 0 and
            "shared_experts" in self.layer_name):
            weight = weight.unsqueeze(0).transpose(1, 2).contiguous()
        else:
            weight = weight.t().contiguous()
        weight = torch_npu.npu_format_cast(weight, 29)
        layer.weight = Parameter(weight, requires_grad=False)

        layer.weight_scale = Parameter(weight_scale.view(-1), requires_grad=False)
        layer.weight_offset = Parameter(weight_offset.view(-1).float(), requires_grad=False)
        return

    def apply_weights(self, layer: torch.nn.Module,
                    x: torch.Tensor,
                    bias: Optional[torch.Tensor],
                    module_name: Optional[str] = "",
                    x_transform: Optional[str] = None,
                    is_prefill: Optional[bool] = True
                    ) -> Union[torch.Tensor, Dict[str, Any]]:
        # activation per-token dynamic quant
        if isinstance(x, Dict):
            x_int8 = x.get('x_int8')
            pertoken_scale = x.get('pertoken_scale')
        else:
            x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        if x_transform == 'AG':
            pertoken_scale = get_tp_group().all_gather(pertoken_scale, dim=0)
            x_int8 = get_tp_group().all_gather(x_int8, dim=0)
        elif x_transform == 'A2A':
            pertoken_scale = get_tp_group().all_to_all(pertoken_scale, dim=0)
            x_int8 = get_tp_group().all_to_all(x_int8, dim=0)


        throw_dequant = getattr(layer, 'throw_dequant', False)
        if throw_dequant and bias is None:
            out = (torch_npu.npu_quant_matmul(x_int8, layer.weight, layer.weight_scale,
                                                bias=None, output_dtype=torch.int32),
                    pertoken_scale)
        else:
            out = torch_npu.npu_quant_matmul(x_int8, layer.weight, layer.weight_scale,
                                    offset=None,
                                    pertoken_scale=pertoken_scale,
                                    bias=bias,
                                    output_dtype=self.params_dtype)
        return out