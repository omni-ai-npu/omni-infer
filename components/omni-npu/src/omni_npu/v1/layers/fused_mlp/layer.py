# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import abstractmethod
from typing import Optional, Union, Dict, Tuple

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.activation import SiluAndMul

from omni_npu.v1.layers.linear import MergedColumnParallelFlashCommLinear, RowParallelFlashCommLinear
from omni_npu.v1.layers.utils import get_npu_execution_type


class FusedMLPMethodBase(QuantizeMethodBase):
    """Base method for FusedMLP

    This method and its subclasses do not define create_weights method.
    The weights' creation is handled by submodules.
    This method only implement the apply method.
    """

    def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
        """Create weights for a layer.

        The weights will be set as attributes of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_part1_gate_up_on_stream(
        self,
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply_part2_activation_on_stream(
        self,
        layer: torch.nn.Module,
        gate_up: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply_part3_down_on_stream(
        self,
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class UnquantizedFusedMLPMethod(FusedMLPMethodBase):

    def apply_part1_gate_up_on_stream(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        with get_npu_execution_type(stream_label):
            gate_up, _ = layer.gate_up_proj(x)
        return gate_up

    def apply_part2_activation_on_stream(
        self,
        layer: torch.nn.Module,
        gate_up: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        with get_npu_execution_type(stream_label):
            x = layer.act_fn(gate_up)
        return x

    def apply_part3_down_on_stream(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        with get_npu_execution_type(stream_label):
            x, _ = layer.down_proj(x)
        return x

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        stream_label: Optional[str | torch.npu.Stream] = None,
    ) -> torch.Tensor:
        gate_up = self.apply_part1_gate_up_on_stream(layer, x, stream_label)
        x = self.apply_part2_activation_on_stream(layer, gate_up, stream_label)
        x = self.apply_part3_down_on_stream(layer, x, stream_label)
        return x


class FusedMLP(torch.nn.Module):
    """FusedMLP layer

    This layer relies on linear layer to create weights and
    implements optimizations that consider MLP module as a whole.
    For example, fusing the dequant of up_gate, swigu and the quant
    of down (dequant_swiglu_quant fused kernel).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        disable_tp: bool = False,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelFlashCommLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=disable_tp,
        )
        self.down_proj = RowParallelFlashCommLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=disable_tp,
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. " "Only silu is supported for now.")
        self.act_fn = SiluAndMul()
        quant_method: Optional[QuantizeMethodBase] = None

        if quant_config is None:
            quant_method = UnquantizedFusedMLPMethod()
        else:
            quant_method = quant_config.get_quant_method(self, prefix)

        assert quant_method is not None
        assert isinstance(quant_method, FusedMLPMethodBase)
        self.quant_method = quant_method

    def forward(self, x, stream_label: Optional[str | torch.npu.Stream] = None):
        output = self.quant_method.apply(self, x, stream_label=stream_label)
        return output
