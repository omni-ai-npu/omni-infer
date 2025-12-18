# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Custom normalization layers."""
import torch
import torch_npu
from typing import Optional, Union, Any
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNormGPU
from vllm.model_executor.custom_op import CustomOp
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank

from omni.models.config_loader.loader import model_extra_config


@CustomOp.register("omni_gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    
    @staticmethod
    def forward_static(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """npu implementation of gemma rms norm"""
        orig_dtype = x.dtype
        if residual is not None:
            if orig_dtype == torch.float16:
                x = x + residual.float()
            else:
                x = x + residual
            residual = x
        x = torch_npu.npu_gemma_rms_norm(x, weight, variance_epsilon)[0]
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)
    
    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_static(
            self.weight,
            self.variance_epsilon,
            x,
            residual,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)
        if not getattr(self, "_is_compiling", False):
            model_cfg = model_extra_config()
            if model_cfg.get("use_gemma_rmsnorm", False):
                self.forward_static = torch.compile(self.forward_static)
                self._is_compiling = True
        return self.forward_native(x, residual)

class RMSNorm(RMSNormGPU):
    def forward(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            quant_symbol: bool = False,
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if quant_symbol:
                x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
                x = {"x_int8": x_int8, "pertoken_scale": pertoken_scale}
            return x, residual

        return torch_npu.npu_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
        )[0]

class RMSNormFlashComm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        module_name: Optional[str] = "",
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size)
        self.module_name = module_name
        self.tp_size = get_tensor_model_parallel_world_size() # get tp size for each module
        self.tp_rank = get_tensor_model_parallel_rank() # get tp rank for each module

    def forward(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            y_transform: str = "",
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if y_transform == "AG":
                x = get_tp_group().all_gather(x, dim=0)
            return x, residual
        else:
            return torch_npu.npu_rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )[0]

    def forward_with_residual(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            y_transform: str = "",
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if y_transform == "AG":
                x = get_tp_group().all_gather(x, dim=0)
            return x, residual
        else:
            residual = x
            x = torch_npu.npu_rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )[0]
            return x, residual