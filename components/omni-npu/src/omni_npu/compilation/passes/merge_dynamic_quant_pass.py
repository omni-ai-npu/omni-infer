# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import functools

import torch
import torch_npu
import torchair
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.compilation.vllm_inductor_pass import VllmInductorPass

from omni_npu.logger import update_configure_vllm_root_logger


update_configure_vllm_root_logger()
logger = init_logger(__name__)


class MergeDynamicQuantPattern:
    """
    Pattern that merges two independent npu_dynamic_quant + npu_quant_matmul
    pairs sharing the same hidden_states input into a single npu_dynamic_quant
    followed by two npu_quant_matmul calls reusing the same quantized tensor.
    """
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    def get_inputs(self):
        hidden_states = functools.partial(
            torch.empty, (8, 4), dtype=torch.bfloat16,
            device="npu", requires_grad=False,
        )
        weight_0 = functools.partial(
            torch.empty, (4, 2), dtype=torch.int8,
            device="npu", requires_grad=False,
        )
        scale_0 = functools.partial(
            torch.empty, (2,), dtype=torch.float32,
            device="npu", requires_grad=False,
        )
        return [hidden_states(), weight_0(), scale_0(), weight_0(), scale_0()]

    def register(self):
        def pattern(
            hidden_states: torch.Tensor,
            weight_0: torch.Tensor, scale_0: torch.Tensor,
            weight_1: torch.Tensor, scale_1: torch.Tensor,
        ):
            x_quant_0, x_scale_0 = torch_npu.npu_dynamic_quant(
                hidden_states, smooth_scales=None)
            y0 = torch_npu.npu_quant_matmul(
                x1=x_quant_0,
                x2=weight_0,
                scale=scale_0,
                pertoken_scale=x_scale_0,
                output_dtype=torch.bfloat16,
            )
            x_quant_1, x_scale_1 = torch_npu.npu_dynamic_quant(
                hidden_states, smooth_scales=None)
            y1 = torch_npu.npu_quant_matmul(
                x1=x_quant_1,
                x2=weight_1,
                scale=scale_1,
                pertoken_scale=x_scale_1,
                output_dtype=torch.bfloat16,
            )
            return (y0, y1)

        def replacement(
            hidden_states: torch.Tensor,
            weight_0: torch.Tensor, scale_0: torch.Tensor,
            weight_1: torch.Tensor, scale_1: torch.Tensor,
        ):
            x_quant_0, x_scale_0 = torch_npu.npu_dynamic_quant(
                hidden_states, smooth_scales=None)
            y0 = torch_npu.npu_quant_matmul(
                x1=x_quant_0,
                x2=weight_0,
                scale=scale_0,
                pertoken_scale=x_scale_0,
                output_dtype=torch.bfloat16,
            )
            y1 = torch_npu.npu_quant_matmul(
                x1=x_quant_0,
                x2=weight_1,
                scale=scale_1,
                pertoken_scale=x_scale_0,
                output_dtype=torch.bfloat16,
            )
            return (y0, y1)

        torchair.register_replacement(pattern, replacement, self.get_inputs())


class MergeDynamicQuantPass(VllmInductorPass):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        dtype = vllm_config.model_config.dtype
        if dtype is not torch.bfloat16:
            return
        MergeDynamicQuantPattern(vllm_config).register()

    def __call__(self, graph: torch.fx.Graph):
        # The pattern is registered into torchair via torchair.register_replacement,
        # so torchair handles the actual graph rewriting; nothing to do here.
        pass
