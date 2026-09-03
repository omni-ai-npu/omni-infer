# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.cache import MambaDType
from vllm.config.model import ModelDType
from vllm.distributed import divide
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.utils.torch_utils import get_kv_cache_torch_dtype

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("MambaStateDtypeCalculatorMomePatch", MambaStateDtypeCalculator)
class MambaStateDtypeCalculatorMomePatch(VLLMPatch):
    """Add mome_state_dtype method to MambaStateDtypeCalculator."""

    _attr_names_to_apply = ["mome_state_dtype"]

    @classmethod
    def mome_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, state_dtype, state_dtype)


@register_patch("MambaStateShapeCalculatorMomePatch", MambaStateShapeCalculator)
class MambaStateShapeCalculatorMomePatch(VLLMPatch):
    """Add mome_state_shape method to MambaStateShapeCalculator."""

    _attr_names_to_apply = ["mome_state_shape"]

    @classmethod
    def mome_state_shape(
        cls,
        q_lora_rank: int,
        kv_lora_rank: int,
        num_heads: int,
        v_head_dim: int,
        kernel_size: int = 1,
        num_spec: int = 0,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        num_tokens = kernel_size - 1 + num_spec
        return (
            (num_tokens, q_lora_rank),
            (num_tokens, kv_lora_rank),
            (num_tokens, num_heads * v_head_dim),
        )
