# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import xgrammar
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_orig_apply_token_bitmask_inplace = xgrammar.apply_token_bitmask_inplace


@register_patch("NPUGrammarBitmaskBackendPatch", xgrammar)
class NPUGrammarBitmaskBackendPatch(VLLMPatch):
    """Force xgrammar's bitmask kernel onto the triton-free ``torch_native``
    backend on NPU to avoid ``import triton`` triggered by ``torch.compile``.
    """

    _attr_names_to_apply = ["apply_token_bitmask_inplace"]

    @staticmethod
    def apply_token_bitmask_inplace(logits: torch.Tensor, grammar_bitmask, *args, **kwargs) -> None:
        if logits.device.type not in ("cpu", "cuda"):
            kwargs["backend"] = "torch_native"
        else:
            kwargs["backend"] = "auto"
        
        return _orig_apply_token_bitmask_inplace(logits, grammar_bitmask, *args, **kwargs)
