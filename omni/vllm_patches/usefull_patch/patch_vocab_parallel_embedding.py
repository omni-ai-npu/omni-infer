# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.utils import set_weight_attrs

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.v1.layers.vocab_parallel_embedding import NPUParallelLMHead

_orig_process_weights_after_loading = (
    UnquantizedEmbeddingMethod.process_weights_after_loading
)


def _patched_process_weights_after_loading(
    self, layer: torch.nn.Module
) -> None:
    _orig_process_weights_after_loading(self, layer)
    # Dummy may skip weight_loader; NZ when not fp32; skip if already marked.
    is_lm_head = isinstance(layer, (ParallelLMHead, NPUParallelLMHead))
    can_apply_nz = not model_extra_config.operator_opt_config.lmhead_fp32
    nz_not_applied = not getattr(layer.weight, "is_weight_nz", False)
    if is_lm_head and can_apply_nz and nz_not_applied:
        layer.weight.data = torch_npu.npu_format_cast(
            layer.weight.data, torch_npu.Format.FRACTAL_NZ
        )
        if not hasattr(layer.weight, "is_weight_nz"):
            set_weight_attrs(layer.weight, {"is_weight_nz": True})


@register_patch("NPUUnquantizedEmbeddingMethod", UnquantizedEmbeddingMethod)
class NPUUnquantizedEmbeddingMethodPatch(VLLMPatch):
    _attr_names_to_apply = ["process_weights_after_loading"]
    process_weights_after_loading = _patched_process_weights_after_loading
