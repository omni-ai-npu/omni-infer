# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Remove layer name check in bind_kv_cache function.
This allows NPU and other platforms to work with models that have
multiple attention layers per decoder block.
"""

from collections import defaultdict

import torch

from vllm.model_executor.layers.attention.attention import Attention
import vllm.v1.worker.gpu_model_runner as gpu_model_runner
from vllm.model_executor.models.utils import extract_layer_index

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


# Create a patched version of bind_kv_cache that removes the layer name check
def bind_kv_cache_patched(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0, "runner.kv_caches should be empty before initialization."

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]


# Register the patch
@register_patch("WorkerUtilsPatch", gpu_model_runner)
class WorkerUtilsPatch(VLLMPatch):
    """Patch to remove layer name check in bind_kv_cache"""

    _attr_names_to_apply = ["bind_kv_cache"]

    # Patch start - replace the function
    bind_kv_cache = bind_kv_cache_patched
    # patch end
