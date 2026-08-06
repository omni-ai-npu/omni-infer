# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This patch is used to hijack the Qwen3Next architecture and apply vLLM core patches.

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
)
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.models.qwen.qwen_hybrid_common import (
    KVCacheUtilsPatch,
    SchedulerPatch,
    align_attention_block_size,
    align_hybrid_kv_cache_groups_to_attention,
    native_get_kv_cache_config_from_groups,
    register_local_qwen3_next_for_hybrid_patch,
    reshape_native_hybrid_kv_cache_tensors,
    set_hybrid_kv_cache_config_fn,
)
from omni_npu.worker.npu_model_runner import NPUModelRunner


def _get_hybrid_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    kv_cache_groups = align_attention_block_size(kv_cache_groups)
    kv_cache_groups = align_hybrid_kv_cache_groups_to_attention(kv_cache_groups)
    return native_get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    )


set_hybrid_kv_cache_config_fn(_get_hybrid_kv_cache_config)
register_local_qwen3_next_for_hybrid_patch(
    success_log="[Omni-NPU] HARD HIJACK SUCCESSFUL: Qwen3NextForCausalLM is now using the local workspace.",
    failure_log="[Omni-NPU] CRITICAL FAIL during Hard Hijack",
)


@register_patch("Qwen3NextReshapeKVCachePatch", NPUModelRunner)
class NPUModelRunnerPatch(VLLMPatch):
    _attr_names_to_apply = ['_reshape_kv_cache_tensors']

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        return reshape_native_hybrid_kv_cache_tensors(self, kv_cache_raw_tensors)
