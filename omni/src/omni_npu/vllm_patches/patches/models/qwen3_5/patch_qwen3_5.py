# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This patch is used to hijack the Qwen3Next architecture for Qwen3.5.

from omni_npu.vllm_patches.patches.models.qwen import patch_qwen3_next as _qwen_next

KVCacheUtilsPatch = _qwen_next.KVCacheUtilsPatch
SchedulerPatch = _qwen_next.SchedulerPatch
NPUModelRunnerPatch = _qwen_next.NPUModelRunnerPatch
_get_hybrid_kv_cache_config = _qwen_next._get_hybrid_kv_cache_config

__all__ = (
    "KVCacheUtilsPatch",
    "SchedulerPatch",
    "NPUModelRunnerPatch",
    "_get_hybrid_kv_cache_config",
)
