# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

from omni.vllm_patches.patches.models.qwen import patch_qwen3_next
from omni.vllm_patches.patches.models.qwen3_5 import patch_qwen3_5


def test_qwen35_patch_reexports_shared_qwen_hybrid_patches():
    assert patch_qwen3_5.KVCacheUtilsPatch is patch_qwen3_next.KVCacheUtilsPatch
    assert patch_qwen3_5.SchedulerPatch is patch_qwen3_next.SchedulerPatch
    assert patch_qwen3_5.NPUModelRunnerPatch is patch_qwen3_next.NPUModelRunnerPatch
    assert (
        patch_qwen3_5._get_hybrid_kv_cache_config
        is patch_qwen3_next._get_hybrid_kv_cache_config
    )
