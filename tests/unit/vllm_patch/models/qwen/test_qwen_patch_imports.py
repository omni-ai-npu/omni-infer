# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Import-time smoke tests for Qwen hybrid patch modules.

These catch NameError / missing hybrid.* references during pytest collection.
"""


def test_qwen_hybrid_common_imports():
    from omni_npu.vllm_patches.patches.models.qwen import qwen_hybrid_common as hybrid

    assert hybrid.QwenHybridSchedulerPatch is not None
    assert hybrid.QwenHybridKVCacheUtilsPatch is not None
    assert hybrid.QwenHybridInputBatchPatch is not None
    assert hybrid.set_hybrid_kv_cache_config_fn is not None


def test_patch_qwen3_next_imports():
    from omni_npu.vllm_patches.patches.models.qwen import patch_qwen3_next as patch_next

    assert patch_next.NPUModelRunnerPatch is not None
    assert patch_next.KVCacheUtilsPatch is not None
    assert callable(patch_next.KVCacheUtilsPatch.get_kv_cache_config_from_groups)


def test_patch_qwen3_5_imports():
    from omni_npu.vllm_patches.patches.models.qwen3_5 import patch_qwen3_5 as patch35

    assert patch35.NPUModelRunnerPatch is not None
    assert patch35.KVCacheUtilsPatch is not None
    assert callable(patch35._get_hybrid_kv_cache_config)
