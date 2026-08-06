# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec

from omni_npu.vllm_patches.patches.models.qwen import patch_qwen3_next as patch_next


def test_empty_groups_returns_minimal_config():
    cfg = patch_next.KVCacheUtilsPatch.get_kv_cache_config_from_groups(
        MagicMock(), [], available_memory=1024
    )

    assert cfg.num_blocks == 1
    assert cfg.kv_cache_tensors == []


@patch(
    "omni_npu.vllm_patches.patches.models.qwen.patch_qwen3_next."
    "native_get_kv_cache_config_from_groups"
)
def test_attention_config_uses_shared_hybrid_builder(native_builder):
    attention = AttentionSpec(
        block_size=129,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )
    group = KVCacheGroupSpec(["layer0"], attention)
    expected_config = MagicMock()
    native_builder.return_value = expected_config

    cfg = patch_next._get_hybrid_kv_cache_config(
        MagicMock(), [group], available_memory=4096
    )

    assert cfg is expected_config
    native_builder.assert_called_once()
    aligned_groups = native_builder.call_args.args[1]
    assert aligned_groups[0].kv_cache_spec.block_size == 256


def test_patch_module_reexports_shared_patch_classes():
    from omni_npu.vllm_patches.patches.models.qwen import qwen_hybrid_common

    assert patch_next.KVCacheUtilsPatch is qwen_hybrid_common.QwenHybridKVCacheUtilsPatch
    assert patch_next.SchedulerPatch is qwen_hybrid_common.QwenHybridSchedulerPatch
    assert patch_next.NPUModelRunnerPatch.__name__ == "NPUModelRunnerPatch"
