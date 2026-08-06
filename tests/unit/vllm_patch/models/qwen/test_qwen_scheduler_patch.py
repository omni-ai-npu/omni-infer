# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.kv_cache_interface import AttentionSpec

from omni.vllm_patches.patches.models.qwen import patch_qwen3_next as patch_next
from omni.vllm_patches.patches.models.qwen import qwen_hybrid_common as hybrid
from omni.vllm_patches.patches.models.qwen3_5 import patch_qwen3_5 as patch35


def test_hybrid_patch_entry_modules_import():
    assert patch_next is not None and patch35 is not None


def test_patch_qwen3_5_exports_shared_hybrid_config_fn():
    assert callable(patch35._get_hybrid_kv_cache_config)


def test_patch_qwen3_5_module_exports_runner_patch():
    assert patch35.NPUModelRunnerPatch is not None
    assert hybrid.VLLMPatch is not None


def test_hybrid_scheduler_and_kv_patches_registered():
    from omni.vllm_patches.patch_manager import PatchManager

    assert "SchedulerPatch" in PatchManager.registered_patches
    assert "KVCacheUtilsPatch" in PatchManager.registered_patches
    assert "QwenHybridInputBatchPatch" in PatchManager.registered_patches
    assert (
        PatchManager.registered_patches["SchedulerPatch"]
        is hybrid.QwenHybridSchedulerPatch
    )
    assert (
        PatchManager.registered_patches["KVCacheUtilsPatch"]
        is hybrid.QwenHybridKVCacheUtilsPatch
    )
    assert (
        PatchManager.registered_patches["QwenHybridInputBatchPatch"]
        is hybrid.QwenHybridInputBatchPatch
    )


def test_scheduler_resolve_block_ids_single_group():
    scheduler = hybrid.SchedulerPatch.__new__(hybrid.SchedulerPatch)
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.return_value = [[10, 11]]
    request = MagicMock()

    assert scheduler._resolve_block_ids(request) == [10, 11]


def test_scheduler_resolve_block_ids_picks_attention_group():
    scheduler = hybrid.SchedulerPatch.__new__(hybrid.SchedulerPatch)
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.return_value = [
        [1, 2],
        [3, 4],
    ]
    attn_group = SimpleNamespace(kv_cache_spec=MagicMock(spec=AttentionSpec))
    mamba_group = SimpleNamespace(kv_cache_spec=object())
    scheduler.kv_cache_manager.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[mamba_group, attn_group]
    )
    request = MagicMock()

    assert scheduler._resolve_block_ids(request) == [3, 4]


def test_scheduler_resolve_block_ids_fallback_without_attention_spec():
    scheduler = hybrid.SchedulerPatch.__new__(hybrid.SchedulerPatch)
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.return_value = [[9, 10], [11, 12]]
    mamba_only = SimpleNamespace(kv_cache_spec=object())
    scheduler.kv_cache_manager.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[mamba_only, mamba_only]
    )
    request = MagicMock()

    assert scheduler._resolve_block_ids(request) == [9, 10]


def test_scheduler_cache_remote_kv_success_reserves_last_token():
    """Remote KV for a full prompt caches num_tokens - 1, not a full block count.

    With block_size=16 and two blocks (32 token slots), min(blocks, num_tokens)
    is 32. Because that equals num_tokens, one token stays for local compute
    (31), matching legacy Qwen hybrid / prefilled-token scheduler behavior.
    This is not block-alignment to 32; it is intentionally num_tokens - 1.
    """
    scheduler = hybrid.SchedulerPatch.__new__(hybrid.SchedulerPatch)
    scheduler.block_size = 16
    scheduler.kv_cache_manager = MagicMock()
    scheduler._resolve_block_ids = MagicMock(return_value=[0, 1])
    request = SimpleNamespace(request_id="r1", num_tokens=32)

    scheduler._cache_remote_kv_success(request)

    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 31)
    assert request.num_computed_tokens == 31


def test_scheduler_cache_remote_kv_success_partial_prompt_no_reserve():
    scheduler = hybrid.SchedulerPatch.__new__(hybrid.SchedulerPatch)
    scheduler.block_size = 16
    scheduler.kv_cache_manager = MagicMock()
    scheduler._resolve_block_ids = MagicMock(return_value=[0, 1])
    request = SimpleNamespace(request_id="r1", num_tokens=33)

    scheduler._cache_remote_kv_success(request)

    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 32)
    assert request.num_computed_tokens == 32


def test_scheduler_cache_remote_kv_success_fewer_blocks_than_prompt():
    scheduler = hybrid.SchedulerPatch.__new__(hybrid.SchedulerPatch)
    scheduler.block_size = 16
    scheduler.kv_cache_manager = MagicMock()
    scheduler._resolve_block_ids = MagicMock(return_value=[0])
    request = SimpleNamespace(request_id="r1", num_tokens=32)

    scheduler._cache_remote_kv_success(request)

    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 16)
    assert request.num_computed_tokens == 16
