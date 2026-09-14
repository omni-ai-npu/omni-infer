# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for hybrid KV-load failure recovery."""

import inspect
from types import SimpleNamespace

from omni_npu.vllm_patches.patches.common import (
    patch_hybrid_kv_load_failure as patch_mod,
)


class _FakeAttentionSpec:
    pass


class _FakeKVCacheManager:
    def __init__(self, block_id_groups):
        self.block_id_groups = block_id_groups

    def get_block_ids(self, request_id):
        return self.block_id_groups[request_id]


def _make_scheduler(block_id_groups):
    return SimpleNamespace(
        block_size=4,
        kv_cache_manager=_FakeKVCacheManager(block_id_groups),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=object()),
                SimpleNamespace(kv_cache_spec=_FakeAttentionSpec()),
            ]
        ),
    )


def test_patch_signature_matches_current_vllm_scheduler():
    patch_signature = inspect.signature(
        patch_mod.HybridKVLoadFailureSchedulerPatch
        ._update_requests_with_invalid_blocks
    )
    scheduler_signature = inspect.signature(
        patch_mod.Scheduler._update_requests_with_invalid_blocks
    )
    assert tuple(patch_signature.parameters) == tuple(
        scheduler_signature.parameters
    )


def test_hybrid_recovery_resets_all_groups(monkeypatch):
    monkeypatch.setattr(patch_mod, "AttentionSpec", _FakeAttentionSpec)
    scheduler = _make_scheduler(
        {"req-1": ([100, 101, 102], [200, 201, 202])}
    )
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=12)

    affected, affected_tokens, blocks_to_evict = (
        patch_mod.HybridKVLoadFailureSchedulerPatch
        ._update_requests_with_invalid_blocks(
            scheduler,
            [request],
            {201},
            {"req-1": 4},
            evict_blocks=True,
        )
    )

    assert affected == {"req-1"}
    assert affected_tokens == 8
    assert blocks_to_evict == {100, 101, 102, 200, 201, 202}
    assert request.num_computed_tokens == 0


def test_single_group_preserves_invalid_block_recovery():
    scheduler = SimpleNamespace(
        block_size=4,
        kv_cache_manager=_FakeKVCacheManager(
            {"req-1": ([100, 101, 102],)}
        ),
    )
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=12)

    affected, affected_tokens, blocks_to_evict = (
        patch_mod.HybridKVLoadFailureSchedulerPatch
        ._update_requests_with_invalid_blocks(
            scheduler, [request], {101}, {}, evict_blocks=True
        )
    )

    assert affected == {"req-1"}
    assert affected_tokens == 8
    assert blocks_to_evict == {101, 102}
    assert request.num_computed_tokens == 4
