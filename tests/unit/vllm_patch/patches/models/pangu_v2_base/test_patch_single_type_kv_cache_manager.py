# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UT for extra-reserved / #52707 allocate_external_computed_blocks."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from omni_npu.vllm_patches.patches.models.pangu_v2_base.patch_single_type_kv_cache_manager import (
    SingleTypeKVCacheManagerExternalAllocPatch,
)

NULL = object()
BLOCK = 128


def _new_blocks(n):
    return [object() for _ in range(n)]


def _no_skipped_tokens(tokens):
    return 0


def _manager(**overrides):
    pool = MagicMock()
    pool.get_new_blocks.side_effect = _new_blocks
    mgr = SimpleNamespace(
        block_size=BLOCK,
        _null_block=NULL,
        req_to_blocks={"req": []},
        block_pool=pool,
        kv_cache_spec=SimpleNamespace(),
        new_block_ids=[],
        num_extra_reserved_blocks=0,
        get_num_skipped_tokens=_no_skipped_tokens,
    )
    for key, value in overrides.items():
        setattr(mgr, key, value)
    mgr.allocate_external_computed_blocks = (
        SingleTypeKVCacheManagerExternalAllocPatch.allocate_external_computed_blocks.__get__(
            mgr, type(mgr)
        )
    )
    return mgr


def test_extra_reserved_mome_skips_pages_that_have_no_offloaded_keys():
    mgr = _manager(num_extra_reserved_blocks=8, kernel_size=5)
    mgr.allocate_external_computed_blocks("req", 0, 8 * BLOCK)

    # Window is kernel_size-1=4 tokens → 7 skipped blocks + 1 real page.
    assert mgr.req_to_blocks["req"][:7] == [NULL] * 7
    assert len(mgr.req_to_blocks["req"]) == 8
    mgr.block_pool.get_new_blocks.assert_called_once_with(1)


def test_extra_reserved_swa_skips_pages_that_have_no_offloaded_keys():
    mgr = _manager(num_extra_reserved_blocks=2, sliding_window=128)
    mgr.allocate_external_computed_blocks("req", 0, 3 * BLOCK)

    # Window is 127 tokens → 2 skipped blocks, then 1 real page.
    assert mgr.req_to_blocks["req"][:2] == [NULL] * 2
    assert len(mgr.req_to_blocks["req"]) == 3
    mgr.block_pool.get_new_blocks.assert_called_once_with(1)


def test_no_external_tokens_pads_nulls_without_allocating():
    mgr = _manager(num_extra_reserved_blocks=8, kernel_size=5)
    mgr.allocate_external_computed_blocks("req", 8 * BLOCK, 0)

    assert mgr.req_to_blocks["req"] == [NULL] * 8
    mgr.block_pool.get_new_blocks.assert_not_called()


def test_negative_allocation_is_clamped_to_zero():
    existing = [object(), object(), object()]
    mgr = _manager(req_to_blocks={"req": list(existing)})
    mgr.allocate_external_computed_blocks("req", BLOCK, BLOCK)

    mgr.block_pool.get_new_blocks.assert_called_once_with(0)
    assert mgr.req_to_blocks["req"] == existing
