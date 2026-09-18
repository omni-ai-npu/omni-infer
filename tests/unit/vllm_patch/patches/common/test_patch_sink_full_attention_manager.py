# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UT for SinkFullAttentionManagerPatch: upstream signature-lag fix (regression guard).

Verifies on stubs (no real NPU/block pool): the patched ``__init__`` forwards
``scheduler_block_size`` to the base, the sink-block reservation still runs,
and the base is called explicitly (no zero-arg ``super()``).
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def patch_mod(load_patch):
    return load_patch("patch_sink_full_attention_manager")


def _make_spec(block_size=16, sink_len=16):
    return SimpleNamespace(block_size=block_size, sink_len=sink_len)


def _make_block_pool(popleft_n_return=("sink_blk_0",)):
    fbq = SimpleNamespace(popleft_n=lambda n: list(popleft_n_return)[:n])
    return SimpleNamespace(null_block=None, free_block_queue=fbq)


def test_patched_init_accepts_scheduler_block_size(patch_mod):
    """The bug: upstream __init__ rejected scheduler_block_size; the patch forwards it."""
    from vllm.v1.core.single_type_kv_cache_manager import SinkFullAttentionManager

    cls = patch_mod.SinkFullAttentionManagerPatch
    saved = SinkFullAttentionManager.__init__
    owners = dict(getattr(SinkFullAttentionManager, "_omni_npu_applied_patches", {}))
    SinkFullAttentionManager._omni_npu_applied_patches = {}
    try:
        cls.apply()
        spec = _make_spec()
        # The coordinator factory calls manager_class(kv_cache_spec, **kwargs),
        # so every arg beyond the spec arrives as a keyword -- mirror that here.
        mgr = SinkFullAttentionManager(
            kv_cache_spec=spec,
            block_pool=_make_block_pool(),
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=16,
            dcp_world_size=1,
            pcp_world_size=1,
        )
        # scheduler_block_size reached the base class unchanged
        assert mgr.scheduler_block_size == 16
        # base class also set block_size from the spec
        assert mgr.block_size == 16
    finally:
        SinkFullAttentionManager.__init__ = saved
        SinkFullAttentionManager._omni_npu_applied_patches = owners


def test_patched_init_reserves_sink_blocks(patch_mod):
    """The sink-block reservation (popleft_n) still runs after the base init."""
    from vllm.v1.core.single_type_kv_cache_manager import SinkFullAttentionManager

    cls = patch_mod.SinkFullAttentionManagerPatch
    saved = SinkFullAttentionManager.__init__
    owners = dict(getattr(SinkFullAttentionManager, "_omni_npu_applied_patches", {}))
    SinkFullAttentionManager._omni_npu_applied_patches = {}
    try:
        cls.apply()
        spec = _make_spec(block_size=16, sink_len=32)  # 2 sink blocks
        pool = _make_block_pool(popleft_n_return=("blk_a", "blk_b"))
        mgr = SinkFullAttentionManager(
            kv_cache_spec=spec,
            block_pool=pool,
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=16,
        )
        assert mgr.sink_blocks == ["blk_a", "blk_b"]
    finally:
        SinkFullAttentionManager.__init__ = saved
        SinkFullAttentionManager._omni_npu_applied_patches = owners


def test_patched_init_does_not_use_zero_arg_super(patch_mod):
    """Regression: an injected plain function has no __class__ cell, so the
    patched __init__ must call the base explicitly (not no-arg super())."""
    import inspect
    src = inspect.getsource(patch_mod.SinkFullAttentionManagerPatch.__init__)
    # The patch must reference SingleTypeKVCacheManager explicitly, not rely on
    # bare super().
    assert "SingleTypeKVCacheManager.__init__" in src
    # Bare super() (with no args) would break for an injected function; assert
    # the explicit base call is the one used.
    assert "super().__init__" not in src
