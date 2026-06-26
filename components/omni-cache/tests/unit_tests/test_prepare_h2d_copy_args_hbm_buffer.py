# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import sys
from unittest.mock import MagicMock, PropertyMock

import pytest
import torch
import multiprocessing

_hbm_mock = MagicMock()
_hbm_mock._is_sliding_window_spec = (
    lambda spec: "SlidingWindow" in type(spec).__name__
)
sys.modules["omni_cache.cache.decode.hbm_buffer_utils"] = _hbm_mock

from omni_cache.cache.transfer_engine.decode import (
    prepare_h2d_copy_args_hbm_buffer,
)

class MockTensor:
    """Real tensor with a mocked data_ptr() for address-level assertions."""

    @staticmethod
    def make(shape, ptr=0x1000):
        t = torch.zeros(shape, dtype=torch.bfloat16)
        t.data_ptr = MagicMock(return_value=ptr)
        return t

class MockOmniCache:
    def __init__(
        self,
        *,
        pool_size=8,
        lane_map=None,
        kv_cache_groups=(),
        hbm_pool=None,
        host_cache=None,
    ):
        self.num_max_batch_pool = pool_size
        base = {i: None for i in range(pool_size)}
        base[0] = "fake-request-idx"
        if lane_map is not None:
            base.update(lane_map)
            base[0] = lane_map.get(0, "fake-request-idx")
        self.record_batch_idx_to_req = base
        self._lane_lock = multiprocessing.Lock()
        self.host_cache = host_cache or MagicMock()
        self.kv_cache_config = MagicMock()
        self.kv_cache_config.kv_cache_groups = list(kv_cache_groups)
        self.hbm_buffer_block_table_pool = hbm_pool or []
        self.block_size = 128
        self.record_req_ids_swa_block = {}


class MockKVCacheGroup:
    def __init__(self, layer_names, spec_kind):
        spec = MagicMock()
        spec.__class__.__name__ = spec_kind
        spec.block_size = 128
        self.layer_names = layer_names
        self.kv_cache_spec = spec


class MockHBMPoolEntry(dict):
    """HBM block-table-pool entry dict with kind / req_offset / buffers."""

    def __init__(self, spec_kind, req_offset=1, layer_buffers=None):
        super().__init__(
            kind=spec_kind,
            req_offset=req_offset,
            hbm_buffer_pool_raw=layer_buffers or {},
        )


class TestLaneClaim:
    def test_claims_first_free_skipping_sentinel(self):
        c = MockOmniCache(lane_map={0: "fake", 1: None, 2: None})
        prepare_h2d_copy_args_hbm_buffer(c, [[], []], "A")
        assert c.record_batch_idx_to_req[1] == "A"

    def test_no_free_raises(self):
        c = MockOmniCache(pool_size=3, lane_map={0: "fake", 1: "B", 2: "C"})
        with pytest.raises(RuntimeError, match="No free HBM lane"):
            prepare_h2d_copy_args_hbm_buffer(c, [[], []], "A")

    def test_sentinel_never_claimed(self):
        c = MockOmniCache(lane_map={0: "fake", 1: None})
        prepare_h2d_copy_args_hbm_buffer(c, [[], []], "X")
        assert c.record_batch_idx_to_req[0] == "fake"
        assert c.record_batch_idx_to_req[1] == "X"

    def test_fills_in_order(self):
        c = MockOmniCache(pool_size=4, lane_map={0: "fake", 1: None, 2: None, 3: None})
        for req in ["A", "B", "C"]:
            prepare_h2d_copy_args_hbm_buffer(c, [[], []], req)
        assert [c.record_batch_idx_to_req[i] for i in range(1, 4)] == ["A", "B", "C"]

    def test_last_claim_raises_next(self):
        c = MockOmniCache(pool_size=2, lane_map={0: "fake", 1: None})
        prepare_h2d_copy_args_hbm_buffer(c, [[], []], "only")
        with pytest.raises(RuntimeError, match="No free HBM lane"):
            prepare_h2d_copy_args_hbm_buffer(c, [[], []], "next")


class TestMamba:
    def test_single_layer(self):
        hc = MagicMock()
        hc.get_block.return_value = [[MockTensor.make((1, 1024))]]
        npu = MockTensor.make((8, 1024))
        g = MockKVCacheGroup(["L0"], "MomeSpec")
        e = MockHBMPoolEntry("mamba", layer_buffers={"L0": npu})
        c = MockOmniCache(kv_cache_groups=[g], hbm_pool=[e], host_cache=hc)
        dev, _, host, sz = prepare_h2d_copy_args_hbm_buffer(c, [[], [[100]]], "A")
        assert len(dev) == 1 and len(host) == 1

    def test_uses_last_block(self):
        hc = MagicMock()
        hc.get_block.return_value = [[MockTensor.make((1, 1024))]]
        g = MockKVCacheGroup(["L0"], "MomeSpec")
        e = MockHBMPoolEntry("mamba", layer_buffers={"L0": MockTensor.make((8, 1024))})
        c = MockOmniCache(kv_cache_groups=[g], hbm_pool=[e], host_cache=hc)
        prepare_h2d_copy_args_hbm_buffer(c, [[], [[10, 20, 30]]], "A")
        hc.get_block.assert_called_with(30)

    def test_multi_layer(self):
        hc = MagicMock()
        hc.get_block.return_value = [[MockTensor.make((1, 1024)), MockTensor.make((1, 1024))]]
        npu = MockTensor.make((8, 1024))
        g = MockKVCacheGroup(["L0", "L1"], "MomeSpec")
        e = MockHBMPoolEntry("mamba", layer_buffers={"L0": npu, "L1": npu})
        c = MockOmniCache(kv_cache_groups=[g], hbm_pool=[e], host_cache=hc)
        dev, _, host, _ = prepare_h2d_copy_args_hbm_buffer(c, [[], [[50]]], "A")
        assert len(dev) == 2 and len(host) == 2

    def test_empty_blocks_raises(self):
        g = MockKVCacheGroup(["L0"], "MomeSpec")
        c = MockOmniCache(kv_cache_groups=[g],
                          hbm_pool=[MockHBMPoolEntry("mamba")])
        with pytest.raises(RuntimeError, match="No src blocks"):
            prepare_h2d_copy_args_hbm_buffer(c, [[], [[]]], "A")

    def test_slot_at_idx_req_plus_one(self):
        hc = MagicMock()
        hc.get_block.return_value = [[MockTensor.make((1, 1024))]]
        npu = MockTensor.make((8, 1024), ptr=0x9000)
        g = MockKVCacheGroup(["L0"], "MomeSpec")
        e = MockHBMPoolEntry("mamba", layer_buffers={"L0": npu})
        # pre-fill lane 1 so our request gets lane 2 (idx_req=2, slot=3)
        c = MockOmniCache(lane_map={0: "fake", 1: "other", 2: None, 3: None},
                          kv_cache_groups=[g], hbm_pool=[e], host_cache=hc,
                          pool_size=4)
        prepare_h2d_copy_args_hbm_buffer(c, [[], [[10]]], "A")
        assert c.record_batch_idx_to_req[2] == "A"


class TestAttention:
    def test_non_swa_uses_block_id_as_slot(self):
        hc = MagicMock()
        hc.batch_layer_copy_to_npu.return_value = ([0xA000], [4096], [0xB000], [2048])
        npu = MockTensor.make((32, 128, 576))
        g = MockKVCacheGroup(["L0"], "DSAAttentionSpec")
        e = MockHBMPoolEntry("attention", layer_buffers={"L0": npu})
        c = MockOmniCache(kv_cache_groups=[g], hbm_pool=[e], host_cache=hc,
                          pool_size=16)
        dev, _, host, _ = prepare_h2d_copy_args_hbm_buffer(c, [[], [[5, 7, 9]]], "A")
        assert dev == [0xA000] and host == [0xB000]

    def test_single_block(self):
        hc = MagicMock()
        hc.batch_layer_copy_to_npu.return_value = ([0x1000], [512], [0x2000], [256])
        npu = MockTensor.make((64, 128, 576))
        g = MockKVCacheGroup(["L0"], "FullAttentionSpec")
        c = MockOmniCache(
            kv_cache_groups=[g],
            hbm_pool=[MockHBMPoolEntry("attention", layer_buffers={"L0": npu})],
            host_cache=hc, pool_size=16)
        dev, _, host, _ = prepare_h2d_copy_args_hbm_buffer(c, [[], [[42]]], "A")
        assert dev == [0x1000]

    def test_empty_blocks_skips_copy(self):
        g = MockKVCacheGroup(["L0"], "DSAAttentionSpec")
        c = MockOmniCache(kv_cache_groups=[g],
                          hbm_pool=[MockHBMPoolEntry("attention")],
                          pool_size=16)
        r = prepare_h2d_copy_args_hbm_buffer(c, [[], [[]]], "A")
        assert r == ([], [], [], [])
        c.host_cache.batch_layer_copy_to_npu.assert_not_called()

    def test_tuple_buf_unpacks_data_scale(self):
        hc = MagicMock()
        hc.batch_layer_copy_to_npu.return_value = ([0xA0], [512], [0xB0], [256])
        data = MockTensor.make((32, 128, 576), ptr=0x1000)
        scale = MockTensor.make((32, 128, 128), ptr=0x2000)
        g = MockKVCacheGroup(["L0"], "DSAAttentionSpec")
        c = MockOmniCache(
            kv_cache_groups=[g],
            hbm_pool=[MockHBMPoolEntry("attention",
                                       layer_buffers={"L0": (data, scale)})],
            host_cache=hc, pool_size=16)
        dev, _, host, _ = prepare_h2d_copy_args_hbm_buffer(c, [[], [[1]]], "A")
        assert dev == [0xA0]


class TestSWA:
    def test_drops_block_zero(self):
        hc = MagicMock()
        hc.batch_layer_copy_to_npu.return_value = ([], [], [], [])
        spec = MagicMock()
        spec.__class__.__name__ = "SlidingWindowSpec"
        spec.block_size = 128
        type(spec).sliding_window = PropertyMock(return_value=1024)
        g = MagicMock()
        g.layer_names = ["L0"]
        g.kv_cache_spec = spec
        npu = MockTensor.make((32, 128, 576))
        c = MockOmniCache(
            kv_cache_groups=[g],
            hbm_pool=[MockHBMPoolEntry("attention", req_offset=8,
                                       layer_buffers={"L0": npu})],
            host_cache=hc, pool_size=16)
        prepare_h2d_copy_args_hbm_buffer(c, [[], [[0, 1, 2, 3, 4, 5, 6]]], "A")
        args, _ = hc.batch_layer_copy_to_npu.call_args
        assert 0 not in args[0], "block 0 must be filtered"

    def test_keeps_only_window(self):
        hc = MagicMock()
        hc.batch_layer_copy_to_npu.return_value = ([], [], [], [])
        spec = MagicMock()
        spec.__class__.__name__ = "SlidingWindowSpec"
        spec.block_size = 128
        type(spec).sliding_window = PropertyMock(return_value=128)
        g = MagicMock()
        g.layer_names = ["L0"]
        g.kv_cache_spec = spec
        npu = MockTensor.make((32, 128, 576))
        c = MockOmniCache(
            kv_cache_groups=[g],
            hbm_pool=[MockHBMPoolEntry("attention", req_offset=4,
                                       layer_buffers={"L0": npu})],
            host_cache=hc, pool_size=16)
        # window = (128+128-1)//128 + 1 = 2
        prepare_h2d_copy_args_hbm_buffer(
            c, [[], [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], "A")
        args, _ = hc.batch_layer_copy_to_npu.call_args
        assert len(args[0]) == 2


class TestMixed:
    def test_mamba_then_attention(self):
        hc = MagicMock()
        hc.get_block.return_value = [[MockTensor.make((1, 1024))]]
        hc.batch_layer_copy_to_npu.return_value = ([0x2000], [512], [0x3000], [256])
        mamba_npu = MockTensor.make((8, 1024))
        attn_npu = MockTensor.make((32, 128, 576))
        c = MockOmniCache(
            kv_cache_groups=[MockKVCacheGroup(["L0"], "MomeSpec"),
                             MockKVCacheGroup(["L1"], "DSAAttentionSpec")],
            hbm_pool=[MockHBMPoolEntry("mamba", layer_buffers={"L0": mamba_npu}),
                      MockHBMPoolEntry("attention", layer_buffers={"L1": attn_npu})],
            host_cache=hc, pool_size=16)
        dev, _, host, _ = prepare_h2d_copy_args_hbm_buffer(
            c, [[], [[10], [5, 7]]], "req-A")
        assert len(dev) >= 1 and len(host) >= 1


class TestEdgeCases:
    def test_empty_groups(self):
        c = MockOmniCache()
        r = prepare_h2d_copy_args_hbm_buffer(c, [[], []], "A")
        assert r == ([], [], [], [])

    def test_mamba_min_size_transfer(self):
        """Copy size is min(npu.nbytes, cpu.nbytes)."""
        cpu = MockTensor.make((1, 50))   # 50 elements × 2 bytes = 100 bytes
        npu = MockTensor.make((8, 100))  # 100 elements × 2 bytes = 200 bytes
        hc = MagicMock()
        hc.get_block.return_value = [[cpu]]
        g = MockKVCacheGroup(["L0"], "MomeSpec")
        e = MockHBMPoolEntry("mamba", layer_buffers={"L0": npu})
        c = MockOmniCache(kv_cache_groups=[g], hbm_pool=[e], host_cache=hc)
        _, _, _, sizes = prepare_h2d_copy_args_hbm_buffer(c, [[], [[1]]], "A")
        assert sizes[0] == 100  # min(200, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
