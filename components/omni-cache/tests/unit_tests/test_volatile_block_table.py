# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for `apply_volatile_block_table` (Task #12).

These are pure tensor transforms; no NPU runtime needed.
"""

import pytest
import torch

from omni_cache.attention.metadata.volatile_block_table import (
    apply_volatile_block_table,
)


def _volatile_table(max_reqs: int, max_blocks: int) -> torch.Tensor:
    """Same construction as `PrefillOmniCache.initialize_device_cache`:
    fake ids start at 1 and run sequentially."""
    n = max_reqs * max_blocks
    return torch.arange(1, 1 + n, dtype=torch.int32).view(max_reqs, max_blocks)


class TestApplyVolatileBlockTable:
    def test_returns_matching_num_reqs(self):
        # 2 reqs, each 2 blocks of size 4 = 8 tokens per req
        vt = _volatile_table(max_reqs=4, max_blocks=2)
        real_bt = torch.tensor([[100, 101], [200, 201], [0, 0], [0, 0]], dtype=torch.int32)
        qsl = torch.tensor([0, 6, 10], dtype=torch.int64)  # 6 + 4 tokens
        slot_mapping = torch.zeros(10, dtype=torch.int64)

        fake_tbl, fake_slots, real_out = apply_volatile_block_table(
            block_tables=real_bt,
            slot_mapping=slot_mapping,
            query_start_loc=qsl,
            volatile_table=vt,
            block_size=4,
        )

        # fake block table is sliced to num_reqs=2
        assert fake_tbl.shape == (2, 2)
        # equals the first two rows of the volatile table
        assert torch.equal(fake_tbl, vt[:2])
        # real table round-trips (clone — not same identity)
        assert torch.equal(real_out, real_bt)
        assert real_out.data_ptr() != real_bt.data_ptr()

    def test_slot_mapping_rebuilds_from_fake_table(self):
        # 1 request, 3 tokens, block_size=2 → blocks [fake0, fake1]
        vt = _volatile_table(max_reqs=2, max_blocks=4)
        real_bt = torch.tensor([[99, 99, 0, 0], [0, 0, 0, 0]], dtype=torch.int32)
        qsl = torch.tensor([0, 3], dtype=torch.int64)
        slot_mapping = torch.tensor([0, 0, 0], dtype=torch.int64)

        fake_tbl, fake_slots, _ = apply_volatile_block_table(
            block_tables=real_bt,
            slot_mapping=slot_mapping,
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            volatile_table=vt,
            block_size=2,
        )

        # fake_slot[0] = vt[0, 0] * 2 + 0 = 1*2 + 0 = 2
        # fake_slot[1] = vt[0, 0] * 2 + 1 = 1*2 + 1 = 3
        # fake_slot[2] = vt[0, 1] * 2 + 0 = 2*2 + 0 = 4
        assert fake_slots.tolist() == [2, 3, 4]

    def test_multiple_reqs_land_in_disjoint_ranges(self):
        # Two requests, ensuring req-1's slots come from its own fake row.
        vt = _volatile_table(max_reqs=4, max_blocks=2)
        real_bt = torch.zeros((4, 2), dtype=torch.int32)
        qsl = torch.tensor([0, 2, 5], dtype=torch.int64)  # req0: 2 toks, req1: 3 toks
        slot_mapping = torch.zeros(5, dtype=torch.int64)

        _, fake_slots, _ = apply_volatile_block_table(
            block_tables=real_bt,
            slot_mapping=slot_mapping,
            query_start_loc=qsl,
            volatile_table=vt,
            block_size=2,
        )

        # req0 uses vt[0] = [1, 2]; tokens 0..1 → slot 1*2+0, 1*2+1 = 2, 3
        # req1 uses vt[1] = [3, 4]; tokens 2..4 → 3*2+0, 3*2+1, 4*2+0 = 6,7,8
        assert fake_slots.tolist() == [2, 3, 6, 7, 8]

    def test_empty_batch_noop(self):
        vt = _volatile_table(4, 2)
        empty_real = torch.zeros((0, 2), dtype=torch.int32)
        empty_slots = torch.zeros((0,), dtype=torch.int64)
        empty_qsl = torch.tensor([0], dtype=torch.int64)

        fake_tbl, fake_slots, real_tbl = apply_volatile_block_table(
            block_tables=empty_real,
            slot_mapping=empty_slots,
            query_start_loc=empty_qsl,
            volatile_table=vt,
            block_size=4,
        )
        assert fake_tbl.numel() == 0
        assert fake_slots.numel() == 0

    def test_overflow_raises(self):
        # volatile_table only has room for 2 reqs; 3-req batch must raise.
        vt = _volatile_table(2, 4)
        real_bt = torch.zeros((3, 4), dtype=torch.int32)
        qsl = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        slot_mapping = torch.zeros(3, dtype=torch.int64)

        import pytest
        with pytest.raises(ValueError):
            apply_volatile_block_table(
                block_tables=real_bt,
                slot_mapping=slot_mapping,
                query_start_loc=qsl,
                volatile_table=vt,
                block_size=4,
            )

    def test_slot_mapping_preserves_dtype(self):
        """Downstream attention kernels may rely on int64 vs int32; don't
        silently upcast or downcast."""
        vt = torch.arange(1, 5, dtype=torch.int32).reshape(2, 2)
        real_bt = torch.tensor([[0, 0], [0, 0]], dtype=torch.int32)
        qsl = torch.tensor([0, 2], dtype=torch.int64)
        for dtype in (torch.int32, torch.int64):
            slot_mapping = torch.zeros(2, dtype=dtype)
            _, fake_slots, _ = apply_volatile_block_table(
                block_tables=real_bt,
                slot_mapping=slot_mapping,
                query_start_loc=qsl,
                volatile_table=vt,
                block_size=4,
            )
            assert fake_slots.dtype == dtype


