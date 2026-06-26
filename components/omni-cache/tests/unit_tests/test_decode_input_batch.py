# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid import (
    PanguNewKVCacheSpecsPatch,
)

try:
    PanguNewKVCacheSpecsPatch.apply()
except ValueError as exc:
    if "already patched" not in str(exc):
        raise

from vllm.v1.kv_cache_interface import (  # noqa: E402
    DSAAttentionSpec,
    FullAttentionSpec,
    MomeSpec,
    ShareKVSlidingWindowSpec,
)
from vllm.v1.utils import CpuGpuBuffer  # noqa: E402

from omni_cache.cache.input_batch import (
    DSAGatherSelectionBlockTable,
    DSAPolicy,
    MomePolicy,
    OmniCacheBlockTable,
    OmniCacheInputBatch,
    OmniCacheSpecPolicy,
    ShareKVSlidingWindowPolicy,
)


class FakeOmniCache:
    def __init__(self, max_num_reqs=4, *, gather_selection=False):
        self.record_batch_idx_to_req = {i: None for i in range(max_num_reqs)}
        self.req_id_to_idx = {}
        self._enable_gs = gather_selection
        self._selection_state_size = 8
        if gather_selection:
            self.num_layers = 2
            self.s_max_block_num = 5

    @property
    def enable_gs(self):
        return self._enable_gs

    @property
    def selection_state_size(self):
        return self._selection_state_size

    @selection_state_size.setter
    def selection_state_size(self, value):
        self._selection_state_size = value

    def reserve_hbm_lane_for_request(self, req_id, lane=None):
        if req_id in self.req_id_to_idx:
            return self.req_id_to_idx[req_id]
        if lane is None:
            lane = next(
                (
                    idx
                    for idx, mapped_req_id in self.record_batch_idx_to_req.items()
                    if mapped_req_id is None
                ),
                None,
            )
        if lane is None or self.record_batch_idx_to_req.get(lane) is not None:
            raise RuntimeError(f"No free HBM lane for request {req_id}")
        self.record_batch_idx_to_req[lane] = req_id
        self.req_id_to_idx[req_id] = lane
        return lane

    def get_hbm_lane_for_request(self, req_id):
        return self.req_id_to_idx.get(req_id)

    def release_hbm_lane_for_request(self, req_id):
        lane = self.req_id_to_idx.pop(req_id, None)
        if lane is None:
            return None
        if self.record_batch_idx_to_req.get(lane) == req_id:
            self.record_batch_idx_to_req[lane] = None
        return lane


def _dsa_spec():
    return DSAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )


def _mome_spec():
    return MomeSpec(
        block_size=1,
        shapes=((1,), (1,), (1,)),
        dtypes=(torch.float16, torch.float16, torch.float16),
        kernel_size=2,
    )


def _share_kv_sliding_window_spec():
    return ShareKVSlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
        sliding_window=64,
    )


def _other_spec():
    return FullAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.float16,
    )


def _make_input_batch(specs, *, max_num_reqs=4, max_model_len=96, omni_cache=None):
    if omni_cache is None:
        omni_cache = FakeOmniCache(max_num_reqs=max_num_reqs)
    return OmniCacheInputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=256,
        block_sizes=[spec.block_size for spec in specs],
        kernel_block_sizes=[spec.block_size for spec in specs],
        kv_cache_specs=specs,
        kv_cache_config=Mock(kv_cache_groups=[]),
        omni_cache=omni_cache,
        is_pooling_model=True,
    )


def _request(req_id, block_ids, *, prompt_token_ids=None, output_token_ids=None):
    if prompt_token_ids is None:
        prompt_token_ids = [1, 2]
    if output_token_ids is None:
        output_token_ids = []
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        prompt_embeds=None,
        output_token_ids=output_token_ids,
        num_tokens=len(prompt_token_ids) + len(output_token_ids),
        num_computed_tokens=0,
        block_ids=block_ids,
        sampling_params=None,
        pooling_params=SimpleNamespace(requires_token_ids=False),
        pooling_states=object(),
        lora_request=None,
    )


def _reserve(batch, req_id, lane=None):
    return batch.omni_cache.reserve_hbm_lane_for_request(req_id, lane=lane)


def test_constructor_preserves_vllm_visible_fields():
    specs = [_dsa_spec()]
    batch = _make_input_batch(specs)

    assert batch.max_num_reqs == 4
    assert batch.max_model_len == 96
    assert batch.max_num_batched_tokens == 16
    assert batch.device == torch.device("cpu")
    assert batch.pin_memory is False
    assert batch.vocab_size == 256
    assert batch.kv_cache_specs == specs


def test_constructor_skips_gs_buffers_without_selection_metadata():
    batch = _make_input_batch([_dsa_spec()])

    assert batch.has_gs_buffers() is False
    assert batch._gs_block_table is None


def test_non_dsa_specs_do_not_create_gs_buffers():
    batch = _make_input_batch([_other_spec()])

    assert batch.has_gs_buffers() is False
    assert batch._gs_block_table is None


def test_constructor_raises_when_enabled_gs_metadata_is_missing():
    cache = FakeOmniCache()
    cache._enable_gs = True

    try:
        _make_input_batch([_dsa_spec()], omni_cache=cache)
    except RuntimeError as exc:
        assert "missing omni_cache attributes" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_constructor_raises_when_gs_enabled_without_dsa_group():
    cache = FakeOmniCache(gather_selection=True)

    try:
        _make_input_batch([_other_spec()], omni_cache=cache)
    except RuntimeError as exc:
        assert "no DSA block table" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_constructor_raises_when_enabled_gs_shape_is_invalid():
    cache = FakeOmniCache(gather_selection=True)
    cache.selection_state_size = 0

    try:
        _make_input_batch([_dsa_spec()], omni_cache=cache)
    except RuntimeError as exc:
        assert "Invalid GatherSelection buffer shape" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_constructor_initializes_gs_buffers_from_selection_metadata():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)
    table = batch.block_table.block_tables[0]

    assert batch.has_gs_buffers() is True
    assert isinstance(table, DSAGatherSelectionBlockTable)
    assert batch._gs_block_table is table
    assert isinstance(table.gs_status, torch.Tensor)
    assert isinstance(table.gs_table, CpuGpuBuffer)
    assert table.gs_status.shape == (2, 4, 8)
    assert table.gs_status.device == batch.device
    assert table.gs_status.dtype == torch.int32
    assert table.gs_table.np.shape == (4, 5)
    assert table.gs_status.tolist() == [[[-1] * 8] * 4] * 2
    assert table.gs_table.np[0].tolist() == [0, 1, 2, 3, 4]
    assert table.gs_table.np[1].tolist() == [5, 6, 7, 8, 9]


def test_commit_gs_buffers_copies_table_without_overwriting_status():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)
    table = batch._gs_block_table
    table.gs_status[0, 0, 0] = 42
    table.gs_table.np[0, 0] = 99

    batch.commit_gs_buffers(2)

    assert batch.get_gs_status_tensor(0, 2).shape == (2, 8)
    assert batch.get_gs_table_tensor(2).shape == (2, 5)
    assert batch.get_gs_status_tensor(0, 2)[0, 0].item() == 42
    assert batch.get_gs_table_tensor(2)[0, 0].item() == 99


def test_dsa_gs_add_row_resets_status_but_not_table():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)
    table = batch._gs_block_table
    table.gs_status[:, 0, :].fill_(77)
    table.gs_table.np[0, 0] = 99

    batch.add_request(_request("req-a", ([11],)))

    assert table.gs_status[:, 0, :].tolist() == [[-1] * 8] * 2
    assert table.gs_table.np[0].tolist() == [99, 1, 2, 3, 4]


def test_dsa_gs_rows_swap_with_request_rows():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)
    batch.add_request(_request("req-a", ([11],)))
    batch.add_request(_request("req-b", ([21],)))
    table = batch._gs_block_table
    table.gs_status[:, 0, :].fill_(10)
    table.gs_status[:, 1, :].fill_(20)
    table.gs_table.np[0] = [100, 101, 102, 103, 104]
    table.gs_table.np[1] = [200, 201, 202, 203, 204]

    batch.swap_states(0, 1)

    assert table.gs_status[:, 0, 0].tolist() == [20, 20]
    assert table.gs_status[:, 1, 0].tolist() == [10, 10]
    assert table.gs_table.np[0].tolist() == [200, 201, 202, 203, 204]
    assert table.gs_table.np[1].tolist() == [100, 101, 102, 103, 104]


def test_dsa_gs_rows_move_with_condense():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)
    batch.add_request(_request("req-a", ([11],)))
    batch.add_request(_request("req-b", ([21],)))
    batch.add_request(_request("req-c", ([31],)))
    table = batch._gs_block_table
    table.gs_status[:, 2, :].fill_(30)
    table.gs_table.np[2] = [300, 301, 302, 303, 304]

    batch.remove_request("req-b")
    batch.condense()

    assert batch.req_id_to_index == {"req-a": 0, "req-c": 1}
    assert table.gs_status[:, 1, 0].tolist() == [30, 30]
    assert table.gs_status[:, 2, :].tolist() == [[-1] * 8] * 2
    assert table.gs_table.np[1].tolist() == [300, 301, 302, 303, 304]
    assert table.gs_table.np[2].tolist() == [5, 6, 7, 8, 9]


def test_dsa_gs_condense_preserves_freed_table_workspace_for_later_add():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)
    batch.add_request(_request("req-a", ([11],)))
    batch.add_request(_request("req-b", ([21],)))
    batch.add_request(_request("req-c", ([31],)))
    table = batch._gs_block_table
    table.gs_table.np[0] = [100, 101, 102, 103, 104]
    table.gs_table.np[1] = [200, 201, 202, 203, 204]
    table.gs_table.np[2] = [300, 301, 302, 303, 304]

    batch.remove_request("req-b")
    batch.condense()
    batch.add_request(_request("req-d", ([41],)))

    assert batch.req_id_to_index == {
        "req-a": 0,
        "req-c": 1,
        "req-d": 2,
    }
    assert table.gs_table.np[0].tolist() == [100, 101, 102, 103, 104]
    assert table.gs_table.np[1].tolist() == [300, 301, 302, 303, 304]
    assert table.gs_table.np[2].tolist() == [200, 201, 202, 203, 204]
    assert table.gs_status[:, 2, :].tolist() == [[-1] * 8] * 2


def test_dsa_gs_reused_slot_resets_status_but_keeps_table_mapping():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)
    batch.add_request(_request("req-a", ([11],)))
    table = batch._gs_block_table
    table.gs_status[:, 0, :].fill_(42)
    table.gs_table.np[0] = [900, 901, 902, 903, 904]

    batch.remove_request("req-a")
    batch.add_request(_request("req-b", ([21],)))

    assert batch.req_id_to_index == {"req-b": 0}
    assert table.gs_status[:, 0, :].tolist() == [[-1] * 8] * 2
    assert table.gs_table.np[0].tolist() == [900, 901, 902, 903, 904]


def test_dsa_gs_clear_resets_status_and_table_layout():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)
    table = batch._gs_block_table
    table.gs_status.fill_(42)
    table.gs_table.np[0] = [900, 901, 902, 903, 904]

    table.clear()

    assert table.gs_status.tolist() == [[[-1] * 8] * 4] * 2
    assert table.gs_table.np[0].tolist() == [0, 1, 2, 3, 4]
    assert table.gs_table.np[1].tolist() == [5, 6, 7, 8, 9]


def test_policy_dispatch_by_spec_type():
    batch = _make_input_batch([
        _dsa_spec(),
        _mome_spec(),
        _share_kv_sliding_window_spec(),
        _other_spec(),
    ])

    assert isinstance(batch.spec_policies[0], DSAPolicy)
    assert isinstance(batch.spec_policies[1], MomePolicy)
    assert isinstance(batch.spec_policies[2], ShareKVSlidingWindowPolicy)
    assert isinstance(batch.spec_policies[3], OmniCacheSpecPolicy)


def test_base_policy_does_not_use_omni_block_table():
    policy = OmniCacheSpecPolicy(0, _other_spec())

    assert policy.initialize_group_block_table(Mock()) is None
    assert policy.uses_omni_block_table() is False
    assert policy.fill_omni_block_row(Mock(), Mock(), 0) is None


def test_first_request_reads_pre_reserved_lane():
    batch = _make_input_batch([_mome_spec()])
    _reserve(batch, "req-a")

    req_index = batch.add_request(_request("req-a", ([10],)))

    assert req_index == 0
    assert batch.get_hbm_lane("req-a") == 0
    assert batch.omni_cache.req_id_to_idx == {"req-a": 0}
    assert batch.omni_cache.record_batch_idx_to_req[0] == "req-a"


def test_removed_lane_is_reused_by_later_request():
    batch = _make_input_batch([_mome_spec()])
    _reserve(batch, "req-a")
    batch.add_request(_request("req-a", ([10],)))
    _reserve(batch, "req-b")
    batch.add_request(_request("req-b", ([20],)))

    assert batch.remove_request("req-a") == 0
    _reserve(batch, "req-c")
    req_index = batch.add_request(_request("req-c", ([30],)))

    assert req_index == 0
    assert batch.get_hbm_lane("req-c") == 0
    assert "req-a" not in batch.omni_cache.req_id_to_idx


def test_lane_remains_stable_after_swap_states():
    batch = _make_input_batch([_mome_spec()])
    _reserve(batch, "req-a")
    batch.add_request(_request("req-a", ([10],)))
    _reserve(batch, "req-b")
    batch.add_request(_request("req-b", ([20],)))

    before = dict(batch.omni_cache.req_id_to_idx)
    batch.swap_states(0, 1)

    assert batch.omni_cache.req_id_to_idx == before
    assert batch.req_id_to_index == {"req-a": 1, "req-b": 0}


def test_lane_remains_stable_after_condense():
    batch = _make_input_batch([_mome_spec()])
    _reserve(batch, "req-a")
    batch.add_request(_request("req-a", ([10],)))
    _reserve(batch, "req-b")
    batch.add_request(_request("req-b", ([20],)))
    _reserve(batch, "req-c")
    batch.add_request(_request("req-c", ([30],)))
    before = dict(batch.omni_cache.req_id_to_idx)

    batch.remove_request("req-b")
    batch.condense()

    assert batch.omni_cache.req_id_to_idx["req-a"] == before["req-a"]
    assert batch.omni_cache.req_id_to_idx["req-c"] == before["req-c"]
    assert "req-b" not in batch.omni_cache.req_id_to_idx


def test_missing_reserved_hbm_lane_raises():
    batch = _make_input_batch([_mome_spec()], max_num_reqs=1)

    try:
        batch.add_request(_request("req-a", ([10],)))
    except RuntimeError as exc:
        assert "No HBM lane reserved" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_mome_group_add_row_writes_lane_plus_one():
    batch = _make_input_batch([_mome_spec()])
    _reserve(batch, "req-a")

    batch.add_request(_request("req-a", ([99],)))

    table = batch.block_table.block_tables[0]
    assert isinstance(table, OmniCacheBlockTable)
    assert table.get_numpy_array()[0, 0] == 1
    assert table.vllm_block_table.np[0, 0] == 99


def test_mome_group_append_row_updates_only_vllm_table():
    batch = _make_input_batch([_mome_spec()])
    _reserve(batch, "req-a")
    batch.add_request(_request("req-a", ([99],)))
    table = batch.block_table.block_tables[0]
    before = table.get_numpy_array()[0].copy()

    batch.block_table.append_row(([123],), 0)

    assert table.get_numpy_array()[0].tolist() == before.tolist()
    assert table.vllm_block_table.np[0, :2].tolist() == [99, 123]
    assert table.num_blocks_per_row[0] == 2


def test_non_mome_groups_preserve_original_block_ids():
    batch = _make_input_batch([_dsa_spec(), _other_spec()])

    batch.add_request(_request("req-a", ([11, 12], [21, 22])))

    dsa_np = batch.block_table.block_tables[0].get_numpy_array()
    other_np = batch.block_table.block_tables[1].get_numpy_array()
    assert dsa_np[0, :2].tolist() == [11, 12]
    assert other_np[0, :2].tolist() == [21, 22]
    assert not hasattr(batch.block_table.block_tables[0], "vllm_block_table")
    assert not hasattr(batch.block_table.block_tables[1], "vllm_block_table")


def test_dsa_gs_wrapper_preserves_scheduler_block_ids():
    cache = FakeOmniCache(gather_selection=True)
    batch = _make_input_batch([_dsa_spec()], omni_cache=cache)

    batch.add_request(_request("req-a", ([11, 12],)))

    table = batch.block_table.block_tables[0]
    assert isinstance(table, DSAGatherSelectionBlockTable)
    assert table.get_numpy_array()[0, :2].tolist() == [11, 12]
    assert not hasattr(table, "vllm_block_table")


def test_mixed_groups_keep_scheduler_ids_in_vllm_tables():
    batch = _make_input_batch([
        _dsa_spec(),
        _mome_spec(),
        _share_kv_sliding_window_spec(),
    ])
    _reserve(batch, "req-a")

    batch.add_request(_request("req-a", ([11], [99], [21])))

    assert batch.block_table.block_tables[0].get_numpy_array()[0, 0] == 11
    assert batch.block_table.block_tables[1].get_numpy_array()[0, 0] == 1
    assert batch.block_table.block_tables[1].vllm_block_table.np[0, 0] == 99
    assert batch.block_table.block_tables[2].get_numpy_array()[0, :6].tolist() == [
        1, 2, 3, 4, 5, 1
    ]
    assert batch.block_table.block_tables[2].vllm_block_table.np[0, 0] == 21


def test_swa_only_batch_assigns_hbm_lane():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")

    batch.add_request(_request("req-a", ([99],)))

    assert batch.get_hbm_lane("req-a") == 0


def test_swa_lane_zero_static_row_uses_req_offset_ring():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")

    batch.add_request(_request("req-a", ([99],)))

    table = batch.block_table.block_tables[0]
    row = table.get_numpy_array()[0]
    assert row[:6].tolist() == [1, 2, 3, 4, 5, 1]
    assert table.vllm_block_table.np[0, 0] == 99


def test_swa_long_prompt_row_preserves_null_padding_phase():
    batch = _make_input_batch(
        [_share_kv_sliding_window_spec()],
        max_model_len=160,
    )
    _reserve(batch, "req-a")

    batch.add_request(
        _request("req-a", ([777, 778, 779],), prompt_token_ids=list(range(96)))
    )

    table = batch.block_table.block_tables[0]
    row = table.get_numpy_array()[0]
    assert row[:8].tolist() == [0, 0, 1, 2, 3, 4, 5, 1]
    assert table.vllm_block_table.np[0, :3].tolist() == [777, 778, 779]


def test_swa_long_prompt_raises_when_block_table_cannot_hold_null_prefix():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    policy = batch.spec_policies[0]
    input_batch = Mock(
        num_prompt_tokens=np.array([176], dtype=np.int32),
        get_hbm_lane_for_row=Mock(return_value=0),
    )
    block_table = Mock(
        get_numpy_array=Mock(return_value=np.zeros((1, 4), dtype=np.int32))
    )

    with pytest.raises(RuntimeError, match="too narrow for skipped blocks"):
        policy.fill_omni_block_row(input_batch, block_table, 0)


def test_swa_lane_one_static_row_starts_after_req_offset():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")

    batch.add_request(_request("req-a", ([99],)))
    _reserve(batch, "req-b")
    batch.add_request(_request("req-b", ([199],)))

    table = batch.block_table.block_tables[0]
    row = table.get_numpy_array()[1]
    assert row[:6].tolist() == [6, 7, 8, 9, 10, 6]
    assert table.vllm_block_table.np[1, 0] == 199


def test_swa_append_row_does_not_change_static_row():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")
    batch.add_request(_request("req-a", ([99],)))
    table = batch.block_table.block_tables[0]
    before = table.get_numpy_array()[0].copy()

    batch.block_table.append_row(([123, 124, 125],), 0)

    assert table.get_numpy_array()[0].tolist() == before.tolist()
    assert table.vllm_block_table.np[0, :4].tolist() == [99, 123, 124, 125]
    assert table.num_blocks_per_row[0] == 4


def test_swa_scheduler_block_ids_are_preserved_in_vllm_table():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")

    batch.add_request(_request("req-a", ([777, 778, 779],)))

    table = batch.block_table.block_tables[0]
    row = table.get_numpy_array()[0]
    assert 777 not in row[:6]
    assert row[:6].tolist() == [1, 2, 3, 4, 5, 1]
    assert table.vllm_block_table.np[0, :3].tolist() == [777, 778, 779]


def test_swa_static_rows_swap_with_request_rows():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")
    batch.add_request(_request("req-a", ([99],)))
    _reserve(batch, "req-b")
    batch.add_request(_request("req-b", ([199],)))

    batch.swap_states(0, 1)

    table = batch.block_table.block_tables[0].get_numpy_array()
    assert table[0, :6].tolist() == [6, 7, 8, 9, 10, 6]
    assert table[1, :6].tolist() == [1, 2, 3, 4, 5, 1]
    vllm_table = batch.block_table.block_tables[0].vllm_block_table.np
    assert vllm_table[0, 0] == 199
    assert vllm_table[1, 0] == 99
    assert batch.get_hbm_lane("req-a") == 0
    assert batch.get_hbm_lane("req-b") == 1


def test_swa_static_row_moves_with_condense():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")
    batch.add_request(_request("req-a", ([99],)))
    _reserve(batch, "req-b")
    batch.add_request(_request("req-b", ([199],)))
    _reserve(batch, "req-c")
    batch.add_request(_request("req-c", ([299],)))

    batch.remove_request("req-b")
    batch.condense()

    table = batch.block_table.block_tables[0].get_numpy_array()
    assert table[0, :6].tolist() == [1, 2, 3, 4, 5, 1]
    assert table[1, :6].tolist() == [11, 12, 13, 14, 15, 11]
    vllm_table = batch.block_table.block_tables[0].vllm_block_table.np
    assert vllm_table[0, 0] == 99
    assert vllm_table[1, 0] == 299
    assert batch.req_id_to_index == {"req-a": 0, "req-c": 1}
    assert batch.get_hbm_lane("req-c") == 2


def test_mome_and_swa_share_one_hbm_lane_per_request():
    batch = _make_input_batch([_mome_spec(), _share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")

    batch.add_request(_request("req-a", ([99], [199])))

    assert batch.get_hbm_lane("req-a") == 0
    assert batch.block_table.block_tables[0].get_numpy_array()[0, 0] == 1
    assert batch.block_table.block_tables[0].vllm_block_table.np[0, 0] == 99
    assert batch.block_table.block_tables[1].get_numpy_array()[0, :6].tolist() == [
        1, 2, 3, 4, 5, 1
    ]
    assert batch.block_table.block_tables[1].vllm_block_table.np[0, 0] == 199


def test_swa_compute_slot_mapping_uses_omni_table():
    batch = _make_input_batch([_share_kv_sliding_window_spec()])
    _reserve(batch, "req-a")
    batch.add_request(_request("req-a", ([777, 778, 779],)))
    table = batch.block_table.block_tables[0]

    table.compute_slot_mapping(
        np.array([0, 0, 0], dtype=np.int32),
        np.array([0, 16, 32], dtype=np.int32),
    )

    assert table.slot_mapping.np[:3].tolist() == [16, 32, 48]


def test_swa_long_prompt_slot_mapping_uses_compact_h2d_slots_after_null_prefix():
    batch = _make_input_batch(
        [_share_kv_sliding_window_spec()],
        max_model_len=160,
    )
    _reserve(batch, "req-a")
    batch.add_request(
        _request("req-a", ([777, 778, 779],), prompt_token_ids=list(range(96)))
    )
    table = batch.block_table.block_tables[0]

    table.compute_slot_mapping(
        np.array([0, 0, 0], dtype=np.int32),
        np.array([32, 48, 112], dtype=np.int32),
    )

    assert table.slot_mapping.np[:3].tolist() == [16, 32, 16]
