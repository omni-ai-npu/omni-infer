import inspect
from types import SimpleNamespace

import torch

from omni_npu.vllm_patches.usefull_patch.common import patch_block_table
from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_hybrid import patch_kv_cache_utils as patch_mod


class Spec:
    pass


def _run(monkeypatch, specs, override):
    captured = {}

    def create(all_specs, grouped_layers):
        captured["specs"] = all_specs
        captured["groups"] = grouped_layers
        return "created-groups"

    monkeypatch.setattr(patch_mod, "create_kv_cache_group_specs", create)
    monkeypatch.setattr(
        patch_mod.envs,
        "OMNI_HYBRID_ATTN_GROUP_SIZE",
        override,
        raising=False,
    )
    monkeypatch.setattr(
        patch_mod, "logger", SimpleNamespace(warning=lambda *args, **kwargs: None)
    )
    result = patch_mod._get_kv_cache_groups_uniform_page_size_patched(specs)
    return result, captured


def test_group_size_override_splits_each_attention_type(monkeypatch):
    full = Spec()
    sliding = Spec()
    specs = {
        "full.0": full,
        "full.1": full,
        "full.2": full,
        "sw.0": sliding,
    }

    result, captured = _run(monkeypatch, specs, override=2)

    assert result == "created-groups"
    assert captured["specs"] is specs
    assert captured["groups"] == [
        ["full.0", "full.2"],
        ["full.1"],
        ["sw.0"],
    ]


def test_close_group_sizes_use_larger_group_to_reduce_padding(monkeypatch):
    full = Spec()
    sliding = Spec()
    specs = {
        **{f"full.{index}": full for index in range(3)},
        **{f"sw.{index}": sliding for index in range(4)},
    }

    _, captured = _run(monkeypatch, specs, override=0)

    assert captured["groups"] == [
        ["full.0", "full.1", "full.2"],
        ["sw.0", "sw.1", "sw.2", "sw.3"],
    ]


def test_kv_cache_utils_patch_registration():
    cls = patch_mod.OverrideGroupSizePatch
    assert cls._target is patch_mod.kv_cache_utils
    assert cls._get_kv_cache_groups_uniform_page_size is (
        patch_mod._get_kv_cache_groups_uniform_page_size_patched
    )


def _run_slot_mapping(
    query_start_loc,
    positions,
    block_table,
    max_num_tokens,
    cp_world_size=1,
    cp_rank=0,
    cp_interleave_size=1,
):
    slot_mapping = torch.empty(max_num_tokens, dtype=torch.int64)
    patch_block_table._compute_slot_mapping_kernel_impl(
        num_tokens=positions.numel(),
        max_num_tokens=max_num_tokens,
        query_start_loc=query_start_loc,
        positions=positions,
        block_table=block_table,
        block_table_stride=block_table.stride(0),
        block_size=4,
        slot_mapping=slot_mapping,
        TOTAL_CP_WORLD_SIZE=cp_world_size,
        TOTAL_CP_RANK=cp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_interleave_size,
        PAD_ID=-1,
        BLOCK_SIZE=1,
    )
    return slot_mapping


def test_slot_mapping_is_vectorized_across_requests():
    query_start_loc = torch.tensor([0, 3, 3, 5], dtype=torch.int32)
    positions = torch.tensor([0, 5, 7, 8, 11], dtype=torch.int64)
    block_table = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ],
        dtype=torch.int32,
    )

    result = _run_slot_mapping(
        query_start_loc,
        positions,
        block_table,
        max_num_tokens=7,
    )

    assert torch.equal(result, torch.tensor([40, 45, 47, 128, 131, -1, -1]))
    source = inspect.getsource(patch_block_table._compute_slot_mapping_kernel_impl)
    assert ".item(" not in source
    assert "repeat_interleave" not in source

    empty_result = _run_slot_mapping(
        query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
        positions=torch.empty(0, dtype=torch.int64),
        block_table=torch.tensor([[10]], dtype=torch.int32),
        max_num_tokens=2,
    )
    assert torch.equal(empty_result, torch.tensor([-1, -1]))


def test_slot_mapping_vectorization_preserves_cp_interleave():
    result = _run_slot_mapping(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        positions=torch.tensor([0, 2, 4, 6], dtype=torch.int64),
        block_table=torch.tensor([[5, 6]], dtype=torch.int32),
        max_num_tokens=6,
        cp_world_size=2,
        cp_rank=0,
        cp_interleave_size=2,
    )

    assert torch.equal(result, torch.tensor([20, -1, 22, -1, -1, -1]))


def _run_slot_mapping_shared(block_table, shared, **cp_kwargs):
    slot_mapping = torch.empty(7, dtype=torch.int64)
    patch_block_table._compute_slot_mapping_kernel_impl(
        num_tokens=5,
        max_num_tokens=7,
        query_start_loc=torch.tensor([0, 3, 3, 5], dtype=torch.int32),
        positions=torch.tensor([0, 5, 7, 8, 11], dtype=torch.int64),
        block_table=block_table,
        block_table_stride=block_table.stride(0),
        block_size=4,
        slot_mapping=slot_mapping,
        TOTAL_CP_WORLD_SIZE=cp_kwargs.get("cp_world_size", 1),
        TOTAL_CP_RANK=cp_kwargs.get("cp_rank", 0),
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kwargs.get("cp_interleave_size", 1),
        PAD_ID=-1,
        BLOCK_SIZE=1,
        shared=shared,
    )
    return slot_mapping


def test_slot_mapping_shared_cache_reuses_group_invariants():
    table_a = torch.tensor(
        [[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32
    )
    table_b = torch.tensor(
        [[50, 51, 52], [60, 61, 62], [70, 71, 72]], dtype=torch.int32
    )

    shared = {}
    result_a = _run_slot_mapping_shared(table_a, shared)
    assert len(shared) == 1
    cached = next(iter(shared.values()))
    result_b = _run_slot_mapping_shared(table_b, shared)
    # Second group must hit the cache, not overwrite it.
    assert next(iter(shared.values())) is cached

    # Same results as fully independent computations.
    assert torch.equal(result_a, torch.tensor([40, 45, 47, 128, 131, -1, -1]))
    assert torch.equal(result_b, torch.tensor([200, 205, 207, 288, 291, -1, -1]))


def test_slot_mapping_shared_cache_matches_uncached_cp_path():
    table = torch.tensor([[5, 6]], dtype=torch.int32)

    def run(shared):
        slot_mapping = torch.empty(6, dtype=torch.int64)
        patch_block_table._compute_slot_mapping_kernel_impl(
            num_tokens=4,
            max_num_tokens=6,
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            positions=torch.tensor([0, 2, 4, 6], dtype=torch.int64),
            block_table=table,
            block_table_stride=table.stride(0),
            block_size=4,
            slot_mapping=slot_mapping,
            TOTAL_CP_WORLD_SIZE=2,
            TOTAL_CP_RANK=0,
            CP_KV_CACHE_INTERLEAVE_SIZE=2,
            PAD_ID=-1,
            BLOCK_SIZE=1,
            shared=shared,
        )
        return slot_mapping

    shared = {}
    first = run(shared)
    second = run(shared)  # cache hit
    expected = torch.tensor([20, -1, 22, -1, -1, -1])
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_slot_mapping_prebuilt_req_indices_matches_fallback():
    query_start_loc = torch.tensor([0, 3, 3, 5], dtype=torch.int32)
    positions = torch.tensor([0, 5, 7, 8, 11], dtype=torch.int64)
    block_table = torch.tensor(
        [[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32
    )
    expected = torch.tensor([40, 45, 47, 128, 131, -1, -1])

    def run(req_indices):
        slot_mapping = torch.empty(7, dtype=torch.int64)
        patch_block_table._compute_slot_mapping_kernel_impl(
            num_tokens=5,
            max_num_tokens=7,
            query_start_loc=query_start_loc,
            positions=positions,
            block_table=block_table,
            block_table_stride=block_table.stride(0),
            block_size=4,
            slot_mapping=slot_mapping,
            TOTAL_CP_WORLD_SIZE=1,
            TOTAL_CP_RANK=0,
            CP_KV_CACHE_INTERLEAVE_SIZE=1,
            PAD_ID=-1,
            BLOCK_SIZE=1,
            req_indices=req_indices,
        )
        return slot_mapping

    exact = torch.tensor([0, 0, 0, 2, 2], dtype=torch.int32)
    assert torch.equal(run(exact), expected)
    oversized = torch.tensor([0, 0, 0, 2, 2, 9, 9], dtype=torch.int32)
    assert torch.equal(run(oversized), expected)


class _FakeBlockTable:
    def __init__(self):
        self.seen = []

    def compute_slot_mapping(
        self, num_reqs, query_start_loc, positions, req_indices, shared
    ):
        self.seen.append((req_indices, shared))


def test_multi_group_resolves_source_once_and_shares_cache():
    fakes = [_FakeBlockTable(), _FakeBlockTable()]
    marker = torch.arange(4, dtype=torch.int32)
    src_calls = []

    def source(num_tokens):
        src_calls.append(num_tokens)
        return marker

    group = SimpleNamespace(block_tables=fakes)
    setattr(group, patch_block_table._REQ_INDICES_SOURCE_ATTR, source)
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    positions = torch.zeros(4, dtype=torch.int64)

    patch_block_table.compute_slot_mapping_multi(
        group, 2, query_start_loc, positions
    )

    assert src_calls == [4]
    (ri0, sh0), = fakes[0].seen
    (ri1, sh1), = fakes[1].seen
    assert ri0 is marker and ri1 is marker
    assert sh0 is sh1 and isinstance(sh0, dict)


def test_multi_group_without_source_passes_none():
    fake = _FakeBlockTable()
    group = SimpleNamespace(block_tables=[fake])
    patch_block_table.compute_slot_mapping_multi(
        group,
        2,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.zeros(4, dtype=torch.int64),
    )
    assert fake.seen[0][0] is None


def _make_block_table(block_table, slot_mapping, cp_world_size=1, cp_rank=0):
    return SimpleNamespace(
        pcp_world_size=cp_world_size,
        dcp_world_size=1,
        pcp_rank=cp_rank,
        dcp_rank=0,
        cp_kv_cache_interleave_size=1,
        block_size=4,
        max_num_batched_tokens=slot_mapping.numel(),
        block_table=SimpleNamespace(gpu=block_table),
        slot_mapping=SimpleNamespace(gpu=slot_mapping),
    )


def test_block_table_compute_slot_mapping_forwards_to_kernel(monkeypatch):
    monkeypatch.setattr(patch_block_table.block_table_module, "PAD_SLOT_ID", -1, raising=False)
    monkeypatch.setattr(
        patch_block_table.block_table_module,
        "_compute_slot_mapping_kernel",
        patch_block_table._compute_slot_mapping_kernel,
        raising=False,
    )
    slot_mapping = torch.empty(7, dtype=torch.int64)
    table = _make_block_table(
        torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32),
        slot_mapping,
    )

    patch_block_table.compute_slot_mapping(
        table,
        3,
        torch.tensor([0, 3, 3, 5], dtype=torch.int32),
        torch.tensor([0, 5, 7, 8, 11], dtype=torch.int64),
        torch.tensor([0, 0, 0, 2, 2], dtype=torch.int32),
        {},
    )

    assert torch.equal(slot_mapping, torch.tensor([40, 45, 47, 128, 131, -1, -1]))


def test_block_table_compute_slot_mapping_uses_cp_ranks(monkeypatch):
    monkeypatch.setattr(patch_block_table.block_table_module, "PAD_SLOT_ID", -1, raising=False)
    monkeypatch.setattr(
        patch_block_table.block_table_module,
        "_compute_slot_mapping_kernel",
        patch_block_table._compute_slot_mapping_kernel,
        raising=False,
    )
    slot_mapping = torch.empty(6, dtype=torch.int64)
    table = _make_block_table(
        torch.tensor([[5, 6]], dtype=torch.int32), slot_mapping, cp_world_size=2
    )
    table.cp_kv_cache_interleave_size = 2

    patch_block_table.compute_slot_mapping(
        table,
        1,
        torch.tensor([0, 4], dtype=torch.int32),
        torch.tensor([0, 2, 4, 6], dtype=torch.int64),
    )

    assert torch.equal(slot_mapping, torch.tensor([20, -1, 22, -1, -1, -1]))


def test_bind_req_indices_source_round_trip():
    patch_cls = patch_block_table.NPUMultiGroupBlockTableSlotMappingPatch
    group = SimpleNamespace(block_tables=[_FakeBlockTable()])
    marker = torch.arange(4, dtype=torch.int32)

    patch_cls._omni_bind_req_indices_source(group, lambda n: marker)
    assert getattr(group, patch_block_table._REQ_INDICES_SOURCE_ATTR) is not None
    patch_block_table.compute_slot_mapping_multi(
        group, 2, torch.tensor([0, 2, 4], dtype=torch.int32), torch.zeros(4, dtype=torch.int64)
    )
    assert group.block_tables[0].seen[0][0] is marker

    patch_cls._omni_bind_req_indices_source(group, None)
    group.block_tables[0].seen.clear()
    patch_block_table.compute_slot_mapping_multi(
        group, 2, torch.tensor([0, 2, 4], dtype=torch.int32), torch.zeros(4, dtype=torch.int64)
    )
    assert group.block_tables[0].seen[0][0] is None
