# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from omni.connector.kv_dump import (
    append_items,
    DecodeReq,
    Dump,
    gen_hash,
    load_layers,
    maybe_dump_kv,
    parse_remote_reqs,
    PrefillReq,
)

KV_DUMP_MODULE = "omni.connector.kv_dump"

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_maybe_dump_kv_singleton():
    maybe_dump_kv.__dict__.pop("dump_prefill", None)
    yield
    maybe_dump_kv.__dict__.pop("dump_prefill", None)

cfg_i32 = {"device": "cpu", "dtype": torch.int32}
cfg_uint8 = {"device": "cpu", "dtype": torch.uint8}
cfg_f32 = {"device": "cpu", "dtype": torch.float32}

# vllm extract_layer_index 要求层名中仅含一个整数
DEFAULT_LAYER_NAME = "model.layers.0.self_attn"


# =========================
# 1. 无效果或简单 mock
# =========================

def _layer_index_from_name(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


@contextmanager
def _patch_kv_dump_misc(*, layer_index_side_effect=None):
    side_effect = layer_index_side_effect or _layer_index_from_name
    with patch(f"{KV_DUMP_MODULE}.extract_layer_index", side_effect=side_effect):
        yield


@contextmanager
def _patch_dump_thread(*, run_id="1704067200"):
    mock_thread = MagicMock()
    with (
        patch(f"{KV_DUMP_MODULE}.threading.Thread", mock_thread),
        patch(f"{KV_DUMP_MODULE}.time.sleep", MagicMock()),
        patch(f"{KV_DUMP_MODULE}._make_run_id", return_value=run_id),
    ):
        yield mock_thread


@contextmanager
def _mock_world_group(rank=0):
    with patch(
        f"{KV_DUMP_MODULE}.get_world_group",
        return_value=SimpleNamespace(rank=rank),
    ):
        yield


# =========================
# 2. 共享 fixture 构造（供后续类测试复用）
# =========================

def _make_uint8_cache(
    num_blocks: int = 4,
    block_elems: int = 32,
    dim: int = 2,
) -> torch.Tensor:
    shape = (num_blocks, block_elems) if dim == 2 else (num_blocks, 4, block_elems)
    return torch.arange(
        num_blocks * block_elems, **cfg_i32
    ).reshape(shape)


def _make_shapes(num_blocks: int = 4, block_elems: int = 32) -> list[list[torch.Tensor]]:
    cache = _make_uint8_cache(num_blocks, block_elems)
    return load_layers([cache])


def _make_new_request(
    req_id: str = "req0",
    tokens: int = 128,
    block_ids=None,
) -> SimpleNamespace:
    if block_ids is None:
        block_ids = [[1, 2]]
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(range(tokens)),
        block_ids=block_ids,
    )


def _make_scheduler_output_prefill(
    req_ids=None,
    new_block_ids=None,
    new_reqs=None,
) -> MagicMock:
    if req_ids is None:
        req_ids = []
    if new_block_ids is None:
        new_block_ids = [None] * len(req_ids)
    cached = SimpleNamespace(
        req_ids=req_ids,
        new_block_ids=new_block_ids,
        num_computed_tokens=[],
    )
    return MagicMock(
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=new_reqs or [],
        kv_connector_metadata=SimpleNamespace(req_params={}),
    )


def _make_scheduler_output_decode(
    req_ids=None,
    computed=None,
    new_block_ids=None,
    new_reqs=None,
    req_params=None,
) -> MagicMock:
    if req_ids is None:
        req_ids = []
    if computed is None:
        computed = [0] * len(req_ids)
    if new_block_ids is None:
        new_block_ids = [None] * len(req_ids)
    meta = SimpleNamespace(req_params=req_params if req_params is not None else {})
    cached = SimpleNamespace(
        req_ids=req_ids,
        new_block_ids=new_block_ids,
        num_computed_tokens=computed,
    )
    return MagicMock(
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=new_reqs or [],
        kv_connector_metadata=meta,
    )


def _make_model_runner(*, kv_role: str = "kv_producer", spec_tokens: int = 0) -> MagicMock:
    layer = SimpleNamespace(kv_cache=[(_make_uint8_cache(),)])
    runner = MagicMock()
    runner.compilation_config.static_forward_context = {
        DEFAULT_LAYER_NAME: layer,
    }
    runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(layer_names=[DEFAULT_LAYER_NAME]),
    ]
    runner.vllm_config.kv_transfer_config.kv_role = kv_role
    runner.num_spec_tokens = spec_tokens
    return runner


def _make_prefill_req(
    req_id: str = "req0",
    tokens: int = 128,
    block_ids=None,
) -> PrefillReq:
    return PrefillReq(_make_new_request(req_id, tokens, block_ids))


class MockPrefillReq:
    """PrefillReq 的轻量 mock，供 DecodeReq / Dump 测试 patch 使用。"""

    def __init__(
        self,
        new=None,
        *,
        req_id: str = "req0",
        tok_num: int = 128,
        blocks=None,
        dump_data=None,
    ):
        if new is not None:
            self.req_id = new.req_id
            self.tok_num = len(new.prompt_token_ids)
            self.blocks = [[idx for idx in group] for group in new.block_ids]
        else:
            self.req_id = req_id
            self.tok_num = tok_num
            self.blocks = [group.copy() for group in (blocks or [[1, 2]])]
        self.block_set = {idx for group in self.blocks for idx in group}
        self._dump_data = dump_data
        self.data = None

    def append_blocks(self, new_blks):
        if new_blks is not None:
            for group, blocks in zip(self.blocks, new_blks):
                for idx in blocks:
                    assert idx not in self.block_set
                    self.block_set.add(idx)
                    group.append(idx)
        return self

    def dump(self, shapes, remote_req=None):
        if self._dump_data is not None:
            data = dict(self._dump_data)
        else:
            data = {f"group{i}": [] for i in range(len(self.blocks))}
        if remote_req is not None:
            data["remote_req_id"] = remote_req
        return data


@contextmanager
def _patch_prefill_req(mock_cls=MockPrefillReq):
    with patch(f"{KV_DUMP_MODULE}.PrefillReq", mock_cls):
        yield mock_cls


def _make_decode_req(
    req_id: str = "req0",
    tokens: int = 128,
    block_ids=None,
) -> DecodeReq:
    return DecodeReq(_make_new_request(req_id, tokens, block_ids))


def _make_decode_groups(
    num_blocks: int = 4,
    pg: int = 128,
    block_elems: int = 8,
) -> list[dict[str, tuple[torch.Tensor, ...]]]:
    cache = torch.arange(
        num_blocks * pg * block_elems, **cfg_i32
    ).reshape(num_blocks, pg, block_elems)
    assert cache.device.type == "cpu"
    return [{"layer0": (cache,)}]


class MockDecodeReq(MockPrefillReq):
    """DecodeReq 的轻量 mock，供 Dump 测试 patch 使用。"""

    def __init__(self, new=None, **kwargs):
        super().__init__(new, **kwargs)
        self.hist_hash = {}
        self.hashed_blocks = {}
        self.hashed_blks = []
        self.hashed_batch = None

    def update(self, tok_num, new_blks, groups, shapes, mtp=0, pg=128):
        self.append_blocks(new_blks)
        self.tok_num = tok_num
        return self

    def finalize(self):
        self.hashed_batch = None


@contextmanager
def _patch_decode_req(mock_cls=MockDecodeReq):
    with patch(f"{KV_DUMP_MODULE}.DecodeReq", mock_cls):
        yield mock_cls


def _make_dump(path: str = "/tmp/kv_dump") -> Dump:
    with _patch_dump_thread(run_id="1704067200"):
        return Dump(path)


# =========================
# 3. 全局函数
# =========================

class TestGenHash:

    def test_known_vector(self):
        buf = torch.tensor([1, 2, 3, 4], **cfg_uint8)
        kind, digest = gen_hash(buf)
        expected = hashlib.md5(buf.cpu().numpy().tobytes()).hexdigest()
        assert digest == expected
        assert kind == "uint8(4).blk"
        assert buf.device.type == "cpu"

    def test_invalid_dtype(self):
        with pytest.raises(ValueError):
            gen_hash(torch.tensor([1.0, 2.0], **cfg_f32))

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            gen_hash(torch.zeros(2, 2, **cfg_uint8))


class TestAppendItems:

    def test_empty_items_no_op(self, tmp_path):
        path = str(tmp_path / "out.json")
        append_items(path, {})
        assert not os.path.exists(path)

    def test_first_write(self, tmp_path):
        path = str(tmp_path / "out.json")
        append_items(path, {"req0": '{"a": 1}'})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"req0": {"a": 1}}

    def test_append_second_batch(self, tmp_path):
        path = str(tmp_path / "out.json")
        append_items(path, {"req0": '{"a": 1}'})
        append_items(path, {"req1": '{"b": 2}'})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"req0": {"a": 1}, "req1": {"b": 2}}

    def test_invalid_items_type(self, tmp_path):
        path = str(tmp_path / "out.json")
        with pytest.raises(TypeError):
            append_items(path, [])  # type: ignore[arg-type]


class TestLoadLayers:

    def test_list_format(self):
        cache = _make_uint8_cache(num_blocks=4, block_elems=16)
        shapes = load_layers([cache])
        assert len(shapes) == 1
        assert len(shapes[0]) == 1
        assert shapes[0][0].dtype == torch.uint8
        assert shapes[0][0].shape[0] == 4
        assert shapes[0][0].device.type == "cpu"
        assert cache.device.type == "cpu"

    def test_dict_format(self):
        cache = _make_uint8_cache(num_blocks=4, block_elems=16)
        with _patch_kv_dump_misc():
            shapes = load_layers({"model.layers.3.self_attn": cache})
        assert len(shapes) == 1
        assert shapes[0][0].shape[0] == 4

    def test_multi_tensor_per_layer(self):
        t1 = _make_uint8_cache(num_blocks=4, block_elems=8)
        t2 = _make_uint8_cache(num_blocks=4, block_elems=8) + 1
        shapes = load_layers([[t1, t2]])
        assert len(shapes) == 1
        assert len(shapes[0]) == 2

    def test_mismatched_block_num(self):
        t1 = _make_uint8_cache(num_blocks=4, block_elems=8)
        t2 = _make_uint8_cache(num_blocks=5, block_elems=8)
        with pytest.raises(ValueError):
            load_layers([t1, t2])

    def test_invalid_format(self):
        with pytest.raises(TypeError):
            load_layers(42)  # type: ignore[arg-type]


class TestParseRemoteReqs:

    def test_extract_prefill_ids(self):
        meta = SimpleNamespace(req_params={
            "decode_a": {"prefill_req_id": "prefill_a"},
            "decode_b": {"prefill_req_id": "prefill_b"},
        })
        scheduler_output = MagicMock(kv_connector_metadata=meta)
        assert parse_remote_reqs(scheduler_output) == {
            "decode_a": "prefill_a",
            "decode_b": "prefill_b",
        }

    def test_empty_params(self):
        scheduler_output = MagicMock(
            kv_connector_metadata=SimpleNamespace(req_params={}))
        assert parse_remote_reqs(scheduler_output) == {}


class TestMaybeDumpKv:

    def test_no_path_bypasses_dump(self):
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return "ok"

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", ""):
            wrapped = maybe_dump_kv(execute_model)
            runner = MagicMock()
            sched = MagicMock()
            assert wrapped(runner, sched) == "ok"
            assert "dump_prefill" not in maybe_dump_kv.__dict__

    def test_with_path_wraps_execute_model(self):
        mock_dump = MagicMock()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=None)
        cm.__exit__ = MagicMock(return_value=False)
        mock_dump.update.return_value = cm

        calls = []

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            calls.append((self, scheduler_output, intermediate_tensors))
            return "result"

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", "/tmp/kv_dump"):
            with patch(f"{KV_DUMP_MODULE}.Dump", return_value=mock_dump):
                wrapped = maybe_dump_kv(execute_model)

                runner = MagicMock()
                sched = MagicMock()
                tensors = MagicMock()
                assert wrapped(runner, sched, tensors) == "result"
                assert calls == [(runner, sched, tensors)]
                mock_dump.update.assert_called_once_with(runner, sched)
                cm.__enter__.assert_called_once()
                cm.__exit__.assert_called_once()

    def test_singleton_reuse(self):
        mock_dump = MagicMock()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=None)
        cm.__exit__ = MagicMock(return_value=False)
        mock_dump.update.return_value = cm

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return 1

        with patch(f"{KV_DUMP_MODULE}._kv_dump_path", "/tmp/kv_dump"):
            with patch(f"{KV_DUMP_MODULE}.Dump", return_value=mock_dump) as dump_cls:
                wrapped = maybe_dump_kv(execute_model)
                runner = MagicMock()
                sched = MagicMock()
                wrapped(runner, sched)
                wrapped(runner, sched)
                dump_cls.assert_called_once_with("/tmp/kv_dump")
                assert mock_dump.update.call_count == 2


# =========================
# 4. PrefillReq
# =========================

class TestPrefillReq:

    def test_init_copies_blocks(self):
        new = _make_new_request(block_ids=[[1, 2], [3]])
        req = PrefillReq(new)
        new.block_ids[0].append(99)
        assert req.blocks == [[1, 2], [3]]
        assert req.block_set == {1, 2, 3}
        assert req.req_id == "req0"
        assert req.tok_num == 128

    def test_append_blocks_success(self):
        req = _make_prefill_req(block_ids=[[1, 2]])
        req.append_blocks(([3],))
        assert req.blocks == [[1, 2, 3]]
        assert req.block_set == {1, 2, 3}
        assert req.append_blocks(None) is req

    def test_append_blocks_duplicate_raises(self):
        req = _make_prefill_req(block_ids=[[1, 2]])
        with pytest.raises(ValueError):
            req.append_blocks(([2],))

    def test_append_blocks_none(self):
        req = _make_prefill_req(block_ids=[[1, 2]])
        before = [group.copy() for group in req.blocks]
        assert req.append_blocks(None) is req
        assert req.blocks == before
        assert req.block_set == {1, 2}

    def test_dump_block_zero(self):
        req = _make_prefill_req(block_ids=[[0, 1]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        group0 = req.dump(shapes)["group0"]
        assert group0[0] == {"block_id": 0}
        assert len(group0[1]) == 2  # block_id + one shape hash layer

    def test_dump_negative_block(self):
        req = _make_prefill_req(block_ids=[[-3]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        blk = req.dump(shapes)["group0"][0]
        assert blk["block_id"] == 3
        assert blk["hashed"] is True

    def test_dump_with_hashes(self):
        req = _make_prefill_req(block_ids=[[1]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        blk = req.dump(shapes)["group0"][0]
        assert blk["block_id"] == 1
        hash_keys = [k for k in blk if k.endswith(".blk")]
        assert len(hash_keys) == len(shapes)
        for key in hash_keys:
            assert isinstance(blk[key], list)
            assert all(":" in item for item in blk[key])

    def test_dump_remote_req_id(self):
        req = _make_prefill_req(block_ids=[[0]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        data = req.dump(shapes, remote_req="prefill_req_1")
        assert data["remote_req_id"] == "prefill_req_1"

    def test_mock_prefill_req_for_upper_layer(self):
        with _patch_prefill_req():
            from omni.connector import kv_dump as mod
            req = mod.PrefillReq(_make_new_request(req_id="patched", block_ids=[[5]]))
            assert isinstance(req, MockPrefillReq)
            assert req.req_id == "patched"
            assert req.blocks == [[5]]
            data = req.dump(None, remote_req="prefill_0")
            assert data["remote_req_id"] == "prefill_0"


# =========================
# 5. DecodeReq
# =========================

class TestDecodeReq:

    def test_update_rest_zero(self):
        req = _make_decode_req(block_ids=[[0, 1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4)
        assert req.update(128, None, groups, shapes) is req
        assert req.tok_num == 128
        assert req.hashed_blks == [0]

    def test_update_with_new_blocks(self):
        req = _make_decode_req(block_ids=[[0, 1]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4)
        req.update(128, ([2],), groups, shapes)
        assert req.blocks == [[0, 1, 2]]
        assert req.block_set == {0, 1, 2}
        assert req.hashed_batch is not None

    def test_check_toks_incremental(self):
        req = _make_decode_req(block_ids=[[0, 1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4, pg=128, block_elems=8)
        req.update(129, None, groups, shapes, pg=128)
        assert "layer0.0" in req.hist_hash
        req.update(130, None, groups, shapes, pg=128)

    def test_check_toks_mismatch_raises(self):
        req = _make_decode_req(block_ids=[[0, 1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4, pg=128, block_elems=8)
        req.update(129, None, groups, shapes, pg=128)
        name = "layer0.0"
        full, rest, sect = req.hist_hash[name]
        # 篡改历史切片，使第二步 update 时 sect[:_rest] 与 hist 不一致
        req.hist_hash[name] = (full, rest, sect + 1)
        with pytest.raises(ValueError):
            req.update(130, None, groups, shapes, pg=128)

    def test_finalize_clears_batch(self):
        req = _make_decode_req(block_ids=[[0, 1, 2]])
        shapes = _make_shapes(num_blocks=4, block_elems=16)
        groups = _make_decode_groups(num_blocks=4)
        req.update(128, None, groups, shapes)
        assert req.hashed_batch is not None
        req.finalize()
        assert req.hashed_batch is None

    def test_mock_decode_req_for_upper_layer(self):
        with _patch_decode_req():
            from omni.connector import kv_dump as mod
            req = mod.DecodeReq(_make_new_request(req_id="dec0", block_ids=[[1]]))
            assert isinstance(req, MockDecodeReq)
            assert req.req_id == "dec0"
            assert req.update(64, ([2],), [], [], pg=128) is req
            assert req.blocks == [[1, 2]]
            req.hashed_batch = object()
            req.finalize()
            assert req.hashed_batch is None


# =========================
# 6. Dump
# =========================

class TestDump:

    def test_init_path_format(self):
        dump = _make_dump("/data/kv_dump")
        assert dump.path == "/data/kv_dump/1704067200/"
        assert dump.reqs == {}
        assert dump.rank is None

    def test_register_once(self):
        dump = _make_dump("/tmp/kv_dump")
        runner = _make_model_runner()
        dump._register(runner)
        shapes_first = dump.shapes
        groups_first = dump.groups
        dump._register(runner)
        assert dump.shapes is shapes_first
        assert dump.groups is groups_first
        assert dump.shapes[0][0].device.type == "cpu"

    def test_prefill_lifecycle(self):
        with _mock_world_group(rank=2):
            dump = _make_dump("/tmp/kv_dump")
            runner = _make_model_runner(kv_role="kv_producer")
            new = _make_new_request(req_id="p0", block_ids=[[0, 1]])

            with dump.update(runner, _make_scheduler_output_prefill(new_reqs=[new])):
                pass

            assert "p0" in dump.reqs
            assert "group0" in dump.reqs["p0"].data
            assert dump.rank == 2

            with dump.update(runner, _make_scheduler_output_prefill()):
                pass

            assert not dump.change.empty()
            req_id, data = dump.change.get()
            assert req_id == "p0"
            assert isinstance(data, dict)
            assert "group0" in data
            assert "p0" not in dump.reqs

    def test_decode_new_req_immediate_dump(self):
        with _mock_world_group():
            dump = _make_dump("/tmp/kv_dump")
            runner = _make_model_runner(kv_role="kv_consumer")
            new = _make_new_request(req_id="d0", block_ids=[[0, 1]])
            sched = _make_scheduler_output_decode(
                new_reqs=[new],
                req_params={"d0": {"prefill_req_id": "p0"}},
            )
            with dump.update(runner, sched):
                pass
            assert not dump.change.empty()
            req_id, data = dump.change.get()
            assert req_id == "d0"
            assert data["remote_req_id"] == "p0"
            assert "group0" in data

    def test_decode_cached_req_update(self):
        with _mock_world_group():
            dump = _make_dump("/tmp/kv_dump")
            runner = _make_model_runner(kv_role="kv_consumer")
            mock_req = MockDecodeReq(req_id="d0", blocks=[[0, 1]])
            dump.reqs = {"d0": mock_req}
            dump.shapes = _make_shapes(num_blocks=4, block_elems=16)
            dump.groups = _make_decode_groups(num_blocks=4)

            sched = _make_scheduler_output_decode(
                req_ids=["d0"],
                computed=[128],
                new_block_ids=[None],
            )
            with dump.update(runner, sched):
                pass

            assert mock_req.tok_num == 128
            assert "d0" in dump.reqs
            assert dump.reqs["d0"] is mock_req

    def test_task_save_json_writes_file(self, tmp_path):
        base = str(tmp_path / "dump")
        dump = _make_dump(base)
        dump.rank = 3
        dump.path = f"{base}/1704067200/"
        os.makedirs(dump.path, exist_ok=True)
        dump.change.put(("req0", {"group0": [{"block_id": 1}]}))

        calls = 0

        def sleep_side_effect(_sec):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise StopIteration

        with (
            patch(f"{KV_DUMP_MODULE}.append_items") as mock_append,
            patch(f"{KV_DUMP_MODULE}.time.sleep", side_effect=sleep_side_effect),
        ):
            with pytest.raises(StopIteration):
                dump._task_save_json()

        mock_append.assert_called_once()
        path, items = mock_append.call_args[0]
        assert path == dump.path + "rank3.json"
        assert "req0" in items

