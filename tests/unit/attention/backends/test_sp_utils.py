# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

MODULE = "omni_npu.attention.backends.utils"

cfg_i32 = {"device": "cpu", "dtype": torch.int32}
cfg_bf16 = {"device": "cpu", "dtype": torch.bfloat16}
cfg_f32 = {"device": "cpu", "dtype": torch.float32}

_PAGE_SIZE = 128
_TABLE_SIZE = 8


def _cdiv(a, b):
    return (a + b - 1) // b


def _init_cp_manager(
    cumlens: list[int],
    computed: list[int],
    page_size: int,
    table_size: int,
    cumlens_np: np.ndarray | None = None,
) -> Any:
    utils = importlib.import_module(MODULE)
    if cumlens_np is None:
        cumlens_np = np.asarray(cumlens, dtype=np.int32)
    return utils.SPManager.init_cp(
        cumlens=torch.tensor(cumlens, **cfg_i32),
        computed_lens=torch.tensor(computed, **cfg_i32),
        cumlens_np=cumlens_np,
        page_size=page_size,
        table_size=table_size,
        block_table_ref=torch.zeros(len(computed), table_size, **cfg_i32),
    )


# =========================
# 0. SeqGolden — canonical full -> sp | cp | tp
# 0b. KVSPGolden — full -> local_blocks (block interleave)
# =========================


class SeqGolden:
    """Canonical full [T] or [T,dim] -> sp / cp / tp goldens (see UT编写原则.md §6)."""

    def __init__(
        self,
        tok: int | None = None,
        cumlens: list[int] | None = None,
        dim: int = 1,
        ranks: int = 4,
        *,
        full: torch.Tensor | None = None,
    ) -> None:
        self.ranks = ranks
        if full is not None:
            self.full = full.clone()
        else:
            assert not (tok is not None and cumlens is not None)
            if cumlens is not None:
                tok = int(cumlens[-1])
            else:
                assert tok is not None
            if dim == 1:
                self.full = torch.arange(tok, **cfg_f32)
            else:
                assert dim % ranks == 0
                self.full = torch.randint(0, 10000, (tok, dim), dtype=torch.int64, device="cpu")
        self.dim = 1 if self.full.dim() == 1 else self.full.size(1)
        assert self.dim == 1 or self.dim % ranks == 0

        self.cumlens = (
            np.asarray(cumlens, dtype=np.int64) if cumlens is not None else None
        )
        self.tok = self.full.size(0)
        self.sp_len = _cdiv(self.tok, ranks)
        self.sp_align_len = self.sp_len * ranks
        self.tp_shard_dim = self.dim // ranks if self.dim > 1 else 1
        if self.tok == self.sp_align_len:
            self.aligned = self.full.clone()
        else:
            self.aligned = self.full.new_zeros(self.sp_align_len, *self.full.shape[1:])
            self.aligned[: self.tok] = self.full
        if self.cumlens is not None:
            frag_num = ranks * 2
            self.cp_len = sum(
                _cdiv(int(l), frag_num) for l in np.diff(self.cumlens)
            ) * 2
        else:
            self.cp_len = None

    def sp(self, rank: int) -> torch.Tensor:
        a = min(self.tok, rank * self.sp_len)
        b = min(self.tok, a + self.sp_len)
        if b == a + self.sp_len:
            return self.full[a:b].clone()
        y = self.full.new_zeros(self.sp_len, *self.full.shape[1:])
        if b > a:
            y[: b - a] = self.full[a:b]
        return y

    def _sp_valid_len(self, rank: int) -> int:
        a = min(self.tok, rank * self.sp_len)
        b = min(self.tok, a + self.sp_len)
        return b - a

    def _cp_segments(self, rank: int):
        assert self.cumlens is not None
        cn = self.cumlens
        frag_num = self.ranks * 2
        frag_lens = np.array(
            [_cdiv(int(l), frag_num) for l in np.diff(cn)],
            dtype=np.int64,
        )
        ends = cn[1:].repeat(2)
        frags = frag_lens.repeat(2)
        frags_base = frags.cumsum() - frags
        left = (
            np.stack(
                [
                    cn[:-1] + rank * frag_lens,
                    cn[:-1] + (frag_num - 1 - rank) * frag_lens,
                ]
            )
            .transpose()
            .flatten()
        )
        right = np.clip(left + frags, a_max=ends, a_min=None)
        for dst, src, end in zip(frags_base, left, right):
            src, end = int(src), int(end)
            if src < end:
                yield int(dst), src, end

    def cp(self, rank: int) -> torch.Tensor:
        # cp_slice layout: only cumlens (full index zigzag); computed 不影响本 golden
        y = self.full.new_zeros(self.cp_len, *self.full.shape[1:])
        for dst, src, end in self._cp_segments(rank):
            y[dst : dst + end - src] = self.full[src:end]
        return y

    def assert_sp(self, rank: int, actual: torch.Tensor) -> None:
        golden = self.sp(rank)
        assert actual.shape == golden.shape
        valid = self._sp_valid_len(rank)
        assert torch.equal(actual[:valid], golden[:valid])

    def assert_cp(self, rank: int, actual: torch.Tensor) -> None:
        assert actual.shape[0] == self.cp_len
        for dst, src, end in self._cp_segments(rank):
            assert torch.equal(actual[dst : dst + end - src], self.full[src:end])

    def tp(self, rank: int) -> torch.Tensor:
        if self.full.dim() == 1:
            assert self.ranks == 1 and rank == 0
            return self.full.clone()
        d = self.tp_shard_dim
        return self.full[:, rank * d : (rank + 1) * d].clone()


class KVSPGolden:
    """full → local_blocks via block interleave only (UT编写原则.md §6.5)."""

    def __init__(
        self,
        q_cumlens: list[int] | np.ndarray,
        kv_lens: list[int] | np.ndarray,
        page_size: int,
        ranks: int,
    ) -> None:
        self.ranks = ranks
        self.page_size = page_size
        self.cycle = page_size * ranks
        self.q_cumlens_np = np.asarray(q_cumlens, dtype=np.int64)
        self.kv_lens_np = np.asarray(kv_lens, dtype=np.int64)
        assert self.q_cumlens_np[0] == 0
        assert self.q_cumlens_np.size == self.kv_lens_np.size + 1
        self.q_lens = np.diff(self.q_cumlens_np)
        self.computed = self.kv_lens_np - self.q_lens
        self.tok = int(self.q_cumlens_np[-1])
        self.full = torch.arange(self.tok, **cfg_f32)

        self.local_blks = _cdiv(self.kv_lens_np, self.cycle)
        b = int(self.kv_lens_np.size)
        self.max_blocks = int(self.local_blks.max()) if b else 0
        self.num_pages = b * self.max_blocks
        self.blk_table = torch.zeros(b, self.max_blocks, **cfg_i32)
        for req in range(b):
            base = req * self.max_blocks
            for vp in range(int(self.local_blks[req])):
                self.blk_table[req, vp] = base + vp

    def _kv_value(self, req: int, kv_pos: int) -> torch.Tensor | None:
        comp = int(self.computed[req])
        if kv_pos < comp:
            return None
        qi = kv_pos - comp
        if qi >= int(self.q_lens[req]):
            return None
        return self.full[int(self.q_cumlens_np[req]) + qi]

    def _batch_slot(self, req: int, serial: int) -> tuple[int, int]:
        pg = self.page_size
        comp = int(self.computed[req])
        idx = serial + comp // self.ranks
        row = int(self.blk_table[req, idx // pg].item())
        return row, idx % pg

    def _written_cells(self, rank: int) -> list[tuple[int, int, torch.Tensor]]:
        pg = self.page_size
        n = self.ranks
        cells: list[tuple[int, int, torch.Tensor]] = []
        for b in range(self.kv_lens_np.size):
            kv_len = int(self.kv_lens_np[b])
            comp = int(self.computed[b])
            serial = 0
            for t in range(comp, kv_len):
                if (t // pg) % n != rank:
                    continue
                row, off = self._batch_slot(b, serial)
                serial += 1
                val = self._kv_value(b, t)
                if val is not None:
                    cells.append((row, off, val))
        return cells

    def local_blocks(self, rank: int) -> torch.Tensor:
        pg = self.page_size
        blocks = self.full.new_full((self.num_pages, pg), -1.0)
        for row, off, val in self._written_cells(rank):
            blocks[row, off] = val
        return blocks

    def assert_local_blocks(self, rank: int, cache: torch.Tensor) -> None:
        view = cache[..., 0] if cache.dim() > 2 else cache
        golden = self.local_blocks(rank)
        assert view.shape == golden.shape
        assert torch.equal(view, golden)

    def reconstruct_from_ag(self, gathered: torch.Tensor) -> torch.Tensor:
        pg = self.page_size
        n = self.ranks
        cycle = self.cycle
        view = gathered[..., 0] if gathered.dim() > 2 else gathered
        pages: list[int] = []
        for b in range(self.kv_lens_np.size):
            lb = int(_cdiv(int(self.kv_lens_np[b]), cycle))
            for vp in range(lb):
                pages.append(int(self.blk_table[b, vp].item()))
        num_local_pages = len(pages)
        row_to_pos = {row: i for i, row in enumerate(pages)}
        recon = self.full.clone()
        for b in range(self.kv_lens_np.size):
            kv_len = int(self.kv_lens_np[b])
            comp = int(self.computed[b])
            q0 = int(self.q_cumlens_np[b])
            serial_on_rank = [0] * n
            for t in range(comp, kv_len):
                r = (t // pg) % n
                serial = serial_on_rank[r]
                idx = serial + comp // n
                row = int(self.blk_table[b, idx // pg].item())
                off = idx % pg
                gather_idx = r * num_local_pages + row_to_pos[row]
                recon[q0 + (t - comp)] = view[gather_idx, off].to(recon.dtype)
                serial_on_rank[r] += 1
        return recon

    def assert_full_from_ag(self, gathered: torch.Tensor) -> None:
        assert torch.equal(self.reconstruct_from_ag(gathered), self.full)


_KVSP_PAGE_SIZE = 32


# =========================
# 1. 环境与模块加载
# =========================


@contextmanager
def _mock_torch_npu_stream() -> Iterator[None]:
    mock_npu = MagicMock()
    mock_npu.current_stream.return_value = MagicMock()
    mock_npu.Stream.return_value = MagicMock()
    mock_npu.stream.side_effect = lambda _: nullcontext()
    with patch("torch.npu", mock_npu, create=True):
        yield


@contextmanager
def _mock_misc() -> Iterator[None]:
    with (
        patch("vllm.logger.init_logger", return_value=MagicMock()),
        patch("vllm.platforms.current_platform", MagicMock(device_type="cpu")),
    ):
        yield


@contextmanager
def _utils_env() -> Iterator[Any]:
    with _mock_torch_npu_stream(), _mock_misc():
        yield importlib.import_module(MODULE)


# =========================
# 2. TaskDist（通信 mock 与自检）
# =========================


class MockGroup:
    def __init__(
        self,
        rank: int,
        dist: "TaskDist",
    ) -> None:
        self.rank_in_group = rank
        self.world_size = dist.ranks
        self.device_group = dist

    def all_gather(
        self,
        x: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        assert dim == 0
        out = x.new_empty(x.size(0) * self.world_size, *x.shape[1:])
        self.device_group._ag_v(out, x, self.device_group)
        return out


class TaskDist:
    thread_local = threading.local()

    def __init__(
        self,
        ranks: int = 4,
    ) -> None:
        self.ranks = ranks
        self.mailboxes = [queue.Queue() for _ in range(ranks)]
        self.errors = queue.Queue()
        self.groups = [MockGroup(r, self) for r in range(ranks)]

    def run(
        self,
        task: Callable[[], None],
    ) -> None:
        for q in self.mailboxes:
            while not q.empty():
                q.get_nowait()
        while not self.errors.empty():
            self.errors.get_nowait()

        with (
            patch(f"{MODULE}.get_tp_group", side_effect=self._group),
            patch("torch.distributed.all_to_all_single", side_effect=self._a2a_v),
            patch("torch.distributed.all_gather_into_tensor", side_effect=self._ag_v),
        ):
            threads = []
            for r in range(self.ranks):
                t = threading.Thread(target=self._worker, args=(r, task))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        assert self.errors.empty(), self.errors.get()

    def _worker(
        self,
        rank: int,
        task: Callable[[], None],
    ) -> None:
        try:
            self.thread_local.rank = rank
            task()
        except Exception as e:
            self.errors.put(e)

    def _group(self) -> MockGroup:
        return self.groups[self.thread_local.rank]

    def _exchange(
        self,
        sends: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        rank = self.thread_local.rank
        for dst, piece in enumerate(sends):
            self.mailboxes[dst].put((rank, piece))
        recvs = [None] * self.ranks
        for _ in range(self.ranks):
            src, piece = self.mailboxes[rank].get()
            recvs[src] = piece
        return recvs

    def _a2a_v(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: "TaskDist",
    ) -> None:
        assert group is self
        assert len(output_split_sizes) == self.ranks
        assert len(input_split_sizes) == self.ranks
        assert sum(output_split_sizes) == output.size(0)
        assert sum(input_split_sizes) == input.size(0)
        sends = [it.clone() for it in torch.split(input, input_split_sizes)]
        recvs = self._exchange(sends)
        for nbytes, (out_view, recv) in zip(
            output_split_sizes, zip(torch.split(output, output_split_sizes), recvs)
        ):
            assert recv.size(0) == nbytes
            out_view.copy_(recv)

    def _ag_v(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: "TaskDist",
    ) -> None:
        assert group is self
        assert output_tensor.size(0) == input_tensor.size(0) * self.ranks
        assert output_tensor.shape[1:] == input_tensor.shape[1:]
        assert output_tensor.numel() == input_tensor.numel() * self.ranks
        piece = input_tensor.clone().flatten()
        recvs = self._exchange([piece] * self.ranks)
        assert len(recvs) == self.ranks
        output_tensor.flatten().copy_(torch.cat(recvs))


@pytest.mark.unit
class TestTaskDist:
    def test_all_to_all(self):
        sp = TaskDist(ranks=4)
        results = {}

        def task():
            rank = sp.thread_local.rank
            group = sp._group()
            inp = torch.tensor([[rank * 10 + d] for d in range(4)], **cfg_f32)
            out = torch.zeros(4, 1, **cfg_f32)
            torch.distributed.all_to_all_single(
                out, inp, [1, 1, 1, 1], [1, 1, 1, 1], group.device_group
            )
            results[rank] = out.clone()

        sp.run(task)
        for dst in range(4):
            for src in range(4):
                assert results[dst][src, 0].item() == src * 10 + dst

    def test_all_gather(self):
        sp = TaskDist(ranks=4)
        results = {}

        def task():
            rank = sp.thread_local.rank
            group = sp._group()
            inp = torch.tensor([[float(rank)]], **cfg_f32)
            out = torch.zeros(4, 1, **cfg_f32)
            torch.distributed.all_gather_into_tensor(out, inp, group.device_group)
            results[rank] = out.clone()

        sp.run(task)
        expected = torch.tensor([[0.0], [1.0], [2.0], [3.0]], **cfg_f32)
        for rank in range(4):
            assert torch.equal(results[rank], expected)

    def test_get_tp_group_per_thread(self):
        sp = TaskDist(ranks=4)
        seen = {}

        def task():
            utils = importlib.import_module(MODULE)
            group = utils.get_tp_group()
            seen[group.rank_in_group] = group.world_size

        with _utils_env():
            sp.run(task)

        assert seen == {0: 4, 1: 4, 2: 4, 3: 4}


# =========================
# 3. sp_ctrl — TestSPCtrl
# =========================


@pytest.mark.unit
class TestSPCtrl:
    """sp_ctrl：align_tokens / slice_tokens / ag_tokens 三项 API 独立测（对应 utils.py §sp_ctrl）。"""

    def _run_align_case(self, tok: int, ranks: int = 4) -> None:
        g = SeqGolden(tok=tok, ranks=ranks)
        sp = TaskDist(ranks=ranks)
        results: dict[int, torch.Tensor] = {}

        def task():
            utils = importlib.import_module(MODULE)
            mgr = utils.SPManager.init_sp(tok)
            results[sp.thread_local.rank] = mgr.align_tokens(g.full).clone()

        with _utils_env():
            sp.run(task)

        for rank in range(ranks):
            assert torch.equal(results[rank], g.aligned)

    def _run_slice_case(self, tok: int, ranks: int = 4, *, aligned: bool = False) -> None:
        g = SeqGolden(tok=tok, ranks=ranks)
        x_in = g.aligned if aligned else g.full
        sp = TaskDist(ranks=ranks)
        results: dict[int, torch.Tensor] = {}

        def task():
            utils = importlib.import_module(MODULE)
            mgr = utils.SPManager.init_sp(tok)
            results[sp.thread_local.rank] = mgr.slice_tokens(x_in).clone()

        with _utils_env():
            sp.run(task)

        for rank in range(ranks):
            assert torch.equal(results[rank], g.sp(rank))

    def _run_ag_case(self, tok: int, ranks: int = 4) -> None:
        g = SeqGolden(tok=tok, ranks=ranks)
        sp = TaskDist(ranks=ranks)
        results: dict[int, torch.Tensor] = {}

        def task():
            utils = importlib.import_module(MODULE)
            rank = sp.thread_local.rank
            mgr = utils.SPManager.init_sp(tok)
            results[rank] = mgr.ag_tokens(g.sp(rank)).clone()

        with _utils_env():
            sp.run(task)

        ref = results[0]
        assert ref.shape[0] == tok
        assert torch.equal(ref, g.full)
        for rank in range(1, ranks):
            assert torch.equal(results[rank], ref)

    def test_slice_exact_16(self):
        self._run_slice_case(16)

    def test_slice_exact_64(self):
        self._run_slice_case(64)

    def test_slice_len_minus_1(self):
        self._run_slice_case(63)

    def test_slice_len_plus_1(self):
        self._run_slice_case(65)

    def test_slice_tiny_lt_rank(self):
        self._run_slice_case(3)

    def test_slice_one_per_rank(self):
        self._run_slice_case(4)

    def test_slice_aligned_len_minus_1(self):
        self._run_slice_case(63, aligned=True)

    def test_slice_aligned_len_plus_1(self):
        self._run_slice_case(65, aligned=True)

    def test_slice_aligned_tiny_lt_rank(self):
        self._run_slice_case(3, aligned=True)

    def test_align_len_minus_1(self):
        self._run_align_case(63)

    def test_align_len_plus_1(self):
        self._run_align_case(65)

    def test_align_tiny_lt_rank(self):
        self._run_align_case(3)

    def test_ag_exact_16(self):
        self._run_ag_case(16)

    def test_ag_exact_64(self):
        self._run_ag_case(64)

    def test_ag_len_minus_1(self):
        self._run_ag_case(63)

    def test_ag_len_plus_1(self):
        self._run_ag_case(65)

    def test_ag_tiny_lt_rank(self):
        self._run_ag_case(3)

    def test_ag_one_per_rank(self):
        self._run_ag_case(4)


# =========================
# 4. cp_reorg — TestCPReorg
# =========================


@pytest.mark.unit
class TestCPReorg:
    """cp_reorg：sp_to_cp / cp_to_sp / sp_to_tp（不测 sp_ctrl 的 slice/ag）。"""

    def _run_cp_case(
        self,
        cumlens: list[int],
        computed: list[int],
        ranks: int = 4,
        page_size: int = _PAGE_SIZE,
        table_size: int = _TABLE_SIZE,
    ) -> None:
        g = SeqGolden(cumlens=cumlens, ranks=ranks)
        sp = TaskDist(ranks=ranks)
        sp_to_cp_out: dict[int, torch.Tensor] = {}
        cp_to_sp_out: dict[int, torch.Tensor] = {}

        def task():
            mgr = _init_cp_manager(cumlens, computed, page_size, table_size)
            rank = sp.thread_local.rank
            sp_to_cp_out[rank] = mgr.sp_to_cp(g.sp(rank)).clone()
            cp_to_sp_out[rank] = mgr.cp_to_sp(g.cp(rank)).clone()

        with _utils_env():
            sp.run(task)

        for rank in range(ranks):
            g.assert_cp(rank, sp_to_cp_out[rank])
            g.assert_sp(rank, cp_to_sp_out[rank])

    def _run_cp_roundtrip_case(
        self,
        cumlens: list[int],
        computed: list[int],
        ranks: int = 4,
        page_size: int = _PAGE_SIZE,
        table_size: int = _TABLE_SIZE,
    ) -> None:
        g = SeqGolden(cumlens=cumlens, ranks=ranks)
        sp = TaskDist(ranks=ranks)
        results: dict[int, torch.Tensor] = {}

        def task():
            mgr = _init_cp_manager(cumlens, computed, page_size, table_size)
            cp_local = mgr.sp_to_cp(g.sp(sp.thread_local.rank))
            results[sp.thread_local.rank] = mgr.cp_to_sp(cp_local).clone()

        with _utils_env():
            sp.run(task)

        for rank in range(ranks):
            g.assert_sp(rank, results[rank])

    def test_cp_roundtrip_smoke(self):
        self._run_cp_roundtrip_case([0, 16], [0])

    def test_cp_single_exact(self):
        self._run_cp_case([0, 16], [0])

    def test_cp_single_padded(self):
        self._run_cp_case([0, 15], [0])

    def test_cp_multi_req(self):
        self._run_cp_case([0, 8, 20], [0, 0])

    def test_cp_prefix_computed(self):
        self._run_cp_case([0, 64], [2])

    def test_cp_uneven_req_boundary(self):
        self._run_cp_case([0, 5, 32], [0, 0])

    def test_cp_sp_exact_64(self):
        self._run_cp_case([0, 64], [0])

    def test_cp_sp_len_minus_1(self):
        self._run_cp_case([0, 63], [0])

    def test_cp_sp_len_plus_1(self):
        self._run_cp_case([0, 65], [0])

    def test_cp_frag_minus_1(self):
        self._run_cp_case([0, 31], [0])

    def test_cp_frag_plus_1(self):
        self._run_cp_case([0, 33], [0])

    def test_cp_short_req(self):
        self._run_cp_case([0, 7], [0])

    def test_cp_many_req_varied_lens(self):
        cumlens = [0, 3, 10, 21, 38, 57, 80, 109, 140, 177, 218]
        self._run_cp_case(cumlens, [0] * (len(cumlens) - 1))

    def _run_sp_to_tp_case(self, T: int, D: int, ranks: int = 4) -> None:
        g = SeqGolden(tok=T, dim=ranks * D, ranks=ranks)
        inputs = {r: g.full.clone() for r in range(ranks)}
        sp = TaskDist(ranks=ranks)
        results: dict[int, torch.Tensor] = {}

        def task():
            utils = importlib.import_module(MODULE)
            mgr = utils.SPManager(utils.get_tp_group())
            rank = sp.thread_local.rank
            results[rank] = mgr.sp_to_tp(inputs[rank]).clone()

        with _utils_env():
            sp.run(task)

        for rank in range(ranks):
            parts = [inputs[s].view(T, ranks, D)[:, rank, :] for s in range(ranks)]
            golden = torch.cat(parts, dim=0)
            assert results[rank].shape == (ranks * T, D)
            assert torch.equal(results[rank], golden)

    def test_sp_to_tp_basic(self):
        self._run_sp_to_tp_case(T=4, D=2)

    def test_sp_to_tp_short_t(self):
        self._run_sp_to_tp_case(T=1, D=4)

    def test_sp_to_tp_d1(self):
        self._run_sp_to_tp_case(T=8, D=1)

    def test_sp_to_tp_large_d(self):
        self._run_sp_to_tp_case(T=3, D=8)


# =========================
# 5. cp_slice — TestCPSlice
# =========================


@pytest.mark.unit
class TestCPSlice:
    """cp_slice：init_cp 触发规划后，各 rank 对全序列 X 调 cp_slice（对应 utils.py §cp_slice）。"""

    def _run_cp_slice_case(
        self,
        query_start_locs: list[int],
        computed_token_lens: list[int],
        rank_count: int = 4,
        kv_page_size: int = _PAGE_SIZE,
        block_table_width: int = _TABLE_SIZE,
    ) -> None:
        g = SeqGolden(cumlens=query_start_locs, ranks=rank_count)
        sp = TaskDist(ranks=rank_count)
        results: dict[int, torch.Tensor] = {}

        def task():
            mgr = _init_cp_manager(
                query_start_locs,
                computed_token_lens,
                kv_page_size,
                block_table_width,
            )
            results[sp.thread_local.rank] = mgr.cp_slice(g.full).clone()

        with _utils_env():
            sp.run(task)

        for rank in range(rank_count):
            assert torch.equal(results[rank], g.cp(rank))

    def test_cp_slice_single_exact(self):
        self._run_cp_slice_case([0, 16], [0])

    def test_cp_slice_uneven_boundary(self):
        self._run_cp_slice_case([0, 5, 32], [0, 0])

    def test_cp_slice_short_req(self):
        self._run_cp_slice_case([0, 7], [0])

    def test_cp_slice_sp_plus_1(self):
        self._run_cp_slice_case([0, 65], [0])


# =========================
# 6. cp_attn — TestCPAttn
# =========================


@pytest.mark.unit
class TestCPAttn:
    """cp_attn：cp_attn_meta 可观测契约（对应 utils.py §cp_attn）。"""

    def _cp_len(
        self,
        cumlens_np: np.ndarray,
        ranks: int,
    ) -> int:
        frag_num = ranks * 2
        frag_lens = _cdiv(np.diff(cumlens_np), frag_num)
        return int(frag_lens.sum() * 2)

    def _run_cp_attn_meta_case(
        self,
        cumlens: list[int],
        computed: list[int],
        ranks: int = 4,
        page_size: int = _PAGE_SIZE,
        table_size: int = _TABLE_SIZE,
    ) -> None:
        cn = np.array(cumlens, dtype=np.int32)
        cp_len = self._cp_len(cn, ranks)
        sp = TaskDist(ranks=ranks)
        results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def task():
            mgr = _init_cp_manager(cumlens, computed, page_size, table_size, cn)
            q_cumlens, kv_lens, _, _ = mgr.cp_attn_meta()
            results[sp.thread_local.rank] = (q_cumlens.clone(), kv_lens.clone())

        with _utils_env():
            sp.run(task)

        for rank in range(ranks):
            q_cumlens, kv_lens = results[rank]
            assert q_cumlens[-1].item() == cp_len
            assert kv_lens.numel() == 2 * len(computed)
            assert kv_lens.ge(torch.tensor(computed, **cfg_i32).repeat_interleave(2)).all()

    def test_cp_attn_meta_single_exact(self):
        self._run_cp_attn_meta_case([0, 16], [0])

    def test_cp_attn_meta_multi_req(self):
        self._run_cp_attn_meta_case([0, 8, 20], [0, 0])

    def test_cp_attn_meta_prefix_computed(self):
        self._run_cp_attn_meta_case([0, 64], [2])


# =========================
# 7. KVSPMaganer — TestKVSP
# =========================


@contextmanager
def _utils_env_or_skip_hccl():
    try:
        with _utils_env() as utils:
            yield utils
    except ImportError as exc:
        if "libhccl.so" in str(exc):
            pytest.skip("requires torch_npu/HCCL runtime")
        raise


@pytest.mark.unit
class TestDummyKVSP:
    def test_dummy_sp_to_local_returns_callable_or_empty_tensor(self):
        with _utils_env_or_skip_hccl() as utils:
            manager = utils.DummyKVSPMaganer()
            source = torch.ones(2, 3, **cfg_f32)

            comm = manager.sp_to_local(source, seperate=True)
            direct = manager.sp_to_local(source, seperate=False)

            assert isinstance(comm, Callable)
            assert torch.equal(comm(), direct)
            assert tuple(direct.shape) == (0, 3)

    def test_dummy_ag_pages_returns_callable_or_none(self):
        with _utils_env_or_skip_hccl() as utils:
            manager = utils.DummyKVSPMaganer()
            cache = torch.ones(2, 3, **cfg_f32)

            comm, send_split, recv_split = manager.ag_pages(cache, seperate=True)
            direct, direct_send_split, direct_recv_split = manager.ag_pages(cache, seperate=False)

            assert isinstance(comm, Callable)
            assert comm() is None
            assert direct is None
            assert send_split is None
            assert recv_split is None
            assert direct_send_split is None
            assert direct_recv_split is None


@pytest.mark.unit
class TestKVSP:
    """KVSPMaganer：select/scatter、sp_to_local/scatter、ag_pages 间接还原。"""

    def _run_kvsp_select_scatter_case(
        self,
        q_cumlens: list[int],
        kv_lens: list[int],
        ranks: int = 4,
        page_size: int = _KVSP_PAGE_SIZE,
    ) -> None:
        g = KVSPGolden(q_cumlens, kv_lens, page_size, ranks)
        cu = np.array(q_cumlens, dtype=np.int32)
        kl = np.array(kv_lens, dtype=np.int32)
        blk = g.blk_table.to(torch.int32)
        caches: dict[int, torch.Tensor] = {}
        sp = TaskDist(ranks=ranks)

        def task():
            utils = importlib.import_module(MODULE)
            mgr = utils.KVSPMaganer(cu, kl, blk, page_size=page_size)
            rank = sp.thread_local.rank
            local = mgr.select_local(g.full)
            slots, _ = mgr.local_slots()
            cache = g.full.new_full((g.num_pages, g.page_size), -1.0)
            cache.view(-1)[slots.long()] = local
            caches[rank] = cache

        with _utils_env():
            sp.run(task)

        for rank in range(ranks):
            g.assert_local_blocks(rank, caches[rank])

    def _run_kvsp_sp_to_local_scatter_case(
        self,
        q_cumlens: list[int],
        kv_lens: list[int],
        ranks: int = 4,
        page_size: int = _KVSP_PAGE_SIZE,
    ) -> None:
        g = KVSPGolden(q_cumlens, kv_lens, page_size, ranks)
        cu = np.array(q_cumlens, dtype=np.int32)
        kl = np.array(kv_lens, dtype=np.int32)
        blk = g.blk_table.to(torch.int32)
        sg = SeqGolden(full=g.full, ranks=ranks)
        caches: dict[int, torch.Tensor] = {}
        sp = TaskDist(ranks=ranks)

        def task():
            utils = importlib.import_module(MODULE)
            mgr = utils.KVSPMaganer(cu, kl, blk, page_size=page_size)
            rank = sp.thread_local.rank
            local = mgr.sp_to_local(sg.sp(rank))
            slots, _ = mgr.local_slots()
            cache = g.full.new_full((g.num_pages, g.page_size), -1.0)
            cache.view(-1)[slots.long()] = local
            caches[rank] = cache

        with _utils_env():
            sp.run(task)

        for rank in range(ranks):
            g.assert_local_blocks(rank, caches[rank])

    def _run_kvsp_ag_case(
        self,
        q_cumlens: list[int],
        kv_lens: list[int],
        ranks: int = 4,
        page_size: int = _KVSP_PAGE_SIZE,
    ) -> None:
        g = KVSPGolden(q_cumlens, kv_lens, page_size, ranks)
        cu = np.array(q_cumlens, dtype=np.int32)
        kl = np.array(kv_lens, dtype=np.int32)
        blk = g.blk_table.to(torch.int32)
        ag_out: dict[int, torch.Tensor] = {}
        sp = TaskDist(ranks=ranks)

        def ag_task():
            utils = importlib.import_module(MODULE)
            mgr = utils.KVSPMaganer(cu, kl, blk, page_size=page_size)
            rank = sp.thread_local.rank
            cache = g.local_blocks(rank).unsqueeze(-1)
            ag_cache, _, _ = mgr.ag_pages(cache)
            ag_out[rank] = ag_cache.clone()

        with _utils_env():
            sp.run(ag_task)

        for rank in range(ranks):
            g.assert_full_from_ag(ag_out[rank])
        for rank in range(1, ranks):
            assert torch.equal(ag_out[rank], ag_out[0])

    def _run_kvsp_case(
        self,
        q_cumlens: list[int],
        kv_lens: list[int],
        ranks: int = 4,
        page_size: int = _KVSP_PAGE_SIZE,
    ) -> None:
        self._run_kvsp_select_scatter_case(q_cumlens, kv_lens, ranks, page_size)
        self._run_kvsp_sp_to_local_scatter_case(q_cumlens, kv_lens, ranks, page_size)
        self._run_kvsp_ag_case(q_cumlens, kv_lens, ranks, page_size)

    def test_kvsp_single_exact(self):
        self._run_kvsp_case([0, 16], [16])

    def test_kvsp_padded(self):
        self._run_kvsp_case([0, 15], [15])

    def test_kvsp_multi_req(self):
        self._run_kvsp_case([0, 8, 20], [8, 12])

    def test_kvsp_prefix(self):
        self._run_kvsp_case([0, 64], [66])

    def test_kvsp_cycle_exact(self):
        self._run_kvsp_case([0, 128], [128])

    def test_kvsp_cycle_plus_1(self):
        self._run_kvsp_case([0, 129], [129])

    def test_kvsp_short_req(self):
        self._run_kvsp_case([0, 5], [5])

    def test_kvsp_many_req_varied_lens(self):
        cumlens = [0, 3, 10, 21, 38, 57, 80, 109, 140, 177, 218]
        kv = [3, 7, 11, 17, 19, 23, 29, 31, 37, 41]
        self._run_kvsp_case(cumlens, kv)


def _causal_conv_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    states: torch.Tensor,
    prefix: torch.Tensor,
    cumlens: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    assert x.dim() == 2 and w.dim() == 2 and states.dim() == 3
    K, D = w.shape
    bs = cumlens.numel() - 1
    assert states.shape[0] == bs and states.size(1) >= K and states.size(2) == D
    y = x if inplace else x.clone()
    for b in range(bs):
        s = int(cumlens[b].item())
        e = int(cumlens[b + 1].item())
        L = e - s
        assert L > 0
        xb = x[s:e]
        computed = int(prefix[b].item())
        hist = (
            xb.new_zeros(K - 1, D) if computed == 0
            else states[b, -(K - 1):].to(xb.dtype)
        )
        padded = torch.cat([hist, xb], dim=0)
        yb = sum(w[k] * padded[k:k + L] for k in range(K))
        reset_idx = min(max(K - 1 - computed, 0), L)
        if reset_idx:
            yb = yb.clone()
            yb[:reset_idx] = 0
        y[s:e] = xb + yb
    return y


@contextmanager
def _mock_conv_sp_ops():
    utils = importlib.import_module(MODULE)

    def select_dim0(x, i):
        return x[i]

    def save_states(cache, index, states):
        cache[index] = states

    def named_stream(_name):
        s = MagicMock()
        s.wait_stream = MagicMock()
        return s

    mock_fused = MagicMock(side_effect=lambda x, *_, **__: x.clone())

    with (
        _mock_torch_npu_stream(),
        patch.object(utils, "select_dim0", side_effect=select_dim0),
        patch.object(utils, "save_states", side_effect=save_states),
        patch.object(utils, "simple_conv", side_effect=_causal_conv_ref),
        patch.object(utils, "named_stream", side_effect=named_stream),
        patch(
            "torch.ops.custom.npu_ai_infra_fused_causal_conv1d",
            mock_fused,
            create=True,
        ),
    ):
        yield


def test_scheme_conv_sp_stages_metadata_with_non_blocking_copy():
    utils = importlib.import_module(MODULE)
    group = MagicMock(rank_in_group=0, world_size=1)
    pinned = []
    to_calls = []
    target_device = torch.device("npu")

    def fake_pin_memory(tensor):
        pinned.append(tensor)
        return tensor

    def tracked_to(tensor, *args, **kwargs):
        to_calls.append((args, kwargs))
        return tensor

    with (
        patch.object(torch.Tensor, "pin_memory", fake_pin_memory),
        patch.object(torch.Tensor, "to", tracked_to),
    ):
        metadata = utils.scheme_conv_sp(
            group,
            np.array([0, 4], dtype=np.int32),
            np.array([0], dtype=np.int32),
            like=MagicMock(device=target_device),
            block_size=4,
            state_len=3,
        )

    async_to_calls = [
        (args, kwargs) for args, kwargs in to_calls if kwargs.get("non_blocking")
    ]
    assert len(pinned) == 4
    assert len(async_to_calls) == 4
    assert all(args == (target_device,) for args, _ in async_to_calls)

    (send_idx, _, _), (prefix, cumlens, reorg_idx), _, _ = metadata
    assert send_idx.dtype == torch.int32
    assert prefix.dtype == torch.int32
    assert cumlens.dtype == torch.int32
    assert reorg_idx.dtype == torch.int32
