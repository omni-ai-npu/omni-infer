# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit / ST tests for MoME sequence-parallel helpers in utils.py
(select_dim0, simple_conv, save_states, scheme_conv_sp, conv_sp).
"""

from __future__ import annotations

import importlib
import queue
import threading
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

MODULE = "omni_npu.attention.backends.utils"

cfg_i32 = {"device": "cpu", "dtype": torch.int32}
cfg_bf16 = {"device": "cpu", "dtype": torch.bfloat16}


@contextmanager
def _mock_torch_npu_stream():
    mock_npu = MagicMock()
    mock_npu.current_stream.return_value = MagicMock()
    mock_npu.Stream.return_value = MagicMock()
    mock_npu.stream.side_effect = lambda _: nullcontext()
    with patch("torch.npu", mock_npu, create=True):
        yield


@contextmanager
def _mock_misc():
    with (
        patch("vllm.logger.init_logger", return_value=MagicMock()),
        patch(f"{MODULE}.current_platform", MagicMock(device_type="cpu"), create=True),
    ):
        yield


@contextmanager
def _utils_env():
    with _mock_torch_npu_stream(), _mock_misc():
        import omni_npu.attention.backends.utils as utils
        yield utils


class TaskDist:

    def __init__(self, ranks: int):
        self.ranks = ranks
        self.thread_local = threading.local()
        self.mailboxes = [queue.Queue() for _ in range(ranks)]
        self.errors = queue.Queue()

        class MockGroup:
            def __init__(self, rank: int, dist: TaskDist):
                self.rank_in_group = rank
                self.world_size = dist.ranks
                self.device_group = dist

            def all_gather(self, x: torch.Tensor, dim: int):
                assert dim == 0
                y = x.new_empty(x.size(0) * self.world_size, *x.shape[1:])
                self.device_group._ag_v(y, x, self.device_group)
                return y

        self.groups = [MockGroup(r, self) for r in range(self.ranks)]

    def _group(self):
        return self.groups[self.thread_local.rank]

    def _exchange(self, sends: list[torch.Tensor]) -> list[torch.Tensor]:
        rank = self.thread_local.rank
        assert len(sends) == self.ranks
        for dst, piece in enumerate(sends):
            self.mailboxes[dst].put((rank, piece))
        recvs = [None] * self.ranks
        for _ in range(self.ranks):
            src, piece = self.mailboxes[rank].get()
            recvs[src] = piece
        return recvs

    def _a2a_v(self,
        output: torch.Tensor,
        input: torch.Tensor,
        recv_split: list[int],
        send_split: list[int],
        group: TaskDist,
    ):
        assert len(recv_split) == self.ranks
        assert len(send_split) == self.ranks
        assert sum(recv_split) == output.size(0)
        assert sum(send_split) == input.size(0)
        assert group is self
        sends = [it.clone() for it in torch.split(input, send_split)]
        recvs = self._exchange(sends)
        assert len(recvs) == self.ranks
        for ref, recv in zip(torch.split(output, recv_split), recvs):
            ref.copy_(recv)

    def _ag_v(self,
        output: torch.Tensor,
        input: torch.Tensor,
        group: TaskDist,
    ):
        assert group is self
        assert output.size(0) == input.size(0) * self.ranks
        assert output.shape[1:] == input.shape[1:]
        assert output.numel() == input.numel() * self.ranks
        piece = input.clone().flatten()
        sends = [piece] * self.ranks
        recvs = self._exchange(sends)
        assert len(recvs) == self.ranks
        output.flatten().copy_(torch.cat(recvs, dim=0))

    def run(self, task):
        for it in self.mailboxes:
            while not it.empty():
                it.get_nowait()
        while not self.errors.empty():
            self.errors.get_nowait()

        def worker(rank):
            try:
                self.thread_local.rank = rank
                task()
            except Exception as e:
                self.errors.put(e)

        with (
            patch(f"{MODULE}.get_tp_group", side_effect=self._group),
            patch("torch.distributed.all_to_all_single", side_effect=self._a2a_v),
            patch("torch.distributed.all_gather_into_tensor", side_effect=self._ag_v),
        ):
            threads = [
                threading.Thread(target=worker, args=(r,))
                for r in range(self.ranks)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert self.errors.empty(), self.errors.get()


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


@pytest.mark.unit
class TestTaskDist:
    def test_all_to_all(self):
        ranks = 4
        sp = TaskDist(ranks=ranks)
        results = {}

        def task():
            rank = sp.thread_local.rank
            inp = torch.tensor(
                [[rank * 10 + d] for d in range(ranks)], **cfg_i32,
            )
            out = torch.zeros(ranks, 1, **cfg_i32)
            splits = [1] * ranks
            torch.distributed.all_to_all_single(
                out, inp, splits, splits, group=sp,
            )
            results[rank] = out.clone()

        sp.run(task)
        for dst in range(ranks):
            assert results[dst].tolist() == [[s * 10 + dst] for s in range(ranks)]

    def test_all_gather(self):
        ranks = 4
        sp = TaskDist(ranks=ranks)
        results = {}

        def task():
            rank = sp.thread_local.rank
            group = sp.groups[rank]
            local = torch.tensor([[rank * 10 + 1], [rank * 10 + 2]], **cfg_bf16)
            gathered = group.all_gather(local, dim=0)
            results[rank] = gathered.clone()

        sp.run(task)
        expected = torch.tensor(
            [[r * 10 + 1, r * 10 + 2] for r in range(ranks)], **cfg_bf16,
        ).view(-1, 1)
        for r in range(ranks):
            assert torch.equal(results[r], expected)

    def test_get_tp_group_per_thread(self):
        with _utils_env():
            ranks = 4
            sp = TaskDist(ranks=ranks)
            results = {}

            def task():
                rank = sp.thread_local.rank
                utils = importlib.import_module(MODULE)
                got = utils.get_tp_group()
                assert got is sp.groups[rank]
                assert got.rank_in_group == rank
                results[rank] = got.rank_in_group

            sp.run(task)
            assert results == {r: r for r in range(ranks)}


@pytest.mark.unit
class TestSelectDim0:

    def test_basic_index(self):
        with _utils_env() as utils:
            x = torch.arange(12, **cfg_bf16).view(4, 3)
            i = torch.tensor([0, 2], **cfg_i32)
            y = utils.select_dim0(x, i)
            assert torch.equal(y, x[[0, 2]])

    def test_3d_tensor(self):
        with _utils_env() as utils:
            x = torch.arange(24, **cfg_bf16).view(4, 2, 3)
            i = torch.tensor([1, 3], **cfg_i32)
            y = utils.select_dim0(x, i)
            assert y.shape == (2, 2, 3)
            assert torch.equal(y, x[[1, 3]].contiguous())

    def test_non_contiguous_raises(self):
        with _utils_env() as utils:
            x = torch.arange(12, **cfg_bf16).view(4, 3).t()
            i = torch.tensor([0], **cfg_i32)
            with pytest.raises(AssertionError):
                utils.select_dim0(x, i)


@pytest.mark.unit
class TestSimpleConv:

    def test_valid_kwargs(self):
        with _utils_env() as utils:
            T, D, k, bs = 8, 16, 3, 2
            x = torch.randn(T, D, **cfg_bf16)
            w = torch.randn(k, D, **cfg_bf16)
            states = torch.zeros(bs, k, D, **cfg_bf16)
            prefix = torch.zeros(bs, **cfg_i32)
            cumlens = torch.tensor([0, 4, 8], **cfg_i32)
            mock_op = MagicMock(return_value=x.clone())
            with patch(
                "torch.ops.custom.npu_ai_infra_fused_causal_conv1d",
                mock_op, create=True,
            ):
                utils.simple_conv(x, w, states, prefix, cumlens, inplace=False)
            mock_op.assert_called_once()
            args, kwargs = mock_op.call_args
            assert args[0] is x and args[1] is w and args[2] is states
            qsl = kwargs["query_start_loc"]
            computed = kwargs["num_computed_tokens"]
            assert qsl is cumlens and computed is prefix
            assert int(qsl[0]) == 0 and int(qsl[-1]) == T
            assert torch.all(qsl[1:] > qsl[:-1])
            assert computed.shape == (bs,)
            assert args[1].shape == (3, D)
            assert D % 16 == 0
            assert kwargs["residual_connection"] == 1
            assert kwargs["inplace"] is False
            assert kwargs["block_size"] >= 2

    def test_shape_mismatch_raises(self):
        with _utils_env() as utils:
            x = torch.randn(4, 16, **cfg_bf16)
            w = torch.randn(3, 16, **cfg_bf16)
            states = torch.randn(1, 2, 16, **cfg_bf16)
            prefix = torch.zeros(1, **cfg_i32)
            cumlens = torch.tensor([0, 4], **cfg_i32)
            with pytest.raises(AssertionError):
                utils.simple_conv(x, w, states, prefix, cumlens)


@pytest.mark.unit
class TestSaveStates:

    def test_valid_kwargs(self):
        with _utils_env() as utils:
            B, m, D = 2, 3, 16
            cache = torch.zeros(8, m, D, **cfg_bf16)
            index = torch.tensor([1, 5], **cfg_i32)
            states = torch.randn(B, m, D, **cfg_bf16)
            mock_op = MagicMock(return_value=None)
            with patch(
                "torch.ops.custom.npu_ai_infra_fused_causal_conv1d",
                mock_op, create=True,
            ):
                utils.save_states(cache, index, states)
            mock_op.assert_called_once()
            args, kwargs = mock_op.call_args
            x_flat, weight, conv_states = args
            assert x_flat.shape == (B * m, D)
            assert weight.shape == (3, D)
            assert D % 16 == 0
            assert conv_states is cache
            qsl = kwargs["query_start_loc"]
            computed = kwargs["num_computed_tokens"]
            assert qsl.numel() == B + 1 and int(qsl[0]) == 0
            assert torch.all(qsl[1:] > qsl[:-1])
            assert int(qsl[-1]) == B * m
            assert computed.shape == (B,)
            assert kwargs["cache_indices"] is index
            assert kwargs["cache_indices"].dim() == 1
            assert kwargs["residual_connection"] == 0
            assert kwargs["block_size"] == m and m >= 2
            assert kwargs["inplace"] is True


@pytest.mark.unit
class TestMomeSpConvST:

    @staticmethod
    def _partition(seqs, offset, unit):
        for i, seq in enumerate(seqs):
            pos = int(offset[i])
            end = int(pos + seq)
            while pos < end:
                idx = pos // unit
                nxt = min(end, (idx + 1) * unit)
                yield i, pos, nxt, idx, bool(nxt == end)
                pos = nxt

    @staticmethod
    def _enum_save_refs(cumlens, computed, block_size, m):
        # 与 scheme_conv_sp 的 save 队列同序：每块 refer(req, tail, m)
        outs = []
        for req, _start, end, _blk, _ in TestMomeSpConvST._partition(
            np.diff(cumlens), computed, block_size,
        ):
            base = int(cumlens[req])
            tail = int(cumlens[req] - computed[req]) + end
            refs = []
            for j in range(m):
                p = tail - m + j
                if p >= base:
                    refs.append(p)
                else:
                    assert p + m >= base
                    refs.append((req, p - base))
            outs.append(refs)
        return outs

    @staticmethod
    def _materialize_refs(refs, Xp, S0):
        m = len(refs)
        out = Xp.new_empty(m, Xp.size(1))
        for j, r in enumerate(refs):
            if isinstance(r, int):
                out[j] = Xp[r]
            else:
                req, off = r
                out[j] = S0[req, off + m]
        return out

    @staticmethod
    def _build_cache_gold(cache, save_idx, save_refs, Xp, S0):
        gold = cache.clone()
        for i, refs in enumerate(save_refs):
            gold[int(save_idx[i])] = TestMomeSpConvST._materialize_refs(
                refs, Xp, S0,
            )
        return gold

    @staticmethod
    def run_sp_case(
        cumlens,
        *,
        computed=None,
        X=None,
        w=None,
        S0=None,
        ranks=4,
        D=16,
        K=3,
        m=5,
        block_size=32,
        inplace=False,
    ):
        cumlens = np.asarray(cumlens, dtype=np.int64)
        bs = cumlens.size - 1
        T = int(cumlens[-1])
        if computed is None:
            computed = np.zeros(bs, dtype=np.int64)
        else:
            computed = np.asarray(computed, dtype=np.int64)
        assert computed.size == bs

        if X is None:
            X = torch.randn(T, D, **cfg_bf16)
        if w is None:
            w = torch.randn(K, D, **cfg_bf16)
        if S0 is None:
            S0 = (
                torch.randn(bs, m, D, **cfg_bf16) if np.any(computed > 0)
                else torch.zeros(bs, m, D, **cfg_bf16)
            )

        sp_len = -(-T // ranks)  # cdiv
        pad_to = sp_len * ranks
        Xp = X.new_zeros(pad_to, D)
        Xp[:T] = X
        xs = [Xp[r * sp_len:(r + 1) * sp_len].clone() for r in range(ranks)]

        save_refs = TestMomeSpConvST._enum_save_refs(
            cumlens, computed, block_size, m,
        )
        n_save = len(save_refs)
        cache = torch.zeros(bs + n_save, m, D, **cfg_bf16)
        cache[:bs] = S0
        init_idx = torch.arange(bs, **cfg_i32)
        save_idx = torch.arange(bs, bs + n_save, **cfg_i32)
        cache_before = cache.clone()
        cache_gold = TestMomeSpConvST._build_cache_gold(
            cache, save_idx, save_refs, Xp, S0,
        )

        y_gold = _causal_conv_ref(
            X, w, S0,
            torch.as_tensor(computed, **cfg_i32),
            torch.as_tensor(cumlens, **cfg_i32),
            inplace=False,
        )

        like = torch.zeros(1, **cfg_i32)
        results = {}
        sp = TaskDist(ranks=ranks)

        with _utils_env(), _mock_conv_sp_ops():

            def task():
                rank = sp.thread_local.rank
                utils = importlib.import_module(MODULE)
                group = utils.get_tp_group()
                meta = utils.scheme_conv_sp(
                    group, cumlens, computed, like=like,
                    block_size=block_size, state_len=m, kernel_size=K,
                )
                _, _, _, save_range = meta
                n = sum(s.stop - s.start for s in save_range)
                assert n == n_save and list(save_idx.shape) == [n_save]
                y = utils.conv_sp(
                    xs[rank], w, cache, init_idx, save_idx, meta, inplace,
                )
                results[rank] = y.clone()

            sp.run(task)
        y_sp = torch.cat([results[r] for r in range(ranks)], dim=0)[:T]
        assert torch.allclose(y_sp, y_gold, atol=1e-2, rtol=1e-2)
        assert torch.allclose(
            cache[save_idx], cache_gold[save_idx], atol=1e-2, rtol=1e-2,
        )
        assert torch.equal(cache[init_idx], cache_before[init_idx])
        return y_sp, cache

    def test_sp_exact(self):
        self.run_sp_case([0, 64])

    def test_sp_len_minus_1(self):
        self.run_sp_case([0, 63])

    def test_sp_len_plus_1(self):
        self.run_sp_case([0, 65])

    def test_block_exact(self):
        self.run_sp_case([0, 32])

    def test_block_minus_1(self):
        self.run_sp_case([0, 31])

    def test_block_plus_1(self):
        self.run_sp_case([0, 33])

    def test_multi_req(self):
        self.run_sp_case([0, 32, 64])

    def test_uneven_req_boundary(self):
        self.run_sp_case([0, 5, 32])

    def test_prefix_computed(self):
        self.run_sp_case([0, 64], computed=[2])

    def test_inplace_path(self):
        self.run_sp_case([0, 64], inplace=True)

    def test_many_req_varied_lens(self):
        # 10 req，长度 3/7/11/17/19/23/29/31/37/41 → T=218，sp_len=55，pad=2
        cumlens = [0, 3, 10, 21, 38, 57, 80, 109, 140, 177, 218]
        self.run_sp_case(cumlens)

    def test_scheme_conv_sp_zero_len_raises(self):
        with _utils_env() as utils:
            like = torch.zeros(1, **cfg_i32)
            group = TaskDist(ranks=4).groups[0]
            with pytest.raises(AssertionError):
                utils.scheme_conv_sp(
                    group, np.array([0, 0], dtype=np.int64), like=like,
                    block_size=32, state_len=5, kernel_size=3,
                )
