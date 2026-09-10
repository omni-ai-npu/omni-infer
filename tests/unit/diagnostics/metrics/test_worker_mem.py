# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.diagnostics.metrics.worker_mem (采样，走 stats 通道，无 prometheus)."""

import sys
import types

import pytest

pytest.importorskip("vllm.logger")

from omni_npu.diagnostics.metrics import worker_mem


@pytest.fixture
def fake_torch(monkeypatch):
    m = types.ModuleType("torch")
    m.npu = types.SimpleNamespace(
        memory_allocated=lambda: 111,
        memory_reserved=lambda: 222,
    )
    monkeypatch.setitem(sys.modules, "torch", m)
    monkeypatch.setattr(worker_mem, "_step", 0)
    yield m


def test_throttle_then_sample(fake_torch, monkeypatch):
    monkeypatch.setattr(worker_mem, "SAMPLE_EVERY_N_STEPS", 3)
    # 前两次节流，返回 None
    assert worker_mem.maybe_sample(5) is None
    assert worker_mem.maybe_sample(5) is None
    # 第三次采到，key 为 rank，值为 alloc/reserved
    snap = worker_mem.maybe_sample(5)
    assert snap == {"5": {"alloc": 111.0, "reserved": 222.0}}


def test_sample_failure_isolated(monkeypatch):
    monkeypatch.setattr(worker_mem, "SAMPLE_EVERY_N_STEPS", 1)
    monkeypatch.setattr(worker_mem, "_step", 0)
    monkeypatch.setitem(sys.modules, "torch", None)  # torch 坏了
    assert worker_mem.maybe_sample(0) is None  # 不抛，返回 None


def test_per_rank_key(fake_torch, monkeypatch):
    monkeypatch.setattr(worker_mem, "SAMPLE_EVERY_N_STEPS", 1)
    monkeypatch.setattr(worker_mem, "_step", 0)
    assert set(worker_mem.maybe_sample(3).keys()) == {"3"}
    assert set(worker_mem.maybe_sample(7).keys()) == {"7"}
