# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``initialize_local_comm_group_list``.

Covers the per-node rank slicing and the re-initialization guard. All
distributed primitives are mocked; no NPU or process group is created.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omni_npu.v1.distributed import parallel_state_ext as mod


pytestmark = pytest.mark.unit


def _patch_world(monkeypatch, ranks, local_size, init_group=None):
    """Fake a `world_size`-rank world spread over nodes of `local_size` dies."""
    monkeypatch.setattr(mod, "_LOCAL_COMM_LIST", None, raising=False)
    monkeypatch.setattr(mod.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        mod,
        "get_world_group",
        lambda: SimpleNamespace(ranks=list(ranks), local_rank=0, device_group="dg"),
    )
    monkeypatch.setattr(mod, "get_npu_device_count", lambda: local_size)
    monkeypatch.setattr(
        mod, "init_model_parallel_group", init_group or MagicMock(return_value="group")
    )


def test_local_comm_group_list_slices_ranks_per_node(monkeypatch):
    """Each node gets a contiguous rank slice of `local_size`."""
    calls = []

    def _init_group(group_ranks, local_rank, backend, **kwargs):
        calls.append(group_ranks)
        return f"group{len(calls)}"

    _patch_world(monkeypatch, ranks=range(4), local_size=2, init_group=_init_group)

    mod.initialize_local_comm_group_list("hccl")

    # world_size 4 / local_size 2 => 2 nodes, ranks split [0,1] and [2,3].
    assert calls[0] == [[0, 1], [2, 3]]
    # One group per server for each of the two nodes, plus the fixed extras.
    num_extra = getattr(mod, "_NUM_COMM_GROUP")
    assert len(calls) == 2 + num_extra
    assert len(getattr(mod, "_LOCAL_COMM_LIST")) == 2 + num_extra


def test_local_comm_group_list_handles_single_node(monkeypatch):
    """A single node yields one group of all ranks."""
    calls = []

    def _init_group(group_ranks, local_rank, backend, **kwargs):
        calls.append(group_ranks)
        return "group"

    _patch_world(monkeypatch, ranks=range(8), local_size=8, init_group=_init_group)

    mod.initialize_local_comm_group_list("hccl")

    assert calls[0] == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert len(getattr(mod, "_LOCAL_COMM_LIST")) == 1 + getattr(mod, "_NUM_COMM_GROUP")


def test_local_comm_group_list_rejects_second_initialization(monkeypatch):
    """Re-initializing without teardown must raise instead of leaking groups."""
    _patch_world(monkeypatch, ranks=range(4), local_size=2)
    monkeypatch.setattr(mod, "_LOCAL_COMM_LIST", ["already"], raising=False)

    with pytest.raises(RuntimeError, match="_LOCAL_COMM_LIST must be None"):
        mod.initialize_local_comm_group_list("hccl")


def test_local_comm_group_list_requires_initialized_distributed(monkeypatch):
    """The guard fires before any group is built."""
    _patch_world(monkeypatch, ranks=range(4), local_size=2)
    monkeypatch.setattr(mod.torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
        mod.initialize_local_comm_group_list("hccl")
