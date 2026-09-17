# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from omni_npu.vllm_patches.patches.common import patch_parallel_state
from omni_npu.vllm_patches.patches.common.patch_parallel_state import (
    GroupCoordinatorPatch,
)


def _make_platform(logical_to_visible, *, out_of_tree=True):
    return SimpleNamespace(
        device_name="npu",
        is_cuda_alike=lambda: False,
        is_xpu=lambda: False,
        is_out_of_tree=lambda: out_of_tree,
        is_tpu=lambda: False,
        is_cpu=lambda: False,
        use_custom_op_collectives=lambda: False,
        logical_device_id_to_visible_device_id=logical_to_visible,
    )


def _build_coordinator(monkeypatch, platform, *, rank, local_rank):
    monkeypatch.setattr(patch_parallel_state, "_get_unique_name", lambda name: name)
    monkeypatch.setattr(patch_parallel_state, "_register_group", lambda group: None)
    monkeypatch.setattr(patch_parallel_state.parallel_state, "_WORLD", None)
    monkeypatch.setattr(
        patch_parallel_state,
        "envs",
        SimpleNamespace(VLLM_DISTRIBUTED_USE_SPLIT_GROUP=False),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        torch.distributed, "new_group", lambda *args, **kwargs: MagicMock()
    )
    # Patch the module object the code under test imports from. Other suites
    # replace vllm / vllm.platforms in sys.modules, so resolving the dotted
    # string through the parent package attribute is not reliable.
    platforms_module = importlib.import_module("vllm.platforms")
    monkeypatch.setattr(platforms_module, "current_platform", platform, raising=False)

    coordinator = GroupCoordinatorPatch.__new__(GroupCoordinatorPatch)
    GroupCoordinatorPatch.__init__(
        coordinator,
        group_ranks=[[0, 1]],
        local_rank=local_rank,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="dp",
    )
    return coordinator


@pytest.mark.parametrize(
    ("rank", "visible_device_index"),
    [
        (0, 0),
        # DP1 under vLLM 0.25 per-DP device sharding: shard-local id 0 maps to
        # the second visible device, not npu:0.
        (1, 1),
    ],
)
def test_group_coordinator_translates_out_of_tree_device(
    monkeypatch, rank, visible_device_index
):
    resolved = []

    def logical_to_visible(device_id):
        resolved.append(device_id)
        return visible_device_index

    coordinator = _build_coordinator(
        monkeypatch, _make_platform(logical_to_visible), rank=rank, local_rank=0
    )

    assert resolved == [0]
    assert coordinator.device_index == 0
    assert coordinator.device == torch.device(f"npu:{visible_device_index}")
    assert coordinator.rank_in_group == rank
    assert coordinator.world_size == 2


def test_group_coordinator_falls_back_to_cpu_device(monkeypatch):
    logical_to_visible = MagicMock()

    coordinator = _build_coordinator(
        monkeypatch,
        _make_platform(logical_to_visible, out_of_tree=False),
        rank=0,
        local_rank=0,
    )

    logical_to_visible.assert_not_called()
    assert coordinator.device == torch.device("cpu")
