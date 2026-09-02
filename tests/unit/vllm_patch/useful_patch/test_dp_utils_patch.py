# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CUDAGraphMode

from omni_npu.vllm_patches.usefull_patch.common import patch_dp_utils


def _make_parallel_config(
    dp_size=2,
    dp_rank=0,
    tp_size=1,
    enable_expert_parallel=False,
):
    return SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        tensor_parallel_size=tp_size,
        num_ubatches=2,
        enable_expert_parallel=enable_expert_parallel,
    )


def _patch_dp_sync_group(monkeypatch, captured, ranks, use_local_synchronization=False):
    """Install the common distributed-group mocks used by DP sync tests."""
    def fake_new_group(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    dp_group = SimpleNamespace(
        ranks=ranks,
        device_group=MagicMock(),
        device=torch.device("cpu"),
        unique_name="dp",
    )
    monkeypatch.setattr(patch_dp_utils, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr("torch.npu.Stream", MagicMock)
    monkeypatch.setattr("torch.Event", MagicMock)
    monkeypatch.setattr(patch_dp_utils.dist, "new_group", fake_new_group)
    monkeypatch.setattr(patch_dp_utils.dist, "get_backend", lambda g: "hccl")
    monkeypatch.setattr(
        "torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options",
        lambda: SimpleNamespace(hccl_config={}),
    )
    if use_local_synchronization:
        monkeypatch.setattr(
            patch_dp_utils.GroupCoordinator,
            "use_local_synchronization",
            True,
            raising=False,
        )


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset lazy-initialized globals before each test."""
    patch_dp_utils._dp_sync_copy_stream = None
    patch_dp_utils._dp_sync_event = None
    patch_dp_utils._dp_sync_device_group = None
    patch_dp_utils._dp_sync_device = None
    patch_dp_utils._dp_sync_group_key = None
    patch_dp_utils._aicpu_dp_sync_init_failed = False
    yield
    patch_dp_utils._dp_sync_copy_stream = None
    patch_dp_utils._dp_sync_event = None
    patch_dp_utils._dp_sync_device_group = None
    patch_dp_utils._dp_sync_device = None
    patch_dp_utils._dp_sync_group_key = None
    patch_dp_utils._aicpu_dp_sync_init_failed = False


@pytest.mark.unit
def test_run_ar_falls_back_when_aicpu_dp_sync_disabled(monkeypatch):
    """When enable_aicpu_dp_sync is False, _run_ar delegates to _original_run_ar."""
    monkeypatch.setattr(
        patch_dp_utils.model_extra_config.parall_config,
        "enable_aicpu_dp_sync",
        False,
        raising=False,
    )

    sentinel = torch.tensor([42])
    monkeypatch.setattr(
        patch_dp_utils,
        "_original_run_ar",
        lambda *args, **kwargs: sentinel,
    )

    result = patch_dp_utils._run_ar(
        should_ubatch=False,
        orig_num_tokens_per_ubatch=10,
        padded_num_tokens_per_ubatch=16,
        cudagraph_mode=0,
        parallel_config=_make_parallel_config(),
    )
    assert torch.equal(result, sentinel)


@pytest.mark.unit
def test_run_ar_populates_cpu_tensor_correctly(monkeypatch):
    """Verify cpu_tensor is populated with the correct values at the correct indices."""
    monkeypatch.setattr(
        patch_dp_utils.model_extra_config.parall_config,
        "enable_aicpu_dp_sync",
        True,
        raising=False,
    )

    dp_size = 4
    dp_rank = 2
    orig_tokens = 100
    padded_tokens = 128
    cudagraph_mode = 2

    captured = {}

    def fake_get_primitives():
        mock_event = MagicMock()
        mock_device_group = MagicMock()
        mock_device = torch.device("cpu")
        return MagicMock(), mock_event, mock_device_group, mock_device

    def fake_npu_stream(s):
        return MagicMock(__enter__=MagicMock(return_value=None),
                         __exit__=MagicMock(return_value=False))

    monkeypatch.setattr(patch_dp_utils, "_get_dp_sync_primitives", fake_get_primitives)
    monkeypatch.setattr("torch.npu.stream", fake_npu_stream)

    def fake_all_reduce(tensor, group=None):
        captured["tensor_before_reduce"] = tensor.clone()

    monkeypatch.setattr(patch_dp_utils.dist, "all_reduce", fake_all_reduce)

    result = patch_dp_utils._run_ar(
        should_ubatch=True,
        orig_num_tokens_per_ubatch=orig_tokens,
        padded_num_tokens_per_ubatch=padded_tokens,
        cudagraph_mode=cudagraph_mode,
        parallel_config=_make_parallel_config(dp_size=dp_size, dp_rank=dp_rank),
    )

    assert result.shape == (4, dp_size)
    assert result[0][dp_rank].item() == orig_tokens
    assert result[1][dp_rank].item() == padded_tokens
    assert result[2][dp_rank].item() == 1  # should_ubatch=True
    assert result[3][dp_rank].item() == cudagraph_mode

    for row in range(4):
        for col in range(dp_size):
            if col != dp_rank:
                assert result[row][col].item() == 0


@pytest.mark.unit
def test_run_ar_cpu_tensor_is_pinned(monkeypatch):
    """Verify that cpu_tensor uses pinned memory."""
    monkeypatch.setattr(
        patch_dp_utils.model_extra_config.parall_config,
        "enable_aicpu_dp_sync",
        True,
        raising=False,
    )

    created_tensors = []
    original_zeros = torch.zeros

    def tracking_zeros(*args, **kwargs):
        t = original_zeros(*args, **kwargs)
        if kwargs.get("pin_memory"):
            created_tensors.append(t)
        return t

    monkeypatch.setattr(torch, "zeros", tracking_zeros)

    def fake_get_primitives():
        mock_stream = MagicMock()
        mock_event = MagicMock()
        return mock_stream, mock_event, MagicMock(), torch.device("cpu")

    monkeypatch.setattr(patch_dp_utils, "_get_dp_sync_primitives", fake_get_primitives)
    monkeypatch.setattr("torch.npu.stream", lambda s: MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False),
    ))
    monkeypatch.setattr(patch_dp_utils.dist, "all_reduce", lambda *a, **kw: None)

    patch_dp_utils._run_ar(
        should_ubatch=False,
        orig_num_tokens_per_ubatch=10,
        padded_num_tokens_per_ubatch=10,
        cudagraph_mode=0,
        parallel_config=_make_parallel_config(),
    )

    assert len(created_tensors) == 1, "Expected one pinned tensor to be created"


@pytest.mark.unit
def test_run_ar_signature_matches_upstream():
    """The active usefull_patch _run_ar must match the upstream signature."""
    import inspect

    upstream_sig = inspect.signature(patch_dp_utils._original_run_ar)
    patched_sig = inspect.signature(patch_dp_utils._run_ar)
    assert list(upstream_sig.parameters.keys()) == list(patched_sig.parameters.keys())


@pytest.mark.unit
def test_get_dp_sync_primitives_is_lazy(monkeypatch):
    """Primitives are created once and reused on subsequent calls."""
    mock_stream = MagicMock()
    mock_event = MagicMock()
    mock_group = MagicMock()
    mock_device = torch.device("cpu")

    call_count = 0

    def fake_stream_ctor():
        nonlocal call_count
        call_count += 1
        return mock_stream

    dp_group = SimpleNamespace(
        ranks=[0, 1],
        device_group=MagicMock(),
        device=mock_device,
        unique_name="dp",
    )
    monkeypatch.setattr(patch_dp_utils, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr("torch.npu.Stream", fake_stream_ctor)
    monkeypatch.setattr("torch.Event", lambda: mock_event)
    monkeypatch.setattr(patch_dp_utils.dist, "new_group", lambda *a, **kw: mock_group)
    monkeypatch.setattr(patch_dp_utils.dist, "get_backend", lambda g: "hccl")
    monkeypatch.setattr(
        "torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options",
        lambda: SimpleNamespace(hccl_config={}),
    )

    result1 = patch_dp_utils._get_dp_sync_primitives()
    result2 = patch_dp_utils._get_dp_sync_primitives()

    assert call_count == 1, "Stream should be created only once"
    assert result1[0] is result2[0]
    assert result1[1] is result2[1]
    assert result1[2] is result2[2]
    assert result1[3] is result2[3]


@pytest.mark.unit
def test_get_dp_sync_primitives_creates_aicpu_group(monkeypatch):
    """Device group should be created with AICPU expansion mode."""
    captured_options = {}
    _patch_dp_sync_group(monkeypatch, captured_options, [0, 1, 2, 3])

    patch_dp_utils._get_dp_sync_primitives()

    pg_options = captured_options.get("pg_options")
    assert pg_options is not None
    assert pg_options.hccl_config["hccl_op_expansion_mode"] == 2


@pytest.mark.unit
def test_run_ar_calls_all_reduce_with_dedicated_group(monkeypatch):
    """all_reduce should use the dedicated AICPU device group, not the default DP group."""
    monkeypatch.setattr(
        patch_dp_utils.model_extra_config.parall_config,
        "enable_aicpu_dp_sync",
        True,
        raising=False,
    )

    dedicated_group = MagicMock(name="dedicated_aicpu_group")
    captured_groups = []

    def fake_get_primitives():
        return (
            MagicMock(),
            MagicMock(),
            dedicated_group,
            torch.device("cpu"),
        )

    def fake_all_reduce(tensor, group=None):
        captured_groups.append(group)

    monkeypatch.setattr(patch_dp_utils, "_get_dp_sync_primitives", fake_get_primitives)
    monkeypatch.setattr("torch.npu.stream", lambda s: MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False),
    ))
    monkeypatch.setattr(patch_dp_utils.dist, "all_reduce", fake_all_reduce)

    patch_dp_utils._run_ar(
        should_ubatch=False,
        orig_num_tokens_per_ubatch=10,
        padded_num_tokens_per_ubatch=16,
        cudagraph_mode=0,
        parallel_config=_make_parallel_config(),
    )

    assert len(captured_groups) == 1
    assert captured_groups[0] is dedicated_group


@pytest.mark.unit
def test_synchronize_dp_ranks_preserves_synced_dp_padding(monkeypatch):
    """A synced FULL cudagraph mode preserves DP padding in usefull_patch."""
    full_cudagraph_mode = CUDAGraphMode.FULL.value
    monkeypatch.setattr(
        patch_dp_utils,
        "_run_ar",
        lambda **kwargs: torch.tensor([
            [1, 128],
            [4, 128],
            [0, 0],
            [full_cudagraph_mode, full_cudagraph_mode],
        ], dtype=torch.int32),
    )

    should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
        patch_dp_utils._synchronize_dp_ranks(
            num_tokens_unpadded=1,
            num_tokens_padded=4,
            should_attempt_ubatching=False,
            cudagraph_mode=full_cudagraph_mode,
            parallel_config=_make_parallel_config(dp_size=2, tp_size=2),
        )
    )

    assert should_ubatch is False
    assert synced_cudagraph_mode == full_cudagraph_mode
    assert num_tokens_across_dp.tolist() == [128, 128]


@pytest.mark.unit
def test_patch_registers_dp_sync_functions():
    """The active patch should register its DP synchronization override."""
    patch_cls = patch_dp_utils.DpUtilsEagerEpPadPatch
    assert patch_cls._attr_names_to_apply == ["_synchronize_dp_ranks"]
    assert patch_cls._synchronize_dp_ranks is patch_dp_utils._synchronize_dp_ranks


@pytest.mark.unit
def test_get_dp_sync_primitives_follows_local_synchronization(monkeypatch):
    """new_group must follow GroupCoordinator.use_local_synchronization for local rendezvous."""
    captured = {}
    _patch_dp_sync_group(monkeypatch, captured, [0, 1], use_local_synchronization=True)

    patch_dp_utils._get_dp_sync_primitives()

    assert captured.get("use_local_synchronization") is True
