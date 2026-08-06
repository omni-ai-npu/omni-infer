# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from omni.vllm_patches.patches.common import patch_dp_utils


def _make_parallel_config(dp_size=2, dp_rank=0, tp_size=1):
    return SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        tensor_parallel_size=tp_size,
        num_ubatches=2,
    )


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset lazy-initialized globals before each test."""
    patch_dp_utils._dp_sync_copy_stream = None
    patch_dp_utils._dp_sync_event = None
    patch_dp_utils._dp_sync_device_group = None
    patch_dp_utils._dp_sync_device = None
    yield
    patch_dp_utils._dp_sync_copy_stream = None
    patch_dp_utils._dp_sync_event = None
    patch_dp_utils._dp_sync_device_group = None
    patch_dp_utils._dp_sync_device = None


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
        should_dp_pad=True,
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
        should_dp_pad=False,
        orig_num_tokens_per_ubatch=orig_tokens,
        padded_num_tokens_per_ubatch=padded_tokens,
        cudagraph_mode=cudagraph_mode,
        parallel_config=_make_parallel_config(dp_size=dp_size, dp_rank=dp_rank),
    )

    assert result.shape == (5, dp_size)
    assert result[0][dp_rank].item() == orig_tokens
    assert result[1][dp_rank].item() == padded_tokens
    assert result[2][dp_rank].item() == 1  # should_ubatch=True
    assert result[3][dp_rank].item() == 0  # should_dp_pad=False
    assert result[4][dp_rank].item() == cudagraph_mode

    for row in range(5):
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
        should_dp_pad=False,
        orig_num_tokens_per_ubatch=10,
        padded_num_tokens_per_ubatch=10,
        cudagraph_mode=0,
        parallel_config=_make_parallel_config(),
    )

    assert len(created_tensors) == 1, "Expected one pinned tensor to be created"


@pytest.mark.unit
def test_run_ar_signature_matches_upstream():
    """Patched _run_ar must have the same signature as upstream to be a drop-in replacement."""
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

    def fake_new_group(*args, **kwargs):
        captured_options.update(kwargs)
        return MagicMock()

    dp_group = SimpleNamespace(
        ranks=[0, 1, 2, 3],
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
        should_dp_pad=True,
        orig_num_tokens_per_ubatch=10,
        padded_num_tokens_per_ubatch=16,
        cudagraph_mode=0,
        parallel_config=_make_parallel_config(),
    )

    assert len(captured_groups) == 1
    assert captured_groups[0] is dedicated_group


@pytest.mark.unit
def test_synchronize_dp_ranks_preserves_synced_dp_padding(monkeypatch):
    """Original synced DP padding is preserved outside the eager all2all path."""
    full_cudagraph_mode = patch_dp_utils.CUDAGraphMode.FULL.value
    monkeypatch.setattr(
        patch_dp_utils,
        "_run_ar",
        lambda **kwargs: torch.tensor([
            [1, 128],
            [4, 128],
            [0, 0],
            [1, 1],
            [full_cudagraph_mode, full_cudagraph_mode],
        ], dtype=torch.int32),
    )
    monkeypatch.setattr(
        patch_dp_utils.torch_npu.npu,
        "get_device_name",
        lambda index=0: "Ascend910B",
    )

    should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
        patch_dp_utils._synchronize_dp_ranks(
            num_tokens_unpadded=1,
            num_tokens_padded=4,
            should_attempt_ubatching=False,
            should_attempt_dp_padding=True,
            cudagraph_mode=full_cudagraph_mode,
            parallel_config=_make_parallel_config(dp_size=2, tp_size=2),
        )
    )

    assert should_ubatch is False
    assert synced_cudagraph_mode == full_cudagraph_mode
    assert num_tokens_across_dp.tolist() == [128, 128]


@pytest.mark.unit
def test_synchronize_dp_ranks_skips_padding_for_a2_all2all(monkeypatch):
    """A2 TP+DP all2all eager path keeps each rank's local padded token count."""
    monkeypatch.setattr(
        patch_dp_utils,
        "_run_ar",
        lambda **kwargs: torch.tensor([
            [1, 128],
            [4, 128],
            [0, 0],
            [1, 1],
            [0, 0],
        ], dtype=torch.int32),
    )
    monkeypatch.setattr(
        patch_dp_utils.torch_npu.npu,
        "get_device_name",
        lambda index=0: "Ascend910B",
    )

    should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
        patch_dp_utils._synchronize_dp_ranks(
            num_tokens_unpadded=1,
            num_tokens_padded=4,
            should_attempt_ubatching=False,
            should_attempt_dp_padding=True,
            cudagraph_mode=0,
            parallel_config=_make_parallel_config(dp_size=2, tp_size=2),
        )
    )

    assert should_ubatch is False
    assert synced_cudagraph_mode == 0
    assert num_tokens_across_dp.tolist() == [4, 128]


@pytest.mark.unit
def test_patch_registers_dp_sync_functions():
    """DpUtilsPatch should register DP sync functions for patching."""
    assert "_run_ar" in patch_dp_utils.DpUtilsPatch._attr_names_to_apply
    assert "_synchronize_dp_ranks" in patch_dp_utils.DpUtilsPatch._attr_names_to_apply
    assert patch_dp_utils.DpUtilsPatch._run_ar is patch_dp_utils._run_ar
    assert (
        patch_dp_utils.DpUtilsPatch._synchronize_dp_ranks
        is patch_dp_utils._synchronize_dp_ranks
    )
