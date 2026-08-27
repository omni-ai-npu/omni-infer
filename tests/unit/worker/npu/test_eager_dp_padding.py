# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for the eager DP-padding fix and its slot-mapping counterpart.

dp_utils.sync_cudagraph_and_dp_padding makes MRv2 pad DP tokens in
eager mode; NPUModelRunnerV2.prepare_attn then keeps the slot mappings at the
unpadded count so len(slot_mapping) == num_actual_tokens still holds.

Upstream callees are stubbed, so these run without a device. The MRv1
counterpart lives in tests/unit/vllm_patch/common/test_dp_utils_patch.py.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


# dispatch_cg_and_sync_dp / LM-head pad target


@pytest.fixture
def dispatch_context(monkeypatch):
    from omni_npu.model_config.config_loader.loader import model_extra_config
    from omni_npu.v1.layers.vocab_parallel_embedding import NPUParallelLMHead
    from omni_npu.worker.npu import dp_utils as npu_dp_utils

    monkeypatch.setattr(NPUParallelLMHead, "_dp_pad_n", 0)

    def configure(*, dp_lmhead=False, local_lmhead=False):
        parallel_config = SimpleNamespace(
            ena_dp_lmhead_parallel=dp_lmhead,
            ena_local_lmhead_parallel=local_lmhead,
        )
        monkeypatch.setattr(model_extra_config, "parall_config", parallel_config)

    return npu_dp_utils, NPUParallelLMHead, configure


def test_dispatch_publishes_global_lmhead_pad_target(
    dispatch_context, monkeypatch
):
    npu_dp_utils, parallel_lmhead, configure = dispatch_context
    configure(dp_lmhead=True)
    batch_desc = object()
    counts = torch.tensor([2, 7, 4], dtype=torch.int32)
    calls = []

    def fake_dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return batch_desc, counts

    monkeypatch.setattr(npu_dp_utils, "_DISPATCH_ORIGINAL", fake_dispatch)

    result_desc, result_counts = npu_dp_utils.dispatch_cg_and_sync_dp(
        "scheduler-output", 3, async_scheduling=True
    )

    assert result_desc is batch_desc
    assert result_counts is counts
    assert calls == [(("scheduler-output", 3), {"async_scheduling": True})]
    assert parallel_lmhead._dp_pad_n == 7


def test_local_lmhead_pad_target_uses_only_local_ranks(
    dispatch_context, monkeypatch
):
    npu_dp_utils, parallel_lmhead, configure = dispatch_context
    configure(local_lmhead=True)
    parallel_state_ext = importlib.import_module(
        "omni_npu.v1.distributed.parallel_state_ext"
    )
    monkeypatch.setattr(
        parallel_state_ext,
        "get_local_world_group",
        lambda: SimpleNamespace(ranks=[0, 2]),
    )

    npu_dp_utils._stash_lmhead_pad_target(
        torch.tensor([3, 9, 5], dtype=torch.int32)
    )

    assert parallel_lmhead._dp_pad_n == 5


def test_lmhead_pad_target_is_unchanged_when_parallelism_is_disabled(
    dispatch_context,
):
    npu_dp_utils, parallel_lmhead, configure = dispatch_context
    configure()
    parallel_lmhead._dp_pad_n = 11

    npu_dp_utils._stash_lmhead_pad_target(
        torch.tensor([3, 9, 5], dtype=torch.int32)
    )

    assert parallel_lmhead._dp_pad_n == 11


def test_dispatch_skips_pad_target_for_an_all_zero_batch(
    dispatch_context, monkeypatch
):
    npu_dp_utils, _, configure = dispatch_context
    configure(dp_lmhead=True)
    batch_desc = object()
    monkeypatch.setattr(
        npu_dp_utils,
        "_DISPATCH_ORIGINAL",
        lambda *args, **kwargs: (batch_desc, None),
    )
    monkeypatch.setattr(
        npu_dp_utils,
        "_stash_lmhead_pad_target",
        lambda counts: pytest.fail(f"unexpected pad target: {counts}"),
    )

    result_desc, result_counts = npu_dp_utils.dispatch_cg_and_sync_dp()

    assert result_desc is batch_desc
    assert result_counts is None


# sync_cudagraph_and_dp_padding


@dataclass
class _ParallelConfig:
    enable_expert_parallel: bool = True
    data_parallel_size: int = 8


@dataclass
class _VllmConfig:
    parallel_config: _ParallelConfig


class _CudaGraphManager:
    """Only the attribute the patch reads, not get_current_vllm_config(),
    which is a contextvar that reads back None at step time."""

    def __init__(self, enable_expert_parallel=True, data_parallel_size=8):
        self.vllm_config = _VllmConfig(
            _ParallelConfig(enable_expert_parallel, data_parallel_size)
        )


def _desc(num_tokens, cg_mode=CUDAGraphMode.NONE, num_reqs=1):
    return BatchExecutionDescriptor(
        cg_mode=cg_mode, num_tokens=num_tokens, num_reqs=num_reqs
    )


@pytest.fixture
def make_wrapper(monkeypatch):
    """Stub the captured upstream sync and hand back the NPU wrapper.

    The module captures the original at import time, so the stub goes onto
    _SYNC_ORIGINAL rather than the upstream module.
    """
    from omni_npu.worker.npu import dp_utils as npu_dp_utils

    def _make(batch_desc, num_tokens_across_dp):
        calls = []

        def fake_sync(cudagraph_manager, *args, **kwargs):
            calls.append((cudagraph_manager, args, kwargs))
            return batch_desc, num_tokens_across_dp

        monkeypatch.setattr(npu_dp_utils, "_SYNC_ORIGINAL", fake_sync)

        return npu_dp_utils.sync_cudagraph_and_dp_padding, calls

    return _make


def test_eager_ep_over_dp_pads_to_max(make_wrapper):
    """The rank below the maximum is padded up, and the vector is levelled."""
    across = torch.tensor([2, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
    wrapper, _ = make_wrapper(_desc(num_tokens=1), across)

    desc, out = wrapper(_CudaGraphManager())

    assert desc.num_tokens == 2
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_reqs == 1, "request count must not be padded along with tokens"
    assert out.tolist() == [2] * 8


def test_rank_already_at_max_still_levels_the_vector(make_wrapper):
    """Regression: the early return used to skip the rewrite.

    batch_desc needs no change on that rank, but the vector does, or it carries
    a different view of the group than its peers.
    """
    across = torch.tensor([3, 3, 3, 3, 2, 2, 2, 2], dtype=torch.int32)
    wrapper, _ = make_wrapper(_desc(num_tokens=3), across)

    desc, out = wrapper(_CudaGraphManager())

    assert desc.num_tokens == 3
    assert out.tolist() == [3] * 8


def test_uniform_batch_is_a_no_op(make_wrapper):
    across = torch.tensor([4] * 8, dtype=torch.int32)
    original = _desc(num_tokens=4)
    wrapper, _ = make_wrapper(original, across)

    desc, out = wrapper(_CudaGraphManager())

    assert desc is original
    assert out.tolist() == [4] * 8


@pytest.mark.parametrize("cg_mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_cudagraph_modes_are_left_alone(make_wrapper, cg_mode):
    """Upstream already padded and rewrote the vector for these."""
    across = torch.tensor([2, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
    original = _desc(num_tokens=1, cg_mode=cg_mode)
    wrapper, _ = make_wrapper(original, across)

    desc, out = wrapper(_CudaGraphManager())

    assert desc is original
    assert out.tolist() == [2, 1, 1, 1, 1, 1, 1, 1]


def test_all_zero_batch_is_left_alone(make_wrapper):
    """num_tokens_across_dp is None means every rank had zero tokens."""
    original = _desc(num_tokens=0, num_reqs=0)
    wrapper, _ = make_wrapper(original, None)

    desc, out = wrapper(_CudaGraphManager())

    assert desc is original
    assert out is None


def test_profile_run_without_manager_is_left_alone(make_wrapper):
    """No manager means the memory-profiling run; every rank runs one shape."""
    across = torch.tensor([2, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
    original = _desc(num_tokens=1)
    wrapper, _ = make_wrapper(original, across)

    desc, out = wrapper(None)

    assert desc is original
    assert out.tolist() == [2, 1, 1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "kwargs",
    [{"enable_expert_parallel": False}, {"data_parallel_size": 1}],
    ids=["no-expert-parallel", "dp-size-1"],
)
def test_padding_only_applies_to_ep_over_dp(make_wrapper, kwargs):
    """Without EP over DP there is no fused collective demanding equal inputs."""
    across = torch.tensor([2, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
    original = _desc(num_tokens=1)
    wrapper, _ = make_wrapper(original, across)

    desc, out = wrapper(_CudaGraphManager(**kwargs))

    assert desc is original
    assert out.tolist() == [2, 1, 1, 1, 1, 1, 1, 1]


def test_arguments_are_forwarded_unchanged(make_wrapper):
    across = torch.tensor([1] * 8, dtype=torch.int32)
    wrapper, calls = make_wrapper(_desc(num_tokens=1), across)

    mgr = _CudaGraphManager()
    wrapper(mgr, "desired", 1, 1, None, 8, 3, num_active_loras=2)

    assert calls == [(mgr, ("desired", 1, 1, None, 8, 3), {"num_active_loras": 2})]


# NPUModelRunnerV2.prepare_attn / prepare_inputs


@dataclass
class _InputBatch:
    num_tokens: int


class _Backend:
    def __init__(self, forward_includes_kv_cache_update=True):
        self.forward_includes_kv_cache_update = forward_includes_kv_cache_update


class _AttnGroup:
    def __init__(self, backend):
        self.backend = backend


class _KVCacheGroup:
    def __init__(self, spec=None):
        self.kv_cache_spec = spec if spec is not None else object()


class _KVCacheConfig:
    def __init__(self, groups=None):
        self.kv_cache_groups = groups if groups is not None else [_KVCacheGroup()]


@pytest.fixture
def runner(monkeypatch):
    """A runner with the upstream halves of both methods stubbed.

    __new__ skips __init__ since a real runner needs a device; super() resolves
    to GPUModelRunner, so patching it there is what gets called.
    """
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2

    slot_mappings = torch.arange(2 * 8, dtype=torch.int64).reshape(2, 8)
    block_tables = ("block-tables",)

    monkeypatch.setattr(
        GPUModelRunner,
        "prepare_attn",
        lambda self, input_batch: (block_tables, slot_mappings),
        raising=False,
    )
    monkeypatch.setattr(
        GPUModelRunner,
        "prepare_inputs",
        lambda self, scheduler_output, batch_desc: ("input-batch", batch_desc),
        raising=False,
    )

    obj = NPUModelRunnerV2.__new__(NPUModelRunnerV2)
    obj.attn_groups = [[_AttnGroup(_Backend())]]
    obj.kv_cache_config = _KVCacheConfig()
    return obj, slot_mappings, block_tables


def test_prepare_inputs_stashes_the_cudagraph_mode(runner):
    obj, _, _ = runner
    desc = _desc(num_tokens=2, cg_mode=CUDAGraphMode.PIECEWISE)

    out = obj.prepare_inputs("sched", desc)

    assert obj._omni_cg_mode == CUDAGraphMode.PIECEWISE
    assert out == ("input-batch", desc)


def test_eager_truncates_slot_mappings_to_unpadded(runner):
    """The case the DSA indexer used to fail on: 8 padded slots, 5 real tokens."""
    obj, slot_mappings, block_tables = runner
    obj._omni_cg_mode = CUDAGraphMode.NONE

    tables, slots = obj.prepare_attn(_InputBatch(num_tokens=5))

    assert tables is block_tables
    assert slots.shape == (2, 5)
    assert torch.equal(slots, slot_mappings[:, :5])


def test_piecewise_also_truncates(runner):
    """Upstream sizes the attention metadata unpadded for anything but FULL."""
    obj, _, _ = runner
    obj._omni_cg_mode = CUDAGraphMode.PIECEWISE

    _, slots = obj.prepare_attn(_InputBatch(num_tokens=5))

    assert slots.shape == (2, 5)


def test_full_cudagraph_keeps_the_padded_slot_mappings(runner):
    """Under FULL the metadata is padded too, so the lengths already agree."""
    obj, slot_mappings, _ = runner
    obj._omni_cg_mode = CUDAGraphMode.FULL

    _, slots = obj.prepare_attn(_InputBatch(num_tokens=5))

    assert slots is slot_mappings


def test_no_truncation_when_lengths_already_match(runner):
    obj, slot_mappings, _ = runner
    obj._omni_cg_mode = CUDAGraphMode.NONE

    _, slots = obj.prepare_attn(_InputBatch(num_tokens=8))

    assert slots is slot_mappings


def test_separate_kv_update_keeps_the_padded_slot_mappings(runner):
    """A backend writing KV outside forward() needs slots matching padded k/v."""
    obj, slot_mappings, _ = runner
    obj._omni_cg_mode = CUDAGraphMode.NONE
    obj.attn_groups = [[_AttnGroup(_Backend(forward_includes_kv_cache_update=False))]]

    _, slots = obj.prepare_attn(_InputBatch(num_tokens=5))

    assert slots is slot_mappings


def test_missing_stash_raises_instead_of_silently_skipping(runner):
    """Fail loud if upstream stops calling prepare_inputs first."""
    obj, _, _ = runner

    with pytest.raises(RuntimeError, match="stashed cudagraph mode"):
        obj.prepare_attn(_InputBatch(num_tokens=5))


def test_separate_kv_update_result_is_cached(runner):
    """Computed once; a later flip of the backend flag must not be re-read."""
    obj, _, _ = runner

    assert obj._omni_has_separate_kv_update() is False
    obj.attn_groups = [[_AttnGroup(_Backend(forward_includes_kv_cache_update=False))]]
    assert obj._omni_has_separate_kv_update() is False


def test_encoder_only_groups_are_ignored(runner):
    """MRv1 excludes them from the same check."""
    from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

    obj, _, _ = runner
    spec = EncoderOnlyAttentionSpec.__new__(EncoderOnlyAttentionSpec)
    obj.kv_cache_config = _KVCacheConfig([_KVCacheGroup(spec)])
    obj.attn_groups = [[_AttnGroup(_Backend(forward_includes_kv_cache_update=False))]]

    assert obj._omni_has_separate_kv_update() is False
