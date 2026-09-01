# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only tests for NPU offload worker helpers and handler paths."""

import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from omni_npu.v1.kv_offload.cpu import npu_worker as worker_mod
from omni_npu.v1.kv_offload.cpu.npu_worker import (
    NPUCPUOffloadingWorker,
    NpuSingleDirectionOffloadingHandler,
    Transfer,
    _block_slice,
    _get_swap_blocks_batch,
    pin_memory_region,
)


def test_block_slice_full_and_partial():
    # CI default device is NPU; keep helpers on CPU. int8 arange is not portable.
    t = torch.arange(12, dtype=torch.int32, device="cpu").to(torch.int8).reshape(3, 4)
    assert torch.equal(_block_slice(t, 1, 0, 4), t[1])
    assert torch.equal(_block_slice(t, 1, 1, 2), t[1, 1:3])


def test_owns_store_block_rotation():
    handler = NpuSingleDirectionOffloadingHandler.__new__(
        NpuSingleDirectionOffloadingHandler
    )
    handler.npu_to_cpu = True
    handler._rotate_store_writers = True
    handler._tp_size = 4
    handler._tp_rank = 1
    assert handler._owns_store_block(1) is True
    assert handler._owns_store_block(2) is False

    handler._rotate_store_writers = False
    assert handler._owns_store_block(2) is True


def _ref_plan_group_copies(
    handler, group_src, group_dst, group_size, refs, src_skip, dst_skip
):
    """Old per-op Python loop; used as oracle for the numpy planner."""
    src_ptrs = []
    dst_ptrs = []
    sizes = []
    for data_ref in refs:
        src_t = handler.src_tensors[data_ref.tensor_idx]
        dst_t = handler.dst_tensors[data_ref.tensor_idx]
        page = data_ref.page_size_bytes
        for i in range(group_size):
            src_sub = src_skip + i
            dst_sub = dst_skip + i
            src_block = int(group_src[src_sub // handler.src_block_size_factor])
            dst_block = int(group_dst[dst_sub // handler.dst_block_size_factor])
            if not handler._owns_store_block(dst_block):
                continue
            src_page = (src_sub % handler.src_block_size_factor) * page
            dst_page = (dst_sub % handler.dst_block_size_factor) * page
            src_ptrs.append(
                int(src_t.data_ptr()) + src_block * int(src_t.stride(0)) + src_page
            )
            dst_ptrs.append(
                int(dst_t.data_ptr()) + dst_block * int(dst_t.stride(0)) + dst_page
            )
            sizes.append(page)
    return (
        np.asarray(src_ptrs, dtype=np.int64),
        np.asarray(dst_ptrs, dtype=np.int64),
        np.asarray(sizes, dtype=np.int64),
    )


def _fill_group_plan(handler, group_src, group_dst, group_size, refs, src_skip, dst_skip):
    group_src = np.asarray(group_src, dtype=np.int64)
    group_dst = np.asarray(group_dst, dtype=np.int64)
    max_ops = int(group_size) * len(refs)
    if max_ops <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    src_ptrs = np.empty(max_ops, dtype=np.int64)
    dst_ptrs = np.empty(max_ops, dtype=np.int64)
    sizes = np.empty(max_ops, dtype=np.int64)
    n = handler._fill_group_descriptors(
        src_ptrs,
        dst_ptrs,
        sizes,
        0,
        group_src,
        group_dst,
        group_size,
        refs,
        src_skip,
        dst_skip,
    )
    return src_ptrs[:n], dst_ptrs[:n], sizes[:n]


def _assert_plan_matches(handler, group_src, group_dst, group_size, refs, src_skip, dst_skip):
    got = _fill_group_plan(
        handler, group_src, group_dst, group_size, refs, src_skip, dst_skip
    )
    exp = _ref_plan_group_copies(
        handler, group_src, group_dst, group_size, refs, src_skip, dst_skip
    )
    np.testing.assert_array_equal(got[0], exp[0])
    np.testing.assert_array_equal(got[1], exp[1])
    np.testing.assert_array_equal(got[2], exp[2])


def test_plan_group_copies_matches_python_oracle():
    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=1)
    refs = handler.kv_cache_groups_data_refs[0]
    _assert_plan_matches(handler, [0, 1, 2], [4, 5, 6], 3, refs, 0, 0)

    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=2)
    refs = handler.kv_cache_groups_data_refs[0]
    # dst packed by factor=2; skip=1 shifts the first logical page.
    _assert_plan_matches(handler, [0, 1, 2, 3], [8, 9, 10], 3, refs, 0, 1)

    handler, npu, cpu = _make_handler(npu_to_cpu=False, factor=2)
    refs = handler.kv_cache_groups_data_refs[0]
    _assert_plan_matches(handler, [2, 3], [0, 1, 2, 3], 3, refs, 1, 0)


def test_plan_group_copies_multi_ref_and_rotation():
    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=1)
    extra_npu = torch.zeros((4, 4), dtype=torch.int8, device="cpu")
    extra_cpu = torch.zeros((4, 4), dtype=torch.int8, device="cpu")
    handler.src_tensors = [npu, extra_npu]
    handler.dst_tensors = [cpu, extra_cpu]
    refs = [
        SimpleNamespace(tensor_idx=0, page_size_bytes=8),
        SimpleNamespace(tensor_idx=1, page_size_bytes=4),
    ]
    _assert_plan_matches(handler, [0, 1], [2, 3], 2, refs, 0, 0)

    handler._rotate_store_writers = True
    handler._tp_size = 2
    handler._tp_rank = 0
    _assert_plan_matches(handler, [0, 1], [2, 3], 2, refs, 0, 0)

    empty = _fill_group_plan(
        handler,
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        0,
        refs,
        0,
        0,
    )
    assert empty[0].size == 0
    assert empty[1].size == 0
    assert empty[2].size == 0


def _two_group_plan_setup():
    """Two KV groups plus src/dst block ids shared by planner tests."""
    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=1)
    extra_npu = torch.zeros((4, 8), dtype=torch.int8, device="cpu")
    extra_cpu = torch.zeros((4, 8), dtype=torch.int8, device="cpu")
    handler.src_tensors = [npu, extra_npu]
    handler.dst_tensors = [cpu, extra_cpu]
    handler.kv_cache_groups_data_refs = [
        [SimpleNamespace(tensor_idx=0, page_size_bytes=8)],
        [SimpleNamespace(tensor_idx=1, page_size_bytes=8)],
    ]
    src = np.asarray([0, 1, 2], dtype=np.int64)
    dst = np.asarray([0, 1, 2], dtype=np.int64)
    return handler, src, dst


def test_plan_transfer_descriptors_two_groups():
    handler, src, dst = _two_group_plan_setup()
    src_np, dst_np, sz_np = handler._plan_transfer_descriptors(
        src, dst, [2, 1], [0, 0], 3, 3
    )
    g0 = _fill_group_plan(handler, src[:2], dst[:2], 2, handler.kv_cache_groups_data_refs[0], 0, 0)
    g1 = _fill_group_plan(handler, src[2:], dst[2:], 1, handler.kv_cache_groups_data_refs[1], 0, 0)
    np.testing.assert_array_equal(src_np, np.concatenate([g0[0], g1[0]]))
    np.testing.assert_array_equal(dst_np, np.concatenate([g0[1], g1[1]]))
    np.testing.assert_array_equal(sz_np, np.concatenate([g0[2], g1[2]]))


def test_for_each_planned_group_matches_iter_and_empty_visitor():
    handler, src, dst = _two_group_plan_setup()
    args = (src, dst, [2, 1], [0, 0], 3, 3)
    expected = list(handler._iter_planned_groups(*args))
    visited = []
    handler._for_each_planned_group(*args, lambda *group: visited.append(group))
    assert visited == expected
    handler._for_each_planned_group(*args)


def test_handler_init_rejects_cpu_npu_mismatch():
    cpu = torch.zeros((2, 8), dtype=torch.int8, device="cpu")
    with pytest.raises(ValueError, match="non-empty and same length"):
        NpuSingleDirectionOffloadingHandler(
            npu_tensors=[],
            cpu_tensors=[cpu],
            block_size_factor=1,
            kv_cache_groups_data_refs=[],
            npu_to_cpu=True,
        )
    with pytest.raises(ValueError, match="npu_tensor must be 2-D int8"):
        NpuSingleDirectionOffloadingHandler(
            npu_tensors=[torch.zeros(8, dtype=torch.int8, device="cpu")],
            cpu_tensors=[cpu],
            block_size_factor=1,
            kv_cache_groups_data_refs=[],
            npu_to_cpu=True,
        )
    with pytest.raises(ValueError, match="device must be npu"):
        NpuSingleDirectionOffloadingHandler(
            npu_tensors=[torch.zeros((2, 8), dtype=torch.int8, device="cpu")],
            cpu_tensors=[cpu],
            block_size_factor=1,
            kv_cache_groups_data_refs=[],
            npu_to_cpu=True,
        )


def test_get_swap_blocks_batch_cache_and_import_paths(monkeypatch):
    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", False)
    assert _get_swap_blocks_batch() is None

    cached = object()
    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", cached)
    assert _get_swap_blocks_batch() is cached

    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", None)
    fake_ext = SimpleNamespace(
        swap_blocks_batch=lambda *a, **k: None,
        host_location="host",
        cann_memcpy_batch=True,
    )
    with patch.dict(
        "sys.modules",
        {"omni_npu.v1.kv_offload.cpu._swap_blocks_batch": fake_ext},
    ):
        assert _get_swap_blocks_batch() is fake_ext.swap_blocks_batch

    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", None)
    stale = SimpleNamespace(
        swap_blocks_batch=lambda *a, **k: "fn",
        host_location="x",
        cann_memcpy_batch=False,
    )
    with patch.dict(
        "sys.modules",
        {"omni_npu.v1.kv_offload.cpu._swap_blocks_batch": stale},
    ):
        assert _get_swap_blocks_batch() is stale.swap_blocks_batch

    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", None)
    broken = SimpleNamespace()  # missing swap_blocks_batch → AttributeError
    with patch.dict(
        "sys.modules",
        {"omni_npu.v1.kv_offload.cpu._swap_blocks_batch": broken},
    ):
        assert _get_swap_blocks_batch() is None
        assert worker_mod._SWAP_BLOCKS_BATCH is False


def test_pin_memory_region_paths():
    region = MagicMock()
    region.is_pinned = True
    pin_memory_region(region)

    region = MagicMock()
    region.is_pinned = False
    region._base = torch.zeros(4096, dtype=torch.int8, device="cpu")
    with patch.object(region._base, "data_ptr", return_value=1), patch(
        "omni_npu.v1.kv_offload.cpu.host_register.pin_host_mmap"
    ) as pin:
        pin_memory_region(region)
        pin.assert_not_called()

    region = MagicMock()
    region.is_pinned = False
    region._base = torch.zeros(4096, dtype=torch.int8, device="cpu")
    region._hugepage = False
    aligned = (region._base.data_ptr() // 4096) * 4096
    with patch.object(region._base, "data_ptr", return_value=aligned), patch(
        "omni_npu.v1.kv_offload.cpu.host_register.pin_host_mmap",
        side_effect=RuntimeError("pin fail"),
    ), patch.object(
        torch.npu, "current_device", side_effect=RuntimeError("no npu")
    ):
        pin_memory_region(region)
        assert region.is_pinned is False

    region = MagicMock()
    region.is_pinned = False
    region._base = torch.zeros(4096, dtype=torch.int8, device="cpu")
    region._hugepage = True
    aligned = (region._base.data_ptr() // 4096) * 4096
    with patch.object(region._base, "data_ptr", return_value=aligned), patch(
        "omni_npu.v1.kv_offload.cpu.host_register.pin_host_mmap"
    ) as pin, patch.object(torch.npu, "current_device", return_value=0):
        pin_memory_region(region)
        pin.assert_called_once()
        assert region.is_pinned is True
        assert region.npu_base is None

    region = MagicMock()
    region.is_pinned = False
    import builtins

    real_import = builtins.__import__

    def selective_import(
        name, global_ns=None, local_ns=None, fromlist=(), level=0
    ):
        if name == "omni_npu.v1.kv_offload.cpu.host_register" or name.endswith(
            ".host_register"
        ):
            raise ImportError("unavailable")
        return real_import(name, global_ns, local_ns, fromlist, level)

    with patch("builtins.__import__", side_effect=selective_import):
        pin_memory_region(region)


def _make_handler(*, npu_to_cpu=True, factor=1, rotate=False, tp_size=1, tp_rank=0):
    npu = torch.zeros((4, 8), dtype=torch.int8, device="cpu")
    cpu = torch.zeros((4, 8 * factor), dtype=torch.int8, device="cpu")
    # Bypass device checks by constructing via __new__ after validating shapes.
    handler = NpuSingleDirectionOffloadingHandler.__new__(
        NpuSingleDirectionOffloadingHandler
    )
    handler.src_tensors = [npu] if npu_to_cpu else [cpu]
    handler.dst_tensors = [cpu] if npu_to_cpu else [npu]
    handler.npu_to_cpu = npu_to_cpu
    handler.kv_cache_groups_data_refs = [
        [SimpleNamespace(tensor_idx=0, page_size_bytes=8)]
    ]
    handler.src_block_size_factor = 1 if npu_to_cpu else factor
    handler.dst_block_size_factor = factor if npu_to_cpu else 1
    handler._mmap_region = MagicMock()
    handler._rotate_store_writers = rotate
    handler._tp_rank = tp_rank
    handler._tp_size = tp_size
    handler._transfer_events = {}
    handler._transfers = deque()
    handler._transfers_by_id = {}
    handler._stream = MagicMock(name="stream")
    handler._event_pool = []
    handler._swap_blocks_batch = None
    handler._batch_direction = 1 if npu_to_cpu else 0
    handler._init_async_submit_state()
    return handler, npu, cpu


class _FakeBlockSpec:
    def __init__(self, block_ids, group_sizes=None, block_indices=None, gpu=False):
        self.block_ids = np.asarray(block_ids, dtype=np.int64)
        self.group_sizes = group_sizes
        self.block_indices = block_indices
        if gpu:
            # isinstance checks against real classes — patch later
            pass


def _concrete_specs():
    """Concrete subclasses — abstract LoadStoreSpec cannot be __new__'d."""
    from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, GPULoadStoreSpec

    class ConcreteGPU(GPULoadStoreSpec):
        @property
        def medium(self):
            return "gpu"

    class ConcreteCPU(BlockIDsLoadStoreSpec):
        @property
        def medium(self):
            return "cpu"

    return ConcreteGPU, ConcreteCPU


def test_handler_transfer_get_finished_wait_shutdown():
    from vllm.v1.kv_offload.base import TransferResult

    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=1)
    ConcreteGPU, ConcreteCPU = _concrete_specs()

    src = ConcreteGPU.__new__(ConcreteGPU)
    src.block_ids = np.asarray([0, 1], dtype=np.int64)
    src.group_sizes = [2]
    src.block_indices = [0]

    dst = ConcreteCPU.__new__(ConcreteCPU)
    dst.block_ids = np.asarray([0, 1], dtype=np.int64)

    end_ev = MagicMock()
    end_ev.query.return_value = True
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=None)
    stream_ctx.__exit__ = MagicMock(return_value=False)

    with patch.object(torch.npu, "Event", return_value=end_ev), patch.object(
        torch.npu, "stream", return_value=stream_ctx
    ), patch.object(torch.npu, "current_stream", return_value=MagicMock()):
        assert handler.transfer_async(7, src, dst) is True

    assert 7 in handler._transfers_by_id
    results = handler.get_finished()
    assert len(results) == 1
    assert isinstance(results[0], TransferResult)
    assert results[0].job_id == 7

    # Wait on missing + present job
    end2 = MagicMock()
    transfer = Transfer(
        job_id=9,
        stream=handler._stream,
        end_event=end2,
    )
    handler._transfers_by_id[9] = transfer
    handler.wait({9, 99})
    end2.synchronize.assert_called_once()

    # Shutdown with pending + mmap cleanup
    end3 = MagicMock()
    pending = Transfer(
        job_id=10,
        stream=handler._stream,
        end_event=end3,
    )
    handler._transfers.append(pending)
    handler._transfers_by_id[10] = pending
    mmap = handler._mmap_region
    handler.shutdown()
    end3.synchronize.assert_called()
    mmap.cleanup.assert_called_once()
    assert handler._mmap_region is None


def test_handler_transfer_batch_path_and_rotation_skip():
    handler, npu, cpu = _make_handler(
        npu_to_cpu=True, factor=1, rotate=True, tp_size=2, tp_rank=0
    )
    swap_fn = MagicMock()
    handler._swap_blocks_batch = swap_fn
    # reuse events from pool
    pooled = MagicMock()
    handler._event_pool = [pooled]

    ConcreteGPU, ConcreteCPU = _concrete_specs()
    src = ConcreteGPU.__new__(ConcreteGPU)
    src.block_ids = np.asarray([0, 1], dtype=np.int64)
    src.group_sizes = [2]
    src.block_indices = [0]
    dst = ConcreteCPU.__new__(ConcreteCPU)
    dst.block_ids = np.asarray([0, 1], dtype=np.int64)

    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=None)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    with patch.object(torch.npu, "stream", return_value=stream_ctx), patch.object(
        torch.npu, "current_stream", return_value=MagicMock()
    ):
        assert handler.transfer_async(1, src, dst) is True
    # rank0 owns even dst blocks → both 0 and? 0 yes, 1 no → one op or two
    assert swap_fn.called
    batched = swap_fn.call_args[0]
    assert int(batched[2].numel()) == 1
    assert int(batched[2][0]) == 8


def test_handler_transfer_rejects_bad_specs():
    handler, _, _ = _make_handler()
    with pytest.raises(TypeError, match="src_spec"):
        handler.transfer_async(1, object(), object())

    ConcreteGPU, ConcreteCPU = _concrete_specs()
    src = ConcreteCPU.__new__(ConcreteCPU)
    src.block_ids = np.asarray([0], dtype=np.int64)
    dst = ConcreteCPU.__new__(ConcreteCPU)
    dst.block_ids = np.asarray([0], dtype=np.int64)
    with pytest.raises(TypeError, match="npu_spec"):
        handler.transfer_async(1, src, dst)

    src = ConcreteGPU.__new__(ConcreteGPU)
    src.block_ids = np.asarray([0], dtype=np.int64)
    src.group_sizes = [1, 1]
    src.block_indices = [0, 0]
    with pytest.raises(ValueError, match="group_sizes length"):
        handler.transfer_async(1, src, dst)


def test_handler_init_happy_path_with_mocked_npu_stream():
    cpu = torch.zeros((2, 8), dtype=torch.int8, device="cpu")
    npu = MagicMock()
    npu.dtype = torch.int8
    npu.ndim = 2
    npu.device = SimpleNamespace(type="npu")
    npu.shape = (2, 8)

    stream = MagicMock()
    with patch.object(torch.npu, "Stream", return_value=stream), patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ):
        handler = NpuSingleDirectionOffloadingHandler(
            npu_tensors=[npu],
            cpu_tensors=[cpu],
            block_size_factor=1,
            kv_cache_groups_data_refs=[[]],
            npu_to_cpu=True,
        )
    assert handler._stream is stream
    assert handler.npu_to_cpu is True

    bad_cpu = MagicMock()
    bad_cpu.dtype = torch.int8
    bad_cpu.ndim = 2
    bad_cpu.device = SimpleNamespace(type="npu")
    bad_cpu.shape = (2, 8)
    with patch.object(torch.npu, "Stream", return_value=stream), patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ):
        with pytest.raises(ValueError, match="cpu_tensors must stay on CPU"):
            NpuSingleDirectionOffloadingHandler(
                npu_tensors=[npu],
                cpu_tensors=[bad_cpu],
                block_size_factor=1,
                kv_cache_groups_data_refs=[[]],
                npu_to_cpu=True,
            )


def test_worker_init_submit_finished_wait_shutdown():
    kv_tensor = MagicMock()
    kv_tensor.page_size_bytes = 8
    kv_tensor.tensor = torch.zeros(16, dtype=torch.int8, device="cpu")
    kv_caches = SimpleNamespace(
        tensors=[kv_tensor],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=8)]],
    )
    mmap_region = MagicMock()
    mmap_region._hugepage = False
    mmap_region.is_pinned = False
    mmap_region.npu_base = None
    mmap_region.create_next_worker_view.return_value = torch.zeros(
        (2, 8), dtype=torch.int8, device="cpu"
    )

    fake_handler = MagicMock()
    fake_handler.transfer_async.return_value = True
    fake_handler.submit.return_value = True
    fake_handler.get_finished.return_value = []
    with patch.object(worker_mod, "pin_memory_region"), patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ), patch.object(
        worker_mod, "NpuSingleDirectionOffloadingHandler", return_value=fake_handler
    ), patch(
        "omni_npu.v1.kv_offload.cpu.npu_worker.PIN_MEMORY", False
    ):
        worker = NPUCPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=1,
            num_cpu_blocks=2,
            mmap_region=mmap_region,
            tp_rank=0,
            tp_size=2,
            rotate_store_writers=True,
        )

    fake_handler.submit.return_value = True
    ConcreteGPU, _ = _concrete_specs()
    src = ConcreteGPU.__new__(ConcreteGPU)
    assert worker.submit_store(1, src, MagicMock()) is True
    assert worker.submit_load(2, MagicMock(), src) is True
    assert fake_handler.submit.call_count == 2
    assert worker.get_finished() == []
    worker.wait({1, 2})
    worker.shutdown()
    assert fake_handler.shutdown.call_count == 2


def test_worker_init_without_mmap_allocates_cpu():
    kv_tensor = MagicMock()
    kv_tensor.page_size_bytes = 4
    kv_tensor.tensor = torch.zeros(8, dtype=torch.int8, device="cpu")
    kv_caches = SimpleNamespace(tensors=[kv_tensor], group_data_refs=[[]])
    fake_handler = MagicMock()
    with patch.object(
        worker_mod,
        "NpuSingleDirectionOffloadingHandler",
        return_value=fake_handler,
    ) as handler_cls, patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ), patch(
        "omni_npu.v1.kv_offload.cpu.npu_worker.PIN_MEMORY", False
    ):
        NPUCPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=1,
            num_cpu_blocks=2,
            mmap_region=None,
        )
    assert handler_cls.call_count == 2


def _transfer_specs(npu_to_cpu: bool):
    ConcreteGPU, ConcreteCPU = _concrete_specs()
    gpu = ConcreteGPU.__new__(ConcreteGPU)
    gpu.block_ids = np.asarray([0, 1], dtype=np.int64)
    gpu.group_sizes = [2]
    gpu.block_indices = [0]
    cpu = ConcreteCPU.__new__(ConcreteCPU)
    cpu.block_ids = np.asarray([0, 1], dtype=np.int64)
    return (gpu, cpu) if npu_to_cpu else (cpu, gpu)


def test_store_submit_returns_before_enqueue():
    handler, _, _ = _make_handler(npu_to_cpu=True, factor=1)
    entered = threading.Event()
    release = threading.Event()
    compute_stream = MagicMock(name="compute_stream")

    def slow_swap(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=2)

    handler._swap_blocks_batch = slow_swap
    src, dst = _transfer_specs(npu_to_cpu=True)
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=None)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    end_ev = MagicMock()
    end_ev.query.return_value = False
    try:
        with patch.object(
            torch.npu, "Event", return_value=end_ev
        ), patch.object(torch.npu, "stream", return_value=stream_ctx), patch.object(
            torch.npu, "current_stream", return_value=compute_stream
        ):
            t0 = time.monotonic()
            assert handler.submit(3, src, dst) is True
            assert (time.monotonic() - t0) < 0.2
            queued = handler._queued.get(3)
            if queued is not None:
                assert queued.wait_stream is compute_stream
            assert 3 in handler._queued or 3 in handler._transfers_by_id
            assert entered.wait(timeout=2)
            handler._stream.wait_stream.assert_called_with(compute_stream)
            assert 3 in handler._queued or 3 in handler._transfers_by_id
            release.set()
            handler.wait({3})
            end_ev.synchronize.assert_called()
    finally:
        release.set()
        handler.shutdown()


def test_load_submit_returns_before_enqueue():
    handler, _, _ = _make_handler(npu_to_cpu=False, factor=1)
    entered = threading.Event()
    release = threading.Event()

    def slow_swap(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=2)

    handler._swap_blocks_batch = slow_swap
    src, dst = _transfer_specs(npu_to_cpu=False)
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=None)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    end_ev = MagicMock()
    end_ev.query.return_value = False
    try:
        with patch.object(
            torch.npu, "Event", return_value=end_ev
        ), patch.object(torch.npu, "stream", return_value=stream_ctx):
            t0 = time.monotonic()
            assert handler.submit(4, src, dst) is True
            assert (time.monotonic() - t0) < 0.2
            assert 4 in handler._queued or 4 in handler._transfers_by_id
            assert entered.wait(timeout=2)
            # enqueue still blocked: job is queued or already in _transfers
            assert 4 in handler._queued or 4 in handler._transfers_by_id
            release.set()
            handler.wait({4})
            end_ev.synchronize.assert_called()
    finally:
        release.set()
        handler.shutdown()
