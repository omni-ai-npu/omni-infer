# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only tests for NPU offload worker helpers and handler paths."""

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
    _block_byte_ptr,
    _block_slice,
    _descriptors_from_ops,
    _get_swap_blocks_batch,
    pin_memory_region,
)


def test_block_slice_full_and_partial():
    # CI default device is NPU; keep helpers on CPU. int8 arange is not portable.
    t = torch.arange(12, dtype=torch.int32, device="cpu").to(torch.int8).reshape(3, 4)
    assert torch.equal(_block_slice(t, 1, 0, 4), t[1])
    assert torch.equal(_block_slice(t, 1, 1, 2), t[1, 1:3])


def test_block_byte_ptr_and_descriptors():
    t = torch.zeros((2, 8), dtype=torch.int8, device="cpu")
    ptr = _block_byte_ptr(t, 1, 2)
    assert ptr == int(t.data_ptr()) + int(t.stride(0)) + 2
    src, dst, sizes = _descriptors_from_ops([(t, 0, 0, t, 1, 0, 8)])
    assert src.numel() == 1
    assert dst.numel() == 1
    assert int(sizes[0]) == 8


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

    start_ev = MagicMock()
    end_ev = MagicMock()
    end_ev.query.return_value = True
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=None)
    stream_ctx.__exit__ = MagicMock(return_value=False)

    with patch.object(torch.npu, "Event", side_effect=[start_ev, end_ev]), patch.object(
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
        start_event=MagicMock(),
        end_event=end2,
        num_bytes=8,
    )
    handler._transfers_by_id[9] = transfer
    handler.wait({9, 99})
    end2.synchronize.assert_called_once()

    # Shutdown with pending + mmap cleanup
    end3 = MagicMock()
    pending = Transfer(
        job_id=10,
        stream=handler._stream,
        start_event=MagicMock(),
        end_event=end3,
        num_bytes=4,
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
    handler._event_pool = [pooled, pooled]

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

    ConcreteGPU, _ = _concrete_specs()
    src = ConcreteGPU.__new__(ConcreteGPU)
    assert worker.submit_store(1, src, MagicMock()) is True
    assert worker.submit_load(2, MagicMock(), src) is True
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
