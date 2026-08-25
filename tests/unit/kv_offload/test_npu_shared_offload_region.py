# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for shared mmap helpers that do not need NPU."""

import errno
import mmap
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from omni_npu.v1.kv_offload.cpu import npu_shared_offload_region as region_mod
from omni_npu.v1.kv_offload.cpu.npu_shared_offload_region import (
    NPUSharedOffloadRegion,
    _align_up,
    _check_shm_free_space,
    _env_flag_enabled,
    _fallback_populate_write,
    _hugetlbfs_mounted,
    _mlock_host_tensor,
    _wait_for_file_size,
)


def test_env_flag_and_align_up(monkeypatch):
    monkeypatch.setenv("OMNI_KV_OFFLOAD_HUGEPAGE", "YES")
    assert _env_flag_enabled("OMNI_KV_OFFLOAD_HUGEPAGE") is True
    monkeypatch.setenv("OMNI_KV_OFFLOAD_HUGEPAGE", "0")
    assert _env_flag_enabled("OMNI_KV_OFFLOAD_HUGEPAGE") is False
    assert _align_up(1, 4096) == 4096
    assert _align_up(4096, 4096) == 4096


def test_hugetlbfs_mounted_true_false_and_oserror():
    mounts = "nodev /dev/hugepages hugetlbfs rw 0 0\n"
    with patch("builtins.open", mock_open(read_data=mounts)):
        assert _hugetlbfs_mounted("/dev/hugepages") is True
        assert _hugetlbfs_mounted("/other") is False
    with patch("builtins.open", side_effect=OSError("denied")):
        assert _hugetlbfs_mounted("/dev/hugepages") is False


def test_check_shm_free_space():
    st = SimpleNamespace(f_bavail=1, f_frsize=4096)
    with patch.object(os, "statvfs", return_value=st):
        _check_shm_free_space(100)
        with pytest.raises(RuntimeError, match="Not enough space"):
            _check_shm_free_space(10_000)


def test_wait_for_file_size_ok_and_timeout():
    fd = 7
    with patch.object(os, "fstat", return_value=SimpleNamespace(st_size=128)):
        _wait_for_file_size(fd, 64, timeout=1.0)
    with patch.object(os, "fstat", return_value=SimpleNamespace(st_size=1)), patch(
        "omni_npu.v1.kv_offload.cpu.npu_shared_offload_region.time.sleep"
    ), patch(
        "omni_npu.v1.kv_offload.cpu.npu_shared_offload_region.time.monotonic",
        side_effect=[0.0, 2.0, 2.0],
    ):
        with pytest.raises(TimeoutError, match="Timed out"):
            _wait_for_file_size(fd, 64, timeout=1.0)


def test_fallback_populate_write_touches_pages():
    size = mmap.PAGESIZE * 2
    mm_obj = mmap.mmap(-1, size)
    try:
        _fallback_populate_write(mm_obj, 0, size)
    finally:
        mm_obj.close()


def test_mlock_host_tensor_success_fail_and_exception():
    tensor = MagicMock()
    tensor.data_ptr.return_value = 1
    tensor.nbytes = 8
    ext = MagicMock()
    with patch.dict(
        "sys.modules",
        {"omni_npu.v1.kv_offload.cpu._zero_copy_npu": ext},
    ):
        ext.mlock_host.return_value = True
        _mlock_host_tensor(tensor)
        ext.mlock_host.return_value = False
        _mlock_host_tensor(tensor)
        ext.mlock_host.side_effect = RuntimeError("boom")
        _mlock_host_tensor(tensor)


def test_create_next_worker_view_requires_rank_and_bounds():
    region = NPUSharedOffloadRegion.__new__(NPUSharedOffloadRegion)
    region.rank = None
    with pytest.raises(RuntimeError, match="requires rank"):
        region.create_next_worker_view(8)

    region.rank = 0
    region._worker_offset = 0
    region._worker_area_end = 4
    with pytest.raises(ValueError, match="exceeds worker area"):
        region.create_next_worker_view(8)


def test_create_next_worker_view_success():
    region = NPUSharedOffloadRegion.__new__(NPUSharedOffloadRegion)
    region.rank = 0
    region.num_blocks = 2
    region._row_stride = 16
    region._worker_offset = 0
    region._worker_area_end = 16
    region._views = []
    region._base = torch.zeros(32, dtype=torch.int8, device="cpu")
    view = region.create_next_worker_view(8)
    assert tuple(view.shape) == (2, 8)
    assert region._worker_offset == 8
    assert region._views[-1] is view


def test_as_npu_view_success_and_canonical_and_memoryview():
    region = NPUSharedOffloadRegion.__new__(NPUSharedOffloadRegion)
    region.npu_base = None
    with pytest.raises(RuntimeError, match="npu_base"):
        region.as_npu_view(torch.zeros(1, dtype=torch.int8, device="cpu"))

    region.npu_base = torch.zeros(16, dtype=torch.int8, device="cpu")
    region._npu_views = []
    cpu_view = torch.as_strided(region.npu_base, size=(2, 4), stride=(8, 1), storage_offset=0)
    npu_view = region.as_npu_view(cpu_view)
    assert tuple(npu_view.shape) == (2, 4)
    assert region._npu_views[-1] is npu_view

    region._canonical_offset = 8
    region._row_stride = 8
    with pytest.raises(ValueError, match="row stride"):
        region.create_next_canonical_view(8)

    region._canonical_offset = 0
    region._row_stride = 16
    region.num_blocks = 2
    region._views = []
    region._base = torch.zeros(32, dtype=torch.int8, device="cpu")
    canon = region.create_next_canonical_view(8)
    assert tuple(canon.shape) == (2, 8)
    assert region._canonical_offset == 8

    region.num_blocks = 1
    region._row_stride = 8
    copied = MagicMock()
    copied.ctypes = SimpleNamespace(data=1)
    viewed = MagicMock()
    viewed.numpy.return_value = copied
    base = MagicMock()
    base.view.return_value = viewed
    base.data_ptr.return_value = 2
    region._base = base
    with pytest.raises(RuntimeError, match="zero-copy"):
        region.create_kv_memoryview()

    # Happy path uses a real contiguous CPU buffer so numpy()/memoryview work.
    region._base = torch.zeros(8, dtype=torch.int8, device="cpu")
    region.num_blocks = 1
    region._row_stride = 8
    mv = region.create_kv_memoryview()
    assert isinstance(mv, memoryview)
    assert mv.nbytes == 8


def test_get_populate_write_fn_paths():
    mmap_obj = MagicMock()
    err = OSError()
    err.errno = errno.EINVAL
    mmap_obj.madvise.side_effect = err
    with patch.object(region_mod.errno, "EINVAL", errno.EINVAL):
        fn = region_mod._get_populate_write_fn(mmap_obj)
        assert fn is region_mod._fallback_populate_write

    mmap_obj2 = MagicMock()
    other = OSError()
    other.errno = errno.EPERM
    mmap_obj2.madvise.side_effect = other
    with pytest.raises(OSError):
        region_mod._get_populate_write_fn(mmap_obj2)

    mmap_obj3 = MagicMock()
    assert region_mod._get_populate_write_fn(mmap_obj3) is region_mod._madvise_populate_write


def _run_region_init(monkeypatch, *, hugepage, rank, num_blocks=1):
    page = mmap.PAGESIZE
    size = num_blocks * page
    monkeypatch.delenv("OMNI_KV_OFFLOAD_HUGEPAGE", raising=False)
    if hugepage:
        monkeypatch.setenv("OMNI_KV_OFFLOAD_HUGEPAGE", "1")
        monkeypatch.setenv("OMNI_KV_OFFLOAD_HUGEPAGE_SIZE", str(page))

    real_mm = mmap.mmap(-1, max(size, page))

    open_calls = {"n": 0}

    def fake_open(path, flags, mode=0o600):
        open_calls["n"] += 1
        if flags & os.O_EXCL:
            return 11
        raise AssertionError(f"unexpected open {path} {flags}")

    with patch.object(os, "open", side_effect=fake_open), patch.object(
        os, "ftruncate"
    ), patch.object(
        os, "statvfs", return_value=SimpleNamespace(f_bavail=10**9, f_frsize=4096)
    ), patch.object(mmap, "mmap", return_value=real_mm), patch.object(
        region_mod, "_get_populate_write_fn", return_value=lambda *a, **k: None
    ), patch.object(region_mod, "_mlock_host_tensor"), patch.object(
        region_mod, "_hugetlbfs_mounted", return_value=True
    ):
        return NPUSharedOffloadRegion("id1", num_blocks, rank, size, page)


def test_region_init_creator_shm_and_hugepage(monkeypatch):
    region = _run_region_init(monkeypatch, hugepage=False, rank=0)
    assert region._creator is True
    assert region.rank == 0
    assert region.mmap_path.endswith("vllm_offload_id1.mmap")
    region.mmap_obj = None  # avoid double-close of anonymous mmap in GC paths

    region_h = _run_region_init(monkeypatch, hugepage=True, rank=None)
    assert region_h._hugepage is True
    assert region_h.rank is None
    region_h.mmap_obj = None


def test_region_init_rejects_bad_page_and_missing_hugetlb(monkeypatch):
    with pytest.raises(ValueError, match="multiple of page size"):
        NPUSharedOffloadRegion("bad", 1, 0, 7, 7)

    monkeypatch.setenv("OMNI_KV_OFFLOAD_HUGEPAGE", "1")
    with patch.object(region_mod, "_hugetlbfs_mounted", return_value=False):
        with pytest.raises(RuntimeError, match="hugetlbfs"):
            NPUSharedOffloadRegion("bad", 1, 0, mmap.PAGESIZE, mmap.PAGESIZE)


def test_region_init_joiner_and_creator_cleanup_on_truncate_fail(monkeypatch):
    page = mmap.PAGESIZE
    monkeypatch.delenv("OMNI_KV_OFFLOAD_HUGEPAGE", raising=False)
    real_mm = mmap.mmap(-1, page)

    def open_joiner(path, flags, mode=0o600):
        if flags & os.O_EXCL:
            raise FileExistsError
        return 22

    with patch.object(os, "open", side_effect=open_joiner), patch.object(
        region_mod, "_wait_for_file_size"
    ), patch.object(mmap, "mmap", return_value=real_mm), patch.object(
        region_mod, "_get_populate_write_fn", return_value=lambda *a, **k: None
    ), patch.object(region_mod, "_mlock_host_tensor"):
        region = NPUSharedOffloadRegion("join", 1, 0, page, page)
        assert region._creator is False
        region.mmap_obj = None

    def open_creator(path, flags, mode=0o600):
        return 33

    with patch.object(os, "open", side_effect=open_creator), patch.object(
        os, "ftruncate", side_effect=OSError("fail")
    ), patch.object(os, "unlink") as unlink, patch.object(os, "close") as close, patch.object(
        os, "statvfs", return_value=SimpleNamespace(f_bavail=10**9, f_frsize=4096)
    ):
        with pytest.raises(OSError, match="fail"):
            NPUSharedOffloadRegion("fail", 1, 0, page, page)
        unlink.assert_called()
        close.assert_called()


def test_region_cleanup_paths():
    region = NPUSharedOffloadRegion.__new__(NPUSharedOffloadRegion)
    region.is_pinned = True
    region._register_ptr = 0x11
    region.rank = 0
    region._npu_views = [object()]
    region.npu_base = object()
    region._views = [object()]
    region._base = object()
    region.mmap_obj = MagicMock()
    region.fd = 9
    region._creator = True
    region.mmap_path = "/tmp/vllm_offload_x.mmap"

    with patch(
        "omni_npu.v1.kv_offload.cpu.host_register.unregister_host_tensor",
        return_value=1,
    ), patch.object(os, "close"), patch.object(os, "unlink"):
        region.cleanup()
    assert region.is_pinned is False
    assert region.fd is None
    assert region._creator is False

    region2 = NPUSharedOffloadRegion.__new__(NPUSharedOffloadRegion)
    region2.is_pinned = True
    region2._register_ptr = 0x22
    region2.rank = 1
    region2._npu_views = None
    region2.npu_base = None
    region2._views = None
    region2._base = None
    region2.mmap_obj = MagicMock()
    region2.mmap_obj.close.side_effect = RuntimeError("close")
    region2.fd = 8
    region2._creator = False
    region2.mmap_path = None
    with patch(
        "omni_npu.v1.kv_offload.cpu.host_register.unregister_host_tensor",
        side_effect=RuntimeError("unreg"),
    ), patch.object(os, "close", side_effect=OSError("fd")):
        region2.cleanup()
