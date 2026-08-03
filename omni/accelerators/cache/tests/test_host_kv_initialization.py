import gc
import mmap
import os
from types import SimpleNamespace

import pytest
import torch


MAPPING_BYTES = 2 * 1024 * 1024
BF16_BYTES = 2
DIRTY_VALUE = 7.0


def _fill_mapping(path):
    fd = os.open(path, os.O_RDWR)
    mapping = mmap.mmap(fd, MAPPING_BYTES, access=mmap.ACCESS_WRITE)
    tensor = torch.frombuffer(
        mapping,
        dtype=torch.bfloat16,
        count=MAPPING_BYTES // BF16_BYTES,
    )
    tensor.fill_(DIRTY_VALUE)
    del tensor
    mapping.close()
    os.close(fd)


def _close_pool(pool):
    pool.shared_tensor_npu = None
    pool.kvi_tensors_swap = []
    pool.kvi_tensors = []
    pool.shared_tensor = None
    gc.collect()
    pool.close()
    del pool.mmap_obj
    del pool.fd


@pytest.mark.parametrize("enable_host_mapping", [False, True])
def test_decode_initializes_entire_host_kv_to_zero(
    monkeypatch,
    tmp_path,
    enable_host_mapping,
):
    monkeypatch.setenv("ENABLE_HOST_MAPPING", "0")
    from omni.accelerators.cache import kv_mem_pool

    monkeypatch.setattr(
        kv_mem_pool,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        kv_mem_pool,
        "NPUTensorRegister",
        lambda: object(),
        raising=False,
    )
    monkeypatch.setattr(kv_mem_pool, "AscendCLStream", lambda: object())
    monkeypatch.setattr(
        kv_mem_pool.model_extra_config.operator_opt_config,
        "enable_dsa",
        True,
    )

    registered_nonzero = []

    def register_identity(self, tensor):
        registered_nonzero.append(int(torch.count_nonzero(tensor).item()))
        return tensor

    monkeypatch.setattr(
        kv_mem_pool.KVCacheMemoryPool,
        "host_swap_device",
        register_identity,
    )
    monkeypatch.setenv(
        "ENABLE_HOST_MAPPING",
        str(int(enable_host_mapping)),
    )

    path = tmp_path / "dirty_host_kv.bin"
    with path.open("wb") as file:
        file.truncate(MAPPING_BYTES)
    _fill_mapping(path)

    pool = None
    try:
        pool = kv_mem_pool.KVCacheMemoryPool(
            hugepage_path=str(path),
            mmap_size=MAPPING_BYTES,
            shape=(1, 1, 2, 128, 704),
            rank=0,
            device=torch.device("cpu"),
        )

        expected_registered_nonzero = [0, 0] if enable_host_mapping else []
        assert registered_nonzero == expected_registered_nonzero
        assert torch.count_nonzero(pool.shared_tensor).item() == 0
        assert all(
            torch.count_nonzero(tensor).item() == 0
            for tensor in pool.kvi_tensors
        )
    finally:
        if pool is not None:
            _close_pool(pool)
