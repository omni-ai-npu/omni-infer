# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import hashlib
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import psutil
import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

GB = 1024 ** 3
MB = 1024 ** 2
BYTE_LENGTH = 8
META_MAX_DIMS = 16
META_OK_IDX = 0
META_NDIM_IDX = 1
META_DTYPE_IDX = 2
META_DEVICE_IDX = 3
META_SHAPE_START_IDX = 4
META_WIDTH = META_SHAPE_START_IDX + META_MAX_DIMS

DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
    torch.int8: 5,
    torch.uint8: 6,
    torch.int16: 7,
    torch.int32: 8,
    torch.int64: 9,
    torch.bool: 10,
}
CODE_TO_DTYPE = {code: dtype for dtype, code in DTYPE_TO_CODE.items()}
DEVICE_TYPE_TO_CODE = {
    "cpu": 1,
    "npu": 2,
}
CODE_TO_DEVICE_TYPE = {code: device for device, code in DEVICE_TYPE_TO_CODE.items()}


def _mm_hash_to_sha256(mm_hash: str) -> str:
    if len(mm_hash) > 1000:
        raise ValueError("mm_hash exceeds the safe length limit for a shared memory identifier.")
    # Filtering out invalid characters, such as '/' and '\0'
    sanitized = mm_hash.replace('/', '_').replace('\0', '_')
    hash_hex = hashlib.sha256(sanitized.encode()).hexdigest()
    # Use Base64 encoding to avoid special characters, which is shorter and more secure.
    return base64.urlsafe_b64encode(hash_hex.encode()).decode().rstrip('=')


@dataclass
class ECSharedMemoryConnectorMetadata(ECConnectorMetadata):
    mm_hashes: list[str]
    # Hashes confirmed to already exist in shared memory (cache hits) for this step.
    # Detected by has_cache_item() in the scheduler process and shipped to the worker so
    # the producer can refresh its eviction LRU on hits (see bind_connector_metadata).
    mm_hashes_hit: list[str]

    def __init__(self) -> None:
        self.mm_hashes = []
        self.mm_hashes_hit = []

    def add_mm_hash(self, mm_hash: str) -> None:
        self.mm_hashes.append(mm_hash)

    def add_mm_hash_hit(self, mm_hash: str) -> None:
        self.mm_hashes_hit.append(mm_hash)


@dataclass(frozen=True)
class _CacheLoadResult:
    tensor: torch.Tensor | None
    ok: bool
    error: str | None
    shm: shared_memory.SharedMemory | None


class ECSharedMemoryConnector(ECConnectorBase):
    """EC connector backed by POSIX shared memory segments."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.count = 0
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")
        available_mem = psutil.virtual_memory().available
        _max_bytes = transfer_config.get_from_extra_config("ec_shared_memory_max_bytes", 10) * GB
        if _max_bytes > available_mem * 0.1:
            logger.info(f"shm ecconnector max bytes > available_mem * 0.1")
        self._max_bytes = min(int(available_mem * 0.1), _max_bytes)
        logger.info(f"shm ecconnector max bytes:{self._max_bytes / GB}GB")
        self._mm_hashes_need_loads: set[str] = set()
        self._mm_hashes_hit: set[str] = set()
        self._mm_hash_sizes: dict[str, int] = {}
        self._mm_hash_refcounts: dict[str, int] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._shm_close_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ec-shm-close")
        self._shm_load_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ec-shm-load")

    @staticmethod
    def _serialize_cache(tensor: torch.Tensor) -> tuple[torch.Tensor, bytes, int]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("encoder cache must be a torch.Tensor")

        cpu_tensor = tensor.detach().contiguous().cpu()
        tensor_meta = {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype).removeprefix("torch."),
        }
        meta_payload = json.dumps(tensor_meta).encode("utf-8")
        total_size_bytes = BYTE_LENGTH + len(meta_payload) + (cpu_tensor.numel() * cpu_tensor.element_size())
        return cpu_tensor, meta_payload, total_size_bytes

    @staticmethod
    def _deserialize_cache(payload: bytes | memoryview) -> torch.Tensor:
        payload_size = len(payload)
        if payload_size < BYTE_LENGTH:
            raise ValueError("EC shared memory payload is too small to contain metadata length.")

        meta_size = int.from_bytes(payload[:BYTE_LENGTH], "little")
        meta_start = BYTE_LENGTH
        meta_end = meta_start + meta_size
        if meta_size <= 0 or meta_end > payload_size:
            raise ValueError(
                f"Invalid EC shared memory metadata length: {meta_size}, "
                f"payload size: {payload_size}.")

        try:
            tensor_meta = json.loads(bytes(payload[meta_start:meta_end]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid EC shared memory cache metadata.") from exc
        if not isinstance(tensor_meta, dict):
            raise ValueError("EC shared memory cache metadata must be an object.")

        try:
            shape_value = tensor_meta["shape"]
            dtype_name = tensor_meta["dtype"]
        except KeyError as exc:
            raise ValueError(
                f"Missing EC shared memory cache metadata field: {exc.args[0]}"
            ) from exc
        if not isinstance(shape_value, list) or any(
            type(dim) is not int or dim < 0 for dim in shape_value
        ):
            raise ValueError(f"Invalid EC shared memory cache tensor shape: {shape_value}")
        if not isinstance(dtype_name, str):
            raise ValueError(f"Invalid EC shared memory cache tensor dtype: {dtype_name}")

        shape = tuple(shape_value)
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported EC shared memory cache tensor dtype: {dtype_name}")
        numel = 1
        for dim in shape:
            numel *= dim
        raw_view = payload[meta_end:]
        required_raw_bytes = numel * torch.empty((), dtype=dtype).element_size()
        if len(raw_view) < required_raw_bytes:
            raise ValueError(
                f"EC shared memory payload raw tensor bytes are incomplete: "
                f"expected {required_raw_bytes}, got {len(raw_view)}.")
        if numel == 0:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            shm_tensor = torch.frombuffer(raw_view, dtype=dtype, count=numel).reshape(shape)
            tensor = shm_tensor.clone()
            del shm_tensor
        return tensor.to(current_platform.device_type)

    @staticmethod
    def _tp_src_rank() -> int:
        tp_group = get_tp_group()
        ranks = getattr(tp_group, "ranks", None)
        if not ranks:
            raise RuntimeError("TP group does not expose ranks for EC cache broadcast.")
        return ranks[0]

    @staticmethod
    def _tp_device_group():
        tp_group = get_tp_group()
        group = getattr(tp_group, "device_group", None)
        if group is None:
            raise RuntimeError("TP group does not expose device_group for EC cache broadcast.")
        return group

    @staticmethod
    def _encode_cache_meta_row(tensor: torch.Tensor | None, load_ok: bool) -> list[int]:
        row = [0] * META_WIDTH
        if not load_ok or tensor is None:
            return row

        shape = tuple(tensor.shape)
        if len(shape) > META_MAX_DIMS:
            raise ValueError(
                f"EC shared memory cache tensor dim {len(shape)} exceeds max {META_MAX_DIMS}.")

        dtype_code = DTYPE_TO_CODE.get(tensor.dtype)
        if dtype_code is None:
            raise ValueError(f"Unsupported EC shared memory cache dtype: {tensor.dtype}")
        device_code = DEVICE_TYPE_TO_CODE.get(tensor.device.type)
        if device_code is None:
            raise ValueError(
                f"Unsupported EC shared memory cache device type: {tensor.device.type}")

        row[META_OK_IDX] = 1
        row[META_NDIM_IDX] = len(shape)
        row[META_DTYPE_IDX] = dtype_code
        row[META_DEVICE_IDX] = device_code
        for idx, dim in enumerate(shape):
            row[META_SHAPE_START_IDX + idx] = int(dim)
        return row

    @staticmethod
    def _decode_cache_meta_row(
        row: torch.Tensor | list[int],
    ) -> tuple[bool, tuple[int, ...] | None, torch.dtype | None, str | None]:
        values = row.tolist() if isinstance(row, torch.Tensor) else row
        if values[META_OK_IDX] == 0:
            return False, None, None, None

        ndim = int(values[META_NDIM_IDX])
        if ndim < 0 or ndim > META_MAX_DIMS:
            raise ValueError(f"Invalid EC shared memory cache tensor ndim: {ndim}")
        shape = tuple(int(values[META_SHAPE_START_IDX + idx]) for idx in range(ndim))
        dtype_code = int(values[META_DTYPE_IDX])
        dtype = CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError(f"Unsupported EC shared memory cache dtype code: {dtype_code}")
        device_code = int(values[META_DEVICE_IDX])
        device_type = CODE_TO_DEVICE_TYPE.get(device_code)
        if device_type is None:
            raise ValueError(f"Unsupported EC shared memory cache device code: {device_code}")
        return True, shape, dtype, device_type

    def _build_cache_meta_tensor(
        self,
        mm_hashes: list[str],
        load_results: dict[str, _CacheLoadResult],
        device: str,
    ) -> torch.Tensor:
        rows = []
        for mm_hash in mm_hashes:
            result = load_results[mm_hash]
            try:
                rows.append(self._encode_cache_meta_row(result.tensor, result.ok))
            except ValueError as exc:
                logger.warning("Skip EC cache load for hash %s: %s", mm_hash, exc)
                rows.append([0] * META_WIDTH)
                continue
            if not result.ok:
                logger.warning("Skip EC cache load for hash %s: %s", mm_hash, result.error)
        return torch.tensor(rows, dtype=torch.int64, device=device)

    def _touch_lru(self, mm_hash: str) -> None:
        if mm_hash in self._lru:
            self._lru.move_to_end(mm_hash)
        else:
            self._lru[mm_hash] = None

    def _current_bytes(self) -> int:
        return sum(self._mm_hash_sizes.values())

    def _close_shm_async(self, shm: shared_memory.SharedMemory) -> None:
        self._shm_close_executor.submit(shm.close)

    def shutdown(self) -> None:
        self._shm_close_executor.shutdown(wait=False)
        self._shm_load_executor.shutdown(wait=False)

    def __del__(self):
        try:
            self.shutdown()
        except Exception as exc:
            logger.warning("Failed to shutdown EC shared memory connector: %s", exc, exc_info=True)

    def bind_connector_metadata(self, connector_metadata: ECConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        # Hit detection (has_cache_item) runs in the scheduler process, while eviction
        # runs on the producer worker. start_load_caches only runs on the consumer
        # (which never evicts), so without this the producer's LRU only advances on
        # save. Refresh it here from the dedicated hit-list so hot entries survive
        # pressure.
        if (self.is_producer and self.role == ECConnectorRole.WORKER
                and connector_metadata is not None):
            hit_hashes = getattr(connector_metadata, "mm_hashes_hit", None) or []
            refreshed = 0
            for mm_hash in hit_hashes:
                # Only move entries the producer actually holds; skip phantoms so we
                # never insert LRU/sizes tracking for segments this worker did not create.
                if mm_hash in self._lru:
                    self._lru.move_to_end(mm_hash)
                    refreshed += 1
            if refreshed:
                logger.debug("EC shm producer refreshed LRU for %d hit hash(es).", refreshed)

    def _evict_if_needed(self, extra_bytes: int) -> None:
        if not self.is_producer or self.role != ECConnectorRole.WORKER:
            return
        while self._current_bytes() + extra_bytes > self._max_bytes and self._lru:
            evict_hash, _ = self._lru.popitem(last=False)
            logger.warning(f"use total memory: {self._current_bytes() / GB}, shm evict_hash:{evict_hash}")
            self._unlink_shm(evict_hash)

    def _unlink_shm(self, mm_hash: str) -> None:
        try:
            shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash))
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logger.debug(f"SHM {mm_hash} already unlinked.")
        except Exception as ex:
            logger.error(f"Unlink failed for {mm_hash}: {ex}", exc_info=True)
        self._mm_hash_sizes.pop(mm_hash, None)
        # Decrease the reference count. If the reference count is 1, the value is cleared to avoid negative values.
        current_count = self._mm_hash_refcounts.get(mm_hash, 0)
        if current_count <= 1:
            self._mm_hash_refcounts.pop(mm_hash, None)
        else:
            self._mm_hash_refcounts[mm_hash] = current_count - 1

    def _load_single_cache(
        self, mm_hash: str
    ) -> _CacheLoadResult:
        try:
            shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash))
        except FileNotFoundError:
            logger.warning("EC shared memory miss for hash %s", mm_hash)
            return _CacheLoadResult(None, False, "missing", None)
        except OSError as exc:
            logger.warning(f"Failed to access shared memory for hash {mm_hash}: {exc}")
            return _CacheLoadResult(None, False, str(exc), None)

        payload_view = None
        try:
            payload_view = memoryview(shm.buf)[:shm.size]
            tensor = self._deserialize_cache(payload_view)
            return _CacheLoadResult(tensor, True, None, shm)
        except ValueError as exc:
            logger.warning("Failed to deserialize EC shared memory for hash %s: %s", mm_hash, exc)
            return _CacheLoadResult(None, False, str(exc), shm)
        except Exception:
            if payload_view is not None:
                payload_view.release()
                payload_view = None
            self._close_shm_async(shm)
            raise
        finally:
            if payload_view is not None:
                payload_view.release()

    def _load_caches_from_shm(
        self, mm_hashes: list[str],
    ) -> dict[str, _CacheLoadResult]:
        if not mm_hashes:
            return {}
        if len(mm_hashes) == 1:
            mm_hash = mm_hashes[0]
            logger.debug(f"start load caches for mm_hash:{mm_hash}")
            return {mm_hash: self._load_single_cache(mm_hash)}

        load_futures = {}
        for mm_hash in mm_hashes:
            logger.debug(f"start load caches for mm_hash:{mm_hash}")
            load_futures[mm_hash] = self._shm_load_executor.submit(
                self._load_single_cache, mm_hash,
            )

        load_results = {}
        first_error = None
        for mm_hash, future in load_futures.items():
            try:
                load_results[mm_hash] = future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            for result in load_results.values():
                if result.shm is not None:
                    self._close_shm_async(result.shm)
            raise first_error
        return load_results

    def _broadcast_cache_metadata(
        self,
        mm_hashes: list[str],
        load_results: dict[str, _CacheLoadResult],
        tp_rank: int,
    ) -> list[list[int]]:
        meta_device = current_platform.device_type
        if tp_rank == 0:
            meta_tensor = self._build_cache_meta_tensor(
                mm_hashes, load_results, meta_device,
            )
        else:
            meta_tensor = torch.empty(
                (len(mm_hashes), META_WIDTH),
                dtype=torch.int64,
                device=meta_device,
            )
        dist.broadcast(
            meta_tensor,
            src=self._tp_src_rank(),
            group=self._tp_device_group(),
        )
        if meta_tensor.device.type != "cpu":
            meta_tensor = meta_tensor.cpu()
        return meta_tensor.tolist()

    def _broadcast_and_store_caches(
        self,
        encoder_cache,
        mm_hashes: list[str],
        load_results: dict[str, _CacheLoadResult],
        meta_rows: list[list[int]] | None,
        tp_rank: int,
        use_tp_broadcast: bool,
    ) -> None:
        for idx, mm_hash in enumerate(mm_hashes):
            logger.debug(f"start load caches for mm_hash:{mm_hash}")
            result = load_results.get(mm_hash)
            tensor = result.tensor if result is not None else None

            if use_tp_broadcast:
                if meta_rows is None:
                    raise RuntimeError("EC cache metadata rows are missing for TP broadcast.")
                load_ok, shape, dtype, device_type = self._decode_cache_meta_row(
                    meta_rows[idx]
                )
                if not load_ok:
                    continue
                if tp_rank != 0:
                    if shape is None or dtype is None or device_type is None:
                        raise ValueError("Incomplete EC cache metadata for TP broadcast.")
                    tensor = torch.empty(
                        shape,
                        dtype=dtype,
                        device=device_type,
                    )
                if tensor is None:
                    raise RuntimeError(
                        f"EC cache tensor is missing for successful hash {mm_hash}."
                    )
                dist.broadcast(
                    tensor,
                    src=self._tp_src_rank(),
                    group=self._tp_device_group(),
                )

            if tensor is None:
                continue
            encoder_cache[mm_hash] = tensor
            self._touch_lru(mm_hash)
            self._mm_hash_refcounts[mm_hash] = self._mm_hash_refcounts.get(mm_hash, 0) + 1

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        metadata: ECConnectorMetadata | None = self._get_connector_metadata()
        if metadata is None:
            logger.warning("In connector.start_load_caches, but the connector metadata is None")
            return
        tp_rank = get_tensor_model_parallel_rank()
        use_tp_broadcast = (
            get_tensor_model_parallel_world_size() > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        mm_hashes_to_load = [mm_hash for mm_hash in metadata.mm_hashes if mm_hash not in encoder_cache]
        should_load_from_shm = (not use_tp_broadcast) or tp_rank == 0
        load_results = {}

        # Ensure opened shm handles are scheduled for close even if broadcast or cache insert fails.
        try:
            if should_load_from_shm:
                load_results = self._load_caches_from_shm(mm_hashes_to_load)

            meta_rows = None
            if use_tp_broadcast and mm_hashes_to_load:
                meta_rows = self._broadcast_cache_metadata(
                    mm_hashes_to_load, load_results, tp_rank,
                )
            self._broadcast_and_store_caches(
                encoder_cache,
                mm_hashes_to_load,
                load_results,
                meta_rows,
                tp_rank,
                use_tp_broadcast,
            )
        finally:
            for result in load_results.values():
                if result.shm is not None:
                    self._close_shm_async(result.shm)

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        if not self.is_producer or self.role != ECConnectorRole.WORKER:
            return
        # 只在rank0保存
        if get_tensor_model_parallel_rank() != 0:
            return

        cpu_tensor, meta_payload, total_size_bytes = self._serialize_cache(encoder_cache[mm_hash])
        self._evict_if_needed(total_size_bytes)

        shm = None
        try:
            shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash), create=True,
                                            size=total_size_bytes)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash))
            if shm.size != total_size_bytes:
                try:
                    shm.close()
                except Exception as exc:
                    logger.warning(
                        "Failed to close stale EC shared memory for hash %s: %s",
                        mm_hash, exc, exc_info=True)
                shm.unlink()
                try:
                    shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash), create=True,
                                                    size=total_size_bytes)
                except Exception as exc:
                    logger.warning(f"Failed to create shared memory for hash {mm_hash}: {exc}")
                    return
        except Exception as exc:
            logger.warning(f"Failed to create shared memory for hash {mm_hash}: {exc}")
            return

        try:
            data_start = BYTE_LENGTH + len(meta_payload)
            shm.buf[:BYTE_LENGTH] = len(meta_payload).to_bytes(BYTE_LENGTH, "little")
            shm.buf[BYTE_LENGTH:data_start] = meta_payload
            if cpu_tensor.numel() > 0:
                shm_tensor = torch.frombuffer(memoryview(shm.buf)[data_start:], dtype=cpu_tensor.dtype,
                                              count=cpu_tensor.numel()).reshape(cpu_tensor.shape)
                shm_tensor.copy_(cpu_tensor)
                del shm_tensor
            self._mm_hash_sizes[mm_hash] = total_size_bytes
            self._touch_lru(mm_hash)
            self._mm_hash_refcounts[mm_hash] = self._mm_hash_refcounts.get(mm_hash, 0) + 1
        finally:
            logger.debug(f"mm_hash:{mm_hash} shm size:{shm.size} {shm.size / MB:.2f} MB")
            shm.close()

        logger.debug("Stored EC shared memory raw tensor cache for hash %s", mm_hash)

    def has_cache_item(self, identifier: str) -> bool:
        # The scheduler asks per media item since v0.25.1; before that it
        # passed the whole request and got one bool per feature. Registering
        # the hash for loading stays on this path, as it did then, even when
        # no cache turns out to exist.
        self._mm_hashes_need_loads.add(identifier)
        logger.debug(f"ecconnector has caches:{identifier}")
        try:
            shared_memory.SharedMemory(name=_mm_hash_to_sha256(identifier)).close()
            self._mm_hashes_hit.add(identifier)
            return True
        except FileNotFoundError:
            return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        self._mm_hashes_need_loads.add(mm_hash)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        meta = ECSharedMemoryConnectorMetadata()
        for mm_hash in self._mm_hashes_need_loads:
            meta.add_mm_hash(mm_hash)
        for mm_hash in self._mm_hashes_hit:
            meta.add_mm_hash_hit(mm_hash)
        self._mm_hashes_need_loads.clear()
        self._mm_hashes_hit.clear()
        return meta
