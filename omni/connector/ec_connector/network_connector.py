# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import queue
import socket as socket_lib
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import lz4.block
import msgspec
import torch
import torch.distributed as dist
import zmq

from vllm import envs
from vllm.config import ECTransferConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Per-request performance diagnostics (Serialized/Deserialized/overhead/etc).
# Off by default (production); set EC_LOG_PERF=1 to enable (benchmark).
EC_LOG_PERF = os.environ.get("EC_LOG_PERF", "") == "1"


def _perf_log(msg: str, *args) -> None:
    if EC_LOG_PERF:
        logger.info(msg, *args)

# ---------------------------------------------------------------------------
# ZMQ multipart protocol constants
# Request:  [identity, empty_delimiter, payload]  (ROUTER side, >= 3 frames)
# Response: [empty_delimiter, header, payload?]      (DEALER side)
# header = msgpack(ECNetworkResponse{status, mode, dtype_code, shape}); the raw
# compressed cache bytes travel as a separate payload frame only for status=OK,
# so the 34 MiB blob never goes through msgpack encode/decode.
# ---------------------------------------------------------------------------
ZMQ_EMPTY_FRAME = b""

SERVER_REQUEST_MIN_FRAMES = 3
SERVER_IDENTITY_FRAME_INDEX = 0
SERVER_DELIMITER_FRAME_INDEX = 1
SERVER_PAYLOAD_FRAME_INDEX = 2

CLIENT_REPLY_MIN_FRAMES = 2
CLIENT_DELIMITER_FRAME_INDEX = 0
CLIENT_HEADER_FRAME_INDEX = 1
CLIENT_PAYLOAD_FRAME_INDEX = 2

# Chunked payload format: [chunk count][chunk length][chunk data]...
# Use network byte order so the header is stable across hosts.
CACHE_META_LENGTH_STRUCT = struct.Struct("!I")
MB = 1024 * 1024
GB = 1024 * MB
EC_LOAD_TIMEOUT_SECONDS = 120.0

# ZMQ/TCP socket buffer sizes for transferring large compressed cache payloads.
# Defaults are small (~128-256 KiB) and cap single-connection throughput well
# below a 10GbE link; 16 MiB lets TCP grow its window to fill the pipe.
EC_ZMQ_SNDBUF = 16 * MB
EC_ZMQ_RCVBUF = 16 * MB
# Hard cap on the number of pooled async DEALER sockets, to bound FD/memory in
# runaway-concurrency scenarios. 32 >> typical batch size (8), so it never
# throttles normal loads; checkout blocks (back-pressure) only past this.
EC_ASYNC_SOCKET_CAP = 32

# Split large payloads into independent LZ4 blocks so several cores can
# compress/decompress concurrently. lz4.block is a GIL-releasing C extension,
# so a ThreadPoolExecutor yields real parallelism. The chunked wire format is:
# [num_chunks:u32] repeated [compressed_len:u32][compressed_block]. Whether the
# payload is "plain" (single lz4 block) or "chunked" is signalled by the
# `mode` field in the ECNetworkResponse header, so no in-band magic is needed.
LZ4_CHUNK_BYTES = 8 * MB
LZ4_MAX_WORKERS = 8

# Lazily-created shared pools; sized to the process CPU affinity. Stored in
# one-element lists so the module-global slot can be mutated in place.
_LZ4_COMPRESS_POOL: list[ThreadPoolExecutor | None] = [None]
_LZ4_DECOMPRESS_POOL: list[ThreadPoolExecutor | None] = [None]
_LZ4_POOL_LOCK = threading.Lock()


def _lz4_worker_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
        cpus = len(affinity) if affinity else (os.cpu_count() or 1)
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(LZ4_MAX_WORKERS, cpus))


def _get_lz4_pool(slot: list[ThreadPoolExecutor | None]) -> ThreadPoolExecutor:
    pool = slot[0]
    if pool is not None:
        return pool
    with _LZ4_POOL_LOCK:
        if slot[0] is None:
            slot[0] = ThreadPoolExecutor(max_workers=_lz4_worker_count())
        return slot[0]


def _compress_payload_parallel(raw_payload: bytes) -> tuple[bytes, str]:
    # Returns (compressed_bytes, mode). mode ("plain"|"chunked") is carried in
    # the response header so the decoder knows the framing without any magic.
    # Single-core (or tiny payload): emit a plain lz4 block to avoid the
    # chunk-framing and join overhead that would only regress latency.
    if _lz4_worker_count() <= 1 or len(raw_payload) <= LZ4_CHUNK_BYTES:
        return lz4.block.compress(raw_payload), "plain"
    view = memoryview(raw_payload)
    chunks = [view[i:i + LZ4_CHUNK_BYTES] for i in range(0, len(raw_payload), LZ4_CHUNK_BYTES)]
    pool = _get_lz4_pool(_LZ4_COMPRESS_POOL)
    blocks = list(pool.map(lz4.block.compress, chunks))
    parts = [CACHE_META_LENGTH_STRUCT.pack(len(blocks))]
    for block in blocks:
        parts.append(CACHE_META_LENGTH_STRUCT.pack(len(block)))
        parts.append(block)
    return b"".join(parts), "chunked"


def _decompress_payload_parallel(payload: bytes, mode: str) -> bytes:
    if mode != "chunked":
        return lz4.block.decompress(payload)
    header_size = CACHE_META_LENGTH_STRUCT.size
    if len(payload) < header_size:
        raise ValueError("EC cache payload is missing the chunk count header.")
    (count,) = CACHE_META_LENGTH_STRUCT.unpack_from(payload)
    offset = header_size
    blocks: list[bytes] = []
    for _ in range(count):
        if offset + header_size > len(payload):
            raise ValueError("EC cache chunk length header is truncated.")
        (size,) = CACHE_META_LENGTH_STRUCT.unpack_from(payload, offset)
        offset += header_size
        if offset + size > len(payload):
            raise ValueError("EC cache chunk data is truncated.")
        blocks.append(payload[offset:offset + size])
        offset += size
    if len(blocks) == 1:
        return lz4.block.decompress(blocks[0])
    pool = _get_lz4_pool(_LZ4_DECOMPRESS_POOL)
    decompressed = list(pool.map(lz4.block.decompress, blocks))
    return b"".join(decompressed)


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


# ---------------------------------------------------------------------------
# Protocol data types
# ---------------------------------------------------------------------------
class STATUS(StrEnum):
    # Response statuses
    HIT = "hit"
    OK = "ok"
    MISS = "miss"
    ERROR = "error"
    # Request commands
    HAS = "has"
    GET = "get"


class ECNetworkRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,
):
    command: STATUS
    mm_hash: str


class ECNetworkResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    dict=True,
):
    status: STATUS
    # Cache metadata, sent in the small header frame (not inside the compressed
    # payload). Present only for status=OK.
    mode: str | None = None
    dtype_code: int | None = None
    shape: tuple[int, ...] | None = None


# ---------------------------------------------------------------------------
# ZMQ request helper
# ---------------------------------------------------------------------------
class ZmqRequestHelper:
    """Encapsulates ZMQ DEALER request/response framing and codec logic."""

    _encoder = msgspec.msgpack.Encoder()

    @staticmethod
    def encode_request(command: STATUS, mm_hash: str) -> bytes:
        return msgspec.msgpack.encode(ECNetworkRequest(command=command, mm_hash=mm_hash))

    @staticmethod
    def decode_response(reply) -> tuple[ECNetworkResponse, memoryview | None]:
        # reply (DEALER side) = [empty, header, payload?] where the payload frame
        # is present only for status=OK. With recv(copy=False) these are Frame
        # objects; .buffer gives a zero-copy memoryview into ZMQ's buffer (the
        # memoryview keeps the Frame alive, so it stays valid through decompress).
        if len(reply) < CLIENT_REPLY_MIN_FRAMES:
            return ECNetworkResponse(status=STATUS.ERROR), None
        if bytes(reply[CLIENT_DELIMITER_FRAME_INDEX]) != ZMQ_EMPTY_FRAME:
            return ECNetworkResponse(status=STATUS.ERROR), None
        try:
            header = msgspec.msgpack.decode(
                reply[CLIENT_HEADER_FRAME_INDEX].buffer, type=ECNetworkResponse,
            )
        except msgspec.DecodeError:
            return ECNetworkResponse(status=STATUS.ERROR), None
        payload = None
        if len(reply) > CLIENT_PAYLOAD_FRAME_INDEX and header.status == STATUS.OK:
            payload = reply[CLIENT_PAYLOAD_FRAME_INDEX].buffer
        return header, payload

    @staticmethod
    def send_and_recv(
        socket: zmq.Socket, payload: bytes,
    ) -> tuple[ECNetworkResponse, memoryview | None, float, float]:
        # Returns (response, payload, transfer_ms, decode_ms) so callers can
        # break the network round-trip out of the total cache-load latency.
        # recv(copy=False) avoids copying the 34 MiB response frame.
        try:
            transfer_start = time.perf_counter()
            socket.send_multipart([ZMQ_EMPTY_FRAME, payload])
            reply = socket.recv_multipart(copy=False)
            transfer_ms = (time.perf_counter() - transfer_start) * 1000
        except zmq.Again:
            return ECNetworkResponse(status=STATUS.ERROR), None, 0.0, 0.0
        decode_start = time.perf_counter()
        response, resp_payload = ZmqRequestHelper.decode_response(reply)
        decode_ms = (time.perf_counter() - decode_start) * 1000
        return response, resp_payload, transfer_ms, decode_ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_local_ip() -> str:
    _socket = socket_lib.socket(socket_lib.AF_INET, socket_lib.SOCK_DGRAM)
    try:
        _socket.connect(("8.8.8.8", 80))
        ip = _socket.getsockname()[0]
    finally:
        _socket.close()
    return ip


@dataclass
class ECNetworkConnectorMetadata(ECConnectorMetadata):
    mm_hashes: list[str]
    mm_hash_endpoints: dict[str, str]

    def __init__(self) -> None:
        self.mm_hashes = []
        self.mm_hash_endpoints = {}

    def add_mm_hash(self, mm_hash: str, endpoint: str | None = None) -> None:
        self.mm_hashes.append(mm_hash)
        if endpoint:
            self.mm_hash_endpoints[mm_hash] = endpoint


@dataclass(frozen=True)
class _NetworkCacheLoadResult:
    tensor: torch.Tensor | None
    ok: bool
    error: str | None
    payload_bytes: int = 0
    timed_out: bool = False


@dataclass(frozen=True)
class _StoredCache:
    # A saved cache in the producer's LRU store. The compressed tensor bytes
    # travel as a raw ZMQ frame; dtype/shape/mode ride in the response header.
    payload: bytes
    dtype_code: int
    shape: tuple[int, ...]
    mode: str


# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_cache(tensor: torch.Tensor) -> _StoredCache:
    total_start = time.perf_counter()
    cpu_copy_start = time.perf_counter()
    cpu_tensor = tensor.detach().cpu()
    cpu_copy_ms = (time.perf_counter() - cpu_copy_start) * 1000

    metadata_start = time.perf_counter()
    dtype_code = DTYPE_TO_CODE[cpu_tensor.dtype]
    shape = tuple(cpu_tensor.shape)
    metadata_encode_ms = (time.perf_counter() - metadata_start) * 1000

    data_encode_start = time.perf_counter()
    data_bytes = (
        cpu_tensor.reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    data_encode_ms = (time.perf_counter() - data_encode_start) * 1000

    compress_start = time.perf_counter()
    compressed_payload, mode = _compress_payload_parallel(data_bytes)
    compress_ms = (time.perf_counter() - compress_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    _perf_log(
        "Serialized EC cache: cpu_copy_ms=%.3f metadata_encode_ms=%.3f "
        "data_encode_ms=%.3f lz4_compress_ms=%.3f total_ms=%.3f "
        "raw_size=%.2f MB compressed_size=%.2f MB "
        "compression_ratio=%.4f",
        cpu_copy_ms,
        metadata_encode_ms,
        data_encode_ms,
        compress_ms,
        total_ms,
        len(data_bytes) / MB,
        len(compressed_payload) / MB,
        (len(compressed_payload) / len(data_bytes)) if data_bytes else 0.0,
    )
    return _StoredCache(
        payload=compressed_payload,
        dtype_code=dtype_code,
        shape=shape,
        mode=mode,
    )


def _deserialize_cache(
    payload: bytes,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    mode: str,
    device: str | None = None,
    net_transfer_ms: float = 0.0,
    net_decode_ms: float = 0.0,
) -> torch.Tensor:
    total_start = time.perf_counter()
    decompress_start = time.perf_counter()
    raw_payload = _decompress_payload_parallel(payload, mode)
    decompress_ms = (time.perf_counter() - decompress_start) * 1000

    metadata_decode_ms = 0.0

    tensor_rebuild_start = time.perf_counter()
    numel = 1
    for dim in shape:
        numel *= dim
    expected_data_bytes = numel * torch.empty((), dtype=dtype).element_size()
    data_bytes = memoryview(raw_payload)
    if len(data_bytes) != expected_data_bytes:
        raise ValueError(
            "EC cache tensor data size mismatch: "
            f"expected={expected_data_bytes}, actual={len(data_bytes)}."
        )
    if expected_data_bytes == 0:
        ec_cache = torch.empty(shape, dtype=dtype)
    else:
        ec_cache = torch.frombuffer(
            data_bytes,
            dtype=dtype,
        ).reshape(shape)
    tensor_rebuild_ms = (time.perf_counter() - tensor_rebuild_start) * 1000

    device_move_start = time.perf_counter()
    target_device = device if device is not None else current_platform.device_type
    if ec_cache.device.type != target_device:
        ec_cache = ec_cache.to(target_device)
    device_move_ms = (time.perf_counter() - device_move_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    _perf_log(
        "Deserialized EC cache: net_transfer_ms=%.3f net_decode_ms=%.3f "
        "lz4_decompress_ms=%.3f "
        "metadata_decode_ms=%.3f tensor_rebuild_ms=%.3f "
        "device_move_ms=%.3f total_ms=%.3f "
        "compressed_size=%.2f MB raw_size=%.2f MB",
        net_transfer_ms,
        net_decode_ms,
        decompress_ms,
        metadata_decode_ms,
        tensor_rebuild_ms,
        device_move_ms,
        total_ms,
        len(payload) / MB,
        len(raw_payload) / MB,
    )
    return ec_cache


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------
class ECNetworkConnector(ECConnectorBase):
    """Network-backed EC connector using ZMQ request/response protocol.

    ``load_ec_async`` controls whether multiple network GETs are submitted to
    the thread pool or loaded serially. ``start_load_caches`` remains blocking.

    With ``ec_zmq_tp_leader_only=true`` (default), only TP rank 0 performs
    network GETs. Load metadata is broadcast once as a fixed-width tensor,
    followed by direct tensor broadcasts for successful cache entries.
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        transfer_config = vllm_config.ec_transfer_config
        self.tp_rank = 0
        self.dp_rank = 0
        self.tp_size = 0
        self.dp_size = 0
        self._init_rank_and_size(role, vllm_config)
        self.ec_async_flag = transfer_config.get_from_extra_config(
            "load_ec_async", True,
        )

        scheme = transfer_config.get_from_extra_config("ec_network_scheme", "tcp")
        if self.is_producer:
            current_rank_port = (
                transfer_config.ec_port + self.dp_rank * self.tp_size + self.tp_rank
            )
            self._producer_endpoint = make_zmq_path(
                scheme=scheme, host=_get_local_ip(), port=current_rank_port,
            )
        else:
            self._init_consumer_endpoints(scheme, transfer_config)

        self._context = zmq.Context.instance()
        self._cache_store: OrderedDict[str, _StoredCache] = OrderedDict()
        self._cache_store_bytes: int = 0
        self._cache_store_max_bytes: int = transfer_config.get_from_extra_config(
            "ec_cache_max_gb", 50,
        ) * GB
        logger.info(f"ecconnector max bytes:{self._cache_store_max_bytes / GB}GB")
        self._cache_lock = threading.Lock()
        self._mm_hashes_need_loads: set[str] = set()
        self._mm_hash_endpoints: dict[str, str] = {}
        self._client_sockets: dict[str, zmq.Socket] = {}
        self._client_lock = threading.Lock()
        self._server_socket: zmq.Socket | None = None
        self._server_thread: threading.Thread | None = None
        self._server_running = threading.Event()

        # Batch loading executors. Per-call futures/results stay local.
        self._ec_load_executor = ThreadPoolExecutor(max_workers=16)
        self._deserialize_executor = ThreadPoolExecutor(max_workers=4)

        # Reusable DEALER sockets for async GETs, pooled per endpoint so we avoid
        # the ~5 ms create+connect+close cost per request. queue.Queue is
        # thread-safe; the dict of pools is guarded by _async_pool_lock.
        self._async_socket_pools: dict[str, "queue.Queue[zmq.Socket]"] = {}
        self._async_pool_lock = threading.Lock()
        self._async_sockets_live = 0

        self._start_server_and_client()

    # -----------------------------------------------------------------------
    # Initialization helpers
    # -----------------------------------------------------------------------
    def _init_rank_and_size(self, role: ECConnectorRole, vllm_config: VllmConfig) -> None:
        if role != ECConnectorRole.SCHEDULER:
            self.dp_rank = vllm_config.parallel_config.data_parallel_rank
            self.dp_size = vllm_config.parallel_config.data_parallel_size
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def _init_consumer_endpoints(
        self, scheme: str, transfer_config: ECTransferConfig,
    ) -> None:
        ec_ports = transfer_config.get_from_extra_config("ec_ports", None)
        ec_ips = [ip.strip() for ip in transfer_config.ec_ip.split(",")]

        if ec_ports is None:
            rank_port = (
                transfer_config.ec_port + self.dp_rank * self.tp_size + self.tp_rank
            )
            self._consumer_endpoints: list[str] = [
                make_zmq_path(scheme=scheme, host=ip, port=rank_port) for ip in ec_ips
            ]
            return

        if isinstance(ec_ports, str):
            ec_ports = [int(p.strip()) for p in ec_ports.split(",") if p.strip()]
        elif isinstance(ec_ports, tuple):
            ec_ports = list(ec_ports)

        if len(ec_ports) != len(ec_ips):
            raise ValueError("The length of ec_ports does not match the length of ec_ips")

        self._consumer_endpoints = []
        for idx, ec_port in enumerate(ec_ports):
            rank_port = ec_port + self.dp_rank * self.tp_size + self.tp_rank
            self._consumer_endpoints.append(
                make_zmq_path(scheme=scheme, host=ec_ips[idx], port=rank_port),
            )

    # -----------------------------------------------------------------------
    # TP leader mode helpers
    # -----------------------------------------------------------------------
    def _ec_zmq_tp_leader_only_enabled(self) -> bool:
        tc = self._vllm_config.ec_transfer_config
        if tc is None:
            return False
        return bool(tc.get_from_extra_config("ec_zmq_tp_leader_only", True))

    def _ec_use_tp_leader_zmq(self) -> bool:
        """Single ZMQ GET on TP rank 0 followed by TP tensor broadcast."""
        if self.is_producer or self.tp_size <= 1:
            return False
        if not self._ec_zmq_tp_leader_only_enabled():
            return False
        return dist.is_available() and dist.is_initialized()

    # -----------------------------------------------------------------------
    # Cache store LRU helpers (must be called under self._cache_lock)
    # -----------------------------------------------------------------------
    def _touch_cache(self, mm_hash: str) -> None:
        if mm_hash in self._cache_store:
            self._cache_store.move_to_end(mm_hash)

    def _evict_cache_if_needed(self, extra_bytes: int) -> None:
        while (
            self._cache_store_bytes + extra_bytes > self._cache_store_max_bytes
            and self._cache_store
        ):
            evict_hash, evict_payload = self._cache_store.popitem(last=False)
            self._cache_store_bytes -= len(evict_payload.payload)
            logger.info(
                "Evicted cache hash %s (%.2f MB), store_total=%.2f MB",
                evict_hash,
                len(evict_payload.payload) / MB,
                self._cache_store_bytes / MB,
            )

    # -----------------------------------------------------------------------
    # Server (producer worker)
    # -----------------------------------------------------------------------
    def _start_server_and_client(self) -> None:
        if self.is_producer and self.role == ECConnectorRole.WORKER:
            self._endpoint: str = self._producer_endpoint
            self._start_server()
        if self.is_producer and self.role == ECConnectorRole.SCHEDULER:
            self._endpoint = self._producer_endpoint
            self._ensure_client_socket(self._producer_endpoint)
        if not self.is_producer:
            self._endpoint = self._consumer_endpoints[0]
            self._ensure_client_socket(self._consumer_endpoints)

    def _start_server(self) -> None:
        if self._server_socket is not None:
            return
        self._server_socket = make_zmq_socket(
            self._context, self._endpoint, zmq.ROUTER, bind=True,
        )
        self._server_socket.setsockopt(zmq.RCVTIMEO, envs.VLLM_RPC_TIMEOUT)
        self._server_socket.setsockopt(zmq.SNDBUF, EC_ZMQ_SNDBUF)
        self._server_socket.setsockopt(zmq.RCVBUF, EC_ZMQ_RCVBUF)
        logger.info(
            "ECNetworkConnector server socket buffers: sndbuf=%d rcvbuf=%d",
            self._server_socket.getsockopt(zmq.SNDBUF),
            self._server_socket.getsockopt(zmq.RCVBUF),
        )
        self._server_running.set()
        self._server_thread = threading.Thread(
            target=self._server_loop,
            name=f"ec-net-rank-{self.tp_rank}",
            daemon=True,
        )
        self._server_thread.start()
        logger.info(
            "DP rank %d TP rank %d ECNetworkConnector server listening on %s",
            self.dp_rank, self.tp_rank, self._endpoint,
        )

    def _server_loop(self) -> None:
        if self._server_socket is None:
            return
        poller = zmq.Poller()
        poller.register(self._server_socket, zmq.POLLIN)
        while self._server_running.is_set():
            events = dict(poller.poll(timeout=envs.VLLM_RPC_TIMEOUT))
            if self._server_socket not in events:
                continue
            framed = self._recv_server_message()
            if framed is None:
                continue
            identity, payload = framed
            request = self._decode_server_request(payload)
            if request is None:
                self._send_server_response(
                    identity, ECNetworkResponse(status=STATUS.ERROR), None,
                )
                continue
            response, resp_payload = self._handle_server_request(request)
            self._send_server_response(identity, response, resp_payload)

    def _recv_server_message(self) -> tuple[bytes, bytes] | None:
        if self._server_socket is None:
            return None
        try:
            message = self._server_socket.recv_multipart()
        except zmq.Again:
            return None
        if len(message) < SERVER_REQUEST_MIN_FRAMES:
            return None
        if message[SERVER_DELIMITER_FRAME_INDEX] != ZMQ_EMPTY_FRAME:
            return None
        return message[SERVER_IDENTITY_FRAME_INDEX], message[SERVER_PAYLOAD_FRAME_INDEX]

    @staticmethod
    def _decode_server_request(payload: bytes) -> ECNetworkRequest | None:
        try:
            return msgspec.msgpack.decode(payload, type=ECNetworkRequest)
        except msgspec.DecodeError:
            return None

    def _handle_server_request(
        self, request: ECNetworkRequest,
    ) -> tuple[ECNetworkResponse, bytes | None]:
        if request.command == STATUS.HAS:
            with self._cache_lock:
                hit = request.mm_hash in self._cache_store
            return ECNetworkResponse(status=STATUS.HIT if hit else STATUS.MISS), None

        if request.command == STATUS.GET:
            with self._cache_lock:
                stored = self._cache_store.get(request.mm_hash)
                if stored is not None:
                    self._touch_cache(request.mm_hash)
            if stored is None:
                return ECNetworkResponse(status=STATUS.MISS), None
            return (
                ECNetworkResponse(
                    status=STATUS.OK,
                    mode=stored.mode,
                    dtype_code=stored.dtype_code,
                    shape=stored.shape,
                ),
                stored.payload,
            )

        return ECNetworkResponse(status=STATUS.ERROR), None

    def _send_server_response(
        self, identity: bytes, response: ECNetworkResponse, payload: bytes | None,
    ) -> None:
        if self._server_socket is None:
            return
        frames = [identity, ZMQ_EMPTY_FRAME, msgspec.msgpack.encode(response)]
        if payload is not None:
            frames.append(payload)
        self._server_socket.send_multipart(frames)

    # -----------------------------------------------------------------------
    # Client (consumer / producer scheduler)
    # -----------------------------------------------------------------------
    def _ensure_client_socket(self, endpoint: str | list[str]) -> zmq.Socket | None:
        if isinstance(endpoint, str):
            if endpoint in self._client_sockets:
                return self._client_sockets[endpoint]
            return self._create_client_socket(endpoint)
        if isinstance(endpoint, list):
            for ep in endpoint:
                self._create_client_socket(ep)
            return None
        raise TypeError(f"Invalid endpoint type: {type(endpoint)}. Expected str or list.")

    def _create_client_socket(self, endpoint: str) -> zmq.Socket:
        socket = make_zmq_socket(self._context, endpoint, zmq.DEALER, bind=False)
        socket.setsockopt(zmq.RCVTIMEO, envs.VLLM_RPC_TIMEOUT)
        socket.setsockopt(zmq.SNDTIMEO, envs.VLLM_RPC_TIMEOUT)
        socket.setsockopt(zmq.SNDBUF, EC_ZMQ_SNDBUF)
        socket.setsockopt(zmq.RCVBUF, EC_ZMQ_RCVBUF)
        self._client_sockets[endpoint] = socket
        transfer_role = "Producer" if self.is_producer else "Consumer"
        logger.info(
            "ECNetworkConnector(%s %s) connected to %s "
            "(socket sndbuf=%d rcvbuf=%d)",
            transfer_role, self.role.name.capitalize(), endpoint,
            socket.getsockopt(zmq.SNDBUF), socket.getsockopt(zmq.RCVBUF),
        )
        return socket

    def _resolve_endpoint(self, mm_hash: str) -> str:
        return self._mm_hash_endpoints.get(mm_hash, self._endpoint)

    def _client_request(
        self, command: STATUS, mm_hash: str,
    ) -> tuple[ECNetworkResponse, memoryview | None, float, float]:
        endpoint = self._resolve_endpoint(mm_hash)
        socket = self._ensure_client_socket(endpoint)
        payload = ZmqRequestHelper.encode_request(command, mm_hash)
        with self._client_lock:
            response, resp_payload, transfer_ms, decode_ms = (
                ZmqRequestHelper.send_and_recv(socket, payload)
            )
        if response.status == STATUS.ERROR:
            logger.warning(
                "ZMQ request failed: command=%s mm_hash=%s endpoint=%s",
                command, mm_hash, endpoint,
            )
        return response, resp_payload, transfer_ms, decode_ms

    def _create_async_socket(self, endpoint: str) -> zmq.Socket:
        socket = make_zmq_socket(self._context, endpoint, zmq.DEALER, bind=False)
        socket.setsockopt(zmq.RCVTIMEO, envs.VLLM_RPC_TIMEOUT)
        socket.setsockopt(zmq.SNDTIMEO, envs.VLLM_RPC_TIMEOUT)
        socket.setsockopt(zmq.SNDBUF, EC_ZMQ_SNDBUF)
        socket.setsockopt(zmq.RCVBUF, EC_ZMQ_RCVBUF)
        return socket

    def _checkout_async_socket(self, endpoint: str) -> zmq.Socket:
        with self._async_pool_lock:
            pool = self._async_socket_pools.get(endpoint)
            if pool is None:
                pool = queue.Queue()
                self._async_socket_pools[endpoint] = pool
        # Fast path: reuse an idle socket.
        try:
            return pool.get_nowait()
        except queue.Empty:
            pass
        # No idle socket: create one if under the cap, otherwise block until a
        # worker returns one (back-pressure instead of unbounded growth).
        with self._async_pool_lock:
            if self._async_sockets_live < EC_ASYNC_SOCKET_CAP:
                self._async_sockets_live += 1
                return self._create_async_socket(endpoint)
        return pool.get()

    def _return_async_socket(self, endpoint: str, socket: zmq.Socket) -> None:
        with self._async_pool_lock:
            pool = self._async_socket_pools.get(endpoint)
        if pool is None:
            socket.close(linger=0)
            return
        pool.put(socket)

    def _async_client_request(
        self, command: STATUS, mm_hash: str,
    ) -> tuple[ECNetworkResponse, memoryview | None, float, float]:
        """Thread-safe ZMQ request using pooled per-call sockets (no
        create/close per GET). A socket is used by exactly one worker at a time
        (the pool is a queue), respecting ZMQ's not-thread-safe constraint."""
        endpoint = self._resolve_endpoint(mm_hash)
        payload = ZmqRequestHelper.encode_request(command, mm_hash)
        socket = self._checkout_async_socket(endpoint)
        healthy = True
        try:
            response, resp_payload, transfer_ms, decode_ms = (
                ZmqRequestHelper.send_and_recv(socket, payload)
            )
        except Exception:
            healthy = False
            raise
        finally:
            if healthy:
                self._return_async_socket(endpoint, socket)
            else:
                with self._async_pool_lock:
                    self._async_sockets_live -= 1
                socket.close(linger=0)
        if response.status == STATUS.ERROR:
            logger.warning(
                "ZMQ async request failed: command=%s mm_hash=%s endpoint=%s",
                command, mm_hash, endpoint,
            )
        return response, resp_payload, transfer_ms, decode_ms

    # -----------------------------------------------------------------------
    # Cache load - batched network path
    # -----------------------------------------------------------------------
    @staticmethod
    def _ordered_unique_hashes(mm_hashes: list[str]) -> list[str]:
        return list(dict.fromkeys(mm_hashes))

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
    def _encode_cache_meta_row(
        tensor: torch.Tensor | None, load_ok: bool,
    ) -> list[int]:
        row = [0] * META_WIDTH
        if not load_ok or tensor is None:
            return row

        shape = tuple(tensor.shape)
        if len(shape) > META_MAX_DIMS:
            raise ValueError(
                f"EC network cache tensor dim {len(shape)} exceeds max {META_MAX_DIMS}."
            )
        dtype_code = DTYPE_TO_CODE.get(tensor.dtype)
        if dtype_code is None:
            raise ValueError(f"Unsupported EC network cache dtype: {tensor.dtype}")
        device_code = DEVICE_TYPE_TO_CODE.get(tensor.device.type)
        if device_code is None:
            raise ValueError(
                f"Unsupported EC network cache device type: {tensor.device.type}"
            )

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
        if len(values) != META_WIDTH:
            raise ValueError(f"Invalid EC network cache metadata width: {len(values)}")
        if values[META_OK_IDX] == 0:
            return False, None, None, None

        ndim = int(values[META_NDIM_IDX])
        if ndim < 0 or ndim > META_MAX_DIMS:
            raise ValueError(f"Invalid EC network cache tensor ndim: {ndim}")
        shape = tuple(
            int(values[META_SHAPE_START_IDX + idx]) for idx in range(ndim)
        )
        if any(dim < 0 for dim in shape):
            raise ValueError(f"Invalid EC network cache tensor shape: {shape}")
        dtype_code = int(values[META_DTYPE_IDX])
        dtype = CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError(
                f"Unsupported EC network cache dtype code: {dtype_code}"
            )
        device_code = int(values[META_DEVICE_IDX])
        device_type = CODE_TO_DEVICE_TYPE.get(device_code)
        if device_type is None:
            raise ValueError(
                f"Unsupported EC network cache device code: {device_code}"
            )
        return True, shape, dtype, device_type

    def _get_mm_hashes_to_load(
        self,
        mm_hashes: list[str],
        encoder_cache: dict,
        use_tp_broadcast: bool,
    ) -> list[str]:
        local_need = [1 if mm_hash not in encoder_cache else 0 for mm_hash in mm_hashes]
        if not use_tp_broadcast or not mm_hashes:
            return [
                mm_hash for mm_hash, need_load in zip(mm_hashes, local_need)
                if need_load
            ]

        need_tensor = torch.tensor(
            local_need,
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        dist.all_reduce(
            need_tensor,
            op=dist.ReduceOp.MAX,
            group=self._tp_device_group(),
        )
        if need_tensor.device.type != "cpu":
            need_tensor = need_tensor.cpu()
        return [
            mm_hash for mm_hash, need_load in zip(mm_hashes, need_tensor.tolist())
            if need_load
        ]

    def _load_single_cache(
        self,
        mm_hash: str,
        device: str | None,
        use_async_workers: bool,
    ) -> _NetworkCacheLoadResult:
        try:
            request = self._async_client_request if use_async_workers else self._client_request
            req_start = time.perf_counter()
            response, payload, net_transfer_ms, net_decode_ms = request(STATUS.GET, mm_hash)
            req_overhead_ms = (
                (time.perf_counter() - req_start) * 1000 - net_transfer_ms - net_decode_ms
            )
        except Exception as exc:
            logger.error("EC network GET failed: mm_hash=%s err=%s", mm_hash, exc)
            return _NetworkCacheLoadResult(None, False, str(exc))

        if response.status != STATUS.OK or payload is None:
            error = f"status={response.status}"
            if response.status == STATUS.OK:
                error = "empty payload"
            logger.warning(
                "EC network cache miss/error: mm_hash=%s %s", mm_hash, error,
            )
            return _NetworkCacheLoadResult(None, False, error)

        dtype = CODE_TO_DTYPE.get(response.dtype_code)
        shape = response.shape
        mode = response.mode
        if dtype is None or shape is None or mode is None:
            error = (
                f"incomplete header dtype_code={response.dtype_code} "
                f"shape={shape} mode={mode}"
            )
            logger.warning("EC network cache header invalid: mm_hash=%s %s", mm_hash, error)
            return _NetworkCacheLoadResult(None, False, error)

        try:
            if use_async_workers:
                deserialize_future = self._deserialize_executor.submit(
                    _deserialize_cache, payload, dtype, shape, mode, device,
                    net_transfer_ms, net_decode_ms,
                )
                tensor = deserialize_future.result()
            else:
                tensor = _deserialize_cache(
                    payload, dtype, shape, mode, device=device,
                    net_transfer_ms=net_transfer_ms, net_decode_ms=net_decode_ms,
                )
        except Exception as exc:
            logger.error(
                "EC network cache deserialize failed: mm_hash=%s err=%s",
                mm_hash, exc,
            )
            return _NetworkCacheLoadResult(
                None, False, str(exc), payload_bytes=len(payload),
            )

        post_start = time.perf_counter()
        result = _NetworkCacheLoadResult(
            tensor, True, None, payload_bytes=len(payload),
        )
        post_ms = (time.perf_counter() - post_start) * 1000
        _perf_log(
            "EC load_single overhead: req_overhead_ms=%.3f post_ms=%.3f "
            "hash=%s payload_mb=%.2f",
            req_overhead_ms, post_ms, mm_hash, len(payload) / MB,
        )
        return result

    @staticmethod
    def _timeout_result() -> _NetworkCacheLoadResult:
        return _NetworkCacheLoadResult(
            None, False, "batch timeout", timed_out=True,
        )

    def _load_caches_from_network(
        self,
        mm_hashes: list[str],
        device: str | None,
        timeout: float,
    ) -> tuple[dict[str, _NetworkCacheLoadResult], bool]:
        if not mm_hashes:
            return {}, False

        deadline = time.monotonic() + timeout
        results: dict[str, _NetworkCacheLoadResult] = {}
        if not self.ec_async_flag or len(mm_hashes) == 1:
            for index, mm_hash in enumerate(mm_hashes):
                if time.monotonic() >= deadline:
                    for timed_out_hash in mm_hashes[index:]:
                        results[timed_out_hash] = self._timeout_result()
                    return results, True
                results[mm_hash] = self._load_single_cache(
                    mm_hash, device, use_async_workers=False,
                )
            return results, False

        futures: dict[str, Future[_NetworkCacheLoadResult]] = {
            mm_hash: self._ec_load_executor.submit(
                self._load_single_cache, mm_hash, device, True,
            )
            for mm_hash in mm_hashes
        }

        for index, mm_hash in enumerate(mm_hashes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out_hashes = mm_hashes[index:]
            else:
                try:
                    results[mm_hash] = futures[mm_hash].result(timeout=remaining)
                    continue
                except FutureTimeoutError:
                    timed_out_hashes = mm_hashes[index:]
                except Exception as exc:
                    logger.error(
                        "EC network load worker failed: mm_hash=%s err=%s",
                        mm_hash, exc,
                    )
                    results[mm_hash] = _NetworkCacheLoadResult(
                        None, False, str(exc),
                    )
                    continue

            for timed_out_hash in timed_out_hashes:
                futures[timed_out_hash].cancel()
                results[timed_out_hash] = self._timeout_result()
            logger.error(
                "EC network cache batch timeout after %.1fs, missing=%s",
                timeout, timed_out_hashes,
            )
            return results, True

        return results, False

    def _reuse_or_load_caches(
        self,
        mm_hashes: list[str],
        encoder_cache: dict,
        device: str | None,
        timeout: float,
        use_tp_broadcast: bool,
    ) -> tuple[dict[str, _NetworkCacheLoadResult], bool]:
        results: dict[str, _NetworkCacheLoadResult] = {}
        hashes_to_fetch = []
        target_device = device or current_platform.device_type
        for mm_hash in mm_hashes:
            if use_tp_broadcast and mm_hash in encoder_cache:
                tensor = encoder_cache[mm_hash]
                try:
                    if tensor.device.type != target_device:
                        tensor = tensor.to(target_device)
                    results[mm_hash] = _NetworkCacheLoadResult(
                        tensor, True, None,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to prepare local EC cache for TP broadcast: "
                        "mm_hash=%s err=%s",
                        mm_hash, exc,
                    )
                    results[mm_hash] = _NetworkCacheLoadResult(
                        None, False, str(exc),
                    )
                continue
            hashes_to_fetch.append(mm_hash)

        fetched, timed_out = self._load_caches_from_network(
            hashes_to_fetch, device, timeout,
        )
        results.update(fetched)
        return results, timed_out

    def _build_cache_meta_tensor(
        self,
        mm_hashes: list[str],
        load_results: dict[str, _NetworkCacheLoadResult],
        device: str,
    ) -> torch.Tensor:
        rows = []
        for mm_hash in mm_hashes:
            result = load_results.get(mm_hash)
            if result is None:
                result = _NetworkCacheLoadResult(None, False, "missing result")
            try:
                rows.append(self._encode_cache_meta_row(result.tensor, result.ok))
            except ValueError as exc:
                logger.warning("Skip EC network cache %s: %s", mm_hash, exc)
                rows.append([0] * META_WIDTH)
                continue
            if not result.ok:
                logger.warning(
                    "Skip EC network cache %s: %s", mm_hash, result.error,
                )
        return torch.tensor(rows, dtype=torch.int64, device=device)

    def _broadcast_cache_metadata(
        self,
        mm_hashes: list[str],
        load_results: dict[str, _NetworkCacheLoadResult],
    ) -> list[list[int]]:
        meta_device = current_platform.device_type
        if self.tp_rank == 0:
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
        encoder_cache: dict,
        mm_hashes: list[str],
        load_results: dict[str, _NetworkCacheLoadResult],
        meta_rows: list[list[int]],
    ) -> int:
        success_count = 0
        for index, mm_hash in enumerate(mm_hashes):
            load_ok, shape, dtype, device_type = self._decode_cache_meta_row(
                meta_rows[index]
            )
            if not load_ok:
                continue

            result = load_results.get(mm_hash)
            tensor = result.tensor if result is not None else None
            if self.tp_rank != 0:
                if shape is None or dtype is None or device_type is None:
                    raise ValueError("Incomplete EC network cache metadata.")
                tensor = torch.empty(shape, dtype=dtype, device=device_type)
            if tensor is None:
                raise RuntimeError(
                    f"EC network cache tensor is missing for successful hash {mm_hash}."
                )

            dist.broadcast(
                tensor,
                src=self._tp_src_rank(),
                group=self._tp_device_group(),
            )
            encoder_cache[mm_hash] = tensor
            success_count += 1
        return success_count

    @staticmethod
    def _store_load_results(
        encoder_cache: dict,
        mm_hashes: list[str],
        load_results: dict[str, _NetworkCacheLoadResult],
    ) -> int:
        success_count = 0
        for mm_hash in mm_hashes:
            result = load_results.get(mm_hash)
            if result is None or not result.ok:
                continue
            if result.tensor is None:
                raise RuntimeError(
                    f"EC network cache tensor is missing for successful hash {mm_hash}."
                )
            encoder_cache[mm_hash] = result.tensor
            success_count += 1
        return success_count

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        total_start = time.perf_counter()
        metadata: ECConnectorMetadata | None = self._get_connector_metadata()
        if metadata is None:
            logger.warning("Connector metadata is None, skipping start_load_caches")
            return
        if not isinstance(metadata, ECNetworkConnectorMetadata):
            raise TypeError("metadata must be ECNetworkConnectorMetadata instance")

        device = kwargs.get("device", None)
        timeout = float(kwargs.get("timeout", EC_LOAD_TIMEOUT_SECONDS))
        mm_hashes = self._ordered_unique_hashes(metadata.mm_hashes)
        for mm_hash in mm_hashes:
            endpoint = metadata.mm_hash_endpoints.get(mm_hash)
            if endpoint is not None:
                self._mm_hash_endpoints[mm_hash] = endpoint

        use_tp_broadcast = self._ec_use_tp_leader_zmq()
        prep_ms = (time.perf_counter() - total_start) * 1000
        mm_hashes_to_load = self._get_mm_hashes_to_load(
            mm_hashes, encoder_cache, use_tp_broadcast,
        )
        load_results: dict[str, _NetworkCacheLoadResult] = {}
        timed_out = False
        success_count = 0
        load_ms = 0.0
        store_ms = 0.0
        try:
            if (not use_tp_broadcast) or self.tp_rank == 0:
                load_start = time.perf_counter()
                load_results, timed_out = self._reuse_or_load_caches(
                    mm_hashes_to_load,
                    encoder_cache,
                    device,
                    timeout,
                    use_tp_broadcast,
                )
                load_ms = (time.perf_counter() - load_start) * 1000

            store_start = time.perf_counter()
            if use_tp_broadcast and mm_hashes_to_load:
                meta_rows = self._broadcast_cache_metadata(
                    mm_hashes_to_load, load_results,
                )
                success_count = self._broadcast_and_store_caches(
                    encoder_cache,
                    mm_hashes_to_load,
                    load_results,
                    meta_rows,
                )
            else:
                success_count = self._store_load_results(
                    encoder_cache, mm_hashes_to_load, load_results,
                )
            store_ms = (time.perf_counter() - store_start) * 1000
        finally:
            _perf_log(
                "EC network start_load_caches finished: requested=%d success=%d "
                "failed=%d tp_rank=%d timed_out=%s elapsed_ms=%.3f "
                "prep_ms=%.3f load_ms=%.3f store_ms=%.3f",
                len(mm_hashes_to_load),
                success_count,
                len(mm_hashes_to_load) - success_count,
                self.tp_rank,
                timed_out,
                (time.perf_counter() - total_start) * 1000,
                prep_ms,
                load_ms,
                store_ms,
            )

    # -----------------------------------------------------------------------
    # Cache save / query
    # -----------------------------------------------------------------------
    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        if not self.is_producer or self.role != ECConnectorRole.WORKER:
            return
        # 存储到rank0即可
        if self.tp_rank != 0:
            return
        stored = _serialize_cache(encoder_cache[mm_hash])
        payload_size = len(stored.payload)
        with self._cache_lock:
            if mm_hash in self._cache_store:
                old = self._cache_store.pop(mm_hash)
                self._cache_store_bytes -= len(old.payload)
            self._evict_cache_if_needed(payload_size)
            self._cache_store[mm_hash] = stored
            self._cache_store_bytes += payload_size
        _perf_log(
            "Stored encoder cache for hash %s (%.2f MB, store_total=%.2f MB)",
            mm_hash,
            payload_size / MB,
            self._cache_store_bytes / MB,
        )

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        # has_cache_item only receives an identifier, so the endpoints the
        # encoder passed on the request are recorded here instead. The
        # scheduler calls this first, before it asks about individual items.
        if self.is_producer:
            return True
        ec_transfer_params = request.ec_transfer_params
        if ec_transfer_params is None or ec_transfer_params.get("mm_hash_endpoints") is None:
            logger.error("Cannot obtain ec_transfer_params from encoder.")
            return True
        mm_hash_endpoints = ec_transfer_params.get("mm_hash_endpoints")
        for feature in request.mm_features:
            mm_hash = feature.identifier
            if mm_hash in mm_hash_endpoints:
                self._mm_hash_endpoints[mm_hash] = mm_hash_endpoints.get(mm_hash)
        return True

    def has_cache_item(self, identifier: str) -> bool:
        # The scheduler asks per media item since v0.25.1; before that it
        # passed the whole request and got one bool per feature. Registering
        # the hash for loading stays on this path, as it did then, even when
        # no endpoint turns out to be known.
        self._mm_hashes_need_loads.add(identifier)
        if self.is_producer:
            return False
        return identifier in self._mm_hash_endpoints

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        meta = ECNetworkConnectorMetadata()
        for mm_hash in self._mm_hashes_need_loads:
            endpoint = (
                self._endpoint if self.is_producer
                else self._mm_hash_endpoints.get(mm_hash)
            )
            meta.add_mm_hash(mm_hash, endpoint)
        self._mm_hashes_need_loads.clear()
        return meta

    def update_connector_output(self, connector_output):
        return

    def update_state_after_alloc(self, request: "Request", index: int):
        return

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    def __del__(self) -> None:
        self._ec_load_executor.shutdown(wait=False)
        self._deserialize_executor.shutdown(wait=False)
        for socket in self._client_sockets.values():
            socket.close(linger=0)
        for pool in self._async_socket_pools.values():
            while True:
                try:
                    pool.get_nowait().close(linger=0)
                except queue.Empty:
                    break
        if self._server_socket is not None:
            self._server_running.clear()
            self._server_socket.close(linger=0)
