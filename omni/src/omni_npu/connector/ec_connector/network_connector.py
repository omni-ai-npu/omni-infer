# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import socket as socket_lib
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

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

# ---------------------------------------------------------------------------
# ZMQ multipart protocol constants
# Request:  [identity, empty_delimiter, payload]  (ROUTER side, >= 3 frames)
# Response: [empty_delimiter, payload]             (DEALER side, >= 2 frames)
# ---------------------------------------------------------------------------
ZMQ_EMPTY_FRAME = b""

SERVER_REQUEST_MIN_FRAMES = 3
SERVER_IDENTITY_FRAME_INDEX = 0
SERVER_DELIMITER_FRAME_INDEX = 1
SERVER_PAYLOAD_FRAME_INDEX = 2

CLIENT_REPLY_MIN_FRAMES = 2
CLIENT_DELIMITER_FRAME_INDEX = 0
CLIENT_PAYLOAD_FRAME_INDEX = 1

EC_CACHE_KEY = "ec_cache"
MB = 1024 * 1024
GB = 1024 * MB


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
    payload: bytes | None = None


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
    def decode_response(reply: list[bytes]) -> ECNetworkResponse:
        if (
            len(reply) < CLIENT_REPLY_MIN_FRAMES
            or reply[CLIENT_DELIMITER_FRAME_INDEX] != ZMQ_EMPTY_FRAME
        ):
            return ECNetworkResponse(status=STATUS.ERROR)
        try:
            return msgspec.msgpack.decode(
                reply[CLIENT_PAYLOAD_FRAME_INDEX], type=ECNetworkResponse,
            )
        except msgspec.DecodeError:
            return ECNetworkResponse(status=STATUS.ERROR)

    @staticmethod
    def send_and_recv(socket: zmq.Socket, payload: bytes) -> ECNetworkResponse:
        try:
            socket.send_multipart([ZMQ_EMPTY_FRAME, payload])
            reply = socket.recv_multipart()
        except zmq.Again:
            return ECNetworkResponse(status=STATUS.ERROR)
        return ZmqRequestHelper.decode_response(reply)


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


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_cache(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save({EC_CACHE_KEY: tensor.detach().cpu()}, buffer)
    return buffer.getvalue()


def _deserialize_cache(payload: bytes, device: str | None = None) -> torch.Tensor:
    buffer = io.BytesIO(payload)
    map_device = device if device is not None else current_platform.device_type
    data = torch.load(buffer, map_location=map_device)
    ec_cache = data[EC_CACHE_KEY]
    if device is not None and ec_cache.device.type != device:
        ec_cache = ec_cache.to(device)
    return ec_cache


# ---------------------------------------------------------------------------
# Main connector
# ---------------------------------------------------------------------------
class ECNetworkConnector(ECConnectorBase):
    """Network-backed EC connector using ZMQ request/response protocol.

    Sync path (load_ec_async=false):
      Only TP rank 0 does ZMQ GET, then broadcast_tensor_dict to other ranks.

    Async path (load_ec_async=true):
      When ec_zmq_tp_leader_only=true (default) and tp_size > 1, only rank 0
      submits ZMQ GET tasks to the thread pool.  The broadcast to other TP
      ranks happens in wait_for_pending_loads() on the main thread where
      torch.distributed collectives are safe.
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
        self._cache_store: OrderedDict[str, bytes] = OrderedDict()
        self._cache_store_bytes: int = 0
        self._cache_store_max_bytes: int = transfer_config.get_from_extra_config(
            "ec_cache_max_gb", 10,
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

        # Async loading state
        self._pending_ec_loads: set[str] = set()
        self._finished_ec_loads: OrderedDict[str, None] = OrderedDict()
        self._failed_ec_loads: OrderedDict[str, None] = OrderedDict()
        self._history_set_max_size: int = 100_000
        self._ec_load_lock = threading.Lock()
        self._ec_load_executor = ThreadPoolExecutor(max_workers=16)
        self._deserialize_executor = ThreadPoolExecutor(max_workers=4)
        self._max_concurrent_ec_loads = 32
        self._tp_broadcast_queue: set[str] = set()

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
        """Single ZMQ GET on tp rank 0 + TP broadcast (sync path only)."""
        if self.is_producer or self.tp_size <= 1:
            return False
        if not self._ec_zmq_tp_leader_only_enabled():
            return False
        if self.ec_async_flag:
            return False
        return torch.distributed.is_initialized()

    def _ec_use_tp_leader_zmq_async(self) -> bool:
        """Only rank 0 does ZMQ GET; broadcast in wait_for_pending_loads."""
        if self.is_producer or self.tp_size <= 1:
            return False
        if not self._ec_zmq_tp_leader_only_enabled():
            return False
        if not self.ec_async_flag:
            return False
        return torch.distributed.is_initialized()

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
            self._cache_store_bytes -= len(evict_payload)
            logger.info(
                "Evicted cache hash %s (%.2f MB), store_total=%.2f MB",
                evict_hash,
                len(evict_payload) / MB,
                self._cache_store_bytes / MB,
            )

    def _trim_history_set(self, od: OrderedDict, max_size: int | None = None) -> None:
        limit = max_size if max_size is not None else self._history_set_max_size
        while len(od) > limit:
            od.popitem(last=False)

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
                self._send_server_response(identity, ECNetworkResponse(status=STATUS.ERROR))
                continue
            self._send_server_response(identity, self._handle_server_request(request))

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

    def _handle_server_request(self, request: ECNetworkRequest) -> ECNetworkResponse:
        if request.command == STATUS.HAS:
            with self._cache_lock:
                hit = request.mm_hash in self._cache_store
            return ECNetworkResponse(status=STATUS.HIT if hit else STATUS.MISS)

        if request.command == STATUS.GET:
            with self._cache_lock:
                cache_payload = self._cache_store.get(request.mm_hash)
                if cache_payload is not None:
                    self._touch_cache(request.mm_hash)
            if cache_payload is None:
                return ECNetworkResponse(status=STATUS.MISS)
            return ECNetworkResponse(status=STATUS.OK, payload=cache_payload)

        return ECNetworkResponse(status=STATUS.ERROR)

    def _send_server_response(self, identity: bytes, response: ECNetworkResponse) -> None:
        if self._server_socket is None:
            return
        self._server_socket.send_multipart(
            [identity, ZMQ_EMPTY_FRAME, msgspec.msgpack.encode(response)],
        )

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
        self._client_sockets[endpoint] = socket
        transfer_role = "Producer" if self.is_producer else "Consumer"
        logger.info(
            "ECNetworkConnector(%s %s) connected to %s",
            transfer_role, self.role.name.capitalize(), endpoint,
        )
        return socket

    def _resolve_endpoint(self, mm_hash: str) -> str:
        return self._mm_hash_endpoints.get(mm_hash, self._endpoint)

    def _client_request(self, command: STATUS, mm_hash: str) -> ECNetworkResponse:
        endpoint = self._resolve_endpoint(mm_hash)
        socket = self._ensure_client_socket(endpoint)
        payload = ZmqRequestHelper.encode_request(command, mm_hash)
        with self._client_lock:
            response = ZmqRequestHelper.send_and_recv(socket, payload)
        if response.status == STATUS.ERROR:
            logger.warning(
                "ZMQ request failed: command=%s mm_hash=%s endpoint=%s",
                command, mm_hash, endpoint,
            )
        return response

    def _async_client_request(self, command: STATUS, mm_hash: str) -> ECNetworkResponse:
        """Thread-safe ZMQ request using a per-call socket (no shared lock)."""
        endpoint = self._resolve_endpoint(mm_hash)
        payload = ZmqRequestHelper.encode_request(command, mm_hash)
        socket = make_zmq_socket(self._context, endpoint, zmq.DEALER, bind=False)
        socket.setsockopt(zmq.RCVTIMEO, envs.VLLM_RPC_TIMEOUT)
        socket.setsockopt(zmq.SNDTIMEO, envs.VLLM_RPC_TIMEOUT)
        try:
            response = ZmqRequestHelper.send_and_recv(socket, payload)
        finally:
            socket.close(linger=0)
        if response.status == STATUS.ERROR:
            logger.warning(
                "ZMQ async request failed: command=%s mm_hash=%s endpoint=%s",
                command, mm_hash, endpoint,
            )
        return response

    # -----------------------------------------------------------------------
    # Cache load – sync path
    # -----------------------------------------------------------------------
    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        if metadata is None:
            logger.warning("Connector metadata is None, skipping start_load_caches")
            return
        if not isinstance(metadata, ECNetworkConnectorMetadata):
            raise TypeError("metadata must be ECNetworkConnectorMetadata instance")
        device = kwargs.get("device", None)

        if not self.ec_async_flag:
            self._sync_load_caches(device, encoder_cache, metadata)
            return

        self._async_submit_loads(device, encoder_cache, metadata)
        self.wait_for_pending_loads(encoder_cache)

    def _sync_load_caches(
        self, device: str | None, encoder_cache: dict, metadata: ECNetworkConnectorMetadata,
    ) -> None:
        for mm_hash in metadata.mm_hashes:
            if mm_hash in metadata.mm_hash_endpoints:
                self._mm_hash_endpoints[mm_hash] = metadata.mm_hash_endpoints[mm_hash]

            if self._ec_use_tp_leader_zmq():
                self._sync_load_with_tp_broadcast(mm_hash, encoder_cache, device)
                continue

            if mm_hash in encoder_cache:
                continue
            response = self._client_request(STATUS.GET, mm_hash)
            if response.status == STATUS.OK and response.payload is not None:
                encoder_cache[mm_hash] = _deserialize_cache(response.payload, device=device)
                logger.info(
                    "Loaded encoder cache for hash %s (%.2f MB)",
                    mm_hash, len(response.payload) / MB,
                )
            else:
                logger.info("EC cache miss for hash %s", mm_hash)

    def _sync_load_with_tp_broadcast(
        self, mm_hash: str, encoder_cache: dict, device: str | None,
    ) -> None:
        tp = get_tp_group()
        local_miss = 1 if mm_hash not in encoder_cache else 0
        need = torch.tensor([local_miss], dtype=torch.int32, device="cpu")
        dist.all_reduce(need, op=dist.ReduceOp.MAX, group=tp.cpu_group)
        if need.item() == 0:
            return

        hit = False
        tensor0: torch.Tensor | None = None
        if self.tp_rank == 0:
            if mm_hash in encoder_cache:
                tensor0 = encoder_cache[mm_hash]
                hit = True
            else:
                response = self._client_request(STATUS.GET, mm_hash)
                if response.status == STATUS.OK and response.payload is not None:
                    tensor0 = _deserialize_cache(response.payload, device=device)
                    hit = True
                else:
                    logger.info("EC cache miss for hash %s", mm_hash)

        hit = tp.broadcast_object(hit if self.tp_rank == 0 else None, src=0)
        if hit:
            to_send = {mm_hash: tensor0} if self.tp_rank == 0 else None
            out = tp.broadcast_tensor_dict(to_send, src=0)
            if out is None or mm_hash not in out:
                raise RuntimeError(f"Failed to broadcast tensor for {mm_hash}")
            encoder_cache[mm_hash] = out[mm_hash]
            logger.info(
                "Loaded encoder cache for hash %s (tp0 zmq + broadcast, %.2f MB)",
                mm_hash, out[mm_hash].nelement() * out[mm_hash].element_size() / MB,
            )

    # -----------------------------------------------------------------------
    # Cache load – async path
    # -----------------------------------------------------------------------
    def _async_submit_loads(
        self, device: str | None, encoder_cache: dict, metadata: ECNetworkConnectorMetadata,
    ) -> None:
        with self._ec_load_lock:
            cached_hashes = set(encoder_cache.keys())
            known_hashes = (
                set(self._finished_ec_loads) | self._pending_ec_loads | cached_hashes
            )
            new_mm_hashes = list(set(metadata.mm_hashes) - known_hashes)

        hashes_to_submit: list[str] = []
        with self._ec_load_lock:
            if new_mm_hashes and len(self._pending_ec_loads) < self._max_concurrent_ec_loads:
                available_slots = self._max_concurrent_ec_loads - len(self._pending_ec_loads)
                hashes_to_submit = new_mm_hashes[:available_slots]
                for mm_hash in hashes_to_submit:
                    self._pending_ec_loads.add(mm_hash)

        if not hashes_to_submit:
            return

        tp_leader_async = self._ec_use_tp_leader_zmq_async()
        logger.info(
            "Async EC load: submitting %d hashes, pending=%d, tp_leader_async=%s, tp_rank=%d",
            len(hashes_to_submit), len(self._pending_ec_loads),
            tp_leader_async, self.tp_rank,
        )

        for mm_hash in hashes_to_submit:
            if mm_hash in metadata.mm_hash_endpoints:
                self._mm_hash_endpoints[mm_hash] = metadata.mm_hash_endpoints[mm_hash]
            if tp_leader_async and self.tp_rank != 0:
                continue
            self._ec_load_executor.submit(
                self._async_load_single, encoder_cache, mm_hash, device,
            )

    def _async_load_single(
        self, encoder_cache: dict, mm_hash: str, device: str | None = None,
    ) -> None:
        tp_leader_async = self._ec_use_tp_leader_zmq_async()
        try:
            response = self._async_client_request(STATUS.GET, mm_hash)
            if response.status == STATUS.OK and response.payload is not None:
                payload_mb = len(response.payload) / MB
                deser_device = "cpu" if tp_leader_async else device
                future = self._deserialize_executor.submit(
                    _deserialize_cache, response.payload, deser_device,
                )
                cache_data = future.result()
                logger.info(
                    "Async loaded encoder cache for hash %s (%.2f MB)",
                    mm_hash, payload_mb,
                )
                with self._ec_load_lock:
                    encoder_cache[mm_hash] = cache_data
                    self._finished_ec_loads[mm_hash] = None
                    self._trim_history_set(self._finished_ec_loads)
                    self._pending_ec_loads.discard(mm_hash)
                    if tp_leader_async:
                        self._tp_broadcast_queue.add(mm_hash)
            else:
                with self._ec_load_lock:
                    self._pending_ec_loads.discard(mm_hash)
                    self._failed_ec_loads[mm_hash] = None
                    self._trim_history_set(self._failed_ec_loads)
                logger.warning(
                    "Async EC load miss/error: mm_hash=%s status=%s",
                    mm_hash, response.status,
                )
        except Exception as ex:
            with self._ec_load_lock:
                self._pending_ec_loads.discard(mm_hash)
                self._failed_ec_loads[mm_hash] = None
                self._trim_history_set(self._failed_ec_loads)
            logger.error("Async EC load exception: mm_hash=%s err=%s", mm_hash, ex)

    # -----------------------------------------------------------------------
    # Wait for pending async loads
    # -----------------------------------------------------------------------
    def wait_for_pending_loads(
        self, encoder_cache: dict, timeout: float = 120.0,
    ) -> None:
        if not self.ec_async_flag:
            return

        if self._ec_use_tp_leader_zmq_async():
            self._wait_and_broadcast_pending(encoder_cache, timeout)
            return

        with self._ec_load_lock:
            pending = set(self._pending_ec_loads)
        if not pending:
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            missing = [h for h in pending if h not in encoder_cache]
            if not missing:
                return
            time.sleep(0.001)

        missing = [h for h in pending if h not in encoder_cache]
        logger.error(
            "wait_for_pending_loads TIMEOUT after %.1fs, missing=%s",
            timeout, missing,
        )

    def _wait_and_broadcast_pending(
        self, encoder_cache: dict, timeout: float = 120.0,
    ) -> None:
        """tp_leader_async: rank 0 waits for ZMQ threads, then all ranks
        collectively broadcast tensors via TP group."""
        tp = get_tp_group()
        start = time.monotonic()

        # Phase 1: rank 0 waits for async threads to finish
        if self.tp_rank == 0:
            deadline = start + timeout
            while time.monotonic() < deadline:
                with self._ec_load_lock:
                    still_pending = len(self._pending_ec_loads)
                if still_pending == 0:
                    break
                time.sleep(0.001)
            with self._ec_load_lock:
                hashes_to_broadcast = sorted(self._tp_broadcast_queue)

        # Phase 2: broadcast hash list to all ranks
        hash_list = tp.broadcast_object(
            hashes_to_broadcast if self.tp_rank == 0 else None, src=0,
        )
        if not hash_list:
            with self._ec_load_lock:
                self._pending_ec_loads.clear()
            return

        # Phase 3: broadcast each tensor from rank 0
        target_device = current_platform.device_type
        for mm_hash in hash_list:
            to_send = None
            if self.tp_rank == 0:
                npu_tensor = encoder_cache[mm_hash].to(target_device)
                to_send = {mm_hash: npu_tensor}
            out = tp.broadcast_tensor_dict(to_send, src=0)
            if out is None or mm_hash not in out:
                raise RuntimeError(f"Failed to broadcast tensor for {mm_hash}")
            encoder_cache[mm_hash] = out[mm_hash]

        # Phase 4: clean up
        with self._ec_load_lock:
            for mm_hash in hash_list:
                self._finished_ec_loads[mm_hash] = None
                self._tp_broadcast_queue.discard(mm_hash)
                self._pending_ec_loads.discard(mm_hash)
            self._trim_history_set(self._finished_ec_loads)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "TP broadcast done: count=%d elapsed_ms=%.1f tp_rank=%d",
            len(hash_list), elapsed_ms, self.tp_rank,
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
        payload = _serialize_cache(encoder_cache[mm_hash])
        payload_size = len(payload)
        with self._cache_lock:
            if mm_hash in self._cache_store:
                old = self._cache_store.pop(mm_hash)
                self._cache_store_bytes -= len(old)
            self._evict_cache_if_needed(payload_size)
            self._cache_store[mm_hash] = payload
            self._cache_store_bytes += payload_size
        logger.info(
            "Stored encoder cache for hash %s (%.2f MB, store_total=%.2f MB)",
            mm_hash,
            payload_size / MB,
            self._cache_store_bytes / MB,
        )

    def has_caches(self, request: "Request") -> list[bool]:
        result: list[bool] = []
        for feature in request.mm_features:
            mm_hash = feature.identifier
            self._mm_hashes_need_loads.add(mm_hash)
            if not self.is_producer:
                ec_transfer_params: ECNetworkConnectorMetadata = request.ec_transfer_params
                if ec_transfer_params is None or ec_transfer_params.get("mm_hash_endpoints") is None:
                    logger.error("Cannot obtain ec_transfer_params from encoder.")
                    result.append(False)
                else:
                    mm_hash_endpoints = ec_transfer_params.get("mm_hash_endpoints")
                    self._mm_hash_endpoints[mm_hash] = mm_hash_endpoints.get(mm_hash)
                    result.append(True)
            else:
                result.append(False)
        return result

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
        if self._server_socket is not None:
            self._server_running.clear()
            self._server_socket.close(linger=0)
