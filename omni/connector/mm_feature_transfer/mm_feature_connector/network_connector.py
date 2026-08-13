# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC, abstractmethod
import asyncio
import msgpack
import numpy as np
import torch
import threading
from typing import Dict, Any, Tuple, List, Optional
import zmq
import zmq.asyncio

from vllm.distributed import parallel_state
from vllm.logger import init_logger

from ..config import ConnectorConfig
from .base import BaseMMFeatureConnector
from .disk_connector import DiskMMFeatureConnector


logger = init_logger(__name__)


class NetworkSender(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Initialize socket and connect to the receiver endpoint."""
        pass

    @abstractmethod
    async def send(
        self,
        hash_key: str,
        payload: bytes,
    ) -> bool:
        """
        Send a multipart message asynchronously with non-blocking semantics.
        Returns True if accepted by ZMQ's internal queue, False if dropped.
        """
        pass


class NetworkReceiver(ABC):
    @abstractmethod
    def bind(self) -> bool:
        """Bind socket to the local endpoint."""
        pass

    @abstractmethod
    async def receive(self, timeout_ms: int | None = None) -> tuple[str, bytes] | None:
        """Poll for the next message. Returns None on timeout."""
        pass


class ZmqAsyncPubSender(NetworkSender):
    """
    A non-blocking ZMQ Pub sender using zmq.asyncio.
    Drops messages immediately if the local queue (SNDHWM) is full.
    """

    def __init__(self, config: dict):
        """
        Config expects:
            - "endpoint": str, e.g., "tcp://192.168.1.10:5555"
            - "high_water_mark": int (optional, default 1000)
            - "linger_ms": int (optional, default 0)
        """
        self.endpoints = config["endpoints"]
        if not isinstance(self.endpoints, list):
            self.endpoints = [self.endpoints]
        self.hwm = config.get("high_water_mark", 1000)
        self.linger = config.get("linger_ms", 0)

        self.context: zmq.asyncio.Context | None = None
        self.socket: zmq.asyncio.Socket | None = None
        self._connected = False

    def connect(self) -> None:
        """Create the context and socket, then connect."""
        if self._connected:
            return

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, self.hwm)
        self.socket.setsockopt(zmq.LINGER, self.linger)
        for endpoint in self.endpoints:
            self.socket.connect(endpoint)
        self._connected = True

    async def send(
        self,
        hash_key: str,
        payload: bytes,
    ) -> bool:
        """
        Attempt to send a multipart message.
        Returns False immediately if the queue is full or a network error occurs.
        """
        if not self._connected or not self.socket:
            logger.warning("send() called before connect()")
            return False

        try:
            await self.socket.send_multipart(
                [hash_key.encode("utf-8"), payload],
                flags=zmq.NOBLOCK,
            )
            return True

        except zmq.Again:
            # SNDHWM reached; drop the message.
            logger.info(
                "ZmqAsyncSubSender[DROP] key=%s, size=%s, reason=QUEUE_FULL", 
                hash_key, len(payload)
            )
            return False

        except zmq.ZMQError as e:
            # Network errors, socket closed, etc.
            logger.info(
                "ZmqAsyncSubSender[DROP] key=%s, size=%s, reason=NETWORK_ERROR_%s", 
                hash_key, len(payload), e.errno
            )
            logger.warning("ZMQ send error: %s", e)
            return False

    def close(self) -> None:
        """Cleanly shut down the socket and context."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self._connected = False


class ZmqAsyncSubReceiver(NetworkReceiver):
    """
    A ZMQ Sub receiver using zmq.asyncio with poller support.
    """

    def __init__(self, config: dict):
        """
        Config expects:
            - "bind_endpoint": str, e.g., "tcp://0.0.0.0:5555"
            - "high_water_mark": int (optional, default 1000)
            - "linger_ms": int (optional, default 0)
            - "poll_timeout_ms": int (optional, default 100)
        """
        self.bind_endpoint = config["bind_endpoint"]
        self.hwm = config.get("high_water_mark", 1000)
        self.linger = config.get("linger_ms", 0)
        self.poll_timeout = config.get("poll_timeout_ms", 100)
        self.topic_filter = config.get("topic_filter", "")  # "" means subscribe to ALL

        self.context: zmq.asyncio.Context | None = None
        self.socket: zmq.asyncio.Socket | None = None
        self.poller: zmq.asyncio.Poller | None = None
        self._bound = False

    def bind(self) -> bool:
        """Create the context, socket, bind, and register the poller."""
        if self._bound:
            return True

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, self.hwm)
        self.socket.setsockopt(zmq.LINGER, self.linger)
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic_filter.encode("utf-8"))

        try:
            self.socket.bind(self.bind_endpoint)
            self._bound = True
            
            self.poller = zmq.asyncio.Poller()
            self.poller.register(self.socket, zmq.POLLIN)

            return True
        except zmq.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                self.close()
                return False
            else:
                raise zmq.ZMQError(f"Failed to bind to {self.bind_endpoint}: {e}") from e

    async def receive(self) -> tuple[str, bytes] | None:
        """
        Poll the socket for incoming data.
        Returns (hash_key, payload) or None if the poll times out.
        """
        if not self._bound or not self.socket or not self.poller:
            logger.warning("receive() called before bind()")
            return None

        # Poll with the specified timeout.
        events = await self.poller.poll(timeout=self.poll_timeout)

        if not events:
            return None  # Timeout

        try:
            # Recv multipart: [hash_bytes, payload]
            hash_bytes, payload = await self.socket.recv_multipart()
            return hash_bytes.decode("utf-8"), payload
        except zmq.ZMQError as e:
            logger.warning(f"Receive error: {e}")
            return None

    def close(self) -> None:
        """Cleanly shut down the socket and context."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self._bound = False


class NetworkTransportFactory:
    @staticmethod
    def create(config: ConnectorConfig):
        if config.is_producer:
            sender_config = {
                "endpoints": config.remote_endpoints,
                "high_water_mark": config.high_water_mark,
                "linger_ms": config.linger_ms
            }
            return ZmqAsyncPubSender(sender_config)
        
        else:
            receiver_config = {
                "bind_endpoint": config.local_endpoint,  
                "high_water_mark": config.high_water_mark,
                "linger_ms": config.linger_ms,
                "topic_filter": config.topic_filter,
                "poll_timeout_ms": config.poll_timeout_ms
            }
            return ZmqAsyncSubReceiver(receiver_config)


class NetworkMMFeatureConnector(BaseMMFeatureConnector):
    def __init__(self, config: ConnectorConfig, local_conn: BaseMMFeatureConnector) -> None:
        super().__init__(config)
        self.local_conn = local_conn
        self.transport = NetworkTransportFactory.create(config)

        if isinstance(self.transport, NetworkSender):
            self.transport.connect()
            logger.info("NetworkMMFeatureConnector(sender) connected to %s.", self.transport.endpoints)
        if isinstance(self.transport, NetworkReceiver):
            self.local_conn.is_consumer = True
            logger.info("NetworkMMFeatureConnector(receiver) attempting to bind...")
            if self.transport.bind():
                logger.info(
                    "NetworkMMFeatureConnector(receiver) successfully bound to %s.", 
                    self.transport.bind_endpoint
                )
                orchestrator_thread = threading.Thread(target=lambda: asyncio.run(self.run_forever()), daemon=True)
                orchestrator_thread.start()
            else:
                logger.warning("The port is already bound by another NetworkMMFeatureConnector(receiver).")
                self.transport = None

    def has_item(self, mm_hash: str) -> bool:
        return self.local_conn.has_item(mm_hash)

    def load(self, mm_hash: str):
        return self.local_conn.load(mm_hash)

    async def save(
        self, 
        mm_hash: str, 
        metadata: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
        serialized_updates: List[Dict[str, Any]],
    ) -> None:
        local_save = asyncio.create_task(
            self.local_conn.save(mm_hash, metadata, tensors, serialized_updates)
        )
        local_save.add_done_callback(
            lambda fut: logger.error("MMFeatureConnector local save failed for %s: %s", mm_hash, fut.exception())
            if fut.exception() else None
        )
        # await local_save
        
        # 直接打包（pack_complex_payload 内部会将 tensor 转 numpy）
        payload_bytes = pack_complex_payload(
            metadata=metadata,
            tensors=tensors,
            serialized_state=serialized_updates
        )
        success = await self.transport.send(mm_hash, payload_bytes)

        logger.debug("MMFeatureConnector sent MM feature %s to %s: %s", mm_hash, self.transport.endpoints, success)

    async def run_forever(self):
        """Start the main receive loop."""
        self._running = True
        logger.info("MMFeatureConnector listening for incoming MM features")

        def done_callback(fut):
            try:
                fut.result() 
            except Exception as e:
                logger.error("MMFeatureConnector local save failed for key %s: %s", hash_key, e)

        while self._running:
            result = await self.transport.receive()
            if result:
                hash_key, payload = result
                metadata, tensors, state = unpack_complex_payload(payload)
                logger.debug("MMFeatureConnector received MM feature %s", hash_key)

                # Hand off to the storage handler.
                # If handler.handle is CPU-heavy, consider using:
                #   await asyncio.get_running_loop().run_in_executor(None, self.handler.handle, hash_key, payload)
                # For now, we call it directly (assuming it's async or fast).
                task = asyncio.create_task(self.local_conn.save(hash_key, metadata, tensors, state))
                task.add_done_callback(done_callback)
            # else: timeout, just loop again (yield control via await above)

        logger.info("NetworkMMFeatureConnector stopped listening")
 
    async def stop(self):
        """Gracefully stop the loop and release resources."""
        self._running = False

        current_task = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks() if t is not current_task]
        await asyncio.gather(*pending, return_exceptions=True)

        if self.transport:
            self.transport.close()

    def __del__(self):
        asyncio.run(self.stop())
        logger.info("NetworkMMFeatureConnector shut down gracefully.")


class DummyMMFeatureConnector(BaseMMFeatureConnector):
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)

    def has_item(self, mm_hash: str) -> bool:
        logger.debug("MM feature %s does not exist in dummy connector", mm_hash)
        return False

    async def save(
        self,
        mm_hash: str,
        metadata: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
        serialized_updates: List[Dict[str, Any]],
    ) -> None:
        logger.debug("Dummy save MMFeature %s", mm_hash)

    def load(self, mm_hash: str):
        logger.debug("Dummy load MMFeature %s", mm_hash)


def pack_complex_payload(
    metadata: Dict[str, Any],
    tensors: Dict[str, torch.Tensor], 
    serialized_state: List[Dict[str, Any]],
) -> bytes:
    """
    Pack complex multimodal data into a single bytes payload.
    """
    # Handle tensors: move to CPU, convert to numpy, get raw bytes + metadata
    tensor_payloads = {}
    tensor_meta = {}
    for name, tensor in tensors.items():
        # Ensure CPU and detached from graph
        if tensor.dtype == torch.bfloat16:
            np_array = tensor.detach().cpu().float().numpy()
        else:
            np_array = tensor.detach().cpu().numpy()
        tensor_payloads[name] = np_array.tobytes()
        tensor_meta[name] = {
            "shape": np_array.shape,
            "dtype": str(np_array.dtype),
            "tensor_device": str(tensor.device),
            "tensor_dtype": str(tensor.dtype).replace("torch.", ""),
        }

    outer_package = {
        "metadata": metadata,
        "tensor_meta": tensor_meta,
        "tensor_data": tensor_payloads, 
        "state_bytes": serialized_state,
    }

    return msgpack.packb(outer_package, use_bin_type=True)


def unpack_complex_payload(raw_bytes: bytes) -> Tuple[Dict, Dict[str, torch.Tensor], List[Dict]]:
    """
    Reverse the packing. Returns (metadata, reconstructed_tensors, state).
    """
    outer = msgpack.unpackb(raw_bytes, raw=False)

    metadata = outer["metadata"]
    tensor_meta = outer["tensor_meta"]

    tensors = {}
    for name, raw_data in outer["tensor_data"].items():
        meta = tensor_meta[name]
        np_array = np.frombuffer(raw_data, dtype=np.dtype(meta["dtype"]))
        np_array = np_array.reshape(meta["shape"])
        tensors[name] = (
            torch.from_numpy(np_array)
            .to(dtype=getattr(torch, meta["tensor_dtype"]), device=meta["tensor_device"])
        )

    state = outer["state_bytes"]

    return metadata, tensors, state
