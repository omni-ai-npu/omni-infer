# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ECNetworkConnector."""

import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import Mock, patch, MagicMock, PropertyMock

import msgspec
import pytest
import torch
import zmq

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.output import SchedulerOutput

from omni_npu.connector.ec_connector.network_connector import (
    STATUS,
    ZMQ_EMPTY_FRAME,
    SERVER_REQUEST_MIN_FRAMES,
    CLIENT_REPLY_MIN_FRAMES,
    CLIENT_HEADER_FRAME_INDEX,
    CLIENT_PAYLOAD_FRAME_INDEX,
    MB,
    GB,
    EC_ASYNC_SOCKET_CAP,
    META_WIDTH,
    META_OK_IDX,
    META_NDIM_IDX,
    META_DTYPE_IDX,
    META_DEVICE_IDX,
    META_SHAPE_START_IDX,
    ECNetworkRequest,
    ECNetworkResponse,
    ECNetworkConnectorMetadata,
    ECNetworkConnector,
    ZmqRequestHelper,
    _get_local_ip,
    _serialize_cache,
    _deserialize_cache,
    _compress_payload_parallel,
    _decompress_payload_parallel,
    _StoredCache,
    _NetworkCacheLoadResult,
    DTYPE_TO_CODE,
    DEVICE_TYPE_TO_CODE,
    CODE_TO_DTYPE,
    CODE_TO_DEVICE_TYPE,
)


# =================== Helper Classes =================== #

class _Frame:
    """Minimal stand-in for a pyzmq Frame (exposes a zero-copy .buffer view)."""

    def __init__(self, data: bytes):
        self.buffer = memoryview(data)


def _make_stored_cache(payload=b"data", dtype_code=2, shape=(4, 64), mode="plain"):
    """Build a _StoredCache for tests that only exercise store access paths."""
    return _StoredCache(payload=payload, dtype_code=dtype_code, shape=shape, mode=mode)


def _all_reduce_fill_one(t, **kw):
    """Mock dist.all_reduce side_effect that fills t with 1 (someone needs it)."""
    return t.fill_(1)


def _all_reduce_fill_zero(t, **kw):
    """Mock dist.all_reduce side_effect that fills t with 0 (no rank needs it)."""
    return t.fill_(0)


def _broadcast_passthrough(t, src=None, **kw):
    """Mock dist.broadcast side_effect that returns t unchanged."""
    return t


def _run_serial_load(connector, hashes):
    """Configure the serial load path and run _load_caches_from_network on CPU."""
    connector.ec_async_flag = False
    tensor = torch.randn(2, 32)
    stored = _serialize_cache(tensor)
    resp = ECNetworkResponse(
        status=STATUS.OK, mode=stored.mode,
        dtype_code=stored.dtype_code, shape=stored.shape,
    )
    connector._client_request = Mock(return_value=(resp, stored.payload, 0.1, 0.1))

    with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
        m.device_type = 'cpu'
        return connector._load_caches_from_network(hashes, "cpu", 10.0)


class MockRequest:
    """Mock Request object for testing."""

    def __init__(self, request_id: str, mm_hashes: list, ec_transfer_params=None):
        self.request_id = request_id
        self.mm_features = []
        self.ec_transfer_params = ec_transfer_params
        for mm_hash in mm_hashes:
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=0, length=100),
            )
            self.mm_features.append(feature)


# =================== Helper Functions =================== #

def make_mock_vllm_config(is_producer=True, ec_ip="127.0.0.1", ec_port=5000,
                          extra_config_overrides=None):
    """Create a mock VllmConfig for testing."""
    defaults = {
        "ec_network_scheme": "tcp",
        "ec_ports": None,
        "load_ec_async": False,
        "ec_cache_max_gb": 10,
        "ec_zmq_tp_leader_only": True,
    }
    if extra_config_overrides:
        defaults.update(extra_config_overrides)

    config = Mock()
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.is_ec_producer = is_producer
    config.ec_transfer_config.get_from_extra_config = Mock(
        side_effect=lambda key, default=None: defaults.get(key, default)
    )
    config.ec_transfer_config.ec_port = ec_port
    config.ec_transfer_config.ec_ip = ec_ip
    config.parallel_config = Mock()
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    return config


def _make_connector_patches():
    """Return context manager patches required for ECNetworkConnector construction."""
    return [
        patch('omni_npu.connector.ec_connector.network_connector.get_tensor_model_parallel_rank',
              return_value=0),
        patch('omni_npu.connector.ec_connector.network_connector._get_local_ip',
              return_value='127.0.0.1'),
        patch('omni_npu.connector.ec_connector.network_connector.make_zmq_path',
              side_effect=lambda scheme, host, port: f'{scheme}://{host}:{port}'),
        patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
              return_value=MagicMock()),
        patch('zmq.Context.instance', return_value=MagicMock()),
        patch('omni_npu.connector.ec_connector.network_connector.envs',
              VLLM_RPC_TIMEOUT=5000),
    ]


def create_connector(role=ECConnectorRole.SCHEDULER, is_producer=True, vllm_config=None):
    """Create an ECNetworkConnector with all external dependencies mocked."""
    if vllm_config is None:
        vllm_config = make_mock_vllm_config(is_producer=is_producer)
    patches = _make_connector_patches()
    started = [p.start() for p in patches]
    try:
        connector = ECNetworkConnector(vllm_config=vllm_config, role=role)
    finally:
        for p in patches:
            p.stop()
    connector._server_running.clear()
    return connector


# =================== Fixtures =================== #

@pytest.fixture
def producer_scheduler_connector():
    """Producer SCHEDULER connector (no server thread)."""
    return create_connector(role=ECConnectorRole.SCHEDULER, is_producer=True)


@pytest.fixture
def producer_worker_connector():
    """Producer WORKER connector (server thread stopped after init)."""
    return create_connector(role=ECConnectorRole.WORKER, is_producer=True)


@pytest.fixture
def consumer_worker_connector():
    """Consumer WORKER connector."""
    return create_connector(role=ECConnectorRole.WORKER, is_producer=False)


@pytest.fixture
def mock_request_3_hashes():
    return MockRequest("req-001", ["hash1", "hash2", "hash3"])


@pytest.fixture
def mock_request_empty():
    return MockRequest("req-empty", [])


# =================== Tests: Constants =================== #

class TestConstants:
    def test_zmq_empty_frame(self):
        assert ZMQ_EMPTY_FRAME == b""

    def test_server_request_min_frames(self):
        assert SERVER_REQUEST_MIN_FRAMES == 3

    def test_client_reply_min_frames(self):
        assert CLIENT_REPLY_MIN_FRAMES == 2

    def test_mb_gb(self):
        assert MB == 1024 * 1024
        assert GB == 1024 * MB

    def test_frame_indices(self):
        assert CLIENT_HEADER_FRAME_INDEX == 1
        assert CLIENT_PAYLOAD_FRAME_INDEX == 2

    def test_dtype_code_roundtrip(self):
        for dtype, code in DTYPE_TO_CODE.items():
            assert CODE_TO_DTYPE[code] == dtype


# =================== Tests: STATUS Enum =================== #

class TestSTATUS:
    def test_response_statuses_exist(self):
        assert STATUS.HIT == "hit"
        assert STATUS.OK == "ok"
        assert STATUS.MISS == "miss"
        assert STATUS.ERROR == "error"

    def test_request_commands_exist(self):
        assert STATUS.HAS == "has"
        assert STATUS.GET == "get"

    def test_is_str_enum(self):
        assert isinstance(STATUS.HIT, str)
        assert isinstance(STATUS.MISS, str)

    def test_all_values_unique(self):
        values = [s.value for s in STATUS]
        assert len(values) == len(set(values))


# =================== Tests: ECNetworkRequest =================== #

class TestECNetworkRequest:
    def test_create_has_request(self):
        req = ECNetworkRequest(command=STATUS.HAS, mm_hash="abc123")
        assert req.command == STATUS.HAS
        assert req.mm_hash == "abc123"

    def test_create_get_request(self):
        req = ECNetworkRequest(command=STATUS.GET, mm_hash="xyz789")
        assert req.command == STATUS.GET
        assert req.mm_hash == "xyz789"

    def test_msgpack_encode_decode_roundtrip(self):
        req = ECNetworkRequest(command=STATUS.HAS, mm_hash="test_hash")
        encoded = msgspec.msgpack.encode(req)
        decoded = msgspec.msgpack.decode(encoded, type=ECNetworkRequest)
        assert decoded.command == req.command
        assert decoded.mm_hash == req.mm_hash

    def test_msgpack_encode_produces_bytes(self):
        req = ECNetworkRequest(command=STATUS.GET, mm_hash="h1")
        encoded = msgspec.msgpack.encode(req)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0


# =================== Tests: ECNetworkResponse =================== #

class TestECNetworkResponse:
    def test_create_response_without_metadata(self):
        resp = ECNetworkResponse(status=STATUS.MISS)
        assert resp.status == STATUS.MISS
        assert resp.mode is None
        assert resp.dtype_code is None
        assert resp.shape is None

    def test_create_response_with_metadata(self):
        resp = ECNetworkResponse(
            status=STATUS.OK, mode="raw", dtype_code=3, shape=(4, 64),
        )
        assert resp.status == STATUS.OK
        assert resp.mode == "raw"
        assert resp.dtype_code == 3
        assert resp.shape == (4, 64)

    def test_msgpack_roundtrip_no_metadata(self):
        resp = ECNetworkResponse(status=STATUS.HIT)
        encoded = msgspec.msgpack.encode(resp)
        decoded = msgspec.msgpack.decode(encoded, type=ECNetworkResponse)
        assert decoded.status == STATUS.HIT
        assert decoded.mode is None
        assert decoded.dtype_code is None
        assert decoded.shape is None

    def test_msgpack_roundtrip_with_metadata(self):
        resp = ECNetworkResponse(
            status=STATUS.OK, mode="lz4", dtype_code=2, shape=(4,),
        )
        encoded = msgspec.msgpack.encode(resp)
        decoded = msgspec.msgpack.decode(encoded, type=ECNetworkResponse)
        assert decoded.status == STATUS.OK
        assert decoded.mode == "lz4"
        assert decoded.dtype_code == 2
        assert decoded.shape == (4,)

    def test_all_statuses(self):
        for status in [STATUS.HIT, STATUS.OK, STATUS.MISS, STATUS.ERROR]:
            resp = ECNetworkResponse(status=status)
            assert resp.status == status

    def test_no_payload_field(self):
        """Guard: payload field must NOT exist (prevents 34MB msgpack regression)."""
        resp = ECNetworkResponse(status=STATUS.OK)
        assert not hasattr(resp, "payload")


# =================== Tests: ZmqRequestHelper =================== #

class TestZmqRequestHelper:
    def test_encode_request_returns_bytes(self):
        result = ZmqRequestHelper.encode_request(STATUS.HAS, "hash1")
        assert isinstance(result, bytes)

    def test_encode_request_roundtrip(self):
        encoded = ZmqRequestHelper.encode_request(STATUS.GET, "myhash")
        decoded = msgspec.msgpack.decode(encoded, type=ECNetworkRequest)
        assert decoded.command == STATUS.GET
        assert decoded.mm_hash == "myhash"

    def test_decode_response_valid(self):
        resp = ECNetworkResponse(
            status=STATUS.OK, mode="raw", dtype_code=3, shape=(2,),
        )
        encoded = msgspec.msgpack.encode(resp)
        reply = [ZMQ_EMPTY_FRAME, _Frame(encoded), _Frame(b"data")]
        response, payload = ZmqRequestHelper.decode_response(reply)
        assert response.status == STATUS.OK
        assert response.mode == "raw"
        assert payload is not None
        assert bytes(payload) == b"data"

    def test_decode_response_too_few_frames(self):
        response, payload = ZmqRequestHelper.decode_response([b"only_one"])
        assert response.status == STATUS.ERROR
        assert payload is None

    def test_decode_response_non_empty_delimiter(self):
        resp = ECNetworkResponse(status=STATUS.OK)
        encoded = msgspec.msgpack.encode(resp)
        reply = [b"nonempty", _Frame(encoded)]
        response, payload = ZmqRequestHelper.decode_response(reply)
        assert response.status == STATUS.ERROR
        assert payload is None

    def test_decode_response_invalid_msgpack(self):
        reply = [ZMQ_EMPTY_FRAME, _Frame(b"not_valid_msgpack")]
        response, payload = ZmqRequestHelper.decode_response(reply)
        assert response.status == STATUS.ERROR
        assert payload is None

    def test_send_and_recv_success(self):
        mock_socket = MagicMock()
        resp = ECNetworkResponse(status=STATUS.HIT)
        encoded = msgspec.msgpack.encode(resp)
        mock_socket.recv_multipart.return_value = [ZMQ_EMPTY_FRAME, _Frame(encoded)]

        response, payload, transfer_ms, decode_ms = ZmqRequestHelper.send_and_recv(
            mock_socket, b"payload",
        )
        assert response.status == STATUS.HIT
        assert payload is None
        assert transfer_ms >= 0
        assert decode_ms >= 0
        mock_socket.send_multipart.assert_called_once_with([ZMQ_EMPTY_FRAME, b"payload"])

    def test_send_and_recv_zmq_again(self):
        mock_socket = MagicMock()
        mock_socket.send_multipart.side_effect = zmq.Again()
        response, payload, _, _ = ZmqRequestHelper.send_and_recv(mock_socket, b"payload")
        assert response.status == STATUS.ERROR
        assert payload is None

    def test_send_and_recv_recv_zmq_again(self):
        mock_socket = MagicMock()
        mock_socket.recv_multipart.side_effect = zmq.Again()
        response, payload, _, _ = ZmqRequestHelper.send_and_recv(mock_socket, b"payload")
        assert response.status == STATUS.ERROR
        assert payload is None

    def test_send_and_recv_ok_with_payload(self):
        mock_socket = MagicMock()
        header = ECNetworkResponse(status=STATUS.OK, mode="plain", dtype_code=2, shape=(4, 64))
        encoded = msgspec.msgpack.encode(header)
        mock_socket.recv_multipart.return_value = [
            ZMQ_EMPTY_FRAME, _Frame(encoded), _Frame(b"data"),
        ]
        response, payload, _, _ = ZmqRequestHelper.send_and_recv(mock_socket, b"req")
        assert response.status == STATUS.OK
        assert response.mode == "plain"
        assert payload is not None
        assert bytes(payload) == b"data"


# =================== Tests: _get_local_ip =================== #

class TestGetLocalIp:
    def test_returns_string(self):
        ip = _get_local_ip()
        assert isinstance(ip, str)

    def test_returns_valid_ip_format(self):
        ip = _get_local_ip()
        parts = ip.split(".")
        assert len(parts) == 4
        for part in parts:
            assert part.isdigit()
            assert 0 <= int(part) <= 255

    def test_socket_closed_after_call(self):
        import socket as socket_lib
        with patch.object(socket_lib, 'socket') as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.getsockname.return_value = ('10.0.0.1', 0)
            mock_socket_cls.return_value = mock_sock
            ip = _get_local_ip()
            mock_sock.close.assert_called_once()
            assert ip == '10.0.0.1'

    def test_socket_closed_on_exception(self):
        import socket as socket_lib
        with patch.object(socket_lib, 'socket') as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.connect.side_effect = OSError("network error")
            mock_socket_cls.return_value = mock_sock
            with pytest.raises(OSError):
                _get_local_ip()
            mock_sock.close.assert_called_once()


# =================== Tests: ECNetworkConnectorMetadata =================== #

class TestECNetworkConnectorMetadata:
    def test_init_empty(self):
        meta = ECNetworkConnectorMetadata()
        assert meta.mm_hashes == []
        assert meta.mm_hash_endpoints == {}

    def test_add_mm_hash_without_endpoint(self):
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("hash1")
        assert "hash1" in meta.mm_hashes
        assert "hash1" not in meta.mm_hash_endpoints

    def test_add_mm_hash_with_endpoint(self):
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("hash1", endpoint="tcp://127.0.0.1:5000")
        assert "hash1" in meta.mm_hashes
        assert meta.mm_hash_endpoints["hash1"] == "tcp://127.0.0.1:5000"

    def test_add_multiple_hashes(self):
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1", "tcp://10.0.0.1:5000")
        meta.add_mm_hash("h2")
        meta.add_mm_hash("h3", "tcp://10.0.0.2:5001")
        assert len(meta.mm_hashes) == 3
        assert meta.mm_hash_endpoints["h1"] == "tcp://10.0.0.1:5000"
        assert "h2" not in meta.mm_hash_endpoints
        assert meta.mm_hash_endpoints["h3"] == "tcp://10.0.0.2:5001"

    def test_add_mm_hash_with_none_endpoint(self):
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("hash1", endpoint=None)
        assert "hash1" in meta.mm_hashes
        assert "hash1" not in meta.mm_hash_endpoints

    def test_add_same_hash_twice(self):
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("hash1", "tcp://127.0.0.1:5000")
        meta.add_mm_hash("hash1", "tcp://127.0.0.1:6000")
        assert meta.mm_hashes.count("hash1") == 2
        assert meta.mm_hash_endpoints["hash1"] == "tcp://127.0.0.1:6000"


# =================== Tests: Serialization / Deserialization =================== #

class TestSerializeDeserialize:
    def test_serialize_returns_stored_cache(self):
        tensor = torch.randn(10, 768)
        result = _serialize_cache(tensor)
        assert isinstance(result, _StoredCache)
        assert isinstance(result.payload, bytes)
        assert len(result.payload) > 0
        assert result.dtype_code == DTYPE_TO_CODE[tensor.dtype]
        assert result.shape == tuple(tensor.shape)
        assert result.mode in ("plain", "chunked")

    def test_serialize_detaches_gradient(self):
        tensor = torch.randn(10, 768, requires_grad=True)
        result = _serialize_cache(tensor)
        assert isinstance(result, _StoredCache)
        assert isinstance(result.payload, bytes)

    def test_deserialize_returns_tensor(self):
        tensor = torch.randn(5, 10)
        serialized = _serialize_cache(tensor)
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as mock_platform:
            mock_platform.device_type = 'cpu'
            restored = _deserialize_cache(
                serialized.payload, tensor.dtype, serialized.shape, serialized.mode,
            )
        assert isinstance(restored, torch.Tensor)

    def test_serialize_deserialize_roundtrip(self):
        original = torch.randn(8, 512)
        serialized = _serialize_cache(original)
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as mock_platform:
            mock_platform.device_type = 'cpu'
            restored = _deserialize_cache(
                serialized.payload, original.dtype, serialized.shape, serialized.mode,
            )
        assert torch.equal(original.cpu(), restored.cpu())
        assert original.shape == restored.shape
        assert original.dtype == restored.dtype

    def test_roundtrip_multiple_shapes(self):
        test_tensors = [
            torch.randn(1, 64),
            torch.randn(16, 256),
            torch.zeros(4, 128),
            torch.ones(2, 32),
        ]
        for tensor in test_tensors:
            serialized = _serialize_cache(tensor)
            with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
                m.device_type = 'cpu'
                restored = _deserialize_cache(
                    serialized.payload, tensor.dtype, serialized.shape, serialized.mode,
                )
            assert torch.equal(tensor.cpu(), restored.cpu())
            assert tensor.shape == restored.shape

    def test_roundtrip_bfloat16(self):
        original = torch.randn(4, 64, dtype=torch.bfloat16)
        stored = _serialize_cache(original)
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            restored = _deserialize_cache(
                stored.payload, original.dtype, stored.shape, stored.mode, device='cpu',
            )
        assert torch.equal(original.cpu(), restored.cpu())
        assert original.dtype == restored.dtype

    def test_roundtrip_multiple_dtypes(self):
        for dtype in [torch.float16, torch.float32, torch.int32, torch.bool]:
            if dtype == torch.bool:
                t = torch.randint(0, 2, (4, 8)).bool()
            else:
                t = torch.randint(0, 10, (4, 8)).to(dtype)
            stored = _serialize_cache(t)
            with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
                m.device_type = 'cpu'
                restored = _deserialize_cache(
                    stored.payload, t.dtype, stored.shape, stored.mode, device='cpu',
                )
            assert torch.equal(t.cpu(), restored.cpu()), f"Failed for {dtype}"


# =================== Tests: ECNetworkConnector Init =================== #

class TestECNetworkConnectorInit:
    def test_init_producer_scheduler(self):
        connector = create_connector(role=ECConnectorRole.SCHEDULER, is_producer=True)
        assert connector.role == ECConnectorRole.SCHEDULER
        assert connector.is_producer is True
        assert connector.tp_rank == 0
        assert connector.dp_rank == 0

    def test_init_producer_worker(self):
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=True)
        assert connector.role == ECConnectorRole.WORKER
        assert connector.is_producer is True

    def test_init_consumer_worker(self):
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=False)
        assert connector.role == ECConnectorRole.WORKER
        assert connector.is_producer is False

    def test_init_cache_store_empty(self):
        connector = create_connector()
        assert len(connector._cache_store) == 0

    def test_init_client_sockets_dict(self):
        connector = create_connector()
        assert isinstance(connector._client_sockets, dict)

    def test_init_server_socket_none_for_non_worker_producer(self):
        connector = create_connector(role=ECConnectorRole.SCHEDULER, is_producer=True)
        assert connector._server_socket is None

    def test_init_server_socket_set_for_worker_producer(self):
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=True)
        assert connector._server_socket is not None

    def test_init_sets_producer_endpoint(self):
        connector = create_connector(role=ECConnectorRole.SCHEDULER, is_producer=True)
        assert connector._endpoint is not None
        assert connector._producer_endpoint is not None

    def test_init_sets_consumer_endpoint(self):
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=False)
        assert connector._endpoint is not None
        assert hasattr(connector, '_consumer_endpoints')

    def test_init_raises_without_ec_transfer_config(self):
        config = Mock()
        config.ec_transfer_config = None
        patches = _make_connector_patches()
        for p in patches:
            p.start()
        try:
            with pytest.raises(ValueError, match="ec_transfer_config must be set"):
                ECNetworkConnector(vllm_config=config, role=ECConnectorRole.SCHEDULER)
        finally:
            for p in patches:
                p.stop()

    def test_init_threading_locks_created(self):
        connector = create_connector()
        assert isinstance(connector._cache_lock, type(threading.Lock()))
        assert isinstance(connector._client_lock, type(threading.Lock()))
        assert isinstance(connector._async_pool_lock, type(threading.Lock()))

    def test_init_async_state(self):
        connector = create_connector()
        assert isinstance(connector._mm_hashes_need_loads, set)
        assert isinstance(connector._cache_store, OrderedDict)
        assert isinstance(connector._async_socket_pools, dict)
        assert isinstance(connector._ec_load_executor, ThreadPoolExecutor)
        assert len(connector._mm_hashes_need_loads) == 0

    def test_init_ec_async_flag_default_false(self):
        connector = create_connector()
        assert connector.ec_async_flag is False

    def test_init_ec_async_flag_true(self):
        config = make_mock_vllm_config(
            is_producer=True,
            extra_config_overrides={"load_ec_async": True},
        )
        connector = create_connector(role=ECConnectorRole.SCHEDULER, vllm_config=config)
        assert connector.ec_async_flag is True

    def test_init_cache_store_max_bytes(self):
        config = make_mock_vllm_config(
            is_producer=True,
            extra_config_overrides={"ec_cache_max_gb": 5},
        )
        connector = create_connector(role=ECConnectorRole.SCHEDULER, vllm_config=config)
        assert connector._cache_store_max_bytes == 5 * GB


# =================== Tests: _init_rank_and_size =================== #

class TestInitRankAndSize:
    def test_scheduler_role_does_not_call_tp_rank(self):
        with patch('omni_npu.connector.ec_connector.network_connector.get_tensor_model_parallel_rank') as mock_tp_rank, \
             patch('omni_npu.connector.ec_connector.network_connector._get_local_ip', return_value='127.0.0.1'), \
             patch('omni_npu.connector.ec_connector.network_connector.make_zmq_path', return_value='tcp://127.0.0.1:5000'), \
             patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket', return_value=MagicMock()), \
             patch('zmq.Context.instance', return_value=MagicMock()), \
             patch('omni_npu.connector.ec_connector.network_connector.envs', VLLM_RPC_TIMEOUT=5000):
            config = make_mock_vllm_config(is_producer=True)
            ECNetworkConnector(vllm_config=config, role=ECConnectorRole.SCHEDULER)
            mock_tp_rank.assert_not_called()

    def test_worker_role_calls_tp_rank(self):
        with patch('omni_npu.connector.ec_connector.network_connector.get_tensor_model_parallel_rank',
                   return_value=0) as mock_tp_rank, \
             patch('omni_npu.connector.ec_connector.network_connector._get_local_ip', return_value='127.0.0.1'), \
             patch('omni_npu.connector.ec_connector.network_connector.make_zmq_path', return_value='tcp://127.0.0.1:5000'), \
             patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket', return_value=MagicMock()), \
             patch('zmq.Context.instance', return_value=MagicMock()), \
             patch('omni_npu.connector.ec_connector.network_connector.envs', VLLM_RPC_TIMEOUT=5000):
            config = make_mock_vllm_config(is_producer=True)
            connector = ECNetworkConnector(vllm_config=config, role=ECConnectorRole.WORKER)
            connector._server_running.clear()
            mock_tp_rank.assert_called_once()

    def test_worker_role_sets_dp_and_tp(self):
        config = make_mock_vllm_config(is_producer=True)
        config.parallel_config.data_parallel_rank = 2
        config.parallel_config.data_parallel_size = 4
        config.parallel_config.tensor_parallel_size = 2

        with patch('omni_npu.connector.ec_connector.network_connector.get_tensor_model_parallel_rank',
                   return_value=1), \
             patch('omni_npu.connector.ec_connector.network_connector._get_local_ip', return_value='127.0.0.1'), \
             patch('omni_npu.connector.ec_connector.network_connector.make_zmq_path', return_value='tcp://127.0.0.1:5000'), \
             patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket', return_value=MagicMock()), \
             patch('zmq.Context.instance', return_value=MagicMock()), \
             patch('omni_npu.connector.ec_connector.network_connector.envs', VLLM_RPC_TIMEOUT=5000):
            connector = ECNetworkConnector(vllm_config=config, role=ECConnectorRole.WORKER)
            connector._server_running.clear()
            assert connector.dp_rank == 2
            assert connector.dp_size == 4
            assert connector.tp_rank == 1
            assert connector.tp_size == 2


# =================== Tests: _init_consumer_endpoints =================== #

class TestInitConsumerEndpoints:
    def test_single_ip_no_ec_ports(self):
        config = make_mock_vllm_config(is_producer=False, ec_ip="10.0.0.1", ec_port=5000)
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=False,
                                     vllm_config=config)
        assert len(connector._consumer_endpoints) == 1

    def test_multiple_ips_no_ec_ports(self):
        config = make_mock_vllm_config(is_producer=False, ec_ip="10.0.0.1,10.0.0.2", ec_port=5000)
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=False,
                                     vllm_config=config)
        assert len(connector._consumer_endpoints) == 2

    def test_ec_ports_mismatch_raises_error(self):
        config = make_mock_vllm_config(
            is_producer=False, ec_ip="10.0.0.1,10.0.0.2", ec_port=5000,
            extra_config_overrides={"ec_ports": [5000]},
        )
        patches = _make_connector_patches()
        for p in patches:
            p.start()
        try:
            with pytest.raises(ValueError, match="The length of ec_ports does not match"):
                ECNetworkConnector(vllm_config=config, role=ECConnectorRole.WORKER)
        finally:
            for p in patches:
                p.stop()

    def test_ec_ports_matching_creates_correct_endpoints(self):
        config = make_mock_vllm_config(
            is_producer=False, ec_ip="10.0.0.1,10.0.0.2", ec_port=5000,
            extra_config_overrides={"ec_ports": [5000, 6000]},
        )
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=False,
                                     vllm_config=config)
        assert len(connector._consumer_endpoints) == 2

    def test_ec_ports_as_string(self):
        config = make_mock_vllm_config(
            is_producer=False, ec_ip="10.0.0.1,10.0.0.2", ec_port=5000,
            extra_config_overrides={"ec_ports": "5000, 6000"},
        )
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=False,
                                     vllm_config=config)
        assert len(connector._consumer_endpoints) == 2

    def test_ec_ports_as_tuple(self):
        config = make_mock_vllm_config(
            is_producer=False, ec_ip="10.0.0.1,10.0.0.2", ec_port=5000,
            extra_config_overrides={"ec_ports": (5000, 6000)},
        )
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=False,
                                     vllm_config=config)
        assert len(connector._consumer_endpoints) == 2


# =================== Tests: Cache store LRU helpers =================== #

class TestCacheStoreLRU:
    def test_touch_cache_moves_to_end(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store["a"] = b"1"
        c._cache_store["b"] = b"2"
        c._cache_store["c"] = b"3"
        c._touch_cache("a")
        assert list(c._cache_store.keys()) == ["b", "c", "a"]

    def test_touch_cache_nonexistent_no_error(self, producer_worker_connector):
        c = producer_worker_connector
        c._touch_cache("nonexistent")

    def test_evict_cache_if_needed(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store_max_bytes = 100
        c._cache_store["a"] = _StoredCache(
            payload=b"x" * 50, dtype_code=3, shape=(1,), mode="raw",
        )
        c._cache_store_bytes = 50
        c._evict_cache_if_needed(60)
        assert "a" not in c._cache_store
        assert c._cache_store_bytes == 0

    def test_evict_cache_multiple_entries(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store_max_bytes = 100
        c._cache_store["a"] = _StoredCache(
            payload=b"x" * 40, dtype_code=3, shape=(1,), mode="raw",
        )
        c._cache_store["b"] = _StoredCache(
            payload=b"y" * 40, dtype_code=3, shape=(1,), mode="raw",
        )
        c._cache_store_bytes = 80
        c._evict_cache_if_needed(30)
        assert c._cache_store_bytes + 30 <= c._cache_store_max_bytes

    def test_evict_cache_no_eviction_needed(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store_max_bytes = 1000
        c._cache_store["a"] = _StoredCache(
            payload=b"x" * 10, dtype_code=3, shape=(1,), mode="raw",
        )
        c._cache_store_bytes = 10
        c._evict_cache_if_needed(10)
        assert "a" in c._cache_store
        assert c._cache_store_bytes == 10


# =================== Tests: TP leader mode helpers =================== #

class TestTPLeaderHelpers:
    def test_ec_zmq_tp_leader_only_enabled_default_true(self, consumer_worker_connector):
        c = consumer_worker_connector
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": True,
            }.get(key, default)
        )
        assert c._ec_zmq_tp_leader_only_enabled() is True

    def test_ec_zmq_tp_leader_only_enabled_false(self, consumer_worker_connector):
        c = consumer_worker_connector
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": False,
            }.get(key, default)
        )
        assert c._ec_zmq_tp_leader_only_enabled() is False

    def test_ec_use_tp_leader_zmq_producer_returns_false(self, producer_worker_connector):
        assert producer_worker_connector._ec_use_tp_leader_zmq() is False

    def test_ec_use_tp_leader_zmq_tp_size_1_returns_false(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.tp_size = 1
        assert c._ec_use_tp_leader_zmq() is False

    def test_ec_use_tp_leader_zmq_disabled_returns_false(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.tp_size = 2
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": False,
            }.get(key, default)
        )
        assert c._ec_use_tp_leader_zmq() is False

    def test_ec_use_tp_leader_zmq_async_flag_returns_false(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.tp_size = 2
        c.ec_async_flag = True
        assert c._ec_use_tp_leader_zmq() is False

    @patch('omni_npu.connector.ec_connector.network_connector.torch.distributed.is_initialized',
           return_value=True)
    def test_ec_use_tp_leader_zmq_all_conditions_met(self, mock_init, consumer_worker_connector):
        c = consumer_worker_connector
        c.tp_size = 2
        c.ec_async_flag = False
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": True,
            }.get(key, default)
        )
        assert c._ec_use_tp_leader_zmq() is True


# =================== Tests: Server =================== #

class TestStartServer:
    def test_start_server_early_return_if_already_started(self, producer_worker_connector):
        connector = producer_worker_connector
        existing_socket = MagicMock()
        connector._server_socket = existing_socket

        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket') as mock_make, \
             patch('omni_npu.connector.ec_connector.network_connector.envs', VLLM_RPC_TIMEOUT=5000):
            connector._start_server()
            mock_make.assert_not_called()

        assert connector._server_socket is existing_socket

    def test_start_server_creates_socket_and_starts_thread(self):
        connector = create_connector(role=ECConnectorRole.SCHEDULER, is_producer=True)
        connector._server_socket = None
        connector._endpoint = "tcp://127.0.0.1:5000"

        mock_sock = MagicMock()
        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_sock), \
             patch('omni_npu.connector.ec_connector.network_connector.envs', VLLM_RPC_TIMEOUT=5000):
            connector._start_server()

        assert connector._server_socket is mock_sock
        assert connector._server_thread is not None
        connector._server_running.clear()


class TestServerLoop:
    """Test _server_loop, _recv_server_message, _decode_server_request, _handle_server_request, _send_server_response."""

    def _run_server_loop_once(self, connector, messages):
        """Run _server_loop for one iteration with given messages."""
        mock_socket = MagicMock()
        connector._server_socket = mock_socket

        call_count = [0]
        def is_set_side_effect():
            call_count[0] += 1
            return call_count[0] <= 1

        connector._server_running = MagicMock()
        connector._server_running.is_set.side_effect = is_set_side_effect

        mock_poller = MagicMock()
        mock_poller.poll.return_value = {mock_socket: zmq.POLLIN}
        mock_socket.recv_multipart.side_effect = [messages] + [zmq.Again()]

        with patch('omni_npu.connector.ec_connector.network_connector.zmq.Poller',
                   return_value=mock_poller):
            connector._server_loop()

        return mock_socket.send_multipart.call_args_list

    def test_server_handles_has_command_hit(self, producer_worker_connector):
        connector = producer_worker_connector
        mm_hash = "cached_hash"
        connector._cache_store[mm_hash] = b"data"

        request = ECNetworkRequest(command=STATUS.HAS, mm_hash=mm_hash)
        payload = msgspec.msgpack.encode(request)
        messages = [b"identity", b"", payload]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 1
        sent = calls[0][0][0]
        response = msgspec.msgpack.decode(sent[2], type=ECNetworkResponse)
        assert response.status == STATUS.HIT

    def test_server_handles_has_command_miss(self, producer_worker_connector):
        connector = producer_worker_connector
        request = ECNetworkRequest(command=STATUS.HAS, mm_hash="missing_hash")
        payload = msgspec.msgpack.encode(request)
        messages = [b"identity", b"", payload]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 1
        sent = calls[0][0][0]
        response = msgspec.msgpack.decode(sent[2], type=ECNetworkResponse)
        assert response.status == STATUS.MISS

    def test_server_handles_get_command_hit(self, producer_worker_connector):
        connector = producer_worker_connector
        mm_hash = "get_hash"
        cache_data = b"tensor_bytes"
        connector._cache_store[mm_hash] = _StoredCache(
            payload=cache_data, dtype_code=3, shape=(1,), mode="raw",
        )

        request = ECNetworkRequest(command=STATUS.GET, mm_hash=mm_hash)
        payload = msgspec.msgpack.encode(request)
        messages = [b"identity", b"", payload]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 1
        sent = calls[0][0][0]
        response = msgspec.msgpack.decode(sent[2], type=ECNetworkResponse)
        assert response.status == STATUS.OK
        assert response.mode == "raw"
        # payload travels as a separate frame (not inside the header)
        assert sent[3] == cache_data

    def test_server_handles_get_command_miss(self, producer_worker_connector):
        connector = producer_worker_connector
        request = ECNetworkRequest(command=STATUS.GET, mm_hash="absent_hash")
        payload = msgspec.msgpack.encode(request)
        messages = [b"identity", b"", payload]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 1
        sent = calls[0][0][0]
        response = msgspec.msgpack.decode(sent[2], type=ECNetworkResponse)
        assert response.status == STATUS.MISS

    def test_server_handles_invalid_payload(self, producer_worker_connector):
        connector = producer_worker_connector
        messages = [b"identity", b"", b"not_valid_msgpack"]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 1
        sent = calls[0][0][0]
        response = msgspec.msgpack.decode(sent[2], type=ECNetworkResponse)
        assert response.status == STATUS.ERROR

    def test_server_handles_unknown_command(self, producer_worker_connector):
        connector = producer_worker_connector
        raw = msgspec.msgpack.encode({"command": "invalid_cmd", "mm_hash": "h"})
        messages = [b"identity", b"", raw]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 1
        sent = calls[0][0][0]
        response = msgspec.msgpack.decode(sent[2], type=ECNetworkResponse)
        assert response.status == STATUS.ERROR

    def test_server_recv_zmq_again_continues(self, producer_worker_connector):
        connector = producer_worker_connector
        mock_socket = MagicMock()
        connector._server_socket = mock_socket

        call_count = [0]
        def is_set_side_effect():
            call_count[0] += 1
            return call_count[0] <= 1

        connector._server_running = MagicMock()
        connector._server_running.is_set.side_effect = is_set_side_effect

        mock_poller = MagicMock()
        mock_poller.poll.return_value = {mock_socket: zmq.POLLIN}
        mock_socket.recv_multipart.side_effect = zmq.Again()

        with patch('omni_npu.connector.ec_connector.network_connector.zmq.Poller',
                   return_value=mock_poller):
            connector._server_loop()

        mock_socket.send_multipart.assert_not_called()

    def test_server_skips_message_with_nonempty_separator(self, producer_worker_connector):
        connector = producer_worker_connector
        request = ECNetworkRequest(command=STATUS.HAS, mm_hash="h")
        payload = msgspec.msgpack.encode(request)
        messages = [b"identity", b"not_empty", payload]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 0

    def test_server_skips_short_message(self, producer_worker_connector):
        connector = producer_worker_connector
        messages = [b"identity", b""]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 0

    def test_server_no_events_no_processing(self, producer_worker_connector):
        connector = producer_worker_connector
        mock_socket = MagicMock()
        connector._server_socket = mock_socket

        call_count = [0]
        def is_set_side_effect():
            call_count[0] += 1
            return call_count[0] <= 2

        connector._server_running = MagicMock()
        connector._server_running.is_set.side_effect = is_set_side_effect

        mock_poller = MagicMock()
        mock_poller.poll.return_value = {}

        with patch('omni_npu.connector.ec_connector.network_connector.zmq.Poller',
                   return_value=mock_poller):
            connector._server_loop()

        mock_socket.send_multipart.assert_not_called()


class TestHandleServerRequest:
    """Direct unit tests for _handle_server_request."""

    def test_has_hit(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store["h1"] = _StoredCache(
            payload=b"data", dtype_code=3, shape=(1,), mode="raw",
        )
        req = ECNetworkRequest(command=STATUS.HAS, mm_hash="h1")
        resp, _ = c._handle_server_request(req)
        assert resp.status == STATUS.HIT

    def test_has_miss(self, producer_worker_connector):
        c = producer_worker_connector
        req = ECNetworkRequest(command=STATUS.HAS, mm_hash="missing")
        resp, _ = c._handle_server_request(req)
        assert resp.status == STATUS.MISS

    def test_get_hit(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store["h1"] = _StoredCache(
            payload=b"payload", dtype_code=3, shape=(1,), mode="raw",
        )
        req = ECNetworkRequest(command=STATUS.GET, mm_hash="h1")
        resp, payload = c._handle_server_request(req)
        assert resp.status == STATUS.OK
        assert resp.mode == "raw"
        assert payload == b"payload"

    def test_get_miss(self, producer_worker_connector):
        c = producer_worker_connector
        req = ECNetworkRequest(command=STATUS.GET, mm_hash="missing")
        resp, _ = c._handle_server_request(req)
        assert resp.status == STATUS.MISS

    def test_unknown_command(self, producer_worker_connector):
        c = producer_worker_connector
        req = ECNetworkRequest(command=STATUS.OK, mm_hash="h")
        resp, _ = c._handle_server_request(req)
        assert resp.status == STATUS.ERROR

    def test_get_touches_cache(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store["a"] = _StoredCache(
            payload=b"1", dtype_code=3, shape=(1,), mode="raw",
        )
        c._cache_store["b"] = _StoredCache(
            payload=b"2", dtype_code=3, shape=(1,), mode="raw",
        )
        req = ECNetworkRequest(command=STATUS.GET, mm_hash="a")
        c._handle_server_request(req)
        assert list(c._cache_store.keys())[-1] == "a"


class TestDecodeServerRequest:
    def test_valid_payload(self):
        req = ECNetworkRequest(command=STATUS.HAS, mm_hash="h1")
        payload = msgspec.msgpack.encode(req)
        result = ECNetworkConnector._decode_server_request(payload)
        assert result is not None
        assert result.command == STATUS.HAS
        assert result.mm_hash == "h1"

    def test_invalid_payload(self):
        result = ECNetworkConnector._decode_server_request(b"garbage")
        assert result is None


class TestRecvServerMessage:
    def test_valid_message(self, producer_worker_connector):
        c = producer_worker_connector
        mock_socket = MagicMock()
        c._server_socket = mock_socket
        mock_socket.recv_multipart.return_value = [b"id", b"", b"payload"]
        result = c._recv_server_message()
        assert result == (b"id", b"payload")

    def test_short_message(self, producer_worker_connector):
        c = producer_worker_connector
        mock_socket = MagicMock()
        c._server_socket = mock_socket
        mock_socket.recv_multipart.return_value = [b"id", b""]
        result = c._recv_server_message()
        assert result is None

    def test_non_empty_delimiter(self, producer_worker_connector):
        c = producer_worker_connector
        mock_socket = MagicMock()
        c._server_socket = mock_socket
        mock_socket.recv_multipart.return_value = [b"id", b"bad", b"payload"]
        result = c._recv_server_message()
        assert result is None

    def test_zmq_again(self, producer_worker_connector):
        c = producer_worker_connector
        mock_socket = MagicMock()
        c._server_socket = mock_socket
        mock_socket.recv_multipart.side_effect = zmq.Again()
        result = c._recv_server_message()
        assert result is None


class TestSendServerResponse:
    def test_sends_correct_format(self, producer_worker_connector):
        c = producer_worker_connector
        mock_socket = MagicMock()
        c._server_socket = mock_socket
        resp = ECNetworkResponse(
            status=STATUS.OK, mode="raw", dtype_code=3, shape=(1,),
        )
        c._send_server_response(b"client_id", resp, b"data")
        mock_socket.send_multipart.assert_called_once()
        args = mock_socket.send_multipart.call_args[0][0]
        assert args[0] == b"client_id"
        assert args[1] == ZMQ_EMPTY_FRAME
        decoded = msgspec.msgpack.decode(args[2], type=ECNetworkResponse)
        assert decoded.status == STATUS.OK
        assert args[3] == b"data"


# =================== Tests: Client Socket Management =================== #

class TestClientSocketManagement:
    def test_ensure_client_socket_returns_existing(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = "tcp://127.0.0.1:5000"
        mock_socket = MagicMock()
        connector._client_sockets[endpoint] = mock_socket
        result = connector._ensure_client_socket(endpoint)
        assert result is mock_socket

    def test_ensure_client_socket_creates_new(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = "tcp://127.0.0.1:9999"
        new_mock = MagicMock()
        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=new_mock), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000):
            result = connector._ensure_client_socket(endpoint)
        assert result is new_mock
        assert endpoint in connector._client_sockets

    def test_ensure_client_socket_list_creates_multiple(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoints = ["tcp://127.0.0.1:6001", "tcp://127.0.0.1:6002"]
        mock_sock = MagicMock()
        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_sock), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000):
            result = connector._ensure_client_socket(endpoints)
        assert result is None
        for ep in endpoints:
            assert ep in connector._client_sockets

    def test_ensure_client_socket_invalid_type_raises(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        with pytest.raises(TypeError, match="Invalid endpoint type"):
            connector._ensure_client_socket(12345)

    def test_create_client_socket_sets_options(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = "tcp://127.0.0.1:7000"
        mock_sock = MagicMock()
        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_sock), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000):
            connector._create_client_socket(endpoint)
        assert mock_sock.setsockopt.call_count >= 2

    def test_resolve_endpoint_default(self, producer_scheduler_connector):
        c = producer_scheduler_connector
        c._endpoint = "tcp://default:5000"
        assert c._resolve_endpoint("unknown_hash") == "tcp://default:5000"

    def test_resolve_endpoint_specific(self, producer_scheduler_connector):
        c = producer_scheduler_connector
        c._mm_hash_endpoints["h1"] = "tcp://specific:5001"
        assert c._resolve_endpoint("h1") == "tcp://specific:5001"


# =================== Tests: _client_request =================== #

class TestClientRequest:
    def test_client_request_hit(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        response = ECNetworkResponse(status=STATUS.HIT)
        reply_data = msgspec.msgpack.encode(response)
        mock_sock.recv_multipart.return_value = [b"", _Frame(reply_data)]
        connector._client_sockets[endpoint] = mock_sock

        response, payload, transfer_ms, decode_ms = connector._client_request(
            STATUS.HAS, "test_hash",
        )
        assert response.status == STATUS.HIT
        assert payload is None
        mock_sock.send_multipart.assert_called_once()

    def test_client_request_miss(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        response = ECNetworkResponse(status=STATUS.MISS)
        mock_sock.recv_multipart.return_value = [b"", _Frame(msgspec.msgpack.encode(response))]
        connector._client_sockets[endpoint] = mock_sock

        response, _, _, _ = connector._client_request(STATUS.HAS, "nonexistent_hash")
        assert response.status == STATUS.MISS

    def test_client_request_ok_with_payload(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        payload = b"cached_tensor_data"
        response = ECNetworkResponse(
            status=STATUS.OK, mode="raw", dtype_code=3, shape=(1,),
        )
        mock_sock.recv_multipart.return_value = [
            b"", _Frame(msgspec.msgpack.encode(response)), _Frame(payload),
        ]
        connector._client_sockets[endpoint] = mock_sock

        response, resp_payload, _, _ = connector._client_request(STATUS.GET, "test_hash")
        assert response.status == STATUS.OK
        assert bytes(resp_payload) == payload

    def test_client_request_timeout_returns_error(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        mock_sock.send_multipart.side_effect = zmq.Again()
        connector._client_sockets[endpoint] = mock_sock

        response, _, _, _ = connector._client_request(STATUS.HAS, "test_hash")
        assert response.status == STATUS.ERROR

    def test_client_request_recv_timeout_returns_error(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        mock_sock.recv_multipart.side_effect = zmq.Again()
        connector._client_sockets[endpoint] = mock_sock

        response, _, _, _ = connector._client_request(STATUS.GET, "test_hash")
        assert response.status == STATUS.ERROR

    def test_client_request_uses_mm_hash_endpoint(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        specific_endpoint = "tcp://10.0.0.2:5001"
        connector._mm_hash_endpoints["special_hash"] = specific_endpoint
        mock_sock = MagicMock()
        response = ECNetworkResponse(status=STATUS.HIT)
        mock_sock.recv_multipart.return_value = [b"", _Frame(msgspec.msgpack.encode(response))]
        connector._client_sockets[specific_endpoint] = mock_sock

        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_sock), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000):
            response, _, _, _ = connector._client_request(STATUS.HAS, "special_hash")
        assert response.status == STATUS.HIT

    def test_client_request_error_logs_warning(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        mock_sock.send_multipart.side_effect = zmq.Again()
        connector._client_sockets[endpoint] = mock_sock

        with patch('omni_npu.connector.ec_connector.network_connector.logger') as mock_logger:
            connector._client_request(STATUS.HAS, "test_hash")
            mock_logger.warning.assert_called_once()


# =================== Tests: _async_client_request =================== #

class TestAsyncClientRequest:
    def test_async_client_request_success(self, consumer_worker_connector):
        c = consumer_worker_connector
        resp = ECNetworkResponse(
            status=STATUS.OK, mode="raw", dtype_code=3, shape=(1,),
        )
        encoded = msgspec.msgpack.encode(resp)
        mock_socket = MagicMock()
        mock_socket.recv_multipart.return_value = [
            ZMQ_EMPTY_FRAME, _Frame(encoded), _Frame(b"data"),
        ]

        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_socket), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000):
            response, payload, _, _ = c._async_client_request(STATUS.GET, "h1")

        assert response.status == STATUS.OK
        assert bytes(payload) == b"data"
        # healthy sockets are returned to the pool, not closed
        pool = c._async_socket_pools.get(c._endpoint)
        assert pool is not None and pool.qsize() == 1
        mock_socket.close.assert_not_called()

    def test_async_client_request_error(self, consumer_worker_connector):
        c = consumer_worker_connector
        mock_socket = MagicMock()
        mock_socket.send_multipart.side_effect = zmq.Again()

        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_socket), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000), \
             patch('omni_npu.connector.ec_connector.network_connector.logger') as mock_logger:
            response, payload, _, _ = c._async_client_request(STATUS.GET, "h1")

        assert response.status == STATUS.ERROR
        assert payload is None
        mock_logger.warning.assert_called_once()

    def test_async_client_request_closes_socket_on_exception(self, consumer_worker_connector):
        c = consumer_worker_connector
        mock_socket = MagicMock()
        mock_socket.send_multipart.side_effect = RuntimeError("boom")

        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_socket), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000):
            with pytest.raises(RuntimeError):
                c._async_client_request(STATUS.GET, "h1")

        mock_socket.close.assert_called_once_with(linger=0)


# =================== Tests: save_caches =================== #

class TestSaveCaches:
    def test_save_caches_stores_payload(self, producer_worker_connector):
        connector = producer_worker_connector
        mm_hash = "save_hash"
        tensor = torch.randn(4, 64)
        encoder_cache = {mm_hash: tensor}

        connector.save_caches(encoder_cache, mm_hash)

        assert mm_hash in connector._cache_store
        stored = connector._cache_store[mm_hash]
        assert isinstance(stored, _StoredCache)
        assert isinstance(stored.payload, bytes)
        assert len(stored.payload) > 0
        assert stored.shape == tuple(tensor.shape)

    def test_save_caches_skips_consumer(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "consumer_hash"
        encoder_cache = {mm_hash: torch.randn(4, 64)}

        connector.save_caches(encoder_cache, mm_hash)
        assert mm_hash not in connector._cache_store

    def test_save_caches_skips_non_worker_role(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        mm_hash = "scheduler_hash"
        encoder_cache = {mm_hash: torch.randn(4, 64)}

        connector.save_caches(encoder_cache, mm_hash)
        assert mm_hash not in connector._cache_store

    def test_save_caches_overwrites_existing(self, producer_worker_connector):
        connector = producer_worker_connector
        mm_hash = "overwrite_hash"
        tensor1 = torch.randn(4, 64)
        tensor2 = torch.randn(4, 64)
        encoder_cache = {mm_hash: tensor1}

        connector.save_caches(encoder_cache, mm_hash)
        first_payload = connector._cache_store[mm_hash]

        encoder_cache[mm_hash] = tensor2
        connector.save_caches(encoder_cache, mm_hash)
        second_payload = connector._cache_store[mm_hash]

        assert mm_hash in connector._cache_store
        assert first_payload != second_payload

    def test_save_caches_updates_bytes_counter(self, producer_worker_connector):
        connector = producer_worker_connector
        mm_hash = "bytes_hash"
        tensor = torch.randn(4, 64)
        encoder_cache = {mm_hash: tensor}

        connector.save_caches(encoder_cache, mm_hash)
        assert connector._cache_store_bytes > 0
        assert connector._cache_store_bytes == len(connector._cache_store[mm_hash].payload)

    def test_save_caches_evicts_when_full(self, producer_worker_connector):
        connector = producer_worker_connector
        connector._cache_store_max_bytes = 100

        encoder_cache = {"h1": torch.randn(4, 64)}
        connector.save_caches(encoder_cache, "h1")

        encoder_cache["h2"] = torch.randn(4, 64)
        connector.save_caches(encoder_cache, "h2")

        remaining = connector._cache_store.get("h2")
        remaining_bytes = len(remaining.payload) if remaining is not None else 0
        assert connector._cache_store_bytes <= connector._cache_store_max_bytes + remaining_bytes

    def test_save_caches_multiple_hashes(self, producer_worker_connector):
        connector = producer_worker_connector
        encoder_cache = {
            "h1": torch.randn(2, 32),
            "h2": torch.randn(3, 48),
        }
        connector.save_caches(encoder_cache, "h1")
        connector.save_caches(encoder_cache, "h2")

        assert "h1" in connector._cache_store
        assert "h2" in connector._cache_store

    def test_save_caches_skips_non_rank0(self, producer_worker_connector):
        connector = producer_worker_connector
        connector.tp_rank = 1
        mm_hash = "non_rank0_hash"
        encoder_cache = {mm_hash: torch.randn(4, 64)}

        connector.save_caches(encoder_cache, mm_hash)
        assert mm_hash not in connector._cache_store


# =================== Tests: start_load_caches (sync) =================== #

class TestStartLoadCaches:
    def _ok_tuple(self, stored, payload=None):
        """Build a (response, payload, transfer_ms, decode_ms) 4-tuple mock value."""
        resp = ECNetworkResponse(
            status=STATUS.OK, mode=stored.mode,
            dtype_code=stored.dtype_code, shape=stored.shape,
        )
        return (resp, payload if payload is not None else stored.payload, 1.0, 0.5)

    def test_start_load_caches_loads_tensor(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "load_hash"
        tensor = torch.randn(4, 64)
        stored = _serialize_cache(tensor)

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        connector._client_request = Mock(return_value=self._ok_tuple(stored))

        encoder_cache = {}
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            connector.start_load_caches(encoder_cache)

        assert mm_hash in encoder_cache
        got = encoder_cache.get(mm_hash)
        assert got is not None
        assert isinstance(got, torch.Tensor)
        assert torch.equal(got.cpu(), tensor.cpu())

    def test_start_load_caches_skips_existing(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "existing_hash"
        existing_tensor = torch.randn(4, 64)

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        connector._client_request = Mock(
            return_value=(ECNetworkResponse(
                status=STATUS.OK, mode="raw", dtype_code=3, shape=(1,),
            ), None, 0.0, 0.0))

        encoder_cache = {mm_hash: existing_tensor}
        connector.start_load_caches(encoder_cache)

        connector._client_request.assert_not_called()
        got = encoder_cache.get(mm_hash)
        assert got is not None
        assert torch.equal(got, existing_tensor)

    def test_start_load_caches_handles_miss(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "miss_hash"

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        connector._client_request = Mock(
            return_value=(ECNetworkResponse(status=STATUS.MISS), None, 0.0, 0.0))

        encoder_cache = {}
        connector.start_load_caches(encoder_cache)
        assert mm_hash not in encoder_cache

    def test_start_load_caches_handles_error_status(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "error_hash"

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        connector._client_request = Mock(
            return_value=(ECNetworkResponse(status=STATUS.ERROR), None, 0.0, 0.0))

        encoder_cache = {}
        connector.start_load_caches(encoder_cache)
        assert mm_hash not in encoder_cache

    def test_start_load_caches_ok_with_none_payload_skips(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "null_payload_hash"

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        connector._client_request = Mock(
            return_value=(ECNetworkResponse(
                status=STATUS.OK, mode="raw", dtype_code=3, shape=(1,),
            ), None, 0.0, 0.0))

        encoder_cache = {}
        connector.start_load_caches(encoder_cache)
        assert mm_hash not in encoder_cache

    def test_start_load_caches_uses_endpoint_from_metadata(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "endpoint_hash"

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash, endpoint="tcp://10.0.0.5:5000")
        connector.bind_connector_metadata(meta)

        stored = _serialize_cache(torch.randn(2, 32))
        connector._client_request = Mock(return_value=self._ok_tuple(stored))

        encoder_cache = {}
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            connector.start_load_caches(encoder_cache)

        assert connector._mm_hash_endpoints.get(mm_hash) == "tcp://10.0.0.5:5000"

    def test_start_load_caches_multiple_hashes(self, consumer_worker_connector):
        connector = consumer_worker_connector
        hashes = ["h1", "h2", "h3"]

        meta = ECNetworkConnectorMetadata()
        for h in hashes:
            meta.add_mm_hash(h)
        connector.bind_connector_metadata(meta)

        stored = _serialize_cache(torch.randn(2, 32))
        connector._client_request = Mock(return_value=self._ok_tuple(stored))

        encoder_cache = {}
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            connector.start_load_caches(encoder_cache)

        assert all(h in encoder_cache for h in hashes)

    def test_start_load_caches_none_metadata_warns(self, consumer_worker_connector):
        connector = consumer_worker_connector
        connector._connector_metadata = None

        encoder_cache = {}
        with patch.object(connector, '_get_connector_metadata', return_value=None):
            with patch('omni_npu.connector.ec_connector.network_connector.logger') as mock_logger:
                connector.start_load_caches(encoder_cache)
                mock_logger.warning.assert_called_once()

    def test_start_load_caches_async_uses_async_client(self, consumer_worker_connector):
        connector = consumer_worker_connector
        connector.ec_async_flag = True

        meta = ECNetworkConnectorMetadata()
        for h in ("h1", "h2"):
            meta.add_mm_hash(h)
        connector.bind_connector_metadata(meta)

        stored = _serialize_cache(torch.randn(2, 4))
        connector._async_client_request = Mock(return_value=self._ok_tuple(stored))

        encoder_cache = {}
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            connector.start_load_caches(encoder_cache)

        assert connector._async_client_request.call_count == 2
        assert "h1" in encoder_cache
        assert "h2" in encoder_cache


# =================== Tests: _load_caches_from_network =================== #

class TestLoadCachesFromNetwork:
    def test_empty_returns_immediately(self, consumer_worker_connector):
        c = consumer_worker_connector
        results, timed_out = c._load_caches_from_network([], "cpu", timeout=1.0)
        assert results == {}
        assert timed_out is False

    def test_serial_path_when_not_async(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = False
        result = _NetworkCacheLoadResult(torch.randn(2, 2), True, None)
        with patch.object(c, '_load_single_cache', return_value=result) as mock_load:
            results, timed_out = c._load_caches_from_network(
                ["h1", "h2"], "cpu", timeout=1.0,
            )
        assert mock_load.call_count == 2
        # Serial loads always use the synchronous client path.
        for call in mock_load.call_args_list:
            assert call.kwargs.get('use_async_workers') is False
        assert timed_out is False
        assert results["h1"] is result
        assert results["h2"] is result

    def test_async_path_submits_to_executor(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        result = _NetworkCacheLoadResult(torch.randn(2, 2), True, None)

        def fake_submit(fn, *args):
            fut = Future()
            fut.set_result(fn(*args))
            return fut

        with patch.object(c, '_load_single_cache', return_value=result) as mock_load, \
             patch.object(c._ec_load_executor, 'submit', side_effect=fake_submit) as mock_submit:
            results, timed_out = c._load_caches_from_network(
                ["h1", "h2"], "cpu", timeout=1.0,
            )
            assert mock_submit.call_count == 2
            # Async loads submit _load_single_cache with use_async_workers=True.
            for call in mock_submit.call_args_list:
                assert call.args[0] == mock_load
                assert call.args[2] == "cpu"
                assert call.args[3] is True
        assert timed_out is False
        assert results["h1"].ok
        assert results["h2"].ok

    def test_serial_timeout_marks_remaining(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = False
        results, timed_out = c._load_caches_from_network(
            ["h1", "h2"], "cpu", timeout=-1.0,
        )
        assert timed_out is True
        assert results["h1"].timed_out
        assert results["h2"].timed_out

    def test_async_timeout_cancels_pending(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True

        def pending_future(*args):
            return Future()

        with patch.object(c._ec_load_executor, 'submit',
                          side_effect=pending_future) as mock_submit:
            results, timed_out = c._load_caches_from_network(
                ["h1", "h2"], "cpu", timeout=-1.0,
            )
        assert timed_out is True
        assert results["h1"].timed_out
        assert results["h2"].timed_out
        assert mock_submit.call_count == 2

    def test_miss_result_passthrough(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = False
        miss = _NetworkCacheLoadResult(None, False, "status=miss")
        with patch.object(c, '_load_single_cache', return_value=miss):
            results, timed_out = c._load_caches_from_network(
                ["h1"], "cpu", timeout=1.0,
            )
        assert timed_out is False
        assert results["h1"] == miss

    def test_serial_single_hash(self, consumer_worker_connector):
        """Serial path with a real _client_request mock and real deserialize."""
        results, timed_out = _run_serial_load(consumer_worker_connector, ["h1"])
        assert timed_out is False
        assert "h1" in results
        assert results["h1"].ok is True
        assert isinstance(results["h1"].tensor, torch.Tensor)

    def test_serial_multiple_hashes(self, consumer_worker_connector):
        results, timed_out = _run_serial_load(consumer_worker_connector, ["h1", "h2"])
        assert timed_out is False
        assert len(results) == 2
        assert all(r.ok for r in results.values())

    def test_async_multiple_hashes(self, consumer_worker_connector):
        """Async path with a real _async_client_request mock and real deserialize."""
        c = consumer_worker_connector
        c.ec_async_flag = True
        tensor = torch.randn(2, 32)
        stored = _serialize_cache(tensor)
        resp = ECNetworkResponse(
            status=STATUS.OK, mode=stored.mode,
            dtype_code=stored.dtype_code, shape=stored.shape,
        )
        c._async_client_request = Mock(return_value=(resp, stored.payload, 0.1, 0.1))

        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            results, timed_out = c._load_caches_from_network(["h1", "h2", "h3"], "cpu", 10.0)
        assert timed_out is False
        assert len(results) == 3
        assert all(r.ok for r in results.values())

    def test_async_partial_miss(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        stored = _serialize_cache(torch.randn(2, 32, dtype=torch.bfloat16))
        ok_resp = ECNetworkResponse(
            status=STATUS.OK, mode=stored.mode,
            dtype_code=stored.dtype_code, shape=stored.shape,
        )
        miss_resp = ECNetworkResponse(status=STATUS.MISS)

        def mock_req(cmd, h):
            if "ok" in h:
                return (ok_resp, stored.payload, 0.1, 0.1)
            return (miss_resp, None, 0.0, 0.0)

        c._async_client_request = Mock(side_effect=mock_req)

        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            results, timed_out = c._load_caches_from_network(["ok1", "miss1"], "cpu", 10.0)
        assert timed_out is False
        assert results["ok1"].ok is True
        assert results["miss1"].ok is False


# =================== Tests: _load_single_cache =================== #

class TestLoadSingleCache:
    def test_success_sync(self, consumer_worker_connector):
        c = consumer_worker_connector
        tensor = torch.randn(2, 2)
        payload = b"serialized_bytes"
        resp = ECNetworkResponse(
            status=STATUS.OK, mode="plain", dtype_code=3, shape=(2, 2),
        )
        with patch.object(c, '_client_request', return_value=(resp, payload, 1.0, 0.5)), \
             patch('omni_npu.connector.ec_connector.network_connector._deserialize_cache',
                   return_value=tensor) as mock_deser:
            result = c._load_single_cache("h1", "cpu", use_async_workers=False)
        assert result.ok
        assert result.tensor is tensor
        assert result.error is None
        assert result.payload_bytes == len(payload)
        # Sync path calls _deserialize_cache directly with the header metadata.
        assert mock_deser.call_args[0][0] == payload
        assert mock_deser.call_args[0][3] == "plain"

    def test_success_async(self, consumer_worker_connector):
        c = consumer_worker_connector
        tensor = torch.randn(2, 2)
        payload = b"data"
        resp = ECNetworkResponse(
            status=STATUS.OK, mode="plain", dtype_code=3, shape=(2, 2),
        )

        def fake_submit(fn, *args, **kwargs):
            fut = Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

        with patch.object(c, '_async_client_request', return_value=(resp, payload, 1.0, 0.5)), \
             patch.object(c._deserialize_executor, 'submit', side_effect=fake_submit), \
             patch('omni_npu.connector.ec_connector.network_connector._deserialize_cache',
                   return_value=tensor):
            result = c._load_single_cache("h1", "cpu", use_async_workers=True)
        assert result.ok
        assert result.tensor is tensor
        assert result.payload_bytes == len(payload)

    def test_miss_response(self, consumer_worker_connector):
        c = consumer_worker_connector
        resp = ECNetworkResponse(status=STATUS.MISS)
        with patch.object(c, '_client_request', return_value=(resp, None, 0.0, 0.0)):
            result = c._load_single_cache("h1", "cpu", use_async_workers=False)
        assert not result.ok
        assert result.tensor is None
        assert "miss" in result.error

    def test_incomplete_header(self, consumer_worker_connector):
        c = consumer_worker_connector
        resp = ECNetworkResponse(status=STATUS.OK, mode=None, dtype_code=3, shape=None)
        with patch.object(c, '_client_request', return_value=(resp, b"data", 1.0, 0.5)):
            result = c._load_single_cache("h1", "cpu", use_async_workers=False)
        assert not result.ok
        assert "incomplete header" in result.error

    def test_exception_handling(self, consumer_worker_connector):
        c = consumer_worker_connector
        with patch.object(c, '_client_request', side_effect=RuntimeError("boom")):
            result = c._load_single_cache("h1", "cpu", use_async_workers=False)
        assert not result.ok
        assert "boom" in result.error

    def test_deserialize_exception(self, consumer_worker_connector):
        c = consumer_worker_connector
        resp = ECNetworkResponse(
            status=STATUS.OK, mode="plain", dtype_code=3, shape=(2, 2),
        )
        with patch.object(c, '_client_request', return_value=(resp, b"data", 1.0, 0.5)), \
             patch('omni_npu.connector.ec_connector.network_connector._deserialize_cache',
                   side_effect=RuntimeError("corrupt")):
            result = c._load_single_cache("h1", "cpu", use_async_workers=False)
        assert not result.ok
        assert "corrupt" in result.error

    def test_load_ok_empty_payload(self, consumer_worker_connector):
        c = consumer_worker_connector
        resp = ECNetworkResponse(status=STATUS.OK)
        with patch.object(c, '_client_request', return_value=(resp, None, 0.0, 0.0)):
            result = c._load_single_cache("h1", "cpu", use_async_workers=False)
        assert not result.ok
        assert "empty payload" in result.error


# =================== Tests: has_cache_item =================== #

class TestHasCacheItem:
    def test_producer_returns_false(self, producer_scheduler_connector, mock_request_3_hashes):
        connector = producer_scheduler_connector
        for feature in mock_request_3_hashes.mm_features:
            assert connector.has_cache_item(feature.identifier) is False

    def test_updates_need_loads(self, producer_scheduler_connector, mock_request_3_hashes):
        connector = producer_scheduler_connector
        hashes = [f.identifier for f in mock_request_3_hashes.mm_features]
        for h in hashes:
            connector.has_cache_item(h)
        for h in hashes:
            assert h in connector._mm_hashes_need_loads

    def test_consumer_uses_endpoints_recorded_by_ensure_cache_available(
            self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "consumer_check_hash"
        endpoint = "tcp://10.0.0.5:5000"
        request = MockRequest("req-c", [mm_hash],
                              ec_transfer_params={"mm_hash_endpoints": {mm_hash: endpoint}})

        connector.ensure_cache_available(request, 0)

        assert connector.has_cache_item(mm_hash) is True
        assert connector._mm_hash_endpoints.get(mm_hash) == endpoint

    def test_consumer_none_transfer_params(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "no_params_hash"
        request = MockRequest("req-no", [mm_hash], ec_transfer_params=None)

        connector.ensure_cache_available(request, 0)

        assert connector.has_cache_item(mm_hash) is False

    def test_consumer_ec_transfer_params_no_mm_hash_endpoints(
            self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "hash_no_endpoints"
        request = MockRequest("req-d", [mm_hash], ec_transfer_params={})

        connector.ensure_cache_available(request, 0)

        assert connector.has_cache_item(mm_hash) is False
        assert mm_hash not in connector._mm_hash_endpoints

    def test_consumer_ec_transfer_params_with_mm_hash_endpoints(
            self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "hash_with_ep"
        request = MockRequest("req-e", [mm_hash],
                              ec_transfer_params={"mm_hash_endpoints": {mm_hash: "tcp://1.2.3.4:5000"}})

        connector.ensure_cache_available(request, 0)

        assert connector.has_cache_item(mm_hash) is True
        assert connector._mm_hash_endpoints[mm_hash] == "tcp://1.2.3.4:5000"

    def test_unknown_identifier_is_registered_for_load(self, consumer_worker_connector):
        # Registration happens even without a cache, as it did when the
        # scheduler asked per request rather than per item.
        connector = consumer_worker_connector
        assert connector.has_cache_item("never_seen") is False
        assert "never_seen" in connector._mm_hashes_need_loads


# =================== Tests: build_connector_meta =================== #

class TestBuildConnectorMeta:
    def test_build_connector_meta_returns_metadata(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        connector._mm_hashes_need_loads.add("h1")

        scheduler_output = Mock(spec=SchedulerOutput)
        meta = connector.build_connector_meta(scheduler_output)

        assert isinstance(meta, ECNetworkConnectorMetadata)

    def test_build_connector_meta_includes_all_hashes(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        connector._mm_hashes_need_loads.add("h1")
        connector._mm_hashes_need_loads.add("h2")
        connector._mm_hashes_need_loads.add("h3")

        scheduler_output = Mock(spec=SchedulerOutput)
        meta = connector.build_connector_meta(scheduler_output)

        assert "h1" in meta.mm_hashes
        assert "h2" in meta.mm_hashes
        assert "h3" in meta.mm_hashes

    def test_build_connector_meta_clears_need_loads(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        connector._mm_hashes_need_loads.add("h1")

        scheduler_output = Mock(spec=SchedulerOutput)
        connector.build_connector_meta(scheduler_output)

        assert len(connector._mm_hashes_need_loads) == 0

    def test_build_connector_meta_empty_state(self, producer_scheduler_connector):
        connector = producer_scheduler_connector

        scheduler_output = Mock(spec=SchedulerOutput)
        meta = connector.build_connector_meta(scheduler_output)

        assert isinstance(meta, ECNetworkConnectorMetadata)
        assert len(meta.mm_hashes) == 0

    def test_build_connector_meta_adds_endpoint_for_producer(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        connector._mm_hashes_need_loads.add("h1")
        connector._endpoint = "tcp://127.0.0.1:5000"

        scheduler_output = Mock(spec=SchedulerOutput)
        meta = connector.build_connector_meta(scheduler_output)

        assert meta.mm_hash_endpoints["h1"] == "tcp://127.0.0.1:5000"

    def test_build_connector_meta_consumer_uses_mm_hash_endpoints(self, consumer_worker_connector):
        connector = consumer_worker_connector
        connector._mm_hashes_need_loads.add("h1")
        connector._mm_hash_endpoints["h1"] = "tcp://10.0.0.1:5000"

        scheduler_output = Mock(spec=SchedulerOutput)
        meta = connector.build_connector_meta(scheduler_output)

        assert meta.mm_hash_endpoints["h1"] == "tcp://10.0.0.1:5000"


# =================== Tests: update_connector_output / update_state_after_alloc =================== #

class TestUpdateMethods:
    def test_update_connector_output_returns_none(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        result = connector.update_connector_output(Mock())
        assert result is None

    def test_update_state_after_alloc_returns_none(self, producer_scheduler_connector,
                                                   mock_request_3_hashes):
        connector = producer_scheduler_connector
        result = connector.update_state_after_alloc(mock_request_3_hashes, index=0)
        assert result is None


# =================== Tests: _start_server_and_client =================== #

class TestStartServerAndClient:
    def test_producer_worker_starts_server(self):
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=True)
        assert connector._server_socket is not None

    def test_producer_scheduler_creates_client(self):
        connector = create_connector(role=ECConnectorRole.SCHEDULER, is_producer=True)
        assert connector._server_socket is None
        assert len(connector._client_sockets) > 0

    def test_consumer_creates_client(self):
        connector = create_connector(role=ECConnectorRole.WORKER, is_producer=False)
        assert len(connector._client_sockets) > 0


# =================== Tests: __del__ cleanup =================== #

class TestCleanup:
    def test_del_closes_client_sockets(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        mock_sock1 = MagicMock()
        mock_sock2 = MagicMock()
        connector._client_sockets["ep1"] = mock_sock1
        connector._client_sockets["ep2"] = mock_sock2

        connector.__del__()

        mock_sock1.close.assert_called_once_with(linger=0)
        mock_sock2.close.assert_called_once_with(linger=0)

    def test_del_closes_server_socket_if_set(self, producer_worker_connector):
        connector = producer_worker_connector
        mock_server_sock = MagicMock()
        connector._server_socket = mock_server_sock

        connector.__del__()

        mock_server_sock.close.assert_called_with(linger=0)

    def test_del_no_server_socket_no_error(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        assert connector._server_socket is None
        connector.__del__()

    def test_del_clears_server_running(self, producer_worker_connector):
        connector = producer_worker_connector
        connector._server_running.set()
        assert connector._server_running.is_set()

        mock_server_sock = MagicMock()
        connector._server_socket = mock_server_sock
        connector.__del__()

        assert not connector._server_running.is_set()

    def test_del_shuts_down_executors(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        connector._ec_load_executor = MagicMock()
        connector._deserialize_executor = MagicMock()

        connector.__del__()

        connector._ec_load_executor.shutdown.assert_called_once_with(wait=False)
        connector._deserialize_executor.shutdown.assert_called_once_with(wait=False)


# =================== Tests: TP broadcast load flow =================== #

class TestTPBroadcastFlow:
    def _make_tp_connector(self):
        config = make_mock_vllm_config(is_producer=False)
        config.parallel_config.tensor_parallel_size = 2
        c = create_connector(role=ECConnectorRole.WORKER, is_producer=False, vllm_config=config)
        c.tp_size = 2
        return c

    # ---- _get_mm_hashes_to_load ----

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_get_hashes_to_load_skips_local_hit(self, mock_dist, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        encoder_cache = {"h1": torch.randn(2, 4)}
        hashes = c._get_mm_hashes_to_load(
            ["h1", "h2"], encoder_cache, use_tp_broadcast=False,
        )
        assert hashes == ["h2"]
        mock_dist.all_reduce.assert_not_called()

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_get_hashes_to_load_tp_any_rank_needs(self, mock_dist, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        mock_dist.all_reduce.side_effect = _all_reduce_fill_one

        encoder_cache = {"h1": torch.randn(2, 4)}
        hashes = c._get_mm_hashes_to_load(
            ["h1", "h2"], encoder_cache, use_tp_broadcast=True,
        )
        # A MAX all-reduce of the need flags says another rank needs both.
        assert hashes == ["h1", "h2"]
        mock_dist.all_reduce.assert_called_once()

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_get_hashes_to_load_tp_all_ranks_have(self, mock_dist, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        mock_dist.all_reduce.side_effect = _all_reduce_fill_zero

        encoder_cache = {"h1": torch.randn(2, 4)}
        hashes = c._get_mm_hashes_to_load(
            ["h1"], encoder_cache, use_tp_broadcast=True,
        )
        assert hashes == []
        mock_dist.all_reduce.assert_called_once()

    # ---- _broadcast_and_store_caches ----

    def _ok_meta_row(self):
        row = [0] * META_WIDTH
        row[0] = 1  # META_OK_IDX
        row[1] = 2  # META_NDIM_IDX -> 2-D
        row[2] = 3  # META_DTYPE_IDX -> float32
        row[3] = 1  # META_DEVICE_IDX -> cpu
        row[4] = 2  # META_SHAPE_START_IDX -> 2
        row[5] = 2  # shape dim 1 -> 2
        return row

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_broadcast_and_store_rank0(self, mock_dist, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        tensor = torch.randn(2, 2)
        result = _NetworkCacheLoadResult(tensor, True, None)
        encoder_cache = {}
        count = c._broadcast_and_store_caches(
            encoder_cache, ["h1"], {"h1": result}, [self._ok_meta_row()],
        )
        assert count == 1
        got = encoder_cache.get("h1")
        assert got is not None
        assert torch.equal(got, tensor)
        mock_dist.broadcast.assert_called_once()

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_broadcast_and_store_non_rank0(self, mock_dist, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 1
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        encoder_cache = {}
        # Non-rank0 has no local result; it allocates an empty tensor that the
        # dist.broadcast is expected to fill in place.
        count = c._broadcast_and_store_caches(
            encoder_cache, ["h1"], {}, [self._ok_meta_row()],
        )
        assert count == 1
        got = encoder_cache.get("h1")
        assert got is not None
        assert got.shape == (2, 2)
        assert got.dtype == torch.float32
        mock_dist.broadcast.assert_called_once()

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_broadcast_and_store_skips_failed(self, mock_dist, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        encoder_cache = {}
        count = c._broadcast_and_store_caches(
            encoder_cache, ["h1"], {}, [[0] * META_WIDTH],
        )
        assert count == 0
        assert "h1" not in encoder_cache
        mock_dist.broadcast.assert_not_called()

    # ---- _store_load_results (non-TP path) ----

    def test_store_load_results(self, consumer_worker_connector):
        c = consumer_worker_connector
        tensor = torch.randn(2, 2)
        ok = _NetworkCacheLoadResult(tensor, True, None)
        fail = _NetworkCacheLoadResult(None, False, "miss")
        encoder_cache = {}
        count = c._store_load_results(
            encoder_cache, ["h1", "h2"], {"h1": ok, "h2": fail},
        )
        assert count == 1
        got = encoder_cache.get("h1")
        assert got is not None
        assert torch.equal(got, tensor)
        assert "h2" not in encoder_cache


# =================== Tests: start_load_caches (TP broadcast) =================== #

class TestStartLoadCachesTPBroadcast:
    def _make_tp_connector(self):
        config = make_mock_vllm_config(is_producer=False)
        config.parallel_config.tensor_parallel_size = 2
        c = create_connector(role=ECConnectorRole.WORKER, is_producer=False, vllm_config=config)
        c.tp_size = 2
        return c

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_non_rank0_receives_via_broadcast(self, mock_dist, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 1
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_tp.ranks = [0, 1]
        mock_get_tp.return_value = mock_tp

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")
        c.bind_connector_metadata(meta)

        def fake_broadcast(tensor, src=None, **kw):
            if tensor.ndim == 2 and tensor.size(1) == META_WIDTH:
                # Metadata buffer: mark h1 as a valid float32 (2,2) cpu tensor.
                tensor.fill_(0)
                tensor[0, 0] = 1
                tensor[0, 1] = 2
                tensor[0, 2] = 3
                tensor[0, 3] = 1
                tensor[0, 4] = 2
                tensor[0, 5] = 2
            elif tensor.numel() == 4:
                tensor.copy_(torch.arange(4, dtype=torch.float32).view(2, 2))
            return tensor

        mock_dist.broadcast.side_effect = fake_broadcast
        mock_dist.all_reduce.side_effect = _all_reduce_fill_one

        encoder_cache = {}
        c.start_load_caches(encoder_cache)

        assert "h1" in encoder_cache
        got = encoder_cache.get("h1")
        assert got is not None
        # Pin BOTH sides to cpu: some earlier module (e.g.
        # test_deepseek_scaling_rope.py) sets `torch.set_default_device("npu")`
        # at import time, which would put the bare torch.arange on npu:0 and
        # break torch.equal(got.cpu(), ...) with a device mismatch.
        assert torch.equal(
            got.cpu(),
            torch.arange(4, dtype=torch.float32).view(2, 2).cpu(),
        )

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_rank0_fetches_and_broadcasts(self, mock_dist, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_tp.ranks = [0, 1]
        mock_get_tp.return_value = mock_tp

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")
        c.bind_connector_metadata(meta)

        tensor = torch.randn(2, 2)
        stored = _serialize_cache(tensor)
        resp = ECNetworkResponse(
            status=STATUS.OK, mode=stored.mode,
            dtype_code=stored.dtype_code, shape=stored.shape,
        )
        c._client_request = Mock(return_value=(resp, stored.payload, 1.0, 0.5))

        mock_dist.all_reduce.side_effect = _all_reduce_fill_one
        mock_dist.broadcast.side_effect = _broadcast_passthrough

        encoder_cache = {}
        c.start_load_caches(encoder_cache)

        assert "h1" in encoder_cache
        assert torch.equal(encoder_cache.get("h1").cpu(), tensor.cpu())


# =================== Tests: Compression (backup union) =================== #

class TestCompression:
    """Cover _compress_payload_parallel/_decompress_payload_parallel round-trips."""

    def test_compress_decompress_plain(self):
        raw = os.urandom(1024)
        compressed, mode = _compress_payload_parallel(raw)
        assert mode == "plain"
        assert _decompress_payload_parallel(compressed, mode) == raw

    def test_compress_decompress_large(self):
        raw = os.urandom(10 * 1024 * 1024)  # 10MB
        compressed, mode = _compress_payload_parallel(raw)
        assert mode in ("plain", "chunked")
        assert _decompress_payload_parallel(compressed, mode) == raw

    def test_mode_consistency(self):
        """Compress then decompress with the returned mode must round-trip."""
        raw = os.urandom(9 * 1024 * 1024)  # 9MB
        comp, mode = _compress_payload_parallel(raw)
        decomp = _decompress_payload_parallel(comp, mode)
        assert decomp == raw


# =================== Tests: Concurrent Compression (backup union) =================== #

class TestConcurrentCompression:
    """Guard: parallel lz4 compress/decompress must not cross data between threads."""

    def test_concurrent_compress_independence(self):
        """8 threads compress different data simultaneously; results must not mix."""
        import concurrent.futures

        def worker(i):
            raw = os.urandom(100_000 + i * 1000)
            compressed, mode = _compress_payload_parallel(raw)
            return _decompress_payload_parallel(compressed, mode) == raw

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(worker, range(8)))
        assert all(results), "At least one thread's data was corrupted"

    def test_concurrent_compress_large_payloads(self):
        """4 threads compress 2MB payloads concurrently without crash."""
        import concurrent.futures

        def worker(i):
            raw = os.urandom(2 * 1024 * 1024)
            compressed, mode = _compress_payload_parallel(raw)
            return _decompress_payload_parallel(compressed, mode) == raw

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(4)))
        assert all(results)

    def test_lazy_pool_init_returns_same_instance(self):
        """Concurrent first-call to _get_lz4_pool must create only one pool."""
        import concurrent.futures
        import omni_npu.connector.ec_connector.network_connector as nc_module
        from omni_npu.connector.ec_connector.network_connector import _get_lz4_pool

        # Reset pool to force lazy init
        nc_module._LZ4_COMPRESS_POOL[0] = None

        results = []

        def worker():
            results.append(_get_lz4_pool(nc_module._LZ4_COMPRESS_POOL))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: worker(), range(8)))

        assert len(results) == 8
        assert all(r is results[0] for r in results), "Pool was created more than once"


# =================== Tests: Socket Pool (backup union) =================== #

class TestSocketPool:
    """Cover _checkout_async_socket / _return_async_socket pool mechanics."""

    def test_checkout_creates_on_empty(self, consumer_worker_connector):
        c = consumer_worker_connector
        mock_sock = MagicMock()
        with patch.object(c, '_create_async_socket', return_value=mock_sock):
            result = c._checkout_async_socket("tcp://test:5000")
        assert result is mock_sock
        assert c._async_sockets_live == 1

    def test_checkout_reuses_from_pool(self, consumer_worker_connector):
        c = consumer_worker_connector
        endpoint = "tcp://reuse:5000"
        mock_sock = MagicMock()
        c._checkout_async_socket(endpoint)  # creates pool
        c._return_async_socket(endpoint, mock_sock)
        with patch.object(c, '_create_async_socket') as mock_create:
            result = c._checkout_async_socket(endpoint)
        assert result is mock_sock
        mock_create.assert_not_called()

    def test_return_increases_pool(self, consumer_worker_connector):
        c = consumer_worker_connector
        endpoint = "tcp://ret:5000"
        c._checkout_async_socket(endpoint)  # init pool
        mock_sock = MagicMock()
        c._return_async_socket(endpoint, mock_sock)
        pool = c._async_socket_pools[endpoint]
        assert pool.qsize() == 1


# =================== Tests: Concurrent Socket Pool (backup union) =================== #

class TestConcurrentSocketPool:
    """Guard: concurrent checkout/return must keep _async_sockets_live accurate."""

    def test_concurrent_checkout_count(self, consumer_worker_connector):
        """8 threads checkout simultaneously; live must equal 8."""
        import concurrent.futures
        c = consumer_worker_connector
        endpoint = "tcp://concurrent-checkout:5000"
        sockets = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            sock = c._checkout_async_socket(endpoint)
            with lock:
                sockets.append(sock)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: worker(), range(8)))

        assert len(sockets) == 8
        assert c._async_sockets_live == 8

    def test_concurrent_checkout_return_cycle(self, consumer_worker_connector):
        """8 threads x 10 rounds of checkout/return; live must stabilize."""
        import concurrent.futures
        c = consumer_worker_connector
        endpoint = "tcp://cycle:5000"
        barrier = threading.Barrier(8)
        errors = []

        def worker():
            barrier.wait()
            try:
                for _ in range(10):
                    sock = c._checkout_async_socket(endpoint)
                    time.sleep(0.001)
                    c._return_async_socket(endpoint, sock)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: worker(), range(8)))

        assert not errors, f"Errors during cycle: {errors}"
        assert c._async_sockets_live <= 8, f"Live sockets leaked: {c._async_sockets_live}"


# =================== Tests: Concurrent Cache Store (backup union) =================== #

class TestConcurrentCacheStore:
    """Guard: concurrent save_caches must not corrupt _cache_store."""

    def test_concurrent_save_different_hashes(self, producer_worker_connector):
        """2 threads save different hashes simultaneously; both must be in store."""
        import concurrent.futures
        c = producer_worker_connector

        def save_hash(h):
            tensor = torch.randn(4, 64)
            c.save_caches({h: tensor}, h)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(save_hash, f"concurrent_h{i}") for i in range(2)]
            concurrent.futures.wait(futures)

        assert "concurrent_h0" in c._cache_store
        assert "concurrent_h1" in c._cache_store
        assert isinstance(c._cache_store["concurrent_h0"], _StoredCache)
        assert isinstance(c._cache_store["concurrent_h1"], _StoredCache)


# =================== Tests: Async Batch Load (backup union) =================== #

class TestAsyncBatchLoad:
    """Guard: async batch loading with ec_async_flag must load all hashes."""

    def test_async_batch_load_all_success(self, consumer_worker_connector):
        """8 hashes loaded via async executor; all must end up in encoder_cache."""
        c = consumer_worker_connector
        c.ec_async_flag = True

        tensor = torch.randn(2, 32)
        stored = _serialize_cache(tensor)
        resp = ECNetworkResponse(
            status=STATUS.OK, mode=stored.mode,
            dtype_code=stored.dtype_code, shape=stored.shape,
        )
        c._async_client_request = Mock(return_value=(resp, stored.payload, 0.1, 0.1))

        hashes = [f"batch_h{i}" for i in range(8)]
        meta = ECNetworkConnectorMetadata()
        for h in hashes:
            meta.add_mm_hash(h)
        c.bind_connector_metadata(meta)

        encoder_cache = {}
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            c.start_load_caches(encoder_cache, device='cpu', timeout=10.0)

        assert len(encoder_cache) == 8, f"Only {len(encoder_cache)} of 8 loaded"
        for h in hashes:
            assert h in encoder_cache
            assert isinstance(encoder_cache.get(h), torch.Tensor)

    def test_async_batch_partial_failure(self, consumer_worker_connector):
        """2 of 4 hashes MISS; only 2 must be in encoder_cache."""
        c = consumer_worker_connector
        c.ec_async_flag = True

        tensor = torch.randn(2, 32)
        stored = _serialize_cache(tensor)
        ok_resp = ECNetworkResponse(
            status=STATUS.OK, mode=stored.mode,
            dtype_code=stored.dtype_code, shape=stored.shape,
        )
        miss_resp = ECNetworkResponse(status=STATUS.MISS)

        def mock_request(cmd, h):
            if h in ("ok1", "ok2"):
                return (ok_resp, stored.payload, 0.1, 0.1)
            return (miss_resp, None, 0.0, 0.0)

        c._async_client_request = Mock(side_effect=mock_request)

        meta = ECNetworkConnectorMetadata()
        for h in ["ok1", "ok2", "miss1", "miss2"]:
            meta.add_mm_hash(h)
        c.bind_connector_metadata(meta)

        encoder_cache = {}
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            c.start_load_caches(encoder_cache, device='cpu', timeout=10.0)

        assert "ok1" in encoder_cache
        assert "ok2" in encoder_cache
        assert "miss1" not in encoder_cache
        assert "miss2" not in encoder_cache


# =================== Tests: Cache Meta Row (backup union) =================== #

class TestEncodeDecodeMetaRow:
    """Direct tests for _encode_cache_meta_row / _decode_cache_meta_row."""

    def test_encode_valid_tensor(self):
        tensor = torch.randn(4, 8, 16, device="cpu")
        row = ECNetworkConnector._encode_cache_meta_row(tensor, load_ok=True)
        assert row[META_OK_IDX] == 1
        assert row[META_NDIM_IDX] == 3
        assert row[META_DTYPE_IDX] == DTYPE_TO_CODE[torch.float32]
        assert row[META_DEVICE_IDX] == DEVICE_TYPE_TO_CODE["cpu"]
        assert row[META_SHAPE_START_IDX] == 4
        assert row[META_SHAPE_START_IDX + 1] == 8
        assert row[META_SHAPE_START_IDX + 2] == 16

    def test_encode_not_ok_returns_zeros(self):
        tensor = torch.randn(4, 8)
        row = ECNetworkConnector._encode_cache_meta_row(tensor, load_ok=False)
        assert row[META_OK_IDX] == 0
        assert all(v == 0 for v in row)

    def test_encode_none_tensor_returns_zeros(self):
        row = ECNetworkConnector._encode_cache_meta_row(None, load_ok=True)
        assert row[META_OK_IDX] == 0

    def test_encode_too_many_dims_raises(self):
        shape = tuple(1 for _ in range(20))
        tensor = torch.randn(*shape, device="cpu")
        with pytest.raises(ValueError, match="exceeds max"):
            ECNetworkConnector._encode_cache_meta_row(tensor, True)

    def test_encode_decode_roundtrip(self):
        tensor = torch.randn(2, 32, dtype=torch.float16, device="cpu")
        row = ECNetworkConnector._encode_cache_meta_row(tensor, True)
        ok, shape, dtype, device_type = ECNetworkConnector._decode_cache_meta_row(row)
        assert ok is True
        assert shape == (2, 32)
        assert dtype == torch.float16
        assert device_type == "cpu"

    def test_decode_not_ok(self):
        row = [0] * META_WIDTH
        ok, shape, dtype, device_type = ECNetworkConnector._decode_cache_meta_row(row)
        assert ok is False
        assert shape is None

    def test_decode_invalid_ndim_raises(self):
        row = [0] * META_WIDTH
        row[META_OK_IDX] = 1
        row[META_NDIM_IDX] = 99
        with pytest.raises(ValueError, match="ndim"):
            ECNetworkConnector._decode_cache_meta_row(row)

    def test_decode_negative_shape_raises(self):
        row = [0] * META_WIDTH
        row[META_OK_IDX] = 1
        row[META_NDIM_IDX] = 1
        row[META_SHAPE_START_IDX] = -5
        with pytest.raises(ValueError, match="shape"):
            ECNetworkConnector._decode_cache_meta_row(row)

    def test_decode_invalid_dtype_raises(self):
        row = [0] * META_WIDTH
        row[META_OK_IDX] = 1
        row[META_NDIM_IDX] = 1
        row[META_DTYPE_IDX] = 999
        with pytest.raises(ValueError, match="dtype code"):
            ECNetworkConnector._decode_cache_meta_row(row)


# =================== Tests: Decompress Error Paths (backup union) =================== #

class TestDecompressErrors:
    """Cover error branches in _decompress_payload_parallel."""

    def test_truncated_chunk_header(self):
        """Payload shorter than the 4-byte chunk-count header -> ValueError."""
        with pytest.raises(ValueError, match="missing the chunk count header"):
            _decompress_payload_parallel(b"abc", "chunked")

    def test_truncated_chunk_length(self):
        """Chunk count says 2 but only 1 chunk length present -> ValueError."""
        import struct
        header = struct.pack("!I", 2)  # claim 2 chunks
        # Only provide 1 chunk length (4 bytes) + tiny data
        partial = header + struct.pack("!I", 4) + b"data"
        with pytest.raises(ValueError, match="truncated"):
            _decompress_payload_parallel(partial, "chunked")

    def test_truncated_chunk_data(self):
        """Chunk length says 10 bytes but only 4 available -> ValueError."""
        import struct
        header = struct.pack("!I", 1)  # 1 chunk
        chunk_header = struct.pack("!I", 10)  # claim 10 bytes
        data = b"abcd"  # only 4 bytes
        with pytest.raises(ValueError, match="truncated"):
            _decompress_payload_parallel(header + chunk_header + data, "chunked")


# =================== Tests: Deserialize Error Paths (backup union) =================== #

class TestDeserializeErrors:
    """Cover error branches in _deserialize_cache."""

    def test_size_mismatch_raises(self):
        """Decompressed data size doesn't match expected -> ValueError."""
        raw = b"\x00" * 10  # arbitrary wrong size
        compressed = _compress_payload_parallel(raw)[0]
        with pytest.raises(ValueError, match="size mismatch"):
            _deserialize_cache(compressed, torch.float32, (4, 64), "plain", device="cpu")

    def test_empty_tensor(self):
        """Shape with 0 numel -> uses torch.empty, no frombuffer."""
        tensor = torch.empty(0, 10)
        stored = _serialize_cache(tensor)
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            result = _deserialize_cache(
                stored.payload, tensor.dtype, tuple(tensor.shape), stored.mode, device='cpu')
        assert result.shape == (0, 10)
        assert result.dtype == torch.float32

    def test_device_move_to_cpu(self):
        """Tensor on cpu, target cpu -> no move needed (covers the else branch)."""
        tensor = torch.randn(4, 8)
        stored = _serialize_cache(tensor)
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            result = _deserialize_cache(
                stored.payload, tensor.dtype, tuple(tensor.shape), stored.mode, device='cpu')
        assert result.device.type == 'cpu'

    def test_device_move_explicit(self):
        """Explicit device='cpu' with cpu tensor -> device_move branch covered."""
        tensor = torch.randn(2, 4)
        stored = _serialize_cache(tensor)
        result = _deserialize_cache(
            stored.payload, tensor.dtype, tuple(tensor.shape), stored.mode, device='cpu')
        assert result.device.type == 'cpu'


# =================== Tests: Meta Row Edge Cases (backup union) =================== #

class TestMetaRowEdgeCases:
    """Cover error branches in _encode/_decode_cache_meta_row."""

    def test_encode_unsupported_dtype(self):
        """dtype not in DTYPE_TO_CODE -> ValueError."""
        class FakeDtype:
            pass
        tensor = MagicMock()
        tensor.shape = (4, 8)
        tensor.dtype = FakeDtype()
        tensor.device.type = "cpu"
        with pytest.raises(ValueError, match="Unsupported.*dtype"):
            ECNetworkConnector._encode_cache_meta_row(tensor, True)

    def test_encode_unsupported_device(self):
        """device type not in DEVICE_TYPE_TO_CODE -> ValueError."""
        tensor = MagicMock()
        tensor.shape = (4, 8)
        tensor.dtype = torch.float32
        tensor.device.type = "tpu"  # not in mapping
        with pytest.raises(ValueError, match="Unsupported.*device"):
            ECNetworkConnector._encode_cache_meta_row(tensor, True)

    def test_decode_invalid_width(self):
        """Row with wrong width -> ValueError."""
        with pytest.raises(ValueError, match="width"):
            ECNetworkConnector._decode_cache_meta_row([0, 1, 2])  # too short

    def test_decode_unsupported_device_code(self):
        """Device code not in CODE_TO_DEVICE_TYPE -> ValueError."""
        row = [0] * META_WIDTH
        row[META_OK_IDX] = 1
        row[META_NDIM_IDX] = 1
        row[META_DTYPE_IDX] = DTYPE_TO_CODE[torch.float32]
        row[META_DEVICE_IDX] = 999
        with pytest.raises(ValueError, match="device code"):
            ECNetworkConnector._decode_cache_meta_row(row)


# =================== Tests: Server Loop Handling (backup union) =================== #

class TestServerLoopHandling:
    """Cover _server_loop actual request dispatch (lines 644-649)."""

    def test_server_loop_dispatches_get(self, producer_worker_connector):
        """Server loop receives a valid GET, dispatches and sends the response."""
        c = producer_worker_connector
        c._cache_store["dispatch_h"] = _make_stored_cache()

        request = ECNetworkRequest(command=STATUS.GET, mm_hash="dispatch_h")
        payload = msgspec.msgpack.encode(request)
        messages = [b"identity", b"", payload]

        mock_socket = MagicMock()
        c._server_socket = mock_socket

        call_count = [0]

        def is_set():
            call_count[0] += 1
            return call_count[0] <= 1

        c._server_running = MagicMock()
        c._server_running.is_set.side_effect = is_set

        mock_poller = MagicMock()
        mock_poller.poll.return_value = {mock_socket: zmq.POLLIN}
        mock_socket.recv_multipart.side_effect = [messages] + [zmq.Again()]

        with patch('omni_npu.connector.ec_connector.network_connector.zmq.Poller',
                   return_value=mock_poller):
            c._server_loop()

        mock_socket.send_multipart.assert_called_once()
        frames = mock_socket.send_multipart.call_args[0][0]
        assert len(frames) == 4  # [identity, empty, header, payload]


# =================== Tests: Socket Pool Edge Cases (backup union) =================== #

class TestSocketPoolEdgeCases:
    """Cover socket pool backpressure + exception paths."""

    def test_checkout_at_cap_blocks(self, consumer_worker_connector):
        """When _async_sockets_live >= CAP, checkout blocks on pool.get()."""
        c = consumer_worker_connector
        c._async_sockets_live = EC_ASYNC_SOCKET_CAP
        endpoint = "tcp://cap:5000"
        # Pre-fill pool so checkout can get one without creating
        import queue as qmod
        pool = qmod.Queue()
        spare_sock = MagicMock()
        pool.put(spare_sock)
        c._async_socket_pools[endpoint] = pool

        result = c._checkout_async_socket(endpoint)
        assert result is spare_sock  # got from pool, not created

    def test_return_to_none_pool_closes(self, consumer_worker_connector):
        """Returning to a non-existent endpoint pool -> close socket."""
        c = consumer_worker_connector
        mock_sock = MagicMock()
        c._return_async_socket("tcp://nonexistent:9999", mock_sock)
        mock_sock.close.assert_called_once_with(linger=0)

    def test_async_request_exception_closes_socket(self, consumer_worker_connector):
        """Exception in send_and_recv -> socket closed, live decremented."""
        c = consumer_worker_connector
        mock_sock = MagicMock()
        mock_sock.send_multipart.side_effect = RuntimeError("boom")
        c._async_sockets_live = 1

        with patch.object(c, '_checkout_async_socket', return_value=mock_sock):
            with pytest.raises(RuntimeError, match="boom"):
                c._async_client_request(STATUS.GET, "h1")

        mock_sock.close.assert_called_once_with(linger=0)
        assert c._async_sockets_live == 0


# =================== Tests: Load Timeout and Errors (backup union) =================== #

class TestLoadTimeoutAndErrors:
    """Cover serial timeout + async timeout/exception paths."""

    def test_serial_timeout(self, consumer_worker_connector):
        """Serial path: timeout=0 -> all hashes get _timeout_result."""
        c = consumer_worker_connector
        c.ec_async_flag = False
        results, timed_out = c._load_caches_from_network(["h1", "h2"], "cpu", 0.0)
        assert timed_out is True
        assert all(r.timed_out for r in results.values())

    def test_async_timeout(self, consumer_worker_connector):
        """Async path: timeout=0 -> futures cancelled, _timeout_result."""
        c = consumer_worker_connector
        c.ec_async_flag = True
        # Fail fast so the executor workers never touch a real socket.
        c._async_client_request = Mock(
            return_value=(ECNetworkResponse(status=STATUS.ERROR), None, 0.0, 0.0))
        results, timed_out = c._load_caches_from_network(["h1", "h2"], "cpu", 0.0)
        assert timed_out is True
        assert all(r.timed_out for r in results.values())

    def test_async_worker_exception(self, consumer_worker_connector):
        """Async path: worker raises exception -> captured, not propagated."""
        c = consumer_worker_connector
        c.ec_async_flag = True
        c._async_client_request = Mock(side_effect=RuntimeError("worker crash"))

        results, timed_out = c._load_caches_from_network(["h1", "h2"], "cpu", 10.0)
        assert timed_out is False
        for r in results.values():
            assert r.ok is False


# =================== Tests: Store Load Results (backup union) =================== #

class TestStoreLoadResults:
    """Cover _store_load_results edge cases."""

    def test_missing_tensor_raises(self, consumer_worker_connector):
        """Result with ok=True but tensor=None -> RuntimeError."""
        c = consumer_worker_connector
        bad_result = _NetworkCacheLoadResult(None, True, None)
        with pytest.raises(RuntimeError, match="missing"):
            c._store_load_results({}, ["h1"], {"h1": bad_result})


# =================== Tests: Start Load Caches Edge Cases (backup union) =================== #

class TestStartLoadCachesEdgeCases:
    """Cover start_load_caches edge branches."""

    def test_wrong_metadata_type_raises(self, consumer_worker_connector):
        """metadata is not ECNetworkConnectorMetadata -> TypeError."""
        c = consumer_worker_connector
        wrong_meta = Mock()
        with patch.object(c, '_get_connector_metadata', return_value=wrong_meta):
            with pytest.raises(TypeError, match="ECNetworkConnectorMetadata"):
                c.start_load_caches({})
