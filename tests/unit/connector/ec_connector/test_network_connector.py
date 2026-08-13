# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ECNetworkConnector."""

import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
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
    EC_CACHE_KEY,
    MB,
    GB,
    ECNetworkRequest,
    ECNetworkResponse,
    ECNetworkConnectorMetadata,
    ECNetworkConnector,
    ZmqRequestHelper,
    _get_local_ip,
    _serialize_cache,
    _deserialize_cache,
)


# =================== Helper Classes =================== #

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

    def test_ec_cache_key(self):
        assert EC_CACHE_KEY == "ec_cache"

    def test_mb_gb(self):
        assert MB == 1024 * 1024
        assert GB == 1024 * MB


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
    def test_create_response_without_payload(self):
        resp = ECNetworkResponse(status=STATUS.MISS)
        assert resp.status == STATUS.MISS
        assert resp.payload is None

    def test_create_response_with_payload(self):
        payload = b"some_data"
        resp = ECNetworkResponse(status=STATUS.OK, payload=payload)
        assert resp.status == STATUS.OK
        assert resp.payload == payload

    def test_msgpack_roundtrip_no_payload(self):
        resp = ECNetworkResponse(status=STATUS.HIT)
        encoded = msgspec.msgpack.encode(resp)
        decoded = msgspec.msgpack.decode(encoded, type=ECNetworkResponse)
        assert decoded.status == STATUS.HIT
        assert decoded.payload is None

    def test_msgpack_roundtrip_with_payload(self):
        payload = b"\x01\x02\x03\x04"
        resp = ECNetworkResponse(status=STATUS.OK, payload=payload)
        encoded = msgspec.msgpack.encode(resp)
        decoded = msgspec.msgpack.decode(encoded, type=ECNetworkResponse)
        assert decoded.status == STATUS.OK
        assert decoded.payload == payload

    def test_all_statuses(self):
        for status in [STATUS.HIT, STATUS.OK, STATUS.MISS, STATUS.ERROR]:
            resp = ECNetworkResponse(status=status)
            assert resp.status == status


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
        resp = ECNetworkResponse(status=STATUS.OK, payload=b"data")
        encoded = msgspec.msgpack.encode(resp)
        reply = [ZMQ_EMPTY_FRAME, encoded]
        result = ZmqRequestHelper.decode_response(reply)
        assert result.status == STATUS.OK
        assert result.payload == b"data"

    def test_decode_response_too_few_frames(self):
        result = ZmqRequestHelper.decode_response([b"only_one"])
        assert result.status == STATUS.ERROR

    def test_decode_response_non_empty_delimiter(self):
        resp = ECNetworkResponse(status=STATUS.OK)
        encoded = msgspec.msgpack.encode(resp)
        reply = [b"nonempty", encoded]
        result = ZmqRequestHelper.decode_response(reply)
        assert result.status == STATUS.ERROR

    def test_decode_response_invalid_msgpack(self):
        reply = [ZMQ_EMPTY_FRAME, b"not_valid_msgpack"]
        result = ZmqRequestHelper.decode_response(reply)
        assert result.status == STATUS.ERROR

    def test_send_and_recv_success(self):
        mock_socket = MagicMock()
        resp = ECNetworkResponse(status=STATUS.HIT)
        encoded = msgspec.msgpack.encode(resp)
        mock_socket.recv_multipart.return_value = [ZMQ_EMPTY_FRAME, encoded]

        result = ZmqRequestHelper.send_and_recv(mock_socket, b"payload")
        assert result.status == STATUS.HIT
        mock_socket.send_multipart.assert_called_once_with([ZMQ_EMPTY_FRAME, b"payload"])

    def test_send_and_recv_zmq_again(self):
        mock_socket = MagicMock()
        mock_socket.send_multipart.side_effect = zmq.Again()
        result = ZmqRequestHelper.send_and_recv(mock_socket, b"payload")
        assert result.status == STATUS.ERROR

    def test_send_and_recv_recv_zmq_again(self):
        mock_socket = MagicMock()
        mock_socket.recv_multipart.side_effect = zmq.Again()
        result = ZmqRequestHelper.send_and_recv(mock_socket, b"payload")
        assert result.status == STATUS.ERROR


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
    def test_serialize_returns_bytes(self):
        tensor = torch.randn(10, 768)
        result = _serialize_cache(tensor)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_serialize_detaches_gradient(self):
        tensor = torch.randn(10, 768, requires_grad=True)
        result = _serialize_cache(tensor)
        assert isinstance(result, bytes)

    def test_deserialize_returns_tensor(self):
        tensor = torch.randn(5, 10)
        serialized = _serialize_cache(tensor)
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as mock_platform:
            mock_platform.device_type = 'cpu'
            restored = _deserialize_cache(serialized)
        assert isinstance(restored, torch.Tensor)

    def test_serialize_deserialize_roundtrip(self):
        original = torch.randn(8, 512)
        serialized = _serialize_cache(original)
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as mock_platform:
            mock_platform.device_type = 'cpu'
            restored = _deserialize_cache(serialized)
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
                restored = _deserialize_cache(serialized)
            assert torch.equal(tensor.cpu(), restored.cpu())
            assert tensor.shape == restored.shape


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
        assert isinstance(connector._ec_load_lock, type(threading.Lock()))

    def test_init_async_state(self):
        connector = create_connector()
        assert isinstance(connector._pending_ec_loads, set)
        assert isinstance(connector._finished_ec_loads, OrderedDict)
        assert isinstance(connector._failed_ec_loads, OrderedDict)
        assert len(connector._pending_ec_loads) == 0

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
        c._cache_store["a"] = b"x" * 50
        c._cache_store_bytes = 50
        c._evict_cache_if_needed(60)
        assert "a" not in c._cache_store
        assert c._cache_store_bytes == 0

    def test_evict_cache_multiple_entries(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store_max_bytes = 100
        c._cache_store["a"] = b"x" * 40
        c._cache_store["b"] = b"y" * 40
        c._cache_store_bytes = 80
        c._evict_cache_if_needed(30)
        assert c._cache_store_bytes + 30 <= c._cache_store_max_bytes

    def test_evict_cache_no_eviction_needed(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store_max_bytes = 1000
        c._cache_store["a"] = b"x" * 10
        c._cache_store_bytes = 10
        c._evict_cache_if_needed(10)
        assert "a" in c._cache_store
        assert c._cache_store_bytes == 10

    def test_trim_history_set(self, producer_worker_connector):
        c = producer_worker_connector
        od = OrderedDict()
        for i in range(200):
            od[f"key_{i}"] = None
        c._trim_history_set(od, max_size=100)
        assert len(od) == 100
        assert "key_100" in od
        assert "key_0" not in od

    def test_trim_history_set_default_max(self, producer_worker_connector):
        c = producer_worker_connector
        od = OrderedDict()
        for i in range(10):
            od[f"key_{i}"] = None
        c._trim_history_set(od)
        assert len(od) == 10


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

    def test_ec_use_tp_leader_zmq_async_producer_returns_false(self, producer_worker_connector):
        assert producer_worker_connector._ec_use_tp_leader_zmq_async() is False

    def test_ec_use_tp_leader_zmq_async_tp_size_1_returns_false(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.tp_size = 1
        c.ec_async_flag = True
        assert c._ec_use_tp_leader_zmq_async() is False

    def test_ec_use_tp_leader_zmq_async_not_async_returns_false(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.tp_size = 2
        c.ec_async_flag = False
        assert c._ec_use_tp_leader_zmq_async() is False

    @patch('omni_npu.connector.ec_connector.network_connector.torch.distributed.is_initialized',
           return_value=True)
    def test_ec_use_tp_leader_zmq_async_all_conditions_met(self, mock_init, consumer_worker_connector):
        c = consumer_worker_connector
        c.tp_size = 2
        c.ec_async_flag = True
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": True,
            }.get(key, default)
        )
        assert c._ec_use_tp_leader_zmq_async() is True


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
        connector._cache_store[mm_hash] = cache_data

        request = ECNetworkRequest(command=STATUS.GET, mm_hash=mm_hash)
        payload = msgspec.msgpack.encode(request)
        messages = [b"identity", b"", payload]

        calls = self._run_server_loop_once(connector, messages)
        assert len(calls) == 1
        sent = calls[0][0][0]
        response = msgspec.msgpack.decode(sent[2], type=ECNetworkResponse)
        assert response.status == STATUS.OK
        assert response.payload == cache_data

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
        c._cache_store["h1"] = b"data"
        req = ECNetworkRequest(command=STATUS.HAS, mm_hash="h1")
        resp = c._handle_server_request(req)
        assert resp.status == STATUS.HIT

    def test_has_miss(self, producer_worker_connector):
        c = producer_worker_connector
        req = ECNetworkRequest(command=STATUS.HAS, mm_hash="missing")
        resp = c._handle_server_request(req)
        assert resp.status == STATUS.MISS

    def test_get_hit(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store["h1"] = b"payload"
        req = ECNetworkRequest(command=STATUS.GET, mm_hash="h1")
        resp = c._handle_server_request(req)
        assert resp.status == STATUS.OK
        assert resp.payload == b"payload"

    def test_get_miss(self, producer_worker_connector):
        c = producer_worker_connector
        req = ECNetworkRequest(command=STATUS.GET, mm_hash="missing")
        resp = c._handle_server_request(req)
        assert resp.status == STATUS.MISS

    def test_unknown_command(self, producer_worker_connector):
        c = producer_worker_connector
        req = ECNetworkRequest(command=STATUS.OK, mm_hash="h")
        resp = c._handle_server_request(req)
        assert resp.status == STATUS.ERROR

    def test_get_touches_cache(self, producer_worker_connector):
        c = producer_worker_connector
        c._cache_store["a"] = b"1"
        c._cache_store["b"] = b"2"
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
        resp = ECNetworkResponse(status=STATUS.OK, payload=b"data")
        c._send_server_response(b"client_id", resp)
        mock_socket.send_multipart.assert_called_once()
        args = mock_socket.send_multipart.call_args[0][0]
        assert args[0] == b"client_id"
        assert args[1] == ZMQ_EMPTY_FRAME
        decoded = msgspec.msgpack.decode(args[2], type=ECNetworkResponse)
        assert decoded.status == STATUS.OK


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
        mock_sock.recv_multipart.return_value = [b"", reply_data]
        connector._client_sockets[endpoint] = mock_sock

        result = connector._client_request(STATUS.HAS, "test_hash")
        assert result.status == STATUS.HIT
        mock_sock.send_multipart.assert_called_once()

    def test_client_request_miss(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        response = ECNetworkResponse(status=STATUS.MISS)
        mock_sock.recv_multipart.return_value = [b"", msgspec.msgpack.encode(response)]
        connector._client_sockets[endpoint] = mock_sock

        result = connector._client_request(STATUS.HAS, "nonexistent_hash")
        assert result.status == STATUS.MISS

    def test_client_request_ok_with_payload(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        payload = b"cached_tensor_data"
        response = ECNetworkResponse(status=STATUS.OK, payload=payload)
        mock_sock.recv_multipart.return_value = [b"", msgspec.msgpack.encode(response)]
        connector._client_sockets[endpoint] = mock_sock

        result = connector._client_request(STATUS.GET, "test_hash")
        assert result.status == STATUS.OK
        assert result.payload == payload

    def test_client_request_timeout_returns_error(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        mock_sock.send_multipart.side_effect = zmq.Again()
        connector._client_sockets[endpoint] = mock_sock

        result = connector._client_request(STATUS.HAS, "test_hash")
        assert result.status == STATUS.ERROR

    def test_client_request_recv_timeout_returns_error(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        endpoint = connector._endpoint
        mock_sock = MagicMock()
        mock_sock.recv_multipart.side_effect = zmq.Again()
        connector._client_sockets[endpoint] = mock_sock

        result = connector._client_request(STATUS.GET, "test_hash")
        assert result.status == STATUS.ERROR

    def test_client_request_uses_mm_hash_endpoint(self, producer_scheduler_connector):
        connector = producer_scheduler_connector
        specific_endpoint = "tcp://10.0.0.2:5001"
        connector._mm_hash_endpoints["special_hash"] = specific_endpoint
        mock_sock = MagicMock()
        response = ECNetworkResponse(status=STATUS.HIT)
        mock_sock.recv_multipart.return_value = [b"", msgspec.msgpack.encode(response)]
        connector._client_sockets[specific_endpoint] = mock_sock

        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_sock), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000):
            result = connector._client_request(STATUS.HAS, "special_hash")
        assert result.status == STATUS.HIT

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
        resp = ECNetworkResponse(status=STATUS.OK, payload=b"data")
        encoded = msgspec.msgpack.encode(resp)
        mock_socket = MagicMock()
        mock_socket.recv_multipart.return_value = [ZMQ_EMPTY_FRAME, encoded]

        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_socket), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000):
            result = c._async_client_request(STATUS.GET, "h1")

        assert result.status == STATUS.OK
        assert result.payload == b"data"
        mock_socket.close.assert_called_once_with(linger=0)

    def test_async_client_request_error(self, consumer_worker_connector):
        c = consumer_worker_connector
        mock_socket = MagicMock()
        mock_socket.send_multipart.side_effect = zmq.Again()

        with patch('omni_npu.connector.ec_connector.network_connector.make_zmq_socket',
                   return_value=mock_socket), \
             patch('omni_npu.connector.ec_connector.network_connector.envs',
                   VLLM_RPC_TIMEOUT=5000), \
             patch('omni_npu.connector.ec_connector.network_connector.logger') as mock_logger:
            result = c._async_client_request(STATUS.GET, "h1")

        assert result.status == STATUS.ERROR
        mock_socket.close.assert_called_once_with(linger=0)
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
        assert isinstance(connector._cache_store[mm_hash], bytes)

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
        assert connector._cache_store_bytes == len(connector._cache_store[mm_hash])

    def test_save_caches_evicts_when_full(self, producer_worker_connector):
        connector = producer_worker_connector
        connector._cache_store_max_bytes = 100

        encoder_cache = {"h1": torch.randn(4, 64)}
        connector.save_caches(encoder_cache, "h1")

        encoder_cache["h2"] = torch.randn(4, 64)
        connector.save_caches(encoder_cache, "h2")

        assert connector._cache_store_bytes <= connector._cache_store_max_bytes + len(connector._cache_store.get("h2", b""))

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
    def test_start_load_caches_loads_tensor(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "load_hash"
        tensor = torch.randn(4, 64)
        payload = _serialize_cache(tensor)

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        mock_response = ECNetworkResponse(status=STATUS.OK, payload=payload)
        connector._client_request = Mock(return_value=mock_response)

        encoder_cache = {}
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            connector.start_load_caches(encoder_cache)

        assert mm_hash in encoder_cache
        assert isinstance(encoder_cache[mm_hash], torch.Tensor)

    def test_start_load_caches_skips_existing(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "existing_hash"
        existing_tensor = torch.randn(4, 64)

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        connector._client_request = Mock(return_value=ECNetworkResponse(status=STATUS.OK, payload=b"data"))

        encoder_cache = {mm_hash: existing_tensor}
        connector.start_load_caches(encoder_cache)

        connector._client_request.assert_not_called()
        assert torch.equal(encoder_cache[mm_hash], existing_tensor)

    def test_start_load_caches_handles_miss(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "miss_hash"

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        connector._client_request = Mock(return_value=ECNetworkResponse(status=STATUS.MISS))

        encoder_cache = {}
        connector.start_load_caches(encoder_cache)
        assert mm_hash not in encoder_cache

    def test_start_load_caches_handles_error_status(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "error_hash"

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(meta)

        connector._client_request = Mock(return_value=ECNetworkResponse(status=STATUS.ERROR))

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
            return_value=ECNetworkResponse(status=STATUS.OK, payload=None))

        encoder_cache = {}
        connector.start_load_caches(encoder_cache)
        assert mm_hash not in encoder_cache

    def test_start_load_caches_uses_endpoint_from_metadata(self, consumer_worker_connector):
        connector = consumer_worker_connector
        mm_hash = "endpoint_hash"

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash(mm_hash, endpoint="tcp://10.0.0.5:5000")
        connector.bind_connector_metadata(meta)

        payload = _serialize_cache(torch.randn(2, 32))
        connector._client_request = Mock(
            return_value=ECNetworkResponse(status=STATUS.OK, payload=payload))

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

        payload = _serialize_cache(torch.randn(2, 32))
        connector._client_request = Mock(
            return_value=ECNetworkResponse(status=STATUS.OK, payload=payload))

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

    def test_start_load_caches_async_delegates(self, consumer_worker_connector):
        connector = consumer_worker_connector
        connector.ec_async_flag = True

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")
        connector.bind_connector_metadata(meta)

        with patch.object(connector, '_async_submit_loads') as mock_async:
            connector.start_load_caches({})
            mock_async.assert_called_once()


# =================== Tests: _async_submit_loads =================== #

class TestAsyncSubmitLoads:
    def test_submits_new_hashes(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")
        meta.add_mm_hash("h2")

        with patch.object(c._ec_load_executor, 'submit') as mock_submit:
            c._async_submit_loads("cpu", {}, meta)

        assert mock_submit.call_count == 2

    def test_skips_already_pending(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        c._pending_ec_loads.add("h1")
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")

        with patch.object(c._ec_load_executor, 'submit') as mock_submit:
            c._async_submit_loads("cpu", {}, meta)

        mock_submit.assert_not_called()

    def test_skips_already_finished(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        c._finished_ec_loads["h1"] = None
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")

        with patch.object(c._ec_load_executor, 'submit') as mock_submit:
            c._async_submit_loads("cpu", {}, meta)

        mock_submit.assert_not_called()

    def test_skips_already_in_encoder_cache(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")

        with patch.object(c._ec_load_executor, 'submit') as mock_submit:
            c._async_submit_loads("cpu", {"h1": torch.randn(2, 2)}, meta)

        mock_submit.assert_not_called()

    def test_respects_max_concurrent_loads(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        c._max_concurrent_ec_loads = 2
        for i in range(2):
            c._pending_ec_loads.add(f"pending_{i}")

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("new_hash")

        with patch.object(c._ec_load_executor, 'submit') as mock_submit:
            c._async_submit_loads("cpu", {}, meta)

        mock_submit.assert_not_called()

    def test_registers_endpoint_from_metadata(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1", endpoint="tcp://10.0.0.1:5000")

        with patch.object(c._ec_load_executor, 'submit'):
            c._async_submit_loads("cpu", {}, meta)

        assert c._mm_hash_endpoints["h1"] == "tcp://10.0.0.1:5000"

    @patch('omni_npu.connector.ec_connector.network_connector.torch.distributed.is_initialized',
           return_value=True)
    def test_tp_leader_async_skips_non_rank0(self, mock_init, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        c.tp_size = 2
        c.tp_rank = 1
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": True,
            }.get(key, default)
        )

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")

        with patch.object(c._ec_load_executor, 'submit') as mock_submit:
            c._async_submit_loads("cpu", {}, meta)

        mock_submit.assert_not_called()


# =================== Tests: _async_load_single =================== #

class TestAsyncLoadSingle:
    def test_success(self, consumer_worker_connector):
        c = consumer_worker_connector
        tensor = torch.randn(2, 4)
        payload = _serialize_cache(tensor)
        resp = ECNetworkResponse(status=STATUS.OK, payload=payload)

        c._pending_ec_loads.add("h1")
        encoder_cache = {}

        with patch.object(c, '_async_client_request', return_value=resp), \
             patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            c._async_load_single(encoder_cache, "h1", device="cpu")

        assert "h1" in encoder_cache
        assert isinstance(encoder_cache["h1"], torch.Tensor)
        assert "h1" in c._finished_ec_loads
        assert "h1" not in c._pending_ec_loads

    def test_miss_response(self, consumer_worker_connector):
        c = consumer_worker_connector
        resp = ECNetworkResponse(status=STATUS.MISS)

        c._pending_ec_loads.add("h1")
        encoder_cache = {}

        with patch.object(c, '_async_client_request', return_value=resp):
            c._async_load_single(encoder_cache, "h1")

        assert "h1" not in encoder_cache
        assert "h1" in c._failed_ec_loads
        assert "h1" not in c._pending_ec_loads

    def test_exception_handling(self, consumer_worker_connector):
        c = consumer_worker_connector
        c._pending_ec_loads.add("h1")
        encoder_cache = {}

        with patch.object(c, '_async_client_request', side_effect=RuntimeError("boom")):
            c._async_load_single(encoder_cache, "h1")

        assert "h1" not in encoder_cache
        assert "h1" in c._failed_ec_loads
        assert "h1" not in c._pending_ec_loads

    @patch('omni_npu.connector.ec_connector.network_connector.torch.distributed.is_initialized',
           return_value=True)
    def test_tp_leader_async_adds_to_broadcast_queue(self, mock_init, consumer_worker_connector):
        c = consumer_worker_connector
        c.tp_size = 2
        c.tp_rank = 0
        c.ec_async_flag = True
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": True,
            }.get(key, default)
        )

        tensor = torch.randn(2, 4)
        payload = _serialize_cache(tensor)
        resp = ECNetworkResponse(status=STATUS.OK, payload=payload)

        c._pending_ec_loads.add("h1")
        encoder_cache = {}

        with patch.object(c, '_async_client_request', return_value=resp):
            c._async_load_single(encoder_cache, "h1", device="cpu")

        assert "h1" in c._tp_broadcast_queue
        assert "h1" in c._finished_ec_loads


# =================== Tests: wait_for_pending_loads =================== #

class TestWaitForPendingLoads:
    def test_returns_immediately_if_not_async(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = False
        c.wait_for_pending_loads({})

    def test_returns_immediately_if_no_pending(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        c.wait_for_pending_loads({})

    def test_waits_for_pending_to_complete(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        c._pending_ec_loads.add("h1")

        encoder_cache = {}

        def fill_cache():
            time.sleep(0.01)
            encoder_cache["h1"] = torch.randn(2, 2)

        t = threading.Thread(target=fill_cache)
        t.start()
        c.wait_for_pending_loads(encoder_cache, timeout=5.0)
        t.join()
        assert "h1" in encoder_cache

    def test_timeout_logs_error(self, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        c._pending_ec_loads.add("h1")

        with patch('omni_npu.connector.ec_connector.network_connector.logger') as mock_logger:
            c.wait_for_pending_loads({}, timeout=0.01)
            mock_logger.error.assert_called_once()

    @patch('omni_npu.connector.ec_connector.network_connector.torch.distributed.is_initialized',
           return_value=True)
    def test_delegates_to_broadcast_when_tp_leader_async(self, mock_init, consumer_worker_connector):
        c = consumer_worker_connector
        c.ec_async_flag = True
        c.tp_size = 2
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": True,
            }.get(key, default)
        )

        with patch.object(c, '_wait_and_broadcast_pending') as mock_broadcast:
            c.wait_for_pending_loads({})
            mock_broadcast.assert_called_once()


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


# =================== Tests: _sync_load_with_tp_broadcast =================== #

class TestSyncLoadWithTPBroadcast:
    def _make_tp_connector(self):
        config = make_mock_vllm_config(is_producer=False)
        config.parallel_config.tensor_parallel_size = 2
        c = create_connector(role=ECConnectorRole.WORKER, is_producer=False, vllm_config=config)
        c.tp_size = 2
        return c

    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_all_ranks_have_cache_skips(self, mock_dist, mock_get_tp):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        need_tensor = torch.tensor([0], dtype=torch.int32)
        mock_dist.all_reduce.side_effect = lambda t, **kw: t.fill_(0)

        encoder_cache = {"h1": torch.randn(2, 4)}
        c._sync_load_with_tp_broadcast("h1", encoder_cache, device="cpu")

        mock_tp.broadcast_object.assert_not_called()

    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_rank0_fetches_and_broadcasts(self, mock_dist, mock_get_tp):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        tensor = torch.randn(2, 4)
        payload = _serialize_cache(tensor)
        resp = ECNetworkResponse(status=STATUS.OK, payload=payload)
        c._client_request = Mock(return_value=resp)

        mock_dist.all_reduce.side_effect = lambda t, **kw: t.fill_(1)
        mock_tp.broadcast_object.return_value = True
        mock_tp.broadcast_tensor_dict.return_value = {"h1": tensor}

        encoder_cache = {}
        with patch('omni_npu.connector.ec_connector.network_connector.current_platform') as m:
            m.device_type = 'cpu'
            c._sync_load_with_tp_broadcast("h1", encoder_cache, device="cpu")

        assert "h1" in encoder_cache
        mock_tp.broadcast_tensor_dict.assert_called_once()

    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_rank0_cache_hit_in_encoder_cache(self, mock_dist, mock_get_tp):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        existing_tensor = torch.randn(2, 4)
        mock_dist.all_reduce.side_effect = lambda t, **kw: t.fill_(1)
        mock_tp.broadcast_object.return_value = True
        mock_tp.broadcast_tensor_dict.return_value = {"h1": existing_tensor}

        encoder_cache = {"h1": existing_tensor}
        c._sync_load_with_tp_broadcast("h1", encoder_cache, device="cpu")

        assert "h1" in encoder_cache

    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_rank0_cache_miss_from_server(self, mock_dist, mock_get_tp):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        resp = ECNetworkResponse(status=STATUS.MISS)
        c._client_request = Mock(return_value=resp)

        mock_dist.all_reduce.side_effect = lambda t, **kw: t.fill_(1)
        mock_tp.broadcast_object.return_value = False

        encoder_cache = {}
        c._sync_load_with_tp_broadcast("h1", encoder_cache, device="cpu")

        assert "h1" not in encoder_cache

    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_non_rank0_receives_broadcast(self, mock_dist, mock_get_tp):
        c = self._make_tp_connector()
        c.tp_rank = 1
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        tensor = torch.randn(2, 4)
        mock_dist.all_reduce.side_effect = lambda t, **kw: t.fill_(1)
        mock_tp.broadcast_object.return_value = True
        mock_tp.broadcast_tensor_dict.return_value = {"h1": tensor}

        encoder_cache = {}
        c._sync_load_with_tp_broadcast("h1", encoder_cache, device="cpu")

        assert "h1" in encoder_cache
        assert torch.equal(encoder_cache["h1"], tensor)


# =================== Tests: _wait_and_broadcast_pending =================== #

class TestWaitAndBroadcastPending:
    def _make_tp_connector(self):
        config = make_mock_vllm_config(is_producer=False)
        config.parallel_config.tensor_parallel_size = 2
        c = create_connector(role=ECConnectorRole.WORKER, is_producer=False, vllm_config=config)
        c.tp_size = 2
        c.ec_async_flag = True
        return c

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    def test_rank0_broadcasts_tensors(self, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        tensor = torch.randn(2, 4)
        c._tp_broadcast_queue.add("h1")
        encoder_cache = {"h1": tensor}

        mock_tp.broadcast_object.return_value = ["h1"]
        mock_tp.broadcast_tensor_dict.return_value = {"h1": tensor}

        c._wait_and_broadcast_pending(encoder_cache, timeout=1.0)

        mock_tp.broadcast_tensor_dict.assert_called_once()
        assert "h1" in c._finished_ec_loads
        assert "h1" not in c._tp_broadcast_queue

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    def test_empty_hash_list_clears_pending(self, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        c._pending_ec_loads.add("h1")
        mock_tp.broadcast_object.return_value = []

        c._wait_and_broadcast_pending({}, timeout=1.0)

        assert len(c._pending_ec_loads) == 0
        mock_tp.broadcast_tensor_dict.assert_not_called()

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    def test_non_rank0_receives_broadcast(self, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 1
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        tensor = torch.randn(2, 4)
        mock_tp.broadcast_object.return_value = ["h1"]
        mock_tp.broadcast_tensor_dict.return_value = {"h1": tensor}

        encoder_cache = {}
        c._wait_and_broadcast_pending(encoder_cache, timeout=1.0)

        assert "h1" in encoder_cache
        assert "h1" in c._finished_ec_loads

    @patch('omni_npu.connector.ec_connector.network_connector.current_platform')
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    def test_rank0_waits_for_pending_to_clear(self, mock_get_tp, mock_platform):
        c = self._make_tp_connector()
        c.tp_rank = 0
        mock_platform.device_type = 'cpu'
        mock_tp = MagicMock()
        mock_get_tp.return_value = mock_tp

        c._pending_ec_loads.add("h1")
        c._tp_broadcast_queue.add("h1")

        def clear_pending():
            time.sleep(0.01)
            with c._ec_load_lock:
                c._pending_ec_loads.clear()

        t = threading.Thread(target=clear_pending)
        t.start()

        tensor = torch.randn(2, 4)
        encoder_cache = {"h1": tensor}
        mock_tp.broadcast_object.return_value = ["h1"]
        mock_tp.broadcast_tensor_dict.return_value = {"h1": tensor}

        c._wait_and_broadcast_pending(encoder_cache, timeout=5.0)
        t.join()

        assert "h1" in c._finished_ec_loads


# =================== Tests: _sync_load_caches with tp_broadcast =================== #

class TestSyncLoadCachesWithTPBroadcast:
    @patch('omni_npu.connector.ec_connector.network_connector.torch.distributed.is_initialized',
           return_value=True)
    @patch('omni_npu.connector.ec_connector.network_connector.get_tp_group')
    @patch('omni_npu.connector.ec_connector.network_connector.dist')
    def test_sync_load_delegates_to_tp_broadcast(self, mock_dist, mock_get_tp, mock_init):
        config = make_mock_vllm_config(is_producer=False)
        config.parallel_config.tensor_parallel_size = 2
        c = create_connector(role=ECConnectorRole.WORKER, is_producer=False, vllm_config=config)
        c.tp_size = 2
        c.ec_async_flag = False
        c._vllm_config.ec_transfer_config.get_from_extra_config = Mock(
            side_effect=lambda key, default=None: {
                "ec_zmq_tp_leader_only": True,
            }.get(key, default)
        )

        meta = ECNetworkConnectorMetadata()
        meta.add_mm_hash("h1")
        c.bind_connector_metadata(meta)

        with patch.object(c, '_sync_load_with_tp_broadcast') as mock_tp_load:
            c.start_load_caches({})
            mock_tp_load.assert_called_once()
