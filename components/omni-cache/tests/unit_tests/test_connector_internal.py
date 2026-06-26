# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_connector_internal.py

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from dataclasses import is_dataclass

# Import module under test
from omni_cache.connector.utils.settings import (
    BASE_DIR,
    OX_PATH,
    OX_LOG_PATH,
    BLOCK_RELEASE_DELAY,
    PER_REQUEST_CONNECTION,
    P_NODE_LIST,
    CLUSTER_LIST,
    CLUSTER_SIZE,
    NODE_IP_SPECS,
    BASE_PORT,
    ZMQ_BASE_PORT,
    P_NODE_PORT_LIST,
)
from omni_cache.connector.utils.metadata import (
    ReqMeta,
    ReqMetaPrefill,
    _build_req_meta,
    DatadistConnectorMetadata,
    DatadistConnectorMetadataPrefill,
    DTypeUtils,
    _SendItem,
    PendingReq,
)


# ==================== Settings module tests ====================

class TestSettingsConstants:
    """Test suite for settings module constants."""

    def test_base_dir_exists(self):
        """Test that BASE_DIR is set."""
        assert BASE_DIR is not None
        assert isinstance(BASE_DIR, str)

    def test_ox_path_format(self):
        """Test that OX_PATH has expected format."""
        assert isinstance(OX_PATH, str)
        assert "ox" in OX_PATH.lower()

    def test_ox_log_path_format(self):
        """Test that OX_LOG_PATH has expected format."""
        assert isinstance(OX_LOG_PATH, str)
        assert "ox" in OX_LOG_PATH.lower() or "data" in OX_LOG_PATH.lower()

    def test_block_release_delay_type(self):
        """Test that BLOCK_RELEASE_DELAY is an integer."""
        assert isinstance(BLOCK_RELEASE_DELAY, int)
        assert BLOCK_RELEASE_DELAY > 0

    def test_per_request_connection_type(self):
        """Test that PER_REQUEST_CONNECTION is an integer."""
        assert isinstance(PER_REQUEST_CONNECTION, int)
        assert PER_REQUEST_CONNECTION > 0

    def test_p_node_list_format(self):
        """Test that P_NODE_LIST has expected format."""
        assert isinstance(P_NODE_LIST, str)

    def test_cluster_list_is_list(self):
        """Test that CLUSTER_LIST is a list."""
        assert isinstance(CLUSTER_LIST, list)

    def test_cluster_size_is_int(self):
        """Test that CLUSTER_SIZE is an integer."""
        assert isinstance(CLUSTER_SIZE, int)
        assert CLUSTER_SIZE > 0

    def test_node_ip_specs_is_list(self):
        """Test that NODE_IP_SPECS is a list."""
        assert isinstance(NODE_IP_SPECS, list)
        assert len(NODE_IP_SPECS) > 0

    def test_base_port_is_int(self):
        """Test that BASE_PORT is an integer."""
        assert isinstance(BASE_PORT, int)
        assert BASE_PORT > 0

    def test_zmq_base_port_is_int(self):
        """Test that ZMQ_BASE_PORT is an integer."""
        assert isinstance(ZMQ_BASE_PORT, int)
        assert ZMQ_BASE_PORT > 0

    def test_p_node_port_list_format(self):
        """Test that P_NODE_PORT_LIST has expected format."""
        assert isinstance(P_NODE_PORT_LIST, str)
        assert str(BASE_PORT) in P_NODE_PORT_LIST


# ==================== Settings environment variable tests ====================

class TestSettingsEnvironmentVariables:
    """Test suite for settings with environment variables."""

    @patch.dict(os.environ, {"OX_PATH": "/custom/ox/path"})
    def test_ox_path_from_env(self):
        """Test that OX_PATH can be overridden via environment."""
        # Need to reimport to get the updated value
        import importlib
        from omni_cache.connector.utils import settings
        importlib.reload(settings)

        assert settings.OX_PATH == "/custom/ox/path"

    @patch.dict(os.environ, {"BASE_PORT": "9999"})
    def test_base_port_from_env(self):
        """Test that BASE_PORT can be overridden via environment."""
        import importlib
        from omni_cache.connector.utils import settings
        importlib.reload(settings)

        assert settings.BASE_PORT == 9999

    @patch.dict(os.environ, {"ZMQ_BASE_PORT": "19999"})
    def test_zmq_base_port_from_env(self):
        """Test that ZMQ_BASE_PORT can be overridden via environment."""
        import importlib
        from omni_cache.connector.utils import settings
        importlib.reload(settings)

        assert settings.ZMQ_BASE_PORT == 19999

    @patch.dict(os.environ, {"P_NODE_LIST": "1.2.3.4,5.6.7.8"})
    def test_p_node_list_from_env(self):
        """Test that P_NODE_LIST can be overridden via environment."""
        import importlib
        from omni_cache.connector.utils import settings
        importlib.reload(settings)

        assert settings.P_NODE_LIST == "1.2.3.4,5.6.7.8"
        assert settings.CLUSTER_LIST == ["1.2.3.4,5.6.7.8"]
        assert settings.CLUSTER_SIZE == 2

    @patch.dict(os.environ, {"P_NODE_LIST": "1.2.3.4,5.6.7.8;9.10.11.12"})
    def test_p_node_list_multiple_clusters(self):
        """Test P_NODE_LIST with multiple clusters."""
        import importlib
        from omni_cache.connector.utils import settings
        importlib.reload(settings)

        assert len(settings.CLUSTER_LIST) == 2
        assert settings.CLUSTER_SIZE == 2  # Size of first cluster


# ==================== ReqMeta tests ====================

class TestReqMeta:
    """Test suite for ReqMeta dataclass."""

    def test_reqmeta_is_dataclass(self):
        """Test that ReqMeta is a dataclass."""
        assert is_dataclass(ReqMeta)

    def test_reqmeta_creation(self):
        """Test creating ReqMeta instance."""
        meta = ReqMeta(
            local_block_ids=[[1, 2, 3], [4, 5, 6]],
            remote_block_ids=[7, 8, 9],
            remote_host="192.168.1.1",
            remote_cluster_id="cluster_1",
            spec_token_ids=[0, 1, 2],
            remote_dp_rank=0,
            remote_request_id="req-123",
        )

        assert meta.local_block_ids == [[1, 2, 3], [4, 5, 6]]
        assert meta.remote_block_ids == [7, 8, 9]
        assert meta.remote_host == "192.168.1.1"
        assert meta.remote_cluster_id == "cluster_1"
        assert meta.spec_token_ids == [0, 1, 2]
        assert meta.remote_dp_rank == 0
        assert meta.remote_request_id == "req-123"

    def test_reqmeta_optional_fields(self):
        """Test ReqMeta with optional fields as None."""
        meta = ReqMeta(
            local_block_ids=[[1, 2]],
            remote_block_ids=[3, 4],
            remote_host="192.168.1.1",
            remote_cluster_id="cluster_1",
            spec_token_ids=None,
            remote_dp_rank=None,
            remote_request_id=None,
        )

        assert meta.spec_token_ids is None
        assert meta.remote_dp_rank is None
        assert meta.remote_request_id is None


# ==================== ReqMetaPrefill tests ====================

class TestReqMetaPrefill:
    """Test suite for ReqMetaPrefill dataclass."""

    def test_reqmetaprefill_is_dataclass(self):
        """Test that ReqMetaPrefill is a dataclass."""
        assert is_dataclass(ReqMetaPrefill)

    def test_reqmetaprefill_creation(self):
        """Test creating ReqMetaPrefill instance."""
        meta = ReqMetaPrefill(finish_time=12345.678)

        assert meta.finish_time == 12345.678


# ==================== _build_req_meta tests ====================

class TestBuildReqMeta:
    """Test suite for _build_req_meta function."""

    def test_build_req_meta_normal(self):
        """Test building ReqMeta with normal parameters."""
        local_block_ids = [[1, 2, 3], [4, 5, 6]]
        kv_transfer_params = {
            "remote_block_ids": [7, 8, 9],
            "remote_host_ip": "192.168.1.1",
            "remote_cluster_id": "cluster_1",
            "spec_token_ids": [0, 1],
            "remote_dp_rank": 1,
            "remote_request_id": "req-456",
        }

        meta = _build_req_meta(local_block_ids, kv_transfer_params)

        assert isinstance(meta, ReqMeta)
        assert meta.local_block_ids == local_block_ids
        assert meta.remote_block_ids == [7, 8, 9]
        assert meta.remote_host == "192.168.1.1"
        assert meta.remote_cluster_id == "cluster_1"
        assert meta.spec_token_ids == [0, 1]
        assert meta.remote_dp_rank == 1
        assert meta.remote_request_id == "req-456"

    def test_build_req_meta_default_dp_rank(self):
        """Test that remote_dp_rank defaults to 0."""
        local_block_ids = [[1, 2]]
        kv_transfer_params = {
            "remote_block_ids": [3, 4],
            "remote_host_ip": "192.168.1.1",
            "remote_cluster_id": "cluster_1",
            "spec_token_ids": None,
        }

        meta = _build_req_meta(local_block_ids, kv_transfer_params)

        assert meta.remote_dp_rank == 0

    def test_build_req_meta_default_request_id(self):
        """Test that remote_request_id defaults to None."""
        local_block_ids = [[1, 2]]
        kv_transfer_params = {
            "remote_block_ids": [3, 4],
            "remote_host_ip": "192.168.1.1",
            "remote_cluster_id": "cluster_1",
            "spec_token_ids": None,
        }

        meta = _build_req_meta(local_block_ids, kv_transfer_params)

        assert meta.remote_request_id is None

    @patch.dict(os.environ, {"ENABLE_MOCK_P": "1"})
    def test_build_req_meta_mock_mode(self):
        """Test building ReqMeta in mock mode."""
        local_block_ids = [[1, 2, 3]]
        kv_transfer_params = {}

        meta = _build_req_meta(local_block_ids, kv_transfer_params)

        assert isinstance(meta, ReqMeta)
        assert meta.remote_host == "0.0.0.0"
        assert meta.remote_cluster_id == "0"
        assert meta.remote_block_ids == [1, 2, 3, 4, 5]


# ==================== DatadistConnectorMetadata tests ====================

class TestDatadistConnectorMetadata:
    """Test suite for DatadistConnectorMetadata class."""

    def test_initialization(self):
        """Test initialization of DatadistConnectorMetadata."""
        metadata = DatadistConnectorMetadata()

        assert isinstance(metadata.requests, dict)
        assert len(metadata.requests) == 0

    def test_add_new_req(self):
        """Test adding a new request."""
        metadata = DatadistConnectorMetadata()

        metadata.add_new_req(
            request_id="req-123",
            local_block_ids=[[1, 2, 3]],
            kv_transfer_params={
                "remote_block_ids": [4, 5, 6],
                "remote_host_ip": "192.168.1.1",
                "remote_cluster_id": "cluster_1",
                "spec_token_ids": None,
            },
        )

        assert "req-123" in metadata.requests
        assert isinstance(metadata.requests["req-123"], ReqMeta)
        assert metadata.requests["req-123"].remote_host == "192.168.1.1"

    def test_add_multiple_reqs(self):
        """Test adding multiple requests."""
        metadata = DatadistConnectorMetadata()

        metadata.add_new_req(
            request_id="req-1",
            local_block_ids=[[1]],
            kv_transfer_params={
                "remote_block_ids": [2],
                "remote_host_ip": "192.168.1.1",
                "remote_cluster_id": "cluster_1",
                "spec_token_ids": None,
            },
        )
        metadata.add_new_req(
            request_id="req-2",
            local_block_ids=[[3]],
            kv_transfer_params={
                "remote_block_ids": [4],
                "remote_host_ip": "192.168.1.2",
                "remote_cluster_id": "cluster_2",
                "spec_token_ids": None,
            },
        )

        assert len(metadata.requests) == 2
        assert metadata.requests["req-1"].remote_host == "192.168.1.1"
        assert metadata.requests["req-2"].remote_host == "192.168.1.2"


# ==================== DatadistConnectorMetadataPrefill tests ====================

class TestDatadistConnectorMetadataPrefill:
    """Test suite for DatadistConnectorMetadataPrefill class."""

    def test_initialization(self):
        """Test initialization of DatadistConnectorMetadataPrefill."""
        metadata = DatadistConnectorMetadataPrefill()

        assert isinstance(metadata.requests, dict)
        assert len(metadata.requests) == 0

    def test_add_new_req(self):
        """Test adding a new request."""
        metadata = DatadistConnectorMetadataPrefill()

        metadata.add_new_req(request_id="req-123", finish_time=12345.678)

        assert "req-123" in metadata.requests
        assert isinstance(metadata.requests["req-123"], ReqMetaPrefill)
        assert metadata.requests["req-123"].finish_time == 12345.678

    def test_add_multiple_reqs(self):
        """Test adding multiple requests."""
        metadata = DatadistConnectorMetadataPrefill()

        metadata.add_new_req(request_id="req-1", finish_time=100.0)
        metadata.add_new_req(request_id="req-2", finish_time=200.0)

        assert len(metadata.requests) == 2
        assert metadata.requests["req-1"].finish_time == 100.0
        assert metadata.requests["req-2"].finish_time == 200.0


# ==================== DTypeUtils tests ====================

class TestDTypeUtils:
    """Test suite for DTypeUtils class."""

    def test_size_int8(self):
        """Test size of int8."""
        assert DTypeUtils.size("int8") == 1
        assert DTypeUtils.size("INT8") == 1
        assert DTypeUtils.size("Int8") == 1

    def test_size_uint8(self):
        """Test size of uint8."""
        assert DTypeUtils.size("uint8") == 1
        assert DTypeUtils.size("byte") == 1

    def test_size_16bit(self):
        """Test size of 16-bit types."""
        assert DTypeUtils.size("int16") == 2
        assert DTypeUtils.size("uint16") == 2
        assert DTypeUtils.size("fp16") == 2
        assert DTypeUtils.size("bf16") == 2

    def test_size_32bit(self):
        """Test size of 32-bit types."""
        assert DTypeUtils.size("int32") == 4
        assert DTypeUtils.size("uint32") == 4
        assert DTypeUtils.size("fp32") == 4
        assert DTypeUtils.size("float32") == 4

    def test_size_64bit(self):
        """Test size of 64-bit types."""
        assert DTypeUtils.size("int64") == 8
        assert DTypeUtils.size("uint64") == 8
        assert DTypeUtils.size("fp64") == 8
        assert DTypeUtils.size("float64") == 8

    def test_size_unsupported_raises_error(self):
        """Test that unsupported dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            DTypeUtils.size("unsupported")

    def test_size_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            DTypeUtils.size("")

    def test_supported_returns_list(self):
        """Test that supported returns a list."""
        supported = DTypeUtils.supported()

        assert isinstance(supported, list)
        assert len(supported) > 0
        assert "int8" in supported
        assert "fp16" in supported
        assert "float32" in supported

    def test_supported_contains_all_types(self):
        """Test that supported contains all expected types."""
        supported = DTypeUtils.supported()

        expected = [
            "int8", "uint8", "byte",
            "int16", "uint16", "fp16", "bf16",
            "int32", "uint32", "fp32", "float32",
            "int64", "uint64", "fp64", "float64",
        ]

        for dtype in expected:
            assert dtype in supported


# ==================== _SendItem tests ====================

class TestSendItem:
    """Test suite for _SendItem dataclass."""

    def test_senditem_is_dataclass(self):
        """Test that _SendItem is a dataclass."""
        assert is_dataclass(_SendItem)

    def test_senditem_creation(self):
        """Test creating _SendItem instance."""
        item = _SendItem(
            request_id="req-123",
            cluster_id=1,
            src_ids=[1, 2, 3],
            dst_ids=[4, 5, 6],
            rank_id=0,
        )

        assert item.request_id == "req-123"
        assert item.cluster_id == 1
        assert item.src_ids == [1, 2, 3]
        assert item.dst_ids == [4, 5, 6]
        assert item.rank_id == 0


# ==================== PendingReq tests ====================

class TestPendingReq:
    """Test suite for PendingReq dataclass."""

    def test_pendingreq_is_dataclass(self):
        """Test that PendingReq is a dataclass."""
        assert is_dataclass(PendingReq)

    def test_pendingreq_creation(self):
        """Test creating PendingReq instance."""
        req = PendingReq(
            request_id="req-123",
            local_block_ids=[[1, 2, 3]],
            remote_block_ids=[4, 5, 6],
            dst_cluster_id="cluster_1",
            remote_request_id="remote-req-456",
            remote_host_ip="192.168.1.1",
            dp_rank=0,
        )

        assert req.request_id == "req-123"
        assert req.local_block_ids == [[1, 2, 3]]
        assert req.remote_block_ids == [4, 5, 6]
        assert req.dst_cluster_id == "cluster_1"
        assert req.remote_request_id == "remote-req-456"
        assert req.remote_host_ip == "192.168.1.1"
        assert req.dp_rank == 0
        assert req.t_sent == 0.0
        assert req.t_resp == 0.0

    def test_pendingreq_default_times(self):
        """Test that PendingReq has correct default time values."""
        req = PendingReq(
            request_id="req-123",
            local_block_ids=[[1]],
            remote_block_ids=[2],
            dst_cluster_id="cluster_1",
            remote_request_id="remote-req",
            remote_host_ip="192.168.1.1",
            dp_rank=0,
        )

        assert req.t_sent == 0.0
        assert req.t_resp == 0.0
        # t_submit should be set to current time by default
        assert req.t_submit > 0


# ==================== Integration tests ====================

class TestConnectorInternalIntegration:
    """Integration tests for connector.utils module."""

    def test_end_to_end_metadata_workflow(self):
        """Test end-to-end workflow with metadata classes."""
        # Create decode metadata
        decode_metadata = DatadistConnectorMetadata()
        decode_metadata.add_new_req(
            request_id="decode-req",
            local_block_ids=[[1, 2, 3]],
            kv_transfer_params={
                "remote_block_ids": [4, 5, 6],
                "remote_host_ip": "192.168.1.1",
                "remote_cluster_id": "cluster_1",
                "spec_token_ids": [0, 1],
            },
        )

        # Create prefill metadata
        prefill_metadata = DatadistConnectorMetadataPrefill()
        prefill_metadata.add_new_req(request_id="prefill-req", finish_time=12345.0)

        # Verify both are working
        assert "decode-req" in decode_metadata.requests
        assert "prefill-req" in prefill_metadata.requests

    def test_dtype_utils_with_metadata(self):
        """Test DTypeUtils integration with metadata workflow."""
        # Use DTypeUtils to calculate sizes
        fp16_size = DTypeUtils.size("fp16")
        fp32_size = DTypeUtils.size("fp32")

        # Create a request with block information
        metadata = DatadistConnectorMetadata()
        metadata.add_new_req(
            request_id="req-with-dtype",
            local_block_ids=[[1, 2]],
            kv_transfer_params={
                "remote_block_ids": [3, 4],
                "remote_host_ip": "192.168.1.1",
                "remote_cluster_id": "cluster_1",
                "spec_token_ids": None,
            },
        )

        # Verify the request is stored correctly
        assert metadata.requests["req-with-dtype"].remote_host == "192.168.1.1"

        # Verify dtype sizes
        assert fp16_size == 2
        assert fp32_size == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
