# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ECSharedMemoryConnector."""
import hashlib
import json
from contextlib import contextmanager
import pytest
import torch
from collections import OrderedDict
from unittest.mock import Mock, patch, MagicMock

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.output import SchedulerOutput

from omni.connector.ec_connector.shared_memory_connector import (
    BYTE_LENGTH,
    ECSharedMemoryConnector,
    ECSharedMemoryConnectorMetadata,
    META_DEVICE_IDX,
    META_DTYPE_IDX,
    META_WIDTH,
    _CacheLoadResult,
)


# ------------------ Mock Classes ------------------ #
class MockRequest:
    """Mock Request class for testing."""

    def __init__(self, request_id: str, mm_hashes: list[str]):
        self.request_id = request_id
        self.mm_features = []
        for mm_hash in mm_hashes:
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=0, length=100),
            )
            self.mm_features.append(feature)


def build_raw_tensor_shm_buffer(tensor: torch.Tensor) -> bytearray:
    """Build the raw shared-memory payload used by ECSharedMemoryConnector."""
    cpu_tensor, meta_payload, total_size = ECSharedMemoryConnector._serialize_cache(tensor)
    mock_buf = bytearray(total_size)
    data_start = BYTE_LENGTH + len(meta_payload)
    mock_buf[:BYTE_LENGTH] = len(meta_payload).to_bytes(BYTE_LENGTH, "little")
    mock_buf[BYTE_LENGTH:data_start] = meta_payload
    if cpu_tensor.numel() > 0:
        shm_tensor = torch.frombuffer(memoryview(mock_buf)[data_start:], dtype=cpu_tensor.dtype,
                                      count=cpu_tensor.numel()).reshape(cpu_tensor.shape)
        shm_tensor.copy_(cpu_tensor)
        del shm_tensor
    return mock_buf


# ------------------ Mock Fixtures ------------------ #
@pytest.fixture
def mock_vllm_config():
    """Fixture providing mock VllmConfig for producer role."""
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.get_from_extra_config = Mock(return_value=True)
    config.ec_transfer_config.is_ec_producer = True  # Producer
    return config


@pytest.fixture
def mock_vllm_config_consumer():
    """Fixture providing mock VllmConfig for consumer role."""
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.get_from_extra_config = Mock(return_value=True)
    config.ec_transfer_config.is_ec_producer = False  # Consumer
    return config


@pytest.fixture
def mock_request_with_3_mm():
    """Fixture providing mock Request with 3 multimodal items."""
    return MockRequest("test_req_123", ["hash1", "hash2", "hash3"])


@pytest.fixture(autouse=True)
def mock_single_tp_world():
    """Keep shared-memory connector tests independent of torch.distributed."""
    with patch(
        'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
        return_value=0
    ), patch(
        'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_world_size',
        return_value=1
    ):
        yield


@pytest.fixture
def tp_broadcast_patches():
    """Patch the common TP broadcast environment for a connector test."""

    @contextmanager
    def apply_patches(connector, tp_rank, device_group, broadcast_side_effect):
        with patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
            return_value=tp_rank
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_world_size',
            return_value=2
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.dist.is_available',
            return_value=True
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.dist.is_initialized',
            return_value=True
        ), patch.object(
            connector, '_tp_src_rank', return_value=0
        ), patch.object(
            connector, '_tp_device_group', return_value=device_group
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.dist.broadcast_object_list',
        ) as mock_broadcast_meta, patch(
            'omni.connector.ec_connector.shared_memory_connector.dist.broadcast',
            side_effect=broadcast_side_effect
        ) as mock_broadcast, patch(
            'omni.connector.ec_connector.shared_memory_connector.current_platform'
        ) as mock_platform:
            mock_platform.device_type = 'cpu'
            yield mock_broadcast_meta, mock_broadcast

    return apply_patches

# ------------------ Unit Tests ------------------ #
class TestECSharedMemoryConnectorMetadata:
    """Test ECSharedMemoryConnectorMetadata functionality."""

    def test_init(self):
        """Test metadata initialization."""
        metadata = ECSharedMemoryConnectorMetadata()
        assert metadata.mm_hashes == []
        assert len(metadata.mm_hashes) == 0

    def test_add_mm_hash(self):
        """Test adding mm_hash to metadata."""
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash("hash1")
        assert len(metadata.mm_hashes) == 1
        assert "hash1" in metadata.mm_hashes

        metadata.add_mm_hash("hash2")
        assert len(metadata.mm_hashes) == 2
        assert "hash2" in metadata.mm_hashes

    def test_add_multiple_mm_hashes(self):
        """Test adding multiple mm_hashes."""
        metadata = ECSharedMemoryConnectorMetadata()
        hashes = ["hash1", "hash2", "hash3"]
        for h in hashes:
            metadata.add_mm_hash(h)
        assert len(metadata.mm_hashes) == 3
        assert metadata.mm_hashes == hashes


class TestECSharedMemoryConnector:
    """Test connector initialization."""

    def test_init_with_ec_transfer_config(self, mock_vllm_config):
        """Test connector initializes correctly with ec_transfer_config."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        assert connector.role == ECConnectorRole.WORKER
        assert connector.count == 0
        assert hasattr(connector, '_max_bytes')
        assert isinstance(connector._lru, OrderedDict)
        assert len(connector._mm_hashes_need_loads) == 0

    def test_init_role_scheduler(self, mock_vllm_config):
        """Test connector initializes as SCHEDULER."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.SCHEDULER
        )
        assert connector.role == ECConnectorRole.SCHEDULER

    def test_init_without_ec_transfer_config(self):
        """Test connector raises error without ec_transfer_config."""
        config = Mock(spec=VllmConfig)
        config.ec_transfer_config = None
        with pytest.raises(ValueError, match="ec_transfer_config must be set"):
            ECSharedMemoryConnector(
                vllm_config=config,
                role=ECConnectorRole.WORKER
            )

    def test_init_memory_calculation(self, mock_vllm_config):
        """Test connector calculates max memory correctly."""
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.available = 10 * 1024 ** 3  # 10 GB
            connector = ECSharedMemoryConnector(
                vllm_config=mock_vllm_config,
                role=ECConnectorRole.WORKER
            )
            # Should use 10% of available, capped at 2GB
            assert connector._max_bytes == int(10 * 1024 ** 3 * 0.1)

    def test_init_uses_eight_shm_load_workers(self, mock_vllm_config):
        """Test shared-memory loads use the benchmarked worker count."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        assert connector._shm_load_executor._max_workers == 8


class TestECSharedMemoryConnectorSerializeDeserialize:
    """Test serialization and deserialization methods."""

    def test_serialize_cache(self, mock_vllm_config):
        """Test _serialize_cache creates raw tensor metadata."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        tensor = torch.randn(10, 768)
        cpu_tensor, meta_payload, total_size = connector._serialize_cache(tensor)
        assert isinstance(cpu_tensor, torch.Tensor)
        assert cpu_tensor.device.type == "cpu"
        assert cpu_tensor.is_contiguous()
        assert isinstance(meta_payload, bytes)
        assert json.loads(meta_payload.decode("utf-8")) == {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        assert total_size == (
            BYTE_LENGTH + len(meta_payload) + tensor.numel() * tensor.element_size())

    def test_deserialize_cache(self, mock_vllm_config):
        """Test _deserialize_cache creates tensor from shared-memory payload."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        tensor = torch.randn(10, 768)
        payload = build_raw_tensor_shm_buffer(tensor)

        with patch(
            'omni.connector.ec_connector.shared_memory_connector.current_platform'
        ) as mock_platform:
            mock_platform.device_type = 'cpu'
            result = connector._deserialize_cache(memoryview(payload))
            assert isinstance(result, torch.Tensor)

    @pytest.mark.parametrize(
        "tensor_meta, error_match",
        [
            ({"shape": [2, 2]}, "Missing.*dtype"),
            ({"shape": "2x2", "dtype": "float32"}, "tensor shape"),
            ({"shape": [2, 2], "dtype": "not_a_dtype"}, "tensor dtype"),
        ],
    )
    def test_deserialize_cache_rejects_invalid_metadata(
        self, mock_vllm_config, tensor_meta, error_match,
    ):
        """Test malformed shared-memory metadata is normalized to ValueError."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        meta_payload = json.dumps(tensor_meta).encode("utf-8")
        payload = (
            len(meta_payload).to_bytes(BYTE_LENGTH, "little") + meta_payload
        )

        with pytest.raises(ValueError, match=error_match):
            connector._deserialize_cache(payload)

    def test_serialize_deserialize_roundtrip(self, mock_vllm_config):
        """Test serialization and deserialization are reversible."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        original = torch.randn(10, 768)

        with patch(
            'omni.connector.ec_connector.shared_memory_connector.current_platform'
        ) as mock_platform:
            mock_platform.device_type = 'cpu'
            payload = build_raw_tensor_shm_buffer(original)
            restored = connector._deserialize_cache(memoryview(payload))
            assert torch.equal(original.cpu(), restored.cpu())

    def test_serialize_deserialize_real(self, mock_vllm_config):
        """Test real serialization and deserialization (end-to-end)."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        # Test with different tensor shapes and values
        test_cases = [
            torch.randn(10, 768),
            torch.randn(25, 1024),
            torch.randn(50, 512),
            torch.zeros(5, 256),
            torch.ones(8, 128),
            torch.randn(15, 768) + 0.5,  # non-zero mean
        ]

        for i, original in enumerate(test_cases):
            payload = build_raw_tensor_shm_buffer(original)
            with patch(
                'omni.connector.ec_connector.shared_memory_connector.current_platform'
            ) as mock_platform:
                mock_platform.device_type = 'cpu'
                restored = connector._deserialize_cache(memoryview(payload))

            # Verify data consistency in real serialization/deserialization
            assert torch.equal(original.cpu(), restored.cpu()), \
                f"Test case {i} failed: tensor values don't match"

            # Verify shape preserved
            assert original.shape == restored.shape, \
                f"Test case {i} failed: shape mismatch"

            # Verify dtype preserved
            assert original.dtype == restored.dtype, \
                f"Test case {i} failed: dtype mismatch"


class TestECSharedMemoryConnectorLRU:
    """Test LRU cache functionality."""

    def test_touch_lru_new_entry(self, mock_vllm_config):
        """Test _touch_lru adds new entry."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._touch_lru("hash1")
        assert "hash1" in connector._lru
        assert len(connector._lru) == 1

    def test_touch_lru_existing_entry(self, mock_vllm_config):
        """Test _touch_lru moves existing entry to end."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._touch_lru("hash1")
        connector._touch_lru("hash2")
        connector._touch_lru("hash3")
        # hash1 should be first (oldest)
        assert list(connector._lru.keys())[0] == "hash1"

        # Touch hash1 again
        connector._touch_lru("hash1")
        # hash1 should now be last (newest)
        assert list(connector._lru.keys())[-1] == "hash1"

    def test_current_bytes(self, mock_vllm_config):
        """Test _current_bytes calculates correctly."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        assert connector._current_bytes() == 0

        connector._mm_hash_sizes["hash1"] = 1000
        connector._mm_hash_sizes["hash2"] = 2000
        assert connector._current_bytes() == 3000


class TestECSharedMemoryConnectorSaveCaches:
    """Test cache saving functionality."""

    @patch('omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
           return_value=0)
    def test_save_caches_saves_to_shared_memory(self, mock_tp_rank, mock_vllm_config):
        """Test save_caches saves tensor to shared memory."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._mm_hash_sizes = {}

        mm_hash = "test_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm_instance = MagicMock()
            _, _, total_size = connector._serialize_cache(encoder_cache[mm_hash])
            mock_shm_instance.size = total_size
            mock_shm_instance.buf = bytearray(mock_shm_instance.size)
            mock_shm.return_value = mock_shm_instance

            connector.save_caches(encoder_cache, mm_hash)

            # Verify shared memory was created
            mock_shm.assert_called()
            assert mm_hash in connector._mm_hash_sizes

    def test_save_caches_consumer_skips(self, mock_vllm_config_consumer):
        """Test cache saving is skipped for consumer."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "test_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}

        # Should not raise and not touch shared memory
        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            connector.save_caches(encoder_cache, mm_hash)
            # Should not be called for consumer
            mock_shm.assert_not_called()

    @patch('omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
           return_value=1)
    def test_save_caches_skips_non_rank0(self, mock_tp_rank, mock_vllm_config):
        """Test save_caches is skipped when tensor model parallel rank != 0."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._mm_hash_sizes = {}

        mm_hash = "test_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            connector.save_caches(encoder_cache, mm_hash)
            mock_shm.assert_not_called()
            assert mm_hash not in connector._mm_hash_sizes

    @patch('omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
           return_value=0)
    def test_save_caches_updates_lru(self, mock_tp_rank, mock_vllm_config):
        """Test save_caches updates LRU cache."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._mm_hash_sizes = {}

        mm_hash = "test_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm_instance = MagicMock()
            _, _, total_size = connector._serialize_cache(encoder_cache[mm_hash])
            mock_shm_instance.size = total_size
            mock_shm_instance.buf = bytearray(mock_shm_instance.size)
            mock_shm.return_value = mock_shm_instance
            connector.save_caches(encoder_cache, mm_hash)
            # LRU should contain the hash
            assert mm_hash in connector._lru
            # Refcount should be 1
            assert connector._mm_hash_refcounts[mm_hash] == 1


class TestECSharedMemoryConnectorLoadCaches:
    """Test cache loading functionality."""

    def test_start_load_caches_loads_from_shared_memory(self, mock_vllm_config):
        """Test start_load_caches loads tensor from shared memory."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "test_hash"
        tensor = torch.randn(10, 768)

        # Setup metadata
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        # Prepare mock shared memory data
        mock_buf = build_raw_tensor_shm_buffer(tensor)

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm_class:
            mock_shm = MagicMock()
            mock_shm.buf = mock_buf
            mock_shm.size = len(mock_buf)
            mock_shm_class.return_value = mock_shm

            # Mock connector current_platform to use cpu
            with patch(
                'omni.connector.ec_connector.shared_memory_connector.current_platform'
            ) as mock_platform:
                mock_platform.device_type = 'cpu'

                encoder_cache = {}
                connector.start_load_caches(encoder_cache)

                assert mm_hash in encoder_cache
                loaded_tensor = encoder_cache.get(mm_hash)
                assert isinstance(loaded_tensor, torch.Tensor)
                assert torch.equal(tensor.cpu(), loaded_tensor.cpu())
                assert tensor.shape == loaded_tensor.shape
                assert tensor.dtype == loaded_tensor.dtype

    def test_start_load_caches_skip_existing(self, mock_vllm_config_consumer):
        """Test cache loading skips already cached items."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "test_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        # Prepopulate cache
        existing_cache = {mm_hash: torch.randn(10, 768)}

        with patch('multiprocessing.shared_memory.SharedMemory'):
            with patch.object(connector, '_deserialize_cache') as mock_deserialize:
                connector.start_load_caches(existing_cache)

                # Should skip - deserialize not called for existing cache
                mock_deserialize.assert_not_called()

                # Verify original cache unchanged
                assert mm_hash in existing_cache

    def test_start_load_caches_submits_all_loads_before_waiting(self, mock_vllm_config_consumer):
        """Test multiple cache loads are dispatched before synchronizing results."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER
        )

        mm_hashes = ["hash1", "hash2", "hash3"]
        metadata = ECSharedMemoryConnectorMetadata()
        for mm_hash in mm_hashes:
            metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        tensors = {mm_hash: torch.full((2, 2), i, dtype=torch.float32)
                   for i, mm_hash in enumerate(mm_hashes)}
        mock_shms = {mm_hash: MagicMock() for mm_hash in mm_hashes}
        completed_loads = set()
        fake_executor = MagicMock()
        futures = {}

        for mm_hash in mm_hashes:
            future = MagicMock()

            def result(hash_key=mm_hash):
                assert fake_executor.submit.call_count == len(mm_hashes)
                completed_loads.add(hash_key)
                return _CacheLoadResult(
                    tensors[hash_key], True, None, mock_shms[hash_key]
                )

            future.result.side_effect = result
            futures[mm_hash] = future

        fake_executor.submit.side_effect = lambda _fn, mm_hash: futures[mm_hash]
        connector._shm_load_executor = fake_executor

        encoder_cache = {}
        with patch.object(connector, '_close_shm_async') as mock_close_async:
            def close_async(_shm):
                assert completed_loads == set(mm_hashes)

            mock_close_async.side_effect = close_async
            connector.start_load_caches(encoder_cache)

        assert fake_executor.submit.call_count == len(mm_hashes)
        assert mock_close_async.call_count == len(mm_hashes)
        assert set(encoder_cache) == set(mm_hashes)
        for mm_hash in mm_hashes:
            assert torch.equal(encoder_cache[mm_hash], tensors[mm_hash])


class TestECSharedMemoryConnectorTPBroadcast:
    """Test TP broadcast path for rank0-only shared-memory reads."""

    def test_tp_src_rank_uses_first_tp_group_rank(self):
        """Test _tp_src_rank uses the first rank from TP group ranks."""
        tp_group = Mock()
        tp_group.ranks = [4, 5]

        with patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tp_group',
            return_value=tp_group
        ):
            assert ECSharedMemoryConnector._tp_src_rank() == 4

    def test_tp_src_rank_raises_when_ranks_missing(self):
        """Test _tp_src_rank raises when TP group ranks are missing."""
        tp_group = Mock()
        tp_group.ranks = []

        with patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tp_group',
            return_value=tp_group
        ), pytest.raises(RuntimeError, match="TP group does not expose ranks"):
            ECSharedMemoryConnector._tp_src_rank()

    def test_tp_device_group_returns_device_group(self):
        """Test _tp_device_group returns device_group when available."""
        device_group = object()
        tp_group = Mock()
        tp_group.device_group = device_group

        with patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tp_group',
            return_value=tp_group
        ):
            assert ECSharedMemoryConnector._tp_device_group() is device_group

    def test_tp_device_group_raises_when_group_missing(self):
        """Test _tp_device_group raises when device_group is missing."""
        tp_group = Mock()
        tp_group.device_group = None

        with patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tp_group',
            return_value=tp_group
        ), pytest.raises(RuntimeError, match="TP group does not expose device_group"):
            ECSharedMemoryConnector._tp_device_group()

    def test_cache_meta_row_roundtrip(self):
        """Test tensor metadata row encodes and decodes cache tensor metadata."""
        tensor = torch.randn(2, 3, dtype=torch.float32)
        row = ECSharedMemoryConnector._encode_cache_meta_row(tensor, True)

        ok, shape, dtype, device_type = ECSharedMemoryConnector._decode_cache_meta_row(row)

        assert ok is True
        assert shape == tuple(tensor.shape)
        assert dtype == tensor.dtype
        assert device_type == tensor.device.type
        assert len(row) == META_WIDTH

    def test_cache_meta_row_padding_and_failure(self):
        """Test tensor metadata row pads shape and encodes failures with ok=0."""
        tensor = torch.randn(2, 3, dtype=torch.float32)
        row = ECSharedMemoryConnector._encode_cache_meta_row(tensor, True)
        assert row[6:] == [0] * (META_WIDTH - 6)

        failed_row = ECSharedMemoryConnector._encode_cache_meta_row(None, False)
        ok, shape, dtype, device_type = ECSharedMemoryConnector._decode_cache_meta_row(failed_row)
        assert ok is False
        assert shape is None
        assert dtype is None
        assert device_type is None

    def test_cache_meta_row_rejects_too_many_dims(self):
        """Test tensor metadata row rejects shapes wider than the fixed header."""
        tensor = torch.empty((1,) * 17)
        with pytest.raises(ValueError, match="exceeds max"):
            ECSharedMemoryConnector._encode_cache_meta_row(tensor, True)

    def test_cache_meta_row_rejects_unsupported_dtype(self):
        """Test tensor metadata rejects dtypes outside the wire mapping."""
        tensor = torch.ones((2, 2), dtype=torch.complex64, device="cpu")

        with pytest.raises(ValueError, match="Unsupported.*dtype"):
            ECSharedMemoryConnector._encode_cache_meta_row(tensor, True)

    @pytest.mark.parametrize(
        "code_index, error_match",
        [
            (META_DTYPE_IDX, "dtype code"),
            (META_DEVICE_IDX, "device code"),
        ],
    )
    def test_cache_meta_row_rejects_unknown_codes(self, code_index, error_match):
        """Test invalid metadata codes fail explicitly during decode."""
        row = ECSharedMemoryConnector._encode_cache_meta_row(
            torch.ones((2, 2), dtype=torch.float32), True
        )
        row[code_index] = 999

        with pytest.raises(ValueError, match=error_match):
            ECSharedMemoryConnector._decode_cache_meta_row(row)

    def test_start_load_caches_rank0_broadcasts_loaded_tensor(self, mock_vllm_config, tp_broadcast_patches):
        """Test rank0 loads shared memory and broadcasts cache to TP peers."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "rank0_hash"
        tensor = torch.randn(4, 8)
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        mock_buf = build_raw_tensor_shm_buffer(tensor)
        mock_shm = MagicMock()
        mock_shm.buf = mock_buf
        mock_shm.size = len(mock_buf)
        device_group = object()

        broadcast_calls = []

        def assert_rank0_broadcast(value, src, group):
            assert src == 0
            assert group is device_group
            broadcast_calls.append(value)
            if len(broadcast_calls) == 1:
                assert value.shape == (1, META_WIDTH)
                ok, shape, dtype, device_type = connector._decode_cache_meta_row(value[0])
                assert ok is True
                assert shape == tuple(tensor.shape)
                assert dtype == tensor.dtype
                assert device_type == "cpu"
            else:
                assert value.shape == tensor.shape
                assert value.dtype == tensor.dtype

        with patch(
            'multiprocessing.shared_memory.SharedMemory', return_value=mock_shm
        ) as mock_shm_class, tp_broadcast_patches(
            connector, 0, device_group, assert_rank0_broadcast
        ) as (mock_broadcast_meta, mock_broadcast):
            encoder_cache = {}
            with patch.object(connector, '_close_shm_async') as mock_close_async:
                connector.start_load_caches(encoder_cache)

        mock_shm_class.assert_called_once()
        mock_close_async.assert_called_once_with(mock_shm)
        mock_broadcast_meta.assert_not_called()
        assert mock_broadcast.call_count == 2
        loaded_tensor = encoder_cache.get(mm_hash)
        assert isinstance(loaded_tensor, torch.Tensor)
        assert loaded_tensor.shape == tensor.shape
        assert loaded_tensor.dtype == tensor.dtype

    def test_start_load_caches_waits_all_loads_before_tp_broadcast(self, mock_vllm_config, tp_broadcast_patches):
        """Test TP broadcast starts only after all submitted shm loads complete."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hashes = ["hash1", "hash2", "hash3"]
        metadata = ECSharedMemoryConnectorMetadata()
        for mm_hash in mm_hashes:
            metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        tensors = {
            mm_hash: torch.full((2, 2), i, dtype=torch.float32)
            for i, mm_hash in enumerate(mm_hashes)
        }
        completed_loads = set()
        fake_executor = MagicMock()
        futures = {}

        for mm_hash in mm_hashes:
            future = MagicMock()

            def result(hash_key=mm_hash):
                completed_loads.add(hash_key)
                return _CacheLoadResult(tensors[hash_key], True, None, None)

            future.result.side_effect = result
            futures[mm_hash] = future

        fake_executor.submit.side_effect = lambda _fn, mm_hash: futures[mm_hash]
        connector._shm_load_executor = fake_executor
        device_group = object()
        broadcast_calls = []

        def assert_all_loads_done(value, src, group):
            assert src == 0
            assert group is device_group
            broadcast_calls.append(value)
            if len(broadcast_calls) == 1:
                assert completed_loads == set(mm_hashes)
                assert value.shape == (len(mm_hashes), META_WIDTH)
                decoded = [connector._decode_cache_meta_row(row) for row in value]
                assert [item[0] for item in decoded] == [True, True, True]

        with tp_broadcast_patches(
            connector, 0, device_group, assert_all_loads_done
        ) as (mock_broadcast_meta, mock_broadcast):
            encoder_cache = {}
            connector.start_load_caches(encoder_cache)

        assert fake_executor.submit.call_count == len(mm_hashes)
        assert completed_loads == set(mm_hashes)
        mock_broadcast_meta.assert_not_called()
        assert mock_broadcast.call_count == len(mm_hashes) + 1
        assert set(encoder_cache) == set(mm_hashes)

    def test_start_load_caches_non_rank0_receives_broadcast_without_shm_read(
            self, mock_vllm_config, tp_broadcast_patches):
        """Test non-rank0 receives broadcasted cache without touching shared memory."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "rank1_hash"
        tensor = torch.randn(4, 8)
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        device_group = object()
        broadcast_calls = []

        def fill_broadcast(value, src, group):
            assert src == 0
            assert group is device_group
            broadcast_calls.append(value)
            if len(broadcast_calls) == 1:
                meta_tensor = torch.tensor(
                    [connector._encode_cache_meta_row(tensor, True)],
                    dtype=torch.int64,
                    device=value.device,
                )
                value.copy_(meta_tensor)
            else:
                value.copy_(tensor)

        with patch(
            'multiprocessing.shared_memory.SharedMemory'
        ) as mock_shm_class, tp_broadcast_patches(
            connector, 1, device_group, fill_broadcast
        ) as (mock_broadcast_meta, mock_broadcast):
            encoder_cache = {}
            connector.start_load_caches(encoder_cache)

        mock_shm_class.assert_not_called()
        mock_broadcast_meta.assert_not_called()
        assert mock_broadcast.call_count == 2
        loaded_tensor = encoder_cache.get(mm_hash)
        assert isinstance(loaded_tensor, torch.Tensor)
        assert loaded_tensor.shape == tensor.shape
        assert loaded_tensor.dtype == tensor.dtype

    def test_start_load_caches_rank0_broadcasts_load_failure(self, mock_vllm_config, tp_broadcast_patches):
        """Test rank0 broadcasts failed shared-memory load and skips tensor broadcast."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "missing_rank0_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        device_group = object()

        def assert_failure_meta(value, src, group):
            assert src == 0
            assert group is device_group
            assert value.shape == (1, META_WIDTH)
            ok, shape, dtype, device_type = connector._decode_cache_meta_row(value[0])
            assert ok is False
            assert shape is None
            assert dtype is None
            assert device_type is None

        with patch(
            'multiprocessing.shared_memory.SharedMemory',
            side_effect=FileNotFoundError("missing")
        ) as mock_shm_class, tp_broadcast_patches(
            connector, 0, device_group, assert_failure_meta
        ) as (mock_broadcast_meta, mock_broadcast):
            encoder_cache = {}
            connector.start_load_caches(encoder_cache)

        mock_shm_class.assert_called_once()
        mock_broadcast_meta.assert_not_called()
        mock_broadcast.assert_called_once()
        assert mm_hash not in encoder_cache

    def test_start_load_caches_rank0_tensor_meta_mixed_success_failure(
            self, mock_vllm_config, tp_broadcast_patches):
        """Test tensor metadata handles mixed successful and failed shm loads."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        ok_hash = "ok_hash"
        missing_hash = "missing_hash"
        tensor = torch.randn(4, 8)
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(ok_hash)
        metadata.add_mm_hash(missing_hash)
        connector.bind_connector_metadata(metadata)

        fake_executor = MagicMock()
        ok_future = MagicMock()
        ok_future.result.return_value = _CacheLoadResult(tensor, True, None, None)
        missing_future = MagicMock()
        missing_future.result.return_value = _CacheLoadResult(
            None, False, "missing", None
        )
        futures = {
            ok_hash: ok_future,
            missing_hash: missing_future,
        }
        fake_executor.submit.side_effect = lambda _fn, mm_hash: futures[mm_hash]
        connector._shm_load_executor = fake_executor
        device_group = object()
        broadcast_calls = []

        def assert_mixed_broadcast(value, src, group):
            assert src == 0
            assert group is device_group
            broadcast_calls.append(value)
            if len(broadcast_calls) == 1:
                assert value.shape == (2, META_WIDTH)
                ok_meta = connector._decode_cache_meta_row(value[0])
                missing_meta = connector._decode_cache_meta_row(value[1])
                assert ok_meta == (True, tuple(tensor.shape), tensor.dtype, tensor.device.type)
                assert missing_meta == (False, None, None, None)
            else:
                assert value.shape == tensor.shape
                assert value.dtype == tensor.dtype

        with tp_broadcast_patches(
            connector, 0, device_group, assert_mixed_broadcast
        ) as (mock_broadcast_meta, mock_broadcast):
            encoder_cache = {}
            connector.start_load_caches(encoder_cache)

        mock_broadcast_meta.assert_not_called()
        assert mock_broadcast.call_count == 2
        assert ok_hash in encoder_cache
        assert missing_hash not in encoder_cache

    def test_start_load_caches_non_rank0_skips_failed_broadcast(
            self, mock_vllm_config, tp_broadcast_patches):
        """Test non-rank0 skips cache insert when rank0 broadcasts load failure."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "missing_rank1_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        device_group = object()

        def fill_failure_meta(value, src, group):
            assert src == 0
            assert group is device_group
            value.copy_(torch.zeros_like(value))

        with patch(
            'multiprocessing.shared_memory.SharedMemory'
        ) as mock_shm_class, tp_broadcast_patches(
            connector, 1, device_group, fill_failure_meta
        ) as (mock_broadcast_meta, mock_broadcast):
            encoder_cache = {}
            connector.start_load_caches(encoder_cache)

        mock_shm_class.assert_not_called()
        mock_broadcast_meta.assert_not_called()
        mock_broadcast.assert_called_once()
        assert mm_hash not in encoder_cache


class TestECSharedMemoryConnectorDelayedClose:
    """Test async shared-memory close executor behavior."""

    def test_close_shm_async_submits_close(self, mock_vllm_config):
        """Test _close_shm_async submits shm.close to the executor."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mock_shm = MagicMock()
        mock_executor = MagicMock()
        connector._shm_close_executor = mock_executor

        connector._close_shm_async(mock_shm)

        mock_executor.submit.assert_called_once_with(mock_shm.close)

    def test_start_load_caches_closes_loaded_shm_async(self, mock_vllm_config):
        """Test successful load schedules shared-memory close asynchronously."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "close_hash"
        tensor = torch.randn(4, 8)
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        mock_buf = build_raw_tensor_shm_buffer(tensor)
        mock_shm = MagicMock()
        mock_shm.buf = mock_buf
        mock_shm.size = len(mock_buf)

        with patch('multiprocessing.shared_memory.SharedMemory', return_value=mock_shm), patch.object(
            connector, '_close_shm_async'
        ) as mock_close_async, patch(
            'omni.connector.ec_connector.shared_memory_connector.current_platform'
        ) as mock_platform:
            mock_platform.device_type = 'cpu'
            encoder_cache = {}
            connector.start_load_caches(encoder_cache)

        assert mm_hash in encoder_cache
        mock_close_async.assert_called_once_with(mock_shm)

    def test_start_load_caches_closes_shm_when_metadata_broadcast_fails(
        self, mock_vllm_config,
    ):
        """Test the outer finally closes shm after a TP broadcast failure."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "broadcast_failure_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        mock_shm = MagicMock()
        result = _CacheLoadResult(torch.ones(2, 2), True, None, mock_shm)

        with patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
            return_value=0
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_world_size',
            return_value=2
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.dist.is_available',
            return_value=True
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.dist.is_initialized',
            return_value=True
        ), patch.object(
            connector, '_load_single_cache', return_value=result
        ), patch.object(
            connector, '_broadcast_cache_metadata', side_effect=RuntimeError("broadcast failed")
        ), patch.object(connector, '_close_shm_async') as mock_close_async, pytest.raises(
            RuntimeError, match="broadcast failed"
        ):
            connector.start_load_caches({})

        mock_close_async.assert_called_once_with(mock_shm)

    def test_start_load_caches_closes_shm_when_cache_insert_fails(
        self, mock_vllm_config,
    ):
        """Test the outer finally closes shm after encoder-cache insertion fails."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "cache_insert_failure_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        mock_shm = MagicMock()
        result = _CacheLoadResult(torch.ones(2, 2), True, None, mock_shm)

        class FailingEncoderCache(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("cache insert failed")

        with patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
            return_value=0
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_world_size',
            return_value=1
        ), patch.object(
            connector, '_load_single_cache', return_value=result
        ), patch.object(connector, '_close_shm_async') as mock_close_async, pytest.raises(
            RuntimeError, match="cache insert failed"
        ):
            connector.start_load_caches(FailingEncoderCache())

        mock_close_async.assert_called_once_with(mock_shm)

    def test_load_single_cache_returns_shm_without_closing(self, mock_vllm_config):
        """Test _load_single_cache leaves close scheduling to the caller."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "load_hash"
        tensor = torch.randn(4, 8)
        mock_buf = build_raw_tensor_shm_buffer(tensor)
        mock_shm = MagicMock()
        mock_shm.buf = mock_buf
        mock_shm.size = len(mock_buf)

        with patch('multiprocessing.shared_memory.SharedMemory', return_value=mock_shm), patch.object(
            connector, '_close_shm_async'
        ) as mock_close_async, patch(
            'omni.connector.ec_connector.shared_memory_connector.current_platform'
        ) as mock_platform:
            mock_platform.device_type = 'cpu'
            result = connector._load_single_cache(mm_hash)

        assert result.ok is True
        assert result.error is None
        assert result.shm is mock_shm
        assert torch.equal(tensor.cpu(), result.tensor.cpu())
        mock_close_async.assert_not_called()

    def test_start_load_caches_closes_shm_after_deserialize_failure(self, mock_vllm_config):
        """Test failed deserialization still defers close to start_load_caches."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "bad_payload_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        mock_shm = MagicMock()
        mock_shm.buf = bytearray(BYTE_LENGTH)
        mock_shm.size = BYTE_LENGTH

        with patch('multiprocessing.shared_memory.SharedMemory', return_value=mock_shm), patch.object(
            connector, '_deserialize_cache', side_effect=ValueError("bad payload")
        ), patch.object(connector, '_close_shm_async') as mock_close_async:
            encoder_cache = {}
            connector.start_load_caches(encoder_cache)

        assert mm_hash not in encoder_cache
        mock_close_async.assert_called_once_with(mock_shm)

    def test_start_load_caches_propagates_unexpected_error_and_closes_shm(
        self, mock_vllm_config,
    ):
        """Test unexpected load failures are fail-fast without leaking shm."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mm_hash = "unexpected_error_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)
        mock_shm = MagicMock()
        mock_shm.buf = bytearray(BYTE_LENGTH)
        mock_shm.size = BYTE_LENGTH

        with patch(
            'multiprocessing.shared_memory.SharedMemory', return_value=mock_shm
        ), patch.object(
            connector, '_deserialize_cache', side_effect=RuntimeError("unexpected")
        ), patch.object(connector, '_close_shm_async') as mock_close_async, pytest.raises(
            RuntimeError, match="unexpected"
        ):
            connector.start_load_caches({})

        mock_close_async.assert_called_once_with(mock_shm)

    def test_shutdown_closes_executor(self, mock_vllm_config):
        """Test shutdown stops async close and load executors."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        mock_close_executor = MagicMock()
        mock_load_executor = MagicMock()
        connector._shm_close_executor = mock_close_executor
        connector._shm_load_executor = mock_load_executor

        connector.shutdown()

        mock_close_executor.shutdown.assert_called_once_with(wait=False)
        mock_load_executor.shutdown.assert_called_once_with(wait=False)


class TestECSharedMemoryConnectorHasCaches:
    """Test cache existence checking."""

    def test_has_caches_all_exist(self, mock_vllm_config, mock_request_with_3_mm):
        """Test has_caches returns True when all caches exist."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm.return_value.close = MagicMock()
            result = connector.has_caches(mock_request_with_3_mm)

            # All should exist (mock returns successfully)
            assert len(result) == 3
            assert all(result)

    def test_has_caches_none_exist(self, mock_vllm_config, mock_request_with_3_mm):
        """Test has_caches returns False when caches don't exist."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm.side_effect = FileNotFoundError("Not found")
            result = connector.has_caches(mock_request_with_3_mm)

            # All should not exist
            assert len(result) == 3
            assert not any(result)

    def test_has_caches_updates_need_loads(self, mock_vllm_config, mock_request_with_3_mm):
        """Test has_caches updates _mm_hashes_need_loads."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        with patch('multiprocessing.shared_memory.SharedMemory'):
            connector.has_caches(mock_request_with_3_mm)

            # Should add all hashes to need_loads
            hashes = [f.identifier for f in mock_request_with_3_mm.mm_features]
            for h in hashes:
                assert h in connector._mm_hashes_need_loads


class TestECSharedMemoryConnectorStateManagement:
    """Test state management methods."""

    def test_update_state_after_alloc(self, mock_vllm_config, mock_request_with_3_mm):
        """Test update_state_after_alloc updates need_loads."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        connector.update_state_after_alloc(mock_request_with_3_mm, index=0)
        assert "hash1" in connector._mm_hashes_need_loads

        connector.update_state_after_alloc(mock_request_with_3_mm, index=1)
        assert "hash2" in connector._mm_hashes_need_loads

        connector.update_state_after_alloc(mock_request_with_3_mm, index=2)
        assert "hash3" in connector._mm_hashes_need_loads

    def test_build_connector_meta(self, mock_vllm_config):
        """Test build_connector_meta creates metadata."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        # Add hashes to need_loads
        connector._mm_hashes_need_loads.add("hash1")
        connector._mm_hashes_need_loads.add("hash2")

        scheduler_output = Mock(spec=SchedulerOutput)
        metadata = connector.build_connector_meta(scheduler_output)

        assert isinstance(metadata, ECSharedMemoryConnectorMetadata)
        assert "hash1" in metadata.mm_hashes
        assert "hash2" in metadata.mm_hashes

        # Need loads should be cleared
        assert len(connector._mm_hashes_need_loads) == 0

    def test_build_connector_meta_empty(self, mock_vllm_config):
        """Test build_connector_meta with empty state."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        scheduler_output = Mock(spec=SchedulerOutput)
        metadata = connector.build_connector_meta(scheduler_output)

        assert isinstance(metadata, ECSharedMemoryConnectorMetadata)
        assert len(metadata.mm_hashes) == 0


class TestECSharedMemoryConnectorEviction:
    """Test memory eviction functionality."""

    def test_evict_if_needed_producer(self, mock_vllm_config):
        """Test _evict_if_needed evicts when memory limit reached."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        # Set small max bytes for testing
        connector._max_bytes = 1000

        # Fill up cache - need big enough to trigger eviction
        connector._mm_hash_sizes["hash1"] = 800
        connector._lru["hash1"] = None
        connector._mm_hash_sizes["hash2"] = 700
        connector._lru["hash2"] = None

        with patch.object(connector, '_unlink_shm') as mock_unlink:
            # Try to add 500 bytes, need to evict both since 800 + 700 > 1000
            connector._evict_if_needed(500)

            # Both should be evicted since memory usage would exceed limit
            assert "hash1" not in connector._lru
            assert "hash2" not in connector._lru
            # Should be called at least once
            assert mock_unlink.call_count >= 1

    def test_evict_if_needed_consumer_skips(self, mock_vllm_config_consumer):
        """Test _evict_if_needed skips for consumer."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config_consumer,
            role=ECConnectorRole.WORKER
        )

        connector._mm_hash_sizes["hash1"] = 8000
        connector._lru["hash1"] = None

        with patch.object(connector, '_unlink_shm') as mock_unlink:
            # Should not evict for consumer
            connector._evict_if_needed(500)
            mock_unlink.assert_not_called()

    def test_evict_if_needed_scheduler_skips(self, mock_vllm_config):
        """Test _evict_if_needed skips for SCHEDULER role."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.SCHEDULER
        )

        connector._mm_hash_sizes["hash1"] = 8000
        connector._lru["hash1"] = None

        with patch.object(connector, '_unlink_shm') as mock_unlink:
            # Should not evict for scheduler (role mismatch)
            connector._evict_if_needed(500)
            mock_unlink.assert_not_called()

    def test_unlink_shm(self, mock_vllm_config):
        """Test _unlink_shm removes shared memory."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "test_hash"
        connector._mm_hash_sizes[mm_hash] = 1000
        connector._mm_hash_refcounts[mm_hash] = 1

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm_class:
            mock_shm = MagicMock()
            mock_shm_class.return_value = mock_shm

            connector._unlink_shm(mm_hash)

            # Should close and unlink
            mock_shm.close.assert_called_once()
            mock_shm.unlink.assert_called_once()
            assert mm_hash not in connector._mm_hash_sizes
            assert mm_hash not in connector._mm_hash_refcounts

    def test_unlink_shm_file_not_found(self, mock_vllm_config):
        """Test _unlink_shm handles FileNotFoundError gracefully."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "test_hash"

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm_class:
            mock_shm_class.side_effect = FileNotFoundError("No such file")

            # Should not raise
            connector._unlink_shm(mm_hash)

    def test_unlink_shm_permission_error(self, mock_vllm_config):
        """Test _unlink_shm handles PermissionError gracefully."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "test_hash"
        connector._mm_hash_sizes[mm_hash] = 1000
        connector._mm_hash_refcounts[mm_hash] = 1

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm_class:
            mock_shm_class.side_effect = PermissionError("Permission denied")

            # Should not raise - exception should be caught and logged
            connector._unlink_shm(mm_hash)

    def test_unlink_shm_os_error(self, mock_vllm_config):
        """Test _unlink_shm handles OSError gracefully."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "test_hash"
        connector._mm_hash_sizes[mm_hash] = 1000
        connector._mm_hash_refcounts[mm_hash] = 1

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm_class:
            mock_shm_class.side_effect = OSError("OS error")

            # Should not raise - exception should be caught and logged
            connector._unlink_shm(mm_hash)


class TestECSharedMemoryConnectorMetadataBinding:
    """Test metadata binding lifecycle."""

    def test_bind_connector_metadata(self, mock_vllm_config):
        """Test binding connector metadata."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash("hash1")

        connector.bind_connector_metadata(metadata)

        assert connector._connector_metadata is metadata

    def test_get_connector_metadata(self, mock_vllm_config):
        """Test getting connector metadata."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        metadata = ECSharedMemoryConnectorMetadata()
        connector.bind_connector_metadata(metadata)

        retrieved = connector._get_connector_metadata()

        assert retrieved is metadata

    def test_start_load_caches_with_none_metadata_returns_before_type_check(
        self, mock_vllm_config,
    ):
        """Test absent connector metadata is handled without a TypeError."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        with patch.object(
            connector, '_get_connector_metadata', return_value=None
        ), patch(
            'omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank'
        ) as mock_tp_rank:
            connector.start_load_caches({})

        mock_tp_rank.assert_not_called()

    def test_bind_and_load_with_empty_metadata(self, mock_vllm_config):
        """Test start_load_caches handles empty metadata gracefully."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        # Bind empty metadata
        encoder_cache: dict[str, torch.Tensor] = {}
        empty_metadata = ECSharedMemoryConnectorMetadata()
        connector.bind_connector_metadata(empty_metadata)

        # Should not raise, should return gracefully
        with patch(
            'omni.connector.ec_connector.shared_memory_connector.current_platform'
        ) as mock_platform:
            mock_platform.device_type = 'cpu'
            connector.start_load_caches(encoder_cache)

            # Cache should remain empty since metadata has no hashes
            assert len(encoder_cache) == 0


class TestECSharedMemoryConnectorEdgeCases:
    """Test edge cases and error handling."""

    @patch('omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
           return_value=0)
    def test_save_caches_empty_tensor(self, mock_tp_rank, mock_vllm_config):
        """Test saving empty tensor."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._mm_hash_sizes = {}

        mm_hash = "empty_hash"
        encoder_cache = {mm_hash: torch.empty(0)}
        _, _, total_size = connector._serialize_cache(encoder_cache[mm_hash])

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm_instance = MagicMock()
            mock_shm_instance.size = total_size
            mock_shm_instance.buf = bytearray(total_size)
            mock_shm.return_value = mock_shm_instance

            # Should not raise
            connector.save_caches(encoder_cache, mm_hash)
            assert connector._mm_hash_sizes[mm_hash] == total_size
            assert mm_hash in connector._lru
            assert connector._mm_hash_refcounts[mm_hash] == 1

    def test_has_caches_empty_request(self, mock_vllm_config):
        """Test has_caches with request that has no MM data."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mock_request = MockRequest("empty_req", [])

        result = connector.has_caches(mock_request)

        assert len(result) == 0
        assert result == []

    @patch('omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
           return_value=0)
    def test_multiple_save_and_reload_refcount(self, mock_tp_rank, mock_vllm_config):
        """Test refcount increases on multiple operations."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._mm_hash_sizes = {}

        mm_hash = "refcount_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm_instance = MagicMock()
            _, _, total_size = connector._serialize_cache(encoder_cache[mm_hash])
            mock_shm_instance.size = total_size
            mock_shm_instance.buf = bytearray(mock_shm_instance.size)
            mock_shm.return_value = mock_shm_instance

            # First save
            connector.save_caches(encoder_cache, mm_hash)
            assert connector._mm_hash_refcounts[mm_hash] == 1

            # Second save (should update, but refcount might increment)
            connector.save_caches(encoder_cache, mm_hash)
            refcount = connector._mm_hash_refcounts[mm_hash]
            assert refcount >= 1

    def test_role_different(self, mock_vllm_config):
        """Test connector with different roles."""
        # SCHEDULER role
        scheduler = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.SCHEDULER
        )
        assert scheduler.role == ECConnectorRole.SCHEDULER

        # WORKER role
        worker = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        assert worker.role == ECConnectorRole.WORKER

    @patch('omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
           return_value=0)
    def test_save_caches_file_already_exists_with_same_size(self, mock_tp_rank, mock_vllm_config):
        """Test save_caches reuses existing shared memory with the same size."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._mm_hash_sizes = {}

        mm_hash = "existing_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}
        mock_buf = build_raw_tensor_shm_buffer(encoder_cache[mm_hash])

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            # First call raises FileExistsError, second call succeeds
            mock_shm_instance = MagicMock()
            _, _, total_size = connector._serialize_cache(encoder_cache[mm_hash])
            mock_shm_instance.size = total_size
            mock_shm_instance.buf = bytearray(mock_shm_instance.size)
            mock_shm.side_effect = [FileExistsError(), mock_shm_instance, mock_shm_instance]

            connector.save_caches(encoder_cache, mm_hash)

            # Should create once, then reopen and reuse the same-size segment.
            assert mock_shm.call_count == 2
            create_call, reuse_call = mock_shm.call_args_list
            assert create_call.kwargs["create"] is True
            assert create_call.kwargs["size"] == total_size
            assert reuse_call.kwargs == {"name": create_call.kwargs["name"]}
            mock_shm_instance.close.assert_called_once()
            mock_shm_instance.unlink.assert_not_called()

    @patch('omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
           return_value=0)
    def test_save_caches_file_already_exists_with_different_size(self, mock_tp_rank, mock_vllm_config):
        """Test save_caches recreates existing shared memory with different size."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._mm_hash_sizes = {}

        mm_hash = "existing_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}
        _, _, total_size = connector._serialize_cache(encoder_cache[mm_hash])

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            old_shm_instance = MagicMock()
            old_shm_instance.size = total_size + 1
            old_shm_instance.buf = bytearray(old_shm_instance.size)
            new_shm_instance = MagicMock()
            new_shm_instance.size = total_size
            new_shm_instance.buf = bytearray(total_size)
            mock_shm.side_effect = [FileExistsError(), old_shm_instance, new_shm_instance]

            connector.save_caches(encoder_cache, mm_hash)

            assert mock_shm.call_count == 3
            old_shm_instance.close.assert_called_once()
            old_shm_instance.unlink.assert_called_once()

    @patch('omni.connector.ec_connector.shared_memory_connector.get_tensor_model_parallel_rank',
           return_value=0)
    def test_save_caches_unlinks_stale_shm_when_close_fails(
            self, mock_tp_rank, mock_vllm_config):
        """Test stale shm is unlinked even if close fails during replacement."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )
        connector._mm_hash_sizes = {}

        mm_hash = "stale_hash"
        encoder_cache = {mm_hash: torch.randn(10, 768)}
        _, _, total_size = connector._serialize_cache(encoder_cache[mm_hash])

        stale_shm = MagicMock()
        stale_shm.size = total_size + 1
        stale_shm.close.side_effect = RuntimeError("close failed")
        new_shm = MagicMock()
        new_shm.size = total_size
        new_shm.buf = bytearray(total_size)

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm.side_effect = [FileExistsError(), stale_shm, new_shm]

            connector.save_caches(encoder_cache, mm_hash)

        assert mock_shm.call_count == 3
        stale_shm.close.assert_called_once()
        stale_shm.unlink.assert_called_once()
        new_shm.close.assert_called_once()

    def test_start_load_caches_file_not_found(self, mock_vllm_config):
        """Test start_load_caches handles FileNotFoundError gracefully."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "nonexistent_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm.side_effect = FileNotFoundError("Cache not found")

            encoder_cache = {}
            connector.start_load_caches(encoder_cache)

            # Cache should remain empty since file was not found
            assert mm_hash not in encoder_cache

    def test_start_load_caches_permission_error(self, mock_vllm_config):
        """Test start_load_caches handles PermissionError gracefully."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "permission_denied_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm.side_effect = PermissionError("Permission denied")

            encoder_cache = {}
            # Should NOT raise - PermissionError is caught by start_load_caches
            connector.start_load_caches(encoder_cache)

            # Cache should remain empty since load failed due to permission error
            assert mm_hash not in encoder_cache

    def test_start_load_caches_os_error(self, mock_vllm_config):
        """Test start_load_caches handles OSError gracefully."""
        connector = ECSharedMemoryConnector(
            vllm_config=mock_vllm_config,
            role=ECConnectorRole.WORKER
        )

        mm_hash = "os_error_hash"
        metadata = ECSharedMemoryConnectorMetadata()
        metadata.add_mm_hash(mm_hash)
        connector.bind_connector_metadata(metadata)

        with patch('multiprocessing.shared_memory.SharedMemory') as mock_shm:
            mock_shm.side_effect = OSError("OS error")

            encoder_cache = {}
            # Should NOT raise - OSError is caught by start_load_caches
            connector.start_load_caches(encoder_cache)

            # Cache should remain empty since load failed due to OS error
            assert mm_hash not in encoder_cache
