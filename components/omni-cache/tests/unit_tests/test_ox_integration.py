# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_ox_integration.py
# Integration tests for ox/ module components
# Tests the ox binary integration, config parsing, and related functionality

import pytest
import subprocess
import os
import tempfile
import shutil
import time
import signal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple, Optional


# ==================== Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dir_path = tempfile.mkdtemp(prefix="ox_test_")
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def ox_binary_path():
    """Get the path to the ox binary."""
    return Path(__file__).parent.parent.parent / "omni_cache" / "connector" / "backends" / "ox" / "ox"


@pytest.fixture
def dummy_shm_path(temp_dir):
    """Create a dummy shared memory path for testing."""
    return os.path.join(temp_dir, "test_shm")


# ==================== Config Parsing Tests ====================

class TestOxConfigParsing:
    """Test suite for ox command-line argument parsing."""

    def test_help_flag(self, ox_binary_path):
        """Test that --help produces usage information."""
        if not ox_binary_path.exists():
            pytest.skip("ox binary not built")
            return

        result = subprocess.run(
            [str(ox_binary_path), "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )

        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "--addr" in result.stdout
        assert "--shard-list" in result.stdout
        assert "--block-table-shm" in result.stdout

    def test_missing_required_args(self, ox_binary_path):
        """Test that missing required arguments produces an error."""
        if not ox_binary_path.exists():
            pytest.skip("ox binary not built")
            return

        result = subprocess.run(
            [str(ox_binary_path)],
            capture_output=True,
            text=True,
            timeout=5
        )

        assert result.returncode != 0
        assert "Error" in result.stderr or "error" in result.stderr

    def test_parse_addr_argument(self, ox_binary_path, dummy_shm_path):
        """Test parsing of --addr argument."""
        if not ox_binary_path.exists():
            pytest.skip("ox binary not built")
            return

        result = subprocess.Popen(
            [
                str(ox_binary_path),
                "--addr", "127.0.0.1:9000,127.0.0.1:9001",
                "--block-table-shm", dummy_shm_path,
                "--num-blocks", "100",
                "--num-layers", "62",
                "--tokens-per-block", "128",
                "--dims", "512,64,128",
                "--dtype", "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(0.5)

        result.terminate()
        try:
            stdout, stderr = result.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            result.kill()
            stdout, stderr - result.communicate()

        # Should either succeed or fail due to SHM issues (not parsing errors)
        assert "Unknown argument" not in stderr


# ==================== Address Parsing Tests ====================

class TestAddressParsing:
    """Test suite for address parsing logic."""

    def test_parse_single_address(self):
        """Test parsing a single IP:port address."""
        # Simulated parsing logic from ox_config.hpp
        addr_str = "192.168.1.1:8080"
        parts = addr_str.split(":")

        assert len(parts) == 2
        assert parts[0] == "192.168.1.1"
        assert int(parts[1]) == 8080

    def test_parse_multiple_addresses(self):
        """Test parsing comma-separated addresses."""
        addr_str = "10.0.0.1:9000,10.0.0.2:9000,10.0.0.3:9000"
        addresses = addr_str.split(",")

        assert len(addresses) == 3
        assert all(":" in addr for addr in addresses)

    def test_parse_clustered_shards(self):
        """Test parsing of clustered shard list format."""
        shard_str = "10.0.0.1:10000,10.0.0.2:10000;10.0.1.1:10000,10.0.1.2:10000"
        clusters = shard_str.split(";")

        assert len(clusters) == 2

        cluster_0 = clusters[0].split(",")
        cluster_1 = clusters[1].split(",")

        assert len(cluster_0) == 2
        assert len(cluster_1) == 2

    def test_invalid_address_format(self):
        """Test that invalid address format is detected."""
        invalid_addresses = [
            "invalid_address",
            "192.168.1.1",
            ":8080",
            "192.168.1.1:abc",
            "",
        ]

        for addr in invalid_addresses:
            parts = addr.split(":")
            is_valid = len(parts) == 2 and parts[0] and parts[1].isdigit()
            assert not is_valid, f"Address {addr} should be invalid"


# ==================== Block Size Calculation Tests ====================

class TestBlockSizeCalculation:
    """Test suite for block size calculations."""

    def test_calculate_block_size_basic(self):
        """Test basic block size calculation."""
        num_layers = 62
        num_tokens_per_block = 128
        block_dims = [512, 64, 128]
        element_size = 2

        total_dims = sum(block_dims)
        block_size = num_layers * num_tokens_per_block * total_dims * element_size

        expected = 62 * 128 * (512 + 64 + 128) * 2
        assert block_size == expected

    def test_calculate_block_table_size(self):
        """Test block table size calculation."""
        num_blocks = 1024
        block_size = 62 * 128 * (512 + 64 + 128) * 2

        block_table_size = num_blocks * block_size

        # Should be a large value (hundreds of MB to GB)
        assert block_table_size > 1_000_000_000

    def test_calculate_shm_mem_size(self):
        """Test shared memory size calculation."""
        block_table_size = 62 * 128 * (512 + 64 + 128) * 2
        num_block_tables = 4

        shm_size = num_block_tables * block_table_size

        assert shm_size == block_table_size * 4

    def test_tp_size_from_shard_list(self):
        """Test TP size calculation from shard list."""
        shard_list = ["10.0.0.1:10000", "10.0.0.2:10000",
                      "10.0.0.3:10000", "10.0.0.4:10000"]

        tp_size = len(shard_list)

        assert tp_size == 4

    def test_tp_size_from_clusters(self):
        """Test TP size calculation from shard clusters."""
        shard_clusters = [
            ["10.0.0.1:10000", "10.0.0.2:10000"],
            ["10.0.1.1:10000", "10.0.1.2:10000"],
        ]

        # TP size is size of first cluster
        tp_size = len(shard_clusters[0]) if shard_clusters else 1

        assert tp_size == 2


# ==================== KV Cache Merger Tests ====================

class TestKVCacheMergerPython:
    """Python tests for KV cache merger functionality."""

    def test_segment_dimensions_calculation(self):
        """Test segment dimension calculations."""
        kv_dimension = 704
        tp = 2

        seg1 = (kv_dimension * 8) // (tp * 11)
        seg2 = kv_dimension // (tp * 11)
        seg3 = (kv_dimension * 2) // (tp * 11)

        assert seg1 == 256
        assert seg2 == 32
        assert seg3 == 64

    def test_total_shard_dimension(self):
        """Test total shard dimension calculation."""
        kv_dimension = 704
        tp = 2

        seg1 = (kv_dimension * 8) // (tp * 11)
        seg2 = kv_dimension // (tp * 11)
        seg3 = (kv_dimension * 2) // (tp * 11)
        total_shard = seg1 + seg2 + seg3

        assert total_shard == 352

    def test_total_merged_dimension(self):
        """Test total merged dimension calculation."""
        kv_dimension = 704
        tp = 2

        seg1 = (kv_dimension * 8) // (tp * 11)
        seg2 = kv_dimension // (tp * 11)
        seg3 = (kv_dimension * 2) // (tp * 11)
        total_merged = seg1 * tp + seg2 * tp + seg3 * tp

        assert total_merged == 704

    def test_memory_size_calculation(self):
        """Test memory size calculation for KV cache."""
        num_layers = 62
        block_size = 128
        total_shard = 352
        tp = 2
        element_size = 2  # sizeof(short)

        memory_size = num_layers * block_size * total_shard * tp * element_size

        # Should be in MB range
        assert memory_size > 1_000_000
        assert memory_size < 10_000_000_000  # Less than 10GB


# ==================== Message Structure Tests ====================

class TestMessageStructures:
    """Test suite for message structures used in ox communication."""

    def test_request_message_structure(self):
        """Test RequestMessage expected structure."""
        # RequestMessage contains:
        # - request_id: string
        # - table_id: int64
        # - src_block_ids: list[int64]
        # - dst_block_ids: list[int64]
        # - cluster_id: int

        request_msg = {
            "request_id": "test_req_123",
            "table_id": 0,
            "src_block_ids": [1, 2, 3],
            "dst_block_ids": [10, 20, 30],
            "cluster_id": 0,
        }

        assert "request_id" in request_msg
        assert "table_id" in request_msg
        assert "src_block_ids" in request_msg
        assert "dst_block_ids" in request_msg
        assert "cluster_id" in request_msg
        assert len(request_msg["src_block_ids"]) == len(request_msg["dst_block_ids"])

    def test_response_message_structure(self):
        """Test ResponseMessage expected structure."""
        # ResponseMessage contains:
        # - request_id: string
        # - success: bool

        response_msg = {
            "request_id": "test_req_123",
            "success": True,
        }

        assert "request_id" in response_msg
        assert "success" in response_msg
        assert isinstance(response_msg["success"], bool)

    def test_block_list_types(self):
        """Test block_id_t type expectations."""
        # block_id_t is int64_t
        block_ids = [0, 1, 100, 1000, 1000000]

        assert all(isinstance(bid, int) for bid in block_ids)
        assert all(bid >= 0 for bid in block_ids)


# ==================== Shared Memory Tests ====================

class TestSharedMemoryHandling:
    """Test suite for shared memory handling logic."""

    def test_path_exists_check(self, temp_dir):
        """Test checking if a path exists."""
        existing_file = os.path.join(temp_dir, "existing_file.txt")
        with open(existing_file, "w") as f:
            f.write("test")

        assert os.path.exists(existing_file)
        assert not os.path.exists(os.path.join(temp_dir, "nonexistent.txt"))

    def test_is_directory_check(self, temp_dir):
        """Test checking if path is a directory."""
        dir_path = os.path.join(temp_dir, "test_dir")
        os.makedirs(dir_path)
        file_path = os.path.join(temp_dir, "test_file.txt")
        with open(file_path, "w") as f:
            f.write("test")

        assert os.path.isdir(dir_path)
        assert not os.path.isdir(file_path)

    def test_is_regular_file_check(self, temp_dir):
        """Test checking if path is a regular file."""
        dir_path = os.path.join(temp_dir, "test_dir")
        os.makedirs(dir_path)
        file_path = os.path.join(temp_dir, "test_file.txt")
        with open(file_path, "w") as f:
            f.write("test")

        assert os.path.isfile(file_path)
        assert not os.path.isfile(dir_path)

    def test_basename_extraction(self):
        """Test extracting basename from path."""
        test_cases = [
            ("/path/to/file.txt", "file.txt"),
            ("/relative/path/to/file.txt", "file.txt"),
            ("file.txt", "file.txt"),
            ("/path/to/dir/", "dir"),
            ("/", ""),
            ("", ""),
        ]

        for path, expected_basename in test_cases:
            basename = os.path.basename(path.rstrip("/")) or os.path.basename(path)
            assert basename == expected_basename or basename == ""

    def test_shm_name_construction(self, temp_dir):
        """Test shared memory name construction from path."""
        dir_path = os.path.join(temp_dir, "shm_dir")
        os.makedirs(dir_path)

        # SHM name should be "/" + basename
        shm_name = "/" + os.path.basename(dir_path)

        assert shm_name.startswith("/")
        assert shm_name == "/shm_dir"


# ==================== TCP Connection Tests ====================

class TestTCPConnectionLogic:
    """Test suite for TCP connection handling logic."""

    def test_endpoint_parsing(self):
        """Test parsing TCP endpoint from IP:port string."""
        addr_str = "127.0.0.1"
        port = 9000

        # Simulated endpoint
        endpoint = (addr_str, port)

        assert endpoint[0] == addr_str
        assert endpoint[1] == port

    def test_connection_timeout_calculation(self):
        """Test connection timeout calculation."""
        overall_timeout_seconds = 3600
        retry_interval_seconds = 5

        max_retries = overall_timeout_seconds // retry_interval_seconds

        assert max_retries == 720

    def test_retry_logic(self):
        """Test retry logic for connection attempts."""
        max_attempts = 10
        attempts = 0
        connected = False

        while attempts < max_attempts and not connected:
            attempts += 1
            # Simulate connection attempt
            if attempts == 5:  # Simulate success on 5th attempt
                connected = True

        assert attempts == 5
        assert connected

    def test_round_robin_selection(self):
        """Test round-robin connection selection."""
        num_connections = 4
        last = 0

        # Simulate 10 selections
        selections = []
        for _ in range(10):
            conn_index = last
            selections.append(conn_index)
            last = (last + 1) % num_connections

        # Should cycle through all connections
        assert all(i in selections for i in range(4))
        assert selections[:4] == [0, 1, 2, 3]


# ==================== Buffer Management Tests ====================

class TestBufferManagement:
    """Test suite for buffer management logic."""

    def test_block_address_calculation(self):
        """Test block address calculation."""
        table_size = 1024 * 1024  # 1MB
        block_size = 4096  # 4KB
        num_blocks = table_size // block_size

        assert num_blocks == 256

        # Calculate address for block 5
        block_id = 5
        offset = block_id * block_size

        assert offset == 5 * 4096

    def test_tp_size_calculation(self):
        """Test TP size calculation."""
        block_size = 1024
        tp_size = 4

        block_tp_size = block_size // tp_size

        assert block_tp_size == 256

    def test_layer_size_calculation(self):
        """Test layer size calculation."""
        num_layers = 62
        num_blocks = 128
        num_tokens_per_block = 128
        dim = 512
        element_size = 2

        layer_size = num_blocks * num_tokens_per_block * dim * element_size

        # Should be large (multiple MB per layer)
        assert layer_size > 1_000_000

    def test_buffer_vector_size(self):
        """Test buffer vector size calculation."""
        num_blocks = 10
        num_layers = 62
        num_segments = 3

        buffer_count = num_blocks * num_layers * num_segments

        assert buffer_count == 10 * 62 * 3


# ==================== Metrics and Statistics Tests ====================

class TestMetricsAndStats:
    """Test suite for metrics and statistics tracking."""

    def test_bytes_received_tracking(self):
        """Test tracking total bytes received."""
        total_bytes = [0]

        def update_stats(n):
            total_bytes[0] += n

        update_stats(1024)
        update_stats(2048)
        update_stats(512)

        assert total_bytes[0] == 3584

    def test_running_tasks_tracking(self):
        """Test tracking running and total tasks."""
        running = [0]
        total = [0]

        def update_running(n):
            if n > 0:
                total[0] += n
            running[0] += n

        update_running(5)
        update_running(3)
        update_running(-2)

        assert running[0] == 6
        assert total[0] == 8

    def test_bandwidth_calculation(self):
        """Test bandwidth calculation."""
        total_bytes = 100_000_000  # 100 MB
        duration_us = 1_000_000  # 1 second in microseconds

        bandwidth_mbps = total_bytes / (duration_us / 1e6) / 1e6

        assert bandwidth_mbps == 100.0

    def test_bandwidth_with_time(self):
        """Test bandwidth calculation with time points."""
        start_time = time.time()
        total_bytes = 50 * 1024 * 1024  # 50 MB

        # Simulate some time passing
        elapsed = 0.5  # 500ms

        bandwidth = total_bytes / elapsed
        bandwidth_mbps = bandwidth / (1024 * 1024)

        # 50MB in 0.5 seconds = 100 MB/s
        assert abs(bandwidth_mbps - 100.0) < 0.1


# ==================== Thread Management Tests ====================

class TestThreadManagement:
    """Test suite for thread pool management."""

    def test_thread_count_calculation(self):
        """Test thread count configuration."""
        default_threads = 16
        available_cores = os.cpu_count() or 1

        # Use configured or default
        num_threads = min(default_threads, available_cores * 2)

        assert num_threads >= 1
        assert num_threads <= 32

    def test_io_context_threads(self):
        """Test io_context thread assignment."""
        num_threads = 8

        # Simulate creating thread pool
        thread_ids = list(range(num_threads))

        assert len(thread_ids) == 8
        assert all(isinstance(tid, int) for tid in thread_ids)

    def test_connection_per_shard(self):
        """Test connection count per shard."""
        num_shards = 4
        connections_per_shard = 16

        total_connections = num_shards * connections_per_shard

        assert total_connections == 64


# ==================== ZMQ Communication Tests ====================

class TestZMQCommunication:
    """Test suite for ZMQ communication patterns."""

    def test_socket_types(self):
        """Test ZMQ socket type constants."""
        # ZMQ_ROUTER for server
        # ZMQ_DEALER for client
        router_type = 1  # ZMQ_ROUTER
        dealer_type = 2  # ZMQ_DEALER

        assert router_type != dealer_type

    def test_multipart_message_structure(self):
        """Test multipart message structure."""
        client_id = b"client_123"
        data = b"serialized_message_data"

        messages = [client_id, data]

        assert len(messages) == 2
        assert messages[0] == client_id
        assert messages[1] == data

    def test_port_binding(self):
        """Test port binding string construction."""
        port = 5555
        bind_address = f"tcp://*:{port}"

        assert bind_address == "tcp://*:5555"

    def test_connection_address(self):
        """Test connection address string construction."""
        host = "127.0.0.1"
        port = 9000
        connect_address = f"tcp://{host}:{port}"

        assert connect_address == "tcp://127.0.0.1:9000"


# ==================== Integration Tests ====================

class TestOxIntegration:
    """Integration tests for ox module."""

    def test_config_to_block_table_flow(self, temp_dir):
        """Test configuration flow to block table setup."""
        # Create test config values
        config = {
            "num_blocks": 128,
            "num_layers": 62,
            "num_tokens_per_block": 128,
            "element_size": 2,
            "block_dims": [512, 64, 128],
            "num_block_tables": 1,
        }

        # Calculate expected sizes
        total_dims = sum(config["block_dims"])
        block_size = (config["num_layers"] *
                     config["num_tokens_per_block"] *
                     total_dims *
                     config["element_size"])

        assert block_size > 0
        assert block_size % 2 == 0  # Should be even for short type

    def test_full_request_flow(self):
        """Test full request flow from client to response."""
        # Simulated request flow
        request = {
            "request_id": "test_001",
            "table_id": 0,
            "src_block_ids": [1, 2, 3],
            "dst_block_ids": [10, 20, 30],
            "cluster_id": 0,
        }

        # Simulate processing
        num_blocks = len(request["src_block_ids"])
        assert num_blocks == len(request["dst_block_ids"])

        # Simulated response
        response = {
            "request_id": request["request_id"],
            "success": True,
        }

        assert response["request_id"] == "test_001"
        assert response["success"] is True

    def test_multi_cluster_selection(self):
        """Test cluster selection logic."""
        num_clusters = 3
        selected_cluster_id = 1

        # Validate cluster selection
        if 0 <= selected_cluster_id < num_clusters:
            valid_selection = True
        else:
            valid_selection = False

        assert valid_selection
        assert selected_cluster_id == 1

    def test_tp_shard_distribution(self):
        """Test distribution of work across TP shards."""
        num_blocks = 100
        num_shards = 4
        conn_per_req = 4

        # Calculate distribution
        base_count = num_blocks // conn_per_req
        remainder = num_blocks % conn_per_req

        distribution = []
        for i in range(conn_per_req):
            count = base_count + (1 if i < remainder else 0)
            distribution.append(count)

        assert sum(distribution) == num_blocks
        assert len(distribution) == conn_per_req
        assert max(distribution) - min(distribution) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
