# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Prefill connector worker implementation for KV transfer operations.

This module handles the prefill-side KV cache transfer to decode nodes.
"""

import json
import queue
import subprocess
import threading
import time
from typing import Dict, List, Tuple

import zmq
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

from vllm.logger import init_logger

logger = init_logger("vllm.v1.omni")
from omni_cache.connector.utils.metadata import (
    DatadistConnectorMetadataPrefill,
    DTypeUtils,
)
from omni_cache.connector.utils.process_utils import (
    stdout_printer,
    stdout_reader,
    stop_logged_process,
    wait_ports,
)
from omni_cache.connector.utils.settings import (
    BASE_PORT,
    BLOCK_RELEASE_DELAY,
    CLUSTER_SIZE,
    OX_PATH,
)


class PrefillConnectorWorker:
    """Implementation of Worker side methods (prefill).

    This class manages the prefill-side KV cache transfer, including:
    - Registering KV caches with the OX transfer service
    - Managing request completion and block release
    - Receiving notifications of completed KV transfers via ZMQ
    """

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, host_port: str):
        """Initialize the PrefillConnectorWorker.

        Args:
            vllm_config: VLLM configuration object
            host_ip: IP address to bind the ZMQ socket
            host_port: Port to bind the ZMQ socket
        """
        # Metadata.
        self.host_ip = host_ip
        self.host_port = host_port
        self.vllm_config = vllm_config
        self.rank = get_tensor_model_parallel_rank()
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        if self.rank == 0:
            self.ctx = zmq.Context()
            self.input_socket = self.ctx.socket(zmq.constants.PULL)
            self.input_socket.bind(f"tcp://{self.host_ip}:{self.host_port}")
            logger.info(f"ConnectWorker bind tcp://{self.host_ip}:{self.host_port}")
            self._transfer_lock = threading.Lock()
            self.receive_req_list: List[str] = []
            thread_name = "prefill_connector_get_pulled_kv_req_list"
            self.thread = threading.Thread(target=self.get_pulled_kv_req_list, daemon=True, name=thread_name)
            self.thread.start()

        # initialize the dict to save requests finish time
        self.requests_finish_time: Dict[str, float] = {}
        self.done_sending_req_ids: Dict[str, float] = {}

    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, omni_cache=None):
        """Register KV caches with the transfer service.

        Args:
            kv_pool_mmap_path: Path to the memory-mapped KV pool
            data_type: Data type of the KV cache
            block_len_dtype: Block length in data type units
            omni_cache: OmniCache instance with cache configuration
        """
        logger.warning(f" ======= OX parameters for P server: {kv_pool_mmap_path=}, {data_type=}, {block_len_dtype=}; {omni_cache.head_sizes=}")
        self._start_p_server_kv_transfer(kv_pool_mmap_path, data_type, block_len_dtype, omni_cache)

    def _start_p_server_kv_transfer(self, kv_pool_mmap_path, data_type, block_len_dtype, omni_cache):
        """Start the P-server KV transfer process.

        Args:
            kv_pool_mmap_path: Path to the memory-mapped KV pool
            data_type: Data type of the KV cache
            block_len_dtype: Block length in data type units
            omni_cache: OmniCache instance with cache configuration
        """
        self.host_block_offset = omni_cache.host_block_offset
        self.tp_rank_local = self.rank % (self.vllm_config.parallel_config.tensor_parallel_size // CLUSTER_SIZE)
        data_type_size = DTypeUtils.size(data_type)
        if self.tp_rank_local == 0 and self.dp_rank == 0:
            cmd = [
                str(OX_PATH),
                "--addr", f"0.0.0.0:{BASE_PORT}",
                "--block-table-shm", str(kv_pool_mmap_path),
                "--num-blocks", str(omni_cache.num_blocks),
                "--num-layers", str(omni_cache.num_layers),
                "--tokens-per-block", str(omni_cache.node_block_size),
                "--dims", ",".join(map(str, omni_cache.head_sizes)),
            ]
            logger.warning(f"<<<Executing {cmd}")

            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1)
            q = queue.Queue()

            t_read = threading.Thread(target=stdout_reader, args=(proc.stdout, q))
            t_print = threading.Thread(target=stdout_printer, args=(q,))
            t_read.daemon = True
            t_print.daemon = True
            t_read.start()
            t_print.start()

            ok, not_ready = wait_ports(host_ports=[("127.0.0.1", BASE_PORT)], timeout_sec=60)
            if not ok:
                stop_logged_process(proc)
                raise RuntimeError(f"[ERROR] P not ready: {sorted(list(not_ready))}")
            logger.info("[READY] P node is ready")

    def start_load_kv(self, metadata: DatadistConnectorMetadataPrefill):
        """Start loading KV cache (no-op for prefill side).

        Args:
            metadata: Prefill metadata object
        """
        pass

    def get_finished(self, metadata: DatadistConnectorMetadataPrefill) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving.
        """
        all_done_sending: set[str] = set()
        all_done_recving: set[str] = set()
        if self.rank == 0:
            # Update requests_finish_time with new finish times from metadata
            with self._transfer_lock:
                current_time = time.monotonic()
                done_retention = max(BLOCK_RELEASE_DELAY, 60)
                for req_id, done_time in list(self.done_sending_req_ids.items()):
                    if current_time - done_time > done_retention:
                        del self.done_sending_req_ids[req_id]

                self.requests_finish_time.update(
                    {
                        req_id: meta.finish_time
                        for req_id, meta in metadata.requests.items()
                        if req_id not in self.done_sending_req_ids
                    }
                )
                # Identify requests whose finish time exceeds BLOCK_RELEASE_DELAY
                out_date_reqs = []
                for req_id, finish_time in self.requests_finish_time.items():
                    if current_time - finish_time > BLOCK_RELEASE_DELAY:
                        out_date_reqs.append(req_id)
                for req_id in out_date_reqs:
                    logger.warning(
                        f"Request {req_id} is out of date, finish time: {self.requests_finish_time[req_id]}. Freeing blocks now."
                    )
                    all_done_sending.add(req_id)
                    self.done_sending_req_ids[req_id] = current_time
                    del self.requests_finish_time[req_id]

            if len(self.receive_req_list) == 0:
                return all_done_sending, all_done_recving

            with self._transfer_lock:
                for req_idx in range(len(self.receive_req_list) - 1, -1, -1):
                    req_id = self.receive_req_list[req_idx]
                    logger.debug(f"Get_finished: request {req_id}")
                    if req_id in self.requests_finish_time:
                        all_done_sending.add(req_id)
                        self.done_sending_req_ids[req_id] = time.monotonic()
                        # if the request's kv has been received, remove it from requests_finish_time
                        del self.requests_finish_time[req_id]
                        self.receive_req_list.pop(req_idx)

        return all_done_sending, all_done_recving

    def get_pulled_kv_req_list(self):
        """Background thread to receive completed KV transfer notifications.

        Listens on the ZMQ socket for notifications from decode nodes
        about completed KV transfers.
        """
        while True:
            try:
                # pyzmq Socket.poll timeout is in milliseconds; check every 1s
                if self.input_socket.poll(timeout=100) > 0:
                    message = self.input_socket.recv_string()
                    id_list = json.loads(message)  # Parse the received JSON string into a list
                    logger.debug("Received: %s", id_list)
                    with self._transfer_lock:
                        self.receive_req_list.extend(id_list)
            except Exception as e:
                logger.error("get pulled kv req list failed: %s", e)
