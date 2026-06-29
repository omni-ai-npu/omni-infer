# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Main decode connector worker implementation."""

import json
import multiprocessing
import os
import pickle
import queue
import subprocess
import threading
from collections import defaultdict
from multiprocessing import current_process
from typing import Dict, List, Optional, Set, Tuple

import torch
import zmq
from concurrent.futures import ThreadPoolExecutor
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

from omni_cache.connector.utils.helpers import align_remote_block_ids
from omni_cache.connector.utils.metadata import DatadistConnectorMetadata, PendingReq, DTypeUtils, _SendItem
from omni_cache.connector.utils.process_utils import handle_exception, stdout_printer, stdout_reader
from omni_cache.connector.utils.settings import (
    OX_LOG_PATH,
    OX_PATH,
    P_NODE_PORT_LIST,
    PER_REQUEST_CONNECTION,
    ZMQ_BASE_PORT,
)
from omni_cache.connector.zmq_transport import RouterDealerClient, ZMQSendProxy

from .kv_loader import KVLoader, handle_future_callback
from .process_manager import ProcessManager, ZMQClientWrapper

logger = init_logger("vllm.v1.omni")


class DecodeConnectorWorker:
    """Worker implementation for datadist (decode)."""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, cluster_id_start: int):
        """Initialize the decode connector worker.

        Args:
            vllm_config: vLLM configuration.
            host_ip: Host IP address.
            cluster_id_start: Starting cluster ID.
        """
        self.vllm_config = vllm_config
        self.cluster_id_start = cluster_id_start
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.host_ip = host_ip

        additional_config = vllm_config.additional_config or {}
        self.async_pull_kv = additional_config.get("async_pull_kv", False)

        # Initialize state
        self._init_state()
        # Initialize ZMQ
        self._init_zmq()
        # Initialize process pool
        self._init_process_pool()

        self.omni_cache = None
        self.kv_loader = KVLoader(self)
        self.process_mgr = ProcessManager(self)

        # Start async pull thread if enabled
        if self.async_pull_kv:
            self._start_async_pull_thread()

    def _init_state(self) -> None:
        """Initialize internal state."""
        self._recving_transfers = queue.Queue()
        self._done_recving_count: defaultdict[str, int] = defaultdict(lambda: 0)
        self._pull_kv_lock = threading.Lock()
        self.queues: Dict[str, queue.Queue] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self._transfer_lock = threading.Lock()

    def _init_zmq(self) -> None:
        """Initialize ZMQ context and sockets."""
        self.ctx = zmq.Context()
        self.zmq_socket_map: Dict[str, zmq.Socket] = {}
        self.endpoint = f"tcp://127.0.0.1:{ZMQ_BASE_PORT}"
        self.zmq_client = ZMQClientWrapper(self.endpoint)

    def _init_process_pool(self) -> None:
        """Initialize process pool and queues."""
        logger.info(" ***** Using single thread to pull kv.")
        self.executor = ThreadPoolExecutor(max_workers=1)

        self._send_q: "multiprocessing.Queue[_SendItem]" = multiprocessing.Queue()
        self._resp_process: Optional[multiprocessing.Process] = None
        self._resp_stop = multiprocessing.Event()

        self._address_process: Optional[multiprocessing.Process] = None
        self._address_stop = multiprocessing.Event()
        self._recv_q: "multiprocessing.Queue[_SendItem]" = multiprocessing.Queue()

        self._h2d_q: "multiprocessing.Queue[PendingReq]" = multiprocessing.Queue(maxsize=1024)
        self._h2d_stop = threading.Event()
        self._h2d_thread: Optional[threading.Thread] = None

        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # Create shared manager
        curr_proc = current_process()
        was_daemon = curr_proc.daemon
        curr_proc.daemon = False
        try:
            self._mgr = multiprocessing.Manager()
        finally:
            curr_proc.daemon = was_daemon
        self._pending = self._mgr.dict()

    def _start_async_pull_thread(self) -> None:
        """Start async pull KV thread."""
        thread_name = f"async_pull_kv_{self.dp_rank}"
        self.thread_on_fast_path_req = threading.Thread(target=self.on_fast_path_req, daemon=True, name=thread_name)
        self.thread_on_fast_path_req.start()
        logger.warning("DecodeConnectorWorker initialized with async_pull_kv enabled.")

    @property
    def pending(self):
        """Get pending requests dict."""
        return self._pending

    @property
    def send_q(self):
        """Get send queue."""
        return self._send_q

    @property
    def recv_q(self):
        """Get receive queue."""
        return self._recv_q

    @property
    def h2d_q(self):
        """Get H2D queue."""
        return self._h2d_q

    @property
    def resp_process(self):
        """Get response process."""
        return self._resp_process

    @resp_process.setter
    def resp_process(self, value):
        self._resp_process = value

    @property
    def resp_stop(self):
        """Get response stop event."""
        return self._resp_stop

    @property
    def address_process(self):
        """Get address process."""
        return self._address_process

    @address_process.setter
    def address_process(self, value):
        self._address_process = value

    @property
    def address_stop(self):
        """Get address stop event."""
        return self._address_stop

    @property
    def h2d_thread(self):
        """Get H2D thread."""
        return self._h2d_thread

    @h2d_thread.setter
    def h2d_thread(self, value):
        self._h2d_thread = value

    @property
    def h2d_stop(self):
        """Get H2D stop event."""
        return self._h2d_stop

    @property
    def monitor_thread(self):
        """Get monitor thread."""
        return self._monitor_thread

    @monitor_thread.setter
    def monitor_thread(self, value):
        self._monitor_thread = value

    @property
    def monitor_stop(self):
        """Get monitor stop event."""
        return self._monitor_stop

    @property
    def recving_transfers(self):
        """Get receiving transfers queue."""
        return self._recving_transfers

    def on_fast_path_req(self) -> None:
        """Fast path request handler for async KV pull."""
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        sub.connect(f"ipc:///tmp/sched-pub-{self.vllm_config.parallel_config.data_parallel_rank_local}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")

        while True:
            serialized_data = sub.recv()
            metadata: DatadistConnectorMetadata = pickle.loads(serialized_data)
            for req_id, meta in metadata.requests.items():
                if len(meta.local_block_ids) > 0 and len(meta.remote_block_ids) > 0:
                    self.start_load_kv(metadata)
                    logger.info(
                        "Received fast path request for request %s with local_block_ids: %s, remote_block_ids: %s.",
                        req_id,
                        len(meta.local_block_ids),
                        len(meta.remote_block_ids),
                    )

    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, omni_cache):
        """Register KV caches with the connector.

        Args:
            kv_pool_mmap_path: Path to KV pool memory map.
            data_type: Data type of KV cache.
            block_len_dtype: Block length in data type units.
            omni_cache: OmniCache instance.
        """
        logger.warning(
            f" ======= OX parameters for D client: "
            f"{omni_cache.num_blocks=};{kv_pool_mmap_path=}, "
            f"{data_type=}, {block_len_dtype=}; {omni_cache.head_sizes}"
        )

        if omni_cache.dp_local_rank == 0:
            self._start_ox_process(omni_cache, kv_pool_mmap_path)

        self.omni_cache = omni_cache
        self.kv_loader.set_omni_cache(omni_cache)
        self.process_mgr.set_omni_cache(omni_cache)

        # Start subprocesses
        curr_proc = current_process()
        was_daemon = curr_proc.daemon
        curr_proc.daemon = False
        try:
            self.process_mgr.ensure_processes_started()
        finally:
            curr_proc.daemon = was_daemon

    def _start_ox_process(self, omni_cache, kv_pool_mmap_path: str) -> None:
        """Start OX subprocess for KV transfer.

        Args:
            omni_cache: OmniCache instance.
            kv_pool_mmap_path: Path to KV pool memory map.
        """
        end_port = f"{ZMQ_BASE_PORT + omni_cache.dp_local_rank}"
        data_type_size = DTypeUtils.size(self.omni_cache.dtype if self.omni_cache else torch.float16)

        cmd = [
            str(OX_PATH),
            "--shard-list",
            str(P_NODE_PORT_LIST),
            "--zmq-port",
            end_port,
            "--block-table-shm",
            str(kv_pool_mmap_path),
            "--num-block-tables",
            str(omni_cache.dp_world_size_local),
            "--num-blocks",
            str(omni_cache.num_blocks),
            "--num-layers",
            str(omni_cache.num_layers),
            "--tokens-per-block",
            str(omni_cache.block_size),
            "--num-connections-per-req",
            str(PER_REQUEST_CONNECTION),
            "--dims",
            ",".join(map(str, omni_cache.head_sizes)),
        ]
        logger.warning(f"<<<Executing {cmd}")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        q = queue.Queue()

        t_read = threading.Thread(target=stdout_reader, args=(proc.stdout, q))
        t_print = threading.Thread(target=stdout_printer, args=(q, OX_LOG_PATH))
        t_read.daemon = True
        t_print.daemon = True
        t_read.start()
        t_print.start()

    def _create_zmq_client(self):
        """Create ZMQ client instance."""
        return RouterDealerClient(self.endpoint)

    def _create_zmq_proxy(self):
        """Create ZMQ proxy instance."""
        return ZMQSendProxy(self._send_q)

    def start_load_kv(self, metadata: DatadistConnectorMetadata) -> None:
        """Start loading KV from prefill node.

        Args:
            metadata: Datadist connector metadata.
        """
        self.kv_loader.start_load_kv(metadata)

    def _read_blocks(self, *args, **kwargs):
        """Delegate to KVLoader._read_blocks."""
        return self.kv_loader._read_blocks(*args, **kwargs)

    def stop_zmq_thread(self, wait: bool = True) -> None:
        """Stop all ZMQ threads and processes.

        Args:
            wait: Whether to wait for clean shutdown.
        """
        self._resp_stop.set()
        self._h2d_stop.set()
        self._address_stop.set()
        self._monitor_stop.set()

        if self._resp_process and wait:
            self._resp_process.join(timeout=2.0)
        if self._address_process and wait:
            self._address_process.join(timeout=2.0)
        if self._h2d_thread and wait:
            self._h2d_thread.join(timeout=2.0)
        if self._monitor_thread and wait:
            self._monitor_thread.join(timeout=2.0)

        self._resp_process = None
        self._h2d_thread = None
        self._address_process = None
        self._monitor_thread = None
        self._mgr.shutdown()

    def _send_pulled_kv_req_list(self, path: str, data: List[str]) -> None:
        """Send pulled KV request list to prefill node.

        Args:
            path: Socket path.
            data: List of request IDs.
        """
        if path in self.zmq_socket_map:
            socket_ = self.zmq_socket_map[path]
        else:
            socket_ = self.ctx.socket(zmq.PUSH)
            socket_.connect(path)
            self.zmq_socket_map[path] = socket_
            logger.info("create new socket path:%s", path)

        try:
            json_data = json.dumps(data)
            socket_.send_string(json_data)
            logger.info("send string %s path:%s", json_data, path)
        except Exception as e:
            logger.error("Failed to send request_ids to prefill: %s", e)

    def get_finished(self, metadata: DatadistConnectorMetadata) -> Tuple[Set[str], Set[str]]:
        """Get finished transfers.

        Args:
            metadata: Datadist connector metadata.

        Returns:
            Tuple of (done_sending, done_recving) sets.
        """
        all_done_sending: Set[str] = set()
        all_done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(all_done_recving) > 0:
            logger.debug("Get_finished: %s requests done recving", len(all_done_recving))
        return all_done_sending, all_done_recving

    def _pop_done_transfers(self, transfers: queue.Queue) -> Set[str]:
        """Pop done transfers from queue.

        Args:
            transfers: Queue of transfer request IDs.

        Returns:
            Set of done request IDs.
        """
        done_req_ids: Set[str] = set()
        while True:
            try:
                req_id = transfers.get_nowait()
            except queue.Empty:
                break
            done_req_ids.add(req_id)
        return done_req_ids

    def _log_network_timing(self, ctx: PendingReq) -> None:
        """Log network timing for a request.

        Args:
            ctx: Pending request context.
        """
        t_submit = ctx.t_submit
        t_sent = ctx.t_sent if ctx.t_sent > 0 else t_submit
        t_resp = ctx.t_resp if ctx.t_resp > 0 else time.time()

        cost_submit_to_send = (t_sent - t_submit) * 1000.0
        cost_send_to_resp = (t_resp - t_sent) * 1000.0
        cost_submit_to_resp = (t_resp - t_submit) * 1000.0

        num_blocks = len(ctx.local_block_ids[0]) if ctx.local_block_ids else 0
        logger.warning(
            " ***** Pull kv timing (network only): req_id:%s, num_blocks:%d, "
            "submit->send: %.3f ms, send->resp: %.3f ms, submit->resp: %.3f ms",
            ctx.request_id,
            num_blocks,
            cost_submit_to_send,
            cost_send_to_resp,
            cost_submit_to_resp,
        )

    def start_load_kv_mock_prefill(self, metadata: DatadistConnectorMetadata) -> None:
        """Mock prefill mode for testing.

        Args:
            metadata: Datadist connector metadata.
        """
        self.kv_loader.start_load_kv_mock_prefill(metadata)

    def _read_blocks_mock_prefill(self, *args, **kwargs):
        """Delegate to KVLoader._read_blocks_mock_prefill."""
        return self.kv_loader._read_blocks_mock_prefill(*args, **kwargs)
