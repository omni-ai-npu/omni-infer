# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import contextlib
import json
from collections.abc import Iterator
import math
import threading
from typing import TYPE_CHECKING, Any, Optional
import zmq
import os
import time
import pickle

from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger

from typing import TYPE_CHECKING, Any, Optional
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
if TYPE_CHECKING:
    from vllm.config import VllmConfig, KVTransferConfig
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request
from vllm.v1.request import Request
from vllm.utils import round_down
from dataclasses import dataclass
from collections import defaultdict
import torch
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)

from vllm.utils import get_open_port
from vllm.v1.request import RequestStatus
import queue
from concurrent.futures import ThreadPoolExecutor

GET_META_MSG = b"get_meta_msg"

logger = init_logger(__name__)

# Introduce the environment variable VLLM_LLMDATADIST_ZMQ_PORT to resolve ZMQ connection conflicts during
# multi-P deployments on the same machine.
# This variable should not be set separately unless specifically required for this scenario.
VLLM_LLMDATADIST_ZMQ_PORT = int(os.environ.get("VLLM_LLMDATADIST_ZMQ_PORT", "5568"))
thread_dump_path = os.environ.get("VLLM_THREAD_DUMP_PATH", "/tmp/vllm_thread_info")

from omni.accelerators.pd.llmdatadist_manager import LLMDataDistManager, LLMDataDistConfig


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_cluster_id: str
    spec_token_ids: Optional[list[int]]


class DatadistConnectorMetadata(KVConnectorMetadata):
    """Metadata for datadist connector."""

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_host=kv_transfer_params["remote_host_ip"],
            remote_cluster_id=kv_transfer_params["remote_cluster_id"],
            spec_token_ids=kv_transfer_params["spec_token_ids"],
        )


class LLMDataDistConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if vllm_config.kv_transfer_config is None:
            raise RuntimeError("vllm_config.kv_transfer_config cannot be None")
        self.datadist_config = LLMDataDistConfig(vllm_config, ignore_load_rank=True)
        self.cluster_id = self.datadist_config.cluster_id_start
        self.local_info = self.datadist_config.local_info
        self.host_ip = self.local_info.server.server_ip
        self.host_port = VLLM_LLMDATADIST_ZMQ_PORT
        self.is_prefill = vllm_config.kv_transfer_config.kv_role == "kv_producer"

        if role == KVConnectorRole.SCHEDULER:
            if self.is_prefill:
                self.connector_scheduler = PrefillConnectorScheduler(self.cluster_id, self.host_ip, str(self.host_port))
            else:
                self.connector_scheduler = DecodeConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            if self.is_prefill:
                self.connector_worker = PrefillConnectorWorker(vllm_config, str(self.host_ip), str(self.host_port))
            else:
                self.connector_worker = DecodeConnectorWorker(vllm_config, str(self.host_ip), self.cluster_id)
            self.connector_scheduler = None

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.build_connector_metadata(scheduler_output)

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.request_finished(request, block_ids, spec_token_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        if not isinstance(self._connector_metadata, DatadistConnectorMetadata):
            raise RuntimeError("self._connector_metadata must be an instance of DatadistConnectorMetadata")
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Connector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Connector does not save explicitly."""
        pass

    def wait_for_save(self):
        """Connector does not save explicitly."""
        pass

class PrefillConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, cluster_id: str, host_ip: str, host_port: str):
        self.cluster_id = cluster_id
        self.host_ip = host_ip
        self.host_port = host_port
        logger.info("Initializing LLMDataDist Scheduler %s %s %s", cluster_id, host_ip, host_port)

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        pass

    def build_connector_metadata(
            self,
            scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = DatadistConnectorMetadata()
        return metadata

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        delay_free_blocks = len(block_ids) > 0
        return delay_free_blocks, dict(
            remote_block_ids=block_ids,
            remote_cluster_id=self.cluster_id,
            remote_host_ip=f"tcp://{self.host_ip}:{self.host_port}",
            spec_token_ids=spec_token_ids
        )


class PrefillConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, host_port: str):
        # Metadata.
        self.host_ip = host_ip
        self.host_port = host_port
        self.rank = get_tensor_model_parallel_rank()
        if self.rank == 0:
            self.ctx = zmq.Context()
            self.input_socket = self.ctx.socket(zmq.constants.PULL)
            self.input_socket.bind(f"tcp://{self.host_ip}:{self.host_port}")
            self._transfer_lock = threading.Lock()
            self.receive_req_list = []
            thread_name = "prefill_connector_get_pulled_kv_req_list"
            self.thread = threading.Thread(target=self.get_pulled_kv_req_list, daemon=True, name=thread_name)
            self.thread.start()
            dump_thread_to_file(self.thread, thread_name, thread_dump_path)
        from omni.accelerators.cache import OmniBiGroupDataDistManager, ENABLED
        if ENABLED:
            manager_cls = OmniBiGroupDataDistManager
            logger.warning(f"PrefillingConnector is using Omni datadist manager for KV transfer.")
            self.datadist_manager = manager_cls(vllm_config)
        else:
            manager_cls = LLMDataDistManager
            self.datadist_manager = manager_cls(vllm_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.datadist_manager.register_memory(kv_caches)
        self.datadist_manager.register_link()
        pass

    def start_load_kv(self, metadata: DatadistConnectorMetadata):
        pass

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving.
        """
        all_done_sending: set[str] = set()
        all_done_recving: set[str] = set()
        if self.rank == 0:
            if len(self.receive_req_list) == 0:
                return all_done_sending, all_done_recving

            with self._transfer_lock:
                for req_id in self.receive_req_list:
                    logger.debug(f"Get_finished: request {req_id}")
                    all_done_sending.add(req_id)
                self.receive_req_list.clear()

        return all_done_sending, all_done_recving

    def get_pulled_kv_req_list(self):
        while True:
            try:
                if self.input_socket.poll(timeout=10) > 0:
                    message = self.input_socket.recv_string()
                    id_list = json.loads(message)  # Parse the received JSON string into a list
                    logger.debug("Received: %s", id_list)
                    with self._transfer_lock:
                        self.receive_req_list.extend(id_list)
            except Exception as e:
                logger.error("get pulled kv req list failed: %s", e)


class DecodeConnectorScheduler:
    """Implementation of Scheduler side methods"""
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}
        self.processed_request: set[str] = set()

        additional_config = vllm_config.additional_config
        if additional_config:
            self.async_pull_kv = additional_config.get("async_pull_kv", False)
        else:
            self.async_pull_kv = False

        if self.async_pull_kv:
            self.context = zmq.Context()
            self.pub = self.context.socket(zmq.PUB)
            self.pub.bind(f"ipc:///tmp/sched-pub-{vllm_config.parallel_config.data_parallel_rank_local}")

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if request.request_id in self.processed_request:
            return 0, False
        self.processed_request.add(request.request_id)
        params = request.kv_transfer_params
        if params is None:
            return 0, False
        logger.debug(
            "DatadistConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if num_computed_tokens % self.block_size != 0:
            raise RuntimeError("num_computed_tokens must be divisible by self.block_size")
        rounded_num_prompt_tokens = self._round_up(
            len(request.prompt_token_ids), self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        return count, count > 0

    def _round_up(self, x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        logger.debug(f"Request id {request.request_id}: blocks length is {len(blocks.blocks)}")
        params = request.kv_transfer_params
        logger.debug(
            "DatadistConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if params is not None:
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_cluster_id", "remote_host_ip")):
                    self._reqs_need_recv[request.request_id] = (
                        request, blocks.get_unhashed_block_ids())
                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s.", params)

    def build_connector_metadata(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = DatadistConnectorMetadata()
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            if req.kv_transfer_params is None:
                logger.warning(f"For reuqest {req_id}: kv_transfer_params now is None")
            else:
                metadata.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            req.kv_transfer_params = None
        self._reqs_need_recv.clear()

        if self.async_pull_kv:
            if scheduler_output is None:
                # Let go fast path
                if metadata.requests:
                    serialized_data = pickle.dumps(metadata)
                    self.pub.send(serialized_data)

        return metadata

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if request.request_id in self.processed_request:
            self.processed_request.remove(request.request_id)
        return False, None


class DecodeConnectorWorker:
    """Worker implementation for datadist."""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, cluster_id: int):
        self.vllm_config = vllm_config
        self.cluster_id = cluster_id
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        additional_config = vllm_config.additional_config
        if additional_config:
            self.async_pull_kv = additional_config.get("async_pull_kv", False)
            self.multi_thread_pull_kv = additional_config.get("multi_thread_pull_kv", False)
            self.multi_rank_pull_kv = additional_config.get("multi_rank_pull_kv", False)
        else:
            self.async_pull_kv = False
            self.multi_thread_pull_kv = False
            self.multi_rank_pull_kv = False
        if self.multi_rank_pull_kv:
            self.multi_thread_pull_kv = True
        from omni.accelerators.cache import OmniBiGroupDataDistManager, ENABLED
        if ENABLED:
            manager_cls = OmniBiGroupDataDistManager
            logger.warning(f"DecodeConnector is using Omni datadist manager for KV transfer.")
            self.datadist_manager = manager_cls(vllm_config)
        else:
            manager_cls = LLMDataDistManager
            self.datadist_manager = manager_cls(vllm_config)
        self._recving_transfers: list = []
        self._done_recving_count: defaultdict[str, int] = defaultdict(lambda: 0)

        self._pull_kv_lock = threading.Lock()
        self.queues = {} # cluster_id -> queue.Queue
        self.threads = {} # cluster_id -> threading.Thread
        if not self.multi_rank_pull_kv:
            if self.multi_thread_pull_kv:
                num_threads = int(os.environ.get("PREFILL_POD_NUM", "1"))
                for pod_idx in range(num_threads):
                    cluster_id = 16 * pod_idx  # Use 16 as the base for cluster_id
                    with self._pull_kv_lock:
                        q = queue.Queue()
                        self.queues[cluster_id] = q
                        thread_name = f"thread_pull_kv_dp_rank_{self.dp_rank}_cluster_id_{cluster_id}"
                        t = threading.Thread(target=self.worker, args=(cluster_id,), daemon=True, name=thread_name)
                        t.start()
                        self.threads[cluster_id] = t  # Store the thread for this cluster_id
                        logger.debug(f" ***** Created a new thread for pulling kv from cluster {cluster_id}.")

                        # Write thread name and native_id to file
                        dump_thread_to_file(t, thread_name, thread_dump_path)
            else:
                max_concurrents = 1
                self.executor = ThreadPoolExecutor(max_workers=max_concurrents)

        self._transfer_lock = threading.Lock()

        self.ctx = zmq.Context()
        self.zmq_socket_map = {}

        if self.async_pull_kv:
            # dp_rank = vllm_config.parallel_config.data_parallel_rank_local
            thread_name = f"async_pull_kv_{self.dp_rank}"
            self.thread_on_fast_path_req = threading.Thread(target=self.on_fast_path_req, daemon=True, name=thread_name)
            self.thread_on_fast_path_req.start()
            logger.warning(f"DecodeConnectorWorker initialized with self.async_pull_kv enabled.")

            # Write thread name and native_id to file
            dump_thread_to_file(self.thread_on_fast_path_req, thread_name, thread_dump_path)

    def on_fast_path_req(self):
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        sub.connect(f"ipc:///tmp/sched-pub-{self.vllm_config.parallel_config.data_parallel_rank_local}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        while True:
            serialized_data = sub.recv()
            metadata = pickle.loads(serialized_data)
            for req_id, meta in metadata.requests.items():
                if (len(meta.local_block_ids) > 0) and (len(meta.remote_block_ids) > 0):
                    self.start_load_kv(metadata)
                    logger.info(
                        "Received fast path request for request %s with "
                        "local_block_ids: %s, remote_block_ids: %s.",
                        req_id,
                        len(meta.local_block_ids),
                        len(meta.remote_block_ids)
                    )

    def worker(self, cluster_id):
        q = self.queues[cluster_id]
        time.sleep(0)
        while True:
            task = q.get()
            if task is None:
                continue
            try:
                self._read_blocks(**task)
            except Exception as e:
                logger.error("KV transfer task failed in thread %s: %s", cluster_id, e)
                self._send_pulled_kv_req_list(task['remote_host_ip'], [task['request_id']])
                raise RuntimeError(f"Failed to pull kv for request:{task['request_id']} from cluster:{cluster_id}.")
            q.task_done()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.datadist_manager.register_memory(kv_caches)
        if self.multi_rank_pull_kv:
            self.registed_link_infos, _ = self.datadist_manager.register_link()
            logger.info(f" ***** registed_link_infos: {self.registed_link_infos}")
            for remote_cluster_id, cluster_ids_dict in self.registed_link_infos.items():
                cluster_ids = cluster_ids_dict[self.cluster_id + self.datadist_manager.local_rank]
                for idx_count, cluster_id in enumerate(cluster_ids):
                    with self._pull_kv_lock:
                        if cluster_id in self.queues:
                            continue
                        q = queue.Queue()
                        self.queues[cluster_id] = q
                        thread_name = f"thread_pull_kv_dp_rank_{self.dp_rank}_cluster_id_{cluster_id}"
                        t = threading.Thread(target=self.worker, args=(cluster_id,), daemon=True, name=thread_name)
                        t.start()
                        self.threads[cluster_id] = t
                        logger.debug(f" ***** Created a new thread for pulling kv from cluster {cluster_id}.")
                        dump_thread_to_file(t, thread_name, thread_dump_path)
        else:
            self.datadist_manager.register_link()

    # Now go asynchronous pull_kv
    def start_load_kv(self, metadata: DatadistConnectorMetadata):
        logger.info(f" ***** start_load_kv: {len(metadata.requests)}")
        futures = []
        for req_id, meta in metadata.requests.items():
            # if the local_block_ids is empty, skip pulling kv for the request
            if len(meta.local_block_ids) == 0:
                logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                continue
            # If local_block_ids is a flat list of int, omni-attention is not used
            # and we can directly use the local_block_ids and remote_block_ids
            if isinstance(meta.local_block_ids[0], int):
                # local_block_ids (kv blocks in D) is more than remote_block_ids (kv blocks in P), cannot correctly pull kv
                # raise RuntimeError to stop the process
                if len(meta.remote_block_ids) < len(meta.local_block_ids):
                    raise RuntimeError(
                        f"Request {req_id} has fewer remote blocks ({len(meta.remote_block_ids)}) "
                        f"than local blocks ({len(meta.local_block_ids)}).")
                # If remote_block_ids is more than local_block_ids, we only need the last N remote blocks
                # where N is the number of local blocks
                elif len(meta.remote_block_ids) > len(meta.local_block_ids):
                    meta.remote_block_ids = meta.remote_block_ids[-len(meta.local_block_ids):]
                logger.info(
                    " ***** start_load_kv for request %s "
                    "Num local_block_ids: %s. Num remote_block_ids: %s.",
                    req_id,
                    len(meta.local_block_ids),
                    len(meta.remote_block_ids)
                )
            # If local_block_ids is a list of lists (e.g., [[], []]), omni-attention is used
            # local_block_ids[0] is a list of local block ids for uncompressed layers
            # local_block_ids[1] is a list of local block ids for compressed layers
            elif isinstance(meta.local_block_ids[0], list):
                # If local_block_ids[0] is a list of lists, we need to ensure that remote_block_ids
                # is a list of lists as well, where each sublist corresponds to the local_block
                meta.remote_block_ids = [meta.remote_block_ids] * len(meta.local_block_ids)
                # If local_block_ids[0] is empty, skip pulling kv for the request
                if len(meta.local_block_ids[0]) == 0:
                    logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                    continue
                # If remote_block_ids in P is less than local_block_ids[0] in D, raise RuntimeError to stop the process
                elif len(meta.remote_block_ids[0]) < len(meta.local_block_ids[0]):
                    raise RuntimeError(
                        f"Request {req_id} has fewer remote blocks ({len(meta.remote_block_ids[0])}) "
                        f"than local blocks ({len(meta.local_block_ids[0])}).")
                # If remote_block_ids in P is more than local_block_ids[0] in D, we only need the last N remote blocks
                elif len(meta.remote_block_ids[0]) > len(meta.local_block_ids[0]):
                    meta.remote_block_ids[0] = meta.remote_block_ids[0][-len(meta.local_block_ids[0]):]
                logger.info(
                    " ***** start_load_kv for request %s "
                    "Num local_block_ids: %s. Num remote_block_ids: %s.",
                    req_id,
                    len(meta.local_block_ids[0]),
                    len(meta.remote_block_ids[0])
                )
            # handle the unexpected case where local_block_ids is not a list of int or list of lists
            else:
                logger.error(f"Unexpected type for meta.local_block_ids[0]: {type(meta.local_block_ids[0])}")
                raise RuntimeError(f"Unexpected type for meta.local_block_ids[0]: {type(meta.local_block_ids[0])}")
            if self.multi_rank_pull_kv:
                # If multi_rank_pull_kv is enabled, each DP rank will pull kv from multiple P ranks
                # and the cluster_ids are obtained from registed_link_infos
                cluster_ids = self.registed_link_infos[meta.remote_cluster_id][self.cluster_id + self.datadist_manager.local_rank]
                # If the local_block_ids is a flat list of int, we can directly use it
                # As multi_rank_pull_kv is designed to pull kv from two P ranks,
                # we split the local_block_ids and remote_block_ids into two parts
                if not isinstance(meta.local_block_ids[0], list):
                    block_thre = len(meta.local_block_ids) // 2
                # If the local_block_ids is a flat list of list, only split the blocks for uncompressed layers
                else:
                    block_thre = len(meta.local_block_ids[0]) // 2
                for idx_cluster, cluster_id in enumerate(cluster_ids):
                    if not isinstance(meta.local_block_ids[0], list):
                        if idx_cluster == 0:
                            local_blocks = meta.local_block_ids[:block_thre]
                            remote_blocks = meta.remote_block_ids[:block_thre]
                            len_local_blocks = len(local_blocks)
                        else:
                            local_blocks = meta.local_block_ids[block_thre:]
                            remote_blocks = meta.remote_block_ids[block_thre:]
                            len_local_blocks = len(local_blocks)
                    else:
                        if idx_cluster == 0:
                            # For uncompressed layers, split the local_block_ids[0] and remote_block_ids
                            # For compressed layers, only pull kv from the second P rank
                            local_blocks = [meta.local_block_ids[0][:block_thre], []]
                            # remote_blocks need to be split as well for getting kv blocks for compressed layers in P
                            remote_blocks = [meta.remote_block_ids[0][:block_thre], []]
                            len_local_blocks = len(local_blocks[0])
                        else:
                            local_blocks = [meta.local_block_ids[0][block_thre:], meta.local_block_ids[1]]
                            remote_blocks = [meta.remote_block_ids[0][block_thre:], meta.remote_block_ids[1]]
                            len_local_blocks = len(local_blocks[0])
                    if len_local_blocks > 0:
                        task = {
                            'request_id': req_id,
                            'dst_cluster_id': cluster_id,
                            'local_block_ids': local_blocks,
                            'remote_block_ids': remote_blocks,
                            'remote_host_ip': meta.remote_host,
                        }
                        logger.warning(f"*********** dst cluster_id is {cluster_id}.")
                        self.queues[cluster_id].put(task)
            elif self.multi_thread_pull_kv:
                cluster_id = int(meta.remote_cluster_id)
                task = {
                    'request_id': req_id,
                    'dst_cluster_id': meta.remote_cluster_id,
                    'local_block_ids': meta.local_block_ids,
                    'remote_block_ids': meta.remote_block_ids,
                    'remote_host_ip': meta.remote_host,
                }
                
                self.queues[cluster_id].put(task)
            else:
                # Use ThreadPoolExecutor to handle the task
                future = self.executor.submit(
                                self._read_blocks,
                                local_block_ids=meta.local_block_ids,
                                remote_block_ids=meta.remote_block_ids,
                                dst_cluster_id=meta.remote_cluster_id, 
                                request_id=req_id,
                                remote_host_ip=meta.remote_host,
                            )
                futures.append(future)

        if not self.multi_thread_pull_kv:
            for future in futures:
                future.add_done_callback(handle_exception)

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_cluster_id: str,
        request_id: str,
        remote_host_ip: str,
    ):
        start = time.time()
        self.datadist_manager.pull_kv(remote_block_ids, local_block_ids, dst_cluster_id)
        self._send_pulled_kv_req_list(remote_host_ip, [request_id])
        with self._transfer_lock:
            self._recving_transfers.append(request_id)
        cost = time.time() - start
        logger.info(f" ***** read block, req_id:{request_id}, cost:{cost:.6f}")


    def _send_pulled_kv_req_list(self, path, data):
        if path in self.zmq_socket_map:
            socket = self.zmq_socket_map[path]
        else:
            socket = self.ctx.socket(zmq.PUSH)
            socket.connect(path)
            self.zmq_socket_map[path] = socket
            logger.info(f"create new socket path:{path}")

        try:
            json_data = json.dumps(data)
            socket.send_string(json_data)
            logger.info(f"send string {json_data} path:{path}")
        except Exception as e:
            logger.error(f"Failed to send reqest_id {json_data} to prefill: {e}")

    def get_finished(self) -> tuple[set[str], set[str]]:
        # for decode size, done_sending is no need
        all_done_sending: set[str] = set()
        with self._transfer_lock:
            all_done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(all_done_recving) > 0:
            logger.debug(
                "Get_finished: %s requests done recving", len(all_done_recving))

        return all_done_sending, all_done_recving

    def _pop_done_transfers(self, transfers: list) -> set[str]:
        done_req_ids: set[str] = set()
        for req_id in transfers:
            done_req_ids.add(req_id)
        self._recving_transfers.clear()
        return done_req_ids

def handle_exception(future):
    if future.exception():
        logger.error(f"Exception occurred in future: {future.exception()}")
        raise future.exception()

def dump_thread_to_file(thread, thread_name: str, folder_path: str):

    timeout = 5  # seconds
    start_time = time.time()
    while not hasattr(thread, "native_id"):
        if time.time() - start_time > timeout:
            logger.error(f"Timeout waiting for thread {thread_name} to have native_id.")
            return
        time.sleep(0.005)

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create folder {folder_path}: {e}")
            return

    file_path = os.path.join(folder_path, thread_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(thread.native_id))
    except Exception as e:
        logger.error(f"Failed to write thread info to {file_path}: {e}")