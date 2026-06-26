# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import os
import pickle
import socket
import threading
import time
from collections import defaultdict
import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    get_dcp_group,
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from .llmdatadist_manager_v1 import LLMDataDistConfig, LLMDataDistManager
from .utils import get_config_from_dict_or_env, TP_Convertor

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

GET_META_MSG = b"get_meta_msg"

thread_dump_path = os.environ.get("VLLM_THREAD_DUMP_PATH", "/tmp/vllm_thread_info")
# seconds, use to free blocks when the request is finished for a long time
BLOCK_RELEASE_DELAY = int(os.environ.get("BLOCK_RELEASE_DELAY", 600))
LLMDATADIST_BASE_PORT = int(os.environ.get("VLLM_LLMDATADIST_BASE_PORT", 15567))
HEARTBEAT_INTERVAL = 5
CLUSTER_HEARTBEAT_TIMEOUT = 60
HEARTBEAT_IPC_PATH = "ipc:///tmp/prefill_llmdatadist_connector_ipc"


def _is_grouped_block_ids(block_ids: Any) -> bool:
    return (
        isinstance(block_ids, (list, tuple))
        and len(block_ids) > 0
        and isinstance(block_ids[0], (list, tuple))
    )


@dataclass
class ReqMeta:
    local_block_ids: list[list[int]]
    remote_block_ids: list[list[int]]
    remote_host: str
    remote_cluster_id: str
    remote_pp_partition: Optional[list[int]]
    spec_token_ids: Optional[list[int]]
    remote_dp_rank: Optional[int]
    remote_request_id: Optional[str]
    token_num: int
    num_external_tokens: Optional[int]
    full_local_block_ids: Optional[list[list[int]]]


@dataclass
class ReqMetaPrefill:
    finish_time: float


class DatadistConnectorMetadata(KVConnectorMetadata):
    """Metadata for datadist connector."""

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        token_num: int = None, # for DCP
        num_external_tokens: Optional[int] = None,
        full_local_block_ids: Optional[list[list[int]]] = None,
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_host=kv_transfer_params["remote_host_ip"],
            remote_cluster_id=kv_transfer_params["remote_cluster_id"],
            spec_token_ids=kv_transfer_params["spec_token_ids"],
            remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),
            remote_pp_partition=kv_transfer_params.get("remote_pp_partition", None),
            remote_request_id=kv_transfer_params.get("remote_request_id", None),
            token_num=token_num,
            num_external_tokens=num_external_tokens,
            full_local_block_ids=full_local_block_ids,
        )


class DatadistConnectorMetadataPrefill(KVConnectorMetadata):
    """Metadata for datadist connector."""

    def __init__(self):
        self.requests: dict[str, ReqMetaPrefill] = {}

    def add_new_req(
        self,
        request_id: str,
        finish_time: float,
    ):
        self.requests[request_id] = ReqMetaPrefill(
            finish_time=finish_time
        )


class LLMDataDistConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole,
                 kv_cache_config: Optional["KVCacheConfig"] = None):
        if vllm_config.kv_transfer_config is None:
            raise RuntimeError("vllm_config.kv_transfer_config cannot be None")

        if vllm_config.model_config.is_deepseek_mla:
            vllm_config.kv_transfer_config.kv_parallel_size = 1
            logger.info(f"Set kv_parallel_size to 1 when use deepseek mla model. {role=}")

        local_host_ip = get_local_ip()
        local_host_port = LLMDATADIST_BASE_PORT
        self.datadist_config = LLMDataDistConfig(vllm_config, local_host_ip, local_host_port, ignore_load_rank=True)
        self.host_cluster_id = self.datadist_config.host_cluster_id
        self.host_ip = local_host_ip
        # Introduce the environment variable VLLM_LLMDATADIST_ZMQ_PORT to resolve ZMQ connection conflicts during
        # multi-P deployments on the same machine.
        # This variable should not be set separately unless specifically required for this scenario.
        self.host_port = get_config_from_dict_or_env(vllm_config.kv_transfer_config, "kv_port",
                                                     "VLLM_LLMDATADIST_ZMQ_PORT", "5568", int)
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.host_port += dp_rank
        self.is_prefill = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        self._connector_metadata: Optional[DatadistConnectorMetadata] = DatadistConnectorMetadata()
        self.pp_partition = self.datadist_config.pp_partition
        if role == KVConnectorRole.SCHEDULER:
            if self.is_prefill:
                self.connector_scheduler = PrefillConnectorScheduler(vllm_config, self.host_cluster_id, self.pp_partition, self.host_ip, str(self.host_port))
            else:
                self.connector_scheduler = DecodeConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            if self.is_prefill:
                self.connector_worker = PrefillConnectorWorker(vllm_config, str(self.host_ip), str(self.host_port), kv_cache_config)
            else:
                self.connector_worker = DecodeConnectorWorker(vllm_config, str(self.host_ip), self.host_cluster_id, kv_cache_config)
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

    def get_finished_count(self) -> int | None:
        """
        Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.
        """
        # for prefill instance, only rank 0 manage the request id from decode instance
        if self.is_prefill:
            return 1
        return None

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.register_kv_caches(kv_caches)

    def unregister_kv_caches(self):
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.unregister_kv_caches()

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.get_finished(self._connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        if not isinstance(self._connector_metadata, Union[DatadistConnectorMetadata, DatadistConnectorMetadataPrefill]):
            raise RuntimeError("self._connector_metadata must be an instance of DatadistConnectorMetadata or DatadistConnectorMetadataPrefill")
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

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [list(group) for group in block_ids])

class PrefillConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config, host_cluster_id: str, pp_partition: list[int], host_ip: str, host_port: str):
        self.vllm_config = vllm_config
        self.host_cluster_id = host_cluster_id
        self.pp_partition = pp_partition
        self.host_ip = host_ip
        self.host_port = host_port
        logger.info("Initializing LLMDataDist Scheduler %s %s %s", host_cluster_id, host_ip, host_port)
        # initialize the dict to save requests finish time
        self.requests_finish_time = dict()

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
        metadata = DatadistConnectorMetadataPrefill()
        # add requests finish time to metadata, to pass to worker connector
        metadata.requests = {req_id: ReqMetaPrefill(finish_time=finish_time)
                     for req_id, finish_time in self.requests_finish_time.items()}
        self.requests_finish_time.clear()
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
        # record the finish time of the request
        if delay_free_blocks:
            self.requests_finish_time[request.request_id] = time.monotonic()

        return delay_free_blocks, dict(
            remote_block_ids=block_ids,
            remote_cluster_id=self.host_cluster_id,
            remote_host_ip=f"tcp://{self.host_ip}:{self.host_port}",
            remote_pp_partition=self.pp_partition,
            spec_token_ids=spec_token_ids,
            remote_dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            remote_request_id=request.request_id,
        )


class PrefillConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, host_port: str,
                 kv_cache_config: Optional["KVCacheConfig"] = None):
        # Metadata.
        self.host_ip = host_ip
        self.host_port = host_port
        self.rank = get_tensor_model_parallel_rank()
        self.pp_rank = get_pp_group().rank_in_group
        self.host_port = str(int(self.host_port) + self.pp_rank)
        self.remote_hb_info_lock = threading.Lock()
        self.remote_hb_info = {}
        self.ctx = zmq.Context()
        self.hb_ipc_client_sockets = {}
        manager_cls = LLMDataDistManager
        datadist_host_port = LLMDATADIST_BASE_PORT
        self.datadist_manager = manager_cls(vllm_config, self.host_ip, datadist_host_port)
        self.kv_cache_config = kv_cache_config
        if self.rank == 0 and self.pp_rank == 0:
            self.input_socket = self.ctx.socket(zmq.constants.PULL)
            self.input_socket.bind(f"tcp://{self.host_ip}:{self.host_port}")
            logger.info(f"ConnectWorker bind tcp://{self.host_ip}:{self.host_port}")
            self._transfer_lock = threading.Lock()
            self.receive_req_list = []
            thread_name = "prefill_connector_get_pulled_kv_req_list"
            self.thread = threading.Thread(target=self.get_pulled_kv_req_list, daemon=True, name=thread_name)
            self.thread.start()
            dump_thread_to_file(self.thread, thread_name, thread_dump_path)

            hb_port = int(os.environ.get("VLLM_LLMDATADIST_HEARTBEAT_PORT", int(datadist_host_port) - 1))
            self.hb_socket = self.ctx.socket(zmq.PUB)
            self.hb_socket.bind(f"tcp://{self.host_ip}:{hb_port}")
            logger.info(f"Prefill create heartbeat publisher: tcp://{self.host_ip}:{hb_port}")

            self.thread = threading.Thread(target=self.heartbeat_timer_func, daemon=True, name="prefill_heartbeat_thread")
            self.thread.start()
        else:
            self.thread = threading.Thread(target=self.heartbeat_server_func, daemon=True, name="prefill_heartbeat_server_thread")
            self.thread.start()

        # initialize the dict to save requests finish time
        self.requests_finish_time = dict()

    def _unlink(self, cluster_id: int):
            try:
                self.datadist_manager.force_unlink(cluster_id)
                logger.info(f"force unlink: {cluster_id}")
            except Exception as e:
                logger.error(f"force_unlink failed: {cluster_id} {e}")

    def heartbeat_timer_func(self):
        logger.info(f"start heart beat thread {threading.current_thread().name}")
        while True:
            cur_time = int(time.time())
            # check remote still alive
            tmp_hb_info = []
            with self.remote_hb_info_lock:
                for remote_cluster_id in self.remote_hb_info:
                    if self.remote_hb_info[remote_cluster_id] + CLUSTER_HEARTBEAT_TIMEOUT < cur_time:
                        tmp_hb_info.append(remote_cluster_id)
                for remote_cluster_id in tmp_hb_info:
                    self.remote_hb_info.pop(remote_cluster_id, None)

            for remote_cluster_id in tmp_hb_info:
                # cluster id of other instance needs
                ip_port, tp_size, pp_size, tp_rank = self.datadist_manager.cluster_id_to_ip_port(int(remote_cluster_id))
                logger.warning(f"remote heartbeat timeout: {ip_port=}, {remote_cluster_id=}")
                port = int(ip_port.split(":")[1])
                if port == 0:
                    self._unlink(int(remote_cluster_id))
                else:
                    rank = port
                    remote_ipc = HEARTBEAT_IPC_PATH + f"_{rank}"
                    if remote_ipc not in self.hb_ipc_client_sockets.keys():
                        socket = self.ctx.socket(zmq.PUSH)
                        socket.connect(remote_ipc)
                        self.hb_ipc_client_sockets[remote_ipc] = socket
                        logger.info(f"create ipc socket connect to {remote_ipc}")
                    else:
                        socket = self.hb_ipc_client_sockets[remote_ipc]
                    socket.send_string(f"{remote_cluster_id}")

            # send heartbeat
            logger.debug(f"prefill publish heartbeat {self.datadist_manager.data_dist_config.host_cluster_id}")
            self.hb_socket.send_string(f"prefill_hb:{self.datadist_manager.data_dist_config.host_cluster_id}")
            time.sleep(HEARTBEAT_INTERVAL)

    def heartbeat_server_func(self):
        self.ipc_socket = self.ctx.socket(zmq.PULL)
        ipc_path = HEARTBEAT_IPC_PATH + f"_{self.rank}"
        self.ipc_socket.bind(ipc_path)
        logger.info(f"create ipc socket {ipc_path}")
        while True:
            data = self.ipc_socket.recv_string()
            logger.info(f"hb server receive ipc data: {data}")
            if data and isinstance(data, int):
                self._unlink(int(data))

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.datadist_manager.register_memory(kv_caches, self.kv_cache_config)

    def unregister_kv_caches(self):
        self.datadist_manager.unregister_link()
        self.datadist_manager.unregister_memory()
        with self.remote_hb_info_lock:
            self.remote_hb_info.clear()
        logger.info("PrefillConnectorWorker unregister_kv_caches done, cleared remote_hb_info")

    def start_load_kv(self, metadata: DatadistConnectorMetadataPrefill):
        pass

    def get_finished(self, metadata: DatadistConnectorMetadataPrefill) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving.
        """
        all_done_sending: set[str] = set()
        all_done_recving: set[str] = set()
        if self.rank == 0 and self.pp_rank == 0:
            # Update requests_finish_time with new finish times from metadata
            with self._transfer_lock:
                self.requests_finish_time.update(
                    {req_id: meta.finish_time for req_id, meta in metadata.requests.items()}
                )
                
                current_time = time.monotonic()
                # Identify requests whose finish time exceeds BLOCK_RELEASE_DELAY
                out_date_reqs = []
                for req_id, finish_time in self.requests_finish_time.items():
                    if current_time - finish_time > BLOCK_RELEASE_DELAY:
                        out_date_reqs.append(req_id)
                    else:
                        # Since the dict is ordered by finish_time, we can break early
                        break
                for req_id in out_date_reqs:
                    logger.warning(
                        f"Request {req_id} is out of date, finish time: {self.requests_finish_time[req_id]}. Freeing blocks now."
                    )
                    all_done_sending.add(req_id)
                    del self.requests_finish_time[req_id]

            if len(self.receive_req_list) == 0:
                return all_done_sending, all_done_recving

            with self._transfer_lock:
                for req_idx in range(len(self.receive_req_list) - 1, -1, -1):
                    req_id = self.receive_req_list[req_idx]
                    logger.debug(f"Get_finished: request {req_id}")
                    if req_id in self.requests_finish_time:
                        all_done_sending.add(req_id)
                        # if the request's kv has been received, remove it from requests_finish_time
                        del self.requests_finish_time[req_id]
                        self.receive_req_list.remove(req_id)

        return all_done_sending, all_done_recving

    def get_pulled_kv_req_list(self):
        while True:
            try:
                if self.input_socket.poll(timeout=10) > 0:
                    message = self.input_socket.recv_string()
                    id_list = json.loads(message)  # Parse the received JSON string into a list
                    logger.debug("Received: %s", id_list)
                    if id_list[0].startswith("decode_hb:"):
                        cluster_id_str = id_list[0].split(":", 1)[1]
                        try:
                            cluster_id = int(cluster_id_str)
                            ip_port, tp_size, pp_size, tp_rank = self.datadist_manager.cluster_id_to_ip_port(cluster_id)
                            logger.debug(f"get heartbeat {ip_port=}, {cluster_id_str=}")
                            # update timestamp
                            with self.remote_hb_info_lock:
                                self.remote_hb_info[cluster_id_str] = int(time.time())
                        except ValueError:
                            logger.warning(f"Invalid heartbeat: {cluster_id_str}")
                    else:
                        with self._transfer_lock:
                            self.receive_req_list.extend(id_list)
            except Exception as e:
                logger.error("get pulled kv req list failed: %s", e)


class DecodeConnectorScheduler:
    """Implementation of Scheduler side methods"""
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._reqs_need_recv: dict[str, tuple[Request, list[list[int]], list[list[int]], int]] = {}
        self.processed_request: set[str] = set()
        self.ctx = zmq.Context()
        self.zmq_socket_map = {}

        additional_config = vllm_config.additional_config
        if additional_config:
            self.async_pull_kv = additional_config.get("async_pull_kv", False)
        else:
            self.async_pull_kv = False

        if self.async_pull_kv:
            self.context = zmq.Context()
            self.pub = self.context.socket(zmq.PUB)
            kv_rank = self.vllm_config.kv_transfer_config.kv_rank
            self.pub.bind(f"ipc:///tmp/sched-pub-{kv_rank}-{vllm_config.parallel_config.data_parallel_rank_local}")

    def _send_pulled_kv_req_list(self, path, data):
        if path in self.zmq_socket_map:
            socket_ = self.zmq_socket_map[path]
        else:
            socket_ = self.ctx.socket(zmq.PUSH)
            socket_.connect(path)
            self.zmq_socket_map[path] = socket_
            logger.info(f"create new socket path:{path}")

        try:
            json_data = json.dumps(data)
            socket_.send_string(json_data)
            logger.info(f"send string {json_data} path:{path}")
        except Exception as e:
            logger.error(
                f"Failed to send reqest_id {json_data} to prefill: {e}")

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if request.request_id in self.processed_request:
            return 0, False
        params = request.kv_transfer_params
        if params is None:
            return 0, False
        logger.debug(
            "DatadistConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if num_computed_tokens % self.block_size != 0:
            raise RuntimeError("num_computed_tokens must be divisible by self.block_size")
        count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
        return count, count > 0

    def _round_up(self, x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def get_unhashed_block_ids(self, blocks: KVCacheBlocks) -> list[list[int]]:
        """Get unhashed block ids, preserving KV cache group boundaries."""
        group_block_ids = [
            [block.block_id for block in group if block.block_hash is None]
            for group in blocks.blocks
        ]
        return group_block_ids

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        logger.debug(f"Request id {request.request_id}: blocks length is {len(blocks.blocks)}")
        params = request.kv_transfer_params
        logger.debug(
            "DatadistConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        self.processed_request.add(request.request_id)
        if params is not None:
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_cluster_id", "remote_host_ip")):
                    full_block_ids = list(blocks.get_block_ids(allow_none=False))
                    self._reqs_need_recv[request.request_id] = (
                        request, self.get_unhashed_block_ids(blocks), full_block_ids, num_external_tokens
                    )
                else:
                    logger.warning("Got invalid KVTransferParams: %s.", params)

    def build_connector_metadata(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = DatadistConnectorMetadata()
        for req_id, (req, block_ids, full_block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert _is_grouped_block_ids(block_ids) or len(block_ids) == 0, f"{block_ids}="
            if req.kv_transfer_params is None:
                logger.warning(f"For reuqest {req_id}: kv_transfer_params now is None")
            else:
                metadata.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    token_num=req.num_prompt_tokens,
                    num_external_tokens=num_external_tokens,
                    full_local_block_ids=full_block_ids,
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
        if request.status == RequestStatus.FINISHED_ABORTED and request.kv_transfer_params is not None:
            self._send_pulled_kv_req_list(request.kv_transfer_params.get("remote_host_ip"), [request.request_id])
        return False, None


class DecodeConnectorWorker:
    """Worker implementation for datadist."""

    def __init__(self, vllm_config: "VllmConfig", host_ip: str, host_cluster_id: int,
                 kv_cache_config: Optional["KVCacheConfig"] = None):
        self.vllm_config = vllm_config
        self.host_cluster_id = host_cluster_id
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.tp_rank = get_tensor_model_parallel_rank()
        self.remote_hb_info_lock = threading.Lock()
        self.remote_hb_info = {}
        additional_config = vllm_config.additional_config
        if additional_config:
            self.async_pull_kv = additional_config.get("async_pull_kv", False)
        else:
            self.async_pull_kv = False

        manager_cls = LLMDataDistManager
        self.datadist_manager = manager_cls(vllm_config, host_ip, 0)

        self._recving_transfers: list = []
        self._done_recving_count: defaultdict[str, int] = defaultdict(lambda: 0)

        self._pull_kv_lock = threading.Lock()
        self.queues = {} # cluster_id -> queue.Queue
        self.threads = {} # cluster_id -> threading.Thread

        self._transfer_lock = threading.Lock()
        self.host_ip = host_ip
        self.kv_cache_config = kv_cache_config

        self.ctx = zmq.Context()
        self.zmq_socket_map_lock = threading.Lock()
        self.zmq_socket_map = {}
        self.hb_server_info = {}
        thread_name = f"decode_worker_hb_{self.datadist_manager.data_dist_config.local_rank}"
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_timer_func, daemon=True, name=thread_name)
        self.heartbeat_thread.start()

        if self.async_pull_kv:
            thread_name = f"async_pull_kv_{self.dp_rank}"
            self.thread_on_fast_path_req = threading.Thread(target=self.on_fast_path_req, daemon=True, name=thread_name)
            self.thread_on_fast_path_req.start()
            logger.warning(f"DecodeConnectorWorker initialized with self.async_pull_kv enabled.")

            # Write thread name and native_id to file
            dump_thread_to_file(self.thread_on_fast_path_req, thread_name, thread_dump_path)

        # TODO: multi_thread_pull_kv and multi_rank_pull_kv are not supported yet
        logger.info(" ***** Using single thread to pull kv.")
        max_concurrents = 1
        self.executor = ThreadPoolExecutor(max_workers=max_concurrents)

        logger.debug("Finish register_kv_caches.")

    def heartbeat_timer_func(self):
        logger.info(f"start heartbeat thread {threading.current_thread().name}")
        while True:
            # recv hb and check remote still alive
            tmp_link_infos = []
            tmp_sub_ips = []
            with self.datadist_manager.registered_link_infos_lock:
                for host_cluster_id, dp_rank, d_rank in self.datadist_manager.registered_link_infos:
                    remote_ip_port, tp_size, pp_size, tp_rank = self.datadist_manager.cluster_id_to_ip_port(host_cluster_id[1])
                    hb_sub_ip = remote_ip_port.split(":")[0]
                    hb_sub_port = int(os.environ.get("VLLM_LLMDATADIST_HEARTBEAT_PORT", int(remote_ip_port.split(":")[1]) - 1))
                    hb_ip_port = f"tcp://" + hb_sub_ip + f":{hb_sub_port + dp_rank}"
                    if hb_ip_port in self.hb_server_info.keys():
                        socket, ts = self.hb_server_info[hb_ip_port]
                        try:
                            data = socket.recv_string(flags=zmq.NOBLOCK)
                            if data:
                                # update timestamp
                                self.hb_server_info[hb_ip_port] = (socket, int(time.time()))
                                logger.debug(f"get heartbeat: {hb_ip_port=}, {data=}")
                        except zmq.error.Again:
                            if ts + CLUSTER_HEARTBEAT_TIMEOUT < int(time.time()):
                                self.hb_server_info.pop(hb_ip_port, None)
                                logger.info(f"remote heartbeat timeout: {hb_ip_port=}, {host_cluster_id=}")
                                tmp_link_infos.append((host_cluster_id, dp_rank, d_rank))
                                tmp_sub_ips.append(hb_sub_ip)
                    else:
                        # create new sub socket
                        socket = self.ctx.socket(zmq.SUB)
                        socket.connect(hb_ip_port)
                        socket.setsockopt_string(zmq.SUBSCRIBE, "prefill_hb")
                        self.hb_server_info[hb_ip_port] = (socket, int(time.time()))
                        logger.info(f"subscribe to {hb_ip_port=}")

            for host_cluster_id, dp_rank, d_rank in tmp_link_infos:
                self.datadist_manager.close_link(host_cluster_id, dp_rank, d_rank)

            with self.zmq_socket_map_lock:
                self.zmq_socket_map = {ip_port: value
                    for ip_port, value in self.zmq_socket_map.items()
                    if ip_port.replace("tcp://", "").split(":")[0] not in tmp_sub_ips}

            #send heartbeat to each remote
            for ip_port in self.zmq_socket_map.keys():
                json_data = json.dumps([f"decode_hb:{self.datadist_manager.data_dist_config.cluster_id}"])
                self.zmq_socket_map[ip_port].send_string(json_data)
                logger.debug(f"decode send hb to {ip_port=}")
            time.sleep(HEARTBEAT_INTERVAL)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.datadist_manager.register_memory(kv_caches, self.kv_cache_config)
        # TODO:put multi-thread_pull_kv and multi_rank_pull_kv related registered_link_infos into queues
        # In single thread pull kv mode, we use a single thread to pull kv
        logger.info(" ***** Using single thread to pull kv.")
        max_concurrents = 1
        self.executor = ThreadPoolExecutor(max_workers=max_concurrents)

        logger.debug("Finish register_kv_caches.")

    def unregister_kv_caches(self):
        self.datadist_manager.unregister_link()
        self.datadist_manager.unregister_memory()
        self.hb_server_info.clear()
        logger.info("DecodeConnectorWorker unregister_kv_caches done, cleared hb_server_info")

    # Now go asynchronous pull_kv
    def start_load_kv(self, metadata: DatadistConnectorMetadata):
        logger.debug(f" ***** start_load_kv: {len(metadata.requests)}")
        futures = []
        for req_id, meta in metadata.requests.items():
            # if the local_block_ids is empty, skip pulling kv for the request
            if len(meta.local_block_ids) == 0:
                if self.tp_rank == 0:
                    logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                continue

            manager = self.datadist_manager
            cluster_ids = manager.get_real_remote_cluster_ids(meta)

            tp_cvt = None
            if get_dcp_group().world_size > 1:
                tp_cvt = TP_Convertor(
                    remote_tp_size=manager.cluster_id_to_ip_port(cluster_ids[0])[1]
                )
                assert meta.token_num is not None
                meta.remote_block_ids = tp_cvt.scheme_reorg(
                    token_num=meta.token_num,
                    tail_blk=meta.local_block_ids[-1],
                    kv_group=manager.registered_kv_caches_tensor,
                    local_block_ids=meta.local_block_ids,
                    remote_block_ids=meta.remote_block_ids,
                )

            if isinstance(meta.local_block_ids[0], int):
                # legacy list[int] block ids is deprecated
                raise RuntimeError(
                    "Flattened block ids for KV transfer is deprecated. Block ids must be "
                    "doubly nested lists of ints, i.e., list[list[int]]."
                    f"{meta.local_block_ids=}, {meta.remote_block_ids=}."
                )

            if not _is_grouped_block_ids(meta.local_block_ids):
                raise RuntimeError(
                    "Grouped local block ids is required "
                    f"for request {req_id}, got {meta.local_block_ids}."
                )

            if not _is_grouped_block_ids(meta.remote_block_ids):
                raise RuntimeError(
                    "Grouped local block ids require grouped remote block ids "
                    f"for request {req_id}, got {meta.remote_block_ids}."
                )

            if len(meta.remote_block_ids) != len(meta.local_block_ids):
                raise RuntimeError(
                    "Grouped local/remote block ids must have the same number "
                    f"of groups for request {req_id}: "
                    f"{len(meta.local_block_ids)} vs {len(meta.remote_block_ids)}."
            )

            aligned_local_block_ids = []
            aligned_remote_block_ids = []
            for group_id, (local_group, remote_group) in enumerate(zip(meta.local_block_ids, meta.remote_block_ids)):
                if len(local_group) == 0:
                    aligned_local_group = []
                    aligned_remote_group = []
                elif len(remote_group) < len(local_group):
                    # NOTE(runze): corner case, reconsider it later
                    aligned_local_group = local_group[:len(remote_group)]
                    aligned_remote_group = remote_group
                    logger.error(f"More local blocks than remote blocks: {local_group=}, {remote_group=}")
                elif len(remote_group) > len(local_group):
                    aligned_local_group = local_group
                    full_local_group = meta.full_local_block_ids[group_id]
                    if len(remote_group) == len(full_local_group) + 1:
                        remote_group = remote_group[:-1]  # trim the lookahead block
                    elif len(remote_group) < len(full_local_group) or len(remote_group) > len(full_local_group) + 1:
                        raise RuntimeError("Length of remote should either match full local or exceed by 1, but got "
                                           f"{len(remote_group)=}, {len(full_local_group)=}, {len(local_group)=}")
                    aligned_remote_group = remote_group[-len(local_group):]
                else:
                    aligned_local_group = local_group
                    aligned_remote_group = remote_group
                aligned_local_block_ids.append(aligned_local_group)
                aligned_remote_block_ids.append(aligned_remote_group)

            if all(len(group) == 0 for group in aligned_local_block_ids):
                if self.tp_rank == 0:
                    logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                continue

            meta.local_block_ids = list(itertools.chain.from_iterable(aligned_local_block_ids))
            meta.remote_block_ids = list(itertools.chain.from_iterable(aligned_remote_block_ids))
            if self.tp_rank == 0:
                logger.info(
                    " ***** start_load_kv for request %s "
                    "Num local_block_ids: %s. Num remote_block_ids: %s.",
                    req_id,
                    len(meta.local_block_ids),
                    len(meta.remote_block_ids)
                )

            # TODO: multi_thread_pull_kv and multi_rank_pull_kv are not supported yet
            # Use ThreadPoolExecutor to handle the task
            future = self.executor.submit(
                self._read_blocks,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                dst_cluster_id=cluster_ids[0],
                request_id=req_id,
                remote_request_id=meta.remote_request_id,
                remote_host_ip=meta.remote_host,
                remote_pp_partition=meta.remote_pp_partition,
                prefill_dp_rank=meta.remote_dp_rank,
                tp_cvt=tp_cvt,
            )
            futures.append(future)

        for future in futures:
            future.add_done_callback(handle_exception)

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_cluster_id: str,
        request_id: str,
        remote_request_id: str,
        remote_host_ip: str,
        remote_pp_partition: list[int],
        prefill_dp_rank: int,
        tp_cvt: TP_Convertor = None,
    ):
        start = time.time()
        self.datadist_manager.pull_kv(remote_block_ids, local_block_ids, dst_cluster_id, prefill_dp_rank, remote_pp_partition)

        if tp_cvt is not None:
            tp_cvt.transfer_done = True

        if self.vllm_config.parallel_config.tensor_parallel_size == 1:
            # tp=1, send to prefill tp rank0 directly.
            self._send_pulled_kv_req_list(remote_host_ip, [remote_request_id])
            with self._transfer_lock:
                self._recving_transfers.append(request_id)
        else:
            # TODO: barrier() is dangerous for multi-thread
            torch.distributed.barrier(group=get_tp_group().cpu_group)
            if get_tensor_model_parallel_rank() == 0:
                self._send_pulled_kv_req_list(remote_host_ip, [remote_request_id])
            with self._transfer_lock:
                self._recving_transfers.append(request_id)
        logger.debug(f" ***** read block, req_id:{request_id}, local_block_ids:{local_block_ids}, remote_block_ids:{remote_block_ids}")
        cost = time.time() - start
        if self.tp_rank == 0:
            logger.info(f" ***** read block, req_id:{request_id}, cost:{cost:.6f}")


    def _send_pulled_kv_req_list(self, path, data):
        with self.zmq_socket_map_lock:
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

    def get_finished(self, metadata: DatadistConnectorMetadata) -> tuple[set[str], set[str]]:
        # for decode size, done_sending is no need
        all_done_sending: set[str] = set()
        with self._transfer_lock:
            all_done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(all_done_recving) > 0:
            logger.debug(
                "Get_finished: %s requests done recving", len(all_done_recving))

        return all_done_sending, all_done_recving

    @staticmethod
    def _pop_done_transfers(transfers: list) -> set[str]:
        done_req_ids: set[str] = set()
        for req_id in transfers:
            done_req_ids.add(req_id)
        transfers.clear()
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

    # ensure the folder exists
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

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
