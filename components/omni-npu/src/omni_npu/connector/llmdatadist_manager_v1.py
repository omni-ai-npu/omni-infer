# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import socket
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Union

import llm_datadist
import torch
from llm_datadist import (
    BlocksCacheKey,
    CacheDesc,
    Cache,
    LLMClusterInfo,
    LLMConfig,
    LLMDataDist,
    LLMException,
    LLMRole,
    LLMStatusCode,
)

from vllm.config import VllmConfig
import vllm.envs as envs
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import KVCacheConfig

from .utils import get_p_start_rank

logger = init_logger(__name__)

_ROLE_STR_TO_ENUM = {
    "kv_producer": LLMRole.PROMPT,
    "kv_consumer": LLMRole.DECODER,
}

TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32,
}

SCHEDULER_LINK_BATCH_SIZE = 32
SCHEDULER_LINK_INTERVAL = 0.5
KV_CACHE_RETRY_TIMES = 1
KV_CACHE_RETRY_WAIT_SECOND = 1
SYNC_KV_TIMEOUT = 5000  # ms
LINK_TIMEOUT = 5000  # ms

RETRYABLE_CODES = [
    LLMStatusCode.LLM_REPEAT_REQUEST,
    LLMStatusCode.LLM_CLUSTER_NUM_EXCEED_LIMIT,
    LLMStatusCode.LLM_PROCESSING_LINK,  # Building chain is in progress
    LLMStatusCode.LLM_DEVICE_OUT_OF_MEMORY,
    LLMStatusCode.LLM_TIMEOUT,
    LLMStatusCode.LLM_WAIT_PROCESS_TIMEOUT,
    LLMStatusCode.LLM_LINK_BUSY,
]

NUM_DIE_PER_MACH = int(os.getenv("NUM_DIE_PER_MACH", "16"))


class LLMDataDistConfig:
    """
    Configuration for the separate deployment.
    """
    def __init__(self, vllm_config: VllmConfig, local_host_ip, host_port, ignore_load_rank=False) -> None:
        self.local_host_ip = local_host_ip
        self.host_port = host_port
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.kv_role_tmp = self.kv_transfer_config.kv_role

        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_rank = 0
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_partition = get_pp_partition(vllm_config.model_config.model_arch_config.total_num_hidden_layers, self.pp_size, envs.VLLM_PP_LAYER_PARTITION)
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        if ignore_load_rank:
            self.rank = -1
            self.local_rank = -1
            self.cluster_id = -1
        else:
            self.rank = get_world_group().rank_in_group
            self.local_rank = get_world_group().local_rank
            # Prefill uses local_rank, Decode uses rank for cluster_id
            port_offset = self.local_rank if self.kv_role_tmp == "kv_producer" else self.rank
            self.cluster_id = ip_port_to_int(f"{self.local_host_ip}:{int(self.host_port)+port_offset}", self.tp_size, self.pp_size)

        # will be used in d side to checkout which P rank is selected to build kv link
        self.kv_parallel_size = self.kv_transfer_config.kv_parallel_size
        self.kv_producer_dp_size = self.kv_transfer_config.kv_connector_extra_config.get("kv_producer_dp_size", 1)

        host_ip_list = self._get_worker_ips()
        self.host_ip_list = host_ip_list

        timestamp_ms = round(time.monotonic() * 1_000)
        # host_cluster_id is a list, in order to handle the case that multi-node for one TP group
        ip_integers = [
            ip_port_to_int(f"{ip}:{host_port}", self.tp_size, self.pp_size)
            for ip in host_ip_list
        ]

        # (timestamp_ms, ip1_int, ip2_int, ip3_int, ...)
        self.host_cluster_id = (timestamp_ms, *ip_integers)

    def _ips_from_config(self) -> list:
        worker_ips = self.kv_transfer_config.kv_connector_extra_config.get("p_node_list")
        if self.is_prefill and worker_ips and isinstance(worker_ips, list) and len(worker_ips) > 0:
            return worker_ips
        return []

    def _ips_from_ray(self) -> list:
        worker_ips = []

        if not self.is_prefill:
            return worker_ips

        try:
            import ray
        except ImportError:
            logger.debug("Ray is not installed; skipping Ray cluster discovery.")
            return worker_ips

        try:
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            nodes = ray.nodes()
        except Exception as e:
            logger.warning(f"Failed to connect/list Ray nodes (address='auto'): {e}. Using local_host_ip.")
            return worker_ips

        ips = []
        head_ip = None

        for node in nodes:
            if node.get("Alive"):
                addr = node.get("NodeManagerAddress")
                if addr:
                    ips.append(addr)
                    gcs_addr = node.get("GcsAddress", "")
                    if addr in gcs_addr:
                        head_ip = addr
            else:
                logger.error("Detected dead node in the Ray cluster. Please check machines' health.")

        if not ips:
            return worker_ips

        if head_ip and head_ip in ips:
            ips.remove(head_ip)
            worker_ips = [head_ip] + ips
        else:
            worker_ips = ips
        return worker_ips

    # get all node ips in a TP group
    def _get_worker_ips(self):
        # priority: config -> ray -> default
        worker_ips = (
            self._ips_from_config() or
            self._ips_from_ray() or
            [self.local_host_ip] # default
        )

        # NOTE:
        # In fact, "worker_ips" are only used for "remote_cluster_id"
        # maintained by prefill-scheduler, who shares the same ip
        # with prefill-heartbeat-receiver (worker at rank0).
        # 
        # Thus, we only care about the local_host of prefill-scheduler,
        # where local_host_ip is exactly the ip of heartbeat-receiver.
        # 
        # Ensure local_host_ip is the first element in the list
        if worker_ips and worker_ips[0] != self.local_host_ip:
            if self.local_host_ip in worker_ips:
                worker_ips.remove(self.local_host_ip)
                worker_ips.insert(0, self.local_host_ip)

        return worker_ips

    @cached_property
    def role(self):
        return _ROLE_STR_TO_ENUM[self.kv_transfer_config.kv_role]

    @cached_property
    def is_prefill(self):
        return self.role == LLMRole.PROMPT

@dataclass
class PPKVCaches:
    cache: Cache
    pp_rank: int
    model_id: int

class LLMDataDistManager:
    def __init__(self, vllm_config: VllmConfig, local_host_ip, host_port):
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.data_dist_config = LLMDataDistConfig(vllm_config, local_host_ip, host_port)
        self.rank = self.data_dist_config.rank
        self.local_rank = self.data_dist_config.local_rank
        self.tp_size = self.data_dist_config.tp_size
        self.tp_rank = self.data_dist_config.tp_rank
        self.dp_size = self.data_dist_config.dp_size
        self.dp_rank = self.data_dist_config.dp_rank
        self.prefill_dp_size = self.data_dist_config.kv_producer_dp_size
        if self.data_dist_config.is_prefill:
            self.decode_id = 0
        else:
            self.decode_id = self.dp_rank // NUM_DIE_PER_MACH

        self.data_dist_option = None
        self.data_dist_engine_is_inited = False
        self.data_dist_engine = self._init_llm_data_dist()

        self.registered_kv_caches = []
        self.registered_kv_caches_tensor = []
        self.registered_remote_pp_partition = None
        self.registered_pp_kv_caches = []
        self.rank_link_info_map = {}
        # the look-up table for pull kv, managed by each dp process
        # { key: (host_cluster_id, prefill_dp_rank, d_rank), value:[prompt_cluster_id_list] }
        self.registered_link_infos = {}
        self.registered_link_infos_lock = threading.Lock()

    def get_real_remote_cluster_ids(self, meta, tp_rank=0):
        # remote_cluster_id: (timestamp, ip1, ip2, ...)
        remote_id_key = tuple(meta.remote_cluster_id) if isinstance(meta.remote_cluster_id, list) else meta.remote_cluster_id

        key = (remote_id_key, meta.remote_dp_rank, self.rank)
        with self.registered_link_infos_lock:
            remote_cluster_ids = self.registered_link_infos.get(key, None)

        if remote_cluster_ids is None:
            old_key = None
            with self.registered_link_infos_lock:
                for (reg_key, reg_dp_rank, reg_rank) in list(self.registered_link_infos.keys()):
                    if (reg_dp_rank == meta.remote_dp_rank and reg_rank == self.rank and
                        any(ip in reg_key[1:] for ip in remote_id_key[1:])):
                        old_key = (reg_key, reg_dp_rank, reg_rank) # reg_key: (time_stamp, ip1_int, .., ip2_int)
                        break
            if old_key:
                self.close_link(old_key[0], meta.remote_dp_rank, self.rank, tp_rank)
                logger.warning(f"Deleted old link with {old_key}")
            logger.warning(f"Could not find remote cluster id from {meta.remote_cluster_id=}, {meta.remote_dp_rank=}.")
            logger.warning(f"Try to build new link with {meta.remote_cluster_id=}, {meta.remote_dp_rank=}...")
            # Ensure register_link also receives hashable data
            self.register_link(remote_id_key, meta.remote_dp_rank, self.rank, tp_rank)
            with self.registered_link_infos_lock:
                remote_cluster_ids = self.registered_link_infos.get(key, None)

        return remote_cluster_ids

    def _init_llm_data_dist(self):
        llm_config = LLMConfig()
        llm_config.device_id = self.local_rank
        llm_config.local_comm_res = ""
        # RoCE timeout is SYNC_KV_TIMEOUT ms， prevent pull kv timeout
        llm_config.sync_kv_timeout = SYNC_KV_TIMEOUT
        llm_config.enable_remote_cache_accessible = True

        # do new_datadist_link
        llm_config.local_comm_res = ""
        # if is prefill, need to listen on specific ip and port to accept decode side connection
        if self.data_dist_config.is_prefill:
            host_ip_t = self.data_dist_config.local_host_ip
            host_port_t = int(self.data_dist_config.host_port) + int(self.data_dist_config.local_rank)
            llm_config.listen_ip_info = f"{host_ip_t}:{host_port_t}"

        options = llm_config.generate_options()
        self.data_dist_option = options
        data_dist = LLMDataDist(self.data_dist_config.role, self.data_dist_config.cluster_id)
        data_dist.init(options)
        self.data_dist_engine_is_inited = True
        logger.info(f"init {self.data_dist_config.kv_role_tmp} success, {self.data_dist_config.cluster_id=}")

        return data_dist

    def _finalize_llm_data_dist(self):
        logger.info(f"finalize LLMDataDist, {self.data_dist_config.cluster_id=}")
        self.data_dist_engine.finalize()
        self.data_dist_engine_is_inited = False

    def _reinit_llm_data_dist(self):
        logger.info(f"reinit LLMDataDist, {self.data_dist_config.cluster_id=}")
        self.data_dist_engine.init(self.data_dist_option)
        self.data_dist_engine_is_inited = True

    # dynamically register link only when is needed
    def register_link(self, host_cluster_id, prefill_dp_rank, d_rank, tp_rank=0):
        prompt_cluster_id_list = self._get_cluster_id_list(host_cluster_id[1:], prefill_dp_rank, d_rank, tp_rank)
        clusters = []
        for PROMPT_CLUSTER_ID in prompt_cluster_id_list:
            cluster = LLMClusterInfo()
            host_ip, tp_size, pp_size, tp_rank = self.cluster_id_to_ip_port(PROMPT_CLUSTER_ID)
            remote_host_ip, port = host_ip.split(':')
            cluster.remote_cluster_id = PROMPT_CLUSTER_ID
            cluster.append_local_ip_info(self._get_local_ip(), 0)
            cluster.append_remote_ip_info(remote_host_ip, int(port))
            clusters.append(cluster)
        ret, _ = self.data_dist_engine.link_clusters(clusters, timeout=LINK_TIMEOUT)
        if ret != LLMStatusCode.LLM_SUCCESS:
            raise Exception("link failed")
        # add the cluster_id to the dict
        if not self.data_dist_config.is_prefill:
            with self.registered_link_infos_lock:
                self.registered_link_infos[(host_cluster_id, prefill_dp_rank, d_rank)] = prompt_cluster_id_list
        logger.info(f"rank:{self.rank} linked to : {remote_host_ip}, {prompt_cluster_id_list=}")

    # close the link when it is confirmed to be broken
    def close_link(self, host_cluster_id, prefill_dp_rank, d_rank, tp_rank=0):
        if not self.data_dist_config.is_prefill:
            prompt_cluster_id_list = self._get_cluster_id_list(host_cluster_id[1:], prefill_dp_rank, d_rank, tp_rank)
        else:
            prompt_cluster_id_list = [host_cluster_id]
        clusters = []
        for PROMPT_CLUSTER_ID in prompt_cluster_id_list:
            cluster = LLMClusterInfo()
            host_ip, tp_size, pp_size, tp_rank = self.cluster_id_to_ip_port(PROMPT_CLUSTER_ID)
            remote_host_ip, port = host_ip.split(':')
            cluster.remote_cluster_id = PROMPT_CLUSTER_ID
            cluster.append_local_ip_info(self._get_local_ip(), 0)
            cluster.append_remote_ip_info(remote_host_ip, int(port))
            clusters.append(cluster)
        ret, _ = self.data_dist_engine.unlink_clusters(clusters, timeout=LINK_TIMEOUT, force=True)
        if ret != LLMStatusCode.LLM_SUCCESS:
            raise Exception("unlink failed")
        # remove the cluster_id from the dict
        if not self.data_dist_config.is_prefill:
            with self.registered_link_infos_lock:
                self.registered_link_infos.pop((host_cluster_id, prefill_dp_rank, d_rank), None)
        logger.info(f"rank:{self.rank} unlinked with : {remote_host_ip}, {prompt_cluster_id_list=}")

    def unregister_link(self):
        if self.data_dist_config.is_prefill:
            self._finalize_llm_data_dist()
        else:
            for host_cluster_id, dp_rank, d_rank in list(self.registered_link_infos.keys()):
                logger.info(f"{d_rank=}, unlink {host_cluster_id=}")
                self.close_link(host_cluster_id, dp_rank, d_rank)

    def force_unlink(self, remote_cluster_id) -> None:
        clusters = []
        cluster_info = LLMClusterInfo()
        logger.warning(f"force unlink cluster {remote_cluster_id=}")
        cluster_info.remote_cluster_id = remote_cluster_id
        clusters.append(cluster_info)
        self.data_dist_engine.unlink_clusters(clusters, timeout=LINK_TIMEOUT, force=True)

    def _pull_blocks(self, src_cache_key, dst_cache, src_blocks, dst_blocks):
        """" pull kv from remote cache to local cache, support return error state if pull kv fails """
        for attempt in range(KV_CACHE_RETRY_TIMES):
            try:
                self.data_dist_engine.cache_manager.pull_blocks(
                    src_cache_key, dst_cache, src_blocks, dst_blocks
                )
                return True
            except LLMException as e:
                code = e.status_code
                if code in RETRYABLE_CODES:
                    logger.info(
                        f"kv cache pull blocks failed, need retry"
                        f"(attempt {attempt + 1}/{KV_CACHE_RETRY_TIMES}): {e}"
                    )
                    if attempt < KV_CACHE_RETRY_TIMES - 1:
                        time.sleep(KV_CACHE_RETRY_WAIT_SECOND)
                        continue
                    logger.error(
                        f"kv cache pull blocks failed after {KV_CACHE_RETRY_TIMES} attempts: {e}"
                    )
                    return False
                else:
                    logger.error(f"kv cache pull blocks failed (non-retryable): {e}")
                    return False
            except (TypeError, ValueError) as e:
                logger.error(f"kv cache pull blocks input error: {e}")
                return False
        logger.error("kv cache pull blocks exhausted attempts without success")
        return False

    def _register_remote_pp_partition(self, prefill_pp_partition):
        self.registered_remote_pp_partition = prefill_pp_partition
        for kv_cache in self.registered_pp_kv_caches:
            logger.info(f"unregister {kv_cache=}")
            self.data_dist_engine.cache_manager.unregister_cache(kv_cache.cache_id)
        self.registered_pp_kv_caches = []
        for model_id, sub_kv_caches_without_pp in enumerate(self.registered_kv_caches_tensor):
            if len(sub_kv_caches_without_pp) >= sum(prefill_pp_partition):
                prefill_pp_partition[-1] += len(sub_kv_caches_without_pp) - sum(prefill_pp_partition)
                sub_kv_caches_list = []
                for length in prefill_pp_partition:
                    sub_kv_caches_list.append(sub_kv_caches_without_pp[:length])
                    sub_kv_caches_without_pp = sub_kv_caches_without_pp[length:]
            else:
                sub_kv_caches_list = [sub_kv_caches_without_pp]
            for i, sub_kv_caches in enumerate(sub_kv_caches_list):
                cache_desc = CacheDesc(num_tensors=len(sub_kv_caches), shape=tuple(sub_kv_caches[0].shape), data_type=TORCH_DTYPE_TO_NPU_DTYPE[sub_kv_caches[0].dtype])
                cache_addrs = [int(item.data_ptr()) for item in sub_kv_caches]
                assert not self.data_dist_config.is_prefill
                cache = self.data_dist_engine.cache_manager.register_blocks_cache(cache_desc, cache_addrs, None)
                if len(sub_kv_caches_list) == 1:
                    self.registered_pp_kv_caches.append(PPKVCaches(cache=cache, pp_rank=len(prefill_pp_partition) - 1, model_id=model_id))
                else:
                    self.registered_pp_kv_caches.append(PPKVCaches(cache=cache, pp_rank=i, model_id=model_id))

    def _modify_cluster_id_by_pp(self, cluster_id, pp_rank):
        ip_port, tp_size, pp_size, _ = self.cluster_id_to_ip_port(cluster_id)
        ip, port = ip_port.split(':')
        port = int(port) + tp_size * pp_rank
        ip_port = f"{ip}:{port}"
        return ip_port_to_int(ip_port, tp_size, pp_size, tp_rank=0)

    def pull_kv(self, src_blocks, tgt_blocks, prompt_cluster_id, prefill_dp_rank, prefill_pp_partition):
        """ pull kv from remote cache to local cache, support to refresh link when pull kv fails """
        torch.npu.set_device(f"npu:{self.local_rank}")
        if prefill_pp_partition is not None and self.registered_remote_pp_partition != prefill_pp_partition:
            self._register_remote_pp_partition(prefill_pp_partition)
        has_pp = prefill_pp_partition is not None
        list_to_pull = self.registered_pp_kv_caches if has_pp else self.registered_kv_caches
        for i, kv_cache in enumerate(list_to_pull):
            if has_pp:
                pp_rank = kv_cache.pp_rank
                model_id = kv_cache.model_id
                kv_cache = kv_cache.cache
            else:
                pp_rank = 0
                model_id = i
            prompt_cluster_id_with_pp = self._modify_cluster_id_by_pp(prompt_cluster_id, pp_rank)
            prompt_cache_key = BlocksCacheKey(
                prompt_cluster_id=prompt_cluster_id_with_pp, model_id=model_id)
            ret = self._pull_blocks(prompt_cache_key, kv_cache,
                                    src_blocks, tgt_blocks)
            if not ret:
                logger.warning(f"======= failed pull kv with {prompt_cluster_id=} ========")
                self._refresh_link(prompt_cluster_id, prefill_dp_rank, self.rank)
                logger.warning(f"======= successfully rebuild kv link with {prompt_cluster_id=} ========")
                ret_updated = self._pull_blocks(prompt_cache_key, kv_cache,
                                src_blocks, tgt_blocks)
                if not ret_updated:
                    raise RuntimeError(f"Failed to pull kv even if rebuild the kv link!")

    def _refresh_link(self, prompt_cluster_id, prefill_dp_rank, d_rank):
        """ refresh the kv link: unlink + link """
        logger.warning(f"======= refresh_link with {prompt_cluster_id=} ========")
        result = self._get_host_cluster_id(prompt_cluster_id, prefill_dp_rank, d_rank)
        if result is None:
            logger.warning(
                f"No prompt_cluster_id={prompt_cluster_id}, prefill_dp_rank={prefill_dp_rank}, d_rank={d_rank}. "
                f"Available registered_link_infos keys: {list(self.registered_link_infos.keys())}"
            )
            return
        (host_cluster_id, prefill_dp_rank, d_rank) = result
        if host_cluster_id is not None:
            self.close_link(host_cluster_id, prefill_dp_rank, d_rank)
            logger.warning(f"======= rebuild_link with {prompt_cluster_id=} ========")
            self.register_link(host_cluster_id, prefill_dp_rank, d_rank)
        else:
            logger.error(f"Unregistered host cluster id!!!")

    # search for the host_cluster_id in key using the prompt_cluster_id in value
    def _get_host_cluster_id(self, prompt_cluster_id, prefill_dp_rank, d_rank):
        """ search for the host_cluster_id in key using the prompt_cluster_id in value """
        with self.registered_link_infos_lock:
            prompt_p_metas = [
                key for key, values in self.registered_link_infos.items()
                if (isinstance(values, list) and
                    prompt_cluster_id in values and
                    len(key) >= 3 and
                    key[1] == prefill_dp_rank and
                    key[2] == d_rank)
            ]
        if not prompt_p_metas:
            return None
        else:
            return prompt_p_metas[0]

    def _get_cluster_id_list(self, host_cluster_ids, prefill_dp_rank, d_rank, tp_rank):
        """ compute the cluster id that should be linked with the target dp rank """
        if isinstance(host_cluster_ids, int):
           host_cluster_ids = [host_cluster_ids]
        ip_ports = []
        for host_cluster_id in host_cluster_ids:
            ip_port, prefill_tp_size, prefill_pp_size, _ = self.cluster_id_to_ip_port(host_cluster_id)
            ip_ports.append(ip_port)
        # decode_tp_size = self.data_dist_config.kv_parallel_size
        decode_tp_size = self.tp_size # set decode_tp_size using parallel_config instead of kv_config
        decode_id = 0
        decode_num = int(os.getenv('DECODE_POD_NUM', "1"))

        p_rank_start = get_p_start_rank(prefill_tp_size, 1, decode_tp_size, self.dp_size,
                                        decode_num, decode_id, d_rank)
        p_rank_list = [p_rank_start + idx * prefill_tp_size for idx in range(self.prefill_dp_size * prefill_pp_size)]
        cluster_id_list = []
        for p_rank in p_rank_list:
            ip_port = ip_ports[p_rank // NUM_DIE_PER_MACH]
            ip, port_str = ip_port.split(':')
            port = int(port_str) + (p_rank % NUM_DIE_PER_MACH)
            cluster_id = ip_port_to_int(f"{ip}:{port}", prefill_tp_size, prefill_pp_size)
            cluster_id_list.append(cluster_id)
        return cluster_id_list

    def _get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip

    # reuse the existing code
    def register_memory(self, kv_caches: dict[str, torch.Tensor], kv_cache_config: KVCacheConfig = None):
        if not self.data_dist_engine_is_inited:
            self._reinit_llm_data_dist()

        if len(self.registered_kv_caches) > 0:
            raise ValueError("Attr `registered_kv_caches` must be empty before register kv_caches.")
        if isinstance(kv_caches, dict):
            flatten_kv_caches = unzip_kv_cache_dict(kv_caches)
        else:
            flatten_kv_caches = unzip_kv_cache_list(kv_caches)

        # dense model.
        flatten_kv_caches = maybe_merge_kv_caches(flatten_kv_caches)
        # spec model.
        flatten_kv_caches = maybe_split_kv_caches_for_spec_layers(flatten_kv_caches)

        for model_id, sub_kv_caches in enumerate(flatten_kv_caches):
            cache_desc = CacheDesc(num_tensors=len(sub_kv_caches), shape=tuple(sub_kv_caches[0].shape),
                                data_type=TORCH_DTYPE_TO_NPU_DTYPE[sub_kv_caches[0].dtype])

            cache_addrs = [int(item.data_ptr()) for item in sub_kv_caches]

            if self.data_dist_config.is_prefill:
                cache_key = BlocksCacheKey(self.data_dist_engine.cluster_id, model_id=model_id)
            else:
                cache_key = None

            cache = self.data_dist_engine.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key)
            self.registered_kv_caches.append(cache)
            self.registered_kv_caches_tensor.append(sub_kv_caches)
        logger.debug(f" ***** registered_kv_caches num:{len(self.registered_kv_caches)}")

    def unregister_memory(self):
        if not self.data_dist_config.is_prefill:
            for kv_cache in self.registered_kv_caches:
                logger.info(f"unregister {kv_cache=}")
                self.data_dist_engine.cache_manager.unregister_cache(kv_cache.cache_id)
        self.registered_kv_caches = []
        self.registered_kv_caches_tensor = []

    def cluster_id_to_ip_port(self, cluster_id):
        """Extract ip_port from int64 cluster id (inverse of ip_port_to_int)."""
        if not isinstance(cluster_id, int):
            raise TypeError("cluster_id must be int type")

        # Extract fields (reverse order of packing)
        pp_size = (cluster_id & 0xFF) + 1          # [0, 8) bits
        tp_size = ((cluster_id >> 8) & 0xFF) + 1   # [8, 16) bits
        port = (cluster_id >> 16) & 0xFFFF         # [16, 32) bits
        ip_int = (cluster_id >> 32) & 0xFFFFFFFF   # [32, 64) bits
        
        ip = socket.inet_ntoa(struct.pack('!I', ip_int))

        return f"{ip}:{port}", tp_size, pp_size, 0  # tp_rank always 0


def unzip_kv_cache_dict(kv_caches: dict[str, Union[torch.Tensor, tuple[torch.Tensor, ...]]]) -> list[list[torch.Tensor]]:
    """
    Unzips a key-value cache dictionary into a flattened structure of tensors.

    Parameters:
    kv_caches (dict): A dictionary where
      - keys are layer names (str),
      - values are either:
        - torch.Tensor: single tensor representing the layer's cache, or
        - tuple(torch.Tensor, ...): multiple tensors for the layer's cache.
            - if each Tensor in the tuple is contiguous: common GQA case where Tensors represent key or value
            - if Tensors are non-contiguous: hybrid attention case where each Tensor is a strided view

    Returns:
    A flattened structure (list) of tensors extracted from "kv_caches".

    Raises:
    ValueError: If the dictionary contains inconsistent types or is empty.
    """
    if not kv_caches:
        raise ValueError("kv_caches dictionary cannot be empty")

    addrs = set()
    shape2tensor = defaultdict(list)

    def _add_distinct_tensor_by_shape(tensor: torch.Tensor) -> bool:
        curr_addr = tensor.data_ptr()
        if curr_addr in addrs:
            return False
        addrs.add(curr_addr)
        shape2tensor[tensor.shape].append(tensor)
        return True

    for layer_name, layer_cache in kv_caches.items():
        if isinstance(layer_cache, tuple):
            if layer_cache[0].is_contiguous():
                for idx, item in enumerate(layer_cache):
                    if not isinstance(item, torch.Tensor):
                        raise TypeError(
                            f"Tuple element at index {idx} in layer '{layer_name}' is not a tensor: {type(item)}"
                        )
                    _add_distinct_tensor_by_shape(item)
            else:
                # Hybrid Attention Case: Tuple of non-contiguous strided views.
                # Extract the single underlying contiguous memory pool.
                num_blocks = layer_cache[0].shape[0]
                reference_tensor = layer_cache[0]

                # 1. Access the shared underlying raw memory
                shared_storage = reference_tensor.untyped_storage()

                # 2. Reconstruct a 1D contiguous tensor from the storage
                raw_tensor = torch.empty(
                    0,
                    dtype=torch.int8,
                    device=reference_tensor.device
                )
                raw_tensor.set_(shared_storage)

                # 3. Reshape into the standard (num_blocks, elements_per_block) format
                contiguous_backing_tensor = raw_tensor.view(num_blocks, -1)

                # 4. Add the single backing tensor instead of the individual views
                _add_distinct_tensor_by_shape(contiguous_backing_tensor)
        else:
            if not isinstance(layer_cache, torch.Tensor):
                raise ValueError(
                    f"Element in layer '{layer_name}' is not a tensor: {type(layer_cache)}"
                )
            _add_distinct_tensor_by_shape(layer_cache)

    flatten_kv_caches = list(shape2tensor.values())
    return flatten_kv_caches


# reuse the existing code
def unzip_kv_cache_list(kv_caches: list[torch.Tensor], ):
    first_kv_cache = kv_caches[0]
    if isinstance(first_kv_cache, tuple):
        cache_num = len(first_kv_cache)
    else:
        cache_num = 1

    flatten_kv_caches = [[] for _ in  range(cache_num)]

    for kv_cache in kv_caches:
        if isinstance(kv_cache, tuple):
            for index, sub_cache in enumerate(kv_cache):
                flatten_kv_caches[index].append(sub_cache)
        else:
            flatten_kv_caches[0].append(kv_cache)
    return flatten_kv_caches

# reuse the existing code
def maybe_merge_kv_caches(flatten_kv_caches):
    # only 1 kvcache tensor with shape (2, b, s, n, d)
    if len(flatten_kv_caches) == 1 and len(flatten_kv_caches[0][0].shape) == 5 and flatten_kv_caches[0][0].shape[0] == 2:
        merged_kv_caches = [[]]
        for sub_kv_caches in flatten_kv_caches[0]:
            merged_kv_caches[0].append(sub_kv_caches[0])
            merged_kv_caches[1].append(sub_kv_caches[1])
        return merged_kv_caches
    return flatten_kv_caches

# reuse the existing code
def maybe_split_kv_caches_for_spec_layers(flatten_kv_caches):
    flatten_kv_caches_split = []
    need_split = False
    for caches in flatten_kv_caches:
        shape_dict = {}
        for cache in caches:
            if str(cache.shape) not in shape_dict:
                shape_dict[str(cache.shape)] = []
            shape_dict[str(cache.shape)].append(cache)

        flatten_kv_caches_split.extend(shape_dict.values())
        if len(shape_dict) > 1 or need_split:
            need_split = True

    if not need_split:
        return flatten_kv_caches
    else:
        return flatten_kv_caches_split

def ip_port_to_int(ip_port, tp_size, pp_size, tp_rank=0):
    """ convert ip_port to int64 cluster id

    layout:
    [ ip (32 bits) | port (16 bits) | tp_size (16 bits) ]
    """
    ip, port_str = ip_port.split(':')
    port = int(port_str)
    if not (0 <= port <= 65535):
        raise ValueError(" port must be in 0-65535 ")
    if not (1 <= tp_size <= (1 << 8)):
        raise ValueError(f" tp size ({tp_size}) must be in 1-{(1 << 8)} ")
    if not (1 <= pp_size <= (1 << 8)):
        raise ValueError(f" pp size ({pp_size}) must be in 1-{(1 << 8)} ")
    # convert IP to 4 byte boolean
    ip_bytes = socket.inet_aton(ip)
    # convert 4 byte IP to 32 bit int
    ip_int = struct.unpack('!I', ip_bytes)[0]
    # now we only contain ip, port, tp_size, tp_rank is ignored for simplification
    # result = (ip_int << 48) | (port << 32) | (tp_size << 16) | (tp_rank)
    result = (ip_int << 32) | (port << 16) | ((tp_size - 1) << 8) | (pp_size - 1)
    return result

def get_pp_partition(num_hidden_layers, pp_size, partition):
    if pp_size == 1:
        return None
    if partition is not None:
        try:
            partitions = [int(layer) for layer in partition.split(",")]
        except ValueError as err:
            raise ValueError(
                "Invalid partition string: {}".format(partition)
            ) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
    else:
        layers_per_partition = num_hidden_layers // pp_size
        partitions = [layers_per_partition for _ in range(pp_size)]

        if remaining_layers := num_hidden_layers % pp_size:
            for i in range(2, remaining_layers + 2):
                partitions[-i] += 1
    return partitions
