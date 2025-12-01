# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import time
from collections import defaultdict, namedtuple
from functools import cached_property

import llm_datadist
import torch
from llm_datadist import (BlocksCacheKey, CacheDesc, LLMConfig,
                          LLMDataDist, LLMRole, RegisterMemStatus, LLMException, LLMStatusCode,
                          Placement, LLMClusterInfo, DataType)

from vllm.config import KVTransferConfig
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from omni.accelerators.pd.ranktable.local_info import LocalInfo
from omni.accelerators.pd.ranktable.rank_table import GlobalRankTable, RankTableConfig
from omni.accelerators.pd.utils import get_p_start_rank, prepare_ranktables
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (get_tp_group, get_dp_group, get_world_group,
                    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                        get_tp_group)
import os
import socket
import struct

logger = init_logger(__name__)

_ROLE_STR_TO_ENUM = {
    "kv_producer": LLMRole.PROMPT,
    "kv_consumer": LLMRole.DECODER
}

TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32
}

SCHEDULER_LINK_BATCH_SIZE = 32
SCHEDULER_LINK_INTERVAL = 0.5
KV_CACHE_RETRY_TIMES = 1
KV_CACHE_RETRY_WAIT_SECOND = 1
SYNC_KV_TIMEOUT = 5000 # ms
LINK_TIMEOUT = 5000 # ms

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
        additional_config = vllm_config.additional_config
        if additional_config:
            self.multi_rank_pull_kv = additional_config.get("multi_rank_pull_kv", False)
        else:
            self.multi_rank_pull_kv = False
        self.local_host_ip = local_host_ip
        self.host_port = host_port
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.kv_role_tmp = self.kv_transfer_config.kv_role

        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.tp_rank = 0
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        if ignore_load_rank:
            self.rank = -1
            self.local_rank = -1
            self.cluster_id = -1
        else:
            self.rank = get_world_group().rank_in_group
            self.local_rank = get_world_group().local_rank
            self.cluster_id = ip_port_to_int(f"{self.local_host_ip}:{int(self.host_port)+self.local_rank}", self.tp_size)

        # will be used in d side to checkout which P rank is selected to build kv link
        self.kv_parallel_size = self.kv_transfer_config.kv_parallel_size
        self.kv_producer_dp_size = self.kv_transfer_config.kv_connector_extra_config.get("kv_producer_dp_size", 1)

        host_ip_list = self._get_worker_ips()
        self.host_ip_list = host_ip_list

        timestamp_ms = round(time.monotonic() * 1_000)
        # host_cluster_id is a list, in order to handle the case that multi-node for one TP group
        ip_integers = [
            ip_port_to_int(f"{ip}:{host_port}", self.tp_size)
            for ip in host_ip_list
        ]
        
        # (timestamp_ms, ip1_int, ip2_int, ip3_int, ...)
        self.host_cluster_id = (timestamp_ms, *ip_integers)

    # get all node ips in a TP group
    def _get_worker_ips(self):
        """Return worker IPs. Only query Ray when Ray is actually available/running.
        
        Behavior:
        - If self.is_prefill is False: return [self.local_host_ip].
        - If Ray is not installed: log and return [self.local_host_ip].
        - If Ray is installed but no cluster is reachable: log and return [self.local_host_ip].
        - If a Ray cluster is reachable: return all Alive nodes' NodeManagerAddress,
          with head node (if detected) placed first.
        """
        # default fallback
        worker_ips = [self.local_host_ip]
        
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

    @cached_property
    def role(self):
        return _ROLE_STR_TO_ENUM[self.kv_transfer_config.kv_role]

    @cached_property
    def is_prefill(self):
        return self.role == LLMRole.PROMPT


class LLMDataDistManager:
    def __init__(self, vllm_config: VllmConfig, local_host_ip, host_port):
        additional_config = vllm_config.additional_config
        if additional_config:  # pragma: no cover
            self.multi_rank_pull_kv = additional_config.get("multi_rank_pull_kv", False)
        else:  # pragma: no cover
            self.multi_rank_pull_kv = False
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.data_dist_config = LLMDataDistConfig(vllm_config, local_host_ip, host_port)
        self.rank = self.data_dist_config.rank
        self.local_rank = self.data_dist_config.local_rank
        self.tp_size = self.data_dist_config.tp_size
        self.tp_rank = self.data_dist_config.tp_rank
        self.dp_size = self.data_dist_config.dp_size
        self.dp_rank = self.data_dist_config.dp_rank
        self.prefill_dp_size = self.data_dist_config.kv_producer_dp_size
        if not self.data_dist_config.is_prefill:
            self.decode_id = self.dp_rank // NUM_DIE_PER_MACH

        self.data_dist_engine = self._init_llm_data_dist()

        self.registered_kv_caches = []
        self.rank_link_info_map = {}
        # the look-up table for pull kv, managed by each dp process
        # { key: (host_cluster_id, prefill_dp_rank, d_rank), value:[prompt_cluster_id_list] }
        self.registered_link_infos = {}

    def get_real_remote_cluster_ids(self, meta, tp_rank=0):
        # remote_cluster_id: (timestamp, ip1, ip2, ...)
        remote_id_key = tuple(meta.remote_cluster_id) if isinstance(meta.remote_cluster_id, list) else meta.remote_cluster_id
        
        key = (remote_id_key, meta.remote_dp_rank, self.rank)
        remote_cluster_ids = self.registered_link_infos.get(key, None)
        
        if remote_cluster_ids is None:
            old_key = None
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
            remote_cluster_ids = self.registered_link_infos.get(key, None)
        
        return remote_cluster_ids

    def _init_llm_data_dist(self):
        data_dist = LLMDataDist(self.data_dist_config.role, self.data_dist_config.cluster_id)
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
        data_dist.init(options)
        logger.info(f"init {self.data_dist_config.kv_role_tmp} success, {self.data_dist_config.cluster_id=}")

        return data_dist

    # dynamically register link only when is needed
    def register_link(self, host_cluster_id, prefill_dp_rank, d_rank, tp_rank=0):
        prompt_cluster_id_list = self._get_cluster_id_list(host_cluster_id[1:], prefill_dp_rank, d_rank, tp_rank)
        clusters = []
        for PROMPT_CLUSTER_ID in prompt_cluster_id_list:
            cluster = LLMClusterInfo()
            host_ip, tp_size, tp_rank = cluster_id_to_ip_port(PROMPT_CLUSTER_ID)
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
            host_ip, tp_size, tp_rank = cluster_id_to_ip_port(PROMPT_CLUSTER_ID)
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
            self.registered_link_infos.pop((host_cluster_id, prefill_dp_rank, d_rank), None)
        logger.info(f"rank:{self.rank} unlinked with : {remote_host_ip}, {prompt_cluster_id_list=}")

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

    def pull_kv(self, src_blocks, tgt_blocks, prompt_cluster_id, prefill_dp_rank):
        """ pull kv from remote cache to local cache, support to refresh link when pull kv fails """
        torch.npu.set_device(f"npu:{self.local_rank}")
        for model_id, kv_cache in enumerate(self.registered_kv_caches):
            prompt_cache_key = BlocksCacheKey(
                prompt_cluster_id=prompt_cluster_id, model_id=model_id)
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
        (host_cluster_id, prefill_dp_rank, d_rank) = \
            self._get_host_cluster_id(prompt_cluster_id, prefill_dp_rank, d_rank)
        if host_cluster_id is not None:
            self.close_link(host_cluster_id, prefill_dp_rank, d_rank)
            logger.warning(f"======= rebuild_link with {prompt_cluster_id=} ========")
            self.register_link(host_cluster_id, prefill_dp_rank, d_rank)
        else:
            raise RuntimeError(f"Unregistered host cluster id!!!")

    # search for the host_cluster_id in key using the prompt_cluster_id in value
    def _get_host_cluster_id(self, prompt_cluster_id, prefill_dp_rank, d_rank):
        """ search for the host_cluster_id in key using the prompt_cluster_id in value """
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
            ip_port, prefill_tp_size, _ = cluster_id_to_ip_port(host_cluster_id)
            ip_ports.append(ip_port)
        decode_tp_size = self.data_dist_config.kv_parallel_size
        decode_id = 0
        decode_num = int(os.getenv('DECODE_POD_NUM', "1"))
        
        p_rank_start = get_p_start_rank(prefill_tp_size, 1, decode_tp_size, self.dp_size,
                                        decode_num, decode_id, d_rank)
        p_rank_list = [p_rank_start + dp_idx * prefill_tp_size for dp_idx in range(self.prefill_dp_size)]
        cluster_id_list = []
        for p_rank in p_rank_list:
            ip_port = ip_ports[p_rank // NUM_DIE_PER_MACH]
            ip, port_str = ip_port.split(':')
            port = int(port_str) + (p_rank % NUM_DIE_PER_MACH)
            cluster_id = ip_port_to_int(f"{ip}:{port}", prefill_tp_size)
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
    def register_memory(self, kv_caches: dict[str, torch.Tensor]):
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
        logger.debug(f" ***** registered_kv_caches num:{len(self.registered_kv_caches)}")

# reuse the existing code
def unzip_kv_cache_dict(kv_caches: dict[str, torch.Tensor], ):
    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    _, first_kv_cache = next(iter(kv_caches.items()))
    if isinstance(first_kv_cache, tuple):
        cache_num = len(first_kv_cache)
    else:
        cache_num = 1

    flatten_kv_caches = [[] for _ in  range(cache_num)]

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.
            raise NotImplementedError
        layer_name = layer_names[0]
        kv_cache = kv_caches[layer_name]
        if isinstance(kv_cache, tuple):
            for index, sub_cache in enumerate(kv_cache):
                flatten_kv_caches[index].append(sub_cache)
        else:
            flatten_kv_caches[0].append(kv_cache)
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

def ip_port_to_int(ip_port, tp_size, tp_rank=0):
    """ convert ip_port to int64 cluster id

    layout:
    [ ip (32 bits) | port (16 bits) | tp_size (16 bits) ]
    """
    ip, port_str = ip_port.split(':')
    port = int(port_str)
    if not (0 <= port <= 65535):
        raise ValueError(" port must be in 0-65535 ")
    # convert IP to 4 byte boolean
    ip_bytes = socket.inet_aton(ip)
    # convert 4 byte IP to 32 bit int
    ip_int = struct.unpack('!I', ip_bytes)[0]
    # now we only contain ip, port, tp_size, tp_rank is ignored for simplification
    # result = (ip_int << 48) | (port << 32) | (tp_size << 16) | (tp_rank)
    result = (ip_int << 32) | (port << 16) | (tp_size & 0xFFFF)
    return result



def cluster_id_to_ip_port(cluster_id):
    """Extract ip_port from int64 cluster id (inverse of ip_port_to_int)."""
    if not isinstance(cluster_id, int):
        raise TypeError("cluster_id must be int type")
    
    # Extract fields (reverse order of packing)
    tp_size = cluster_id & 0xFFFF              # Lower 16 bits
    port = (cluster_id >> 16) & 0xFFFF         # Next 16 bits
    ip_int = (cluster_id >> 32) & 0xFFFFFFFF   # Upper 32 bits
    
    ip = socket.inet_ntoa(struct.pack('!I', ip_int))

    return f"{ip}:{port}", tp_size, 0  # tp_rank always 0