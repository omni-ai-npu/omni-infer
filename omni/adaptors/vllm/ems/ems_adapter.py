import itertools
import threading
import time
import queue
from typing import List, Tuple

import torch
import numpy as np
from ems import Ems, CcKvOption, EmsException, EmsConfig, EmsErrorCode, CcConfig_v1, ContextCache_v1
from ems.cc_v1.cc_config import KVCacheType
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, get_pp_group, get_dp_group, in_the_same_node_as
from vllm.logger import logger

from omni.adaptors.vllm.ems.ems_env import EmsEnv

from .zmq_comm import ZmqComm, CommMethod, SocketType

class EmsAdapter:
    _LOG_INTERVAL = 10
    
    def __init__(self, vllm_config: VllmConfig):
        self.block_size = vllm_config.cache_config.block_size
        self.is_mla = self._is_mla(vllm_config)
        logger.info(f"[EMS] init ems adapter, is_mla: {self.is_mla}, vllm config: {vllm_config}")
        self.context_caching = self._init_context_caching()
        self.status_checker = EmsStatusChecker()
        self.load_futures = {}
        self.save_futures = {}

        self.last_log_time = time.perf_counter()
        self.num_total_blocks = 0
        self.num_hit_blocks = 0

        self.rank = vllm_config.parallel_config.rank
        self.world_size = vllm_config.parallel_config.world_size
        self.local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.comm_method = vllm_config.kv_transfer_config.kv_connector_extra_config.get("comm_method", "tcp")
        self.zmq_ip = vllm_config.kv_transfer_config.kv_connector_extra_config.get("ems_zmq_ip", "127.0.0.1")
        self.zmq_port = vllm_config.kv_transfer_config.kv_connector_extra_config.get("ems_zmq_port", "50500")

        self.setup_zmq_comm()
    
    def setup_zmq_comm(self) -> None:
        self.used_port = str(int(self.zmq_port) + self.local_dp_rank)
        self.socket_type = SocketType.Receiver if self.rank == 0 else SocketType.Sender

        if self.comm_method == "tcp":
            self.comm_method = CommMethod.TCP
            logger.info(f"(dp rank {self.local_dp_rank}) rank {self.rank} setup ZMQ communication with "
                        f"(comm_method=\"{self.comm_method.name}\", socket_type=\"{self.socket_type.name}\", "
                        f"zmq_ip=\"{self.zmq_ip}\", zmq_port={self.used_port}).")
        elif self.comm_method == "ipc":
            self.comm_method = CommMethod.IPC
            logger.info(f"(dp rank {self.local_dp_rank}) rank {self.rank} setup ZMQ communication with "
                        f"(comm_method=\"{self.comm_method.name}\", socket_type=\"{self.socket_type.name}\"")
        else:
            self.comm_method = CommMethod.TCP
            logger.error(f"Undefined communication method: \"{self.comm_method}\", "
                         f"use TCP instead.")
            logger.info(f"(dp rank {self.local_dp_rank}) rank {self.rank} setup ZMQ communication with "
                        f"(comm_method=\"{self.comm_method.name}\", socket_type=\"{self.socket_type.name}\", "
                        f"zmq_ip=\"{self.zmq_ip}\", zmq_port={self.used_port}).")
        
        self.zmq_comm = ZmqComm(
            ip=self.zmq_ip,
            port=self.used_port,
            dp_rank=self.local_dp_rank,
            comm_method=self.comm_method,
            socket_type=self.socket_type,
        )
        if self.rank == 0:
            self.lock = threading.Lock()  # rank0用于控制流程的线程锁
            self.req_info = dict()  # rank0用于存储请求数据聚合结果的字典
            self.gather_reduce_thread = threading.Thread(target=self.gather_reduce_func, name="ems_worker_comm")
            self.stop_gr_thread = False
            self.gather_reduce_thread.start()

    def gather_reduce_func(self) -> None:
        logger.info(f"(dp rank {self.local_dp_rank}) rank {self.rank} is ready for gather&reduce.")
        while not self.stop_gr_thread:
            try:
                res = self.zmq_comm.recv()
            except Exception as e:
                logger.error(f"(dp rank {self.local_dp_rank}) [ZMQ] (rank {self.rank})"
                             f"socket.recv_pyobj() failed, error: {e}.")
            else:
                if res is not None:
                    self.update_req_info(res)

    def update_req_info(self, data) -> None:
        # data: {
        #     "from_rank": int,
        #     "timestamp": float,
        #     "item": List[Tuple[str, int]],
        # }
        # 主线程和子线程可能同时对self.req_info进行操作，需要上线程锁
        with self.lock:
            for req_id, value in data["item"]:
                if req_id in self.req_info:
                    self.req_info[req_id]["ranks"].add(
                        data["from_rank"]
                    )
                    self.req_info[req_id]["earliest"] = min(
                        data["timestamp"], self.req_info[req_id]["earliest"]
                    )
                    self.req_info[req_id]["value"] = min(
                        value, self.req_info[req_id]["value"]
                    )
                else:
                    self.req_info[req_id] = {
                        "ranks": {data["from_rank"]},  # 接收到消息的rank id集合
                        "earliest": data["timestamp"],  # （已知的）最早发送时间
                        "value": value,  # req_id对应的操作成功数
                    }
    
    # 保留ZMQ通信关闭方法
    def close_zmq_comm(self) -> None:
        self.stop_gr_thread = True
        time.sleep(1)
        self.zmq_comm.terminate()
    
    def register_kv_caches(self, kv_caches: dict[str, Tuple[torch.Tensor]]) -> None:
        kv_caches_list: List[List[torch.Tensor]] = []

        for _, kv_caches_layer in kv_caches.items():
            num_kv = len(kv_caches_layer)
            kv_tensor_list: List[torch.Tensor] = [kv_caches_layer[idx] for idx in range(num_kv)]
            kv_caches_list.append(kv_tensor_list)
        
        try:
            self.context_caching.register_kvcache(kv_caches_list)
            logger.info(f"[EMS] ems adapter register kv caches success.")
        except EmsException as e:
            logger.error(f"[EMS] ems adapter register kv caches failed, error: {e}.")
            # 初始化和注册失败，直接抛异常，由上层处理
            raise

    def _cal_block_offsets(self, block_ids: List[int], block_size: int) -> List[int]:
        return [block_size] * len(block_ids)
    
    def _cal_block_slot_mapping(self, block_ids: List[int], block_size: int) -> List[int]:
        block_ids_np = np.array(block_ids)
        range_arr = np.arange(block_size)
        result_matrix = block_ids_np[:, np.newaxis] * block_size + range_arr
        flattened_result = result_matrix.ravel()
        return flattened_result.tolist()
    
    def async_load(self, req_id: str, block_hashes: List[int], block_ids: List[int], num_computed_blocks: int) -> None:
        future = None
        submit_time = time.perf_counter()

        if not self._check_params(req_id, block_hashes, block_ids):
            self.load_futures[req_id] = (future, submit_time, num_computed_blocks)
            return
        
        option = CcKvOption(write_rcache=EmsEnv.ems_enable_write_rcache, read_local_only=EmsEnv.ems_enable_read_local_only, timeout=EmsEnv.ems_timeout)
        offsets = self._cal_block_offsets(block_ids, self.block_size)
        slot_mapping = self._cal_block_slot_mapping(block_ids, self.block_size)
        try:
            future = self.context_caching.async_load(slot_mapping=slot_mapping, hashes=block_hashes, offsets=offsets,
                                                     option=option)
        except EmsException as e:
            self._process_exception(e)
            logger.error(f"[EMS] req {req_id} async load failed, error: {e}.")

        self.load_futures[req_id] = (future, submit_time, num_computed_blocks)
    
    def async_save(self, req_id: str, block_hashes: List[int], block_ids: List[int]) -> None:
        future = None
        submit_time = time.perf_counter()

        if not self._check_params(req_id, block_hashes, block_ids):
            self.save_futures[req_id] = (future, submit_time)
            return
        
        option = CcKvOption(write_rcache=EmsEnv.ems_enable_write_rcache, read_local_only=EmsEnv.ems_enable_read_local_only, timeout=EmsEnv.ems_timeout)
        offsets = self._cal_block_offsets(block_ids, self.block_size)
        slot_mapping = self._cal_block_slot_mapping(block_ids, self.block_size)
        try:
            future = self.context_caching.async_save(slot_mapping=slot_mapping, hashes=block_hashes, offsets=offsets,
                                                     option=option)
        except EmsException as e:
            self._process_exception(e)
            logger.error(f"[EMS] req {req_id} async save failed, error: {e}.")

        self.save_futures[req_id] = (future, submit_time)
    
    def get_finished_load_reqs(self) -> List[Tuple[str, int]]:
        finished_reqs = []

        for req_id, (future, submit_time, num_computed_blocks) in list(self.load_futures.items()):
            if future and not self.context_caching.is_ready(future):
                continue

            num_success_tokens = 0
            try:
                if future:
                    result = self.context_caching.get_result(future)
                    logger.info(f"[EMS] req {req_id} async load result is {result}, "
                                f"cost time: {round(time.perf_counter() - submit_time, 6)} s.")
                    num_success_tokens = result.success * self.block_size
                    self._update_hit_info(result.success, result.total)
            except EmsException as e:
                self._process_exception(e)
                logger.error(f"[EMS] req {req_id} async load result failed, error: {e}.")
            finally:
                # 考虑apc开启场景，计算总的computed tokens返回作为最终计算复用的token数量
                num_total_computed_tokens = num_success_tokens + num_computed_blocks * self.block_size
                finished_reqs.append((req_id, num_total_computed_tokens))
                del self.load_futures[req_id]
        
        self._print_hit_info()
        updated_reqs = self.update_finished_reqs(finished_reqs)
        return updated_reqs
    

    def update_finished_reqs(self, finished_reqs) -> List[Tuple[str, int]]:
        data = {
            "from_rank": self.rank,  # 本地rank id
            "timestamp": time.perf_counter(),  # 数据发送时间
            "item": finished_reqs,  # 本地finished_reqs结果
        }

        if self.rank == 0:
            self.update_req_info(data)

            global_finished_reqs = []  # 用于记录全局结果
            # 主线程和子线程可能同时对self.req_info进行操作，需要上线程锁
            with self.lock:
                to_pop = []  # 用于记录本次返回后需要移除的req_id
                for req_id in self.req_info:
                    if len(self.req_info[req_id]["ranks"]) == self.world_size:
                        to_pop.append(req_id)
                        global_finished_reqs.append((req_id, self.req_info[req_id]["value"]))
                # 已经返回的，从req_info中移除其req_id
                for req_id in to_pop:
                    self.req_info.pop(req_id)
            return global_finished_reqs
        else:
            try:
                self.zmq_comm.send(data)
            except Exception as e:
                logger.error(f"(dp rank {self.local_dp_rank}) [ZMQ] (rank {self.rank}) "
                             f"socket.send_pyobj() failed, error: {e}.")
            return finished_reqs
    
    def get_finished_save_reqs(self) -> List[Tuple[str, int]]:
        finished_reqs = []

        for req_id, (future, submit_time) in list(self.save_futures.items()):
            if future and not self.context_caching.is_ready(future):
                continue

            num_success_tokens = 0
            try:
                if future:
                    result = self.context_caching.get_result(future)
                    logger.info(f"[EMS] req {req_id} async save result is {result}, "
                                f"cost time: {round(time.perf_counter() - submit_time, 6)} s.")
                    num_success_tokens = result.success * self.block_size
            except EmsException as e:
                self._process_exception(e)
                logger.error(f"[EMS] req {req_id} async save result failed, error: {e}.")
            finally:
                finished_reqs.append((req_id, num_success_tokens))
                del self.save_futures[req_id]
        
        return finished_reqs
    
    def sync_save_reqs(self) -> None:
        for req_id, (future, submit_time) in self.save_futures.items():
            if future is None:
                continue

            try:
                result = self.context_caching.get_result(future)
                logger.info(f"[EMS] req {req_id} async save result is {result}, "
                            f"cost time: {round(time.perf_counter() - submit_time, 6)} s.")
            except EmsException as e:
                self._process_exception(e)
                logger.error(f"[EMS] req {req_id} async save result failed, error: {e}.")
            
        self.save_futures.clear()
    
    def _init_context_caching(self) -> "ContextCache_v1":
        tp_group = get_tp_group()
        pp_group = get_pp_group()
        cc_config_v1 = CcConfig_v1(rank_id=tp_group.rank,
                                   device_id=tp_group.local_rank,
                                   model_id=EmsEnv.model_id,
                                   tp_world_size=tp_group.world_size,
                                   pp_world_size=pp_group.world_size,
                                   rank_in_tp_group=tp_group.rank_in_group,
                                   rank_in_pp_group=pp_group.rank_in_group,
                                   llm_engine=EmsEnv.llm_engine)
        if self.is_mla:
            cc_config_v1.kvcache_type = KVCacheType.MLA
        ems_config = EmsConfig(access_id=EmsEnv.access_id, access_key=EmsEnv.access_key, cc_config_v1=cc_config_v1)

        context_caching = None
        try:
            Ems.init(ems_config)
            context_caching = Ems.get_cc()
        except EmsException as e:
            logger.error(f"[EMS] init ems failed, error: {e}.")
            # 初始化和注册失败，直接抛异常，由上层处理
            raise

        return context_caching
    
    def _check_params(self, req_id: str, block_hashes: List[int], block_ids: List[int]) -> bool:
        if not self.context_caching:
            logger.error("[EMS] context caching is not initialized.")
            return False
        
        if not self.status_checker.get_status():
            logger.error("[EMS] context caching status is unhealthy.")
            return False
        
        if len(block_hashes) == 0 or len(block_hashes) != len(block_ids):
            logger.error(f"[EMS] req {req_id} block hashes or block ids is invalid, "
                         f"block hashes len is {len(block_hashes)}, block ids len is {len(block_ids)}.")
            return False
        
        return True
    
    def _is_mla(self, vllm_config: VllmConfig):
        return vllm_config.model_config.use_mla
    
    def _update_hit_info(self, num_hit_blocks: int, num_total_blocks: int) -> None:
        if self.rank != 0:
            return
        
        self.num_hit_blocks += num_hit_blocks
        self.num_total_blocks += num_total_blocks
    
    def _print_hit_info(self) -> None:
        if self.rank != 0:
            return
        
        cur_time = time.perf_counter()
        if cur_time - self.last_log_time < self._LOG_INTERVAL:
            return

        self.last_log_time = cur_time
        hit_rate = self.num_hit_blocks / self.num_total_blocks * 100 if self.num_total_blocks != 0 else 0.0
        logger.info(f"[EMS] ems cache hit rate: {hit_rate:.1f}%, num_hit_blocks: {self.num_hit_blocks}, "
                    f"num_total_blocks: {self.num_total_blocks}.")
        
    def _process_exception(self, e):
        if e.status_code() == EmsErrorCode.EMS_INVALID_ARGUMENT:
            return
        self.status_checker.set_status(False)
    
class EmsStatusChecker:
    SLEEP_SECONDS = 30

    def __init__(self):
        self._ems_ok = True
        self._start_cc_health_check()
    
    def get_status(self):
        return self._ems_ok
    
    def set_status(self, status: bool):
        self._ems_ok = status
    
    def _check_and_update(self):
        while True:
            time.sleep(self.SLEEP_SECONDS)
            try:
                self._ems_ok = Ems.check_health()
                if self._ems_ok:
                    logger.debug("EMS health status is ok.")
                else:
                    logger.info("EMS health status is abnormal.")
            except Exception as e:
                self._ems_ok = False
                logger.exception(f"EMS health status exception, {e}")
    
    def _start_cc_health_check(self):
        """启动一个新线程来执行定时任务"""
        health_check_thread = threading.Thread(target=self._check_and_update, name="ems-health")
        health_check_thread.daemon = True
        health_check_thread.start()
        logger.info("Start EMS health check.")
