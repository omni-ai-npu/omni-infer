from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple, List

import llm_datadist  # type: ignore
import torch
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from omni.adaptors.vllm.ems.ems_adapter import EmsAdapter
from omni.adaptors.vllm.ems.ems_env import EmsEnv


class OperationType(Enum):
    NoOp = 0
    Load = 1
    Save = 2


@dataclass
class CcReqMeta:
    req_id: str
    num_computed_blocks: int
    num_total_blocks: int
    block_hashes: List[int]
    block_ids: Optional[List[int]]
    operation: OperationType


@dataclass
class CcConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: List[CcReqMeta] = []
    
    def add_requests(self, req_meta: CcReqMeta):
        self.requests.append(req_meta)


class CcConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = CcConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = CcConnectorWorker(vllm_config)
    
    # ===============================
    # Scheduler-side methods
    # ===============================

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)
    
    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        return self.connector_scheduler.request_finished(request, block_ids)
    
    # ===============================
    # Worker-side methods
    # ===============================

    def register_kv_caches(self, kv_caches: dict[str, Tuple[torch.Tensor]]):
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return self.connector_worker.get_finished(finished_req_ids)
    
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self.connector_worker.start_load_kv(self._connector_metadata)
    
    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata",
                      **kwargs) -> None:
        pass

    def wait_for_save(self):
        self.connector_worker.wait_for_save(self._connector_metadata)


class CcConnectorScheduler:
    def __init__(self, vllm_config: VllmConfig):
        logger.info(f"[EMS] CcConnectorScheduler init.")
        self.block_size = vllm_config.cache_config.block_size

        self.processed_requests: set[str] = set()
        self.meta_load_reqs: dict[str, CcReqMeta] = {}
    
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        if request.request_id in self.processed_requests:
            logger.debug(f"req {request.request_id} already in processed requests.")
            return 0, False
        self.processed_requests.add(request.request_id)

        num_computed_blocks = num_computed_tokens // self.block_size
        num_total_blocks = (len(request.prompt_token_ids) - 1) // self.block_size

        if not self._need_load(num_computed_blocks, num_total_blocks):
            logger.debug(f"req {request.request_id} no need to load, num_computed_blocks: {num_computed_blocks}, "
                         f"num_total_blocks: {num_total_blocks}.")
            return 0, False
        
        block_hashes = self._cal_block_hashes(request.prompt_token_ids, self.block_size)[
            num_computed_blocks:num_total_blocks]
        req_meta = CcReqMeta(req_id=request.request_id,
                             num_computed_blocks=num_computed_blocks,
                             num_total_blocks=num_total_blocks,
                             block_hashes=block_hashes,
                             block_ids=None,
                             operation=OperationType.NoOp)
        self.meta_load_reqs[request.request_id] = req_meta
        logger.debug(f"req {request.request_id} meta: {req_meta}.")

        return (num_total_blocks - num_computed_blocks) * self.block_size, True
    
    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int):
        if request.request_id not in self.meta_load_reqs:
            logger.debug(f"req {request.request_id} not in meta_load_reqs.")
            return
        
        if num_external_tokens == 0:
            logger.debug(f"req {request.request_id} num_external_tokens is 0, no need to update block ids.")
            self.meta_load_reqs[request.request_id].operation = OperationType.NoOp
            return
        
        req_meta = self.meta_load_reqs[request.request_id]
        num_block_ids = len(blocks.get_block_ids()[0])
        # 如果开启chunked prefill，分配的block数量会小于num_total_blocks
        if req_meta.num_total_blocks > num_block_ids:
            logger.debug(f"req {request.request_id} block ids num ({num_block_ids}) less than "
                         f"total blocks ({req_meta.num_total_blocks}).")
            req_meta.num_total_blocks = num_block_ids
        req_meta.block_ids = blocks.get_block_ids()[0][req_meta.num_computed_blocks:req_meta.num_total_blocks]
        req_meta.operation = OperationType.Load
        logger.debug(f"req {request.request_id} update block ids, meta: {req_meta}.")
    
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        connector_meta = CcConnectorMetadata()

        for req_id, req_meta in self.meta_load_reqs.items():
            if req_meta.operation != OperationType.Load:
                continue

            connector_meta.add_requests(req_meta)
        self.meta_load_reqs.clear()

        for new_req in scheduler_output.scheduled_new_reqs:
            num_total_tokens = scheduler_output.num_scheduled_tokens[new_req.req_id] + new_req.num_computed_tokens
            num_computed_blocks = new_req.num_computed_tokens // self.block_size
            num_total_blocks = num_total_tokens // self.block_size

            if not self._need_save(num_computed_blocks, num_total_blocks):
                logger.debug(f"[EMS] req {new_req.req_id} no need to save, num_computed_blocks: {num_computed_blocks}, "
                             f"num_total_blocks: {num_total_blocks}.")
                continue

            block_hashes = self._cal_block_hashes(new_req.prompt_token_ids, self.block_size)[
                           num_computed_blocks:num_total_blocks]
            block_ids = new_req.block_ids[0][num_computed_blocks:num_total_blocks]
            req_meta = CcReqMeta(req_id=new_req.req_id,
                                 num_computed_blocks=num_computed_blocks,
                                 num_total_blocks=num_total_blocks,
                                 block_hashes=block_hashes,
                                 block_ids=block_ids,
                                 operation=OperationType.Save)
            logger.debug(f"req {new_req.req_id} need save, meta: {req_meta}.")
            connector_meta.add_requests(req_meta)
        
        return connector_meta
    
    def request_finished(self, request: "Request", block_ids: list[int], ) -> tuple[bool, Optional[dict[str, Any]]]:
        if request.request_id in self.processed_requests:
            self.processed_requests.remove(request.request_id)
        
        logger.debug(f"req {request.request_id} finished.")
        return False, None
    
    def _need_load(self, num_computed_blocks: int, num_total_blocks: int) -> bool:
        # 长度小于ems_num_min_reuse_tokens，不load
        if num_total_blocks * self.block_size < EmsEnv.ems_num_min_reuse_tokens:
            return False
        # load block数量小于ems_num_min_load_blocks， 不load
        if num_total_blocks - num_computed_blocks <= EmsEnv.ems_num_min_load_blocks:
            return False
        
        return True
    
    def _need_save(self, num_computed_blocks: int, num_total_blocks: int) -> bool:
        # 长度小于ems_num_min_reuse_tokens, 不save
        if num_total_blocks * self.block_size < EmsEnv.ems_num_min_reuse_tokens:
            return False
        
        if num_total_blocks <= num_computed_blocks:
            return False
        
        return True
    
    def _cal_block_hashes(self, token_ids: List[int], block_size) -> List[int]:
        result: List[int] = []
        prev_block_hash = 0
        num_blocks = len(token_ids) // block_size

        for block_id in range(num_blocks):
            block_hash = self._cal_block_hash(token_ids[block_id * block_size:(block_id + 1) * block_size],
                                              prev_block_hash)
            result.append(block_hash)
            prev_block_hash = block_hash

        return result
    
    def _cal_block_hash(self, block_token_ids: List[int], prev_block_hash: int) -> int:
        return hash((prev_block_hash, *block_token_ids))
    
class CcConnectorWorker:
    SPLITTER = "+@-"

    def __init__(self, vllm_config: VllmConfig):
        self.ems_adapter = EmsAdapter(vllm_config)

    def register_kv_caches(self, kv_caches: dict[str, Tuple[torch.Tensor]]):
        return self.ems_adapter.register_kv_caches(kv_caches)
    
    def start_load_kv(self, metadata: CcConnectorMetadata):
        self.ems_adapter.sync_save_reqs()

        for request in metadata.requests:
            if request.operation != OperationType.Load:
                continue

            self.ems_adapter.async_load(request.req_id, request.block_hashes, request.block_ids, request.num_computed_blocks)
    
    def wait_for_save(self, metadata: CcConnectorMetadata):
        for request in metadata.requests:
            if request.operation != OperationType.Save:
                continue

            self.ems_adapter.async_save(request.req_id, request.block_hashes, request.block_ids)
    
    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        finished_load_reqs = self.ems_adapter.get_finished_load_reqs()

        all_done_recving: set[str] = {f"{req_id}{self.SPLITTER}{num_success_tokens}"
                                      for req_id, num_success_tokens in finished_load_reqs}
        all_done_sending: set[str] = set()

        if all_done_sending or all_done_recving:
            logger.debug(f"[EMS] all_done_sending: {all_done_sending}, all_done_recving: {all_done_recving}.")
        return all_done_sending, all_done_recving