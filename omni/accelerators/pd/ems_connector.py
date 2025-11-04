from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.forward_context import ForwardContext
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from omni.accelerators.pd.cc_connector import CcConnector
from omni.accelerators.pd.llmdatadist_connector_v1 import LLMDataDistConnector


@dataclass
class EmsConnectorMetadata(KVConnectorMetadata):
    metadata: tuple[KVConnectorMetadata, ...]


class EmsConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        self._cc_connector = CcConnector(vllm_config, role)
        self._pd_connector = LLMDataDistConnector(vllm_config, role)
        self._connectors: list[KVConnectorBase_V1] = [self._cc_connector, self._pd_connector]
    
    # ===============================
    # Scheduler-side methods
    # ===============================

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        self._pd_connector.get_num_new_matched_tokens(request, num_computed_tokens)
        return self._cc_connector.get_num_new_matched_tokens(request, num_computed_tokens)
    
    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        for connector in self._connectors:
            connector.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        return EmsConnectorMetadata(metadata=tuple(c.build_connector_meta(scheduler_output) for c in self._connectors))

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        # CcConnector的返回值没有实际意义，无需处理
        self._cc_connector.request_finished(request, block_ids)
        return self._pd_connector.request_finished(request, block_ids)
    
    # ===============================
    # Worker-side methods
    # ===============================

    def register_kv_caches(self, kv_caches: dict[str, Tuple[torch.Tensor]]):
        for connector in self._connectors:
            connector.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        for connector, connector_meta in zip(self._connectors, connector_metadata.metadata):
            connector.bind_connector_metadata(connector_meta)
    
    def clear_connector_metadata(self) -> None:
        for connector in self._connectors:
            connector.clear_connector_metadata()
    
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        for connector in self._connectors:
            connector.start_load_kv(forward_context, **kwargs)
    
    def wait_for_layer_load(self, layer_name: str) -> None:
        for connector in self._connectors:
            connector.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata",
                      **kwargs) -> None:
        for connector in self._connectors:
            connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        for connector in self._connectors:
            connector.wait_for_save()
    
    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # prefill 实例中，cc connector只返回loading，saving为None
        saving, loading = self._cc_connector.get_finished(finished_req_ids)
        # prefill 实例中，llmdatadist connector只返回sending，recving为None
        sending, recving = self._pd_connector.get_finished(finished_req_ids)
        return sending, loading