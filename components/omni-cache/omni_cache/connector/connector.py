# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Main OmniCache connector implementation."""

import itertools
import socket
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole, SupportsHMA,
)
from vllm.logger import init_logger

from omni_cache.connector.utils.helpers import resolve_prefill_endpoint, get_config_from_dict_or_env
from omni_cache.connector.utils.metadata import (
    DatadistConnectorMetadata, DatadistConnectorMetadataPrefill,
)
from omni_cache.connector.utils.settings import (
    CLUSTER_SIZE, NODE_IP_SPECS, P_NODE_LIST,
)
from omni_cache.connector.scheduler import PrefillConnectorScheduler, DecodeConnectorScheduler

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext

logger = init_logger(__name__)


class OmniCacheConnector(KVConnectorBase_V1, SupportsHMA):
    """OmniCache connector for KV transfer between prefill and decode."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None
    ):
        """Initialize the OmniCache connector.

        Args:
            vllm_config: vLLM configuration.
            role: Connector role (SCHEDULER or WORKER).
            kv_cache_config: KV cache configuration.
        """
        if vllm_config.kv_transfer_config is None:
            raise RuntimeError("vllm_config.kv_transfer_config cannot be None")

        if vllm_config.model_config.is_deepseek_mla:
            vllm_config.kv_transfer_config.kv_parallel_size = 1
            logger.info("Set kv_parallel_size to 1 when using deepseek MLA model.")

        self.is_prefill = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        self._connector_metadata = None

        if self.is_prefill:
            self._init_prefill_config(vllm_config)
        else:
            self._init_decode_config()

        self._init_role(role, vllm_config)

    def _init_prefill_config(self, vllm_config: VllmConfig) -> None:
        """Initialize prefill-specific configuration."""
        endpoint = resolve_prefill_endpoint(
            vllm_config=vllm_config,
            node_ip_specs=NODE_IP_SPECS,
            cluster_size=CLUSTER_SIZE,
            p_node_list=P_NODE_LIST,
            get_local_ip=self._get_local_ip,
            get_config_value=get_config_from_dict_or_env,
        )
        self.cluster_id_start = endpoint.cluster_id_start
        self.host_ip = endpoint.host_ip
        self.host_port = endpoint.host_port
        self._host_port_str = str(endpoint.host_port)

    def _init_decode_config(self) -> None:
        """Initialize decode-specific configuration."""
        self.host_ip = "127.0.0.1"
        self.cluster_id_start = 0

    def _init_role(self, role: KVConnectorRole, vllm_config: VllmConfig) -> None:
        """Initialize role-specific components."""
        self.connector_scheduler = None
        self.connector_worker = None

        if role == KVConnectorRole.SCHEDULER:
            if self.is_prefill:
                self.connector_scheduler = PrefillConnectorScheduler(
                    vllm_config, self.cluster_id_start, self.host_ip, self._host_port_str
                )
            else:
                self.connector_scheduler = DecodeConnectorScheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            if self.is_prefill:
                from omni_cache.connector.prefill import PrefillConnectorWorker
                self.connector_worker = PrefillConnectorWorker(
                    vllm_config, self.host_ip, self._host_port_str
                )
            else:
                from omni_cache.connector.decode import DecodeConnectorWorker
                self.connector_worker = DecodeConnectorWorker(
                    vllm_config, self.host_ip, self.cluster_id_start
                )

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [list(group) for group in block_ids])

    # ========== Scheduler Side Methods ==========

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int
    ) -> Tuple[int, bool]:
        """Get number of new matched tokens."""
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int
    ):
        """Update state after block allocation."""
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Build connector metadata."""
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.build_connector_metadata(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: List[int],
        spec_token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[dict]]:
        """Handle request finish."""
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.request_finished(
            request, block_ids, spec_token_ids or []
        )

    def get_finished_count(self) -> int | None:  # noqa: D401
        # In OmniCache the decode-side ZMQ pull is performed only by TP
        # rank 0 (see kv_loader._read_blocks). Only one worker marks the
        # transfer finished, but KVOutputAggregator expects N = world_size
        # notifications by default. Report 1 so the scheduler sees the
        # request done after a single notification.
        return 1

    def _get_finished_count_legacy(self) -> int | None:
        """Get expected completion count."""
        if self.is_prefill:
            return 1
        return None

    # ========== Worker Side Methods ==========

    def register_kv_caches(self, kv_pool_mmap_path, data_type, block_len_dtype, omni_cache=None):
        """Register KV caches."""
        data_type = 'bf16'
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.register_kv_caches(
            kv_pool_mmap_path, data_type, block_len_dtype, omni_cache
        )

    def get_finished(
        self,
        finished_req_ids: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Get finished transfers."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.get_finished(self._connector_metadata)

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs
    ) -> None:
        """Start loading KV from prefill."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        if not isinstance(self._connector_metadata, (DatadistConnectorMetadata, DatadistConnectorMetadataPrefill)):
            raise RuntimeError(
                "self._connector_metadata must be DatadistConnectorMetadata or "
                "DatadistConnectorMetadataPrefill"
            )
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Connector does not do layerwise loading."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs
    ) -> None:
        """Connector does not save explicitly."""
        pass

    def wait_for_save(self):
        """Connector does not save explicitly."""
        pass


__all__ = [
    "OmniCacheConnector",
]
