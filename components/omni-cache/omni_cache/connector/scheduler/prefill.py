# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Prefill scheduler implementation for OmniCache connector."""

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

from omni_cache.connector.utils.metadata import (
    DatadistConnectorMetadataPrefill,
    ReqMetaPrefill,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

logger = init_logger(__name__)


class PrefillConnectorScheduler:
    """Scheduler side implementation for prefill role."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        ox_shard_list: str,
        host_ip: str,
        host_port: str
    ):
        """Initialize the prefill scheduler.

        Args:
            vllm_config: vLLM configuration.
            ox_shard_list: OX shard endpoint list for this P cluster.
            host_ip: Host IP address.
            host_port: Host port.
        """
        self.vllm_config = vllm_config
        self.ox_shard_list = ox_shard_list
        self.host_ip = host_ip
        self.host_port = host_port
        logger.info(
            "[DYNAMIC-TOPO] Initializing OmniCacheConnector Scheduler ox_shard_list=%s host=%s:%s",
            ox_shard_list, host_ip, host_port
        )
        self.requests_finish_time: Dict[str, float] = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int
    ) -> Tuple[int, bool]:
        """Get number of new matched tokens for a request.

        Args:
            request: The request.
            num_computed_tokens: Number of already computed tokens.

        Returns:
            Tuple of (count, has_match).
        """
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int
    ) -> None:
        """Update state after block allocation.

        Args:
            request: The request.
            blocks: Allocated blocks.
            num_external_tokens: Number of external tokens.
        """
        pass

    def build_connector_metadata(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> "KVConnectorMetadata":
        """Build connector metadata for worker.

        Args:
            scheduler_output: Scheduler output.

        Returns:
            Connector metadata.
        """
        metadata = DatadistConnectorMetadataPrefill()
        metadata.requests = {
            req_id: ReqMetaPrefill(finish_time=finish_time)
            for req_id, finish_time in self.requests_finish_time.items()
        }
        self.requests_finish_time.clear()
        return metadata

    def request_finished(
        self,
        request: "Request",
        block_ids: List[int],
        spec_token_ids: Optional[List[int]] = None
    ) -> Tuple[bool, Optional[dict]]:
        """Handle request finish event.

        Args:
            request: The finished request.
            block_ids: Block IDs used by the request.
            spec_token_ids: Speculative token IDs.

        Returns:
            Tuple of (delay_free_blocks, kv_transfer_params).
        """
        spec_token_ids = spec_token_ids or []
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            self.requests_finish_time[request.request_id] = time.monotonic()
            logger.warning(
                "[DYNAMIC-TOPO] KV produced req_id=%s ox_shard_list=%s host=%s:%s blocks=%d",
                request.request_id, self.ox_shard_list,
                self.host_ip, self.host_port, len(block_ids),
            )

        kv_transfer_params = dict(
            remote_block_ids=block_ids,
            remote_cluster_id=self.ox_shard_list,
            remote_host_ip=f"tcp://{self.host_ip}:{self.host_port}",
            spec_token_ids=spec_token_ids,
            remote_dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            remote_request_id=request.request_id
        )
        return delay_free_blocks, kv_transfer_params
