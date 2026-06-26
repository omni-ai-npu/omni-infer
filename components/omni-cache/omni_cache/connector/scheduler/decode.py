# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Decode scheduler implementation for OmniCache connector."""

import itertools
import os
import pickle
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import zmq
from vllm.logger import init_logger

from omni_cache.cache.utils.support import _is_pangu_v2_model
from omni_cache.connector.utils.helpers import should_round_prompt_tokens
from omni_cache.connector.utils.metadata import DatadistConnectorMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

logger = init_logger(__name__)


class DecodeConnectorScheduler:
    """Scheduler side implementation for decode role."""

    def __init__(self, vllm_config: "VllmConfig"):
        """Initialize the decode scheduler.

        Args:
            vllm_config: vLLM configuration.
        """
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._reqs_need_recv: Dict[str, Tuple["Request", List[int]]] = {}
        self.processed_request: Set[str] = set()

        self.is_pangu_v2 = _is_pangu_v2_model(vllm_config)

        additional_config = vllm_config.additional_config or {}
        self.async_pull_kv = additional_config.get("async_pull_kv", False)

        if self.async_pull_kv:
            self.context = zmq.Context()
            self.pub = self.context.socket(zmq.PUB)
            self.pub.bind(
                f"ipc:///tmp/sched-pub-{vllm_config.parallel_config.data_parallel_rank_local}"
            )

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
        prompt_len = len(request.prompt_token_ids)
        if request.request_id in self.processed_request:
            logger.error(
                "@@@@@@ MATCH_SKIP reason=processed req_id=%s prompt_len=%s "
                "local_hit_tokens=%s",
                request.request_id,
                prompt_len,
                num_computed_tokens,
            )
            return 0, False

        params = request.kv_transfer_params
        if params is None:
            logger.error(
                "@@@@@@ MATCH_SKIP reason=no_params req_id=%s prompt_len=%s "
                "local_hit_tokens=%s",
                request.request_id,
                prompt_len,
                num_computed_tokens,
            )
            return 0, False

        logger.debug(
            "DatadistConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params
        )

        if num_computed_tokens % self.block_size != 0:
            raise RuntimeError(
                "num_computed_tokens must be divisible by self.block_size"
            )

        patches = os.getenv("OMNI_NPU_VLLM_PATCHES", "")
        omni_cache_enabled = os.getenv("ENABLE_OMNI_CACHE", "0") == "1"
        host_mapping_enabled = os.getenv("ENABLE_HOST_MAPPING", "1") == "1"
        should_round = should_round_prompt_tokens(
            patches=patches,
            omni_cache_enabled=omni_cache_enabled,
            host_mapping_enabled=host_mapping_enabled,
        )

        prompt_tokens = len(request.prompt_token_ids)
        target_prompt_tokens = (
            self._round_up(prompt_tokens, self.block_size)
            if should_round else prompt_tokens
        )
        count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
        return count, count > 0

    def _round_up(self, x: int, y: int) -> int:
        """Round x up to nearest multiple of y."""
        return ((x + y - 1) // y) * y

    # def get_unhashed_block_ids(self, blocks) -> list[int]:
    #     """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
    #     return [block.block_id for group in blocks.blocks for block in group if block.block_hash is None]
    def get_unhashed_block_ids(self, blocks) -> tuple[list[int], ...]:
        return tuple(
            [block.block_id for block in group if block.block_hash is None]
            for group in blocks.blocks
        )

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
        logger.debug(
            f"Request id {request.request_id}: blocks length is {len(blocks.blocks)}"
        )
        params = request.kv_transfer_params
        logger.debug(
            "DatadistConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params
        )

        if params is not None:
            if params.get("remote_block_ids"):
                if (all(p in params for p in ("remote_cluster_id", "remote_host_ip"))
                    or int(os.getenv("ENABLE_MOCK_P", "0"))):
                    if self.is_pangu_v2:
                        self._reqs_need_recv[request.request_id] = (
                            request, self.get_unhashed_block_ids(blocks))
                    else:
                        self._reqs_need_recv[request.request_id] = (
                            request, blocks.get_unhashed_block_ids())
                else:
                    logger.warning("Got invalid KVTransferParams: %s.", params)

        self.processed_request.add(request.request_id)

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
        metadata = DatadistConnectorMetadata()
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            if isinstance(block_ids, tuple):
                # Pangu V2 hybrid: block_ids is tuple[list[int], ...] per group.
                formatted_ids = list(block_ids)
            else:
                # Non-hybrid: block_ids is already a flat list[int].
                formatted_ids = block_ids

            if req.kv_transfer_params is None:
                logger.warning(f"For request {req_id}: kv_transfer_params now is None")
            else:
                metadata.add_new_req(
                    request_id=req_id,
                    local_block_ids=formatted_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            req.kv_transfer_params = None
        self._reqs_need_recv.clear()

        if self.async_pull_kv:
            if scheduler_output is None and metadata.requests:
                serialized_data = pickle.dumps(metadata)
                self.pub.send(serialized_data)

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
        if request.request_id in self.processed_request:
            self.processed_request.remove(request.request_id)
        return False, None
