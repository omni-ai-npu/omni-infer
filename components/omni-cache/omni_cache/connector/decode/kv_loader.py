# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""KV loading logic for decode connector."""

import itertools
import os
import time
from typing import TYPE_CHECKING, List, Optional

from vllm.logger import init_logger

from omni_cache.connector.utils.helpers import align_remote_block_ids
from omni_cache.connector.utils.metadata import PendingReq

if TYPE_CHECKING:
    from omni_cache.connector.utils.metadata import DatadistConnectorMetadata

logger = init_logger("vllm.v1.omni")


def _get_kv_cache_spec(omni_cache, grp_idx: int):
    if omni_cache is None:
        return None
    try:
        group = omni_cache.kv_cache_config.kv_cache_groups[grp_idx]
    except (AttributeError, IndexError, TypeError):
        return None
    return getattr(group, "kv_cache_spec", None)


def _is_mome_group(omni_cache, grp_idx: int) -> bool:
    spec = _get_kv_cache_spec(omni_cache, grp_idx)
    return (
        getattr(spec, "mamba_type", None) == "mome"
        or type(spec).__name__ == "MomeSpec"
    )


def _is_dsa_group(omni_cache, grp_idx: int) -> bool:
    spec = _get_kv_cache_spec(omni_cache, grp_idx)
    return type(spec).__name__ == "DSAAttentionSpec"


def handle_future_callback(future):
    """Handle future completion callback."""
    from omni_cache.connector.utils.process_utils import handle_exception
    handle_exception(future, logger)


class KVLoader:
    """Handles KV loading operations from prefill nodes."""

    def __init__(self, worker):
        self.worker = worker
        self.omni_cache = None

    def set_omni_cache(self, omni_cache):
        """Set the omni cache reference."""
        self.omni_cache = omni_cache

    def start_load_kv(self, metadata: "DatadistConnectorMetadata") -> None:
        """Start loading KV blocks from prefill node.

        Args:
            metadata: Datadist connector metadata containing request info.
        """
        if int(os.getenv("ENABLE_MOCK_P", "0")):
            self.start_load_kv_mock_prefill(metadata)
            return

        futures = []
        for req_id, meta in metadata.requests.items():
            future = self._process_request(req_id, meta)
            if future:
                futures.append(future)

        for future in futures:
            future.add_done_callback(handle_future_callback)

    def _process_request(self, req_id: str, meta):
        """Process a single request's KV load.

        Args:
            req_id: Request ID.
            meta: Request metadata.

        Returns:
            Future object if request is processed, None otherwise.
        """
        logger.debug(f" ***** start_load_kv: processing request {req_id}")

        if len(meta.local_block_ids) == 0:
            logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
            return None

        # block_table is a list --> non-hybrid attn
        if isinstance(meta.local_block_ids[0], int):
            return self._process_flat_blocks(req_id, meta)
        elif isinstance(meta.local_block_ids[0], list):
            return self._process_nested_blocks(req_id, meta)
        else:
            logger.error(f"Unexpected type for meta.local_block_ids[0]: {type(meta.local_block_ids[0])}")
            raise RuntimeError(f"Unexpected type for meta.local_block_ids[0]: {type(meta.local_block_ids[0])}")

    def _process_flat_blocks(self, req_id: str, meta):
        """Process flat block list (non-omni-attention case).

        Args:
            req_id: Request ID.
            meta: Request metadata.

        Returns:
            Future object.
        """
        print(meta.remote_block_ids, meta.local_block_ids)
        if isinstance(meta.remote_block_ids[0], list):
            meta.remote_block_ids = list(itertools.chain(*meta.remote_block_ids))
        meta.local_block_ids, meta.remote_block_ids = align_remote_block_ids(
            meta.local_block_ids,
            meta.remote_block_ids,
        )

        if len(meta.remote_block_ids) != len(meta.local_block_ids):
            logger.debug("look ahead token num is greater than 0")

        logger.info(
            " ***** start_load_kv for request %s "
            "Num local_block_ids: %s. Num remote_block_ids: %s.",
            req_id,
            len(meta.local_block_ids),
            len(meta.remote_block_ids)
        )

        meta.local_block_ids = [list(itertools.chain(*meta.local_block_ids)), meta.local_block_ids]
        meta.remote_block_ids = list(itertools.chain(*meta.remote_block_ids))
        return self._submit_read_blocks(req_id, meta)

    def _process_nested_blocks(self, req_id: str, meta):
        """Process nested block list (omni-attention case).

        Args:
            req_id: Request ID.
            meta: Request metadata.

        Returns:
            Future object or None.
        """
        # Skip only when every group has zero blocks.
        if all(len(meta.local_block_ids[i]) == 0 for i in range(len(meta.local_block_ids))):
            logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
            return None

        if len(meta.local_block_ids) != len(meta.remote_block_ids):
            raise RuntimeError(
                f"Group count mismatch for request {req_id}: "
                f"local has {len(meta.local_block_ids)} groups, "
                f"remote has {len(meta.remote_block_ids)} groups."
            )

        has_lookahead = False
        for grp_idx in range(len(meta.local_block_ids)):
            if _is_dsa_group(self.omni_cache, grp_idx):
                if len(meta.local_block_ids[grp_idx]) < len(meta.remote_block_ids[grp_idx]):
                    if len(meta.remote_block_ids[grp_idx]) - len(meta.local_block_ids[grp_idx]) == 1:
                        has_lookahead = True
                        break
                    else:
                        raise RuntimeError(
                            f"Remote block count exceeds local by more than 1 in DSA group {grp_idx} "
                            f"for request {req_id}: local has {len(meta.local_block_ids[grp_idx])} blocks, "
                            f"remote has {len(meta.remote_block_ids[grp_idx])} blocks."
                        )
        for grp_idx in range(len(meta.local_block_ids)):
            is_mome_group = _is_mome_group(self.omni_cache, grp_idx)
            remote_overflow = (
                "tail" if is_mome_group
                else "head"
            )
            meta.local_block_ids[grp_idx], meta.remote_block_ids[grp_idx] = align_remote_block_ids(
                meta.local_block_ids[grp_idx],
                meta.remote_block_ids[grp_idx],
                remote_overflow=remote_overflow,
                remote_tail_skip=1 if is_mome_group and has_lookahead else 0,
            )

        for grp_idx in range(len(meta.local_block_ids)):
            logger.info(
                " ***** start_load_kv for request %s, group %s: "
                "Num local_block_ids: %s. Num remote_block_ids: %s.",
                req_id, grp_idx,
                len(meta.local_block_ids[grp_idx]),
                len(meta.remote_block_ids[grp_idx]),
            )

        # Flatten nested groups into [flat_list, per_group_tuple] for
        # downstream consumers that expect local_block_ids[0] to be flat.
        meta.local_block_ids = [list(itertools.chain(*meta.local_block_ids)), meta.local_block_ids]
        meta.remote_block_ids = list(itertools.chain(*meta.remote_block_ids))
        return self._submit_read_blocks(req_id, meta)

    def _submit_read_blocks(self, req_id: str, meta):
        """Submit read blocks task to executor.

        Args:
            req_id: Request ID.
            meta: Request metadata.

        Returns:
            Future object.
        """
        remote_blocks = (
            meta.remote_block_ids if isinstance(meta.remote_block_ids[0], int)
            else meta.remote_block_ids[0]
        )

        return self.worker.executor.submit(
            self._read_blocks,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=remote_blocks,
            dst_cluster_id=meta.remote_cluster_id,
            request_id=req_id,
            remote_request_id=meta.remote_request_id,
            remote_host_ip=meta.remote_host,
            remote_dp_rank=meta.remote_dp_rank,
        )

    def _read_blocks(
        self,
        local_block_ids: List[List[int]],
        remote_block_ids: List[int],
        dst_cluster_id: str,
        request_id: str,
        remote_request_id: Optional[str],
        remote_host_ip: str,
        remote_dp_rank: int = 0,
    ):
        """Read blocks from remote prefill node.

        Only TP rank 0 actually issues the ZMQ pull request. All TPs on
        the same node share one mmap'd host pool (hugetlbfs), so a single
        pull makes the bytes visible to every rank — and having all 8 TPs
        send with the same dp_local_rank identity makes OX's
        dealer-routing deliver responses to only a subset of senders,
        which then deadlocks the TP barrier in _post_success.
        """
        start = time.time()
        logger.debug(
            f"<<<<<<<<< {request_id=}; {remote_dp_rank=}; "
            f"{local_block_ids=}; {remote_block_ids=} >>>>>>>>>"
        )

        tp_rank = getattr(self.omni_cache, "tp_rank", 0)
        tp_world_size = getattr(self.omni_cache, "tp_world_size", 1)
        if tp_world_size > 1 and tp_rank != 0:
            logger.debug(
                "skip ZMQ pull on TP rank %d (world=%d) for req %s "
                "(TP rank 0 is the single sender when TP>1)",
                tp_rank, tp_world_size, request_id,
            )
            return

        # Verification aid: `OMNI_CACHE_SKIP_OX_PULL=1` bypasses the
        # actual byte transfer but still lets the scheduler mark the
        # request as "received" so decode proceeds. Used to confirm
        # whether attention KV bytes are actually being transferred —
        # if decode output is identical with this set vs unset, the
        # transfer is a no-op and decode is generating from zeroed HBM.
        import os as _os
        if int(_os.getenv("OMNI_CACHE_SKIP_OX_PULL", "0")):
            req_skip = PendingReq(
                request_id=request_id,
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                dst_cluster_id=dst_cluster_id,
                remote_request_id=remote_request_id,
                remote_host_ip=remote_host_ip,
                dp_rank=self.omni_cache.dp_local_rank,
                t_submit=start,
            )
            req_skip.t_sent = start
            req_skip.t_resp = start
            self.worker.pending[request_id] = req_skip
            # Directly mark as done-receiving without issuing ZMQ.
            self.worker.recv_q.put(request_id)
            logger.warning(
                "[SKIP_OX_PULL=1] req=%s marked received without transfer",
                request_id,
            )
            return

        req = PendingReq(
            request_id=request_id,
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
            dst_cluster_id=dst_cluster_id,
            remote_request_id=remote_request_id,
            remote_host_ip=remote_host_ip,
            dp_rank=self.omni_cache.dp_local_rank,
            t_submit=start,
        )
        req.t_sent = start
        self.worker.pending[request_id] = req

        # Drop (src, dst) pairs where either side is the null block (id 0).
        # Transferring against block 0 overwrites the shared null/sentinel
        # slot with whatever happens to live in the prefill's block 0 at
        # this moment, and that junk later bleeds into any decode layer
        # whose block_table still points at null (rolled-out SWA positions),
        # producing non-deterministic / looping output across runs.
        dst_flat = local_block_ids[0]
        if len(dst_flat) == len(remote_block_ids):
            pairs = [(s, d) for s, d in zip(remote_block_ids, dst_flat)
                     if s != 0 and d != 0]
            src_clean = [s for s, _ in pairs]
            dst_clean = [d for _, d in pairs]
        else:
            src_clean = remote_block_ids
            dst_clean = dst_flat

        self.worker.zmq_client.send_request(
            request_id=request_id,
            cluster_id=dst_cluster_id,
            src_id_list=src_clean,
            dst_id_list=dst_clean,
            rank_id=self.omni_cache.dp_local_rank,
            src_dp_rank=remote_dp_rank,
        )

        logger.warning(
            "Adding Sent Queue req_id=%s in %.6f s",
            request_id, time.time() - start
        )

    def start_load_kv_mock_prefill(self, metadata: "DatadistConnectorMetadata") -> None:
        """Mock prefill mode for testing.

        Args:
            metadata: Datadist connector metadata.
        """
        logger.warning(" ***** MOCK P is enabled")
        logger.info(f" ***** start_load_kv: {len(metadata.requests)}")

        futures = []
        for req_id, meta in metadata.requests.items():
            if len(meta.local_block_ids) == 0:
                logger.info(f" ***** Request {req_id} has 0 local blocks, skip load kv.")
                continue

            future = self.worker.executor.submit(
                self._read_blocks_mock_prefill,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=[0],
                dst_cluster_id="0",
                request_id=req_id,
                remote_request_id=req_id,
                remote_host_ip="0.0.0.0",
                remote_dp_rank=0,
            )
            futures.append(future)

        for future in futures:
            future.add_done_callback(handle_future_callback)

    def _read_blocks_mock_prefill(
        self,
        local_block_ids: List[List[int]],
        remote_block_ids: List[int],
        dst_cluster_id: str,
        request_id: str,
        remote_request_id: Optional[str],
        remote_host_ip: str,
        remote_dp_rank: int = 0,
    ):
        """Mock read blocks for testing without actual prefill.

        Args:
            local_block_ids: Local block IDs.
            remote_block_ids: Remote block IDs (ignored in mock).
            dst_cluster_id: Destination cluster ID.
            request_id: Request ID.
            remote_request_id: Remote request ID.
            remote_host_ip: Remote host IP.
            remote_dp_rank: Remote DP rank.
        """
        logger.warning("<<<<<< In _read_blocks_mock_prefill")
        start = time.time()

        req = PendingReq(
            request_id=request_id,
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
            dst_cluster_id=dst_cluster_id,
            remote_request_id=remote_request_id,
            remote_host_ip=remote_host_ip,
            dp_rank=self.omni_cache.dp_local_rank,
            t_submit=start,
        )

        self.worker.pending[request_id] = req

        try:
            self.worker.recv_q.put(request_id)
        except Exception as e:
            raise RuntimeError(f"Error: {e}")
