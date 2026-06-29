# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Process and thread management for decode connector."""

import asyncio
import multiprocessing
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional

import torch
import zmq
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tp_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from omni_cache.connector.decode.worker import DecodeConnectorWorker

logger = init_logger("vllm.v1.omni")


class ZMQClientWrapper:
    """Wrapper for ZMQ client operations."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._send_q = None
        self._client = None

    def set_send_queue(self, send_q):
        """Set the send queue."""
        self._send_q = send_q

    def set_client(self, client):
        """Set the underlying client."""
        self._client = client

    def send_request(self, **kwargs):
        """Send a request through the underlying client."""
        if self._client:
            return self._client.send_request(**kwargs)
        return False


class ProcessManager:
    """Manages subprocesses and threads for decode connector."""

    def __init__(self, worker: "DecodeConnectorWorker"):
        self.worker = worker
        self.omni_cache = None

    def set_omni_cache(self, omni_cache):
        """Set the omni cache reference."""
        self.omni_cache = omni_cache

    def ensure_processes_started(self) -> None:
        """Ensure all worker processes and threads are started."""
        self._start_resp_process()
        self._start_address_process()
        self._start_h2d_thread()
        self._start_monitor_thread()

    def _start_resp_process(self) -> None:
        """Start ZMQ response process."""
        if not (self.worker.resp_process and self.worker.resp_process.is_alive()):
            self.worker.resp_stop.clear()
            self.worker.resp_process = multiprocessing.get_context("fork").Process(
                target=self._process_zmq, name="ZMQ-Recv-Thread", daemon=True
            )
            self.worker.resp_process.start()
            self.worker.zmq_client.set_client(self.worker._create_zmq_proxy())
            logger.info("ZMQ receive thread started at %s", self.worker.endpoint)

    def _start_address_process(self) -> None:
        """Start H2D address process."""
        if not (self.worker.address_process and self.worker.address_process.is_alive()):
            self.worker.address_stop.clear()
            self.worker.address_process = multiprocessing.get_context("fork").Process(
                target=self._process_h2d_address, name="Address-Process", daemon=True
            )
            self.worker.address_process.start()
            logger.info("Address-Process started at %s", self.worker.endpoint)

    def _start_h2d_thread(self) -> None:
        """Start H2D worker thread."""
        if not (self.worker.h2d_thread and self.worker.h2d_thread.is_alive()):
            self.worker.h2d_stop.clear()
            self.worker.h2d_thread = threading.Thread(target=self._h2d_worker, name="H2D-Worker", daemon=True)
            self.worker.h2d_thread.start()
            logger.info("H2D worker thread started")

    def _start_monitor_thread(self) -> None:
        """Start process monitor thread."""
        if not (self.worker.monitor_thread and self.worker.monitor_thread.is_alive()):
            self.worker.monitor_stop.clear()
            self.worker.monitor_thread = threading.Thread(
                target=self._monitor_processes, name="Process Monitor", daemon=True
            )
            self.worker.monitor_thread.start()
            logger.info("Process Monitor thread started")

    def _monitor_processes(self) -> None:
        """Monitor subprocess health."""
        while not self.worker.monitor_stop.is_set():
            is_alive = [
                self.worker.resp_process.is_alive() if self.worker.resp_process else False,
                self.worker.address_process.is_alive() if self.worker.address_process else False,
            ]
            if sum(is_alive) != len(is_alive):
                logger.warning(f"[resp_process, address_process] alive status = {is_alive}")
            time.sleep(1)

    def _process_zmq(self) -> None:
        """Process ZMQ messages in subprocess."""
        client = self.worker._create_zmq_client()

        async def sender():
            while not self.worker.resp_stop.is_set():
                try:
                    item = await asyncio.to_thread(self.worker.send_q.get)
                except asyncio.CancelledError:
                    break
                t0 = time.time()
                ok = client.send_request(
                    request_id=item.request_id,
                    cluster_id=item.cluster_id,
                    src_id_list=item.src_ids,
                    dst_id_list=item.dst_ids,
                    rank_id=item.rank_id,
                    src_dp_rank=getattr(item, "src_dp_rank", 0),
                )
                if not ok:
                    raise ValueError(f"Send failed for req_id={item.request_id}")
                logger.warning("Sent req_id=%s in %.6f s", item.request_id, time.time() - t0)

        async def receiver():
            while not self.worker.resp_stop.is_set():
                try:
                    resp = await asyncio.to_thread(client.receive_response, None)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    raise RuntimeError(f" ***** Failed to receive zmq response with error code {e}") from e

                req_id = resp.get("request_id")
                success = bool(resp.get("success"))
                if not req_id:
                    raise ValueError(f"Received response without request_id: {resp}")
                if not success:
                    raise ValueError(f"Failed to pull kv of request {req_id}")

                try:
                    await asyncio.to_thread(self.worker.recv_q.put, req_id)
                except asyncio.CancelledError:
                    break

                self._handle_pulled_kv(req_id)

        async def async_main():
            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())
            while not self.worker.resp_stop.is_set():
                await asyncio.sleep(0.5)
            sender_task.cancel()
            receiver_task.cancel()

        asyncio.run(async_main())

    def _handle_pulled_kv(self, req_id: str) -> None:
        """Handle pulled KV notification.

        Args:
            req_id: Request ID.
        """
        ctx = self.worker.pending.get(req_id)
        if not ctx:
            return

        # Only TP rank 0 pulls KV (see KVLoader._read_blocks). The ack
        # back to prefill is a pure control message — no cross-TP sync
        # needed.
        if ctx.remote_request_id is not None:
            self.worker._send_pulled_kv_req_list(ctx.remote_host_ip, [ctx.remote_request_id])

    def _process_h2d_address(self) -> None:
        """Process H2D address mappings in subprocess."""
        while not self.worker.address_stop.is_set():
            batch_device_mem = []
            batch_device_max = []
            batch_host_mem = []
            batch_host_sizes = []
            ctxs = []
            start_time = time.time()

            while True:
                if os.getenv("ENABLE_HOST_MAPPING", "1") == "1":
                    # Slot 0 is a permanent sentinel. All other lanes
                    # [1, num_max_batch_pool) are available for real
                    # requests. Gate when all lanes are saturated.
                    if int(os.getenv("USE_OMNI_INPUT_BATCH", "0")):
                        req_to_lane = self.omni_cache.req_id_to_idx
                        if req_to_lane is None:
                            raise RuntimeError("req_id_to_idx is not initialized")
                        real_full = len(req_to_lane) >= self.omni_cache.num_max_batch_pool
                        if real_full:
                            break
                    else:
                        rmap = self.omni_cache.record_batch_idx_to_req
                        if rmap is not None:
                            real_full = all(
                                v is not None for k, v in rmap.items() if int(k) < self.omni_cache.num_max_batch_pool
                            )
                            if real_full:
                                break

                try:
                    req_id = self.worker.recv_q.get_nowait()
                except queue.Empty:
                    break

                ctx = self.worker.pending.get(req_id)
                if ctx:
                    ctx.t_resp = time.time()
                if not ctx:
                    raise ValueError("Orphan response for unknown req_id=%s", req_id)

                self.worker._log_network_timing(ctx)

                device_mem, device_max, host_mem, host_sizes = self.omni_cache.build_h2d_ops(
                    ctx.local_block_ids, ctx.request_id
                )
                batch_device_mem.extend(device_mem)
                batch_device_max.extend(device_max)
                batch_host_mem.extend(host_mem)
                batch_host_sizes.extend(host_sizes)
                ctxs.append(ctx)

            if len(ctxs) > 0:
                logger.debug(
                    f" ***** process batch copy address: len(req_id): {len(ctxs)}, cost: {time.time() - start_time} s"
                )
                self.worker.h2d_q.put((batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes, ctxs))

    def _h2d_worker(self) -> None:
        """H2D worker thread main loop."""
        self.omni_cache.host_cache.ascend_cl_stream.create()

        while not self.worker.h2d_stop.is_set():
            batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes, ctxs = self.worker.h2d_q.get()
            t_h2d_start = time.time()

            try:
                self._post_success(batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes, ctxs)
            except Exception as e:
                logger.exception(
                    "H2D worker error on req_id=%s: %s", ",".join([str(ctx.request_id) for ctx in ctxs]), e
                )
            finally:
                for ctx in ctxs:
                    self.worker.pending.pop(ctx.request_id, None)

            t_h2d_end = time.time()
            logger.debug(
                " **** Time cost of decode synchronize_h2d is %.3f ms len(req_id):%s",
                (t_h2d_end - t_h2d_start) * 1000.0,
                len(ctxs),
            )
            total_cost = [round(t_h2d_end - ctx.t_submit, 6) for ctx in ctxs]
            logger.debug(f" **** Read block Total: len(req_id): {len(ctxs)}, cost: {total_cost} s")

    def _post_success(self, batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes, ctxs):
        """Post successful H2D transfer.

        Args:
            batch_device_mem: Device memory addresses.
            batch_device_max: Device memory max sizes.
            batch_host_mem: Host memory addresses.
            batch_host_sizes: Host memory sizes.
            ctxs: List of request contexts.
        """
        if self.omni_cache is None or self.omni_cache.device_cache is None:
            raise RuntimeError("Error! omni_cache is None or device_cache is None.")

        self.omni_cache.synchronize_h2d(batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes)

        # KV diagnostics: probe decode host pool (Stage 3) — host bytes before H2D
        from tools.diagnostics.probes import probe_decode_host

        probe_decode_host(self.omni_cache, ctxs)

        # DSA-split: post-pull copy into secondary host pool
        # If the DSA-split feature is on, copy the leading head_size
        # bytes of every pulled DSA slot into the narrow-head
        # secondary pool. The primary already holds the full 704-byte
        # slot; the secondary is what the attention kernel reads via
        # its MMU alias into HBM. This runs ONLY on the TP rank that
        # actually pulled (by the same convention as the primary
        # H2D above). Best-effort — any failure here is logged and
        # swallowed so a malformed layout cannot stall the pull.
        _dsa_pool = getattr(self.omni_cache, "dsa_host_pool", None)
        if _dsa_pool is not None:
            try:
                # kvi_tensors_swap[0] is the 4-D (L,B,T,H) reshape; fallback to shared_tensor only if missing.
                _pkvi = getattr(self.omni_cache.host_cache, "kvi_tensors", None)
                primary_shared = (
                    _pkvi[0]
                    if (_pkvi and len(_pkvi) > 0)
                    else getattr(self.omni_cache.host_cache, "shared_tensor", None)
                )
                _pkvi_swap = getattr(self.omni_cache.host_cache, "kvi_tensors_swap", None)
                primary_swap = _pkvi_swap[0] if (_pkvi_swap and len(_pkvi_swap) > 0) else None
                # If still not 4-D, the copy helper will error and be caught.
                # Resolve the index of the DSA group so we can pick only DSA blocks.
                dsa_group_idx = None
                kv_cfg = getattr(self.omni_cache, "kv_cache_config", None)
                if kv_cfg is not None:
                    for _gi, _g in enumerate(kv_cfg.kv_cache_groups):
                        if type(_g.kv_cache_spec).__name__ == "DSAAttentionSpec":
                            dsa_group_idx = _gi
                            break
                # DSA secondary pool is aligned 1:1 with the primary's DSA-only
                # per-layer pool, so dsa_layer_indices is just range(num_dsa_layers).
                dsa_layers = list(range(_dsa_pool.num_layers))
                all_block_ids = []
                for ctx in ctxs:
                    blks = getattr(ctx, "local_block_ids", None)
                    if not blks or dsa_group_idx is None:
                        continue
                    per_group = None
                    if len(blks) >= 2 and isinstance(blks[1], (list, tuple)):
                        per_group = blks[1]
                    elif isinstance(blks[0], (list, tuple)):
                        per_group = blks
                    if per_group is None or dsa_group_idx >= len(per_group):
                        continue
                    for b in per_group[dsa_group_idx]:
                        try:
                            all_block_ids.append(int(b))
                        except (TypeError, ValueError):
                            pass
                if dsa_layers and all_block_ids and primary_shared is not None:
                    _dsa_pool.copy_from_primary(
                        primary_shared,
                        dsa_layers,
                        list(set(all_block_ids)),
                        primary_device_tensor=primary_swap,
                    )
            except Exception as _exc:
                logger.warning("[DSA-SPLIT] post-pull copy failed: %s", _exc)
                if os.getenv("ENABLE_OMNI_CACHE_DSA_SPLIT", "0") == "1":
                    raise RuntimeError(f"[DSA-SPLIT] post-pull copy failed: {_exc}") from _exc

        # Only TP rank 0 pulls KV (single-sender model — see
        # kv_loader._read_blocks). Scheduler queries rank 0's
        # recving_transfers only, so no cross-TP sync needed here. For
        # Pangu V2 the host pool is shared mmap'd hugepages across all
        # TPs, and the attention kernel reads via MMU alias — the
        # per-TP H2D above is a no-op.

        # KV diagnostics: probe decode HBM (Stage 4) after H2D copy.
        #
        # Before reading HBM bytes, drain EVERY device stream. The H2D is
        # queued on `ascend_cl_stream` (and `synchronize_h2d_decode`
        # already syncs that one), but the MoMe per-layer state tensors
        # are *aliases* over the same byte ranges the attention kernel
        # touches during the decode forward; if decode on the compute
        # stream has already been issued when the probe's `.cpu()` runs,
        # the bytes we read reflect a partially-shifted conv_state
        # (typically just the newest token `k=num_total_tokens-1`,
        # masking the H2D fill with only the fresh-slot data).
        import torch as _torch

        try:
            _torch.npu.synchronize()
        except Exception:
            pass

        from tools.diagnostics.probes import probe_decode_hbm

        probe_decode_hbm(self.omni_cache, ctxs)

        for ctx in ctxs:
            self.worker.recving_transfers.put(ctx.request_id)
