#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
test_zmq_stress.py — ZMQ stress test for mock_ox

Mimics the connector D-side worker's ZMQ communication pattern:

  Connector architecture (real):
    vLLM scheduler → KVLoader._read_blocks (ThreadPoolExecutor, max_workers=1)
      → ZMQSendProxy.send_request → multiprocessing.Queue (unbounded)
        → _process_zmq subprocess:
            sender()   coroutine: drain queue → DEALER socket.send()
            receiver() coroutine: DEALER socket.recv() → recv_q

  This script mirrors that:
    producer_task × 1 (fixed rate) → asyncio.Queue → sender_task × 1 → DEALER.send()
                                                                   │
    receiver_task × 1 ←── DEALER.recv() ←──────────────────────────┘

  Key design choices matching the real connector:
    - Only 1 coroutine touches the DEALER socket for send (sender_task)
    - Only 1 coroutine touches the DEALER socket for recv (receiver_task)
    - sender and receiver run independently (no flow control / no windowing)
    - Single producer enqueues at a fixed rate (--rate), matching the real
      connector where KVLoader._read_blocks is single-threaded and driven
      by the decode scheduler. The rate should be slightly below the sender's
      max throughput to prevent unbounded queue growth.

Usage:
  python test_zmq_stress.py --rate 900
  python test_zmq_stress.py --rate 900 --zmq-port 16666 --num-blocks 4

The test runs indefinitely. If a response is missing (timeout) or a request_id
mismatch is detected, it prints an error and exits with code 1.
"""

import argparse
import asyncio
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Optional

import msgpack
import zmq
import zmq.asyncio

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ZMQ_PORT = 15555
DEFAULT_RATE = 900              # requests per second (matching connector's single-threaded producer)
DEFAULT_NUM_BLOCKS = 4          # number of block IDs per request
DEFAULT_NUM_OX_THREADS = 16
DEFAULT_RECV_TIMEOUT_MS = 30000 # 30s — if no response in this time, likely lost

# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------
_ut_log_file = None


def ut_log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}"
    print(line, flush=True)
    if _ut_log_file:
        _ut_log_file.write(line + "\n")
        _ut_log_file.flush()


# ---------------------------------------------------------------------------
# Ox process management (mirrors connector's DecodeConnectorWorker pattern)
# ---------------------------------------------------------------------------
def _stdout_reader(pipe, q: queue.Queue):
    """Read lines from ox stdout and put them on a queue."""
    for line in iter(pipe.readline, ""):
        q.put(line)
    q.put(None)  # sentinel
    pipe.close()


def _stdout_printer(q: queue.Queue, log_path: str):
    """Consume lines from queue and write to ox log file."""
    os.makedirs(log_path, exist_ok=True)
    log_file_path = os.path.join(log_path, "ox_log_d_client.log")
    with open(log_file_path, "w", buffering=1) as f:
        while True:
            try:
                line = q.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                break
            f.write(line)
            f.flush()


def launch_mock_ox(ox_binary: str, zmq_port: int, num_threads: int,
                   log_dir: str) -> subprocess.Popen:
    """Launch mock_ox subprocess, mirroring connector's _start_ox_process."""
    cmd = [
        ox_binary,
        "--zmq-port", str(zmq_port),
        "--num-blocks", "1024",
        "--num-threads", str(num_threads),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Start reader/printer threads (same pattern as connector)
    q: queue.Queue = queue.Queue()
    reader_thread = threading.Thread(target=_stdout_reader, args=(proc.stdout, q), daemon=True)
    printer_thread = threading.Thread(target=_stdout_printer, args=(q, log_dir), daemon=True)
    reader_thread.start()
    printer_thread.start()

    return proc


def stop_mock_ox(proc: subprocess.Popen, timeout: float = 5.0):
    """Gracefully stop the mock_ox process."""
    try:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ZMQ DEALER client (mirrors connector's RouterDealerClient)
#
# IMPORTANT: This class is NOT thread-safe. Only one coroutine should call
# send_request at a time, and only one coroutine should call receive_response
# at a time. This matches the real connector where a single sender() and a
# single receiver() coroutine operate on the DEALER socket.
# ---------------------------------------------------------------------------
class StressDealerClient:
    """Async DEALER socket client that talks to mock_ox's ROUTER.

    Uses zmq.asyncio so both send and recv are native coroutines running
    in the same event loop thread — no thread-safety issues, no blocking.
    """

    def __init__(self, server_address: str, client_id: Optional[bytes] = None):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)

        if client_id is None:
            client_id = f"stress_{uuid.uuid4().hex[:8]}".encode("utf-8")
        self.socket.setsockopt(zmq.IDENTITY, client_id)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.client_id = client_id

        self.socket.connect(server_address)
        ut_log(f"DEALER connected to {server_address} with ID: {client_id.decode()}")

    async def send_request(self, request_id: str, cluster_id: int,
                     src_ids: list, dst_ids: list,
                     rank_id: int, src_dp_rank: int = 0) -> bool:
        """Send a request to mock_ox (same format as connector)."""
        try:
            request_data = {
                "request_id": request_id,
                "table_id": rank_id,
                "src_block_ids": src_ids,
                "dst_block_ids": dst_ids,
                "cluster_id": cluster_id,
                "src_dp_rank": src_dp_rank,
            }
            packed = msgpack.packb(request_data)
            await self.socket.send(packed)
            return True
        except Exception as e:
            ut_log(f"ERROR sending request {request_id}: {e}")
            return False

    async def receive_response(self, timeout_ms: int = 1000) -> Optional[dict]:
        """Receive a response from mock_ox."""
        try:
            if await self.socket.poll(timeout_ms, zmq.POLLIN):
                data = await self.socket.recv()
                return msgpack.unpackb(data)
        except Exception as e:
            ut_log(f"ERROR receiving response: {e}")
        return None

    def close(self):
        self.socket.close()
        self.context.term()


# ---------------------------------------------------------------------------
# Async stress test logic
#
# Architecture (matching connector):
#
#   producer_task × 1 (fixed rate) ──→ asyncio.Queue ──→ sender_task × 1 ──→ DEALER.send()
#                                                                           │
#   receiver_task × 1 ←── DEALER.recv() ←──────────────────────────────────┘
#
# The producer enqueues at a fixed rate (--rate req/s), matching the real
# connector where KVLoader._read_blocks is single-threaded and driven by
# the decode scheduler. The rate should be slightly below the sender's
# max throughput to prevent unbounded queue growth.
# ---------------------------------------------------------------------------

# Item passed from producer to sender (mirrors _SendItem in connector)
_SendItem = tuple  # (request_id, cluster_id, src_ids, dst_ids, rank_id, src_dp_rank)


async def producer_task(send_q: asyncio.Queue, rate: float,
                        num_blocks: int, stop_event: asyncio.Event):
    """Enqueue requests at a fixed rate into the send queue.

    Mirrors KVLoader._read_blocks putting _SendItem onto multiprocessing.Queue.
    Uses token-bucket pacing to maintain the target rate.
    """
    interval = 1.0 / rate
    req_counter = 0
    while not stop_event.is_set():
        request_id = f"r{req_counter}"
        src_ids = list(range(req_counter * num_blocks, (req_counter + 1) * num_blocks))
        dst_ids = src_ids.copy()

        await send_q.put((request_id, 0, src_ids, dst_ids, 0, 0))
        req_counter += 1

        await asyncio.sleep(interval)


async def sender_task(client: StressDealerClient, send_q: asyncio.Queue,
                      pending: dict, pending_lock: asyncio.Lock,
                      stats: dict, stop_event: asyncio.Event):
    """Drain the send queue and send requests on the DEALER socket.

    Mirrors connector's _process_zmq / sender() coroutine — a single coroutine
    that loops: get from queue → socket.send(). This is the ONLY coroutine
    that calls client.send_request, ensuring thread safety on the zmq socket.
    """
    while not stop_event.is_set():
        try:
            item = await asyncio.wait_for(send_q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        request_id, cluster_id, src_ids, dst_ids, rank_id, src_dp_rank = item

        # Register in pending BEFORE send. mock_ox processes requests
        # extremely fast; if we register after send, the receiver can
        # get the response before the sender registers the request_id,
        # causing a false "UNEXPECTED" error.
        async with pending_lock:
            pending[request_id] = time.monotonic()

        # Async send — yields control back to the event loop, allowing the
        # receiver to run between sends.  Both send and recv are native
        # coroutines on the same zmq.asyncio socket, so they are safe.
        ok = await client.send_request(
            request_id=request_id,
            cluster_id=cluster_id,
            src_ids=src_ids,
            dst_ids=dst_ids,
            rank_id=rank_id,
            src_dp_rank=src_dp_rank,
        )

        if ok:
            stats["sent"] += 1
            if stats["sent"] % 100 == 0:
                ut_log(f"[SENDER] total sent: {stats['sent']} "
                       f"queue_len: {send_q.qsize()}")
        else:
            # Send failed — remove from pending so timeout won't fire
            async with pending_lock:
                pending.pop(request_id, None)
            ut_log(f"[SENDER] SEND FAILED for {request_id} - skipping, not counted")


async def receiver_task(client: StressDealerClient,
                        pending: dict, pending_lock: asyncio.Lock,
                        stats: dict, stop_event: asyncio.Event,
                        timeout_ms: int):
    """Continuously receive responses and verify against pending requests.

    Mirrors connector's _process_zmq / receiver() coroutine — a single
    coroutine that loops: socket.recv() → process. This is the ONLY
    coroutine that calls client.receive_response.
    """
    while not stop_event.is_set():
        try:
            resp = await client.receive_response(1000)
        except Exception as e:
            ut_log(f"[RECEIVER] Exception: {e}")
            continue

        if resp is None:
            # Poll timeout (1s) — check for stale pending requests
            async with pending_lock:
                now = time.monotonic()
                stale = {rid: t for rid, t in pending.items()
                         if now - t > timeout_ms / 1000.0}
                if stale:
                    ut_log(f"[RECEIVER] *** TIMEOUT *** {len(stale)} pending requests "
                           f"exceeded {timeout_ms}ms without response!")
                    for rid in list(stale.keys())[:5]:
                        ut_log(f"  Missing: {rid} (waiting {now - stale[rid]:.1f}s)")
                    if len(stale) > 5:
                        ut_log(f"  ... and {len(stale) - 5} more")
                    ut_log("[RECEIVER] *** ZMQ MESSAGE LOSS DETECTED ***")
                    stop_event.set()
                    return
            continue

        req_id = resp.get("request_id", "")
        success = resp.get("success", False)

        async with pending_lock:
            if req_id in pending:
                latency = time.monotonic() - pending.pop(req_id)
                stats["recv"] += 1
                if stats["recv"] % 100 == 0:
                    ut_log(f"[RECEIVER] total recv: {stats['recv']} "
                           f"pending: {len(pending)} latency: {latency*1000:.1f}ms")
            else:
                ut_log(f"[RECEIVER] *** UNEXPECTED *** response for unknown req_id={req_id}")
                stop_event.set()
                return

        if not success:
            ut_log(f"[RECEIVER] *** FAILURE *** response for req_id={req_id} success=False")
            stop_event.set()
            return


async def periodic_stats(pending: dict, pending_lock: asyncio.Lock,
                         stats: dict, send_q: asyncio.Queue,
                         stop_event: asyncio.Event):
    """Print periodic stats with send/recv rates."""
    prev_sent = 0
    prev_recv = 0
    prev_time = time.monotonic()
    while not stop_event.is_set():
        await asyncio.sleep(5)
        now = time.monotonic()
        elapsed = now - prev_time
        cur_sent = stats['sent']
        cur_recv = stats['recv']
        send_rate = (cur_sent - prev_sent) / elapsed if elapsed > 0 else 0
        recv_rate = (cur_recv - prev_recv) / elapsed if elapsed > 0 else 0
        async with pending_lock:
            n_pending = len(pending)
        ut_log(f"[STATS] sent={cur_sent} recv={cur_recv} "
               f"diff={cur_sent - cur_recv} "
               f"pending={n_pending} queue={send_q.qsize()} "
               f"send_rate={send_rate:.0f}/s recv_rate={recv_rate:.0f}/s")
        prev_sent = cur_sent
        prev_recv = cur_recv
        prev_time = now


async def run_stress_test(zmq_port: int, rate: float, num_blocks: int,
                          num_ox_threads: int, ox_binary: str, log_dir: str,
                          recv_timeout_ms: int):
    """Main stress test coroutine."""
    global _ut_log_file

    # Set up UT log
    os.makedirs(log_dir, exist_ok=True)
    ut_log_path = os.path.join(log_dir, "ut_stress.log")
    _ut_log_file = open(ut_log_path, "w", buffering=1)
    ut_log(f"UT log: {ut_log_path}")
    ut_log(f"OX log: {os.path.join(log_dir, 'ox_log_d_client.log')}")

    # Launch mock_ox
    ut_log(f"Launching mock_ox: {ox_binary} --zmq-port {zmq_port} --num-threads {num_ox_threads}")
    proc = launch_mock_ox(ox_binary, zmq_port, num_ox_threads, log_dir)

    # Wait for mock_ox to start up
    await asyncio.sleep(2)

    # Check if process is still alive
    if proc.poll() is not None:
        ut_log(f"ERROR: mock_ox exited prematurely with code {proc.returncode}")
        return 1

    # Create DEALER client
    server_address = f"tcp://localhost:{zmq_port}"
    client = StressDealerClient(server_address)

    # Shared state
    pending = {}                # request_id -> send timestamp (only for successfully sent)
    pending_lock = asyncio.Lock()
    stats = {"sent": 0, "recv": 0}
    stop_event = asyncio.Event()

    # Send queue — unbounded, matching connector's multiprocessing.Queue()
    send_q: asyncio.Queue = asyncio.Queue()

    ut_log(f"Starting stress test: rate={rate}/s num_blocks={num_blocks}")
    ut_log(f"Architecture: 1 producer ({rate}/s) → queue → 1 sender → DEALER → "
           f"mock_ox ROUTER → 1 receiver")

    # Launch tasks
    tasks = []

    # 1 receiver task (single, matching connector)
    tasks.append(asyncio.create_task(
        receiver_task(client, pending, pending_lock, stats, stop_event, recv_timeout_ms)))

    # 1 sender task (single, matching connector)
    tasks.append(asyncio.create_task(
        sender_task(client, send_q, pending, pending_lock, stats, stop_event)))

    # 1 producer task (fixed rate, matching connector's single-threaded KVLoader)
    tasks.append(asyncio.create_task(
        producer_task(send_q, rate, num_blocks, stop_event)))

    # Periodic stats
    tasks.append(asyncio.create_task(
        periodic_stats(pending, pending_lock, stats, send_q, stop_event)))

    # Monitor ox process health
    async def monitor_ox():
        while not stop_event.is_set():
            await asyncio.sleep(1)
            if proc.poll() is not None:
                ut_log(f"ERROR: mock_ox process died with code {proc.returncode}")
                stop_event.set()
                return

    tasks.append(asyncio.create_task(monitor_ox()))

    # Wait for stop signal
    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        ut_log("Interrupted by user")

    ut_log("Stopping test...")
    stop_event.set()

    # Give tasks a moment to finish
    await asyncio.sleep(0.5)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Final stats
    ut_log(f"Final stats: sent={stats['sent']} recv={stats['recv']} "
           f"diff={stats['sent'] - stats['recv']}")

    if stats["sent"] != stats["recv"]:
        ut_log("RESULT: FAIL - message loss detected!")
        result = 1
    else:
        ut_log("RESULT: PASS - no message loss")
        result = 0

    # Cleanup
    client.close()
    stop_mock_ox(proc)

    if _ut_log_file:
        _ut_log_file.close()

    return result


def main():
    parser = argparse.ArgumentParser(description="ZMQ stress test for mock_ox")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE,
                        help=f"Request enqueue rate in req/s (default: {DEFAULT_RATE})")
    parser.add_argument("--zmq-port", type=int, default=DEFAULT_ZMQ_PORT,
                        help=f"ZMQ port for mock_ox ROUTER (default: {DEFAULT_ZMQ_PORT})")
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS,
                        help=f"Number of block IDs per request (default: {DEFAULT_NUM_BLOCKS})")
    parser.add_argument("--num-ox-threads", type=int, default=DEFAULT_NUM_OX_THREADS,
                        help=f"Number of IO threads for mock_ox (default: {DEFAULT_NUM_OX_THREADS})")
    parser.add_argument("--ox-binary", type=str, default=None,
                        help="Path to mock_ox binary (default: same dir as this script)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files (default: /tmp/zmq_stress_<timestamp>)")
    parser.add_argument("--recv-timeout", type=int, default=DEFAULT_RECV_TIMEOUT_MS,
                        help=f"Receive timeout in ms before declaring loss (default: {DEFAULT_RECV_TIMEOUT_MS})")

    args = parser.parse_args()

    # Default ox binary path: original ox backend directory (where mock_ox is built)
    if args.ox_binary is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # mock_ox lives in the ox backend directory, not alongside this test script
        ox_backend_dir = os.path.join(script_dir, "..", "..", "omni_cache", "connector", "backends", "ox")
        args.ox_binary = os.path.join(os.path.normpath(ox_backend_dir), "mock_ox")

    if not os.path.isfile(args.ox_binary):
        print(f"ERROR: mock_ox binary not found at {args.ox_binary}")
        print("Build it first: make -f Makefile.mock")
        sys.exit(1)

    if args.log_dir is None:
        args.log_dir = f"/tmp/zmq_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    rc = asyncio.run(run_stress_test(
        zmq_port=args.zmq_port,
        rate=args.rate,
        num_blocks=args.num_blocks,
        num_ox_threads=args.num_ox_threads,
        ox_binary=args.ox_binary,
        log_dir=args.log_dir,
        recv_timeout_ms=args.recv_timeout,
    ))
    sys.exit(rc)


if __name__ == "__main__":
    main()