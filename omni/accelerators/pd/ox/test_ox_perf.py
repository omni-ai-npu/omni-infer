#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
OX Transfer Performance Test

Measures KV transfer performance of the OX backend across machines.
Run on multiple machines: one (or more) as P (Provider/Prefill), one as D (Demander/Decode).

The P-side starts an OX TCP server; the D-side starts an OX ZMQ+TCP client,
then sends ZMQ pull requests and measures end-to-end latency / bandwidth.

Usage:
  # On P-side machine:
  python test_ox_perf.py --role P --p-addr 0.0.0.0:15077 [options]

  # On D-side machine:
  python test_ox_perf.py --role D --shard-list 10.0.0.1:15077 --zmq-port 17555 [options]

Examples:
  # P-side (single node)
  python test_ox_perf.py --role P --p-addr 0.0.0.0:15077

  # D-side — sequential mode (default)
  python test_ox_perf.py --role D --shard-list 10.0.0.1:15077 \\
      --num-requests 100 --blocks-per-request 10

  # D-side — pipeline mode (max throughput)
  python test_ox_perf.py --role D --shard-list 10.0.0.1:15077 \\
      --num-requests 200 --blocks-per-request 20 --pipeline

  # End-to-end data verification (enable on both P and D)
  python test_ox_perf.py --role P --p-addr 0.0.0.0:15077 --tokens-per-block 64 \\
      --blocks-per-request 2048 --verify-data --verify-tp-size 2 --verify-tp-rank 0
  python test_ox_perf.py --role P --p-addr 0.0.0.0:15078 --tokens-per-block 64 \\
      --blocks-per-request 2048 --verify-data --verify-tp-size 2 --verify-tp-rank 1
  python test_ox_perf.py --role D --shard-list 10.0.0.1:15077,10.0.0.2:15078 \\
      --tokens-per-block 128 --blocks-per-request 2048 --verify-data

  # D-side — multi-shard P cluster
  python test_ox_perf.py --role D --shard-list 10.0.0.1:15077,10.0.0.2:15077

Data verification is disabled by default. When --verify-data is enabled, P initializes
the source blocks with a deterministic pattern and D validates the destination blocks
byte-for-byte after the benchmark. P and D must use the same --blocks-per-request:
P initializes only source block IDs 0 through N-1, where N is that value.
"""

import argparse
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import msgpack
import zmq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_block_size(num_layers, tokens_per_block, dims, dtype):
    """Compute the byte size of a single KV block."""
    return num_layers * tokens_per_block * sum(dims) * dtype


def parse_tp_size(shard_list_str):
    """Parse tp_size from --shard-list, matching C++ Config::tp_size().

    Format: semicolons separate clusters, commas separate nodes within a cluster.
    tp_size = number of nodes in the first cluster.
    E.g. "10.0.0.1:15077,10.0.0.2:15077" -> tp_size=2
         "10.0.0.1:15077"                 -> tp_size=1
    """
    if not shard_list_str:
        return 1
    first_cluster = shard_list_str.split(";")[0]
    nodes = [n.strip() for n in first_cluster.split(",") if n.strip()]
    return max(1, len(nodes))


def create_shm_file(path, size):
    """Create (or overwrite) a file of *size* bytes for OX mmap."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.ftruncate(fd, size)
    finally:
        os.close(fd)

    print(f"[UT] Created shm file: {path}  "
          f"size={size} bytes ({size / (1024 ** 3):.3f} GiB)")


def _expected_slice_bytes(block_id, segment, layer, rank, size):
    """Return deterministic bytes for one D-side layerwise TP slice."""
    start = (block_id * 131 + segment * 47 + layer * 19 + rank * 73) & 0xFF
    one_cycle = bytes(range(start, 256)) + bytes(range(start))
    return (one_cycle * ((size + len(one_cycle) - 1) // len(one_cycle)))[:size]


def _layerwise_slices(block_ids, num_blocks, num_layers, tokens_per_block, dims, dtype, tp_size, rank):
    """Yield (file_offset, size, block_id, segment, layer) in OX layerwise-buffer order."""
    segment_offset = 0
    for segment, dim in enumerate(dims):
        block_layer_size = tokens_per_block * dim * dtype
        if block_layer_size % tp_size:
            raise ValueError(f"segment {segment} cannot be evenly split across tp_size={tp_size}")
        layer_size = num_blocks * block_layer_size
        slice_size = block_layer_size // tp_size
        for block_id in block_ids:
            for layer in range(num_layers):
                offset = (segment_offset + layer * layer_size + block_id * block_layer_size +
                          rank * slice_size)
                yield offset, slice_size, block_id, segment, layer
        segment_offset += num_layers * layer_size


def initialize_verification_blocks(args, block_ids):
    """Seed one P rank with the bytes D expects in that rank's layerwise slice."""
    block_ids = list(block_ids)
    with open(args.block_table_shm, "r+b", buffering=0) as shm:
        for offset, size, block_id, segment, layer in _layerwise_slices(
                block_ids, args.num_blocks, args.num_layers, args.tokens_per_block,
                args.dims, args.dtype, 1, 0):
            shm.seek(offset)
            shm.write(_expected_slice_bytes(
                block_id, segment, layer, args.verify_tp_rank, size))
    initialized_range = "none" if not block_ids else f"0..{len(block_ids) - 1}"
    print(f"[UT] Initialized P rank {args.verify_tp_rank}/{args.verify_tp_size} for data verification "
          f"on source blocks {initialized_range}.")


def verify_destination_blocks(args, block_ids, tp_size):
    """Check every D-side rank slice against the deterministic data from its P rank."""
    block_ids = list(block_ids)
    with open(args.block_table_shm, "rb", buffering=0) as shm:
        for rank in range(tp_size):
            for offset, size, block_id, segment, layer in _layerwise_slices(
                    block_ids, args.num_blocks, args.num_layers, args.tokens_per_block,
                    args.dims, args.dtype, tp_size, rank):
                shm.seek(offset)
                actual = shm.read(size)
                expected = _expected_slice_bytes(block_id, segment, layer, rank, size)
                if actual == expected:
                    continue
                if len(actual) != size:
                    raise RuntimeError(
                        f"data verification failed: block={block_id} rank={rank} expected {size} bytes, got {len(actual)}")
                mismatch = next(i for i, (got, want) in enumerate(zip(actual, expected)) if got != want)
                raise RuntimeError(
                    "data verification failed: "
                    f"block={block_id} segment={segment} layer={layer} rank={rank} offset={mismatch} "
                    f"expected=0x{expected[mismatch]:02x} actual=0x{actual[mismatch]:02x}")
    print(f"[UT] Data verification passed: {len(block_ids)} blocks across {tp_size} P ranks match.")


def wait_for_port(host, port, timeout=600):
    """Block until *host:port* accepts TCP connections or *timeout* expires."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((host, port))
            s.close()
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# OX subprocess management
# ---------------------------------------------------------------------------

class OXProcess:
    """Manage an OX subprocess with stdout logging."""

    def __init__(self, cmd, log_path=None):
        print(f"[UT] Starting OX: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self._log_file = open(log_path, "a")

        self._stdout_q: queue.Queue = queue.Queue()

        def _reader():
            for line in self.proc.stdout:
                self._stdout_q.put(line)
                if self._log_file:
                    self._log_file.write(line)
                    self._log_file.flush()

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    @property
    def returncode(self):
        return self.proc.returncode

    def is_alive(self):
        return self.proc.poll() is None

    def stop(self, timeout=10):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self._log_file:
            self._log_file.close()
        print("[UT] OX process stopped.")


# ---------------------------------------------------------------------------
# P-side
# ---------------------------------------------------------------------------

def run_p_side(args):
    block_size = compute_block_size(args.num_layers, args.tokens_per_block,
                                    args.dims, args.dtype)
    shm_size = args.num_blocks * block_size
    create_shm_file(args.block_table_shm, shm_size)
    if args.verify_data:
        initialize_verification_blocks(args, range(min(args.blocks_per_request, args.num_blocks)))

    # Parse bind address
    bind_ip, bind_port = args.p_addr.rsplit(":", 1)
    bind_port = int(bind_port)

    cmd = [
        str(args.ox_path),
        "--addr", args.p_addr,
        "--block-table-shm", args.block_table_shm,
        "--num-blocks", str(args.num_blocks),
        "--num-layers", str(args.num_layers),
        "--tokens-per-block", str(args.tokens_per_block),
        "--dims", ",".join(map(str, args.dims)),
        "--dtype", str(args.dtype),
        "--num-threads", str(args.num_threads),
    ]

    log_path = os.path.join(args.log_dir, "ox_perf_p.log")
    ox = OXProcess(cmd, log_path)

    print(f"[UT] Waiting for P-side OX on {bind_ip}:{bind_port} ...")
    if not wait_for_port(bind_ip, bind_port, timeout=args.wait_timeout):
        ox.stop()
        print("[UT] ERROR: P-side OX failed to start within timeout.", file=sys.stderr)
        sys.exit(1)

    print(f"[UT] P-side OX ready on {bind_ip}:{bind_port}.")
    print("[UT] P-side is running. Press Ctrl+C to stop.")

    try:
        ox.proc.wait()
    except KeyboardInterrupt:
        print("\n[UT] Caught Ctrl+C, stopping P-side OX ...")
    finally:
        ox.stop()


# ---------------------------------------------------------------------------
# D-side
# ---------------------------------------------------------------------------

def run_d_side(args):
    block_size = compute_block_size(args.num_layers, args.tokens_per_block,
                                    args.dims, args.dtype)
    tp_size = parse_tp_size(args.shard_list)
    # Actual bytes transferred per block over the wire = block_size / tp_size,
    # matching C++ bt.block_tp_size() used in global_stats_update().
    block_tp_size = block_size // tp_size

    print(f"[UT] tp_size={tp_size}, block_size={block_size} bytes, "
          f"block_tp_size={block_tp_size} bytes")

    shm_size = args.num_block_tables * args.num_blocks * block_size
    create_shm_file(args.block_table_shm, shm_size)

    cmd = [
        str(args.ox_path),
        "--shard-list", args.shard_list,
        "--zmq-port", str(args.zmq_port),
        "--block-table-shm", args.block_table_shm,
        "--num-block-tables", str(args.num_block_tables),
        "--num-blocks", str(args.num_blocks),
        "--num-layers", str(args.num_layers),
        "--tokens-per-block", str(args.tokens_per_block),
        "--num-connections-per-req", str(args.num_connections_per_req),
        "--num-connections", str(args.num_connections),
        "--dims", ",".join(map(str, args.dims)),
        "--dtype", str(args.dtype),
        "--num-threads", str(args.num_threads),
    ]

    log_path = os.path.join(args.log_dir, "ox_perf_d.log")
    ox = OXProcess(cmd, log_path)

    # Give OX time to establish TCP connections to P-side.
    # The C++ code retries every 5 s with 3600 s overall timeout.
    print(f"[UT] Waiting {args.connect_wait}s for D-side OX to connect to P-side ...")
    time.sleep(args.connect_wait)

    try:
        if not ox.is_alive():
            print("[UT] ERROR: D-side OX process exited prematurely.", file=sys.stderr)
            sys.exit(1)
        run_benchmark(args, block_tp_size, tp_size)
        if args.verify_data:
            verify_destination_blocks(
                args, range(min(args.blocks_per_request, args.num_blocks)), tp_size)
    finally:
        ox.stop()


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def _build_request(request_id, src_ids, dst_ids):
    return msgpack.packb({
        "request_id": request_id,
        "table_id": 0,
        "src_block_ids": src_ids,
        "dst_block_ids": dst_ids,
        "cluster_id": 0,
        "src_dp_rank": 0,
    })


def _warmup(socket, num_requests, blocks_per_request, num_blocks):
    """Send a few requests to warm up TCP connections."""
    if num_requests <= 0:
        return
    print(f"[UT] Warming up ({num_requests} requests) ...")
    src_ids = list(range(min(blocks_per_request, num_blocks)))
    dst_ids = list(range(min(blocks_per_request, num_blocks)))

    for i in range(num_requests):
        data = _build_request(f"warmup_{i}", src_ids, dst_ids)
        socket.send(data)

    # Collect warmup responses
    for i in range(num_requests):
        if socket.poll(60_000, zmq.POLLIN):
            socket.recv()
        else:
            print(f"[UT] WARNING: warmup request {i} timed out (60 s)")

    print("[UT] Warmup complete.")


def _run_sequential(socket, num_requests, blocks_per_request, num_blocks):
    """Send one request at a time; measure per-request latency."""
    latencies = []
    src_ids = list(range(min(blocks_per_request, num_blocks)))
    dst_ids = list(range(min(blocks_per_request, num_blocks)))

    for i in range(num_requests):
        data = _build_request(f"seq_{i}", src_ids, dst_ids)
        t_send = time.perf_counter()
        socket.send(data)

        if socket.poll(120_000, zmq.POLLIN):
            socket.recv()
            t_recv = time.perf_counter()
            latencies.append(t_recv - t_send)
        else:
            print(f"[UT] ERROR: sequential request {i} timed out (120 s)!")
            break

        # Progress every 10 %
        step = max(1, num_requests // 10)
        if (i + 1) % step == 0:
            print(f"  [{i + 1}/{num_requests}]  "
                  f"latency={latencies[-1] * 1000:.3f} ms")

    return latencies


def _run_pipeline(socket, num_requests, blocks_per_request, num_blocks):
    """Fire all requests, then collect all responses; measure per-request latency
    and overall throughput."""
    src_ids = list(range(min(blocks_per_request, num_blocks)))
    dst_ids = list(range(min(blocks_per_request, num_blocks)))

    send_times = {}

    # --- send phase ---
    t_batch_start = time.perf_counter()
    for i in range(num_requests):
        rid = f"pipe_{i}"
        data = _build_request(rid, src_ids, dst_ids)
        send_times[rid] = time.perf_counter()
        socket.send(data)
    t_send_done = time.perf_counter()

    # --- recv phase ---
    recv_times = {}
    for i in range(num_requests):
        if socket.poll(120_000, zmq.POLLIN):
            raw = socket.recv()
            resp = msgpack.unpackb(raw)
            rid = resp.get("request_id", f"pipe_{i}")
            recv_times[rid] = time.perf_counter()
        else:
            print(f"[UT] ERROR: pipeline request {i} timed out (120 s)!")
            break

    t_batch_end = time.perf_counter()

    # Build latency list matched by request_id
    latencies = []
    for rid, t_s in send_times.items():
        if rid in recv_times:
            latencies.append(recv_times[rid] - t_s)

    return latencies, t_batch_start, t_batch_end


def _print_results(latencies, block_tp_size, blocks_per_request, mode,
                   num_requests, tp_size, batch_start=None, batch_end=None):
    """Print a summary table.

    block_tp_size: bytes actually transferred per block over the wire
                   (= block_size / tp_size), matching C++ bt.block_tp_size().
    """
    if not latencies:
        print("[UT] No latency data collected.")
        return

    total_data = len(latencies) * blocks_per_request * block_tp_size
    sum_lat = sum(latencies)

    sorted_lat = sorted(latencies)
    avg_lat = sum_lat / len(latencies)
    min_lat = sorted_lat[0]
    max_lat = sorted_lat[-1]
    med_lat = sorted_lat[len(sorted_lat) // 2]
    p90_lat = sorted_lat[int(len(sorted_lat) * 0.90)]
    p99_lat = sorted_lat[int(len(sorted_lat) * 0.99)]

    # Throughput based on sum of individual latencies (sequential wall-clock)
    bw_gbit = (total_data * 8) / (sum_lat * 1e9)
    bw_gib = total_data / (sum_lat * (1024 ** 3))

    print()
    print("=" * 64)
    print(f"  OX Transfer Performance  —  {mode} mode")
    print("=" * 64)
    print(f"  Requests completed:   {len(latencies)} / {num_requests}")
    print(f"  TP size:              {tp_size}")
    print(f"  Blocks / request:     {blocks_per_request}")
    print(f"  Block tp size:        {block_tp_size / (1024 ** 2):.3f} MiB  "
          f"(block_size / tp_size)")
    print(f"  Data / request:       {blocks_per_request * block_tp_size / (1024 ** 2):.3f} MiB")
    print(f"  Total data:           {total_data / (1024 ** 3):.3f} GiB")
    print("-" * 64)
    print(f"  Avg latency:          {avg_lat * 1000:.3f} ms")
    print(f"  Min latency:          {min_lat * 1000:.3f} ms")
    print(f"  Max latency:          {max_lat * 1000:.3f} ms")
    print(f"  Median latency:       {med_lat * 1000:.3f} ms")
    print(f"  P90 latency:          {p90_lat * 1000:.3f} ms")
    print(f"  P99 latency:          {p99_lat * 1000:.3f} ms")
    print("-" * 64)
    print(f"  Throughput (sum-lat): {bw_gbit:.3f} Gbit/s  |  {bw_gib:.3f} GiB/s")
    print(f"  Blocks/s (sum-lat):   {len(latencies) * blocks_per_request / sum_lat:.1f}")

    if batch_start is not None and batch_end is not None:
        wall = batch_end - batch_start
        bw_wall_gbit = (total_data * 8) / (wall * 1e9)
        bw_wall_gib = total_data / (wall * (1024 ** 3))
        print(f"  Wall-clock time:      {wall:.3f} s")
        print(f"  Throughput (wall):    {bw_wall_gbit:.3f} Gbit/s  |  {bw_wall_gib:.3f} GiB/s")
        print(f"  Blocks/s (wall):      {len(latencies) * blocks_per_request / wall:.1f}")

    print("=" * 64)


def run_benchmark(args, block_tp_size, tp_size):
    zmq_addr = f"tcp://127.0.0.1:{args.zmq_port}"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    client_id = f"perf_{uuid.uuid4().hex[:8]}".encode("utf-8")
    sock.setsockopt(zmq.IDENTITY, client_id)
    sock.connect(zmq_addr)
    # Allow some time for ZMQ connection handshake
    time.sleep(1)
    print(f"[UT] ZMQ DEALER connected to {zmq_addr}  (id={client_id.decode()})")

    # Warmup
    _warmup(sock, args.warmup_requests, args.blocks_per_request, args.num_blocks)

    # Benchmark
    print(f"[UT] Benchmark: {args.num_requests} requests × "
          f"{args.blocks_per_request} blocks/req  "
          f"(block_tp={block_tp_size / (1024 ** 2):.3f} MiB, "
          f"req={args.blocks_per_request * block_tp_size / (1024 ** 2):.3f} MiB)")

    if args.pipeline:
        latencies, t_start, t_end = _run_pipeline(
            sock, args.num_requests, args.blocks_per_request, args.num_blocks)
        _print_results(latencies, block_tp_size, args.blocks_per_request,
                       "pipeline", args.num_requests, tp_size, t_start, t_end)
    else:
        latencies = _run_sequential(
            sock, args.num_requests, args.blocks_per_request, args.num_blocks)
        _print_results(latencies, block_tp_size, args.blocks_per_request,
                       "sequential", args.num_requests, tp_size)

    sock.close()
    ctx.term()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="OX Transfer Performance Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- role ----
    p.add_argument("--role", required=True, choices=["P", "D"],
                   help="P = Provider/Prefill (TCP server); "
                        "D = Demander/Decode (ZMQ client + TCP client)")

    # ---- OX binary ----
    p.add_argument("--ox-path", default=None,
                   help="Path to the OX binary (default: <script_dir>/ox)")

    # ---- shared memory ----
    p.add_argument("--block-table-shm", default=None,
                   help="File path for the block-table mmap file "
                        "(default: /tmp/ox_perf_{p|d}_shm)")

    # ---- KV geometry (must match on both sides) ----
    p.add_argument("--num-blocks", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=62)
    p.add_argument("--tokens-per-block", type=int, default=128)
    p.add_argument("--dims", default="512,64,128",
                   help="Comma-separated KV dimension split (default: 512,64,128)")
    p.add_argument("--dtype", type=int, default=2,
                   help="Element size in bytes: 1=int8, 2=bfloat16 (default: 2)")

    # ---- threading / connections ----
    p.add_argument("--num-threads", type=int, default=16)
    p.add_argument("--num-connections", type=int, default=16,
                   help="TCP connections per shard (D-side only)")
    p.add_argument("--num-connections-per-req", type=int, default=8,
                   help="Max connections used per single request (D-side only)")

    # ---- P-side specific ----
    p.add_argument("--p-addr", default="0.0.0.0:15077",
                   help="P-side bind address (default: 0.0.0.0:15077)")

    # ---- D-side specific ----
    p.add_argument("--shard-list", default=None,
                   help="Shard list for D-side, e.g. 10.0.0.1:15077  "
                        "or 10.0.0.1:15077,10.0.0.2:15077 for multi-shard")
    p.add_argument("--zmq-port", type=int, default=17555)
    p.add_argument("--num-block-tables", type=int, default=1)

    # ---- benchmark parameters (D-side only) ----
    p.add_argument("--num-requests", type=int, default=100,
                   help="Number of benchmark requests (default: 100)")
    p.add_argument("--blocks-per-request", type=int, default=10,
                   help="Number of KV blocks per request (default: 10)")
    p.add_argument("--warmup-requests", type=int, default=5,
                   help="Number of warmup requests (default: 5)")
    p.add_argument("--pipeline", action="store_true",
                   help="Pipeline mode: fire all requests then collect responses "
                        "(measures max throughput)")
    p.add_argument("--verify-data", action="store_true",
                   help="Initialize deterministic P-side source data and verify D-side destination data "
                        "byte-for-byte after the benchmark. Enable on both P and D runs.")
    p.add_argument("--verify-tp-size", type=int, default=None,
                   help="Total P ranks used by --verify-data on the P side")
    p.add_argument("--verify-tp-rank", type=int, default=None,
                   help="This P process rank used by --verify-data, in [0, verify-tp-size)")

    # ---- other ----
    p.add_argument("--wait-timeout", type=int, default=600,
                   help="Seconds to wait for P-side OX to become ready (default: 600)")
    p.add_argument("--connect-wait", type=int, default=10,
                   help="Seconds to wait for D-side OX to connect to P-side "
                        "before benchmarking (default: 10)")
    p.add_argument("--log-dir", default="/data/ox_log",
                   help="Directory for OX log files (default: /data/ox_log)")

    args = p.parse_args()

    # Derived defaults
    args.dims = [int(d) for d in args.dims.split(",")]
    if args.ox_path is None:
        args.ox_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ox")
    if args.block_table_shm is None:
        args.block_table_shm = f"/tmp/ox_perf_{args.role.lower()}_shm"

    # Validate
    if not os.path.isfile(args.ox_path):
        p.error(f"OX binary not found: {args.ox_path}")
    if args.role == "D" and args.shard_list is None:
        p.error("--shard-list is required when --role=D")
    if args.blocks_per_request > args.num_blocks:
        p.error("--blocks-per-request cannot exceed --num-blocks")
    if args.verify_data and args.role == "P":
        if args.verify_tp_size is None or args.verify_tp_rank is None:
            p.error("P-side --verify-data requires --verify-tp-size and --verify-tp-rank")
        if args.verify_tp_size < 1 or not 0 <= args.verify_tp_rank < args.verify_tp_size:
            p.error("--verify-tp-rank must be in [0, --verify-tp-size)")

    # Dispatch
    if args.role == "P":
        run_p_side(args)
    else:
        run_d_side(args)


if __name__ == "__main__":
    main()
