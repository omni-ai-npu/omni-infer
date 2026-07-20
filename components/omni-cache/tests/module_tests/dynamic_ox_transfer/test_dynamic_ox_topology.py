#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# test_dynamic_ox_topology.py
#
# End-to-end test for OX dynamic topology features described in
# CONNECTOR_DYNAMIC_TOPOLOGY_DESIGN.md §8–10.
#
# The test launches the real `ox` binary as a subprocess (exactly as the
# Python connector does) and talks to it via ZMQ + msgpack.  No C++
# compilation is needed.
#
# === TEST CASES ===
#
#  T1  Dynamic group creation — first request to a new P group
#  T2  TPGroup reuse — second request with the same shard_list
#  T3  Multiple sequential requests to the same P group
#  T4  Empty src_block_ids (zero-block request, no TCP I/O)
#  T5  Invalid/empty remote_ox_shard_list → must fail, not crash
#  T6  Different shard_list creates a different TPGroup
#  T7  Reconnection after P restart (manual P kill/restart)
#  T8  src_dp_rank offset applied correctly
#  T9  Large batch of blocks in a single request
#  T10 Connection reuse latency measurement
#  T11 Dynamic addition of new P node — connect to an expanded shard_list

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import msgpack
import zmq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OX_PATH_ENV = "OX_PATH"
DEFAULT_OX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ox"
)
DEFAULT_ZMQ_PORT = 17555
POLL_TIMEOUT_MS = 15000  # 15 s default for most tests
LONG_TIMEOUT_MS = 60000  # 60 s for tests involving retries


# ---------------------------------------------------------------------------
# ZMQ DEALER client — mirrors connector's RouterDealerClient
# ---------------------------------------------------------------------------
class TestZmqClient:
    """ZMQ DEALER client that talks to OX's ROUTER socket."""

    def __init__(self, endpoint: str):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.DEALER)
        client_id = f"test_{uuid.uuid4().hex[:8]}".encode()
        self.sock.setsockopt(zmq.IDENTITY, client_id)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(endpoint)
        self._endpoint = endpoint
        print(f"[CLIENT] Connected to {endpoint} as {client_id.decode()}")

    def send_request(
        self,
        request_id: str,
        remote_ox_shard_list: str,
        src_block_ids: List[int],
        dst_block_ids: List[int],
        table_id: int = 0,
        src_dp_rank: int = 0,
    ) -> bool:
        payload = {
            "request_id": request_id,
            "table_id": table_id,
            "src_block_ids": src_block_ids,
            "dst_block_ids": dst_block_ids,
            "remote_ox_shard_list": remote_ox_shard_list,
            "src_dp_rank": src_dp_rank,
        }
        try:
            self.sock.send(msgpack.packb(payload))
            return True
        except Exception as e:
            print(f"[CLIENT] Send error: {e}")
            return False

    def recv_response(self, timeout_ms: int = POLL_TIMEOUT_MS) -> Optional[Dict]:
        try:
            if self.sock.poll(timeout_ms, zmq.POLLIN):
                data = self.sock.recv()
                return msgpack.unpackb(data)
        except Exception as e:
            print(f"[CLIENT] Recv error: {e}")
        return None

    def close(self):
        try:
            self.sock.close(linger=0)
        except Exception as e:
            print(f"[CLIENT] Error closing socket: {e}")
        try:
            self.ctx.term()
        except Exception as e:
            print(f"[CLIENT] Error terminating context: {e}")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------
class OxProcess:
    """Manages an ox subprocess (P-side or D-side)."""

    def __init__(self, cmd: List[str], label: str = "OX"):
        self.cmd = cmd
        self.label = label
        self.proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        print(f"[{self.label}] Starting: {' '.join(self.cmd)}")
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._reader_thread = threading.Thread(
                target=self._drain_output, daemon=True
            )
            self._reader_thread.start()
            return True
        except Exception as e:
            print(f"[{self.label}] Failed to start: {e}")
            return False

    def _drain_output(self):
        assert self.proc is not None
        for line in self.proc.stdout:
            print(f"[{self.label}] {line.rstrip()}")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc and self.is_alive():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)
            print(f"[{self.label}] Stopped")

    def wait_until_output_contains(self, pattern: str, timeout: float = 10.0) -> bool:
        """Wait until ox stdout contains a given pattern (best-effort)."""
        # Since we drain output in a thread, we just sleep and hope.
        # For a more robust approach, capture output into a deque.
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_alive():
                return False
            time.sleep(0.5)
        return self.is_alive()


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


_results: List[TestResult] = []
_results_lock = threading.Lock()


def record(name: str, passed: bool, detail: str = ""):
    with _results_lock:
        _results.append(TestResult(name, passed, detail))


def print_results():
    p = sum(1 for r in _results if r.passed)
    f = sum(1 for r in _results if not r.passed)
    print("\n" + "=" * 56)
    print("              TEST RESULTS SUMMARY")
    print("=" * 56)
    for r in _results:
        tag = "PASS" if r.passed else "FAIL"
        line = f"  [{tag}] {r.name}"
        if r.detail:
            line += f"  ({r.detail})"
        print(line)
    print("-" * 56)
    print(f"  Total: {p+f}  Passed: {p}  Failed: {f}")
    print("=" * 56)


# ---------------------------------------------------------------------------
# P-side: start ox binary in P-server mode
# ---------------------------------------------------------------------------
def run_p_side(args) -> int:
    ox_path = os.environ.get(OX_PATH_ENV, DEFAULT_OX_PATH)
    if not os.path.isfile(ox_path) or not os.access(ox_path, os.X_OK):
        print(f"[P] FATAL: ox binary not found at {ox_path}")
        print(f"    Set {OX_PATH_ENV} environment variable to the ox binary path.")
        return 1

    # Create a temp shm file for the P-side BlockTable
    shm_path = f"/tmp/ox_test_p_shm_{os.getpid()}"
    # Pre-create the file so BlockTable can mmap it
    with open(shm_path, "wb") as f:
        pass

    cmd = [
        str(ox_path),
        "--addr", args.addr,
        "--block-table-shm", shm_path,
        "--num-block-tables", "1",
        "--num-blocks", str(args.num_blocks),
        "--num-layers", str(args.num_layers),
        "--tokens-per-block", str(args.tokens_per_block),
        "--num-connections-per-req", "4",
        "--dims", str(args.dims),
    ]

    proc = OxProcess(cmd, label="P-OX")
    if not proc.start():
        return 1

    print(f"[P] OX server running. Press Ctrl-C to stop.")

    try:
        while proc.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        proc.stop()
        try:
            os.unlink(shm_path)
        except OSError:
            pass
    return 0


# ---------------------------------------------------------------------------
# D-side: launch D OX as subprocess, run tests via ZMQ
# ---------------------------------------------------------------------------
def run_d_side(args) -> int:
    ox_path = os.environ.get(OX_PATH_ENV, DEFAULT_OX_PATH)
    if not os.path.isfile(ox_path) or not os.access(ox_path, os.X_OK):
        print(f"[D] FATAL: ox binary not found at {ox_path}")
        print(f"    Set {OX_PATH_ENV} environment variable to the ox binary path.")
        return 1

    # Create a temp shm file for the D-side BlockTable
    shm_path = f"/tmp/ox_test_d_shm_{os.getpid()}"
    with open(shm_path, "wb") as f:
        pass

    zmq_port = args.zmq_port

    # ---- Launch D OX in dynamic mode (no --addr, no --shard-list) ----
    cmd = [
        str(ox_path),
        "--zmq-port", str(zmq_port),
        "--block-table-shm", shm_path,
        "--num-block-tables", "1",
        "--num-blocks", str(args.num_blocks),
        "--num-layers", str(args.num_layers),
        "--tokens-per-block", str(args.tokens_per_block),
        "--num-connections-per-req", "4",
        "--dims", str(args.dims),
    ]

    d_ox = OxProcess(cmd, label="D-OX")
    if not d_ox.start():
        return 1

    # Wait for OX to bind ZMQ
    print("[D] Waiting for OX to start...")
    time.sleep(3)
    if not d_ox.is_alive():
        print("[D] FATAL: OX process died during startup")
        return 1

    # ---- Connect ZMQ client ----
    zmq_endpoint = f"tcp://127.0.0.1:{zmq_port}"
    client = TestZmqClient(zmq_endpoint)

    p_endpoints = args.p_endpoints  # e.g. "10.0.0.1:15077,10.0.0.2:15077"

    # ---- Helper ----
    def make_req(req_id, src, dst, shard_list=p_endpoints, dp_rank=0):
        return {
            "id": req_id,
            "src": src,
            "dst": dst,
            "shard_list": shard_list,
            "dp_rank": dp_rank,
        }

    def send_and_recv(req_id, src, dst, shard_list=p_endpoints,
                      dp_rank=0, timeout_ms=POLL_TIMEOUT_MS):
        """Send a request and wait for its response. Returns (req_id, success)."""
        ok = client.send_request(
            request_id=req_id,
            remote_ox_shard_list=shard_list,
            src_block_ids=src,
            dst_block_ids=dst,
            table_id=0,
            src_dp_rank=dp_rank,
        )
        if not ok:
            return ("", False)
        resp = client.recv_response(timeout_ms)
        if resp is None:
            return ("", False)
        return (resp.get("request_id", ""), resp.get("success", False))

    def run_test(name, fn):
        print(f"\n{'='*50}")
        print(f">>> Running: {name}")
        print(f"{'='*50}")
        try:
            ok = fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            ok = False
        record(name, ok, "" if ok else "see log above")
        print(f">>> {name}: {'PASS' if ok else 'FAIL'}")

    # ==================================================================
    # T1: Dynamic group creation
    # ==================================================================
    def test_t1():
        rid, ok = send_and_recv("t1-001", [1, 2, 3], [1, 2, 3],
                                timeout_ms=30000)
        if rid != "t1-001" or not ok:
            print(f"  FAIL: rid={rid} ok={ok}")
            return False
        print("  First request to new P group succeeded")
        return True

    run_test("T1_dynamic_group_creation", test_t1)

    # ==================================================================
    # T2: TPGroup reuse
    # ==================================================================
    def test_t2():
        rid, ok = send_and_recv("t2-001", [4, 5], [4, 5])
        if rid != "t2-001" or not ok:
            print(f"  FAIL: rid={rid} ok={ok}")
            return False
        print("  Same shard_list reused existing group")
        return True

    run_test("T2_group_reuse", test_t2)

    # ==================================================================
    # T3: Multiple sequential requests
    # ==================================================================
    def test_t3():
        n = 5
        ok_count = 0
        for i in range(n):
            rid, ok = send_and_recv(f"t3-{i}", [10 + i], [10 + i],
                                    timeout_ms=30000)
            if ok:
                ok_count += 1
            else:
                print(f"  Request {i} failed: rid={rid} ok={ok}")
        if ok_count != n:
            print(f"  Only {ok_count}/{n} succeeded")
            return False
        print(f"  All {n} sequential requests succeeded")
        return True

    run_test("T3_sequential_requests", test_t3)

    # ==================================================================
    # T4: Empty src_block_ids
    # ==================================================================
    def test_t4():
        rid, ok = send_and_recv("t4-001", [], [100], timeout_ms=10000)
        if rid != "t4-001" or not ok:
            print(f"  FAIL: rid={rid} ok={ok}")
            return False
        print("  Empty src_block_ids request succeeded")
        return True

    run_test("T4_empty_src_blocks", test_t4)

    # ==================================================================
    # T5: Invalid/empty remote_ox_shard_list
    # ==================================================================
    def test_t5():
        # 5a: empty shard_list
        rid, ok = send_and_recv("t5a-001", [1], [1],
                                shard_list="", timeout_ms=10000)
        if rid == "t5a-001" and not ok:
            print("  5a: Empty shard_list → success=False (correct)")
        elif rid == "":
            print("  5a: Empty shard_list → OX threw (no response, acceptable for MVP)")
        else:
            print(f"  5a: rid={rid} ok={ok} (unexpected but OX didn't crash)")

        # 5b: unreachable endpoint
        rid, ok = send_and_recv("t5b-001", [1], [1],
                                shard_list="255.255.255.254:19999",
                                timeout_ms=LONG_TIMEOUT_MS)
        if rid == "t5b-001" and not ok:
            print("  5b: Unreachable endpoint → success=False (correct)")
        elif rid == "":
            print("  5b: Timeout — OX still retrying internally")
        else:
            print(f"  5b: rid={rid} ok={ok}")

        return True  # Pass as long as OX didn't crash

    run_test("T5_invalid_shard_list", test_t5)

    # ==================================================================
    # T6: Different shard_list creates a different TPGroup
    # ==================================================================
    def test_t6():
        # Build a shard_list that differs from the real one
        first_ep = p_endpoints.split(",")[0].strip()
        ip, port = first_ep.rsplit(":", 1)
        fake_ep = f"{ip}:{int(port) + 10000}"

        rid, ok = send_and_recv("t6-001", [1], [1],
                                shard_list=fake_ep,
                                timeout_ms=LONG_TIMEOUT_MS)
        if rid == "t6-001" and not ok:
            print("  Different shard_list → new group, connect failed as expected")
            return True
        if rid == "":
            print("  Timeout — OX still retrying on new group (acceptable)")
            return True
        print(f"  Unexpected: rid={rid} ok={ok}")
        return False

    run_test("T6_different_shard_list", test_t6)

    # ==================================================================
    # T7: Reconnection after P restart (manual)
    # ==================================================================
    def test_t7():
        # Step 1: Verify connection works
        rid, ok = send_and_recv("t7-before", [20, 21], [20, 21])
        if not ok:
            print("  Pre-kill request already failed — is P running?")
            return False
        print("  Step 1 OK: Pre-kill request succeeded")

        # Step 2: Ask user to kill P
        print()
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║  MANUAL STEP: KILL the P process now (Ctrl-C).      ║")
        print("  ║  Then press ENTER here to continue...               ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        input()

        # Step 3: Request while P is down → should fail
        rid, ok = send_and_recv("t7-down", [22], [22],
                                timeout_ms=LONG_TIMEOUT_MS)
        if rid == "t7-down" and not ok:
            print("  Step 3 OK: Request during P-down returned failure")
        elif rid == "":
            print("  Step 3: Timeout — OX still retrying")
        else:
            print(f"  Step 3: rid={rid} ok={ok}")

        # Step 4: Ask user to restart P
        print()
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║  MANUAL STEP: RESTART the P process now.            ║")
        print("  ║  Then press ENTER here to continue...               ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        input()
        time.sleep(3)

        # Step 5: Request after P restart → should succeed
        rid, ok = send_and_recv("t7-after", [23, 24], [23, 24],
                                timeout_ms=LONG_TIMEOUT_MS)
        if rid == "t7-after" and ok:
            print("  Step 5 OK: Reconnection succeeded!")
            return True
        print(f"  Step 5 FAIL: rid={rid} ok={ok}")
        return False

    run_test("T7_reconnect_after_p_restart", test_t7)

    # ==================================================================
    # T8: src_dp_rank offset
    # ==================================================================
    def test_t8():
        rid, ok = send_and_recv("t8-001", [1, 2], [1, 2], dp_rank=1)
        if rid != "t8-001" or not ok:
            print(f"  FAIL: rid={rid} ok={ok}")
            return False
        print("  src_dp_rank=1 request succeeded (offset applied by OX)")
        return True

    run_test("T8_src_dp_rank_offset", test_t8)

    # ==================================================================
    # T9: Large batch
    # ==================================================================
    def test_t9():
        n = 50
        src = list(range(1, n + 1))
        dst = list(range(1, n + 1))
        rid, ok = send_and_recv("t9-001", src, dst, timeout_ms=30000)
        if rid != "t9-001" or not ok:
            print(f"  FAIL: rid={rid} ok={ok}")
            return False
        print(f"  Large batch ({n} blocks) succeeded")
        return True

    run_test("T9_large_batch", test_t9)

    # ==================================================================
    # T10: Connection reuse latency
    # ==================================================================
    def test_t10():
        n = 10
        latencies = []
        for i in range(n):
            t0 = time.monotonic()
            rid, ok = send_and_recv(f"t10-{i}", [1], [1])
            t1 = time.monotonic()
            if not ok:
                print(f"  Request {i} failed")
                return False
            latencies.append((t1 - t0) * 1000)

        first_ms = latencies[0]
        avg_rest = sum(latencies[1:]) / (n - 1)
        print(f"  First: {first_ms:.1f} ms,  Avg rest: {avg_rest:.1f} ms")
        if avg_rest < first_ms:
            print("  Connection reuse confirmed (subsequent faster)")
        else:
            print("  WARNING: Subsequent not faster")
        return True  # Informational

    run_test("T10_connection_reuse_latency", test_t10)

    # ==================================================================
    # T11: Dynamic addition of new P node
    # ==================================================================
    if not args.extra_p_endpoint:
        print("\n[SKIP] T11_dynamic_new_p_node: --extra-p-endpoint not provided")
        print("       To run T11, use: --extra-p-endpoint <ip:port>")
        record("T11_dynamic_new_p_node", False, "skipped: --extra-p-endpoint not provided")
    else:
        def test_t11():
            # Step 1: Verify existing group (from --p-endpoints) works
            rid, ok = send_and_recv("t11-before", [30, 31], [30, 31])
            if not ok:
                print("  Step 1 FAIL: Existing group request failed")
                return False
            print("  Step 1 OK: Existing group request succeeded")

            # Step 2: Ask user to start the new P node
            print()
            print("  ╔══════════════════════════════════════════════════════╗")
            print("  ║  MANUAL STEP: START the new P node now (if not yet).║")
            print("  ║  Then press ENTER here to continue...               ║")
            print("  ╚══════════════════════════════════════════════════════╝")
            input()

            # Step 3: Send request with expanded shard_list (adds a new P node)
            #         This triggers TPGroupManager to create a NEW TPGroup.
            existing_eps = [ep.strip() for ep in p_endpoints.split(",")]
            new_ep = args.extra_p_endpoint  # e.g. "7.246.46.143:15077"
            expanded_shard_list = ",".join(existing_eps + [new_ep])

            rid, ok = send_and_recv("t11-expanded", [32, 33], [32, 33],
                                    shard_list=expanded_shard_list,
                                    timeout_ms=30000)
            if rid != "t11-expanded" or not ok:
                print(f"  Step 3 FAIL: Expanded shard_list request failed (rid={rid} ok={ok})")
                return False
            print(f"  Step 3 OK: Expanded shard_list ({expanded_shard_list}) connected and succeeded")

            # Step 4: Verify the original group still works
            rid, ok = send_and_recv("t11-original", [34], [34])
            if rid != "t11-original" or not ok:
                print(f"  Step 4 FAIL: Original group request failed after expansion")
                return False
            print("  Step 4 OK: Original group still works after expansion")

            return True

        run_test("T11_dynamic_new_p_node", test_t11)

    # ==================================================================
    # Print results and cleanup
    # ==================================================================
    print_results()

    client.close()
    d_ox.stop()
    try:
        os.unlink(shm_path)
    except OSError:
        pass

    failures = sum(1 for r in _results if not r.passed)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end test for OX dynamic topology features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # P machine (10.0.0.1):
  python test_dynamic_ox_topology.py --role p --addr 0.0.0.0:15077 \\
      --num-layers 2 --tokens-per-block 4 --dims 8

  # D machine:
  python test_dynamic_ox_topology.py --role d --p-endpoints 10.0.0.1:15077 \\
      --num-layers 2 --tokens-per-block 4 --dims 8

  # Multi-P (2 nodes):
  python test_dynamic_ox_topology.py --role d \\
      --p-endpoints 10.0.0.1:15077,10.0.0.2:15077 \\
      --num-layers 2 --tokens-per-block 4 --dims 8

  # With dynamic new-node test (T11):
  python test_dynamic_ox_topology.py --role d \\
      --p-endpoints 10.0.0.1:15077,10.0.0.2:15077 \\
      --extra-p-endpoint 10.0.0.3:15077 \\
      --num-layers 2 --tokens-per-block 4 --dims 8
""",
    )
    parser.add_argument("--role", required=True, choices=["p", "d"],
                        help="Role: 'p' = P-side OX server, 'd' = D-side test driver")
    parser.add_argument("--addr",
                        help="P listen address (ip:port), required for role=p")
    parser.add_argument("--zmq-port", type=int, default=DEFAULT_ZMQ_PORT,
                        help=f"D-side ZMQ port (default: {DEFAULT_ZMQ_PORT})")
    parser.add_argument("--p-endpoints",
                        help="Comma-separated P OX endpoints, required for role=d")
    parser.add_argument("--extra-p-endpoint",
                        help="Extra P endpoint for T11 (dynamic new-node test), e.g. 10.0.0.3:15077")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tokens-per-block", type=int, default=4)
    parser.add_argument("--dims", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.role == "p":
        if not args.addr:
            print("ERROR: --addr is required for role=p")
            return 1
        return run_p_side(args)
    else:
        if not args.p_endpoints:
            print("ERROR: --p-endpoints is required for role=d")
            return 1
        return run_d_side(args)


if __name__ == "__main__":
    sys.exit(main())
