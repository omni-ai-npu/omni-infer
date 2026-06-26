# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Test master worker failover.

This test verifies that when the master worker dies (simulating a coredump),
a new worker takes over as master and scheduling continues to work.
"""

import pytest
import os
import subprocess
import time
import requests
import signal
import re
from pathlib import Path

from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess
import port_manager

ENCODE_NUM = 1  # Need at least 1 encode to avoid mock_model path
PREFILL_NUM = 2
DECODE_NUM = 2


def get_nginx_workers():
    """Get list of nginx worker PIDs."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nginx: worker"],
            capture_output=True,
            text=True
        )
        pids = [int(line) for line in result.stdout.strip().split('\n') if line]
        return pids
    except Exception as e:
        print(f"[WARN] Failed to get nginx workers: {e}")
        return []


def get_master_worker_pid_pid_from_logs(log_file, exclude_pid=None):
    """Parse nginx error log to find which worker was master, optionally excluding a known dead pid."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        # Look for lines like: "Worker 0: Init timer, pid: 12345, g_state: ..., is_master_worker 1"
        pattern = r'Worker \d+: Init timer, pid: (\d+), .*, is_master_worker 1'
        matches = re.findall(pattern, content)
        if matches:
            # Return the most recent master worker pid that is alive
            for pid_str in reversed(matches):
                pid = int(pid_str)
                if exclude_pid is not None and pid == exclude_pid:
                    continue
                # Verify this pid is still alive
                try:
                    os.kill(pid, 0)  # Signal 0 just checks existence
                    return pid
                except ProcessLookupError:
                    continue
    except Exception as e:
        print(f"[WARN] Failed to parse log for master worker: {e}")
    return None


def get_master_worker_pid_from_logs(log_file):
    """Parse nginx error log to find the most recent master worker that is alive."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        pattern = r'Worker \d+: Init timer, pid: (\d+), .*, is_master_worker 1'
        matches = re.findall(pattern, content)
        if matches:
            # Return the most recent master worker pid that is alive
            for pid_str in reversed(matches):
                pid = int(pid_str)
                try:
                    os.kill(pid, 0)  # Signal 0 just checks existence
                    return pid
                except ProcessLookupError:
                    continue
    except Exception as e:
        print(f"[WARN] Failed to parse log for master worker: {e}")
    return None


def get_current_master_pid(log_file, current_workers):
    """Get current master pid from logs, filtering out dead workers."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        pattern = r'Worker \d+: Init timer, pid: (\d+), .*, is_master_worker 1'
        matches = re.findall(pattern, content)
        if matches:
            # Return the most recent master worker pid that is in current workers list
            for pid_str in reversed(matches):
                pid = int(pid_str)
                if pid in current_workers:
                    return pid
    except Exception as e:
        print(f"[WARN] Failed to parse log for master worker: {e}")
    return None


def kill_worker(pid, force=False):
    """Kill a worker process. Use SIGKILL to simulate coredump."""
    try:
        sig = signal.SIGKILL if force else signal.SIGTERM
        os.kill(pid, sig)
        print(f"[INFO] Sent {sig.name} to worker {pid}")
        return True
    except ProcessLookupError:
        print(f"[WARN] Worker {pid} already dead")
        return False
    except PermissionError:
        print(f"[ERROR] No permission to kill worker {pid}")
        return False


@pytest.fixture(scope="module")
def failover_env():
    """
    Setup: Start EPD mock server + omni-proxy with encode/prefill/decode endpoints.
    Teardown: Stop omni-proxy and mock server.
    """
    os.environ["no_proxy"] = "localhost,127.0.0.1"

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file_epd()
        proxy_port = ports["proxy_port"]
        encode_ports = ports["encode"]
        prefill_ports = ports["prefill"]
        decode_ports = ports["decode"]
        print(f"\n[DEBUG] Skipping fixture, {proxy_port=}, encode={encode_ports}, prefill={prefill_ports}, decode={decode_ports}")
        yield {
            "proxy_port": proxy_port,
            "prefill_ports": prefill_ports,
            "decode_ports": decode_ports,
        }
        return

    # Load ports for encode + prefill + decode (EPD mode)
    ports = port_manager.load_ports_epd(ENCODE_NUM, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    encode_ports = ports["encode"]
    prefill_ports = ports["prefill"]
    decode_ports = ports["decode"]

    # Start EPD mock server first (encode=1, prefill=2, decode=2)
    print(f"\n[SETUP] Starting EPD mock server: encode={encode_ports}, prefill={prefill_ports}, decode={decode_ports}")
    processes = start_mock_server(ENCODE_NUM, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Failed to start EPD mock server")
    time.sleep(2)

    # Start proxy after mock server is ready
    print(f"\n[SETUP] Starting omni-proxy on port {proxy_port}")
    ret = setup_proxy(
        proxy_port=proxy_port,
        encode_port_list=encode_ports,
        prefill_port_list=prefill_ports,
        decode_port_list=decode_ports,
        stream_ops="add",
        max_request_slots=1024,
        worker_processes=4
    )
    if ret.returncode != 0:
        cleanup_subprocess(processes)
        pytest.fail(f"Failed to start omni-proxy")

    yield {
        "proxy_port": proxy_port,
        "prefill_ports": prefill_ports,
        "decode_ports": decode_ports,
        "processes": processes,
    }

    # Teardown
    print("\n[TEARDOWN] Stopping omni-proxy...")
    teardown_proxy()

    print(f"\n[TEARDOWN] Stopping EPD mock server...")
    cleanup_subprocess(processes)


def wait_proxy_health(port, timeout=30):
    """Wait for proxy to become healthy."""
    url = f"http://127.0.0.1:{port}/omni_proxy/metrics"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code >= 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Proxy health check timed out after {timeout}s")


def send_test_request(proxy_port, request_id="test_failover"):
    """Send a test chat completions request to the proxy."""
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": request_id
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Hi, respond with a short word."}],
        "stream": False
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    return response


def test_master_worker_failover(failover_env):
    """
    Test that when master worker dies, a new worker takes over scheduling.

    Steps:
    1. Verify initial state - proxy works
    2. Find master worker from logs
    3. Kill master worker (SIGKILL to simulate coredump)
    4. Wait for nginx to respawn new worker
    5. Send request - should be scheduled by new master
    6. Verify request succeeds
    """
    proxy_port = failover_env["proxy_port"]
    prefill_ports = failover_env["prefill_ports"]
    log_file = Path(__file__).parent / "nginx_error.log"

    print(f"\n[TEST] Initial state check...")
    # First, verify proxy works initially
    resp = send_test_request(proxy_port, "initial_request")
    print(f"[TEST] Initial request status: {resp.status_code}")
    assert resp.status_code == 200, f"Initial request failed: {resp.text}"
    print(f"[TEST] Initial request succeeded")

    # Give time for logs to flush
    time.sleep(1)

    # Find master worker from logs
    master_pid = get_master_worker_pid_from_logs(log_file)
    if master_pid is None:
        pytest.skip("Could not determine master worker from logs")

    print(f"\n[TEST] Master worker PID: {master_pid}")

    # Get initial worker count
    initial_workers = get_nginx_workers()
    print(f"[TEST] Initial worker PIDs: {initial_workers}")
    initial_worker_count = len(initial_workers)

    # Kill master worker with SIGKILL (simulates coredump, no graceful exit)
    print(f"\n[TEST] Killing master worker {master_pid} with SIGKILL...")
    killed = kill_worker(master_pid, force=True)
    assert killed, "Failed to kill master worker"

    # Wait for nginx to respawn a new worker
    print(f"[TEST] Waiting for nginx to respawn worker...")
    max_wait = 30
    start = time.time()
    new_master_found = False
    new_master_pid = None

    while time.time() - start < max_wait:
        time.sleep(1)

        # Check if workers are back
        current_workers = get_nginx_workers()
        print(f"[TEST] Current workers after kill: {current_workers}")

        # Check if master worker log appears for a new worker
        potential_master = get_master_worker_pid_from_logs(log_file)
        if potential_master and potential_master != master_pid:
            new_master_pid = potential_master
            new_master_found = True
            print(f"[TEST] New master worker detected: PID {new_master_pid}")
            break

        # Also check if we have enough workers again
        if len(current_workers) >= initial_worker_count:
            print(f"[TEST] Worker count restored: {len(current_workers)}")

    assert new_master_found, "New master worker was not elected within timeout"
    print(f"\n[TEST] New master worker {new_master_pid} elected successfully")

    # Wait a bit for the new master to initialize
    time.sleep(3)

    # Send a request to verify scheduling still works
    print(f"\n[TEST] Sending request after master failover...")
    try:
        resp = send_test_request(proxy_port, "post_failover_request")
        print(f"[TEST] Post-failover request status: {resp.status_code}")
        print(f"[TEST] Post-failover response: {resp.text[:200]}...")
        assert resp.status_code == 200, f"Request after failover failed: {resp.text}"
        print(f"[TEST] Post-failover request succeeded!")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request after failover failed with exception: {e}")

    print(f"\n[TEST] Master worker failover test PASSED")


def test_non_master_worker_death_does_not_affect_scheduling(failover_env):
    """
    Test that when a non-master worker dies, scheduling continues unaffected.
    """
    proxy_port = failover_env["proxy_port"]
    prefill_ports = failover_env["prefill_ports"]
    log_file = Path(__file__).parent / "nginx_error.log"

    print(f"\n[TEST] Initial state check...")
    resp = send_test_request(proxy_port, "before_nonmaster_kill")
    assert resp.status_code == 200
    print(f"[TEST] Initial request succeeded")

    time.sleep(1)

    # Find master worker
    master_pid = get_master_worker_pid_from_logs(log_file)
    if master_pid is None:
        pytest.skip("Could not determine master worker from logs")

    print(f"\n[TEST] Master worker PID: {master_pid}")

    # Get all workers and kill a non-master one
    workers = get_nginx_workers()
    print(f"[TEST] All workers: {workers}")

    non_master_workers = [w for w in workers if w != master_pid]
    if not non_master_workers:
        pytest.skip("No non-master workers to kill")

    # Kill a non-master worker
    victim = non_master_workers[0]
    print(f"\n[TEST] Killing non-master worker {victim}...")
    killed = kill_worker(victim, force=True)
    assert killed

    # Wait a bit
    time.sleep(2)

    # Verify master is still alive
    workers_after = get_nginx_workers()
    print(f"[TEST] Workers after non-master kill: {workers_after}")
    assert master_pid in workers_after, "Master worker died unexpectedly"

    # Send request - should still work
    print(f"\n[TEST] Sending request after non-master death...")
    resp = send_test_request(proxy_port, "after_nonmaster_kill")
    print(f"[TEST] Request status: {resp.status_code}")
    assert resp.status_code == 200
    print(f"[TEST] Non-master worker death test PASSED")


def test_repeated_worker_kills_stress(failover_env):
    """
    Stress test: repeatedly kill master and non-master workers,
    verify requests still work after each kill.
    """
    proxy_port = failover_env["proxy_port"]
    prefill_ports = failover_env["prefill_ports"]
    log_file = Path(__file__).parent / "nginx_error.log"

    NUM_KILLS = 10  # Number of times to kill workers
    REQUESTS_PER_ROUND = 5  # Number of requests to send after each kill

    print(f"\n[STRESS TEST] Starting repeated kill stress test")
    print(f"[STRESS TEST] Will kill workers {NUM_KILLS} times, send {REQUESTS_PER_ROUND} requests each round")

    for round_num in range(1, NUM_KILLS + 1):
        print(f"\n[STRESS TEST] === Round {round_num}/{NUM_KILLS} ===")

        # Get current workers
        workers = get_nginx_workers()
        print(f"[STRESS TEST] Current workers: {workers}")

        # Get current master from logs (filtering out dead workers)
        current_master = get_current_master_pid(log_file, workers)
        if current_master is None:
            # If we can't find master in logs, do initial request to ensure one gets elected
            resp = send_test_request(proxy_port, f"round_{round_num}_initial")
            assert resp.status_code == 200
            time.sleep(1)
            current_master = get_master_worker_pid_from_logs(log_file)

        print(f"[STRESS TEST] Current master: {current_master}")

        # Determine which worker to kill
        non_master_workers = [w for w in workers if w != current_master]

        # Alternate between killing master and non-master
        if round_num % 2 == 1 and non_master_workers:
            # Odd rounds: kill non-master worker
            victim = non_master_workers[round_num % len(non_master_workers)]
            victim_type = "non-master"
        else:
            # Even rounds: kill master worker
            victim = current_master
            victim_type = "master"

        print(f"[STRESS TEST] Killing {victim_type} worker {victim}...")
        killed = kill_worker(victim, force=True)
        assert killed, f"Failed to kill {victim_type} worker"

        # Wait for respawn
        print(f"[STRESS TEST] Waiting for worker respawn...")
        time.sleep(3)

        # Verify workers are back
        workers_after = get_nginx_workers()
        print(f"[STRESS TEST] Workers after kill: {workers_after}")
        assert len(workers_after) >= 1, "No workers alive after kill"

        # If we killed master, send a request to ensure new master gets elected
        if victim_type == "master":
            print(f"[STRESS TEST] Sent probe request to trigger master election...")
            probe_resp = send_test_request(proxy_port, f"round_{round_num}_probe")
            print(f"[STRESS TEST] Probe request status: {probe_resp.status_code}")
            time.sleep(1)  # Give time for election

            # Now check for new master
            new_master = get_current_master_pid(log_file, workers_after)
            print(f"[STRESS TEST] New master elected: {new_master}")
            assert new_master != victim and new_master is not None, \
                "Master was killed but no new master elected"
        else:
            new_master = get_current_master_pid(log_file, workers_after)
            print(f"[STRESS TEST] Current master: {new_master}")

        # Send multiple requests to verify scheduling works
        print(f"[STRESS TEST] Sending {REQUESTS_PER_ROUND} requests to verify...")
        for req_num in range(1, REQUESTS_PER_ROUND + 1):
            request_id = f"stress_round_{round_num}_req_{req_num}"
            try:
                resp = send_test_request(proxy_port, request_id)
                print(f"[STRESS TEST] Round {round_num} Request {req_num}/{REQUESTS_PER_ROUND}: {resp.status_code}")
                assert resp.status_code == 200, f"Request {request_id} failed: {resp.text}"
            except requests.exceptions.RequestException as e:
                pytest.fail(f"Request {request_id} failed with exception: {e}")

        print(f"[STRESS TEST] Round {round_num} completed successfully")

    print(f"\n[STRESS TEST] All {NUM_KILLS} rounds completed successfully!")
    print(f"[STRESS TEST] Repeated worker kill stress test PASSED")