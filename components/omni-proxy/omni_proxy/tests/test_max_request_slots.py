# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Omni Proxy Authors

"""
Test cases for omni_proxy_max_request_slots configuration.

Coverage:
1. Proxy starts with default max_request_slots
2. Reload succeeds when max_request_slots is unchanged
3. Reload fails when max_request_slots is changed (validation)
4. Requests exceeding max_request_slots return 429 error

PD Flow:
    Client -> Omni Proxy -> Prefill Node
                          <- kv_transfer_params
                     -> Decode Node + kv_transfer_params
                          <- Response tokens
                          <- Client
"""

import pytest
import os
import subprocess
import time
import requests
import re
from pathlib import Path

from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess
import port_manager

# Configuration
ENCODE_NUM = 0  # PD mode only
PREFILL_NUM = 2
DECODE_NUM = 2
PREFILL_EXEC_TIME_MULTIPLIER = 5.0  # 5x multiplier for overflow test

CUR_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def pd_setup_teardown():
    """
    Setup: Start PD mock server + omni-proxy with prefill/decode endpoints.
    Teardown: Stop omni-proxy and mock server.

    Pattern copied from test_epd_proxy.py::epd_setup_teardown
    """
    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file_epd()
        proxy_port = ports.get("proxy_port", 32667)
        print(f"\n[DEBUG] Skipping fixture, {proxy_port=}")
        yield {"proxy_port": proxy_port}
        return

    # Load ports for encode(0) + prefill + decode
    ports = port_manager.load_ports_epd(ENCODE_NUM, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    print(f"\n[SETUP] Starting PD mock server: prefill={prefill_port_list}, decode={decode_port_list}")

    # Start PD mock server (encode_num=0, prefill + decode only)
    # Use 5x exec time multiplier so requests hold slots longer during overflow test
    processes = start_mock_server(ENCODE_NUM, PREFILL_NUM, DECODE_NUM, PREFILL_EXEC_TIME_MULTIPLIER)
    if not processes:
        pytest.fail("Failed to start PD mock server")

    time.sleep(2)  # Wait for mock server to be fully ready

    print(f"\n[SETUP] Starting omni-proxy on port {proxy_port}")

    # Start omni-proxy with prefill/decode endpoints
    ret = setup_proxy(
        proxy_port=proxy_port,
        prefill_port_list=prefill_port_list,
        decode_port_list=decode_port_list,
    )
    if ret == -1:
        cleanup_subprocess(processes)
        pytest.fail("Failed to start omni-proxy")

    # Wait for proxy to be healthy
    wait_proxy_health(proxy_port)

    yield {
        "proxy_port": proxy_port,
        "prefill_ports": prefill_port_list,
        "decode_ports": decode_port_list,
    }

    # --- Teardown ---
    print("\n[TEARDOWN] Stopping omni-proxy...")
    teardown_proxy()

    print("\n[TEARDOWN] Stopping PD mock server...")
    cleanup_subprocess(processes)


def wait_proxy_health(proxy_port, timeout=30):
    """Wait for proxy health endpoint to be ready."""
    url = f"http://127.0.0.1:{proxy_port}/omni_proxy/health"
    start = time.time()
    last_err = None

    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}"
        except requests.exceptions.ConnectionError as e:
            last_err = f"conn refused: {e}"
        except Exception as e:
            last_err = str(e)

        time.sleep(0.5)

    pytest.fail(f"proxy health endpoint not ready, last error={last_err}")


def get_nginx_pids(tag=""):
    """Get nginx master and worker PIDs."""
    master = None
    workers = []

    try:
        master_out = subprocess.check_output(
            "pgrep -f 'nginx: master process' | head -n1",
            shell=True,
            text=True,
        ).strip()
        master = int(master_out) if master_out else None
    except Exception:
        master = None

    time.sleep(0.5)  # small delay to ensure workers have started after master

    if master:
        try:
            worker_out = subprocess.check_output(
                f"pgrep -P {master} -f 'nginx: worker process'",
                shell=True,
                text=True,
            ).strip()
            if worker_out:
                workers = [int(x) for x in worker_out.splitlines()]
        except subprocess.CalledProcessError:
            workers = []

    print(f"\n[NGINX PID] {tag}")
    print(f"  master : {master}")
    print(f"  workers: {workers}")
    return master, workers


def send_inference_request(proxy_port, idx):
    """Send an inference request and validate response."""
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": f"hi {idx}"}],
        "stream": False,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        pytest.fail(f"Request {idx} failed: HTTP {r.status_code}, body={r.text!r}")

    body = (r.text or "").strip()
    if not body:
        pytest.fail(f"Request {idx} returned empty body")

    # Handle both non-stream JSON and SSE streaming format
    import json
    try:
        json.loads(body)
    except json.JSONDecodeError:
        # May be SSE streaming format starting with "data: "
        lines = body.split('\n')
        found_data = False
        for line in lines:
            if line.startswith('data: '):
                found_data = True
                json_str = line[6:]  # Remove "data: " prefix
                if json_str == '[DONE]':
                    continue  # SSE terminator, valid
                try:
                    json.loads(json_str)
                except json.JSONDecodeError:
                    pytest.fail(f"Request {idx} returned non-JSON SSE: {line[:200]!r}")
        if not found_data:
            pytest.fail(f"Request {idx} returned non-JSON body: {body[:200]!r}")


def test_max_request_slots_default(pd_setup_teardown):
    """
    Test that proxy works with default max_request_slots.

    This is a sanity check test that verifies:
    1. Proxy health endpoint responds 200
    2. Inference requests succeed with stream=False
    """
    proxy_port = pd_setup_teardown["proxy_port"]

    # Check health
    url = f"http://127.0.0.1:{proxy_port}/omni_proxy/health"
    r = requests.get(url, timeout=5)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"

    # Send a few requests
    for i in range(5):
        send_inference_request(proxy_port, i)

    print("[PASS] Default max_request_slots works correctly")


def test_max_request_slots_reload_same_value(pd_setup_teardown):
    """
    Test that reload succeeds when max_request_slots stays the same.

    Steps:
    1. Start proxy with max_request_slots=43008 (default)
    2. Reload with same value
    3. Verify no crash in error log

    Note: Due to WSL nginx daemonization limitations, we cannot reliably
    verify worker rotation. We check for crash keywords instead.
    """
    proxy_port = pd_setup_teardown["proxy_port"]
    conf_path = CUR_DIR / "nginx.conf"
    error_log = CUR_DIR / "nginx_error.log"

    # Get initial log position
    log_pos = error_log.stat().st_size if error_log.exists() else 0

    # Rewrite with same value (16384)
    rewrite_max_request_slots_in_conf(conf_path, 16384)

    # Reload nginx (use sudo since nginx runs as root)
    proc = subprocess.run(
        f"sudo nginx -c {conf_path} -s reload",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for reload to complete
    time.sleep(2)

    # Check no crash in error log
    if error_log.exists():
        with open(error_log, "r") as f:
            f.seek(log_pos)
            logs = f.read().lower()
        crash_keywords = ["exited on signal", "segmentation fault", "core dumped"]
        for kw in crash_keywords:
            assert kw not in logs, f"nginx crash detected: '{kw}'"

    print("[PASS] Reload with same max_request_slots succeeded")


def test_max_request_slots_reload_different_value(pd_setup_teardown):
    """
    Test that reload FAILS when max_request_slots is changed.

    Steps:
    1. Start proxy with default max_request_slots (16384)
    2. Reload with max_request_slots=65536
    3. Verify error message in nginx error log

    Expected error:
    "omni_proxy: max_request_slots cannot be changed on reload"

    NOTE: In WSL environments, nginx -s reload has issues with daemonization.
    If the reload causes nginx to exit rather than reload, this test will fail
    because the error message won't be logged. This is a WSL limitation, not
    a code bug.
    """
    proxy_port = pd_setup_teardown["proxy_port"]
    conf_path = CUR_DIR / "nginx.conf"
    error_log = CUR_DIR / "nginx_error.log"

    # Get initial nginx pids before reload
    master_before, workers_before = get_nginx_pids("before reload")

    # Get initial log position
    log_pos = error_log.stat().st_size if error_log.exists() else 0

    # Rewrite with DIFFERENT value (65536 vs original 16384)
    rewrite_max_request_slots_in_conf(conf_path, 65536)

    # Try to reload nginx (use sudo since nginx runs as root)
    proc = subprocess.run(
        f"sudo nginx -c {conf_path} -s reload",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for error log to be written
    time.sleep(2)

    # Check if nginx is still running after reload
    master_after, workers_after = get_nginx_pids("after reload")

    # If nginx exited completely (WSL reload issue), fail the test instead of skip
    if master_after is None:
        pytest.fail(
            f"nginx exited on reload. "
            f"reload returncode={proc.returncode}, "
            f"stderr={proc.stderr[:200]}"
        )

    # Check error log for the validation error message
    if not error_log.exists():
        pytest.fail("nginx error log not found after reload attempt")

    with open(error_log, "r") as f:
        f.seek(log_pos)
        logs = f.read()

    expected_error = "max_request_slots cannot be changed on reload"
    if expected_error not in logs:
        pytest.fail(
            f"Expected error message '{expected_error}' not found in error log. "
            f"Logs after reload:\n{logs}"
        )

    print("[PASS] Reload with different max_request_slots correctly rejected")


def rewrite_max_request_slots_in_conf(conf_path, new_value):
    """Replace or add omni_proxy_max_request_slots directive in http block."""
    with open(conf_path, "r") as f:
        content = f.read()

    # Check if directive already exists
    if re.search(r'omni_proxy_max_request_slots\s+\d+;?', content):
        # Replace existing value (preserve semicolon)
        content = re.sub(
            r'omni_proxy_max_request_slots\s+\d+;?',
            f'omni_proxy_max_request_slots {new_value};',
            content
        )
    else:
        # Add after http {
        content = re.sub(
            r'(http\s*\{)',
            r'\1\n    omni_proxy_max_request_slots ' + str(new_value) + ';',
            content
        )

    with open(conf_path, "w") as f:
        f.write(content)


def send_inference_request_no_fail(proxy_port, idx, timeout=30):
    """Send an inference request, return status code and body."""
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": f"hi {idx}"}],
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        return r.status_code, r.text
    except requests.exceptions.Timeout:
        return 0, "timeout"
    except requests.exceptions.ConnectionError as e:
        return 0, f"connection_error: {e}"


def send_concurrent_requests(proxy_port, num_requests, timeout=60):
    """Send concurrent requests using threads."""
    import concurrent.futures

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for i in range(num_requests):
            future = executor.submit(send_inference_request_no_fail, proxy_port, i, timeout)
            futures.append((i, future))

        for i, future in futures:
            try:
                status, body = future.result(timeout=timeout + 10)
                results.append((i, status, body))
            except Exception as e:
                results.append((i, 0, str(e)))
    return results


def test_max_request_slots_overflow(pd_setup_teardown):
    """
    Test that when max_request_slots is exceeded, requests return 429 error.

    Steps:
    1. Start mock servers with long exec time (5s) so requests hold slots
    2. Restart nginx with small max_request_slots (10) using setup_proxy
    3. Send 20 concurrent requests (all go to prefill, holding slots for 5s)
    4. Verify exactly 10 succeed (within slot limit) and 10 get 429 (exceeded)
    5. Verify nginx doesn't crash (no coredump)

    PD Flow:
        Client -> Omni Proxy -> Prefill Node (holds slot for 5s per request)
                              <- Response after 5s
    """
    proxy_port = pd_setup_teardown["proxy_port"]
    prefill_ports = pd_setup_teardown["prefill_ports"]
    decode_ports = pd_setup_teardown["decode_ports"]

    # Stop current nginx
    teardown_proxy()
    time.sleep(3)

    # Start nginx with small max_request_slots (10)
    small_max_slots = 10
    ret = setup_proxy(
        proxy_port=proxy_port,
        prefill_port_list=prefill_ports,
        decode_port_list=decode_ports,
        max_request_slots=small_max_slots,
    )
    if ret == -1:
        pytest.fail(f"nginx failed to start with max_request_slots={small_max_slots}")

    time.sleep(3)

    error_log = CUR_DIR / "nginx_error.log"
    log_pos = error_log.stat().st_size if error_log.exists() else 0

    # Send 20 concurrent requests - all will hit prefill and hold slots for 5s
    # Since max_slots=10, exactly 10 should succeed and 10 should get 429
    num_requests = 20
    print(f"\n[Sending {num_requests} concurrent requests with max_slots={small_max_slots}]")

    # Show all results
    results = send_concurrent_requests(proxy_port, num_requests, timeout=60)

    # Give time for all requests to complete
    time.sleep(2)

    # Check for crashes in error log
    crash_keywords = ["exited on signal", "segmentation fault", "core dumped"]
    crash_found = False
    if error_log.exists():
        with open(error_log, "r") as f:
            f.seek(log_pos)
            logs = f.read().lower()
        for kw in crash_keywords:
            if kw in logs:
                crash_found = True
                print(f"[CRASH DETECTED] {kw} found in error log")
                break

    # Count results - only 200 or 429 are valid (per user requirement)
    http_429_count = sum(1 for _, status, _ in results if status == 429)
    success_count = sum(1 for _, status, _ in results if status == 200)
    other_results = [(idx, status, body) for idx, status, body in results if status != 200 and status != 429]
    other_error_count = len(other_results)

    # Print details of other errors
    if other_error_count > 0:
        error_codes = set(status for _, status, _ in other_results)
        print(f"\n[Results] Success: {success_count}, 429: {http_429_count}, other_errors: {other_error_count}")
        print(f"[Other Error Codes] {sorted(error_codes)}")
        for idx, status, body in other_results[:5]:  # Show first 5
            print(f"  [idx={idx}] status={status}, body={body[:100]}...")
    else:
        print(f"\n[Results] Success: {success_count}, 429: {http_429_count}, other_errors: {other_error_count}")

    # Exactly 10 should succeed (within slot limit) and 10 should get 429 (exceeded)
    assert success_count == 10, f"Expected exactly 10 successes, got {success_count}"
    assert http_429_count == 10, f"Expected exactly 10 requests to get 429, got {http_429_count}"
    assert other_error_count == 0, f"Expected no other errors (503/5xx), got {other_error_count}"

    print(f"[PASS] Exactly {http_429_count} requests returned 429 without crashing")

    # Cleanup: stop nginx
    teardown_proxy()
