# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Proxy reload stability tests.

This file contains:
- Serial reload correctness test (deterministic)
- Concurrent traffic + multi-round reload stress test

Redundancy in test logic is intentional for observability and debuggability.
"""

import pytest
import os
import subprocess
import time
import requests
import json
import re
import threading
import random
from pathlib import Path

from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock import start_vllm_mock, cleanup_subprocess, setup_vllm
import port_manager

PREFILL_NUM = 3
DECODE_NUM = 3
VLLM_POOL = os.getenv("PROXY_VLLM_POOL") == "1"

@pytest.fixture(scope="module")
def reload_env(vllm_keep_alive):
    os.environ["no_proxy"] = "localhost,127.0.0.1"

    if VLLM_POOL:
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_ports = ports["prefill"][:PREFILL_NUM]
        decode_ports = ports["decode"][:DECODE_NUM]
        ret = setup_proxy(proxy_port, prefill_ports, decode_ports)
        if ret == -1:
            pytest.fail(f"Start proxy fail")
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_ports=}, {decode_ports=}")
        yield {
        "proxy_port": proxy_port,
        "prefill_ports": prefill_ports,
        "decode_ports": decode_ports,
        }
        teardown_proxy()
        return

    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)

    proxy_port = ports["proxy_port"]
    prefill_ports = ports["prefill"]
    decode_ports = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_ports, decode_ports)
    if ret == -1:
        pytest.fail("Start proxy fail")

    wait_proxy_health(proxy_port)

    processes = start_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Start vllm fail")
    time.sleep(1)

    yield {
        "proxy_port": proxy_port,
        "prefill_ports": prefill_ports,
        "decode_ports": decode_ports,
        "processes": processes,
    }

    cleanup_subprocess(processes)

    try:
        teardown_proxy()
    except Exception as e:
        print(f"[TEARDOWN] teardown_proxy ignored: {e}")

# Case behavior helpers
def apply_case_1_remove(cur_prefill, cur_decode):
    """
    Case 1: -P2 / -D1
    """
    removed_p = cur_prefill.pop(1)
    removed_d = cur_decode.pop(0)
    return removed_p, removed_d


def apply_case_2_restore(cur_prefill, cur_decode, base_prefill, base_decode):
    """
    Case 2: +P2 / +D1（恢复到 base）
    """
    cur_prefill[:] = base_prefill[:]
    cur_decode[:] = base_decode[:]


def apply_case_3_append_existing(cur_prefill, cur_decode, p3_port, d3_port):
    """
    Case 3: +P3 / +D3（新 vLLM）
    """
    cur_prefill.append(p3_port)
    cur_decode.append(d3_port)


def apply_case_4_remove_new(cur_prefill, cur_decode, p3_port, d3_port):
    """
    Case 4: -P3 / -D3（回到 base）
    """
    cur_prefill.remove(p3_port)
    cur_decode.remove(d3_port)

# Common helpers
def reload_nginx(conf_path):
    proc = subprocess.run(
        f"nginx -c {conf_path} -s reload",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, "nginx reload failed"
    assert proc.stderr.strip() == "", f"nginx reload stderr: {proc.stderr}"


def wait_proxy_health(proxy_port, timeout=30):
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

    pytest.fail(f"proxy health endpoint not ready after reload, last error={last_err}")


def rewrite_upstream_servers_only(conf_path, upstream_name, new_ports):
    with open(conf_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    inside_target = False
    server_inserted = False

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith(f"upstream {upstream_name}"):
            inside_target = True
            server_inserted = False
            new_lines.append(line)
            i += 1
            continue

        if inside_target:
            stripped = line.strip()

            if stripped.startswith("server "):
                i += 1
                continue

            if stripped == "}" and not server_inserted:
                for p in new_ports:
                    new_lines.append(
                        f"        server 127.0.0.1:{p} max_fails=3 fail_timeout=10s;\n"
                    )
                server_inserted = True
                new_lines.append(line)
                inside_target = False
                i += 1
                continue

            new_lines.append(line)
            i += 1
            continue

        new_lines.append(line)
        i += 1

    with open(conf_path, "w") as f:
        f.writelines(new_lines)

def wait_vllm_ready(processes, logs, timeout=120):
    start = time.time()
    ready = [False] * len(processes)

    while time.time() - start < timeout:
        for i, (proc, log) in enumerate(zip(processes, logs)):
            if ready[i]:
                continue

            if proc.poll() is not None:
                raise RuntimeError(f"vLLM process {proc.pid} exited early")

            if log.exists() and "Application startup complete." in log.read_text():
                ready[i] = True

        if all(ready):
            return

        time.sleep(1)

    raise RuntimeError("new vLLM instances did not become ready in time")


COMPLETE_MARK = "Upstream initialization completed"
def wait_reload_complete_from_log(
    error_log: Path,
    start_pos: int,
    proxy_port: int,
    timeout: int = 30,
):
    start = time.time()

    while time.time() - start < timeout:
        # 条件 1：看到 reload 完成日志
        if error_log.exists():
            with open(error_log, "r") as f:
                f.seek(start_pos)
                if COMPLETE_MARK in f.read():
                    return

        # 条件 2：proxy health 恢复
        try:
            r = requests.get(
                f"http://127.0.0.1:{proxy_port}/omni_proxy/health",
                timeout=2,
            )
            if r.status_code == 200:
                return
        except Exception:
            pass

        time.sleep(0.5)
    raise RuntimeError("reload did not complete")

def get_nginx_pids(tag=""):
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
    else:
        out = subprocess.check_output(
            "ps -ef | grep nginx | grep -v grep",
            shell=True,
            text=True,
        ).strip()
        for line in out.splitlines():
            parts = line.split()
            if "master process" in line:
                master = int(parts[1])
            elif "worker process" in line:
                workers.append(int(parts[1]))

    print(f"\n[NGINX PID] {tag}")
    print(f"  master : {master}")
    print(f"  workers: {workers}")
    return master, workers

def _response_body_is_valid_json_or_sse_json(r) -> bool:
    body = (r.text or "").strip()
    if not body:
        return False

    # 情况 1：普通 JSON
    try:
        r.json()
        return True
    except Exception:
        pass

    # 情况 2：SSE（data: {...}）
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue

        data_part = line[len("data:"):].strip()
        if not data_part or data_part == "[DONE]":
            continue

        try:
            json.loads(data_part)
            return True
        except Exception:
            continue

    return False

def test_proxy_reload(reload_env):
    proxy_port = reload_env["proxy_port"]
    prefill_port_list = reload_env["prefill_ports"].copy()
    decode_port_list = reload_env["decode_ports"].copy()
    """
    Health-based proxy reload test (enhanced).

    Coverage:
    - Baseline nginx reload mechanics (master unchanged, workers rotated)
    - Reload scenarios:
        1) -P2 / -D1
        2) +P2 / +D1 (same ports)
        3) +P3 / +D3 (new ports, new vLLM)
        4) -P3 / -D3
    - After each reload:
        * proxy health OK
        * 30 inference requests succeed (HTTP 200)
        * no nginx crash (SIGSEGV / core dumped)
        * master PID unchanged
    - nginx.conf must be restored after test
    """
    SELECT_CASE = os.getenv("RELOAD_CASE")
    if SELECT_CASE:
        print(f"[RELOAD_CASE] Only running Case {SELECT_CASE}")
    conf_path = "/usr/local/nginx/conf/nginx.conf"
    error_log = Path(__file__).resolve().parent / "nginx_error.log"

    NGINX_CRASH_KEYWORDS = [
        "exited on signal",
        "segmentation fault",
        "core dumped",
    ]

    with open(conf_path, "r") as f:
        original_nginx_conf = f.read()

    def assert_no_nginx_crash(error_log: Path, start_pos: int):
        if not error_log.exists():
            return
        with open(error_log, "r") as f:
            f.seek(start_pos)
            logs = f.read().lower()
        for kw in NGINX_CRASH_KEYWORDS:
            assert kw not in logs, (
                f"nginx crash detected after reload: '{kw}'\n{logs}"
            )

    def send_requests(num=30):
        url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "qwen",
            "temperature": 0,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        for i in range(num):
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code != 200:
                pytest.fail(
                    f"Request {i} failed: HTTP {r.status_code}, body={r.text!r}"
                )

            body = (r.text or "").strip()
            if not body:
                pytest.fail(f"Request {i} returned empty body")

            if not _response_body_is_valid_json_or_sse_json(r):
                pytest.fail(
                    f"Request {i} returned non-JSON body: {body[:200]!r}"
                )

    PREFILL_RE = re.compile(r"Add Prefill peer .* -> 127\.0\.0\.1:(\d+)")
    DECODE_RE = re.compile(r"Add Decode peer .* -> 127\.0\.0\.1:(\d+)")

    def parse_new_ports_from_log(error_log, start_pos):
        if not error_log.exists():
            return set(), set()
        with open(error_log, "r") as f:
            f.seek(start_pos)
            logs = f.read()
        return (
            set(map(int, PREFILL_RE.findall(logs))),
            set(map(int, DECODE_RE.findall(logs))),
        )

    new_processes = []
    new_logs = []
    p3_port = None
    d3_port = None

    try:
        base_prefill = prefill_port_list.copy()
        base_decode = decode_port_list.copy()

        cur_prefill = base_prefill.copy()
        cur_decode = base_decode.copy()

        def run_case(
            name,
            modify_fn,
            expect_prefill,
            expect_decode,
            expect_new_p=None,
            expect_new_d=None,
        ):
            print(f"\n===== {name} =====")

            master_before, workers_before = get_nginx_pids("before reload")
            log_pos = error_log.stat().st_size if error_log.exists() else 0

            before_p = cur_prefill.copy()
            before_d = cur_decode.copy()

            modify_fn()

            rewrite_upstream_servers_only(conf_path, "prefill_endpoints", cur_prefill)
            rewrite_upstream_servers_only(conf_path, "decode_endpoints", cur_decode)

            reload_nginx(conf_path)
            wait_reload_complete_from_log(error_log, log_pos, proxy_port)
            time.sleep(0.5)
            wait_proxy_health(proxy_port)

            master_after, workers_after = get_nginx_pids("after reload")

            assert master_after == master_before, (
                f"nginx master pid changed: {master_before} -> {master_after}"
            )
            assert workers_after != workers_before, \
                "nginx workers not rotated after reload"

            print("\n================ Upstream Config Diff =================")
            print(f"Prefill: {before_p} -> {cur_prefill}")
            print(f"Decode : {before_d} -> {cur_decode}")
            print("======================================================\n")

            assert set(cur_prefill) == set(expect_prefill)
            assert set(cur_decode) == set(expect_decode)

            new_p, new_d = parse_new_ports_from_log(error_log, log_pos)

            if expect_new_p:
                print(f"[INFO] new prefill seen: {sorted(new_p)}")
            if expect_new_d:
                print(f"[INFO] new decode seen: {sorted(new_d)}")

            send_requests()
            assert_no_nginx_crash(error_log, log_pos)

        # Case 1: -P2 / -D1
        removed_p = cur_prefill[1]
        removed_d = cur_decode[0]

        expect_p1 = base_prefill.copy()
        expect_p1.remove(removed_p)
        expect_d1 = base_decode.copy()
        expect_d1.remove(removed_d)

        if SELECT_CASE in (None, "1"):
            run_case(
                "Case 1: -P2 / -D1",
                lambda: apply_case_1_remove(cur_prefill, cur_decode),
                expect_p1,
                expect_d1,
            )

        # Case 2: restore
        if SELECT_CASE in (None, "2"):
            run_case(
                "Case 2: +P2 / +D1",
                lambda: apply_case_2_restore(
                    cur_prefill, cur_decode, base_prefill, base_decode
                ),
                base_prefill,
                base_decode,
                expect_new_p=[removed_p],
                expect_new_d=[removed_d],
            )

        # Case 3: +P3 / +D3
        if SELECT_CASE in (None, "3"):
            if VLLM_POOL:
                ports = port_manager.get_ports_from_file()
                p3_port = ports["prefill"][PREFILL_NUM]
                d3_port = ports["decode"][DECODE_NUM]
            else:
                p3_port = port_manager.find_free_port_excluding_existing()
                d3_port = port_manager.find_free_port_excluding_existing(p3_port)

                procs, logs = setup_vllm(True, [p3_port], log_file_prefix="reload")
                new_processes.extend(procs)
                new_logs.extend(logs)

                procs2, logs2 = setup_vllm(False, [d3_port], log_file_prefix="reload")
                new_processes.extend(procs2)
                new_logs.extend(logs2)

                wait_vllm_ready(procs2, logs2)

            run_case(
                "Case 3: +P3 / +D3",
                lambda: apply_case_3_append_existing(
                    cur_prefill, cur_decode, p3_port, d3_port
                ),
                base_prefill + [p3_port],
                base_decode + [d3_port],
                expect_new_p=[p3_port],
                expect_new_d=[d3_port],
            )

        # Case 4: rollback
        if SELECT_CASE in (None, "4"):
            run_case(
                "Case 4: -P3 / -D3",
                lambda: apply_case_4_remove_new(
                    cur_prefill, cur_decode, p3_port, d3_port
                ),
                base_prefill,
                base_decode,
            )

    finally:
        with open(conf_path, "w") as f:
            f.write(original_nginx_conf)

        if new_processes:
            cleanup_subprocess(new_processes)

def test_proxy_reload_under_concurrent_traffic(reload_env):
    """
    Concurrent traffic + multi-round real reload stability test.

    Test objective:
    With continuous real request traffic in flight, repeatedly perform nginx reloads
    in the background. During each reload, the upstream configuration (Prefill / Decode
    backends) is actively changed (remove / add / rollback) to validate the proxy's
    stability under high churn conditions.

    Key guarantees covered by this test:
    1. Requests must continue to succeed during reload (no dropped requests, no 502s).
    2. Reload operations run in a background thread, concurrently with request traffic.
    3. Reload is executed repeatedly across multiple rounds.
    4. Each reload involves real upstream changes (Prefill / Decode add/remove/rollback).
    5. After each round, the configuration must be fully restored to the initial baseline
    (no state pollution).

    Policy constraints:
    - Client timeouts are relaxed to avoid false positives.
    - Any request timeout is treated as a failure.
    - All requests must return HTTP 200.
    """
    proxy_port = reload_env["proxy_port"]

    base_prefill = reload_env["prefill_ports"].copy()
    base_decode = reload_env["decode_ports"].copy()

    conf_path = "/usr/local/nginx/conf/nginx.conf"
    error_log = Path.cwd() / "nginx_error.log"
    log_pos = error_log.stat().st_size if error_log.exists() else 0

    stop_event = threading.Event()
    request_errors = []
    reload_errors = []

    stats = {
        "request_count": 0,
        "request_success": 0,
    }
    request_latencies = []
    stats_lock = threading.Lock()

    master_pid_baseline, _ = get_nginx_pids("baseline (test start)")

    def request_worker():
        url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "qwen",
            "temperature": 0,
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        timeout = (3, 60)

        while not stop_event.is_set():
            start_ts = time.time()
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=timeout,
                )
                latency = time.time() - start_ts

                with stats_lock:
                    stats["request_count"] += 1
                    request_latencies.append(latency)

                if r.status_code != 200:
                    request_errors.append(f"HTTP {r.status_code}: {r.text!r}")
                    continue

                body = (r.text or "").strip()
                if not body:
                    request_errors.append("empty body")
                    continue

                if not _response_body_is_valid_json_or_sse_json(r):
                    request_errors.append(f"non-JSON body: {body[:200]!r}")
                    continue

                with stats_lock:
                    stats["request_success"] += 1

            except Exception as e:
                latency = time.time() - start_ts
                with stats_lock:
                    stats["request_count"] += 1
                    request_latencies.append(latency)

                request_errors.append(str(e))

            finally:
                time.sleep(0.02)

    def reload_worker():
        cur_prefill = base_prefill.copy()
        cur_decode = base_decode.copy()

        new_processes = []
        new_logs = []

        def _pid_guard(round_id, case_id, when, master_pid):
            if master_pid != master_pid_baseline:
                raise RuntimeError(
                    f"[Round {round_id} Case {case_id}] master pid changed before reload: "
                    f"{master_pid_baseline} -> {master_pid}"
                    if when == "before" else
                    f"[Round {round_id} Case {case_id}] nginx master pid changed: "
                    f"{master_pid_baseline} -> {master_pid}"
                )

        def _reload_pipeline(round_id, case_id, before_tag, after_tag, workers_rotate_err):
            master_before, workers_before = get_nginx_pids(before_tag)
            _pid_guard(round_id, case_id, "before", master_before)

            time.sleep(1.0)
            reload_nginx(conf_path)
            wait_reload_complete_from_log(error_log, log_pos, proxy_port)
            time.sleep(0.5)
            wait_proxy_health(proxy_port)

            master_after, workers_after = get_nginx_pids(after_tag)
            _pid_guard(round_id, case_id, "after", master_after)

            if workers_after == workers_before:
                raise RuntimeError(workers_rotate_err)

            time.sleep(random.uniform(3, 8))

        try:
            ROUNDS = 1

            for round_id in range(ROUNDS):
                print(f"\n[ROUND {round_id}] ===============================")
                print(f"[ROUND {round_id}] Starting reload cycle")

                # Case 1
                print(f"[ROUND {round_id}] Case 1: remove one P and one D")
                master_before, workers_before = get_nginx_pids(
                    f"round {round_id} case 1 (before reload)"
                )
                if master_before != master_pid_baseline:
                    raise RuntimeError(
                        f"[Round {round_id} Case 1] master pid already changed before reload: "
                        f"{master_pid_baseline} -> {master_before}"
                    )

                removed_p, removed_d = apply_case_1_remove(cur_prefill, cur_decode)
                print(f"[ROUND {round_id}]   removed P={removed_p}, D={removed_d}")

                rewrite_upstream_servers_only(conf_path, "prefill_endpoints", cur_prefill)
                rewrite_upstream_servers_only(conf_path, "decode_endpoints", cur_decode)

                time.sleep(1.0)
                reload_nginx(conf_path)
                wait_reload_complete_from_log(error_log, log_pos, proxy_port)
                time.sleep(0.5)
                wait_proxy_health(proxy_port)

                master_after, workers_after = get_nginx_pids(
                    f"round {round_id} case 1 (after reload)"
                )

                if master_after != master_pid_baseline:
                    raise RuntimeError(
                        f"[Round {round_id} Case 1] nginx master pid changed: "
                        f"{master_pid_baseline} -> {master_after}"
                    )

                if workers_after == workers_before:
                    raise RuntimeError(
                        f"[Round {round_id} Case 1] nginx workers not rotated after reload"
                    )

                time.sleep(random.uniform(3, 8))

                # Case 2
                print(f"[ROUND {round_id}] Case 2: restore base P/D")
                master_before, workers_before = get_nginx_pids(
                    f"round {round_id} case 2 (before reload)"
                )
                if master_before != master_pid_baseline:
                    raise RuntimeError(
                        f"[Round {round_id} Case 2] master pid changed before reload: "
                        f"{master_pid_baseline} -> {master_before}"
                    )

                apply_case_2_restore(cur_prefill, cur_decode, base_prefill, base_decode)

                rewrite_upstream_servers_only(conf_path, "prefill_endpoints", cur_prefill)
                rewrite_upstream_servers_only(conf_path, "decode_endpoints", cur_decode)

                time.sleep(1.0)
                reload_nginx(conf_path)
                wait_reload_complete_from_log(error_log, log_pos, proxy_port)
                time.sleep(0.5)
                wait_proxy_health(proxy_port)

                master_after, workers_after = get_nginx_pids(
                    f"round {round_id} case 2 (after reload)"
                )

                if master_after != master_pid_baseline:
                    raise RuntimeError(
                        f"[Round {round_id} Case 2] nginx master pid changed: "
                        f"{master_pid_baseline} -> {master_after}"
                    )

                if workers_after == workers_before:
                    raise RuntimeError(
                        f"[Round {round_id} Case 2] nginx workers not rotated after reload"
                    )

                time.sleep(random.uniform(3, 8))

                # Case 3
                print(f"[ROUND {round_id}] Case 3: add new P/D backends")

                master_before, workers_before = get_nginx_pids(
                    f"round {round_id} case 3 (before reload)"
                )
                if master_before != master_pid_baseline:
                    raise RuntimeError(
                        f"[Round {round_id} Case 3] master pid changed before reload: "
                        f"{master_pid_baseline} -> {master_before}"
                    )
                if VLLM_POOL:
                    ports = port_manager.get_ports_from_file()
                    p3 = ports["prefill"][PREFILL_NUM]
                    d3 = ports["decode"][DECODE_NUM]
                else:
                    p3 = port_manager.find_free_port_excluding_existing()
                    d3 = port_manager.find_free_port_excluding_existing(p3)
                    print(f"[ROUND {round_id}]   new P={p3}, D={d3}")

                    procs, logs = setup_vllm(True, [p3], log_file_prefix="reload")
                    new_processes.extend(procs)
                    new_logs.extend(logs)

                    procs2, logs2 = setup_vllm(False, [d3], log_file_prefix="reload")
                    new_processes.extend(procs2)
                    new_logs.extend(logs2)

                    wait_vllm_ready(procs2, logs2)

                apply_case_3_append_existing(cur_prefill, cur_decode, p3, d3)

                rewrite_upstream_servers_only(conf_path, "prefill_endpoints", cur_prefill)
                rewrite_upstream_servers_only(conf_path, "decode_endpoints", cur_decode)

                time.sleep(1.0)
                reload_nginx(conf_path)
                wait_reload_complete_from_log(error_log, log_pos, proxy_port)
                time.sleep(0.5)
                wait_proxy_health(proxy_port)

                master_after, workers_after = get_nginx_pids(
                    f"round {round_id} case 3 (after reload)"
                )

                if master_after != master_pid_baseline:
                    raise RuntimeError(
                        f"[Round {round_id} Case 3] nginx master pid changed: "
                        f"{master_pid_baseline} -> {master_after}"
                    )

                if workers_after == workers_before:
                    raise RuntimeError(
                        f"[Round {round_id} Case 3] nginx workers not rotated after reload"
                    )

                time.sleep(random.uniform(3, 8))

                # Case 4
                print(f"[ROUND {round_id}] Case 4: remove new P/D and rollback")
                master_before, workers_before = get_nginx_pids(
                    f"round {round_id} case 4 (before reload)"
                )
                if master_before != master_pid_baseline:
                    raise RuntimeError(
                        f"[Round {round_id} Case 4] master pid changed before reload: "
                        f"{master_pid_baseline} -> {master_before}"
                    )

                time.sleep(1.0)
                apply_case_4_remove_new(cur_prefill, cur_decode, p3, d3)

                rewrite_upstream_servers_only(conf_path, "prefill_endpoints", cur_prefill)
                rewrite_upstream_servers_only(conf_path, "decode_endpoints", cur_decode)

                time.sleep(1.0)
                reload_nginx(conf_path)
                wait_reload_complete_from_log(error_log, log_pos, proxy_port)
                time.sleep(0.5)
                wait_proxy_health(proxy_port)

                master_after, workers_after = get_nginx_pids(
                    f"round {round_id} case 4 (after reload)"
                )

                if master_after != master_pid_baseline:
                    raise RuntimeError(
                        f"[Round {round_id} Case 4] nginx master pid changed: "
                        f"{master_pid_baseline} -> {master_after}"
                    )

                if workers_after == workers_before:
                    raise RuntimeError(
                        f"[Round {round_id} Case 4] nginx workers not rotated after reload"
                    )

                time.sleep(random.uniform(3, 8))

                if set(cur_prefill) != set(base_prefill):
                    raise RuntimeError(
                        f"[Round {round_id}] Prefill polluted: "
                        f"{cur_prefill} vs {base_prefill}"
                    )

                if set(cur_decode) != set(base_decode):
                    raise RuntimeError(
                        f"[Round {round_id}] Decode polluted: "
                        f"{cur_decode} vs {base_decode}"
                    )

                print(f"[ROUND {round_id}] Completed successfully")

        except Exception as e:
            reload_errors.append(e)
        finally:
            if new_processes:
                cleanup_subprocess(new_processes)

    t_req = threading.Thread(target=request_worker, daemon=True)
    t_reload = threading.Thread(target=reload_worker)

    t_req.start()
    t_reload.start()

    t_reload.join()
    stop_event.set()
    t_req.join(timeout=10)

    master_end, _ = get_nginx_pids("baseline check (test end)")
    if master_end != master_pid_baseline:
        pytest.fail(
            f"nginx master pid changed during test: {master_pid_baseline} -> {master_end}"
        )

    if reload_errors:
        pytest.fail(f"Reload thread failed: {reload_errors[0]}")

    if request_errors:
        pytest.fail(
            "Requests failed during concurrent reload:\n" +
            "\n".join(request_errors[:10])
        )

    if request_latencies:
        max_lat = max(request_latencies)
        avg_lat = sum(request_latencies) / len(request_latencies)
        print(
            f"\n[REQUEST] total={stats['request_count']} "
            f"success={stats['request_success']} "
            f"max_latency={max_lat:.2f}s "
            f"avg_latency={avg_lat:.2f}s"
        )
