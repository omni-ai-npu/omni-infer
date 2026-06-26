# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import time
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest
import requests

import utils
import port_manager 
from run_proxy import setup_proxy, teardown_proxy, generate_proxy_endpoints
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess


# Configuration
PREFILL_NUM = 3
DECODE_NUM = 3
proxy_port = 7000
prefill_port_list = None
decode_port_list = None
CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../omni_proxy.sh"

@pytest.fixture(scope="module")
def setup_teardown():
    global proxy_port
    global prefill_port_list
    global decode_port_list

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file_epd()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"]
        decode_port_list = ports["decode"]
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        return

    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list,
                      stream_ops="off")
    if ret == -1:
        pytest.fail(f"Start proxy fail")

    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start mock server fail")

    yield

    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)


def setup_proxy_parallel(proxy_port=7000, prefill_port_list=None, decode_port_list=None):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '123'
    ports = port_manager.get_ports_from_file_epd()
    prefill_list = generate_proxy_endpoints(ports["prefill"][:PREFILL_NUM])
    decode_list = generate_proxy_endpoints(ports["decode"][:DECODE_NUM])
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CUR_DIR}/nginx_balance.conf",
            "--core-num", "4",
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", f"{CUR_DIR}/nginx_error_balance.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access_balance.log",
            "--omni-proxy-pd-policy", "parallel",
            "--stream-ops", "add",
            "--no-reuseport",
            "--keepalive-nginx",
            
        ]
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SETUP] Script succeeded. Output:\n{result.stdout}")

        return 0
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Setup script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)
        
        return 1


def setup_proxy_trace_header(proxy_port=7000, prefill_port_list=None, decode_port_list=None):
    ports = port_manager.get_ports_from_file_epd()
    prefill_list = generate_proxy_endpoints(ports["prefill"][:PREFILL_NUM])
    decode_list = generate_proxy_endpoints(ports["decode"][:DECODE_NUM])
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CUR_DIR}/nginx_balance.conf",
            "--core-num", "4",
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", f"{CUR_DIR}/nginx_error_balance.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access_balance.log",
            "--stream-ops", "add",
            "--no-reuseport",
            "--keepalive-nginx",
            "--set_trace_headers_force"
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SETUP] Script succeeded. Output:\n{result.stdout}")
        return 0

    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Setup script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)
        return 1
    

# --- PD POLICY: PARALLEL TEST ---
def test_chat_completions_with_proxy_parallel(setup_teardown):
    proxy_port = port_manager.find_free_port()
    ret = setup_proxy_parallel(proxy_port)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")
    ngx_pid = utils.get_ngx_pid()
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    
    assert response.status_code == 200
    
    utils.teardown_proxy_balance(ngx_pid)


# --- SET TRACE HEADER TEST ---
def test_chat_completions_with_proxy_set_trace(setup_teardown):
    proxy_port_set_trace = port_manager.find_free_port()
    ret = setup_proxy_trace_header(proxy_port_set_trace)
    # wait proxy service ready
    ngx_pid = utils.get_ngx_pid()
    time.sleep(5) 
    if not ret == 0:
        pytest.fail(f"Start proxy fail")

    url = f"http://127.0.0.1:{proxy_port_set_trace}/v1/chat/completions"
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    response = requests.post(url, json=data, timeout=60)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200
    utils.teardown_proxy_balance(ngx_pid)