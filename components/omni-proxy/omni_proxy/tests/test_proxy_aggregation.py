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
from run_vllm_mock import start_vllm_mock, cleanup_subprocess


# Configuration
PREFILL_NUM = 3
DECODE_NUM = 3
proxy_port = 7000
prefill_port_list = None
decode_port_list = None
CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../omni_proxy.sh"

@pytest.fixture(scope="module")
def setup_teardown(vllm_keep_alive):
    global proxy_port
    global prefill_port_list
    global decode_port_list

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"]
        decode_port_list = ports["decode"]
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        return
    
    if os.getenv("PROXY_VLLM_POOL") == "1":
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"][:PREFILL_NUM]
        decode_port_list = ports["decode"][:DECODE_NUM]
        ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list)
        if ret == -1:
            pytest.fail(f"Start proxy fail")
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        teardown_proxy()
        return    
    
    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")

    processes = start_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start vllm fail")

    yield

    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)


def setup_proxy_aggregation(proxy_port=7000, prefill_port_list=None, decode_port_list=None):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '123'
    ports = port_manager.get_ports_from_file()
    prefill_list = generate_proxy_endpoints(ports["prefill"][:PREFILL_NUM])
    decode_list = generate_proxy_endpoints(ports["decode"][:DECODE_NUM])
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CUR_DIR}/nginx_balance.conf",
            "--core-num", "4",
            "--listen-port", f"{proxy_port}",
            "--decode-endpoints", decode_list,
            "--log-file", f"{CUR_DIR}/nginx_error_balance.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access_balance.log",
            "--omni-proxy-pd-policy", "aggregation",
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

def test_chat_completions_with_proxy_aggregation(setup_teardown):
    proxy_port = port_manager.find_free_port()
    ret = setup_proxy_aggregation(proxy_port)
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

