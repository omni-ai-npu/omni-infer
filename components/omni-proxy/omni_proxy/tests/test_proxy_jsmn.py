# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os

import pytest
import requests

from pathlib import Path
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock import start_vllm_mock, cleanup_subprocess
import port_manager

# Configuration
PREFILL_NUM = 3
DECODE_NUM = 3
proxy_port = 7000
prefill_port_list = None
decode_port_list = None

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
    if ret == -1:
        pytest.fail(f"Start proxy fail")

    processes = start_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start vllm fail")

    yield

    # --- Teardown: Shut down all instances ---
    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)

def test_chat_completions_with_proxy_stream_options(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = [
        {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
        },
        {
        "Content-Type": "application/json",
        "X-Request-Id": "12346"
        },
    ]
    data = [
        {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True
        }
    },

    {
        "model": "qwen",
        "temperature": 0,
        # "max_tokens": None,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True,
    },

    ]
    for idx in range(len(data)):
        response = requests.post(url, headers=headers[idx], json=data[idx], timeout=30)
        print("Status Code:", response.status_code)
        print("Response Headers:", response.headers)
        print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
        assert response.status_code == 200