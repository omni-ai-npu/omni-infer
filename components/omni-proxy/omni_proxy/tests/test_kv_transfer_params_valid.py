# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Tests for kv_transfer_params handling - valid case.

When prefill returns a valid kv_transfer_params object, the proxy should
correctly forward the request to decode.
"""

import json
import os
import time

import pytest
import requests

import port_manager
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess


# Configuration
PREFILL_NUM = 1
DECODE_NUM = 1
proxy_port = 7200


@pytest.fixture(scope="module")
def valid_kv_setup_teardown():
    """
    Setup: Start mock server (prefill + decode) + omni-proxy in PD mode.
    """
    global proxy_port

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file_epd()
        proxy_port = ports.get("proxy_port", 7200)
        print(f"\n[DEBUG] Skipping fixture, {proxy_port=}")
        yield
        return

    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    print(f"\n[SETUP] Starting PD mock server: prefill={prefill_port_list}, decode={decode_port_list}")

    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Failed to start PD mock server")

    time.sleep(2)

    print(f"\n[SETUP] Starting omni-proxy on port {proxy_port} (PD mode)")

    ret = setup_proxy(
        proxy_port=proxy_port,
        prefill_port_list=prefill_port_list,
        decode_port_list=decode_port_list,
    )
    if ret.returncode != 0:
        cleanup_subprocess(processes)
        pytest.fail("Failed to start omni-proxy")

    yield {
        "proxy_port": proxy_port,
        "prefill_port_list": prefill_port_list,
        "decode_port_list": decode_port_list,
    }

    print("\n[TEARDOWN] Stopping omni-proxy...")
    teardown_proxy()

    print(f"\n[TEARDOWN] Stopping PD mock server...")
    cleanup_subprocess(processes)


def test_kv_transfer_params_valid_continue_to_decode(valid_kv_setup_teardown):
    """
    Test that when prefill returns a valid kv_transfer_params object,
    the proxy correctly forwards the request to decode.
    """
    context = valid_kv_setup_teardown
    proxy_port = context["proxy_port"]

    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-kv-valid-001"
    }

    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }

    print(f"\n[TEST] Sending request to proxy (prefill returns valid kv_transfer_params)")
    print(f"[TEST] Proxy URL: {url}")

    response = requests.post(url, headers=headers, json=data, timeout=60)

    print(f"[TEST] Response status: {response.status_code}")
    print(f"[TEST] Response headers: {dict(response.headers)}")
    print(f"[TEST] Response content length: {len(response.content)}")
    print(f"[TEST] Response content (preview): {response.text[:300] if response.text else '(empty)'}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    # Check content-type to handle both streaming and non-streaming responses
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        # Streaming response - accumulate chunks
        print(f"[TEST] Received streaming response")
        token_cnt = 0
        for line in response.iter_lines():
            if line and line.startswith(b"data:"):
                json_str = line[len(b"data:"):].strip()
                if json_str == b"[DONE]":
                    break
                try:
                    chunk = json.loads(json_str)
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        token_cnt += 1
                except json.JSONDecodeError:
                    pass
        print(f"[TEST] Received {token_cnt} tokens")
        assert token_cnt > 0, "Expected at least one token in streaming response"
    else:
        # Non-streaming JSON response
        resp_json = response.json()
        assert "choices" in resp_json, f"Expected 'choices' in response"
        assert len(resp_json["choices"]) > 0, "Expected at least one choice in response"

    print("[TEST] test_kv_transfer_params_valid_continue_to_decode PASSED")