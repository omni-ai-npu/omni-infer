# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import json
import time
from pathlib import Path

import requests
import pytest

import port_manager
from run_proxy import setup_proxy, generate_proxy_endpoints, teardown_proxy
from run_vllm_mock import start_vllm_mock, cleanup_subprocess

# Configuration
PREFILL_NUM = 3
DECODE_NUM = 3
CUR_DIR = Path(__file__).parent
proxy_port = 7000
prefill_port_list = None
decode_port_list = None
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


def test_metrics_endpoint_err(setup_teardown):
    metrics_url = f"http://127.0.0.1:{proxy_port}/omni_proxy/metrics"
    metrics_resp = requests.get(metrics_url, timeout=10)
    print("Metrics Status Code:", metrics_resp.status_code)
    print("Metrics Body (preview):", metrics_resp.text[:200] + "..." if len(metrics_resp.text) > 200 else metrics_resp.text)
    assert metrics_resp.status_code >= 200


def test_chat_completions(setup_teardown):
    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/chat/completions"
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

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200


def test_chat_completions_stream(setup_teardown):
    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/chat/completions"
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
    # Enable streaming response handling
    with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as resp:
        resp.raise_for_status()  # Raise exception for HTTP error status codes
        token_cnt = 0
        for line in resp.iter_lines():
            if line:
                # Process Server-Sent Events (SSE) lines
                if line.startswith(b"data:"):
                    json_str = line[len(b"data:"):].strip()
                    if json_str == b"[DONE]":
                        print(f"Stream finished. get {token_cnt} output\n")
                        break
                    try:
                        chunk = json.loads(json_str)
                        # Extract content (assuming gpt-compatible format)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            token_cnt += 1
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"\nFailed to decode JSON: {json_str}")


def test_chat_completions_invalid_server(setup_teardown):
    invaild_port = port_manager.find_free_port_excluding_existing()
    url = f"http://127.0.0.1:{invaild_port}/v1/chat/completions"
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

    with pytest.raises(requests.exceptions.ConnectionError):
        response = requests.post(url, headers=headers, json=data, timeout=3)
        print("Status Code:", response.status_code)
        print("Response Headers:", response.headers)


def test_chat_completions_with_proxy(setup_teardown):
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


def test_chat_completions_non_stream_json(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200


def test_chat_completions_invalid_json_body(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }

    response = requests.post(url, headers=headers, data="{invalid_json", timeout=10)
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
    assert response.status_code >= 400


def test_chat_completions_missing_messages(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
    assert response.status_code >= 400


def test_chat_completions_messages_wrong_type(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "messages": "not-a-list",
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Body:", response.text)
    assert response.status_code >= 400


def test_metrics_endpoint(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False
    }
    resp = requests.post(url, headers=headers, json=data, timeout=30)
    assert resp.status_code == 200

    metrics_url = f"http://127.0.0.1:{proxy_port}/omni_proxy/metrics"
    metrics_resp = requests.get(metrics_url, timeout=10)
    print("Metrics Status Code:", metrics_resp.status_code)
    print("Metrics Body (preview):", metrics_resp.text[:200] + "..." if len(metrics_resp.text) > 200 else metrics_resp.text)
    assert metrics_resp.status_code == 200
    assert "# HELP" in metrics_resp.text
    assert "vllm:requests_success_total" in metrics_resp.text
    assert "vllm:time_to_first_token_seconds_bucket" in metrics_resp.text


def test_jsmn_nomen_trigger_with_proxy(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "jsmn-nomem"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False
    }

    # Add enough top-level fields to exceed 256 JSMN tokens.
    # Each key/value string adds two tokens, plus the object token itself.
    for i in range(140):
        data[f"k{i}"] = f"v{i}"

    response = requests.post(url, headers=headers, json=data, timeout=60)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200


def test_jsmn_parse_cached_reuse_log_with_proxy(setup_teardown):
    error_log = CUR_DIR / "nginx_error.log"
    before_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    before_hits = before_text.count("omni_origin_body_jsmn_cached: reuse cached tokens")

    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "jsmn-cache-reuse",
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "cache-check"}],
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    assert response.status_code == 200

    # Give nginx log flush a short time window, then verify cache-hit log increased.
    after_hits = before_hits
    for _ in range(10):
        after_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
        after_hits = after_text.count("omni_origin_body_jsmn_cached: reuse cached tokens")
        if after_hits > before_hits:
            break
        time.sleep(0.2)

    assert after_hits > before_hits, (
        "Expected omni_origin_body_jsmn_cached cache reuse log to increase after request, "
        "but no new 'reuse cached tokens' entry was found."
    )