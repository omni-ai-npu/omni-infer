# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Tests for input_ids injection in decode body rewrite.

When the proxy has --omni-proxy-model-path set, the tokenizer produces input_ids
that are injected into both prefill and decode request bodies. This test uses
the EPD mock server (which returns kv_transfer_params in prefill responses) so
the proxy proceeds to the decode phase and exercises the decode body input_ids
injection code path (omni_pd_body_rewrite.c lines 838-903).
"""
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

import port_manager
from run_proxy import generate_proxy_endpoints, teardown_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess

PREFILL_NUM = 1
DECODE_NUM = 1
CUR_DIR = Path(__file__).parent
proxy_port = 7000
proxy_script_path = f"{CUR_DIR}/../omni_proxy.sh"


def setup_proxy_with_model_path(proxy_port, prefill_port_list, decode_port_list):
    """Start proxy with --omni-proxy-model-path to enable tokenizer."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "123"
    if '/usr/sbin' not in env.get('PATH', ''):
        env['PATH'] = '/usr/sbin:' + env.get('PATH', '')

    prefill_list = generate_proxy_endpoints(prefill_port_list)
    decode_list = generate_proxy_endpoints(decode_port_list)

    cmd = [
        "bash", proxy_script_path,
        "--nginx-conf-file", f"{CUR_DIR}/nginx.conf",
        "--core-num", "1",
        "--listen-port", str(proxy_port),
        "--prefill-endpoints", prefill_list,
        "--decode-endpoints", decode_list,
        "--log-file", f"{CUR_DIR}/nginx_error.log",
        "--log-level", "info",
        "--access-log-file", f"{CUR_DIR}/nginx_access.log",
        "--stream-ops", "add",
        "--omni-proxy-model-path", f"{CUR_DIR}/mock_model",
        "--no-reuseport",
    ]
    try:
        print(f"\n[SETUP] Starting proxy with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        print(f"[SETUP] Script succeeded. Output:\n{result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Setup script failed with exit code {e.returncode}.\n"
              f"STDERR: {e.stderr}\nSTDOUT: {e.stdout}")
        return -1


@pytest.fixture(scope="module")
def setup_teardown():
    global proxy_port

    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Failed to start EPD mock server")

    time.sleep(2)

    ret = setup_proxy_with_model_path(proxy_port, prefill_port_list, decode_port_list)
    if ret == -1:
        cleanup_subprocess(processes)
        pytest.fail("Start proxy fail")

    yield

    teardown_proxy()
    cleanup_subprocess(processes)


def _get_error_log_text():
    log_file = CUR_DIR / "nginx_error.log"
    if log_file.exists():
        return log_file.read_text(encoding="utf-8", errors="ignore")
    return ""


def test_input_ids_decode_non_stream(setup_teardown):
    """Non-stream request with tokenizer enabled through full PD pipeline.
    The EPD mock prefill returns kv_transfer_params, so the proxy generates
    a decode request body with input_ids injected."""
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "input-ids-decode-001"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Hello world"}],
        "stream": False
    }

    log_before = _get_error_log_text()
    before_tokenize_count = log_before.count("[Tokenize-")

    response = requests.post(url, headers=headers, json=data, timeout=120)
    print("Status Code:", response.status_code)
    print("Response Body (preview):", response.text[:300])
    assert response.status_code == 200

    time.sleep(1)
    log_after = _get_error_log_text()
    after_tokenize_count = log_after.count("[Tokenize-")
    assert after_tokenize_count > before_tokenize_count, \
        "Tokenizer should have been invoked (Tokenize log expected)"

    new_log = log_after[len(log_before):]
    assert "kv_transfer_params is null" not in new_log and \
           "kv_transfer_params is absent" not in new_log, \
        "Expected kv_transfer_params to be present (EPD mock should return it)"


def test_input_ids_decode_stream(setup_teardown):
    """Streaming request with tokenizer through full PD pipeline.
    Exercises decode body input_ids injection for streaming mode."""
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "input-ids-decode-stream-001"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True
    }

    log_before = _get_error_log_text()
    before_count = log_before.count("[Tokenize-")

    response = requests.post(url, headers=headers, json=data, timeout=120)
    print("Status Code:", response.status_code)
    content_type = response.headers.get("content-type", "")
    print("Content-Type:", content_type)
    assert response.status_code == 200

    time.sleep(1)
    log_after = _get_error_log_text()
    after_count = log_after.count("[Tokenize-")
    assert after_count > before_count, \
        "Tokenizer should have been invoked for streaming request"


def test_input_ids_decode_long_prompt(setup_teardown):
    """Long prompt to exercise input_ids injection with a larger token array."""
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "input-ids-decode-long-001"
    }
    long_content = "Please explain the following topics in detail: " + \
                   ", ".join([f"topic_{i}" for i in range(50)])
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": long_content}],
        "stream": False
    }

    log_before = _get_error_log_text()
    before_count = log_before.count("[Tokenize-")

    response = requests.post(url, headers=headers, json=data, timeout=120)
    print("Status Code:", response.status_code)
    print("Response Body (preview):", response.text[:300])
    assert response.status_code == 200

    time.sleep(1)
    log_after = _get_error_log_text()
    after_count = log_after.count("[Tokenize-")
    assert after_count > before_count, \
        "Tokenizer should have been invoked for long prompt"


def test_input_ids_decode_multi_turn(setup_teardown):
    """Multi-turn conversation producing more tokens for the injection loop."""
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "input-ids-decode-multi-001"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "stream": False
    }

    log_before = _get_error_log_text()
    before_count = log_before.count("[Tokenize-")

    response = requests.post(url, headers=headers, json=data, timeout=120)
    print("Status Code:", response.status_code)
    print("Response Body (preview):", response.text[:300])
    assert response.status_code == 200

    time.sleep(1)
    log_after = _get_error_log_text()
    after_count = log_after.count("[Tokenize-")
    assert after_count > before_count, \
        "Tokenizer should have been invoked for multi-turn conversation"
