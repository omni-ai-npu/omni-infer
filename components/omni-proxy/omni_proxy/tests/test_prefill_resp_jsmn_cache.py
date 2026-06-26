# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
JSMN cache reuse tests for prefill_response_body.

Verifies that omni_prefill_resp_jsmn_cached reuses parsed tokens when
the same prefill_response_body JSON is encountered multiple times per request.
"""

import os
import time
from pathlib import Path

import pytest
import requests

import port_manager
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess


PREFILL_NUM = 2
DECODE_NUM = 2
proxy_port = 7100
CUR_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def pd_setup_teardown():
    global proxy_port

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file_epd()
        proxy_port = ports.get("proxy_port", 7100)
        yield
        return

    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Failed to start PD mock server")

    time.sleep(2)

    ret = setup_proxy(
        proxy_port=proxy_port,
        prefill_port_list=prefill_port_list,
        decode_port_list=decode_port_list,
    )
    if ret == -1:
        cleanup_subprocess(processes)
        pytest.fail("Failed to start omni-proxy")

    yield

    teardown_proxy()
    cleanup_subprocess(processes)


def test_prefill_resp_jsmn_cached_reuse_log_with_proxy(pd_setup_teardown):
    error_log = CUR_DIR / "nginx_error.log"
    before_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    before_hits = before_text.count("omni_prefill_resp_jsmn_cached: reuse cached tokens")

    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "prefill-jsmn-cache-reuse",
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

    # Wait for log flush
    after_hits = before_hits
    for _ in range(10):
        after_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
        after_hits = after_text.count("omni_prefill_resp_jsmn_cached: reuse cached tokens")
        if after_hits > before_hits:
            break
        time.sleep(0.2)

    assert after_hits > before_hits, (
        "Expected omni_prefill_resp_jsmn_cached cache reuse log to increase after request, "
        "but no new 'reuse cached tokens' entry was found."
    )