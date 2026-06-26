# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Large Prompt JSON Parsing Tests for omni-proxy (PD separation mode).

Test cases verify that jsmn_parse auto-expansion works correctly with
large token counts (e.g., prompts with 256k+ tokens).

Test topology:
    - PD separation mode (no encode endpoint)
    - Uses vllm_mock_server_epd.py with prefill + decode nodes
    - omni-proxy configured without encode-endpoints
"""

import os
import pytest
import requests
import json
import time
from pathlib import Path

from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess
import port_manager


# Configuration - PD mode (no encode)
PREFILL_NUM = 2
DECODE_NUM = 2
proxy_port = 7100


@pytest.fixture(scope="module")
def pd_setup_teardown():
    """
    Setup: Start mock server (prefill + decode only) + omni-proxy in PD mode.
    Teardown: Stop omni-proxy and mock server.
    """
    global proxy_port

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file_epd()
        proxy_port = ports.get("proxy_port", 7100)
        print(f"\n[DEBUG] Skipping fixture, {proxy_port=}")
        yield
        return

    # Load ports for prefill + decode (no encode for PD mode)
    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    print(f"\n[SETUP] Starting PD mock server: prefill={prefill_port_list}, decode={decode_port_list}")

    # Start mock server with prefill + decode only (encode_num=0)
    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Failed to start PD mock server")

    time.sleep(2)

    print(f"\n[SETUP] Starting omni-proxy on port {proxy_port} (PD mode)")

    # Start omni-proxy WITHOUT encode-endpoints to trigger PD mode
    # Use stream_ops="off" to preserve original stream setting (don't force streaming)
    ret = setup_proxy(
        proxy_port=proxy_port,
        prefill_port_list=prefill_port_list,
        decode_port_list=decode_port_list,
        stream_ops="off",
    )
    if ret.returncode != 0:
        cleanup_subprocess(processes)
        pytest.fail(f"Failed to start omni-proxy")

    yield

    # --- Teardown ---
    print("\n[TEARDOWN] Stopping omni-proxy...")
    teardown_proxy()

    print(f"\n[TEARDOWN] Stopping PD mock server...")
    cleanup_subprocess(processes)


def make_large_prompt_tokens(num_tokens):
    """
    Generate a token ID list to create a large JSON
    that will stress-test jsmn_parse token expansion.
    """
    base_pattern = [148899, 24, 13, 17, 13, 809, 28523, 662]
    tokens = []
    while len(tokens) < num_tokens:
        tokens.extend(base_pattern)
    return tokens[:num_tokens]


def test_large_prompt_non_stream(pd_setup_teardown):
    """
    Test non-streaming /v1/completions request with large prompt token list.
    Verifies jsmn_parse auto-expansion works correctly.
    """
    url = f"http://127.0.0.1:{proxy_port}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-large-prompt-001"
    }

    # 200 turns * 2 messages/turn * ~8 tokens/message = ~3200 tokens
    # Use enough tokens to stress-test jsmn_parse
    prompt_tokens = make_large_prompt_tokens(3200)

    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "prompt": prompt_tokens,
        "stream": False
    }

    print(f"\n[TEST] Sending large prompt non-stream request (prompt len={len(prompt_tokens)}) to {url}")
    response = requests.post(url, headers=headers, json=data, timeout=60)

    print(f"[TEST] Response status: {response.status_code}")
    print(f"[TEST] Response body (preview): {response.text[:500]}")

    # Assert HTTP 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    # Assert JSON response contains choices
    resp_json = response.json()
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"
    assert len(resp_json["choices"]) > 0, "Expected at least one choice in response"

    print(f"[TEST] Large prompt non-stream test passed.")


def test_large_prompt_stream(pd_setup_teardown):
    """
    Test streaming /v1/completions request with large prompt token list.
    Verifies jsmn_parse auto-expansion works with streaming responses.
    """
    url = f"http://127.0.0.1:{proxy_port}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-large-prompt-stream-001"
    }

    prompt_tokens = make_large_prompt_tokens(3200)

    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "prompt": prompt_tokens,
        "stream": True
    }

    print(f"\n[TEST] Sending large prompt stream request (prompt len={len(prompt_tokens)}) to {url}")
    token_cnt = 0

    with requests.post(url, headers=headers, json=data, stream=True, timeout=60) as resp:
        print(f"[TEST] Response status: {resp.status_code}")
        assert resp.status_code == 200

        for line in resp.iter_lines():
            if line:
                if line.startswith(b"data:"):
                    json_str = line[len(b"data:"):].strip()
                    if json_str == b"[DONE]":
                        print(f"\n[TEST] Stream finished. Total tokens: {token_cnt}")
                        break
                    try:
                        chunk = json.loads(json_str)
                        # Completions API uses "text" field, chat uses "delta"
                        content = chunk.get("choices", [{}])[0].get("text", "") or \
                                  chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            token_cnt += 1
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass

    assert token_cnt > 0, "Expected at least one token in stream"
    print(f"\n[TEST] Large prompt stream test passed.")


def test_repeated_parsing_reuses_cached_tokens(pd_setup_teardown):
    """
    Test that repeated parsing of the same JSON reuses cached tokens.
    Verifies omni_origin_body_jsmn_cached cache hit behavior.
    """
    error_log = Path(__file__).parent / "nginx_error.log"
    backup_log = Path(__file__).parent / "nginx_error.log.bak"
    original_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    backup_log.write_text(original_text, encoding="utf-8")
    error_log.write_text("", encoding="utf-8")

    url = f"http://127.0.0.1:{proxy_port}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-cache-reuse-001",
    }

    prompt_tokens = make_large_prompt_tokens(1600)

    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 10,
        "prompt": prompt_tokens,
        "stream": False,
    }

    # First request
    response1 = requests.post(url, headers=headers, json=data, timeout=30)
    assert response1.status_code == 200, f"Expected 200, got {response1.status_code}: {response1.text}"

    # Second request with same JSON (should hit cache)
    headers["X-Request-Id"] = "test-cache-reuse-002"
    response2 = requests.post(url, headers=headers, json=data, timeout=30)
    assert response2.status_code == 200, f"Expected 200, got {response2.status_code}: {response2.text}"

    # Wait and check nginx error log for cache reuse log
    time.sleep(0.5)
    after_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    cache_hit_count = after_text.count("omni_origin_body_jsmn_cached: reuse cached tokens")

    print(f"\n[TEST] Cache reuse log entries: {cache_hit_count}")
    assert cache_hit_count > 0, (
        "Expected 'omni_origin_body_jsmn_cached: reuse cached tokens' in error log after second request, "
        "but no cache reuse entry was found."
    )
    print(f"[TEST] Cache reuse test passed.")


def test_no_jsmn_nomem_error_in_logs(pd_setup_teardown):
    """
    Test that nginx error log does NOT contain JSMN_ERROR_NOMEM after requests.
    This confirms auto-expansion is working and not failing with out-of-memory.
    """
    error_log = Path(__file__).parent / "nginx_error.log"
    backup_log = Path(__file__).parent / "nginx_error.log.bak"
    original_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    backup_log.write_text(original_text, encoding="utf-8")
    error_log.write_text("", encoding="utf-8")

    url = f"http://127.0.0.1:{proxy_port}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-no-nomem-001",
    }

    prompt_tokens = make_large_prompt_tokens(4800)

    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "prompt": prompt_tokens,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=data, timeout=60)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    # Wait for log to flush
    time.sleep(0.5)

    after_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""

    # Check no JSMN_ERROR_NOMEM or "jsmn_parse failed" errors
    nomem_count = after_text.count("JSMN_ERROR_NOMEM")
    parse_fail_count = after_text.count("jsmn_parse failed")

    print(f"\n[TEST] JSMN_ERROR_NOMEM entries: {nomem_count}, parse failures: {parse_fail_count}")

    assert nomem_count == 0, f"Found {nomem_count} JSMN_ERROR_NOMEM entries in error log - auto-expansion may have failed"
    assert parse_fail_count == 0, f"Found {parse_fail_count} jsmn_parse failures in error log"

    print(f"[TEST] No JSMN_ERROR_NOMEM in logs - auto-expansion working correctly.")


def test_large_prompt_token_2049(pd_setup_teardown):
    """
    Test /v1/completions request with prompt token list length of 2049.
    Verifies proxy can correctly parse prompts near common token limits.
    """
    error_log = Path(__file__).parent / "nginx_error.log"
    backup_log = Path(__file__).parent / "nginx_error.log.bak"
    original_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    backup_log.write_text(original_text, encoding="utf-8")
    error_log.write_text("", encoding="utf-8")

    url = f"http://127.0.0.1:{proxy_port}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-token-2049-001",
    }

    prompt_tokens = make_large_prompt_tokens(2049)

    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 10,
        "prompt": prompt_tokens,
        "stream": False,
    }

    print(f"\n[TEST] Sending request with 2049 tokens (prompt len={len(prompt_tokens)}) to {url}")
    response = requests.post(url, headers=headers, json=data, timeout=60)

    print(f"[TEST] Response status: {response.status_code}")
    print(f"[TEST] Response body (preview): {response.text[:500]}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"
    assert len(resp_json["choices"]) > 0, "Expected at least one choice in response"

    # Verify no jsmn parse errors in logs
    time.sleep(0.5)
    after_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    nomem_count = after_text.count("JSMN_ERROR_NOMEM")
    parse_fail_count = after_text.count("jsmn_parse failed")
    parse_fail_count2 = after_text.count("first pass failed")

    print(f"[TEST] JSMN_ERROR_NOMEM entries: {nomem_count}, parse failures: {parse_fail_count}, first pass failures: {parse_fail_count2}")

    assert nomem_count == 0, f"Found {nomem_count} JSMN_ERROR_NOMEM entries"
    assert parse_fail_count == 0, f"Found {parse_fail_count} jsmn_parse failures"
    assert parse_fail_count2 == 0, f"Found {parse_fail_count2} first pass failures"

    print(f"[TEST] Large prompt token 2049 test passed.")


def test_completions_prompt_token_2049(pd_setup_teardown):
    """
    Test /v1/completions endpoint with prompt token list length of 2049.
    Verifies proxy can correctly parse the completions API format with large token IDs.
    """
    error_log = Path(__file__).parent / "nginx_error.log"
    backup_log = Path(__file__).parent / "nginx_error.log.bak"
    original_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    backup_log.write_text(original_text, encoding="utf-8")
    error_log.write_text("", encoding="utf-8")

    url = f"http://127.0.0.1:{proxy_port}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-completions-2049-001",
    }

    # Generate a token ID list with 2049 elements
    # Using a repeating pattern of common token IDs
    base_pattern = [148899, 24, 13, 17, 13, 809, 28523, 662]  # 8 tokens per pattern
    prompt_token_ids = base_pattern * 256 + [148899]  # 2048 + 1 = 2049
    assert len(prompt_token_ids) == 2049, f"Expected 2049 tokens, got {len(prompt_token_ids)}"

    data = {
        "model": "deepseek",
        "prompt": prompt_token_ids,
        "max_tokens": 10,
        "temperature": 0,
        "stream": False,
    }

    print(f"\n[TEST] Sending /v1/completions request with prompt len={len(prompt_token_ids)} to {url}")
    response = requests.post(url, headers=headers, json=data, timeout=60)

    print(f"[TEST] Response status: {response.status_code}")
    print(f"[TEST] Response body (preview): {response.text[:500]}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"
    assert len(resp_json["choices"]) > 0, "Expected at least one choice in response"

    # Verify no jsmn parse errors in logs
    time.sleep(0.5)
    after_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    nomem_count = after_text.count("JSMN_ERROR_NOMEM")
    parse_fail_count = after_text.count("jsmn_parse failed")
    parse_fail_count2 = after_text.count("first pass failed")

    print(f"[TEST] JSMN_ERROR_NOMEM entries: {nomem_count}, parse failures: {parse_fail_count}, first pass failures: {parse_fail_count2}")

    assert nomem_count == 0, f"Found {nomem_count} JSMN_ERROR_NOMEM entries"
    assert parse_fail_count == 0, f"Found {parse_fail_count} jsmn_parse failures"
    assert parse_fail_count2 == 0, f"Found {parse_fail_count2} first pass failures"

    print(f"[TEST] Completions prompt token 2049 test passed.")


@pytest.mark.parametrize("num_tokens", [20, 256, 257, 511, 512, 513, 1023, 1024, 1025, 262144])
def test_completions_boundary_token_counts(pd_setup_teardown, num_tokens):
    """
    Test /v1/completions endpoint with various boundary token list lengths.
    Covers power-of-2 and adjacent values: 20, 256, 257, 511, 512, 513, 1023, 1024, 1025, 256k.
    """
    url = f"http://127.0.0.1:{proxy_port}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": f"test-boundary-{num_tokens}-001",
    }

    prompt_tokens = make_large_prompt_tokens(num_tokens)
    assert len(prompt_tokens) == num_tokens, f"Expected {num_tokens} tokens, got {len(prompt_tokens)}"

    data = {
        "model": "deepseek",
        "prompt": prompt_tokens,
        "max_tokens": 10,
        "temperature": 0,
        "stream": False,
    }

    print(f"\n[TEST] Sending request with {num_tokens} tokens to {url}")
    response = requests.post(url, headers=headers, json=data, timeout=60)

    print(f"[TEST] Response status: {response.status_code}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"
    assert len(resp_json["choices"]) > 0, "Expected at least one choice in response"

    print(f"[TEST] Boundary token count {num_tokens} test passed.")

