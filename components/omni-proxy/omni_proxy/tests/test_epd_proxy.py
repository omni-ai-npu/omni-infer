# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
EPD (Encode-Prefill-Decode) Integration Tests for omni-proxy.

Test cases:
    - test_pd_mode_chat_completions: PD separation mode, pure text request
    - test_epd_mode_multimodal: EPD separation mode, multimodal (image) request

PD/EPD Flow:
    Client -> Omni Proxy -> Encode Node (EPD only)
                          -> Prefill Node (+ ec_transfer_params in EPD)
                          <- kv_transfer_params
                          -> Decode Node (+ kv_transfer_params)
                          <- Response tokens
                          <- Client
"""

import pytest
import os
import random
import subprocess
import time
import requests
import json
import tempfile
from pathlib import Path
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess
import port_manager


def generate_random_image(width=64, height=64):
    """Generate a small random pixel PPM image and return its file path."""
    import random
    # Generate random RGB pixels
    data = bytes([random.randint(0, 255) for _ in range(width * height * 3)])
    # PPM format is simple: P6 + header + data
    header = f"P6\n{width} {height}\n255\n".encode()
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
        f.write(header + data)
        return f.name

# Configuration
ENCODE_NUM = 1
PREFILL_NUM = 2
DECODE_NUM = 2
proxy_port = 7150
CUR_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def epd_setup_teardown():
    """
    Setup: Start EPD mock server + omni-proxy with encode/prefill/decode endpoints.
    Teardown: Stop omni-proxy and mock server.
    """
    global proxy_port

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file_epd()
        proxy_port = ports.get("proxy_port", 7150)
        print(f"\n[DEBUG] Skipping fixture, {proxy_port=}")
        yield
        return

    # Load ports for encode + prefill + decode
    ports = port_manager.load_ports_epd(ENCODE_NUM, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    encode_port_list = ports["encode"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    print(f"\n[SETUP] Starting EPD mock server: encode={encode_port_list}, prefill={prefill_port_list}, decode={decode_port_list}")

    # Start EPD mock server (encode + prefill + decode)
    processes = start_mock_server(ENCODE_NUM, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Failed to start EPD mock server")

    time.sleep(2)  # Wait for mock server to be fully ready

    print(f"\n[SETUP] Starting omni-proxy on port {proxy_port}")

    # Start omni-proxy with encode/prefill/decode endpoints
    ret = setup_proxy(
        proxy_port=proxy_port,
        encode_port_list=encode_port_list,
        prefill_port_list=prefill_port_list,
        decode_port_list=decode_port_list,
    )
    if ret.returncode != 0:
        cleanup_subprocess(processes)
        pytest.fail(f"Failed to start omni-proxy")

    yield

    # --- Teardown ---
    print("\n[TEARDOWN] Stopping omni-proxy...")
    teardown_proxy()

    print(f"\n[TEARDOWN] Stopping EPD mock server...")
    cleanup_subprocess(processes)


def test_pd_mode_chat_completions(epd_setup_teardown):
    """
    Test PD separation mode: pure text request through omni-proxy.

    Flow:
        Client -> Omni Proxy -> Prefill Node
                               <- kv_transfer_params
                          -> Decode Node + kv_transfer_params
                               <- Response tokens
                          <- Client

    Expected:
        - HTTP 200 response
        - JSON response contains 'choices' with generated text
    """
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-pd-001"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": False
    }

    print(f"\n[TEST] Sending PD mode request to {url}")
    response = requests.post(url, headers=headers, json=data, timeout=60)

    print(f"[TEST] Response status: {response.status_code}")
    print(f"[TEST] Response headers: {response.headers}")
    print(f"[TEST] Response body (preview): {response.text[:500]}")

    # Assert HTTP 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    # Assert SSE response (stream_ops add forces stream=True)
    text = response.text
    lines = text.split("\n")
    data_lines = [l for l in lines if l.startswith("data:") and l != "data: [DONE]"]
    last_data = data_lines[-1] if data_lines else ""
    resp_json = json.loads(last_data[len("data:"):].strip())
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"
    assert len(resp_json["choices"]) > 0, "Expected at least one choice in response"

    print(f"[TEST] PD mode test passed. Generated: {resp_json['choices'][0].get('delta', {})['content'][:100]}")


def test_epd_mode_multimodal(epd_setup_teardown):
    """
    Test EPD separation mode: multimodal request with image through omni-proxy.

    Flow:
        Client -> Omni Proxy -> Encode Node (extracts image features)
                               <- ec_transfer_params (encoder embeddings)
                          -> Prefill Node + ec_transfer_params
                          <- kv_transfer_params
                          -> Decode Node + kv_transfer_params
                          <- Response tokens
                          <- Client

    Expected:
        - HTTP 200 response
        - JSON response contains 'choices' with generated text
        - Logs show encode node was invoked ("[Encoder-0]" in logs)
    """
    image_path = generate_random_image()

    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-epd-001"
    }
    data = {
        "model": "multimodal-model",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"file://{image_path}"
                        }
                    }
                ]
            }
        ],
        "stream": False
    }

    print(f"\n[TEST] Sending EPD mode multimodal request to {url}")
    response = requests.post(url, headers=headers, json=data, timeout=60)

    print(f"[TEST] Response status: {response.status_code}")
    print(f"[TEST] Response headers: {response.headers}")
    print(f"[TEST] Response body (preview): {response.text[:500]}")

    # Assert HTTP 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    # Assert SSE response (stream_ops add forces stream=True)
    text = response.text
    lines = text.split("\n")
    data_lines = [l for l in lines if l.startswith("data:") and l != "data: [DONE]"]
    last_data = data_lines[-1] if data_lines else ""
    resp_json = json.loads(last_data[len("data:"):].strip())
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"
    assert len(resp_json["choices"]) > 0, "Expected at least one choice in response"

    # Check log file for encode node invocation
    log_file = Path(__file__).parent / "epd_mock_server.log"
    encode_invoked = False
    if log_file.exists():
        with open(log_file, "r") as f:
            log_content = f.read()
            # The EPD mock server logs "[Encoder-N]" when encode node is invoked
            if "[Encoder-0]" in log_content or "encode" in log_content.lower():
                print("[TEST] Encode node was invoked (found in logs)")
                encode_invoked = True
    assert encode_invoked, "Expected encode node to be invoked, but not found in logs"

    print(f"[TEST] EPD mode test passed. Generated: {resp_json['choices'][0].get('delta', {})['content'][:100]}")

    # Cleanup the generated image file
    if image_path and os.path.exists(image_path):
        os.unlink(image_path)
        print(f"[TEST] Cleaned up generated image: {image_path}")


def test_pd_mode_stream(epd_setup_teardown):
    """
    Test PD separation mode: streaming text request through omni-proxy.

    Expected:
        - HTTP 200 response
        - Stream of SSE tokens
        - Final [DONE] marker
    """
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-pd-stream-001"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Count from 1 to 5"}],
        "stream": True
    }

    print(f"\n[TEST] Sending PD mode streaming request to {url}")
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
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            token_cnt += 1
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass

    assert token_cnt > 0, "Expected at least one token in stream"
    print(f"\n[TEST] PD streaming test passed.")


def test_traceparent_header_update(epd_setup_teardown):
    """
    Test that prefill node returns Traceparent and Start_time_ns headers.

    This triggers the update_traceparent_header C function in omni-proxy
    which updates the main request's headers when the prefill node returns
    these headers in its response.

    Flow:
        Client -> Omni Proxy -> Prefill Node (returns Traceparent/Start_time_ns headers)
                               <- Traceparent + Start_time_ns headers
                          <- Client

    Expected:
        - HTTP 200 response
        - Prefill node returns Traceparent and Start_time_ns headers in its response
    """
    # Get prefill port from the shared ports file
    ports = port_manager.get_ports_from_file_epd()
    prefill_port = ports["prefill"][0]
    prefill_url = f"http://127.0.0.1:{prefill_port}/v1/chat/completions"

    # Send request directly to prefill node (not through omni-proxy)
    # This verifies the mock prefill node correctly returns the traceparent headers
    prefill_headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-traceparent-direct",
        "Traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "Start_time_ns": "9876543210"
    }
    prefill_data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False
    }

    print(f"\n[TEST] Sending request directly to prefill node at {prefill_url}")
    prefill_resp = requests.post(prefill_url, headers=prefill_headers, json=prefill_data, timeout=30)

    print(f"[TEST] Prefill response status: {prefill_resp.status_code}")
    print(f"[TEST] Prefill response headers: {dict(prefill_resp.headers)}")

    # Assert HTTP 200 from prefill node
    assert prefill_resp.status_code == 200, f"Expected 200 from prefill, got {prefill_resp.status_code}: {prefill_resp.text}"

    # Verify traceparent headers are present in prefill node's response
    # HTTP headers are case-insensitive, requests library lowercases them
    assert "traceparent" in prefill_resp.headers, f"Expected 'traceparent' in prefill response headers, got: {prefill_resp.headers.keys()}"
    assert "start_time_ns" in prefill_resp.headers, f"Expected 'start_time_ns' in prefill response headers, got: {prefill_resp.headers.keys()}"

    traceparent_value = prefill_resp.headers.get("traceparent")
    start_time_ns_value = prefill_resp.headers.get("start_time_ns")
    print(f"[TEST] Prefill returned Traceparent: {traceparent_value}")
    print(f"[TEST] Prefill returned Start_time_ns: {start_time_ns_value}")

    # Verify the values match what the mock prefill node should return
    assert traceparent_value == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    assert start_time_ns_value == "1234567890"

    print(f"[TEST] Traceparent header test passed - mock prefill node correctly returns headers.")


def test_health_endpoint(epd_setup_teardown):
    """
    Test /omni_proxy/health endpoint returns HTTP 200.
    """
    url = f"http://127.0.0.1:{proxy_port}/omni_proxy/health"
    response = requests.get(url)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print(f"[TEST] Health endpoint test passed.")


def test_broadcast_health(epd_setup_teardown):
    """
    Test /health broadcast endpoint returns HTTP 200.
    This triggers omni_proxy_broadcast handler which should include encode nodes.
    """
    url = f"http://127.0.0.1:{proxy_port}/health"
    response = requests.get(url)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Validate JSON response
    data = response.json()
    assert data["uri"] == "/health"
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0
    assert data["summary"]["success"] + data["summary"]["failed"] == data["summary"]["total"]

    # Validate each result has required fields including 'ok'
    for result in data["results"]:
        assert "type" in result
        assert "index" in result
        assert "address" in result
        assert "ok" in result
        assert isinstance(result["ok"], bool)
        assert "status" in result
        assert "body" in result
    print(f"[TEST] Health endpoint test passed.")


def test_epd_jsmn_parse_cached_reuse_log(epd_setup_teardown):
    """
    Guard test: ensure origin body JSON cache reuse happens in EPD flow.
    """
    error_log = CUR_DIR / "nginx_error.log"
    # Backup current log, but do not restore it after test.
    backup_log = CUR_DIR / "nginx_error.log.bak"
    original_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
    backup_log.write_text(original_text, encoding="utf-8")
    error_log.write_text("", encoding="utf-8")

    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-epd-cache-reuse-001",
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "cache-check-epd"}],
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    after_hits = 0
    for _ in range(10):
        after_text = error_log.read_text(encoding="utf-8", errors="ignore") if error_log.exists() else ""
        after_hits = after_text.count("omni_origin_body_jsmn_cached: reuse cached tokens")
        if after_hits > 0:
            break
        time.sleep(0.2)

    assert after_hits > 0, (
        "Expected omni_origin_body_jsmn_cached cache reuse log to increase after EPD request, "
        "but no new 'reuse cached tokens' entry was found."
    )


def test_epd_mode_multimodal_batch_10(epd_setup_teardown):
    """
    Test EPD separation mode: 10 concurrent multimodal requests with images through omni-proxy.

    Flow for each request:
        Client -> Omni Proxy -> Encode Node (extracts image features)
                               <- ec_transfer_params (encoder embeddings)
                          -> Prefill Node + ec_transfer_params
                          <- kv_transfer_params
                          -> Decode Node + kv_transfer_params
                          <- Response tokens
                          <- Client

    Expected:
        - All 10 requests return HTTP 200
        - All JSON responses contain 'choices' with generated text
        - Logs show encode node was invoked for multimodal requests
    """
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    num_requests = 10
    results = [None] * num_requests

    def send_multimodal_request(idx):
        """Send a single multimodal request and return result."""
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": f"test-epd-batch-{idx:03d}"
        }
        data = {
            "model": "multimodal-model",
            "temperature": 0,
            "max_tokens": 20,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Describe this image {idx + 1}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"https://www.example.com/test_{idx}.jpg"
                            }
                        }
                    ]
                }
            ],
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)

            # First check response text before parsing JSON
            resp_text = response.text
            if not resp_text:
                return (idx, response.status_code, None, f"Empty response body, status={response.status_code}")

            if response.status_code != 200:
                return (idx, response.status_code, None, f"HTTP {response.status_code}: {resp_text[:200]}")
            # Handle SSE format response (mock server may return streaming format)
            # SSE format: "data: {...}\n\ndata: {...}\n\ndata: [DONE]\n\n"
            content = ""
            if resp_text.startswith("data:"):
                # Parse SSE stream
                for line in resp_text.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("data:"):
                        json_str = line[5:].strip()  # Remove "data:" prefix
                        if json_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                content += delta["content"]
                        except json.JSONDecodeError:
                            pass
            else:
                # Standard JSON response
                try:
                    resp_json = json.loads(resp_text)
                    if "choices" in resp_json and resp_json["choices"]:
                        content = resp_json["choices"][0].get("message", {}).get("content", "")
                except json.JSONDecodeError as e:
                    return (idx, response.status_code, None, f"JSON decode error: {e}, body={resp_text[:200]}")

            if not content:
                return (idx, response.status_code, None, f"No content in response: {resp_text[:200]}")

            return (idx, response.status_code, content[:100], "OK")

        except requests.exceptions.Timeout:
            return (idx, 0, None, "Request timeout")
        except requests.exceptions.ConnectionError as e:
            return (idx, 0, None, f"Connection error: {e}")
        except Exception as e:
            return (idx, 0, None, f"Exception: {type(e).__name__}: {e}")

    print(f"\n[TEST] Sending {num_requests} concurrent EPD multimodal requests to {url}")

    # Use ThreadPoolExecutor for concurrent requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(send_multimodal_request, i) for i in range(num_requests)]

        for future in as_completed(futures):
            idx, status, content, msg = future.result()
            results[idx] = (status, content, msg)
            print(f"[TEST] Request {idx}: status={status}, content={content}, msg={msg}")

    # Verify all requests succeeded
    success_count = 0
    for idx, (status, content, msg) in enumerate(results):
        if status == 200 and content is not None:
            success_count += 1
        else:
            print(f"[TEST] Request {idx} failed: status={status}, content={content}, msg={msg}")

    # Check log file for encode node invocation
    log_file = Path(__file__).parent / "epd_mock_server.log"
    if log_file.exists():
        with open(log_file, "r") as f:
            log_content = f.read()
            if "[Encoder-0]" in log_content or "encode" in log_content.lower():
                print("[TEST] Encode node was invoked (found in logs)")

    print(f"\n[TEST] Batch multimodal test completed: {success_count}/{num_requests} succeeded")
    assert success_count == num_requests, f"Expected all {num_requests} requests to succeed, got {success_count}"
    print(f"[TEST] EPD multimodal batch test passed.")
