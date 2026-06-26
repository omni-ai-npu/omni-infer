# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Tests for kv_transfer_params handling - null and absent cases.

When prefill returns kv_transfer_params: null, the proxy should directly return
the prefill response to client without forwarding to decode, and strip the
kv_transfer_params field from the response.

When prefill response does not contain kv_transfer_params at all, the proxy
should also directly return the prefill response to client.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

import port_manager
from run_proxy import setup_proxy, teardown_proxy


# =============================================================================
# Custom Mock Servers
# =============================================================================

class PrefillNullKVHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params: null in response.
    The field is placed AFTER usage (last position).
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            request_data = {}

        stream_mode = request_data.get("stream", False)
        max_tokens = request_data.get("max_tokens", 20)

        if stream_mode:
            # Streaming response with kv_transfer_params: null in final chunk
            for i in range(max_tokens):
                chunk = {
                    "id": f"prefill-{self.server.request_count}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "mock-prefill",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"tok{i}"},
                        "finish_reason": None
                    }]
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                time.sleep(0.01)

            final_chunk = {
                "id": f"prefill-{self.server.request_count}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "mock-prefill",
                "choices": [{
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                },
                "kv_transfer_params": None
            }
            self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            # Non-streaming response: kv_transfer_params at the end
            response = {
                "id": f"prefill-{self.server.request_count}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mock-prefill",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Prefill response with null kv_transfer_params"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                },
                "kv_transfer_params": None
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class PrefillNullKVFirstHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server where kv_transfer_params: null is the FIRST field.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        # Manually construct JSON with kv_transfer_params as first key
        response_str = json.dumps({
            "kv_transfer_params": None,
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "kv_transfer_params as first field"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(response_str.encode())

    def log_message(self, format, *args):
        return


class PrefillNullKVMiddleHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server where kv_transfer_params: null is in the MIDDLE.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        # Manually construct JSON with kv_transfer_params in the middle
        response_str = json.dumps({
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "kv_transfer_params": None,
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "kv_transfer_params in middle"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(response_str.encode())

    def log_message(self, format, *args):
        return


class PrefillAbsentKVHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that does NOT include kv_transfer_params at all.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        # Response without kv_transfer_params field
        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response without kv_transfer_params"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class DecodeCountHandler(BaseHTTPRequestHandler):
    """
    Mock decode server that tracks request count.
    """
    def do_POST(self):
        self.server.decode_request_count += 1
        print(f"[DecodeCountHandler] Received request #{self.server.decode_request_count}")

        response = {
            "id": f"decode-{self.server.decode_request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-decode",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Decode response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


def start_mock_server(port, handler_class):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_class)
    server.request_count = 0
    server.decode_request_count = 0
    server.port = port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stop_mock_server(server):
    if server:
        server.shutdown()
        server.server_close()


def get_free_port_in_range(start, end):
    import socket
    for port in range(start, end):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError("No free port found")


def make_proxy_fixture(prefill_handler_class, port_base):
    """Factory to create a pytest fixture with a specific prefill handler."""

    @pytest.fixture(scope="module")
    def fixture():
        proxy_port = get_free_port_in_range(port_base, port_base + 100)
        prefill_port = get_free_port_in_range(port_base + 100, port_base + 200)
        decode_port = get_free_port_in_range(port_base + 200, port_base + 300)

        print(f"\n[SETUP] Starting mock servers: prefill={prefill_port}, decode={decode_port}")
        print(f"[SETUP] Proxy port: {proxy_port}")

        prefill_server = start_mock_server(prefill_port, prefill_handler_class)
        decode_server = start_mock_server(decode_port, DecodeCountHandler)

        time.sleep(1)

        ret = setup_proxy(
            proxy_port=proxy_port,
            prefill_port_list=[prefill_port],
            decode_port_list=[decode_port],
        )
        if ret.returncode != 0:
            stop_mock_server(prefill_server)
            stop_mock_server(decode_server)
            pytest.fail("Failed to start omni-proxy")

        yield {
            "proxy_port": proxy_port,
            "prefill_server": prefill_server,
            "decode_server": decode_server,
        }

        print("\n[TEARDOWN] Stopping omni-proxy...")
        teardown_proxy()

        print("[TEARDOWN] Stopping mock servers...")
        stop_mock_server(prefill_server)
        stop_mock_server(decode_server)

    return fixture


# =============================================================================
# Fixtures for each scenario
# =============================================================================

null_kv_setup_teardown = make_proxy_fixture(PrefillNullKVHandler, 50000)
null_kv_first_setup_teardown = make_proxy_fixture(PrefillNullKVFirstHandler, 50300)
null_kv_middle_setup_teardown = make_proxy_fixture(PrefillNullKVMiddleHandler, 50600)
absent_kv_setup_teardown = make_proxy_fixture(PrefillAbsentKVHandler, 50900)


# =============================================================================
# Helper
# =============================================================================

def send_non_streaming_request(proxy_port, request_id="test"):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": request_id
    }
    data = {
        "model": "mock-model",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    return requests.post(url, headers=headers, json=data, timeout=30)


# =============================================================================
# Tests: kv_transfer_params: null at end (original scenario)
# =============================================================================

def test_kv_transfer_params_null_no_decode_forward(null_kv_setup_teardown):
    """
    When prefill returns kv_transfer_params: null (at end of JSON),
    proxy returns prefill response directly WITHOUT forwarding to decode,
    and the kv_transfer_params field is stripped from the response.
    """
    context = null_kv_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-null-001")

    print(f"[TEST] Response status: {response.status_code}")
    print(f"[TEST] Response body: {response.text[:500]}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped from response, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_null_no_decode_forward PASSED")


def test_kv_transfer_params_null_streaming(null_kv_setup_teardown):
    """
    Streaming request with kv_transfer_params: null should also return
    prefill response directly without forwarding to decode.
    """
    context = null_kv_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "test-kv-null-stream-001"
    }
    data = {
        "model": "mock-model",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True
    }

    received_data = False
    with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as resp:
        assert resp.status_code == 200

        for line in resp.iter_lines():
            if line:
                received_data = True
                if line.startswith(b"data:"):
                    json_str = line[len(b"data:"):].strip()
                    if json_str == b"[DONE]":
                        break

    assert received_data, "Expected to receive stream data"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_null_streaming PASSED")


# =============================================================================
# Tests: kv_transfer_params: null as first field
# =============================================================================

def test_kv_transfer_params_null_first_field(null_kv_first_setup_teardown):
    """
    When kv_transfer_params: null is the FIRST key in the JSON,
    proxy should still strip it and return response directly.
    """
    context = null_kv_first_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-null-first-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"
    assert "usage" in resp_json, f"Expected 'usage' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_null_first_field PASSED")


# =============================================================================
# Tests: kv_transfer_params: null in middle position
# =============================================================================

def test_kv_transfer_params_null_middle_field(null_kv_middle_setup_teardown):
    """
    When kv_transfer_params: null is in the MIDDLE of the JSON,
    proxy should still strip it and return response directly,
    preserving all other fields intact.
    """
    context = null_kv_middle_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-null-middle-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"
    assert "usage" in resp_json, f"Expected 'usage' in response, got: {resp_json}"
    assert "id" in resp_json, f"Expected 'id' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_null_middle_field PASSED")


# =============================================================================
# Tests: kv_transfer_params absent (field not present at all)
# =============================================================================

def test_kv_transfer_params_absent_no_decode_forward(absent_kv_setup_teardown):
    """
    When prefill response does NOT contain kv_transfer_params at all,
    proxy should return prefill response directly without forwarding to decode.
    """
    context = absent_kv_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-absent-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should not appear in response, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_absent_no_decode_forward PASSED")


# =============================================================================
# Mock Handlers for missing required fields scenarios
# =============================================================================

class PrefillMissingBlockIdsHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params WITHOUT remote_block_ids.
    Has all other fields including routed_experts_* fields.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with missing remote_block_ids"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "remote_cluster_id": [2830905970, 546633382273941520],
                "remote_host_ip": "tcp://7.150.8.42:5568",
                "spec_token_ids": [],
                "remote_dp_rank": 0,
                "remote_request_id": "cmpl-12345-0-8d04166d",
                "routed_experts_shape": [9, 46, 8],
                "routed_experts_dtype": "uint8",
                "routed_experts_str_len": 20,
                "routed_experts_str": "ABCDEFGHIJKLMNOPQRST",
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class PrefillMissingHostIpHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params WITHOUT remote_host_ip.
    Has all other fields including routed_experts_* fields.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with missing remote_host_ip"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "remote_cluster_id": [2830905970, 546633382273941520],
                "remote_block_ids": [[1], [2], [3], [4], [5], [6], [7], [8]],
                "spec_token_ids": [],
                "remote_dp_rank": 0,
                "remote_request_id": "cmpl-12345-0-8d04166d",
                "routed_experts_shape": [9, 46, 8],
                "routed_experts_dtype": "uint8",
                "routed_experts_str_len": 20,
                "routed_experts_str": "ABCDEFGHIJKLMNOPQRST",
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class PrefillMissingClusterIdHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params WITHOUT remote_cluster_id.
    Has all other fields including routed_experts_* fields.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with missing remote_cluster_id"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "remote_host_ip": "tcp://7.150.8.42:5568",
                "remote_block_ids": [[1], [2], [3], [4], [5], [6], [7], [8]],
                "spec_token_ids": [],
                "remote_dp_rank": 0,
                "remote_request_id": "cmpl-12345-0-8d04166d",
                "routed_experts_shape": [9, 46, 8],
                "routed_experts_dtype": "uint8",
                "routed_experts_str_len": 20,
                "routed_experts_str": "ABCDEFGHIJKLMNOPQRST",
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class PrefillOnlyNonRequiredFieldsHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params with ONLY non-required fields.
    Missing all 3 required fields: remote_cluster_id, remote_host_ip, remote_block_ids.
    Has routed_experts_* fields.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with only non-required fields"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "spec_token_ids": [],
                "remote_dp_rank": 0,
                "remote_request_id": "cmpl-12345-0-8d04166d",
                "routed_experts_shape": [9, 46, 8],
                "routed_experts_dtype": "uint8",
                "routed_experts_str_len": 20,
                "routed_experts_str": "ABCDEFGHIJKLMNOPQRST",
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class PrefillEmptyKVParamsHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params as empty object {}.
    No required fields, no optional fields.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with empty kv_transfer_params"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {}
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class PrefillFieldValueNullHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params where remote_cluster_id is null.
    But the field exists (just value is null). This should be treated as PRESENT.
    Has all other valid fields.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with remote_cluster_id=null but field exists"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "remote_cluster_id": None,
                "remote_host_ip": "tcp://7.150.8.42:5568",
                "remote_block_ids": [[1], [2], [3], [4], [5], [6], [7], [8]],
                "spec_token_ids": [],
                "remote_dp_rank": 0,
                "remote_request_id": "cmpl-12345-0-8d04166d",
                "routed_experts_shape": [9, 46, 8],
                "routed_experts_dtype": "uint8",
                "routed_experts_str_len": 20,
                "routed_experts_str": "ABCDEFGHIJKLMNOPQRST",
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


# =============================================================================
# Fixtures for missing required fields scenarios
# =============================================================================

missing_block_ids_setup_teardown = make_proxy_fixture(PrefillMissingBlockIdsHandler, 51200)
missing_host_ip_setup_teardown = make_proxy_fixture(PrefillMissingHostIpHandler, 51500)
missing_cluster_id_setup_teardown = make_proxy_fixture(PrefillMissingClusterIdHandler, 51800)
only_non_required_setup_teardown = make_proxy_fixture(PrefillOnlyNonRequiredFieldsHandler, 52100)
empty_kv_params_setup_teardown = make_proxy_fixture(PrefillEmptyKVParamsHandler, 52400)
field_value_null_setup_teardown = make_proxy_fixture(PrefillFieldValueNullHandler, 52700)


# =============================================================================
# Tests: missing remote_block_ids
# =============================================================================

def test_kv_transfer_params_missing_block_ids(missing_block_ids_setup_teardown):
    """
    When kv_transfer_params exists but remote_block_ids is missing,
    proxy should strip kv_transfer_params and return prefill response directly.
    """
    context = missing_block_ids_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-missing-block-ids-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_missing_block_ids PASSED")


# =============================================================================
# Tests: missing remote_host_ip
# =============================================================================

def test_kv_transfer_params_missing_host_ip(missing_host_ip_setup_teardown):
    """
    When kv_transfer_params exists but remote_host_ip is missing,
    proxy should strip kv_transfer_params and return prefill response directly.
    """
    context = missing_host_ip_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-missing-host-ip-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_missing_host_ip PASSED")


# =============================================================================
# Tests: missing remote_cluster_id
# =============================================================================

def test_kv_transfer_params_missing_cluster_id(missing_cluster_id_setup_teardown):
    """
    When kv_transfer_params exists but remote_cluster_id is missing,
    proxy should strip kv_transfer_params and return prefill response directly.
    """
    context = missing_cluster_id_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-missing-cluster-id-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_missing_cluster_id PASSED")


# =============================================================================
# Tests: only non-required fields (missing all 3 required fields)
# =============================================================================

def test_kv_transfer_params_only_non_required_fields(only_non_required_setup_teardown):
    """
    When kv_transfer_params exists but only has non-required fields
    (missing all 3 required: remote_cluster_id, remote_host_ip, remote_block_ids),
    proxy should strip kv_transfer_params and return prefill response directly.
    """
    context = only_non_required_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-only-non-required-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_only_non_required_fields PASSED")


# =============================================================================
# Tests: empty kv_transfer_params object
# =============================================================================

def test_kv_transfer_params_empty_object(empty_kv_params_setup_teardown):
    """
    When kv_transfer_params is an empty object {},
    proxy should strip kv_transfer_params and return prefill response directly.
    """
    context = empty_kv_params_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-empty-object-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped, got: {resp_json}"
    assert "choices" in resp_json, f"Expected 'choices' in response, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    assert decode_count == 0, f"Decode should NOT be called, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_empty_object PASSED")


# =============================================================================
# Tests: field exists but value is null (should be treated as PRESENT)
# =============================================================================

def test_kv_transfer_params_field_value_null(field_value_null_setup_teardown):
    """
    When kv_transfer_params has a required field but value is null (e.g. remote_cluster_id: null),
    the field still EXISTS, so this should be treated as KV_TRANSFER_PRESENT.
    Proxy should forward to decode (field exists despite null value).
    """
    context = field_value_null_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-field-null-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    # Note: kv_transfer_params is only in prefill response, not decode response.
    # So we cannot check for it in resp_json (which is decode response).
    # Instead, we verify that decode WAS called - this proves the proxy correctly
    # detected that all required fields exist (even with null values).

    decode_count = decode_server.decode_request_count
    # Decode SHOULD be called because required fields exist (even with null value)
    assert decode_count > 0, f"Decode SHOULD be called when fields exist, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_field_value_null PASSED")


# =============================================================================
# Additional tests: scenarios without routed_experts_* fields (complex nested data)
# =============================================================================

class PrefillOnlyRequiredFieldsHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params with ONLY the 3 required fields.
    No routed_experts_* fields - tests basic case with minimal fields.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with only required fields"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "remote_cluster_id": [1234567890, 9876543210],
                "remote_host_ip": "tcp://192.168.1.100:8080",
                "remote_block_ids": [[1, 2], [3, 4]]
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class PrefillRequiredPlusOptionalHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params with required fields
    plus some optional fields (spec_token_ids, remote_dp_rank, remote_request_id).
    No routed_experts_* - tests that optional fields don't affect validation.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with required plus optional fields"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "remote_cluster_id": [2830905970, 546633382273941520],
                "remote_host_ip": "tcp://7.150.8.42:5568",
                "remote_block_ids": [[1], [2], [3]],
                "spec_token_ids": [100, 200, 300],
                "remote_dp_rank": 2,
                "remote_request_id": "req-12345"
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


class PrefillEmptyArrayVsMissingHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns kv_transfer_params where remote_block_ids is EMPTY ARRAY [].
    Empty array [] is DIFFERENT from missing - field EXISTS with value [].
    Should be treated as PRESENT (field exists).
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with empty array remote_block_ids"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "remote_cluster_id": [12345],
                "remote_host_ip": "tcp://10.0.0.1:9090",
                "remote_block_ids": []
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


# =============================================================================
# Fixtures for tests without routed_experts_* fields
# =============================================================================

only_required_setup_teardown = make_proxy_fixture(PrefillOnlyRequiredFieldsHandler, 53000)
required_plus_optional_setup_teardown = make_proxy_fixture(PrefillRequiredPlusOptionalHandler, 53300)
empty_array_vs_missing_setup_teardown = make_proxy_fixture(PrefillEmptyArrayVsMissingHandler, 53600)


# =============================================================================
# Tests: kv_transfer_params with only required fields
# =============================================================================

def test_kv_transfer_params_only_required_fields(only_required_setup_teardown):
    """
    When kv_transfer_params has ONLY the 3 required fields (no optional, no routed_experts_*),
    proxy should forward to decode - all required fields exist.
    """
    context = only_required_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-required-only-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    decode_count = decode_server.decode_request_count
    # Decode SHOULD be called because all 3 required fields exist
    assert decode_count > 0, f"Decode SHOULD be called when required fields exist, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_only_required_fields PASSED")


# =============================================================================
# Tests: kv_transfer_params with required fields plus optional fields
# =============================================================================

def test_kv_transfer_params_required_plus_optional(required_plus_optional_setup_teardown):
    """
    When kv_transfer_params has required fields PLUS optional fields (no routed_experts_*),
    proxy should forward to decode - all required fields exist.
    """
    context = required_plus_optional_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-required-optional-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    decode_count = decode_server.decode_request_count
    # Decode SHOULD be called because all 3 required fields exist
    assert decode_count > 0, f"Decode SHOULD be called when required fields exist, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_required_plus_optional PASSED")


# =============================================================================
# Tests: kv_transfer_params with empty array vs missing field
# =============================================================================

def test_kv_transfer_params_empty_array_vs_missing(empty_array_vs_missing_setup_teardown):
    """
    When kv_transfer_params has remote_block_ids: [] (empty array),
    the field EXISTS with value [], which is DIFFERENT from being missing.
    Should be treated as KV_TRANSFER_PRESENT and forward to decode.
    """
    context = empty_array_vs_missing_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-empty-array-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    decode_count = decode_server.decode_request_count
    # Decode SHOULD be called because remote_block_ids field EXISTS (even if empty array)
    assert decode_count > 0, f"Decode SHOULD be called when field exists (empty array), but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_empty_array_vs_missing PASSED")


# =============================================================================
# Tests: remote_host_ip exists at top level but missing in kv_transfer_params
# =============================================================================

class PrefillHostIpAtTopLevelHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns remote_host_ip at top level (same level as kv_transfer_params),
    but NOT inside kv_transfer_params.
    The field inside kv_transfer_params is missing.
    Should be treated as KV_TRANSFER_MISSING_FIELDS - not forwarded to decode.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with remote_host_ip at top level"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "remote_host_ip": "tcp://10.0.0.1:9090",
            "kv_transfer_params": {
                "remote_cluster_id": [12345],
                "remote_block_ids": [[1], [2]],
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


host_ip_toplevel_setup_teardown = make_proxy_fixture(PrefillHostIpAtTopLevelHandler, 53900)


def test_kv_transfer_params_host_ip_at_toplevel(host_ip_toplevel_setup_teardown):
    """
    When remote_host_ip exists at top level but is MISSING inside kv_transfer_params,
    the required field inside kv_transfer_params is still considered missing.
    Should NOT forward to decode - remove kv_transfer_params and return prefill response.
    """
    context = host_ip_toplevel_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-host-ip-toplevel-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    # kv_transfer_params should be stripped because required field is missing
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped when required field missing inside it, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    # Decode should NOT be called because required field inside kv_transfer_params is missing
    assert decode_count == 0, f"Decode should NOT be called when required field missing, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_host_ip_at_toplevel PASSED")


# =============================================================================
# Tests: remote_host_ip exists after kv_transfer_params at top level
# =============================================================================

class PrefillHostIpAfterKVParamsHandler(BaseHTTPRequestHandler):
    """
    Mock prefill server that returns remote_host_ip AFTER kv_transfer_params at top level.
    The field inside kv_transfer_params is missing.
    Should be treated as KV_TRANSFER_MISSING_FIELDS - not forwarded to decode.
    """
    def do_POST(self):
        self.server.request_count += 1

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        response = {
            "id": f"prefill-{self.server.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-prefill",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Prefill response with remote_host_ip after kv_transfer_params"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "kv_transfer_params": {
                "remote_cluster_id": [12345],
                "remote_block_ids": [[1], [2]],
            },
            "remote_host_ip": "tcp://10.0.0.1:9090"
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        return


host_ip_after_kv_setup_teardown = make_proxy_fixture(PrefillHostIpAfterKVParamsHandler, 54200)


def test_kv_transfer_params_host_ip_after_kv(host_ip_after_kv_setup_teardown):
    """
    When remote_host_ip exists AFTER kv_transfer_params at top level,
    but is MISSING inside kv_transfer_params,
    the required field inside kv_transfer_params is still considered missing.
    Should NOT forward to decode - remove kv_transfer_params and return prefill response.
    """
    context = host_ip_after_kv_setup_teardown
    proxy_port = context["proxy_port"]
    decode_server = context["decode_server"]
    decode_server.decode_request_count = 0

    response = send_non_streaming_request(proxy_port, "test-kv-host-ip-after-001")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    resp_json = response.json()
    # kv_transfer_params should be stripped because required field is missing
    assert "kv_transfer_params" not in resp_json, \
        f"kv_transfer_params should be stripped when required field missing inside it, got: {resp_json}"

    decode_count = decode_server.decode_request_count
    # Decode should NOT be called because required field inside kv_transfer_params is missing
    assert decode_count == 0, f"Decode should NOT be called when required field missing, but got {decode_count} requests"

    print("[TEST] test_kv_transfer_params_host_ip_after_kv PASSED")