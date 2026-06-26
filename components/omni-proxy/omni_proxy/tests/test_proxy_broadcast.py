# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import subprocess
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request

import pytest
import requests

import port_manager
import utils
from run_proxy import generate_proxy_endpoints
from proxy_mock_broadcast import start_mock_server, stop_mock_server

CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../omni_proxy.sh"
CONF_PATH = CUR_DIR / "nginx_broadcast.conf"
PREFILL_NUM = 4
DECODE_NUM = 4
used = set()

def setup_proxy_broadcast(proxy_port=7000, prefill_port_list = None, decode_port_list = None):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '123'
    
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CONF_PATH}",
            "--core-num", "4",
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_port_list,
            "--decode-endpoints", decode_port_list,
            "--log-file", f"{CUR_DIR}/nginx_error.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access.log",
            "--stream-ops", "add",
            "--no-reuseport",
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

@pytest.fixture(scope="module")
def broadcast_enc():
    global used
    proxy_port = port_manager.find_free_port()
    used.add(proxy_port)
    decode_ports = port_manager.find_n_free_ports(DECODE_NUM, used)
    used.update(set(decode_ports))
    prefill_ports = port_manager.find_n_free_ports(PREFILL_NUM + 1, used)
    used.update(set(prefill_ports))
    decode_servers = [start_mock_server(d) for d in decode_ports]
    prefill_servers = [start_mock_server(p) for p in prefill_ports[:PREFILL_NUM]]
    decode_ports_list = generate_proxy_endpoints(decode_ports)
    prefill_port_list = generate_proxy_endpoints(prefill_ports)

    ret = setup_proxy_broadcast(proxy_port, prefill_port_list, decode_ports_list) 
    ngx_id = utils.get_ngx_pid()

    if ret != 0:
        pytest.fail("Start proxy fail")
    time.sleep(0.5)

    yield {
        "proxy_port": proxy_port,
        "prefill_servers": prefill_servers,
        "decode_servers": decode_servers,
    }

    utils.teardown_proxy_balance(ngx_id)

    for srv in prefill_servers + decode_servers:
        stop_mock_server(srv)


def test_broadcast_locations_are_generated(broadcast_enc):
    text = CONF_PATH.read_text(encoding="utf-8")

    assert "location = /sleep" in text
    assert "location = /wake_up" in text
    assert text.count("omni_proxy_broadcast on;") >= 2


def test_internal_broadcast_sub_location_is_generated(broadcast_enc):
    text = CONF_PATH.read_text(encoding="utf-8")

    assert "location = /omni_proxy_broadcast_sub" in text
    assert "proxy_pass http://$arg_target;" in text

def test_proxy_start_and_stop_with_broadcast_locations(broadcast_enc):
    proxy_port = broadcast_enc["proxy_port"]
    url = f"http://127.0.0.1:{proxy_port}/sleep"

    for _ in range(20):
        try:
            resp = requests.post(url, json={"level": 1}, timeout=10)
            break
        except Exception:
            time.sleep(0.2)

    assert resp.status_code is not None, "proxy started but /sleep was never reachable"
    assert 100 <= resp.status_code < 600

    # Validate JSON response
    data = resp.json()
    assert data["uri"] == "/sleep"
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

    response_code = None
    for _ in range(20):
        try:
            req = urllib.request.Request(
                url,
                data=b'{"level": 1}',
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                response_code = resp.getcode()
            break
        except urllib.error.HTTPError as e:
            response_code = e.code
            break
        except Exception:
            time.sleep(0.2)
    assert response_code is not None, "proxy started but /sleep was never reachable"
    assert 100 <= response_code < 600

    url = f"http://127.0.0.1:{proxy_port}/wake_up?tags=weights&tags=kv_cache"
    response_code = None
    for _ in range(20):
        try:
            req = urllib.request.Request(
                url,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                response_code = resp.getcode()
            break
        except urllib.error.HTTPError as e:
            response_code = e.code
            break
        except Exception:
            time.sleep(0.2)
    assert response_code is not None, "proxy started but /sleep was never reachable"
    assert 100 <= response_code < 600