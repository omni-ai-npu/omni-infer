# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import subprocess
import time
from pathlib import Path
import re

import pytest

CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../omni_proxy.sh"
error_log = Path(__file__).parent / "nginx_error_buffer_test.log"
access_log = Path(__file__).parent / "nginx_access_buffer_test.log"
nginx_conf = Path(__file__).parent / "nginx_buffer_test.conf"


def truncate_log_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w'):
        pass


def extract_all_values(conf_text: str, directive: str) -> list:
    """Extract all values for a given directive from nginx config text."""
    pattern = rf"{re.escape(directive)}\s+([^;]+);"
    return re.findall(pattern, conf_text)


def test_buffer_size_params_dry_run(tmp_path):
    """Test that buffer size parameters are correctly generated in nginx.conf via dry-run."""
    nginx_conf_file = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file", str(nginx_conf_file),
        "--core-num", "1",
        "--prefill-endpoints", "127.0.0.1:8001,127.0.0.1:8002",
        "--decode-endpoints", "127.0.0.1:9001,127.0.0.1:9002",
        "--client-max-body-size", "100M",
        "--client-body-buffer-size", "50M",
        "--subrequest-output-buffer-size", "20M",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf_file.exists()
    conf_text = nginx_conf_file.read_text()

    # Verify custom values are correctly set
    client_max_body_values = extract_all_values(conf_text, "client_max_body_size")
    assert len(set(client_max_body_values)) == 1, \
        f"client_max_body_size has inconsistent values: {client_max_body_values}"
    assert client_max_body_values[0] == "100M", \
        f"client_max_body_size expected '100M', got '{client_max_body_values[0]}'"

    client_body_buffer_values = extract_all_values(conf_text, "client_body_buffer_size")
    assert len(set(client_body_buffer_values)) == 1, \
        f"client_body_buffer_size has inconsistent values: {client_body_buffer_values}"
    assert client_body_buffer_values[0] == "50M", \
        f"client_body_buffer_size expected '50M', got '{client_body_buffer_values[0]}'"

    subrequest_output_values = extract_all_values(conf_text, "subrequest_output_buffer_size")
    assert len(set(subrequest_output_values)) == 1, \
        f"subrequest_output_buffer_size has inconsistent values: {subrequest_output_values}"
    assert subrequest_output_values[0] == "20M", \
        f"subrequest_output_buffer_size expected '20M', got '{subrequest_output_values[0]}'"


def test_buffer_size_params_default_values_dry_run(tmp_path):
    """Test that default buffer size values are used when parameters are not specified."""
    nginx_conf_file = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file", str(nginx_conf_file),
        "--core-num", "1",
        "--prefill-endpoints", "127.0.0.1:8001,127.0.0.1:8002",
        "--decode-endpoints", "127.0.0.1:9001,127.0.0.1:9002",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf_file.exists()
    conf_text = nginx_conf_file.read_text()

    # Verify default values are correctly set
    client_max_body_values = extract_all_values(conf_text, "client_max_body_size")
    assert len(set(client_max_body_values)) == 1, \
        f"client_max_body_size has inconsistent values: {client_max_body_values}"
    assert client_max_body_values[0] == "10M", \
        f"client_max_body_size expected '10M', got '{client_max_body_values[0]}'"

    client_body_buffer_values = extract_all_values(conf_text, "client_body_buffer_size")
    assert len(set(client_body_buffer_values)) == 1, \
        f"client_body_buffer_size has inconsistent values: {client_body_buffer_values}"
    assert client_body_buffer_values[0] == "1M", \
        f"client_body_buffer_size expected '1M', got '{client_body_buffer_values[0]}'"

    subrequest_output_values = extract_all_values(conf_text, "subrequest_output_buffer_size")
    assert len(set(subrequest_output_values)) == 1, \
        f"subrequest_output_buffer_size has inconsistent values: {subrequest_output_values}"
    assert subrequest_output_values[0] == "1M", \
        f"subrequest_output_buffer_size expected '1M', got '{subrequest_output_values[0]}'"


def test_buffer_size_params_100x_startup():
    """Test that proxy starts successfully with buffer size values set to 100x default."""
    truncate_log_file(error_log)
    truncate_log_file(access_log)

    # 100x default values:
    # client_max_body_size: 10M * 100 = 1000M
    # client_body_buffer_size: 1M * 100 = 100M
    # subrequest_output_buffer_size: 1M * 100 = 100M
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file", str(nginx_conf),
        "--core-num", "1",
        "--listen-port", "7151",
        "--prefill-endpoints", "127.0.0.1:8001,127.0.0.1:8002",
        "--decode-endpoints", "127.0.0.1:9001,127.0.0.1:9002",
        "--log-file", str(error_log),
        "--log-level", "info",
        "--access-log-file", str(access_log),
        "--client-max-body-size", "1000M",
        "--client-body-buffer-size", "100M",
        "--subrequest-output-buffer-size", "100M",
        "--keepalive-nginx",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check that the script started without errors
    assert result.returncode == 0, f"Proxy start failed: {result.stderr}"

    # Wait for nginx to start
    time.sleep(2)

    # Check if the specific nginx process is running
    check_proc = subprocess.run(["pgrep", "-f", "nginx.*nginx_buffer_test.conf"],
                                capture_output=True, text=True)
    assert check_proc.stdout.strip() != "", \
        "nginx process not found after startup"

    # Stop only this specific nginx process
    stop_cmd = ["nginx", "-c", str(nginx_conf), "-s", "stop"]
    subprocess.run(stop_cmd, capture_output=True, text=True)
    time.sleep(1)

    # Verify the generated config has consistent values
    assert nginx_conf.exists()
    conf_text = nginx_conf.read_text()

    client_max_body_values = extract_all_values(conf_text, "client_max_body_size")
    assert len(set(client_max_body_values)) == 1, \
        f"client_max_body_size has inconsistent values: {client_max_body_values}"
    assert client_max_body_values[0] == "1000M", \
        f"client_max_body_size expected '1000M', got '{client_max_body_values[0]}'"

    client_body_buffer_values = extract_all_values(conf_text, "client_body_buffer_size")
    assert len(set(client_body_buffer_values)) == 1, \
        f"client_body_buffer_size has inconsistent values: {client_body_buffer_values}"
    assert client_body_buffer_values[0] == "100M", \
        f"client_body_buffer_size expected '100M', got '{client_body_buffer_values[0]}'"

    subrequest_output_values = extract_all_values(conf_text, "subrequest_output_buffer_size")
    assert len(set(subrequest_output_values)) == 1, \
        f"subrequest_output_buffer_size has inconsistent values: {subrequest_output_values}"
    assert subrequest_output_values[0] == "100M", \
        f"subrequest_output_buffer_size expected '100M', got '{subrequest_output_values[0]}'"


def test_buffer_size_params_with_encode_dry_run(tmp_path):
    """Test that buffer size parameters work correctly with encode endpoints."""
    nginx_conf_file = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file", str(nginx_conf_file),
        "--core-num", "1",
        "--encode-endpoints", "127.0.0.1:9201,127.0.0.1:9202",
        "--prefill-endpoints", "127.0.0.1:8001,127.0.0.1:8002",
        "--decode-endpoints", "127.0.0.1:9001,127.0.0.1:9002",
        "--client-max-body-size", "200M",
        "--client-body-buffer-size", "80M",
        "--subrequest-output-buffer-size", "40M",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf_file.exists()
    conf_text = nginx_conf_file.read_text()

    # Verify encode_sub location exists
    assert "location ~ ^/encode_sub(?<orig>/.*)$" in conf_text, \
        "encode_sub location not found in config"

    # Verify custom values are correctly set
    client_max_body_values = extract_all_values(conf_text, "client_max_body_size")
    assert len(set(client_max_body_values)) == 1, \
        f"client_max_body_size has inconsistent values: {client_max_body_values}"
    assert client_max_body_values[0] == "200M", \
        f"client_max_body_size expected '200M', got '{client_max_body_values[0]}'"

    client_body_buffer_values = extract_all_values(conf_text, "client_body_buffer_size")
    assert len(set(client_body_buffer_values)) == 1, \
        f"client_body_buffer_size has inconsistent values: {client_body_buffer_values}"
    assert client_body_buffer_values[0] == "80M", \
        f"client_body_buffer_size expected '80M', got '{client_body_buffer_values[0]}'"

    subrequest_output_values = extract_all_values(conf_text, "subrequest_output_buffer_size")
    assert len(set(subrequest_output_values)) == 1, \
        f"subrequest_output_buffer_size has inconsistent values: {subrequest_output_values}"
    assert subrequest_output_values[0] == "40M", \
        f"subrequest_output_buffer_size expected '40M', got '{subrequest_output_values[0]}'"


def test_buffer_size_params_encode_with_default_values_dry_run(tmp_path):
    """Test that default buffer size values are used when encode endpoints are specified."""
    nginx_conf_file = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file", str(nginx_conf_file),
        "--core-num", "1",
        "--encode-endpoints", "127.0.0.1:9201,127.0.0.1:9202",
        "--prefill-endpoints", "127.0.0.1:8001,127.0.0.1:8002",
        "--decode-endpoints", "127.0.0.1:9001,127.0.0.1:9002",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf_file.exists()
    conf_text = nginx_conf_file.read_text()

    # Verify encode_sub location exists
    assert "location ~ ^/encode_sub(?<orig>/.*)$" in conf_text, \
        "encode_sub location not found in config"

    # Verify default values are correctly set for all buffer size params
    client_max_body_values = extract_all_values(conf_text, "client_max_body_size")
    assert len(set(client_max_body_values)) == 1, \
        f"client_max_body_size has inconsistent values: {client_max_body_values}"
    assert client_max_body_values[0] == "10M", \
        f"client_max_body_size expected '10M', got '{client_max_body_values[0]}'"

    client_body_buffer_values = extract_all_values(conf_text, "client_body_buffer_size")
    assert len(set(client_body_buffer_values)) == 1, \
        f"client_body_buffer_size has inconsistent values: {client_body_buffer_values}"
    assert client_body_buffer_values[0] == "1M", \
        f"client_body_buffer_size expected '1M', got '{client_body_buffer_values[0]}'"

    subrequest_output_values = extract_all_values(conf_text, "subrequest_output_buffer_size")
    assert len(set(subrequest_output_values)) == 1, \
        f"subrequest_output_buffer_size has inconsistent values: {subrequest_output_values}"
    assert subrequest_output_values[0] == "1M", \
        f"subrequest_output_buffer_size expected '1M', got '{subrequest_output_values[0]}'"