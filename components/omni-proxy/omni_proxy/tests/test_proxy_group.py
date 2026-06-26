# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import pytest
import os
import subprocess
import time
import re
import requests
import json
from pathlib import Path
from run_proxy import setup_proxy, teardown_proxy, graceful_quit_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess
import port_manager
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PREFILL_NUM = 4
DECODE_NUM = 4
proxy_port = 7000
prefill_port_list = None
decode_port_list = None
prefill_groups = "0:2,1:1,2:1"
decode_groups = "0:1,1:2,2:1"

CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../omni_proxy.sh"
error_log = Path(__file__).parent / "nginx_error.log"
access_log = Path(__file__).parent / "nginx_access.log"


def truncate_log_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w'):
        pass


@pytest.fixture(scope="module")
def setup_teardown():
    global proxy_port
    global prefill_port_list
    global decode_port_list

    if os.getenv("SKIP_FIXTURE") == "1":
        truncate_log_file(access_log)
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"]
        decode_port_list = ports["decode"]
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        return

    truncate_log_file(error_log)
    truncate_log_file(access_log)
    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list,
                      prefill_groups=prefill_groups, decode_groups=decode_groups,
                      stream_ops="off")
    if ret == -1:
        pytest.fail(f"Start proxy fail")

    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start mock server fail")

    yield

    # --- Teardown: Shut down all instances ---
    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)


def test_group_config_dry_run(tmp_path):
    nginx_conf = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file",
        str(nginx_conf),
        "--core-num",
        "1",
        "--prefill-endpoints",
        "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004",
        "--decode-endpoints",
        "127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006",
        "--omni-proxy-prefill-groups",
        "0:2,1:1,2:1",
        "--omni-proxy-decode-groups",
        "0:3,1:1,2:2",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf.exists()
    conf_text = nginx_conf.read_text()
    assert "omni_proxy_prefill_groups 0:2 1:1 2:1;" in conf_text
    assert "omni_proxy_decode_groups 0:3 1:1 2:2;" in conf_text


def test_prefill_group_num_not_match_dry_run(tmp_path):
    nginx_conf = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file",
        str(nginx_conf),
        "--prefill-endpoints",
        "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004",
        "--decode-endpoints",
        "127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006",
        "--omni-proxy-prefill-groups",
        "0:2,1:1",
        "--omni-proxy-decode-groups",
        "0:3,1:1,2:2",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf.exists()
    conf_text = nginx_conf.read_text()
    assert "omni_proxy_prefill_groups 0:2 1:1;" in conf_text
    assert "omni_proxy_decode_groups 0:3 1:1 2:2;" in conf_text
    cmd = ['nginx', '-c', str(nginx_conf), '-t']
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0


def test_decode_group_num_not_match_dry_run(tmp_path):
    nginx_conf = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file",
        str(nginx_conf),
        "--prefill-endpoints",
        "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004",
        "--decode-endpoints",
        "127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006",
        "--omni-proxy-prefill-groups",
        "0:2,1:1,2:1",
        "--omni-proxy-decode-groups",
        "1:1,2:2",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf.exists()
    conf_text = nginx_conf.read_text()
    assert "omni_proxy_prefill_groups 0:2 1:1 2:1;" in conf_text
    assert "omni_proxy_decode_groups 1:1 2:2;" in conf_text
    cmd = ['nginx', '-c', str(nginx_conf), '-t']
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0


def test_prefill_group_not_set_dry_run(tmp_path):
    nginx_conf = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file",
        str(nginx_conf),
        "--core-num",
        "1",
        "--prefill-endpoints",
        "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004",
        "--decode-endpoints",
        "127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006",
        "--omni-proxy-decode-groups",
        "0:3,1:1,2:2",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf.exists()
    conf_text = nginx_conf.read_text()
    assert "omni_proxy_decode_groups 0:3 1:1 2:2;" in conf_text
    cmd = ['nginx', '-c', str(nginx_conf), '-t']
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0


def test_decode_group_not_set_dry_run(tmp_path):
    nginx_conf = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file",
        str(nginx_conf),
        "--core-num",
        "1",
        "--prefill-endpoints",
        "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004",
        "--decode-endpoints",
        "127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006",
        "--omni-proxy-prefill-groups",
        "0:2,1:1,2:1",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf.exists()
    conf_text = nginx_conf.read_text()
    assert "omni_proxy_prefill_groups 0:2 1:1 2:1;" in conf_text
    cmd = ['nginx', '-c', str(nginx_conf), '-t']
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0


def test_group_id_not_match_dry_run(tmp_path):
    nginx_conf = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file",
        str(nginx_conf),
        "--core-num",
        "1",
        "--prefill-endpoints",
        "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004",
        "--decode-endpoints",
        "127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006",
        "--omni-proxy-prefill-groups",
        "0:2,1:1,2:1",
        "--omni-proxy-decode-groups",
        "0:3,1:1,2:1,3:1",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf.exists()
    conf_text = nginx_conf.read_text()
    assert "omni_proxy_prefill_groups 0:2 1:1 2:1;" in conf_text
    assert "omni_proxy_decode_groups 0:3 1:1 2:1 3:1;" in conf_text
    cmd = ['nginx', '-c', str(nginx_conf), '-t']
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0


def test_upstream_group_num_not_match_dry_run(tmp_path):
    nginx_conf = tmp_path / "nginx.conf"
    cmd = [
        "bash",
        proxy_script_path,
        "--nginx-conf-file",
        str(nginx_conf),
        "--core-num",
        "1",
        "--prefill-endpoints",
        "127.0.0.1:8001,127.0.0.1:8002,127.0.0.1:8003,127.0.0.1:8004",
        "--decode-endpoints",
        "127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003,127.0.0.1:9004,127.0.0.1:9005,127.0.0.1:9006",
        "--omni-proxy-prefill-groups",
        "0:2,1:1,2:1",
        "--omni-proxy-decode-groups",
        "0:3,1:1,2:3",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert nginx_conf.exists()
    conf_text = nginx_conf.read_text()
    assert "omni_proxy_prefill_groups 0:2 1:1 2:1;" in conf_text
    assert "omni_proxy_decode_groups 0:3 1:1 2:3;" in conf_text
    cmd = ['nginx', '-c', str(nginx_conf), '-t']
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0


def test_proxy_access_log_contains_request_id(setup_teardown):
    request_id = "req-log-9876"
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": request_id
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Log request id"}],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    assert response.status_code == 200

    access_log = Path(__file__).parent / "nginx_access.log"
    deadline = time.time() + 5
    while time.time() < deadline:
        if access_log.exists():
            log_text = access_log.read_text()
            if request_id in log_text:
                return
        time.sleep(0.2)

    pytest.fail("request_id not found in proxy access log")


def parse_group_mapping(groups_str: str) -> dict[int, int]:
    mapping = {}
    current_idx = 0
    for part in groups_str.split(","):
        if not part.strip():
            continue
        group_id_str, count_str = part.split(":")
        group_id = int(group_id_str)
        count = int(count_str)
        for i in range(count):
            mapping[current_idx] = group_id
            current_idx += 1
    return mapping


def test_group_affinity_logged_indices(setup_teardown):
    request_id = "req-group-affinity-0001"
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": request_id
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Check group affinity"}],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    assert response.status_code == 200

    access_log = Path(__file__).parent / "nginx_access.log"
    line = None
    deadline = time.time() + 5
    while time.time() < deadline:
        if access_log.exists():
            for entry in access_log.read_text().splitlines():
                if request_id in entry:
                    line = entry
                    break
        if line:
            break
        time.sleep(0.2)

    assert line is not None, "request_id not found in proxy access log"

    prefill_match = re.search(r'"prefill_idx":"(?P<prefill>\d+)"', line)
    decode_match = re.search(r'"decode_idx":"(?P<decode>\d+)"', line)
    assert prefill_match is not None, f"prefill_idx missing in log line: {line}"
    assert decode_match is not None, f"decode_idx missing in log line: {line}"

    prefill_idx = int(prefill_match.group("prefill"))
    decode_idx = int(decode_match.group("decode"))

    prefill_to_group = parse_group_mapping(prefill_groups)
    decode_to_group = parse_group_mapping(decode_groups)

    assert prefill_idx in prefill_to_group, f"prefill_idx {prefill_idx} out of range for groups: {prefill_groups}"
    assert decode_idx in decode_to_group, f"decode_idx {decode_idx} out of range for groups: {decode_groups}"

    prefill_group = prefill_to_group[prefill_idx]
    decode_group = decode_to_group[decode_idx]

    assert prefill_group == decode_group, (
        f"Group mismatch! "
        f"prefill_idx={prefill_idx} group {prefill_group}, "
        f"decode_idx={decode_idx} group {decode_group}. "
        f"Log line: {line}"
    )


def test_group_affinity_logged_indices_concurrent_post(setup_teardown):
    num_requests = 10
    request_ids = [f"req-group-affinity-{i:04d}" for i in range(num_requests)]
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers_base = {"Content-Type": "application/json"}
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Check group affinity"}],
        "stream": False
    }

    def send_request(req_id: str):
        headers = {**headers_base, "X-Request-Id": req_id}
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            assert resp.status_code == 200, f"Request {req_id} failed with {resp.status_code}"
            return req_id, True
        except Exception as e:
            return req_id, e

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request, rid) for rid in request_ids]
        for future in as_completed(futures):
            req_id, result = future.result()
            if isinstance(result, Exception):
                raise AssertionError(f"Request {req_id} raised exception: {result}")

    access_log = Path(__file__).parent / "nginx_access.log"
    deadline = time.time() + 20
    found_logs = {}
    while time.time() < deadline:
        if access_log.exists():
            content = access_log.read_text()
            for line in content.splitlines():
                for rid in request_ids:
                    if rid in line and rid not in found_logs:
                        found_logs[rid] = line
        if len(found_logs) == num_requests:
            break
        time.sleep(0.2)

    missing = set(request_ids) - set(found_logs.keys())
    assert not missing, f"Missing log entries for request IDs: {missing}"

    prefill_to_group = parse_group_mapping(prefill_groups)
    decode_to_group = parse_group_mapping(decode_groups)

    for req_id, line in found_logs.items():
        prefill_match = re.search(r'"prefill_idx":"(?P<prefill>\d+)"', line)
        decode_match = re.search(r'"decode_idx":"(?P<decode>\d+)"', line)
        assert prefill_match is not None, f"[{req_id}] prefill_idx missing in log: {line}"
        assert decode_match is not None, f"[{req_id}] decode_idx missing in log: {line}"

        prefill_idx = int(prefill_match.group("prefill"))
        decode_idx = int(decode_match.group("decode"))

        assert prefill_idx in prefill_to_group, (
            f"[{req_id}] prefill_idx {prefill_idx} out of range for prefill_groups: {prefill_groups}"
        )
        assert decode_idx in decode_to_group, (
            f"[{req_id}] decode_idx {decode_idx} out of range for decode_groups: {decode_groups}"
        )

        prefill_group = prefill_to_group[prefill_idx]
        decode_group = decode_to_group[decode_idx]

        assert prefill_group == decode_group, (
            f"[{req_id}] Group mismatch! "
            f"prefill_idx={prefill_idx} group {prefill_group}, "
            f"decode_idx={decode_idx} group {decode_group}. "
            f"Log: {line}"
        )


def parse_group_mappings_from_error_log(error_log_path: Path) -> tuple[dict[int, int], dict[int, int]]:
    prefill_map = {}
    decode_map = {}

    if not error_log_path.exists():
        raise FileNotFoundError(f"Nginx error log not found: {error_log_path}")

    content = error_log_path.read_text()

    prefill_pattern = re.compile(
        r'Add Prefill peer \d+ in endpoint\[(?P<idx>\d+)\] .*? \(group=(?P<group>\d+)\)'
    )
    decode_pattern = re.compile(
        r'Add Decode peer \d+ in endpoint\[(?P<idx>\d+)\] .*? \(group=(?P<group>\d+)\)'
    )

    for line in content.splitlines():
        m = prefill_pattern.search(line)
        if m:
            idx = int(m.group("idx"))
            group = int(m.group("group"))
            prefill_map[idx] = group
            continue

        m = decode_pattern.search(line)
        if m:
            idx = int(m.group("idx"))
            group = int(m.group("group"))
            decode_map[idx] = group

    if not prefill_map:
        raise AssertionError("No 'Add Prefill peer' entries found in error log")
    if not decode_map:
        raise AssertionError("No 'Add Decode peer' entries found in error log")

    return prefill_map, decode_map


def parse_indices(spec, total_length: int) -> list[int]:
    """
    support:
      - "all": [0, 1, ..., total_length-1]
      - single int: 0 -> [0]
      - str: "1-3" -> [1, 2, 3]
      - mixed list: [0, "2-3"] -> [0, 2, 3]
      - empty -> []

    :param spec: int / str / list
    :param total_length:
    :return: list
    """
    if spec == "all":
        return list(range(total_length))

    if isinstance(spec, int):
        if spec < 0 or spec >= total_length:
            raise IndexError(f"Index {spec} out of range for length {total_length}")
        return [spec]

    if isinstance(spec, str):
        if spec == "":
            return []
        if '-' in spec:
            parts = spec.split('-', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid range format: {spec!r}")
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError as e:
                raise ValueError(f"Invalid range format: {spec!r}") from e
            if start > end:
                raise ValueError(f"Range start > end: {spec!r}")
            if start < 0 or end >= total_length:
                raise IndexError(f"Range {spec!r} out of bounds for length {total_length}")
            return list(range(start, end + 1))
        else:
            idx = int(spec)
            if idx < 0 or idx >= total_length:
                raise IndexError(f"Index {idx} out of range for length {total_length}")
            return [idx]

    if isinstance(spec, list):
        result = []
        for item in spec:
            result.extend(parse_indices(item, total_length))
        return result

    raise ValueError(f"Unsupported index spec type: {type(spec).__name__}, value: {spec!r}")


def resolve_ports(spec, port_list: list[int]) -> list[int]:
    indices = parse_indices(spec, total_length=len(port_list))
    return [port_list[i] for i in indices]


def generate_test_params():
    cases = []
    cases.append(("single_group", "0", "1-2", "0:1", "0:2"))
    cases.append(("two_group", "all", "0-2", "0:2,1:2", "0:1,1:2"))
    cases.append(("two_group_continuous", [0, 2, 3], [1, 3], "0:2,1:1", "0:1,1:1"))
    cases.append(("three_group", "1-3", "all", "0:1,1:1,2:1", "0:1,1:1,2:2"))
    return cases


PARAMS = generate_test_params()
TEST_IDS = [case[0] for case in PARAMS]


@pytest.mark.parametrize(
    "test_id, prefill_spec, decode_spec, test_prefill_groups, test_decode_groups",
    PARAMS,
    ids=TEST_IDS
)
def test_group_affinity_logged_indices_with_reload(setup_teardown, test_id,
                                                   prefill_spec, decode_spec, test_prefill_groups, test_decode_groups):
    error_log = Path(__file__).parent / "nginx_error.log"
    access_log = Path(__file__).parent / "nginx_access.log"
    conf_file = Path(__file__).parent / "nginx.conf"

    error_log.write_text("")
    access_log.write_text("")

    # reload only modify port list
    prefill_ports = resolve_ports(prefill_spec, prefill_port_list)
    decode_ports = resolve_ports(decode_spec, decode_port_list)
    ret = setup_proxy(proxy_port, prefill_ports, decode_ports,
                      prefill_groups=test_prefill_groups, decode_groups=test_decode_groups, dry_run=True)
    if ret == -1:
        pytest.fail(f"Start proxy fail")
    try:
        cmd = ["nginx", "-c", conf_file, "-s", "reload"]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Setup script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)
        pytest.fail(f"reload proxy fail")
    time.sleep(1)

    num_requests = 10
    request_ids = [f"req-group-affinity-{i:04d}" for i in range(num_requests)]
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers_base = {"Content-Type": "application/json"}
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Check group affinity"}],
        "stream": False
    }

    def send_request(req_id: str):
        headers = {**headers_base, "X-Request-Id": req_id}
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        assert resp.status_code == 200, f"Request {req_id} failed: {resp.status_code}"
        return req_id

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request, rid) for rid in request_ids]
        for future in as_completed(futures):
            future.result()

    prefill_to_group, decode_to_group = parse_group_mappings_from_error_log(error_log)

    deadline = time.time() + 20
    found_logs = {}
    while time.time() < deadline:
        if access_log.exists():
            for line in access_log.read_text().splitlines():
                for rid in request_ids:
                    if rid in line and rid not in found_logs:
                        found_logs[rid] = line
        if len(found_logs) == num_requests:
            break
        time.sleep(0.2)

    assert len(found_logs) == num_requests, f"Missing logs: expected {num_requests}, got {len(found_logs)}"

    total_matching = sum(
        1 for line in access_log.read_text().splitlines()
        if any(rid in line for rid in request_ids)
    )
    assert total_matching == num_requests, f"Unexpected number of log lines: {total_matching}"

    for req_id, line in found_logs.items():
        prefill_match = re.search(r'"prefill_idx":"(?P<prefill>\d+)"', line)
        decode_match = re.search(r'"decode_idx":"(?P<decode>\d+)"', line)
        assert prefill_match, f"[{req_id}] missing prefill_idx in: {line}"
        assert decode_match, f"[{req_id}] missing decode_idx in: {line}"

        prefill_idx = int(prefill_match.group("prefill"))
        decode_idx = int(decode_match.group("decode"))

        assert prefill_idx in prefill_to_group, (
            f"[{req_id}] prefill_idx {prefill_idx} not found in error.log prefill peers. "
            f"Known prefill indices: {sorted(prefill_to_group.keys())}"
        )
        assert decode_idx in decode_to_group, (
            f"[{req_id}] decode_idx {decode_idx} not found in error.log decode peers. "
            f"Known decode indices: {sorted(decode_to_group.keys())}"
        )

        prefill_group = prefill_to_group[prefill_idx]
        decode_group = decode_to_group[decode_idx]

        assert prefill_group == decode_group, (
            f"[{req_id}] Group mismatch! "
            f"prefill_idx={prefill_idx} (group={prefill_group}) vs "
            f"decode_idx={decode_idx} (group={decode_group}). "
            f"Log: {line}"
        )

    # restore proxy
    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list,
                      prefill_groups=prefill_groups, decode_groups=decode_groups, dry_run=True)
    if ret == -1:
        pytest.fail(f"Start proxy fail")
    cmd = ["nginx", "-c", conf_file, "-s", "reload"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)