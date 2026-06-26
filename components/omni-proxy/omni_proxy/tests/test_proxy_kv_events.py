# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import msgpack
import pytest
import zmq

import port_manager
import utils
from run_proxy import generate_proxy_endpoints, setup_proxy, teardown_proxy


PREFILL_NUM = 3
DECODE_NUM = 3
CUR_DIR = Path(__file__).parent
proxy_script_path = f"{CUR_DIR}/../omni_proxy.sh"


@pytest.fixture(scope="module")
def setup_teardown(vllm_keep_alive):
    global proxy_port
    global prefill_port_list
    global decode_port_list

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "123"

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

    yield

    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")


def kv_proxy_only():
    ports = port_manager.find_n_free_ports(PREFILL_NUM + DECODE_NUM)
    proxy_port_apc = port_manager.find_free_port()
    ret = setup_proxy_apc(proxy_port_apc, ports[:PREFILL_NUM], ports[DECODE_NUM:PREFILL_NUM + DECODE_NUM])

    if ret != 0:
        pytest.fail("Start proxy fail")

    time.sleep(5)
    prefill_port = ports[:PREFILL_NUM]
    decode_port = ports[DECODE_NUM:PREFILL_NUM + DECODE_NUM]
    return prefill_port, decode_port


def setup_proxy_apc(proxy_port=7000, prefill_port_list=None, decode_port_list=None):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "123"

    prefill_list = generate_proxy_endpoints(prefill_port_list)
    decode_list = generate_proxy_endpoints(decode_port_list)

    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CUR_DIR}/nginx_kv.conf",
            "--core-num", "4",
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", f"{CUR_DIR}/nginx_error_kv.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access_kv.log",
            "--stream-ops", "add",
            "--omni-proxy-model-path", f"{CUR_DIR}/mock_model/",
            "--no-reuseport",
            "--keepalive-nginx"
        ]
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
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


def test_kv_block_removed_event(setup_teardown):
    prefill_port, decode_port = kv_proxy_only()

    ngx_pid = utils.get_ngx_pid()

    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)

    try:
        pub.bind(f"tcp://*:{prefill_port[0] + 100}")
        pub.bind(f"tcp://*:{decode_port[0] + 100}")
    except zmq.ZMQError as exc:
        pub.close(0)
        context.term()
        pytest.skip(f"kv events port already in use: {exc}")

    try:
        # Allow time for SUB to connect and subscribe
        time.sleep(0.5)

        block_hash = 123456789
        block_hash_neg = -123456789

        block_stored = [
            "BlockStored",
            [block_hash],
            123,
            [1, 2, 3],
            128,
            1,
            "cpu",
            "lora_name",
            0,
        ]

        block_stored_neg = [
            "BlockStored",
            [block_hash_neg],
            -123,
            [],
            -128,
            -1,
            "cpu",
            None,
            1,
        ]

        block_stored_zero = [
            "BlockStored",
            [],
            0,
            [0],
            None,
            None,
            None,
            None,
            None,
        ]

        block_stored_str_lora = [
            "BlockStored",
            [],
            0,
            [0],
            None,
            "lora_id",
            None,
            "lora_name",
            2,
        ]

        block_removed = [
            "BlockRemoved",
            [block_hash],
            "cpu",
            0,
        ]

        block_removed_err = [
            "BlockRemoved",
            None,
            None,
            None,
        ]

        block_removed_err_size = [
            "BlockRemoved",
            None,
        ]

        block_blkcleard = [
            "AllBlocksCleared",
        ]

        block_err_str = [
            0,
        ]

        block_err_type = "wrong type instead of list"

        block_err_stored_err_size = [
            "BlockStored",
            [block_hash],
        ]

        block_err_event = [
            "Unknown",
        ]

        batches_events = [
            [block_stored],
            [block_removed],
            [block_blkcleard],
            [block_stored, block_removed],
            [block_stored_neg],
            block_err_type,
            [block_err_str],
            [block_stored_zero],
            [block_stored_str_lora],
            [block_removed_err_size],
            [block_err_stored_err_size],
            [block_err_event],
            [block_removed_err],
        ]

        for seq, events in enumerate(batches_events):
            payload = msgpack.packb([time.time(), events], use_bin_type=True)
            pub.send_multipart([b"", str(seq).encode("utf-8"), payload])
            time.sleep(0.2)

        events_num = len(batches_events)

        non_msgpack_payload = b"not-msgpack"
        pub.send_multipart([b"", str(events_num).encode("utf-8"), non_msgpack_payload])

        invalid_payload = msgpack.packb({"bad": 1}, use_bin_type=True)
        pub.send_multipart([b"", str(events_num + 1).encode("utf-8"), invalid_payload])

        timestamp_payload = msgpack.packb([1, block_stored], use_bin_type=True)
        pub.send_multipart([b"", str(events_num + 2).encode("utf-8"), timestamp_payload])

        # topic only
        pub.send_multipart([b""])
        time.sleep(0.1)

        # topic + seq only
        pub.send_multipart([b"", b"16"])
        time.sleep(0.1)

        # topic + seq + payload + extra
        pub.send_multipart([b"", b"17", b"payload", b"extra"])
        time.sleep(0.2)

        log_file = f"{CUR_DIR}/nginx_error_kv.log"
        LOG_LINE_NUM = 73
        targets = count_kv_events(log_file, LOG_LINE_NUM)
        targets_mapping = {
            "Received KV event batch with 1 events": 22,
            "Received KV event batch with 2 events": 2,
            "Received KV event batch with 9 events": 2,
            "Removed block hash 123456789u from radix tree": 4,
            "OBJECT TYPE: 0": 0,
            "Failed to parse KV event batch": 4,
        }
        assert targets == targets_mapping

    finally:
        utils.teardown_proxy_balance(ngx_pid)
        try:
            pub.unbind(f"tcp://*:{prefill_port + 100}")
        except Exception:
            pass
        pub.close(0)
        context.term()


def count_kv_events(log_path, tail_lines=133):
    targets = {
        "Received KV event batch with 1 events": 0,
        "Received KV event batch with 2 events": 0,
        "Received KV event batch with 9 events": 0,
        "Removed block hash 123456789u from radix tree": 0,
        "OBJECT TYPE: 0": 0,
        "Failed to parse KV event batch": 0,
    }

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        last_lines = deque(f, maxlen=tail_lines)

    for line in last_lines:
        for key in targets:
            if key in line:
                targets[key] += 1

    return targets
