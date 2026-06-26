# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os

import pytest

import port_manager
from run_vllm_mock import start_vllm_mock, cleanup_subprocess

DEFAULT_PREFILL_MAX = 4
DEFAULT_DECODE_MAX = 4


@pytest.fixture(scope="package", autouse=True)
def vllm_keep_alive():
    if os.getenv("PROXY_VLLM_POOL") != "1":
        yield
        return

    prefill_max = int(os.getenv("VLLM_PREFILL_MAX", DEFAULT_PREFILL_MAX))
    decode_max = int(os.getenv("VLLM_DECODE_MAX", DEFAULT_DECODE_MAX))

    port_manager.ensure_ports_file(prefill_max, decode_max)
    processes = start_vllm_mock(prefill_max, decode_max)
    if not processes:
        pytest.fail("Start vllm fail")

    yield

    cleanup_subprocess(processes)
