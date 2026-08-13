# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import contextvars
import os
import threading
import multiprocessing
from datetime import datetime
import logging
import sys
from pathlib import Path
import socket
import requests

from omni_npu import envs


_trace_mm_hash_to_req_id: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar("_trace_mm_hash_to_req_id", default=None)
)


def trace_set_mm_hash_req_map(mapping: dict[str, str]) -> None:
    """Set mm_hash -> request_id map for the current async/thread context."""
    _trace_mm_hash_to_req_id.set(mapping)


def trace_clear_mm_hash_req_map() -> None:
    _trace_mm_hash_to_req_id.set(None)


def trace_lookup_req_id(mm_hash: str | None) -> str | None:
    """Look up request_id for a single mm_hash."""
    mapping = _trace_mm_hash_to_req_id.get() or {}
    return mapping[mm_hash] if mm_hash in mapping else None


def trace_lookup_req_ids_for_load(mm_hashes: list[str]) -> list[str]:
    """Look up request_ids for a batch of mm_hashes to load, deduplicated."""
    mapping = _trace_mm_hash_to_req_id.get() or {}
    return list({mapping[mm_hash] for mm_hash in mm_hashes if mm_hash in mapping})


def safe_print(directory, message):
    process_id = multiprocessing.current_process().pid
    thread_id = threading.get_ident()

    Path(directory).mkdir(parents=True, exist_ok=True)
    
    filename = f"log_pid_{process_id}_tid_{thread_id}.log"
    filepath = os.path.join(directory, filename)

    logger = logging.getLogger(f"safe_print_{process_id}_{thread_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    
    if not logger.handlers:
        handler = logging.FileHandler(filepath)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.info(message)

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"Error getting local IP: {e}"

ip_str = get_ip()
trace_output_directory = envs.OMNI_TRACE_OUTPUT_DIRECTORY
