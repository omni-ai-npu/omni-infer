# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import functools
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
import torch_npu

_current_stream = None

_OMNI_STREAM = {}
_OMNI_EVENT = {}

@functools.lru_cache(maxsize=1)
def _npu_device_name() -> str:
    return torch_npu.npu.get_device_name(0)

def _is_ascend(prefix: str) -> bool:
    return _npu_device_name().startswith(prefix)

def on_ascend910b() -> bool:
    """Check if the device is an Ascend910b (A2) device."""
    return _is_ascend("Ascend910B")

def on_ascend910() -> bool:
    """Check if the device is an Ascend910 (A3) device."""
    return _is_ascend("Ascend910")

def on_ascend950() -> bool:
    """Check if the device is an Ascend950 (A5) device."""
    return _is_ascend("Ascend950")

def get_nth_last_sep_pos(s: str, sep: str = '.', n: int = 2) -> int:
    if n < 1 or not sep:
        return -1
    
    current_pos = len(s)
    for _ in range(n):
        current_pos = s.rfind(sep, 0, current_pos)
        if current_pos == -1:
            return -1
    return current_pos


def get_last_two_parts(s: str, sep: str = '.') -> str:
    second_last_sep_pos = get_nth_last_sep_pos(s, sep=sep, n=2)

    if second_last_sep_pos == -1:
        return s
    
    return s[second_last_sep_pos + 1:]


def current_stream() -> torch.npu.Stream:
    """
    replace `torch.npu.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.npu.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.npu.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.npu.current_stream()`.

    """
    global _current_stream
    if _current_stream is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _current_stream = torch.npu.current_stream()
    return _current_stream


def get_stream(name: str) -> torch.npu.Stream:
    global _OMNI_STREAM
    if _OMNI_STREAM.get(name) is None:
        _OMNI_STREAM[name] = torch.npu.Stream()
    return _OMNI_STREAM[name]


def get_event(name: str) -> torch.npu.Event:
    global _OMNI_EVENT
    if _OMNI_EVENT.get(name) is None:
        _OMNI_EVENT[name] = torch.npu.Event()
    return _OMNI_EVENT[name]


def switch_npu_stream(enable_switch: bool, stream_name: str):
    if enable_switch:
        stream = get_stream(stream_name)
        return torch.npu.stream(stream)
    else:
        return nullcontext()


@contextmanager
def limit_core_num(enable_limit, target_stream, cube_num, vector_num):
    stream = None
    if enable_limit:
        if isinstance(target_stream, str):
            stream = get_stream(target_stream)
        elif isinstance(target_stream, torch.npu.Stream):
            stream = target_stream
        if stream is not None:
            torch.npu.set_stream_limit(stream, cube_num, vector_num)
    yield
    if enable_limit and stream is not None:
        torch.npu.reset_stream_limit(stream)


def record_stream(
    enable_record: bool,
    input: torch.Tensor,
    target_stream: Optional[str | torch.npu.Stream],
):
    if enable_record:
        if isinstance(target_stream, str):
            stream = get_stream(target_stream)
        elif isinstance(target_stream, torch.npu.Stream):
            stream = target_stream
        else:
            return
        input.record_stream(stream)


def wait_event(
    enable_wait: bool,
    target_stream: Optional[str | torch.npu.Stream],
    event_name: str,
):
    if enable_wait:
        if isinstance(target_stream, str):
            stream = get_stream(target_stream)
        elif isinstance(target_stream, torch.npu.Stream):
            stream = target_stream
        else:
            return
        stream.wait_event(get_event(event_name))


def record_event(
    enable_record,
    target_stream: Optional[str | torch.npu.Stream],
    event_name: str,
):
    if enable_record:
        if isinstance(target_stream, str):
            stream = get_stream(target_stream)
        elif isinstance(target_stream, torch.npu.Stream):
            stream = target_stream
        else:
            return
        stream.record_event(get_event(event_name))


def wait_stream(
    enable_wait: bool,
    target_stream_1: Optional[str | torch.npu.Stream],
    target_stream_2: Optional[str | torch.npu.Stream],
):
    if enable_wait:
        if isinstance(target_stream_1, str):
            stream_1 = get_stream(target_stream_1)
        elif isinstance(target_stream_1, torch.npu.Stream):
            stream_1 = target_stream_1
        else:
            return
        if isinstance(target_stream_2, str):
            stream_2 = get_stream(target_stream_2)
        elif isinstance(target_stream_2, torch.npu.Stream):
            stream_2 = target_stream_2
        stream_1.wait_stream(stream_2)
