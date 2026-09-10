# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""ACL Graph utility control-flow tests for Model Runner V2."""

from importlib import import_module


def _aclgraph_utils():
    return import_module("omni_npu.worker.npu.aclgraph_utils")


def test_capture_window_restores_nested_state():
    cudagraph = _aclgraph_utils()

    assert cudagraph._capturing_default() is False
    with cudagraph.CaptureWindow():
        assert cudagraph._capturing_default() is True
        with cudagraph.CaptureWindow():
            assert cudagraph._capturing_default() is True
        assert cudagraph._capturing_default() is True
    assert cudagraph._capturing_default() is False


def test_capturing_descriptor_reads_thread_local_then_instance():
    """未赋值时回落到线程局部标志，赋过值以实例上的为准。"""
    cudagraph = _aclgraph_utils()
    descriptor = cudagraph._CapturingDescriptor()

    class _Ctx:
        capturing = descriptor

    assert _Ctx.capturing is descriptor  # obj 为 None 时返回描述符自身
    ctx = _Ctx()
    assert ctx.capturing is False
    with cudagraph.CaptureWindow():
        assert ctx.capturing is True
        ctx.capturing = 0
        assert ctx.capturing is False
    assert ctx.capturing is False
