# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def make_recording_event_type(records):
    """Create a torch.npu.Event test double that appends operations to records."""

    class RecordingEvent:
        def record(self):
            records.append("record")

        def wait(self, stream):
            records.append(("wait", stream))

    return RecordingEvent


def make_core_limit_context(records):
    """Create a core-limit context manager that records requested limits."""

    @contextmanager
    def limit_core_num(cube, vector):
        records.append((cube, vector))
        yield

    return limit_core_num


def configure_multistream_npu(monkeypatch, submitted, core_limits, main_stream):
    """Install the common NPU stream and graph-scope doubles."""
    monkeypatch.setattr(torch.npu, "Event", make_recording_event_type(submitted))

    def current_stream():
        return main_stream

    def stream(_stream):
        return nullcontext()

    monkeypatch.setattr(torch.npu, "current_stream", current_stream)
    monkeypatch.setattr(torch.npu, "stream", stream)
    monkeypatch.setattr(
        torch.npu,
        "npugraph_ex",
        SimpleNamespace(
            scope=SimpleNamespace(
                limit_core_num=make_core_limit_context(core_limits)
            )
        ),
    )


def make_marked_constant(recorder, name, result):
    """Create a test callback that records its name and returns a fixed result."""

    def callback(*_args, **_kwargs):
        return recorder(name, result)

    return callback


def make_marked_input(recorder, name):
    """Create a test callback that records its name and returns its first input."""

    def callback(value, *_args, **_kwargs):
        return recorder(name, value)

    return callback


def run_mome_out_partition_case(
    attention_cls,
    mla_mod,
    monkeypatch,
    requires_partition,
    get_mome_args,
):
    attention = attention_cls.__new__(attention_cls)
    torch.nn.Module.__init__(attention)
    mome_output = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    split_output = (mome_output[:, :2], mome_output[:, 2:])
    split = MagicMock(return_value=split_output)
    monkeypatch.setattr(mla_mod, "split_tensor_along_last_dim", split)
    attention.use_mome = True
    attention.kv_b_proj = SimpleNamespace(tp_size=1)
    attention.o_proj = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        requires_input_partition=MagicMock(return_value=requires_partition),
    )
    attention._apply_mome = MagicMock(return_value=mome_output)

    output = attention._maybe_mome_out(
        torch.zeros_like(mome_output), get_mome_args
    )

    attention.o_proj.requires_input_partition.assert_called_once_with()
    if requires_partition:
        split.assert_called_once_with(mome_output, num_partitions=2)
        torch.testing.assert_close(output, split_output[1])
        if not output.is_contiguous():
            pytest.fail("partitioned mome output must be contiguous")
    else:
        split.assert_not_called()
        if output is not mome_output:
            pytest.fail("unpartitioned path must return the raw mome tensor")
