# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


@contextmanager
def mock_torch_npu_stream():
    mock_npu = MagicMock()
    mock_npu.current_stream.return_value = MagicMock()
    mock_npu.Stream.return_value = MagicMock()

    def _stream_ctx(_stream):
        return nullcontext()

    mock_npu.stream.side_effect = _stream_ctx
    with (
        patch("torch.npu", mock_npu, create=True),
        patch.object(torch.Tensor, "record_stream", return_value=None),
    ):
        yield


def run_maybe_mome_out_partition_case(
    attention_cls,
    split_module,
    monkeypatch,
    requires_partition,
    get_mome_args=None,
):
    attention = attention_cls.__new__(attention_cls)
    torch.nn.Module.__init__(attention)
    mome_output = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    split_output = (mome_output[:, :2], mome_output[:, 2:])
    split = MagicMock(return_value=split_output)
    monkeypatch.setattr(split_module, "split_tensor_along_last_dim", split)
    attention.use_mome = True
    attention.kv_b_proj = SimpleNamespace(tp_size=1)
    attention.o_proj = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        requires_input_partition=MagicMock(return_value=requires_partition),
    )
    attention._apply_mome = MagicMock(return_value=mome_output)
    if get_mome_args is None:
        get_mome_args = MagicMock()
    output = attention._maybe_mome_out(
        torch.zeros_like(mome_output), get_mome_args
    )
    attention.o_proj.requires_input_partition.assert_called_once_with()
    if requires_partition:
        split.assert_called_once_with(mome_output, num_partitions=2)
        torch.testing.assert_close(output, split_output[1])
        assert output.is_contiguous()
    else:
        split.assert_not_called()
        assert output is mome_output
