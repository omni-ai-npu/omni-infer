# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for omni_npu.compilation.utils module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from omni_npu.compilation.utils import (
    _capture_kwargs,
    _get_or_create_workspace,
    capture_graph_task,
)


class TestCaptureKwargs:
    """Tests for _capture_kwargs."""

    def test_capture_kwargs_only_weak_ref_selected_keys(self):
        op_desc = SimpleNamespace(weak_ref_keys=("query", "value"))
        query = torch.randn(2, 2)
        key = torch.randn(2, 2)
        value = torch.randn(2, 2)
        kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "none_field": None,
        }

        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
            mock_weak_ref.side_effect = lambda x: f"weak_ref_{id(x)}"
            captured = _capture_kwargs(op_desc, kwargs)

        assert captured["query"].startswith("weak_ref_")
        assert captured["value"].startswith("weak_ref_")
        assert captured["key"] is key
        assert captured["none_field"] is None
        assert mock_weak_ref.call_count == 2


class TestGetOrCreateWorkspace:
    """Tests for _get_or_create_workspace."""

    def test_get_or_create_workspace_caches_workspace(self):
        num_tokens = 16
        query = torch.randn(1, 8)
        workspace = torch.randn(4)
        workspace_fn = MagicMock(return_value=workspace)
        op_desc = SimpleNamespace(workspace_fn=workspace_fn)
        graph_params = SimpleNamespace(workspaces={num_tokens: {}})

        with patch("omni_npu.compilation.utils.get_graph_params", return_value=graph_params):
            with patch("omni_npu.compilation.utils.weak_ref_tensors", side_effect=lambda x: x):
                first = _get_or_create_workspace(op_desc, {"query": query}, num_tokens)
                second = _get_or_create_workspace(op_desc, {"query": query}, num_tokens)

        assert first is workspace
        assert second is workspace
        workspace_fn.assert_called_once_with(query=query)
        assert graph_params.workspaces[num_tokens][workspace_fn] is workspace


class TestCaptureGraphTask:
    """Tests for capture_graph_task."""

    def test_capture_graph_task_records_entry(self):
        num_tokens = 32
        layer_name = "layer_0"
        stream = MagicMock()
        event = MagicMock()
        handle = MagicMock()
        workspace = torch.randn(2)

        query = torch.randn(2, 4)
        key = torch.randn(2, 4)
        out_tensors = [torch.randn(2, 4), torch.randn(2, 4)]
        op_kwargs = {"query": query, "key": key}

        op_out_fn = MagicMock()
        op_desc = SimpleNamespace(
            op_out_fn=op_out_fn,
            workspace_fn=MagicMock(return_value=workspace),
            weak_ref_keys=("query",),
        )
        graph_params = SimpleNamespace(
            task_entries={num_tokens: {}},
            workspaces={num_tokens: {}},
        )

        with patch("omni_npu.compilation.utils.get_graph_params", return_value=graph_params):
            with patch("omni_npu.compilation.utils.weak_ref_tensors", side_effect=lambda x: x):
                with patch("torch_npu.npu.current_stream", return_value=stream):
                    with patch("torch.npu.ExternalEvent", return_value=event):
                        with patch("torch.npu.graph_task_group_begin") as mock_begin:
                            with patch("torch.npu.graph_task_group_end", return_value=handle) as mock_end:
                                capture_graph_task(
                                    op_desc,
                                    op_kwargs,
                                    out_tensors,
                                    num_tokens,
                                    layer_name,
                                )

        mock_begin.assert_called_once_with(stream)
        mock_end.assert_called_once_with(stream)
        event.wait.assert_called_once_with(stream)
        event.reset.assert_called_once_with(stream)
        op_out_fn.assert_called_once_with(
            query=query,
            key=key,
            workspace=workspace,
            out=out_tensors,
        )

        assert layer_name in graph_params.task_entries[num_tokens]
        entry = graph_params.task_entries[num_tokens][layer_name]
        assert entry.op_desc is op_desc
        assert entry.captured_kwargs["query"] is query
        assert entry.captured_kwargs["key"] is key
        assert entry.out_tensors == out_tensors
        assert entry.handle is handle
        assert entry.event is event
