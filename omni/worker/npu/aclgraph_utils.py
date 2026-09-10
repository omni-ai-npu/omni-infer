# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACL Graph capture support for the NPU V2 model runner.

Scoped to Pangu's default use_aicpu_fa_tiling=true: attention reads its
tiling from device tensors at execution time, so nothing registers an ACL
graph task and a replay needs no out-of-graph refresh. The refresh and the
attention-metadata handoff it consumes therefore live on
feature/mrv2_graph_full, not here; assert_no_acl_tasks_registered() fails
loudly if that assumption ever stops holding.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

from vllm.config import CUDAGraphMode


_CAPTURING = threading.local()
_CAPTURING_ATTR = "_omni_npu_capturing"


def _capturing_default() -> bool:
    return getattr(_CAPTURING, "value", False)


class _CapturingDescriptor:
    """ForwardContext.capturing with a thread-local V2 capture fallback."""

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj.__dict__.get(_CAPTURING_ATTR, _capturing_default())

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[_CAPTURING_ATTR] = bool(value)


class CaptureWindow:
    """Raise the thread-local capturing flag for one recorded pass."""

    def __init__(self) -> None:
        # Placeholder; __enter__ records the value actually restored on exit.
        self._previous = False

    def __enter__(self) -> "CaptureWindow":
        self._previous = _capturing_default()
        _CAPTURING.value = True
        return self

    def __exit__(self, *exc: Any) -> None:
        _CAPTURING.value = self._previous


def _force_full_mode(forward_fn):
    """Run the recorded pass as FULL, with the capturing flag raised.

    Both signals are scoped to this one call rather than to torch.cuda.graph:
    the wrapper is only installed for the recorded pass, so the warmup pass --
    which runs outside the graph and must register nothing -- is untouched,
    and unrelated code entering torch.cuda.graph no longer trips the flag.
    """

    def forward_fn_full(cg_mode):
        with CaptureWindow():
            return forward_fn(
                CUDAGraphMode.FULL if cg_mode is CUDAGraphMode.NONE else cg_mode
            )

    return forward_fn_full


def build_capture_with_full_mode(original_capture):
    """Report FULL to the pass that upstream records with NONE.

    Upstream passes NONE because no inner wrapper takes over once V2 records
    from the outside, but omni reads that field as "am I in a graph" in three
    places. Rewriting the argument rather than the ForwardContext field also
    keeps set_forward_context synthesizing the BatchDescriptor that MoME
    dereferences. The warmup pass keeps NONE: it runs outside the graph.
    """

    @functools.wraps(original_capture)
    def capture(self, create_forward_fn, *args, **kwargs):
        def create_forward_fn_full(desc, warmup):
            forward_fn, attn_state = create_forward_fn(desc, warmup)
            # desc.cg_mode is already a resolved runtime mode: dual configs
            # such as FULL_DECODE_ONLY are split by decode_mode()/mixed_mode()
            # when the capture descs are built, so only PIECEWISE and FULL
            # reach here and a new dual config needs no change.
            if warmup or desc.cg_mode is not CUDAGraphMode.FULL:
                return forward_fn, attn_state
            return _force_full_mode(forward_fn), attn_state

        return original_capture(self, create_forward_fn_full, *args, **kwargs)

    return capture


def ensure_graph_params(compilation_config) -> None:
    """Give capture_graph_task a table to write into, even on this branch.

    Nothing should register while attention takes its tiling from the aicpu,
    but two sites ignore that flag. Reaching one with graph_params unset
    dereferences None inside _get_or_create_workspace -- loud yet
    undiagnosable. Let the registration succeed instead, so
    assert_no_acl_tasks_registered() can say what happened and where to
    look.
    """
    from omni_npu.compilation.acl_graph import get_graph_params, set_graph_params

    if get_graph_params() is not None:
        return  # set_graph_params raises when called twice
    set_graph_params(set(compilation_config.cudagraph_capture_sizes or []))


def assert_no_acl_tasks_registered() -> None:
    """Fail if capture registered tasks this branch cannot refresh.

    Two capture_graph_task call sites do not check use_aicpu_fa_tiling --
    NPUAttentionBackendImpl.forward's FIA_V2 branch and
    NPUDeepseekMLAAttention._apply_standard_attention -- so a model reaching
    either would need the replay-time refresh that only
    feature/mrv2_graph_full carries. capture_graph_task records event.wait
    into the graph, and nothing here ever records the event, so the first
    replay would hang rather than compute a wrong result. Raise at the end
    of capture instead.
    """
    from omni_npu.compilation.acl_graph import get_graph_params

    graph_params = get_graph_params()
    if graph_params is None:
        return
    registered = {
        size: len(entries)
        for size, entries in graph_params.task_entries.items()
        if entries
    }
    if registered:
        raise RuntimeError(
            "[omni-npu/mrv2] capture registered ACL graph tasks "
            f"({registered}), but this branch omits the replay-time refresh. "
            "Restore feature/mrv2_graph_full before running this model."
        )


def reset_for_testing() -> None:
    # The thread-local is the only process state left: the capture wrapper
    # is applied as a patch and needs no per-test teardown.
    _CAPTURING.value = False
