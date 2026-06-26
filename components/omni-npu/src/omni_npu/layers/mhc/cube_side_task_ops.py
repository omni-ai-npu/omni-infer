# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.utils.torch_utils import direct_register_custom_op

from omni_npu.layers.utils import (
    CUBE_SIDE_STREAM_NAME,
    CUBE_SIDE_TASKS_KEY,
    CubeSideTask,
    named_stream,
)

# Holds {task_key: [h_res_tensor]} (1-element list) used to ferry the
# post-sinkhorn tensor produced inside the closure back to DecoderLayer.forward.
MHC_HOLDER_KEY = "mhc_h_res_holders"

# Holds {layer_key: done_event} — produced by cube_side_run, consumed by
# cube_side_wait. Lives in forward_context.additional_kwargs so the lifetime
# is per-forward (auto-cleaned). Symmetric with MHC_HOLDER_KEY.
CUBE_SIDE_PENDING_KEY = "cube_side_pending_events"


def mhc_register(
    holder_layer_name: str, task_key: str, h_res: torch.Tensor
) -> torch.Tensor:
    fwctx = get_forward_context()
    layer = fwctx.no_compile_layers[holder_layer_name]
    mhc = getattr(layer, "mhc_module", None)
    if mhc is None:
        return h_res

    h_res_holder = [h_res]

    def _sinkhorn() -> None:
        # Runner invokes us inside torch.npu.stream(side_stream).
        h_res_holder[0].record_stream(torch.npu.current_stream())
        h_res_holder[0] = mhc.mhc_sinkhorn(h_res_holder[0])

    fwctx.additional_kwargs.setdefault(CUBE_SIDE_TASKS_KEY, {})[task_key] = (
        CubeSideTask(fn=_sinkhorn)
    )
    fwctx.additional_kwargs.setdefault(MHC_HOLDER_KEY, {})[task_key] = h_res_holder
    return h_res


def mhc_register_fake(
    holder_layer_name: str, task_key: str, h_res: torch.Tensor
) -> torch.Tensor:
    return h_res


def mhc_fetch(task_key: str, fallback: torch.Tensor) -> torch.Tensor:
    fwctx = get_forward_context()
    holders = fwctx.additional_kwargs.get(MHC_HOLDER_KEY)
    if not holders:
        return fallback
    holder = holders.pop(task_key, None)
    if holder is None:
        return fallback
    result = holder[0]
    # New h_res was allocated on the side stream by mhc_sinkhorn; pin its
    # storage to the current (main) stream so the caching allocator doesn't
    # free it while the downstream mhc_post is still consuming. Done here —
    # not in DecoderLayer.forward — because torch.compile does not support
    # tensor.record_stream in its traced region.
    result.record_stream(torch.npu.current_stream())
    return result


def mhc_fetch_fake(task_key: str, fallback: torch.Tensor) -> torch.Tensor:
    return fallback


direct_register_custom_op(
    op_name="mhc_register",
    op_func=mhc_register,
    fake_impl=mhc_register_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="mhc_fetch",
    op_func=mhc_fetch,
    fake_impl=mhc_fetch_fake,
    dispatch_key="PrivateUse1",
)


def maybe_register_mhc_task(
    holder_layer_name: str, task_key: Optional[str], h_res: torch.Tensor
) -> torch.Tensor:
    """Register a side-stream sinkhorn task, or pass h_res through unchanged.

    task_key is None when the multistream optimization is disabled (or not
    applicable for the call site), in which case no task is registered and
    the caller is expected to invoke mhc_sinkhorn synchronously later.
    """
    if task_key is None:
        return h_res
    return torch.ops.vllm.mhc_register(holder_layer_name, task_key, h_res)


def resolve_mhc_h_res(
    mhc_module, task_key: Optional[str], h_res: torch.Tensor
) -> torch.Tensor:
    """Return the post-sinkhorn h_res.

    When task_key is set, fetch the side-stream result registered earlier.
    When task_key is None, run sinkhorn synchronously on the main stream.
    """
    if task_key is None:
        return mhc_module.mhc_sinkhorn(h_res)
    return torch.ops.vllm.mhc_fetch(task_key, h_res)


def _consume_cube_side_task(layer_key: Optional[str]) -> Optional[CubeSideTask]:
    if not layer_key or not is_forward_context_available():
        return None
    tasks = get_forward_context().additional_kwargs.get(CUBE_SIDE_TASKS_KEY)
    if not tasks:
        return None
    return tasks.pop(layer_key, None)


def cube_side_run(layer_key: str, x: torch.Tensor) -> torch.Tensor:
    """Consume the task registered under layer_key (if any), fire it on the
    cube-side stream, stash the done_event for the matching wait op.

    Returns x unchanged so Dynamo sees a tensor flow through the op (keeps
    it from being optimized away — the real effect is on stream state).
    """
    task = _consume_cube_side_task(layer_key)
    if task is None:
        return x
    side = named_stream(CUBE_SIDE_STREAM_NAME)
    side.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side):
        task.fn()
        task.done_event = torch.npu.Event()
        task.done_event.record()
    fwctx = get_forward_context()
    fwctx.additional_kwargs.setdefault(CUBE_SIDE_PENDING_KEY, {})[layer_key] = (
        task.done_event
    )
    return x


def cube_side_run_fake(layer_key: str, x: torch.Tensor) -> torch.Tensor:
    return x


def cube_side_wait(layer_key: str, y: torch.Tensor) -> torch.Tensor:
    """Wait on the cube-side task fired earlier by cube_side_run (if any)."""
    if not is_forward_context_available():
        return y
    pending = get_forward_context().additional_kwargs.get(CUBE_SIDE_PENDING_KEY)
    if not pending:
        return y
    event = pending.pop(layer_key, None)
    if event is not None:
        torch.npu.current_stream().wait_event(event)
    return y


def cube_side_wait_fake(layer_key: str, y: torch.Tensor) -> torch.Tensor:
    return y


direct_register_custom_op(
    op_name="cube_side_run",
    op_func=cube_side_run,
    fake_impl=cube_side_run_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="cube_side_wait",
    op_func=cube_side_wait,
    fake_impl=cube_side_wait_fake,
    dispatch_key="PrivateUse1",
)
