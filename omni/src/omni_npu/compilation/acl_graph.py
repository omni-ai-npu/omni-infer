# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import copy
import dataclasses
import gc
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
from unittest.mock import patch

import numpy as np
import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

global_recapture = False


def set_aclgraph_recapture(enable: bool):
    global global_recapture
    if not enable or global_recapture:
        return

    global_recapture = True


def consume_aclgraph_recapture() -> bool:
    global global_recapture
    if not global_recapture:
        return False
    global_recapture = False
    return True


def reset_stale_aclgraph_resources(wrappers=None) -> None:
    graph_params = get_graph_params()
    wrappers = list(wrappers) if wrappers is not None else []
    if graph_params is None and not wrappers:
        return

    free_before, _ = torch.npu.mem_get_info()

    # Drop cached graph-task metadata so recapture rebuilds it.
    if graph_params is not None:
        for task_entries in graph_params.task_entries.values():
            task_entries.clear()
        for workspaces in graph_params.workspaces.values():
            workspaces.clear()

    if wrappers:
        # Release every captured NPUGraph before retiring the shared pool.
        for wrapper in wrappers:
            wrapper.reset_captured_graphs()

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()

    if wrappers:
        type(current_platform)._global_graph_pool = None
        new_pool = current_platform.get_global_graph_pool()
        for wrapper in wrappers:
            wrapper.graph_pool = new_pool
            wrapper.need_static_compile = False

    free_after, _ = torch.npu.mem_get_info()
    logger.info(
        "Stale ACLGraph resource reset: released graphs, cleared graph_params, "
        "destructed the old shared pool and rotated to a fresh global pool, "
        "freed %.2f GiB",
        (free_after - free_before) / (1 << 30),
    )


def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    if hasattr(torch.ops._C_ascend, "weak_ref_tensor") and isinstance(tensor, torch.Tensor):
        return torch.ops._C_ascend.weak_ref_tensor(tensor)
    else:
        return tensor


def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]]
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.

    This function should be used in the following scenario:
    When a tensor is created during graph capture, and it's held by a method
    that's not part of the graph, we don't really need to store it, but we
    **do need** its buffer pointer. If we don't handle this, it cannot
    be garbage collected, leading to a memory leak. To avoid this,
    we should create a weak reference to the tensor.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")


@dataclass
class OpDescriptor:
    """Describes how to capture and update an op in a graph task."""
    op_out_fn: Callable
    workspace_fn: Callable
    weak_ref_keys: tuple[str, ...] = ()
    compute_dynamic_kwargs: Optional[Callable] = None


@dataclass
class GraphTaskEntry:
    """Stored state for a single captured graph task."""
    op_desc: OpDescriptor
    captured_kwargs: dict[str, Any]
    out_tensors: list[Any]
    handle: Any
    event: Any


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Optional[Any] = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 graph_pool: Any = None,
                 cudagraph_options: Optional[CUDAGraphOptions] = None,
                 update_stream: torch.npu.Stream = None,
                 attn_layer_names: list[str] = [],
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.graph_pool = graph_pool
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config
        self.update_stream = update_stream
        self.attn_layer_names = attn_layer_names

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self.npugraph_ex_config = (self.vllm_config.additional_config or {}).get(
            "npugraph_ex_config", {}
        )
        self.need_super_kernel_optimize = False
        self.need_static_compile = False
        if self.npugraph_ex_config.get("enable", False):
            self.need_super_kernel_optimize = \
                self.npugraph_ex_config.get("super_kernel_optimize", False)
            self.need_static_compile = (
                self.need_super_kernel_optimize
                or self.npugraph_ex_config.get("static_kernel_compile", False)
            )

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        if self.graph_pool is None:
            # get_global_graph_pool() must stay ACLGraph-only. If other code
            # keeps references to this shared pool, recapture may not fully
            # release the stale pool when rotating to a fresh one.
            self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"aclgraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def reset_captured_graphs(self) -> None:
        """Eagerly release all captured graphs before a recapture.

        The reset happens against the pool the graphs were captured into,
        which decrements that pool's use_count in the allocator. Rotating
        self.graph_pool to a fresh shared handle is handled by the caller
        after the old pool has been destructed.
        """
        for entry in self.concrete_aclgraph_entries.values():
            aclgraph = entry.aclgraph
            entry.aclgraph = None
            entry.output = None
            if aclgraph is not None:
                aclgraph.reset()

    def __call__(self, *args, **kwargs):
        logger.debug("<<< ACLGraphWrapper is being called.")
        begin = time.time()
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            aclgraph_runtime_mode == CUDAGraphMode.NONE
            or aclgraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = \
                ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)",
                             self.runtime_mode.name, entry.batch_descriptor)
            if self.need_static_compile and entry.aclgraph is None:
                # Compile once before the first capture so static kernels are ready.
                logger.debug("<<< Running first compile for static kernel compilation")
                self.runnable(*args, **kwargs)

            input_addresses = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    input_addresses.append(arg.data_ptr())
            entry.input_addresses = input_addresses
            aclgraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                forward_context.capturing = True
                with torch.npu.graph(aclgraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's aclgraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.aclgraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise aclgraph mode, because
                        # the output of the last graph will not be used by
                        # any other acl graph.
                        output = weak_ref_tensors(output)

            # ensure all tensors in graph_params are weak-refed to avoid memory leak, 
            # since they are not needed after capture, 
            # and we only need the buffer pointer for replay.
            ensure_weak_ref_graph_params()

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)

            entry.aclgraph = aclgraph

            compilation_counter.num_cudagraph_captured += 1
            end = time.time()
            logger.debug(f"<<< Capturing aclgraph time {end - begin =}s")

            if self.need_super_kernel_optimize:
                logger.debug("<<< Running aclgraph super kernel optimize")
                entry.aclgraph.super_kernel_optimize(optimize_options={
                    "dcci_before_kernel_start": [
                        ".*GroupedMatmul.*",
                        ".*AiInfraSparseFlashAttentionPioneer.*",
                        ".*AiInfraKvRmsNormRopeCache.*",
                        ".*AiInfraScatterBlockUpdate.*",
                        ".*AiInfraFusedInferAttentionSink.*"
                    ],
                    "dcci_disable_on_kernel": [".*"]
                })

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    new_input_addresses.append(arg.data_ptr())
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}")

        logger.debug(f"<<< Replaying aclgraph, {batch_descriptor=}")

        entry.aclgraph.replay()

        # Updates after replay to improve performance,
        # temporal ordering ensured via inter-stream events.
        self._update_graph_tasks(
            self.update_stream, forward_context,
        )

        return entry.output

    def _update_graph_tasks(self, update_stream, forward_context):
        """Update captured graph tasks with dynamic kwargs on a side stream.

        After an ACL graph replay, sequence-length metadata may
        have changed.  This method re-issues every captured attention op with
        the updated kwargs so the device-side task graph stays in sync.

        The updates run on ``update_stream`` (not the main compute stream) for
        overlap; temporal ordering with the main stream is ensured by the
        per-task ExternalEvent recorded at the end of each update.
        """
        if forward_context.attn_metadata is None:
            raise RuntimeError("attn_metadata is empty in forward_context," \
                              "cannot update graph tasks for attention layers.")

        graph_params = get_graph_params()
        runtime_shape = forward_context.batch_descriptor.num_tokens
        task_entries = graph_params.task_entries.get(runtime_shape)
        workspaces = graph_params.workspaces.get(runtime_shape)

        with torch.npu.stream(update_stream):
            for layer_name, task_entry in task_entries.items():
                if layer_name not in self.attn_layer_names:
                    continue
                update_fn = task_entry.op_desc.compute_dynamic_kwargs
                if update_fn is None:
                    task_entry.event.record(update_stream)
                    continue
                dynamic_kwargs = update_fn(
                    forward_context, layer_name, self.vllm_config,
                )
                if dynamic_kwargs is None:
                    task_entry.event.record(update_stream)
                    continue

                updated_kwargs = {
                    **task_entry.captured_kwargs,
                    **dynamic_kwargs,
                }
                if (updated_kwargs.get("input_layout", "").startswith("BSND")
                    and "actual_seq_lengths" in updated_kwargs
                ):
                    updated_kwargs["actual_seq_lengths"] = None

                workspace = workspaces[task_entry.op_desc.workspace_fn]
                out_tensors = task_entry.out_tensors

                torch.npu.graph_task_update_begin(update_stream, task_entry.handle)
                task_entry.op_desc.op_out_fn(
                    **updated_kwargs,
                    workspace=workspace,
                    out=out_tensors,
                )
                torch.npu.graph_task_update_end(update_stream)
                task_entry.event.record(update_stream)


def ensure_weak_ref_graph_params():
    """Weak-ref all captured tensors in task entries and workspaces."""
    graph_params = get_graph_params()
    for task_entries in graph_params.task_entries.values():
        for task_entry in task_entries.values():
            task_entry.out_tensors = weak_ref_tensors(task_entry.out_tensors)
            for key in task_entry.op_desc.weak_ref_keys:
                if key in task_entry.captured_kwargs and task_entry.captured_kwargs[key] is not None:
                    task_entry.captured_kwargs[key] = weak_ref_tensor(task_entry.captured_kwargs[key])

    for workspaces in graph_params.workspaces.values():
        for workspace_fn in workspaces:
            workspaces[workspace_fn] = weak_ref_tensors(workspaces[workspace_fn])


@dataclass
class GraphParams:
    task_entries: dict[int, dict[str, GraphTaskEntry]]
    workspaces: dict[int, dict[Callable, torch.Tensor]]

_graph_params: Optional[GraphParams] = None


def set_graph_params(aclgraph_capture_sizes: set[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        task_entries={size: dict() for size in aclgraph_capture_sizes},
        workspaces={size: dict() for size in aclgraph_capture_sizes},
    )


def get_graph_params():
    return _graph_params
