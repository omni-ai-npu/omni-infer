# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import functools
from typing import TypeVar, Union

import torch
import torch.nn as nn

import vllm.compilation.decorators as _dec_mododule
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.config import CUDAGraphMode


logger = init_logger(__name__)


_T = TypeVar('_T', bound=type[nn.Module])

def support_ge_compile(
        cls: _T,
        dynamic_arg_dims: dict[str, Union[int, list[int]]],
        *args, **kwargs
) -> _T:
    from omni_npu.compilation.ge_wrapper import TorchNpuCompilerWrapperWithCustomDispatcher
    from vllm.compilation.counter import compilation_counter
    from vllm.config import VllmConfig

    if TorchNpuCompilerWrapperWithCustomDispatcher in cls.__bases__:
        return cls

    cls.__bases__ = cls.__bases__ + (TorchNpuCompilerWrapperWithCustomDispatcher,)

    old_init = cls.__init__

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.vllm_config = vllm_config
        compilation_counter.num_models_seen += 1
        TorchNpuCompilerWrapperWithCustomDispatcher.__init__(
            self, vllm_config, dynamic_arg_dims)

    cls.__init__ = __init__
    cls.__call__ = TorchNpuCompilerWrapperWithCustomDispatcher.__call__

    return cls

def _bypass_prefill(self, *args, **kwargs):
    """
    patch vllm's _support_torch_compile's __call__
    If any prefill request exists, torch.all_to_all_single will be used
    in MoE layers, which involves CPU operations and cannot be compiled.
    We use the non-compiled forward for this case.
    """
    attn_metadata = get_forward_context().attn_metadata
    has_prefill = attn_metadata is None or attn_metadata[next(iter(attn_metadata))].num_prefills > 0
    # FIXME (zhao): currently we only support full cudagraph mode for compiled graphs.
    if has_prefill or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL:
        logger.debug(f"<<< use original forward")
        return True, self.forward(*args, **kwargs)
    return False, None

def _wrap_call(original_call):
    @functools.wraps(original_call)
    def _new_call(self, *args, **kwargs):
        hit, retval = _bypass_prefill(self, *args, **kwargs)
        logger.debug(f"<<< {hit=}, {retval=}")
        if hit:
            return retval
        logger.debug(f"<<< {hit=}, {retval=}, use original_call")
        model_output = original_call(self, *args, **kwargs)
        if isinstance(model_output, (tuple, list)) and len(model_output) == 1:
            hidden_states = model_output[0]
            if isinstance(hidden_states, list) and \
                    len(hidden_states) == 1 and \
                    isinstance(hidden_states[0], torch.Tensor):
                    hidden_states = hidden_states[0]
            return hidden_states
        else:
            return model_output
    return _new_call

def _patched_mark_dynamic():
    import inspect
    import sys
    origin_torch_compile = _dec_mododule._support_torch_compile
    origin_torch_compile_str = inspect.getsource(origin_torch_compile)
    new_torch_compile_str = origin_torch_compile_str.replace('torch._dynamo.mark_dynamic',
                                                             'torch._dynamo.maybe_mark_dynamic')
    new_torch_compile = {}
    module_globals = sys.modules[_dec_mododule.__name__].__dict__
    exec(new_torch_compile_str, module_globals, new_torch_compile)
    _dec_mododule._support_torch_compile = new_torch_compile["_support_torch_compile"]
    logger.debug("<<< _patched_mark_dynamic applied!")


def _patch_piecewise_backend():
    import vllm.compilation.piecewise_backend as _piecewise_module
    from vllm.config.utils import Range

    if getattr(_piecewise_module.PiecewiseBackend, "_omni_npu_patched", False):
        return

    original_call = _piecewise_module.PiecewiseBackend.__call__

    def _infer_runtime_shape_from_args(args):
        for arg in args:
            if hasattr(arg, "shape") and len(arg.shape) > 0:
                return int(arg.shape[0])
        return None

    def _get_fallback_range_entry(self, args):
        if not self.sym_shape_indices:
            runtime_shape = _infer_runtime_shape_from_args(args)
        else:
            runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape is not None:
            range_entry = self._find_range_for_shape(runtime_shape)
            if range_entry is not None:
                return range_entry

            range = Range(start=runtime_shape, end=runtime_shape)
            if range not in self.range_entries:
                self.range_entries[range] = _piecewise_module.RangeEntry(
                    compile_range=range
                )
                self.to_be_compiled_ranges.add(range)
            return self.range_entries[range]

        if self.compile_sizes:
            range = Range(start=self.compile_sizes[0], end=self.compile_sizes[0])
            return self.range_entries[range]

        if self.compile_ranges:
            return self.range_entries[self.compile_ranges[0]]

        raise RuntimeError(
            "Cannot determine fallback compile range for piecewise graph"
        )

    @functools.wraps(original_call)
    def _patched_call(self, *args):
        range_entry = _get_fallback_range_entry(self, args)
        self._maybe_compile_for_range_entry(range_entry, args)
        return range_entry.runnable(*args)

    _piecewise_module.PiecewiseBackend.__call__ = _patched_call
    _piecewise_module.PiecewiseBackend._omni_npu_patched = True
    logger.debug("<<< PiecewiseBackend.__call__ patched!")

def patch_compile_decorators():
    import os
    use_gegraph = os.getenv("TORCH_COMPILE_GE", "False").lower() == "true"
    if use_gegraph:
        logger.debug("<<< patch_compile_decorators:use ge graph!")
        _dec_mododule._support_torch_compile = support_ge_compile
    else:
        _patch_piecewise_backend()
        _patched_mark_dynamic()
        _original_decorator = _dec_mododule._support_torch_compile
        def _patched_support_torch_compile(cls, *args, **kwargs):
            cls = _original_decorator(cls, *args, **kwargs)

            cls.__call__ = _wrap_call(cls.__call__)
            logger.debug("<<< cls.__call__ wrapped!")
            return cls

        _dec_mododule._support_torch_compile = _patched_support_torch_compile
        logger.debug("<<< _patched_support_torch_compile applied!")

