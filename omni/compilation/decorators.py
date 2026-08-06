# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import functools

import torch

import vllm.compilation.decorators as _dec_mododule
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.config import CUDAGraphMode


logger = init_logger(__name__)

_COMPILE_DECORATORS_PATCHED = False


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
        logger.debug(f"<<< hit={hit}, retval is Tensor, shape={retval.shape if hasattr(retval, 'shape') else 'N/A'}")
        if hit:
            return retval
        logger.debug(f"<<< hit={hit}, use original_call")
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


def _patch_piecewise_backend():
    """Run a precompiled entry when the graph has no symbolic shape input."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    piecewise_backend = _piecewise_module.PiecewiseBackend
    if getattr(piecewise_backend, "_omni_npu_static_range_patched", False):
        return

    original_call = piecewise_backend.__call__

    def _infer_runtime_shape_from_args(args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim > 0:
                return int(arg.shape[0])
        return None

    @functools.wraps(original_call)
    def _patched_call(self, *args):
        if self.sym_shape_indices:
            return original_call(self, *args)

        runtime_shape = _infer_runtime_shape_from_args(args)
        assert runtime_shape is not None, (
            "Cannot infer runtime shape for PiecewiseBackend without SymInt inputs"
        )
        range_entry = self._find_range_for_shape(runtime_shape)
        assert range_entry is not None, (
            f"Shape {runtime_shape} is outside compile ranges "
            f"{self.compile_ranges}"
        )
        return range_entry.runnable(*args)

    piecewise_backend.__call__ = _patched_call
    piecewise_backend._omni_npu_static_range_patched = True
    logger.debug("<<< PiecewiseBackend static range dispatch patched!")


def _patched_mark_dynamic():
    """Use maybe_mark_dynamic instead of mark_dynamic for backed dynamic shapes."""
    import torch._dynamo as dynamo

    if getattr(dynamo, "_omni_npu_maybe_mark_dynamic", False):
        return

    maybe_mark_dynamic = getattr(dynamo, "maybe_mark_dynamic", None)
    if maybe_mark_dynamic is None:
        logger.warning(
            "torch._dynamo.maybe_mark_dynamic is unavailable; "
            "skip omni-npu mark_dynamic patch"
        )
        return

    dynamo.mark_dynamic = maybe_mark_dynamic
    dynamo._omni_npu_maybe_mark_dynamic = True
    logger.debug("<<< _patched_mark_dynamic applied!")


def patch_compile_decorators():
    global _COMPILE_DECORATORS_PATCHED
    if _COMPILE_DECORATORS_PATCHED:
        return
    _patch_piecewise_backend()
    _patched_mark_dynamic()
    _original_decorator = _dec_mododule._support_torch_compile

    def _patched_support_torch_compile(cls, *args, **kwargs):
        cls = _original_decorator(cls, *args, **kwargs)

        cls.__call__ = _wrap_call(cls.__call__)
        logger.debug("<<< cls.__call__ wrapped!")
        return cls

    _dec_mododule._support_torch_compile = _patched_support_torch_compile
    _COMPILE_DECORATORS_PATCHED = True
    logger.debug("<<< _patched_support_torch_compile applied!")
