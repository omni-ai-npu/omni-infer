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
    """Run precompiled ranges, compiling an exact range when none matches."""
    # Intentional: this module monkey-patches vLLM internals.
    # pylint: disable=protected-access
    import vllm.compilation.piecewise_backend as _piecewise_module
    from vllm.config.utils import Range

    piecewise_backend = _piecewise_module.PiecewiseBackend
    if getattr(piecewise_backend, "_omni_npu_static_range_patched", False):
        return

    original_call = piecewise_backend.__call__

    def _infer_runtime_shape_from_args(args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim > 0:
                return int(arg.shape[0])
        return None

    def _get_runtime_shape(self, args):
        if self.sym_shape_indices:
            return int(args[self.sym_shape_indices[0]])
        return _infer_runtime_shape_from_args(args)

    def _get_or_create_exact_range_entry(self, runtime_shape):
        compile_range = Range(start=runtime_shape, end=runtime_shape)
        range_entry = self.range_entries.get(compile_range)
        if range_entry is None:
            range_entry = _piecewise_module.RangeEntry(
                compile_range=compile_range
            )
            self.range_entries[compile_range] = range_entry
            logger.warning(
                "Runtime shape %s is outside configured compile ranges %s; "
                "compiling exact fallback range %s.",
                runtime_shape,
                self.compile_ranges,
                compile_range,
            )
        return range_entry

    def _compile_exact_range_entry(self, range_entry, args):
        if range_entry.compiled:
            return
        if self.graph is None:
            raise RuntimeError(
                "Cannot compile an exact fallback range when PiecewiseBackend "
                "was initialized from precompiled artifacts"
            )

        # Match the legacy omni-npu single-size path: compile with the real
        # runtime arguments. In particular, this preserves independent fixed
        # dimensions such as the leading MRoPE dimension (3) while specializing
        # only the runtime token dimension (for example 97).
        self._log_compile_start(range_entry.compile_range)
        range_entry.runnable = self.vllm_backend.compiler_manager.compile(
            self.graph,
            list(args),
            self.vllm_backend.inductor_config,
            self.compilation_config,
            compile_range=range_entry.compile_range,
            graph_index=self.piecewise_compile_index,
            num_graphs=self.total_piecewise_compiles,
            is_encoder=self.vllm_backend.is_encoder,
        )
        range_entry.compiled = True
        if self.is_last_graph:
            self.vllm_backend.compiler_manager.save_to_file()

    @functools.wraps(original_call)
    def _patched_call(self, *args):
        runtime_shape = _get_runtime_shape(self, args)
        if runtime_shape is None:
            raise RuntimeError(
                "Cannot determine runtime shape for PiecewiseBackend"
            )

        range_entry = self._find_range_for_shape(runtime_shape)
        if range_entry is not None:
            # Preserve upstream dispatch for the normal symbolic-shape path.
            if self.sym_shape_indices:
                return original_call(self, *args)
            return range_entry.runnable(*args)

        range_entry = _get_or_create_exact_range_entry(self, runtime_shape)
        _compile_exact_range_entry(self, range_entry, args)
        return range_entry.runnable(*args)

    piecewise_backend.__call__ = _patched_call
    piecewise_backend._omni_npu_static_range_patched = True
    logger.debug("<<< PiecewiseBackend exact-range fallback patched!")


def _patched_mark_dynamic():
    """Use maybe_mark_dynamic instead of mark_dynamic for backed dynamic shapes."""
    # Intentional: this module monkey-patches torch internals.
    # pylint: disable=protected-access
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
    # Intentional: this module monkey-patches vLLM internals.
    # pylint: disable=protected-access
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
