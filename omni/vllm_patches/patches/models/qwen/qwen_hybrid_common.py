# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared Scheduler / KV-cache utils patches for Qwen3-Next and Qwen3.5."""

from __future__ import annotations

import logging
import os
from copy import copy
from collections.abc import Callable
from dataclasses import replace
from importlib import import_module

import numpy as np
import torch
import torch_npu
import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.model_executor.models.registry import (
    ModelRegistry,
    _ModelInfo,
    _RegisteredModel,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request

from omni_npu.attention.backends.attention import (
    NPUAttentionBackend,
    NPUAttentionBackendImpl,
)
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.plugin_decorators import reinitialize_input_batch_decorator
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu_model_runner import NPUModelRunner

logger = logging.getLogger(__name__)

native_get_kv_cache_config_from_groups = kv_cache_utils.get_kv_cache_config_from_groups
_FIA_BLOCK_SIZE_GRANULARITY = 128
_FIA_MAX_BLOCK_SIZE = 1024

QWEN3_NEXT_REGISTRY_MODEL_INFO = _ModelInfo(
    architecture="Qwen3NextForCausalLM",
    is_text_generation_model=True,
    is_pooling_model=False,
    attn_type="decoder",
    default_seq_pooling_type="LAST",
    default_tok_pooling_type="ALL",
    supports_cross_encoding=False,
    supports_multimodal=False,
    supports_multimodal_raw_input_only=False,
    requires_raw_input_tokens=False,
    supports_multimodal_encoder_tp_data=False,
    supports_pp=True,
    has_inner_state=True,
    is_attention_free=False,
    is_hybrid=True,
    has_noops=False,
    supports_mamba_prefix_caching=False,
    supports_transcription=False,
    supports_transcription_only=False,
)

_hybrid_kv_cache_config_fn: Callable[
    [VllmConfig, list[KVCacheGroupSpec], int], KVCacheConfig
] | None = None
_npu_attention_backend_forward = NPUAttentionBackendImpl.forward


def set_hybrid_kv_cache_config_fn(
    fn: Callable[[VllmConfig, list[KVCacheGroupSpec], int], KVCacheConfig],
) -> None:
    global _hybrid_kv_cache_config_fn
    _hybrid_kv_cache_config_fn = fn


def register_local_qwen3_next_model(model_cls: type) -> None:
    ModelRegistry.models["Qwen3NextForCausalLM"] = _RegisteredModel(
        interfaces=QWEN3_NEXT_REGISTRY_MODEL_INFO,
        model_cls=model_cls,
    )


def apply_local_qwen3_next_registry(
    model_cls: type,
    *,
    success_log: str,
    failure_log: str,
) -> None:
    try:
        register_local_qwen3_next_model(model_cls)
        logger.info(success_log)
    except Exception:
        logger.exception(failure_log)


def register_local_qwen3_next_for_hybrid_patch(
    *,
    success_log: str,
    failure_log: str,
) -> None:
    Qwen3NextForCausalLM = import_module(
        "omni_npu.v1.models.qwen.qwen3_next"
    ).Qwen3NextForCausalLM

    apply_local_qwen3_next_registry(
        Qwen3NextForCausalLM,
        success_log=success_log,
        failure_log=failure_log,
    )


def uniform_type_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    group = kv_cache_groups[0]
    num_blocks = available_memory // group.kv_cache_spec.page_size_bytes
    num_blocks = kv_cache_utils.may_override_num_blocks(vllm_config, num_blocks)
    per_layer_specs = group.kv_cache_spec.kv_cache_specs
    kv_cache_tensors = [
        KVCacheTensor(
            size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
            shared_by=[layer_name],
        )
        for layer_name in group.layer_names
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def get_unpadded_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(kv_cache_spec, MambaSpec):
        return replace(kv_cache_spec, page_size_padded=None)
    return kv_cache_spec


def get_hybrid_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    bytes_per_block = 0
    for i in range(group_size):
        for group in kv_cache_groups:
            if i >= len(group.layer_names):
                continue
            kv_cache_spec = get_unpadded_kv_cache_spec(group.kv_cache_spec)
            bytes_per_block += kv_cache_spec.page_size_bytes
    return bytes_per_block


def build_hybrid_kv_cache_tensors(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
) -> list[KVCacheTensor]:
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    kv_cache_tensors = []
    for i in range(group_size):
        for group in kv_cache_groups:
            if i >= len(group.layer_names):
                continue
            kv_cache_spec = get_unpadded_kv_cache_spec(group.kv_cache_spec)
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=kv_cache_spec.page_size_bytes * num_blocks,
                    shared_by=[group.layer_names[i]],
                )
            )
    return kv_cache_tensors


def get_supported_attention_block_size(block_size: int) -> int:
    if block_size <= _FIA_BLOCK_SIZE_GRANULARITY:
        return _FIA_BLOCK_SIZE_GRANULARITY
    rounded = (
        (block_size + _FIA_BLOCK_SIZE_GRANULARITY - 1)
        // _FIA_BLOCK_SIZE_GRANULARITY
        * _FIA_BLOCK_SIZE_GRANULARITY
    )
    if rounded > _FIA_MAX_BLOCK_SIZE:
        raise ValueError(
            f"Attention block_size {block_size} cannot be rounded to a "
            f"supported FIA block size <= {_FIA_MAX_BLOCK_SIZE}."
        )
    return rounded


def align_attention_block_size(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> list[KVCacheGroupSpec]:
    aligned_groups = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, AttentionSpec):
            supported_block_size = get_supported_attention_block_size(spec.block_size)
            if supported_block_size != spec.block_size:
                group = KVCacheGroupSpec(
                    group.layer_names,
                    replace(spec, block_size=supported_block_size),
                )
        aligned_groups.append(group)
    return aligned_groups


def get_attention_kv_cache_spec(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> AttentionSpec:
    attention_specs = {
        group.kv_cache_spec
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, AttentionSpec)
    }
    if not attention_specs:
        raise ValueError("Qwen hybrid KV cache requires an attention group")
    if len(attention_specs) != 1:
        page_sizes = {spec.page_size_bytes for spec in attention_specs}
        raise ValueError(f"Attention specs must match: {page_sizes}")
    return attention_specs.pop()


def align_hybrid_kv_cache_groups_to_attention(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> list[KVCacheGroupSpec]:
    attention_spec = get_attention_kv_cache_spec(kv_cache_groups)
    page_size = attention_spec.page_size_bytes
    aligned_groups = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, MambaSpec):
            unpadded_spec = replace(spec, page_size_padded=None)
            if unpadded_spec.page_size_bytes > page_size:
                raise ValueError(
                    "Mamba page size must not exceed the attention page size: "
                    f"{unpadded_spec.page_size_bytes=} {page_size=}"
                )
            spec = replace(spec, page_size_padded=page_size)
            group = KVCacheGroupSpec(group.layer_names, spec)
        aligned_groups.append(group)
    return aligned_groups


def reshape_mamba_kv_cache(
    raw_tensor: torch.Tensor,
    kv_cache_spec: MambaSpec,
    num_blocks: int,
) -> tuple[torch.Tensor, ...]:
    state_tensors = []
    offset = 0
    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
        dtype_size = get_dtype_size(dtype)
        target_shape = (num_blocks, *shape)
        size_bytes = int(np.prod(target_shape)) * dtype_size
        final_tensor = raw_tensor[offset:offset + size_bytes].view(dtype).view(target_shape)
        offset += size_bytes
        if not final_tensor.is_contiguous():
            raise ValueError("Mamba state tensor must be contiguous")
        state_tensors.append(final_tensor)
    return tuple(state_tensors)


def reshape_native_mamba_kv_cache(
    raw_tensor: torch.Tensor,
    kv_cache_spec: MambaSpec,
    num_blocks: int,
) -> tuple[torch.Tensor, ...]:
    state_layout = []
    for index, (shape, dtype) in enumerate(
        zip(kv_cache_spec.shapes, kv_cache_spec.dtypes)
    ):
        state_size_bytes = int(np.prod(shape)) * get_dtype_size(dtype)
        state_layout.append((index, shape, dtype, state_size_bytes))
    state_layout.sort(key=lambda item: item[3], reverse=True)

    storage_offsets_bytes = {}
    storage_offset_bytes = 0
    for index, _, _, state_size_bytes in state_layout:
        storage_offsets_bytes[index] = storage_offset_bytes
        storage_offset_bytes += state_size_bytes

    state_tensors = []
    for index, (shape, dtype) in enumerate(
        zip(kv_cache_spec.shapes, kv_cache_spec.dtypes)
    ):
        dtype_size = get_dtype_size(dtype)
        num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
        target_shape = (num_blocks, *shape)
        stride = torch.empty(target_shape).stride()
        target_stride = (num_element_per_page, *stride[1:])
        storage_offset_bytes = storage_offsets_bytes[index]
        if storage_offset_bytes % dtype_size != 0:
            raise ValueError("Mamba state offset must align with dtype size")
        state_tensor = torch.as_strided(
            raw_tensor.view(dtype),
            size=target_shape,
            stride=target_stride,
            storage_offset=storage_offset_bytes // dtype_size,
        )
        state_tensors.append(state_tensor)
        storage_offset_bytes += stride[0] * dtype_size
    return tuple(state_tensors)


def reshape_native_attention_kv_cache(
    raw_tensor: torch.Tensor,
    num_blocks: int,
    kv_cache_spec: AttentionSpec,
) -> tuple[torch.Tensor, ...]:
    block_size = kv_cache_spec.block_size
    num_kv_heads = kv_cache_spec.num_kv_heads
    head_size = kv_cache_spec.head_size
    dtype = kv_cache_spec.dtype
    head_size_v = getattr(kv_cache_spec, "head_size_v", None)
    if head_size_v is None or head_size_v == head_size:
        kv_shapes = [
            NPUAttentionBackend.get_kv_cache_shape(
                num_blocks, block_size, num_kv_heads, head_size
            )
        ] * 2
    else:
        kv_shapes = [
            NPUAttentionBackend.get_kv_cache_shape(
                num_blocks, block_size, num_kv_heads, head_size
            ),
            NPUAttentionBackend.get_kv_cache_shape(
                num_blocks, block_size, num_kv_heads, head_size_v
            ),
        ]

    dtype_size = dtype.itemsize
    page_size = kv_cache_spec.page_size_bytes
    raw_tensor = raw_tensor.view(dtype=dtype)
    if page_size % dtype_size != 0:
        raise RuntimeError(
            f"KV cache page size {page_size} is not aligned to {dtype}."
        )
    if raw_tensor.numel() * dtype_size != page_size * num_blocks:
        raise RuntimeError(
            f"Raw tensor has {raw_tensor.numel() * dtype_size} bytes, while "
            f"the expected KV cache size is {page_size * num_blocks} bytes."
        )

    tensors = []
    storage_offset_bytes = 0
    elements_per_page = page_size // dtype_size
    for tensor_shape in kv_shapes:
        stride = torch.empty(tensor_shape).stride()
        tensor_size_bytes = stride[0] * dtype_size
        if storage_offset_bytes + tensor_size_bytes > page_size:
            raise RuntimeError(
                "KV cache tensors exceed one page: "
                f"{storage_offset_bytes + tensor_size_bytes} > {page_size}."
            )
        tensors.append(
            torch.as_strided(
                raw_tensor,
                size=tensor_shape,
                stride=(elements_per_page, *stride[1:]),
                storage_offset=storage_offset_bytes // dtype_size,
            )
        )
        storage_offset_bytes += tensor_size_bytes
    return tuple(tensors)


def maybe_get_page_dense_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if key_cache.dim() != 3 or value_cache.dim() != 3:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if key_cache.stride()[1:] != value_cache.stride()[1:]:
        return None

    block_size, hidden_size = key_cache.shape[1:]
    dense_block_stride = block_size * hidden_size
    dense_stride = (dense_block_stride, hidden_size, 1)
    page_stride = (dense_block_stride * 2, hidden_size, 1)
    if key_cache.stride() != page_stride:
        return None
    if value_cache.stride() != page_stride:
        return None
    if value_cache.storage_offset() - key_cache.storage_offset() != dense_block_stride:
        return None

    kernel_blocks = key_cache.shape[0] * 2 - 1
    kernel_shape = (kernel_blocks, block_size, hidden_size)
    return (
        torch.as_strided(key_cache, size=kernel_shape, stride=dense_stride),
        torch.as_strided(value_cache, size=kernel_shape, stride=dense_stride),
    )


def scatter_native_attention_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_actual_tokens: int | None = None,
) -> None:
    slots = slot_mapping.reshape(-1)
    if num_actual_tokens is not None:
        slots = slots[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
    block_size = key_cache.shape[1]
    dense_kv = maybe_get_page_dense_kv_cache(key_cache, value_cache)
    if dense_kv is not None:
        key_cache, value_cache = dense_kv
        block_ids = slots // block_size
        block_offsets = slots % block_size
        kernel_slots = (block_ids * 2 * block_size + block_offsets).view(-1, 1)
        torch_npu.npu_scatter_nd_update_(
            key_cache.view(-1, key.shape[-1]),
            kernel_slots,
            key,
        )
        torch_npu.npu_scatter_nd_update_(
            value_cache.view(-1, value.shape[-1]),
            kernel_slots,
            value,
        )
        return

    block_ids = slots // block_size
    block_offsets = slots % block_size
    key_cache[block_ids, block_offsets] = key
    value_cache[block_ids, block_offsets] = value


def should_update_native_attention_kv_cache(
    backend: NPUAttentionBackendImpl,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    attn_metadata,
) -> bool:
    has_custom_models = "omni_custom_models" in os.environ.get("VLLM_PLUGINS", "")
    kv_rmsnorm_enabled = getattr(
        model_extra_config.operator_opt_config,
        "enable_kv_rmsnorm_rope_cache",
        False,
    )
    return (
        backend.kv_sharing_target_layer_name is None
        and key is not None
        and value is not None
        and getattr(attn_metadata, "slot_mapping", None) is not None
        and not (kv_rmsnorm_enabled and has_custom_models)
    )


def reshape_non_attention_kv_cache(
    raw_tensor: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    num_blocks: int,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(kv_cache_spec, MambaSpec):
        raise NotImplementedError(
            f"Unsupported kv_cache_spec type: {type(kv_cache_spec)}"
        )
    return reshape_mamba_kv_cache(raw_tensor, kv_cache_spec, num_blocks)


def reshape_native_hybrid_kv_cache_tensors(
    runner,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    kv_caches: dict[str, torch.Tensor] = {}
    has_tensor, has_tuple = False, False
    for group in runner._kv_cache_spec_attn_group_iterator():
        kv_cache_spec = group.kv_cache_spec
        attn_backend = group.backend

        for layer_name in group.layer_names:
            if layer_name in getattr(runner, 'runner_only_attn_layers', []):
                continue

            raw_tensor = kv_cache_raw_tensors[layer_name]
            if raw_tensor.numel() % kv_cache_spec.page_size_bytes != 0:
                raise ValueError(
                    f"{kv_cache_spec=}, {raw_tensor.numel()=}, "
                    f"{kv_cache_spec.page_size_bytes=}"
                )

            num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes

            if isinstance(kv_cache_spec, AttentionSpec):
                kv_cache_tensors = reshape_native_attention_kv_cache(
                    raw_tensor,
                    num_blocks,
                    kv_cache_spec,
                )
                if (
                    isinstance(kv_cache_tensors, torch.Tensor)
                    and kv_cache_tensors.is_contiguous()
                ):
                    has_tensor = True
            elif isinstance(kv_cache_spec, MambaSpec):
                kv_cache_tensors = reshape_native_mamba_kv_cache(
                    raw_tensor, kv_cache_spec, num_blocks
                )
                has_tuple = True
            else:
                raise NotImplementedError(
                    f"Unsupported kv cache spec type: {type(kv_cache_spec)}"
                )
            kv_caches[layer_name] = kv_cache_tensors

    if has_tensor and has_tuple:
        if hasattr(runner, '_update_hybrid_attention_mamba_layout'):
            runner._update_hybrid_attention_mamba_layout(kv_caches)

    return kv_caches


def _get_attention_group_block_size(kv_cache_config: KVCacheConfig) -> int | None:
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, AttentionSpec):
            return kv_cache_spec.block_size
    return None


@register_patch("QwenNativeAttentionBackendPatch", NPUAttentionBackend)
class QwenNativeAttentionBackendPatch(VLLMPatch):
    _attr_names_to_apply = ["reshape_kv_cache"]

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: AttentionSpec,
    ) -> tuple[torch.Tensor, ...]:
        return reshape_native_attention_kv_cache(
            raw_tensor,
            num_blocks,
            kv_cache_spec,
        )


@register_patch("QwenNativeAttentionImplPatch", NPUAttentionBackendImpl)
class QwenNativeAttentionImplPatch(VLLMPatch):
    _attr_names_to_apply = ["forward"]

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: tuple,
        attn_metadata,
        output: torch.Tensor | None = None,
        trace_flag: bool = True,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_kv_nz = model_extra_config.operator_opt_config.kv_nz
        if (
            not use_kv_nz
            and len(kv_cache) >= 2
            and should_update_native_attention_kv_cache(
                self,
                key,
                value,
                attn_metadata,
            )
        ):
            reshaped_key = key.view(
                -1,
                self.num_kv_heads * key.shape[-1],
            ).contiguous()
            reshaped_value = value.reshape(
                -1,
                self.num_kv_heads * value.shape[-1],
            )
            scatter_native_attention_kv_cache(
                kv_cache[0],
                kv_cache[1],
                attn_metadata.slot_mapping,
                reshaped_key,
                reshaped_value,
                attn_metadata.num_actual_tokens,
            )
            key = None
            value = None

        if len(kv_cache) >= 2:
            dense_kv = maybe_get_page_dense_kv_cache(
                kv_cache[0],
                kv_cache[1],
            )
            block_tables = getattr(attn_metadata, "block_tables", None)
            if dense_kv is not None and block_tables is not None:
                dense_key, dense_value = dense_kv
                kv_cache = (dense_key, dense_value, *kv_cache[2:])
                attn_metadata = copy(attn_metadata)
                attn_metadata.block_tables = block_tables * 2

        result = _npu_attention_backend_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            trace_flag,
            output_scale,
            output_block_scale,
        )
        return result


@register_patch("SchedulerPatch", Scheduler)
class QwenHybridSchedulerPatch(VLLMPatch):
    _attr_names_to_apply = ["_update_waiting_for_remote_kv"]

    def _resolve_block_ids(self, request: Request):
        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
        if len(block_ids) == 1:
            return block_ids[0]

        for idx, group in enumerate(
            self.kv_cache_manager.kv_cache_config.kv_cache_groups
        ):
            if isinstance(group.kv_cache_spec, AttentionSpec):
                return block_ids[idx]
        # Hybrid models always register an AttentionSpec group; if none is found
        # (misconfig/tests), use the first group's flat block list. The legacy
        # inline loop left block_ids as a nested list when no break fired, which
        # broke len(block_ids) in cache_blocks.
        return block_ids[0]

    def _cache_remote_kv_success(self, request: Request) -> None:
        block_ids = self._resolve_block_ids(request)
        # Cap by allocated KV blocks, then by prompt length.
        num_computed_tokens = min(
            len(block_ids) * self.block_size, request.num_tokens
        )
        # When remote KV covers the full prompt, leave one token for the engine
        # to compute locally on the next step (same as prefilled-token patches).
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
        request.num_computed_tokens = num_computed_tokens

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        if self.connector is None:
            raise RuntimeError("connector must be set for remote KV recv")
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            if request.num_computed_tokens:
                self.kv_cache_manager.cache_blocks(
                    request, request.num_computed_tokens
                )
            else:
                self.kv_cache_manager.free(request)
            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            self._cache_remote_kv_success(request)

        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True


@register_patch("KVCacheUtilsPatch", kv_cache_utils)
class QwenHybridKVCacheUtilsPatch(VLLMPatch):
    _attr_names_to_apply = [
        "get_kv_cache_config_from_groups",
        "unify_hybrid_kv_cache_specs",
    ]

    @staticmethod
    def get_kv_cache_config_from_groups(
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        if not kv_cache_groups:
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )

        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            return uniform_type_kv_cache_config(
                vllm_config, kv_cache_groups, available_memory
            )

        if _hybrid_kv_cache_config_fn is None:
            raise RuntimeError(
                "set_hybrid_kv_cache_config_fn() must be called by the active "
                "Qwen hybrid patch module before patches are applied"
            )
        return _hybrid_kv_cache_config_fn(
            vllm_config, kv_cache_groups, available_memory
        )

    @staticmethod
    def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]) -> None:
        """NPU hybrid path: skip vLLM sliding-window -> FullAttention unification.

        Matches upstream ``kv_cache_utils.unify_hybrid_kv_cache_specs`` signature:
        mutates ``kv_cache_spec`` in place (we leave it unchanged) and returns
        None. ``get_kv_cache_groups`` only calls this for side effects.
        """
        return


@register_patch("QwenHybridInputBatchPatch", NPUModelRunner)
class QwenHybridInputBatchPatch(VLLMPatch):
    _attr_names_to_apply = ["may_reinitialize_input_batch"]

    @classmethod
    def apply(cls) -> None:
        target = cls._target
        if "_omni_npu_applied_patches" not in target.__dict__:
            inherited_records = getattr(target, "_omni_npu_applied_patches", {})
            patch_records = dict(inherited_records)
            for name in cls._attr_names_to_apply:
                patch_records.pop(name, None)
            setattr(target, "_omni_npu_applied_patches", patch_records)
        super().apply()

    @staticmethod
    def _check_cpu_offload_disabled(cpu_offload_gb: float) -> None:
        if cpu_offload_gb == 0:
            return
        raise RuntimeError(
            "Cannot re-initialize the input batch when CPU weight "
            "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "
            "for more details."
        )

    @reinitialize_input_batch_decorator
    def may_reinitialize_input_batch(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int],
    ) -> None:
        block_sizes = []
        adjusted_kernel_block_sizes = []
        attention_block_size = _get_attention_group_block_size(kv_cache_config)
        if attention_block_size is None:
            attention_block_size = self.cache_config.block_size
        for kv_cache_group, kernel_block_size in zip(
            kv_cache_config.kv_cache_groups,
            kernel_block_sizes,
        ):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            if isinstance(kv_cache_spec, MambaSpec):
                block_sizes.append(attention_block_size)
                adjusted_kernel_block_sizes.append(attention_block_size)
            else:
                block_sizes.append(kv_cache_spec.block_size)
                adjusted_kernel_block_sizes.append(kernel_block_size)

        if block_sizes != [attention_block_size] or adjusted_kernel_block_sizes != [
            attention_block_size
        ]:
            QwenHybridInputBatchPatch._check_cpu_offload_disabled(
                self.cache_config.cpu_offload_gb
            )
            self._init_npu_input_batch(block_sizes, adjusted_kernel_block_sizes)


KVCacheUtilsPatch = QwenHybridKVCacheUtilsPatch
SchedulerPatch = QwenHybridSchedulerPatch
