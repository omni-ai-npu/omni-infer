# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compatibility patches for OpenPangu interleaved MRoPE.

Keep vLLM's ``MRotaryEmbeddingInterleaved`` class object intact.  vLLM 0.25.1
uses that class identity to select an out-of-tree implementation in
``CustomOp.__new__``.  Replacing the class object breaks the relationship with
the already registered NPU subclass and can return an object whose
``nn.Module.__init__`` was never called.

Only the OpenPangu MRoPE construction path is extended here.  All ordinary
language-model RoPE requests continue to delegate to vLLM's original
``get_rope`` implementation.
"""

from typing import Any, Literal, Optional

import numpy as np
import torch

from vllm.distributed import get_pp_group
from vllm.model_executor.layers import rotary_embedding
import vllm.model_executor.layers.rotary_embedding as _rope_mod
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    MRotaryEmbeddingInterleaved,
    _ROPE_DICT,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


_orig_get_rope = _rope_mod.get_rope


@register_patch("rotary_embeddingPatch", rotary_embedding)
class RotaryEmbeddingModulePatch(VLLMPatch):
    """Add the legacy wrapper without changing the normal vLLM RoPE path."""

    _attr_names_to_apply = ["get_rope_wrapper"]
    _position_cache: dict[np.dtype, np.ndarray] = {}

    @classmethod
    def apply(cls) -> None:
        """Apply the module and class patches as one backwards-compatible unit."""
        super().apply()
        MRotaryEmbeddingPositionPatch.apply()
        MRotaryEmbeddingInterleavedPatch.apply()

    @classmethod
    def _get_np_position_slice(
        cls,
        start: int,
        end: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Return a cached ``np.arange`` slice for ``[start, end)``."""
        cache = cls._position_cache.get(dtype)
        cache_size = 0 if cache is None else cache.shape[0]
        if cache_size < end:
            new_size = max(end, cache_size * 2 if cache_size > 0 else 1)
            cache = np.arange(new_size, dtype=dtype)
            cls._position_cache[dtype] = cache
        return cache[start:end]

    @staticmethod
    def get_rope_wrapper(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        is_neox_style: bool = True,
        rope_scaling: Optional[dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
        partial_rotary_factor: float = 1.0,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        num_hidden_layers_cache: int = 1,
    ) -> _rope_mod.RotaryEmbedding:
        """Build OpenPangu interleaved MRoPE or delegate to vLLM."""
        if rope_scaling is not None and rope_scaling.get("mrope_interleaved") is True:
            if dtype is None:
                dtype = torch.get_default_dtype()

            rope_scaling_tuple = {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in rope_scaling.items()
            }
            rope_scaling_args = tuple(rope_scaling_tuple.items())

            if partial_rotary_factor < 1.0:
                rotary_dim = int(rotary_dim * partial_rotary_factor)

            key = (
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                rope_scaling_args,
                None,
                dtype,
                num_hidden_layers_cache,
            )
            if key in _ROPE_DICT:
                return _ROPE_DICT[key]

            # A pipeline stage owns only one copy of the shared RoPE cache.
            effective_cache_layers = (
                1 if get_pp_group().world_size > 1 else num_hidden_layers_cache
            )
            rotary_emb = rotary_embedding.MRotaryEmbeddingInterleaved(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_scaling.get("mrope_section"),
                mrope_interleaved=True,
                rotary_mode=rope_scaling.get("rotary_mode", "half"),
                num_hidden_layers_cache=effective_cache_layers,
            )
            _ROPE_DICT[key] = rotary_emb
            return rotary_emb

        # vLLM 0.25.1 reads rotary_dim/partial_rotary_factor from
        # rope_parameters rather than positional arguments.
        rope_parameters = {} if rope_scaling is None else rope_scaling.copy()
        rope_parameters.setdefault("rope_theta", base)
        rope_parameters.setdefault("rope_type", "default")
        if partial_rotary_factor < 1.0:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        if rotary_dim != head_size:
            rope_parameters["rope_dim"] = rotary_dim

        return _orig_get_rope(
            head_size,
            max_position,
            is_neox_style,
            rope_parameters,
            dtype,
            dual_chunk_attention_config,
        )


class MRotaryEmbeddingPositionPatch(VLLMPatch):
    """Retain the allocation-free decode position update."""

    _target = MRotaryEmbedding
    _attr_names_to_apply = ["get_next_input_positions_tensor"]

    @staticmethod
    def get_next_input_positions_tensor(
        out: np.ndarray,
        out_offset: int,
        mrope_position_delta: int,
        context_len: int,
        num_new_tokens: int,
    ) -> None:
        if num_new_tokens <= 0:
            return

        start = mrope_position_delta + context_len
        if num_new_tokens == 1:
            out[:, out_offset] = start
            return

        end = start + num_new_tokens
        values = RotaryEmbeddingModulePatch._get_np_position_slice(
            start, end, out.dtype
        )
        out[:, out_offset:out_offset + num_new_tokens] = values


class MRotaryEmbeddingInterleavedPatch(VLLMPatch):
    """Extend vLLM's class in place so its NPU OOT registration stays valid."""

    _target = MRotaryEmbeddingInterleaved
    _attr_names_to_apply = ["__init__", "get_cos_sin"]

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[list[int]] = None,
        mrope_interleaved: bool = True,
        rotary_mode: Literal["half", "interleave"] = "half",
        num_hidden_layers_cache: int = 1,
    ) -> None:
        if mrope_section is None:
            raise ValueError("mrope_section cannot be None")
        if sum(mrope_section) != rotary_dim // 2:
            raise ValueError("sum(mrope_section) must equal rotary_dim // 2")
        if not mrope_interleaved:
            raise ValueError(
                "mrope_interleaved must be true when mrope_section is provided"
            )
        if rotary_mode not in ("half", "interleave"):
            raise ValueError("rotary_mode must be 'half' or 'interleave'")
        if num_hidden_layers_cache < 1:
            raise ValueError("num_hidden_layers_cache must be at least 1")

        # Call the stable vLLM base explicitly: this function is transplanted
        # onto MRotaryEmbeddingInterleaved, so zero-argument super() would refer
        # to this patch class instead of the target class.
        MRotaryEmbedding.__init__(
            self,
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.rotary_mode = rotary_mode

        if len(mrope_section) == 2:
            height, width = mrope_section
            mrope_dim = self.get_mrope_interleaved_id_list(height, width, 0)
        elif len(mrope_section) == 3:
            temporal, height, width = mrope_section
            mrope_dim = self.get_mrope_interleaved_id_list(
                temporal, height, width, force_last=True
            )
        else:
            raise ValueError("mrope_section must contain two or three sections")

        self.mrope_dim = mrope_dim * 2
        self.mrope_section_3d = [1] * len(self.mrope_dim)
        self.layer_cache = None
        self.layer_counts = 0
        self.num_hidden_layers_cache = num_hidden_layers_cache

        # Warm the host-side decode position cache.  This does not participate
        # in cos/sin computation and therefore cannot alter model numerics.
        RotaryEmbeddingModulePatch._get_np_position_slice(
            0,
            max_position_embeddings * 4 + self.cache_max_position_num,
            np.dtype(np.int64),
        )

    def get_cos_sin(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rebuild_pos_emb(positions)


# Keep the historical class import working for downstream deployments.  The
# patch manager itself uses the stable registration name ``rotary_embeddingPatch``.
rotary_embeddingPatch = RotaryEmbeddingModulePatch
