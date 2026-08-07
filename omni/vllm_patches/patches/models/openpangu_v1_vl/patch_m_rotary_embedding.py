# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from typing import Any, Dict, Optional, List, Literal
from einops import rearrange
import numpy as np

import torch
import torch_npu

from vllm.distributed import get_pp_group
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT 
import vllm.model_executor.layers.rotary_embedding as _rope_mod 
from vllm.model_executor.layers import rotary_embedding

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


_orig_get_rope = _rope_mod.get_rope


@register_patch("rotary_embeddingPatch", rotary_embedding)
class rotary_embeddingPatch(VLLMPatch):
    _attr_names_to_apply = ['get_rope_wrapper', 'MRotaryEmbeddingInterleaved']
    # Class-level NumPy cache for decode position slices, keyed by dtype.
    _position_cache: Dict[np.dtype, np.ndarray] = {}

    #####patch start: for pangu72B-VL
    @classmethod
    def _get_np_position_slice(
        cls,
        start: int,
        end: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Get a cached np.arange slice for [start, end)."""
        cache = cls._position_cache.get(dtype)
        cache_size = 0 if cache is None else cache.shape[0]
        if cache_size < end:
            new_size = max(end, cache_size * 2 if cache_size > 0 else 1)
            cache = np.arange(new_size, dtype=dtype)
            cls._position_cache[dtype] = cache
        return cache[start:end]

    @staticmethod 
    def get_next_input_positions_tensor(
        out: np.ndarray,
        out_offset: int,
        mrope_position_delta: int,
        context_len: int,
        num_new_tokens: int,
    ) -> None:
        """Get next input positions tensor using cached NumPy arange slices."""
        if num_new_tokens <= 0:
            return

        start = mrope_position_delta + context_len
        # Decode commonly appends one token; avoid slice/cache overhead.
        if num_new_tokens == 1:
            out[:, out_offset] = start
            return

        end = start + num_new_tokens
        values = rotary_embeddingPatch._get_np_position_slice(start, end, out.dtype)
        out[:, out_offset : out_offset + num_new_tokens] = values

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
        num_hidden_layers_cache: int = 1
    ) -> _rope_mod.RotaryEmbedding:
        if rope_scaling is not None and rope_scaling.get("mrope_interleaved") == True:
            # adapt Replacing legacy 'type' key with 'rope_type' in 0.6.3
            if dtype is None:
                dtype = torch.get_default_dtype()
            if rope_scaling is not None:
                rope_scaling_tuple = {k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()}
                rope_scaling_args = tuple(rope_scaling_tuple.items())
            else:
                rope_scaling_args = None
            
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
                num_hidden_layers_cache
            )
            if key in _ROPE_DICT:
                return _ROPE_DICT[key]

            mrope_section = rope_scaling.get("mrope_section")
            rotary_mode = rope_scaling.get("rotary_mode", "half")
            num_hidden_layers_cache = 1 if get_pp_group().world_size > 1 else num_hidden_layers_cache

            rotary_emb = rotary_embedding.MRotaryEmbeddingInterleaved(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=mrope_section,
                mrope_interleaved=True,
                rotary_mode=rotary_mode,
                num_hidden_layers_cache=num_hidden_layers_cache
            )

            _ROPE_DICT[key] = rotary_emb
            return rotary_emb

        if rope_scaling is None:
            rope_parameters = {"rope_theta": base, "rope_type": "default"}
        else:
            rope_parameters = rope_scaling.copy()
            if "rope_theta" not in rope_parameters:
                rope_parameters["rope_theta"] = base
            if "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = "default"

        return _orig_get_rope(
            head_size,
            rotary_dim,
            max_position,
            is_neox_style,
            rope_parameters,
            dtype,
            partial_rotary_factor,
            dual_chunk_attention_config,
        )
    #####patch start: for pangu72B-VL

    rotary_embedding.get_rope_wrapper = get_rope_wrapper


    #####patch start: for pangu72B-VL
    class MRotaryEmbeddingInterleaved(MRotaryEmbedding):
        """Rotary Embedding with Multimodal Sections and Interleaved Support."""

        def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: float,
            is_neox_style: bool,
            dtype: torch.dtype,
            mrope_section: Optional[List[int]] = None,
            mrope_interleaved: bool = True,
            rotary_mode: Literal["half", "interleaved"] = "half",
            num_hidden_layers_cache: int = 1
        ) -> None:
            # Enlarge max_position_embeddings for video inputs
            self.cache_max_position_num = max_position_embeddings
            super().__init__(
                head_size,
                rotary_dim,
                self.cache_max_position_num,
                base,
                is_neox_style,
                dtype,
            )

            self.mrope_section = mrope_section
            self.mrope_interleaved = mrope_interleaved
            self.rotary_mode = rotary_mode

            if self.mrope_section is None:
                raise ValueError("mrope_section cannot be None.")
            if sum(self.mrope_section) != rotary_dim // 2:
                raise ValueError(
                    "Sum of mrope_section must equal rotary_dim // 2.")
            if not self.mrope_interleaved:
                raise ValueError(
                    "mrope_interleaved must be True when mrope_section is provided.")

            # Generate interleaved indices
            if len(mrope_section) == 2:
                h_num, w_num = mrope_section[0], mrope_section[1]
                mrope_dim = self.get_mrope_interleaved_id_list(h_num, w_num, 0)
            elif len(mrope_section) == 3:
                t_num, h_num, w_num = mrope_section[0], mrope_section[1], mrope_section[2]
                mrope_dim = self.get_mrope_interleaved_id_list(
                    t_num, h_num, w_num, force_last=True)
            else:
                raise AssertionError(
                    "Cannot support the length of mrope section is not 2 or 3.")

            mrope_dim = mrope_dim * 2
            self.mrope_dim = mrope_dim
            self.mrope_section_3d = [1] * len(self.mrope_dim)

            self.layer_cache = None
            self.layer_counts = 0
            self.num_hidden_layers_cache = num_hidden_layers_cache

            rotary_embeddingPatch._get_np_position_slice(
                0,
                max_position_embeddings * 4 + self.cache_max_position_num,
                np.dtype(np.int64),
            )

        def _rebuild_pos_emb(
            self,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Interleave the rotary embedding"""
            cos_sin = self.cos_sin_cache[positions]
            
            cos_sin = torch.cat(
                [m[self.mrope_dim[i]]
                    for i, m in enumerate(cos_sin.split(self.mrope_section_3d, dim=-1))],
                dim=-1,
            )
            cos, sin = cos_sin.chunk(2, dim=-1)
            
            if self.rotary_mode == 'half':
                cos = torch.cat((cos, cos), dim=-1)
                sin = torch.cat((sin, sin), dim=-1)
            elif self.rotary_mode == 'interleave':
                cos = rearrange(torch.stack((cos, cos), dim=-1), "... d two -> ...(d two)", two=2)
                sin = rearrange(torch.stack((sin, sin), dim=-1), "... d two -> ...(d two)", two=2)
            else:
                raise ValueError("only support half or interleave")
            cos = cos.reshape(-1, 1, 1, self.rotary_dim)
            sin = sin.reshape(-1, 1, 1, self.rotary_dim)

            return cos, sin

        def get_cos_sin(
            self,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._rebuild_pos_emb(positions)
        
        def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: Optional[torch.Tensor] = None
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Forward pass with interleaved rotary embedding."""
            if self.layer_counts % self.num_hidden_layers_cache == 0:
                cos, sin = self._rebuild_pos_emb(positions)
                self.layer_cache = (cos, sin)
                self.layer_counts = 0
            else:
                cos, sin = self.layer_cache
            self.layer_counts += 1

            num_tokens = query.shape[0]

            query = query.view(num_tokens, 1, -1, self.head_size)
            query_rot = query[..., :self.rotary_dim]
            query_pass = query[..., self.rotary_dim:]

            key = key.view(num_tokens, 1, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]

            query_rot = torch_npu.npu_rotary_mul(query_rot.contiguous(), cos, sin, rotary_mode=self.rotary_mode)
            key_rot = torch_npu.npu_rotary_mul(key_rot.contiguous(), cos, sin, rotary_mode=self.rotary_mode)

            query = torch.cat((query_rot, query_pass), dim=-1).reshape(num_tokens, -1)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(num_tokens, -1)

            return query, key

        @staticmethod
        def get_mrope_interleaved_id_list(a: int, b: int, c: int, force_last: bool = False) -> List[int]:
            """
            Generate an interleaved list of indices for multi-modal rotary embedding.

            Args:
                a: Number of indices for first modality
                b: Number of indices for second modality
                c: Number of indices for third modality
                force_last: Whether to force the last element to be from the first modality

            Returns:
                List of interleaved indices
            """
            if force_last:
                a -= 1

            counts = {0: a, 1: b, 2: c}
            placed = {k: 0 for k in counts}
            rem = counts.copy()
            seq: List[int] = []
            last = None

            total = a + b + c
            for _ in range(total):
                # Candidates: remaining > 0 and ≠ last
                cands = [k for k in rem if rem[k] > 0 and k != last]
                if not cands:
                    # If only last remains, relax the condition
                    cands = [k for k in rem if rem[k] > 0]

                # Select the rarest candidate
                try:
                    best = min(cands, key=lambda k: (placed[k] / counts[k], k))
                except KeyError:
                    best = 0

                seq.append(best)
                placed[best] += 1
                rem[best] -= 1
                last = best

            if force_last:
                seq.append(0)

            return seq
    #####patch end

    rotary_embedding.MRotaryEmbeddingInterleaved = MRotaryEmbeddingInterleaved
    rotary_embedding.MRotaryEmbedding.get_next_input_positions_tensor = staticmethod(
        get_next_input_positions_tensor
    )