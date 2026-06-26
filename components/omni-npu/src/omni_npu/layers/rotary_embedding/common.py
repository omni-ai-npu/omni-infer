# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Common utility functions for rotary positional embeddings.

This module contains utility functions that are reuse.
"""

from typing import Optional
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.common import (rotate_gptj,
                                                                rotate_neox)

logger = init_logger(__name__)


def apply_rotary_emb_full_dim(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Apply rotary embedding using PyTorch operations for full head size.

    Args:
        x: Input tensor of shape [..., head_size]
        cos: Cosine values of shape [..., head_size]
        sin: Sine values of shape [..., head_size]
        is_neox_style: Whether to use Neox-style or GPT-J-style rotation
    Returns:
        Rotated tensor
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x_rotated = rotate_neox(x)
    else:
        x_rotated = rotate_gptj(x)
    return x * cos + x_rotated * sin    


def get_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get cos and sin values for the given positions.

    Args:
        positions: Position indices
        offsets: Optional position offsets

    Returns:
        Tuple of (cos, sin) tensors with shape [batch, 1, 1, rotary_dim]
    """
    if not isinstance(positions, torch.Tensor):
        raise ValueError("positions must be a torch.Tensor.")
    positions = torch.add(positions, offsets) if offsets is not None else positions
    cos = cos[positions].view(-1, 1, 1, cos.shape[-1])
    sin = sin[positions].view(-1, 1, 1, sin.shape[-1])
    return cos, sin


class CachedCosSinMixin:
    """Mixin to expose cached cos/sin lookup with optional seqlen support."""

    def get_cos_sin(
        self,
        positions_or_seqlen: int | torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(positions_or_seqlen, int):
            return super().get_cos_sin(positions_or_seqlen)
        if isinstance(positions_or_seqlen, torch.Tensor) and positions_or_seqlen.ndim == 0:
            return super().get_cos_sin(positions_or_seqlen)
        return get_cos_sin(
            self.cos_cached,
            self.sin_cached,
            positions_or_seqlen,
            offsets,
        )
