# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Attention metadata construction for OmniCache."""

from .attention import fake_build_attention
from .hybrid_attention import construct_fake_attn_metadata
from .compress import fake_build_compress, compute_compress_outlen
from .sliding_window_attention import post_process_fake_metadata
from .dsa import get_volatile_metadata
from .volatile_block_table import apply_volatile_block_table
from .hbm_lane import resolve_batch_idx_reqs

__all__ = [
    "fake_build_attention",
    "construct_fake_attn_metadata",
    "fake_build_compress",
    "post_process_fake_metadata",
    "get_volatile_metadata",
    "compute_compress_outlen",
    "apply_volatile_block_table",
    "resolve_batch_idx_reqs",
]
