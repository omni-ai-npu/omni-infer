# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Core gather selection operations for OmniCache.

This module provides the core gather_selection function which is responsible
for selecting relevant KV cache blocks based on top-k indices for compressed
attention mechanisms during decode.
"""

import os

import torch
import torch_npu

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache


def _resolve_gs_table_status(
    omni_cache: "DecodeOmniCache",
    layer_idx: int,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(os.getenv("USE_OMNI_INPUT_BATCH", "0")):
        input_batch = omni_cache.input_batch
        if input_batch is None or not hasattr(input_batch, "has_gs_buffers") or not input_batch.has_gs_buffers():
            raise RuntimeError("GatherSelection buffers are not available on OmniCacheInputBatch")
        return (
            input_batch.get_gs_table_tensor(num_reqs),
            input_batch.get_gs_status_tensor(layer_idx, num_reqs),
        )

    return (
        omni_cache.selection_kv_block_table[:num_reqs],
        omni_cache.selection_kv_block_status[layer_idx, :num_reqs],
    )


def gather_selection(
    omni_cache: "DecodeOmniCache",
    attn_kwargs: Dict[str, Any],
    layer_idx: int,
    compress_ratio: int = 1,
    attn_type: str = "SAS",
) -> None:
    """执行 gather selection 操作。

    从完整的 KV cache 中根据 top-k 索引选择相关的 KV 块，
    用于压缩注意力机制。

    Args:
        omni_cache: DecodeOmniCache 实例
        attn_kwargs: 包含注意力相关张量的字典
        layer_idx: 当前层索引
        compress_ratio: 压缩比率
    """
    if attn_type == "SAS":
        sas_gather_selection(omni_cache, attn_kwargs, layer_idx, compress_ratio)
    elif attn_type == "DSA":
        dsa_gather_selection(omni_cache, attn_kwargs, layer_idx, compress_ratio)


def dsa_gather_selection(
    omni_cache: "DecodeOmniCache", attn_kwargs: Dict[str, Any], layer_idx: int, compress_ratio: int = 1
) -> None:
    bsz_seq, _, topk = attn_kwargs["sparse_indices"].size()
    bs = attn_kwargs["actual_seq_lengths_kv"].size(0)
    selection_topk_indices_npu = attn_kwargs["sparse_indices"]
    if "key_rope" not in attn_kwargs:
        if layer_idx == 0:
            omni_cache.full_k_rope = torch.zeros(
                [attn_kwargs["value"].size(0), attn_kwargs["value"].size(1), 1],
                device=attn_kwargs["value"].device,
                dtype=attn_kwargs["value"].dtype,
            )
        full_k_rope_npu = omni_cache.full_k_rope
    else:
        full_k_rope_npu = attn_kwargs["key_rope"].squeeze(-2)

    selection_k_rope = omni_cache.selection_k_rope[layer_idx]
    selection_kv_cache_npu = omni_cache.selection_kv_cache[layer_idx]
    selection_kv_block_table_npu, selection_kv_block_status_npu = _resolve_gs_table_status(omni_cache, layer_idx, bs)

    full_kv_block_table_npu = attn_kwargs["block_table"]
    full_kv_cache_npu = attn_kwargs["value"].squeeze(-2)
    full_kv_actual_seq_npu = attn_kwargs["actual_seq_lengths_kv"] // compress_ratio
    full_q_actual_seq_npu = attn_kwargs["actual_seq_lengths_query"]

    selection_kwargs = {
        "selection_k_rope": selection_k_rope,
        "selection_kv_cache": selection_kv_cache_npu,
        "selection_kv_block_table": selection_kv_block_table_npu,
        "selection_kv_block_status": selection_kv_block_status_npu,
        "selection_topk_indices": selection_topk_indices_npu,
        "full_k_rope": full_k_rope_npu,
        "full_kv_cache": full_kv_cache_npu,
        "full_kv_block_table": full_kv_block_table_npu,
        "full_kv_actual_seq": full_kv_actual_seq_npu,
        "full_q_actual_seq": full_q_actual_seq_npu,
        "selection_topk_block_size": 1,
    }

    reusedNum = torch_npu.npu_gather_selection_kv_cache(**selection_kwargs)

    # 计算复用率
    q_seq_diff = torch.cat([full_q_actual_seq_npu[0:1], full_q_actual_seq_npu[1:] - full_q_actual_seq_npu[:-1]])
    total_blocks = torch.clamp(full_kv_actual_seq_npu, max=topk) * q_seq_diff
    omni_cache.reuse_rate[layer_idx] = omni_cache.reuse_rate[
        layer_idx
    ] * omni_cache.record_smooth_alpha + reusedNum.sum() / total_blocks.sum() * (1 - omni_cache.record_smooth_alpha)

    selection_k_rope = selection_k_rope.unsqueeze(-2)
    selection_kv_cache_npu = selection_kv_cache_npu.unsqueeze(-2)

    attn_kwargs["value"] = selection_kv_cache_npu
    attn_kwargs["block_table"] = selection_kv_block_table_npu


def sas_gather_selection(
    omni_cache: "DecodeOmniCache", attn_kwargs: Dict[str, Any], layer_idx: int, compress_ratio: int = 1
) -> None:
    from vllm.logger import init_logger

    logger = init_logger("vllm.v1.omni")

    selection_topk_indices_npu = attn_kwargs["cmp_sparse_indices"]

    bs = attn_kwargs["seqused_kv"].size(0)

    if layer_idx == 0:
        omni_cache.full_k_rope = torch.zeros(
            [attn_kwargs["cmp_kv"].size(0), attn_kwargs["cmp_kv"].size(1), 1],
            device=attn_kwargs["cmp_kv"].device,
            dtype=attn_kwargs["cmp_kv"].dtype,
        )

    full_kv_block_table_npu = attn_kwargs["cmp_block_table"]
    full_kv_cache_npu = attn_kwargs["cmp_kv"].squeeze(-2)
    full_k_rope_npu = omni_cache.full_k_rope
    full_kv_actual_seq_npu = attn_kwargs["seqused_kv"] // compress_ratio
    full_q_actual_seq_npu = attn_kwargs["cu_seqlens_q"][1:]

    selection_k_rope = omni_cache.selection_k_rope[layer_idx]
    selection_kv_cache_npu = omni_cache.selection_kv_cache[layer_idx]
    selection_kv_block_table_npu, selection_kv_block_status_npu = _resolve_gs_table_status(omni_cache, layer_idx, bs)

    selection_kwargs = {
        "selection_k_rope": selection_k_rope,
        "selection_kv_cache": selection_kv_cache_npu,
        "selection_kv_block_table": selection_kv_block_table_npu,
        "selection_kv_block_status": selection_kv_block_status_npu,
        "selection_topk_indices": selection_topk_indices_npu,
        "full_k_rope": full_k_rope_npu,
        "full_kv_cache": full_kv_cache_npu,
        "full_kv_block_table": full_kv_block_table_npu,
        "full_kv_actual_seq": full_kv_actual_seq_npu,
        "full_q_actual_seq": full_q_actual_seq_npu,
        "selection_topk_block_size": 1,
    }

    reusedNum = torch_npu.npu_gather_selection_kv_cache(**selection_kwargs)

    attn_kwargs["cmp_kv"] = selection_kv_cache_npu.unsqueeze(-2)
    attn_kwargs["cmp_block_table"] = selection_kv_block_table_npu


__all__ = ["gather_selection"]
