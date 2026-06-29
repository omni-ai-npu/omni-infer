# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared helpers for attention metadata-builder extension tests."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch


def patch_cpu_allocators(monkeypatch):
    """Drop CPU-unsupported pinned-memory kwargs from common tensor allocators."""
    torch_empty = torch.empty
    torch_full = torch.full

    def empty_without_pin(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_empty(*args, **kwargs)

    def full_without_pin(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_full(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_without_pin)
    monkeypatch.setattr(torch, "full", full_without_pin)


def load_extension_module(monkeypatch, module_path: Path, module_prefix: str):
    patch_cpu_allocators(monkeypatch)
    module_name = f"{module_prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_vllm_config(
    *,
    enable_prefix_caching: bool = False,
    max_model_len: int = 4096,
    block_size: int = 128,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    num_speculative_tokens: int | None = None,
    router_sliding_window: int = 3,
):
    hf_config = SimpleNamespace(router_sliding_window=router_sliding_window)
    hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=512,
        qk_rope_head_dim=64,
        v_head_dim=512,
    )
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=hf_config,
        hf_text_config=hf_text_config,
        max_model_len=max_model_len,
        get_num_attention_heads=lambda parallel_config: 1,
        get_head_size=lambda: 576,
    )
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            disable_flashinfer_prefill=True,
            use_cudnn_prefill=False,
            use_trtllm_ragged_deepseek_prefill=False,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            max_cudagraph_capture_size=None,
        ),
        kv_transfer_config=None,
        model_config=model_config,
        parallel_config=SimpleNamespace(
            cp_kv_cache_interleave_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        speculative_config=None
        if num_speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=num_speculative_tokens),
    )


def patch_current_vllm_config(monkeypatch, vllm_config):
    import importlib

    config_mod = importlib.import_module("vllm.config")
    monkeypatch.setattr(config_mod, "_current_vllm_config", vllm_config, raising=False)
    config_vllm_mod = importlib.import_module("vllm.config.vllm")
    monkeypatch.setattr(
        config_vllm_mod, "_current_vllm_config", vllm_config, raising=False
    )


def make_common_attn_metadata(
    *,
    query_lens: list[int],
    seq_lens: list[int] | None = None,
    first_column: list[int] | None = None,
    slot_mapping: list[int] | torch.Tensor | None = None,
    max_blocks: int | None = None,
    block_fill_value: int = 777,
):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    if seq_lens is None:
        seq_lens = [100 + i + q_len for i, q_len in enumerate(query_lens)]
    assert len(seq_lens) == len(query_lens)

    starts = [0]
    for q_len in query_lens:
        starts.append(starts[-1] + q_len)
    query_start_loc = torch.tensor(starts, dtype=torch.int32)

    if first_column is not None:
        assert len(first_column) == len(query_lens)
        block_table = torch.tensor(
            [[value, value + 100] for value in first_column],
            dtype=torch.int32,
        )
    else:
        assert max_blocks is not None
        block_table = torch.full(
            (len(seq_lens), max_blocks),
            block_fill_value,
            dtype=torch.int32,
        )

    if slot_mapping is None:
        slot_mapping_tensor = torch.arange(sum(query_lens), dtype=torch.int64)
    elif isinstance(slot_mapping, torch.Tensor):
        slot_mapping_tensor = slot_mapping
    else:
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(seq_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping_tensor,
    )


def make_decode_cache(idx_reqs: list[int] | None):
    req_ids = [f"req-{i}" for i in range(len(idx_reqs or []))]
    return SimpleNamespace(
        num_max_batch_pool=max(8, len(req_ids) + 1),
        record_batch_idx_to_req={}
        if idx_reqs is None
        else {idx: req_id for req_id, idx in zip(req_ids, idx_reqs)},
        req_ids_update_buffer=None if idx_reqs is None else req_ids,
        req_id_to_idx=None
        if idx_reqs is None
        else {req_id: idx for req_id, idx in zip(req_ids, idx_reqs)},
    )
