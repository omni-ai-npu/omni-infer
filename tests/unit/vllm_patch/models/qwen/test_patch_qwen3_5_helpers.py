# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

import torch
from types import SimpleNamespace

from omni.vllm_patches.patches.models.qwen import qwen_hybrid_common as hybrid
from omni.vllm_patches.patches.models.qwen3_5 import patch_gdn_attn


def test_mask_padded_state_indices_masks_one_dimensional_indices():
    indices = torch.tensor([3, 4, 5], dtype=torch.int32)
    query_start_loc_cpu = torch.tensor([0, 1, 1, 3], dtype=torch.int32)

    result = patch_gdn_attn._mask_padded_state_indices(
        indices,
        query_start_loc_cpu,
    )

    assert result.tolist() == [3, patch_gdn_attn.PAD_SLOT_ID, 5]
    assert indices.tolist() == [3, 4, 5]


def test_mask_padded_state_indices_masks_spec_rows():
    indices = torch.tensor([[3, 4], [5, 6], [7, 8]], dtype=torch.int32)
    query_start_loc_cpu = torch.tensor([0, 1, 1, 3], dtype=torch.int32)

    result = patch_gdn_attn._mask_padded_state_indices(
        indices,
        query_start_loc_cpu,
    )

    assert result.tolist() == [
        [3, 4],
        [patch_gdn_attn.PAD_SLOT_ID, patch_gdn_attn.PAD_SLOT_ID],
        [7, 8],
    ]


def test_mask_padded_state_indices_returns_none_and_unmodified_without_padding():
    assert patch_gdn_attn._mask_padded_state_indices(
        None,
        torch.tensor([0, 1], dtype=torch.int32),
    ) is None
    indices = torch.tensor([3, 4], dtype=torch.int32)
    result = patch_gdn_attn._mask_padded_state_indices(
        indices,
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert result is indices


def test_page_dense_helper_returns_none_for_non_page_strided_cache():
    key_cache = torch.zeros(2, 2, 4)
    value_cache = torch.zeros(2, 2, 4)

    assert hybrid.maybe_get_page_dense_kv_cache(key_cache, value_cache) is None


def test_qwen35_patch_module_exports_shared_entrypoints():
    from omni.vllm_patches.patches.models.qwen3_5 import patch_qwen3_5
    from omni.vllm_patches.patches.models.qwen import patch_qwen3_next

    assert patch_qwen3_5.NPUModelRunnerPatch is patch_qwen3_next.NPUModelRunnerPatch
    assert patch_qwen3_5.SchedulerPatch is patch_qwen3_next.SchedulerPatch


def _builder(monkeypatch, *, use_spec_decode):
    monkeypatch.setattr(
        patch_gdn_attn,
        "GDNAttentionMetadata",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    builder = patch_gdn_attn.GDNAttentionMetadataBuilderPatch.__new__(
        patch_gdn_attn.GDNAttentionMetadataBuilderPatch
    )
    builder.use_spec_decode = use_spec_decode
    builder.num_spec = 1
    builder.decode_cudagraph_max_bs = 0
    builder.use_full_cuda_graph = False
    return builder


def test_gdn_metadata_builder_populates_non_spec_cpu_lists(monkeypatch):
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
        block_table_tensor=torch.tensor([[4], [5]], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=3,
        compute_num_computed_tokens=lambda: torch.tensor([0, 1]),
    )
    monkeypatch.setattr(
        patch_gdn_attn,
        "split_decodes_and_prefills",
        lambda _m, decode_threshold: (1, 1, 1, 2),
    )
    monkeypatch.setattr(
        patch_gdn_attn,
        "compute_causal_conv1d_metadata",
        lambda _loc: (None, None, None),
    )

    result = _builder(monkeypatch, use_spec_decode=False).build(
        0, metadata, fast_build=False
    )

    assert result.non_spec_query_start_loc_cpu == [0, 1, 3]
    assert result.non_spec_seqlens_list == [1, 2]
    assert result.spec_sequence_masks is None
    assert result.has_initial_state.tolist() == [False, True]


def test_gdn_metadata_builder_handles_speculative_batch(monkeypatch):
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
        block_table_tensor=torch.tensor([[4, 6], [5, 7]], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=3,
        compute_num_computed_tokens=lambda: torch.tensor([1, 2]),
    )
    monkeypatch.setattr(
        patch_gdn_attn,
        "compute_causal_conv1d_metadata",
        lambda _loc: (None, None, None),
    )

    result = _builder(monkeypatch, use_spec_decode=True).build(
        0,
        metadata,
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([1, -1], dtype=torch.int32),
    )

    assert result.num_spec_decodes == 1
    assert result.num_accepted_tokens.tolist() == [1]
    assert result.spec_token_indx.numel() == 1
    assert result.non_spec_token_indx.numel() == 2
