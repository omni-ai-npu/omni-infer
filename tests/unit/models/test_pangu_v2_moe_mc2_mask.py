# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the Pangu V2 MoE MC2 active-token mask."""

from types import SimpleNamespace

import pytest
import torch

from omni_npu.v1.models.pangu import pangu_v2_moe as pangu_moe


def _patch_forward_context(monkeypatch, attn_metadata) -> None:
    monkeypatch.setattr(
        pangu_moe,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=attn_metadata),
    )


@pytest.mark.unit
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_npu_get_mc2_mask_returns_full_true_mask_without_runtime_mask(
    monkeypatch,
    num_tokens,
):
    """A missing runtime mask must use a shape-stable all-True fallback."""
    attn_metadata = SimpleNamespace(
        decode=SimpleNamespace(mc2_mask=None),
        num_decode_tokens=num_tokens,
        num_actual_tokens=num_tokens,
    )
    _patch_forward_context(monkeypatch, attn_metadata)
    tokens = torch.zeros((num_tokens, 8), dtype=torch.int32)

    mask = pangu_moe.npu_get_mc2_mask(tokens)

    assert mask.shape == (num_tokens,)
    assert mask.dtype == torch.bool
    assert mask.device == tokens.device
    assert torch.all(mask)


@pytest.mark.unit
def test_npu_get_mc2_mask_returns_sliced_valid_decode_mask(monkeypatch):
    """A valid decode mask must be preserved and sliced to num_tokens."""
    runtime_mask = torch.tensor([True, True, False, False], dtype=torch.bool)
    attn_metadata = SimpleNamespace(
        decode=SimpleNamespace(mc2_mask=runtime_mask),
        num_decode_tokens=3,
        num_actual_tokens=3,
    )
    _patch_forward_context(monkeypatch, attn_metadata)
    tokens = torch.zeros((3, 8), dtype=torch.int32)

    mask = pangu_moe.npu_get_mc2_mask(tokens)

    assert torch.equal(mask, runtime_mask[:3])
    assert mask.device == tokens.device


@pytest.mark.unit
def test_npu_get_mc2_mask_runtime_and_fake_metadata_match(monkeypatch):
    """Runtime and fake implementations must expose the same tensor metadata."""
    num_tokens = 16
    attn_metadata = SimpleNamespace(
        decode=SimpleNamespace(mc2_mask=None),
        num_decode_tokens=num_tokens,
        num_actual_tokens=num_tokens,
    )
    _patch_forward_context(monkeypatch, attn_metadata)
    tokens = torch.zeros((num_tokens, 8), dtype=torch.int32)

    runtime_mask = pangu_moe.npu_get_mc2_mask(tokens)
    fake_mask = pangu_moe.npu_get_mc2_mask_fake(tokens)

    assert runtime_mask.shape == fake_mask.shape == (num_tokens,)
    assert runtime_mask.dtype == fake_mask.dtype == torch.bool
    assert runtime_mask.device == fake_mask.device == tokens.device


@pytest.mark.unit
@pytest.mark.parametrize(
    "attn_metadata",
    [
        pytest.param(None, id="missing-metadata"),
        pytest.param({}, id="empty-metadata-dict"),
        pytest.param(
            SimpleNamespace(
                decode=None,
                num_decode_tokens=4,
                num_actual_tokens=4,
            ),
            id="missing-decode-metadata",
        ),
        pytest.param(
            SimpleNamespace(
                decode=SimpleNamespace(
                    mc2_mask=torch.tensor([True, False], dtype=torch.bool)
                ),
                num_decode_tokens=4,
                num_actual_tokens=4,
            ),
            id="runtime-mask-too-short",
        ),
        pytest.param(
            SimpleNamespace(
                decode=SimpleNamespace(
                    mc2_mask=torch.tensor([True, True, False, False])
                ),
                num_decode_tokens=2,
                num_actual_tokens=4,
            ),
            id="mixed-prefill-and-decode",
        ),
    ],
)
def test_npu_get_mc2_mask_falls_back_when_runtime_mask_is_unusable(
    monkeypatch,
    attn_metadata,
):
    """Invalid or unavailable runtime metadata must use the safe fallback."""
    _patch_forward_context(monkeypatch, attn_metadata)
    tokens = torch.zeros((4, 8), dtype=torch.int32)

    mask = pangu_moe.npu_get_mc2_mask(tokens)

    assert torch.equal(mask, torch.ones(4, dtype=torch.bool))
