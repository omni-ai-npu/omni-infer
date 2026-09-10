# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""EDG-01 ~ EDG-07: Boundary condition tests."""

import pytest
import torch
from .decoder_layer_stub import (
    TestMTPConfig,
    TestMTPPredictor,
)


def _make_predictor(**overrides) -> TestMTPPredictor:
    """Create a predictor with custom config values."""
    cfg = TestMTPConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if "head_dim" not in overrides:
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    return TestMTPPredictor(cfg)


class TestBatchSizes:
    """EDG-01 ~ EDG-02: Single and large batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 32])
    def test_batch_size(self, batch_size):
        pred = _make_predictor()
        B, H = batch_size, TestMTPConfig.hidden_size

        out = pred(
            input_ids=torch.zeros(B, dtype=torch.int64),
            positions=torch.zeros(B, dtype=torch.int64),
            previous_hidden_states=torch.arange(B * H, dtype=torch.float32).view(B, H) * 0.01,
            inputs_embeds=None,
            spec_step_idx=0,
        )
        assert out.shape == (B, H)


class TestMTPLayerCounts:
    """EDG-03 ~ EDG-04: Edge numbers of MTP layers."""

    @pytest.mark.parametrize("num_mtp", [1, 4])
    def test_mtp_layer_count(self, num_mtp):
        pred = _make_predictor(num_nextn_predict_layers=num_mtp)
        B, H = 2, TestMTPConfig.hidden_size

        out = pred(
            input_ids=torch.zeros(B, dtype=torch.int64),
            positions=torch.zeros(B, dtype=torch.int64),
            previous_hidden_states=torch.arange(B * H, dtype=torch.float32).view(B, H) * 0.01,
            inputs_embeds=None,
            spec_step_idx=0,
        )
        assert out.shape == (B, H)

        if num_mtp > 1:
            out2 = pred(
                input_ids=torch.zeros(B, dtype=torch.int64),
                positions=torch.zeros(B, dtype=torch.int64),
                previous_hidden_states=torch.arange(B * H, dtype=torch.float32).view(B, H) * 0.01,
                inputs_embeds=None,
                spec_step_idx=num_mtp + 1,  # wrap
            )
            assert out2.shape == (B, H)


class TestVocabSizes:
    """EDG-05 ~ EDG-06: Very small and moderate vocab sizes."""

    @pytest.mark.parametrize("vocab_size", [16, 1024])
    def test_vocab_size(self, vocab_size):
        pred = _make_predictor(vocab_size=vocab_size)
        B, H = 2, TestMTPConfig.hidden_size

        hidden = pred(
            input_ids=torch.zeros(B, dtype=torch.int64),
            positions=torch.zeros(B, dtype=torch.int64),
            previous_hidden_states=torch.arange(B * H, dtype=torch.float32).view(B, H) * 0.01,
            inputs_embeds=None,
            spec_step_idx=0,
        )
        logits = pred.compute_logits(hidden, spec_step_idx=0)
        assert logits.shape == (B, vocab_size)


class TestHiddenSizes:
    """EDG-07: Tiny hidden size."""

    def test_small_hidden_size(self):
        pred = _make_predictor(
            hidden_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            intermediate_size=64,
        )
        B, H = 2, 32

        out = pred(
            input_ids=torch.zeros(B, dtype=torch.int64),
            positions=torch.zeros(B, dtype=torch.int64),
            previous_hidden_states=torch.arange(B * H, dtype=torch.float32).view(B, H) * 0.01,
            inputs_embeds=None,
            spec_step_idx=0,
        )
        assert out.shape == (B, H)
