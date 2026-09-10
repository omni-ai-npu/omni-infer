# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MST-01 ~ MST-06: Multi-step speculative prediction tests.

Validates the spec_step_idx modulo cycling logic and state propagation
through the real OpenPanguMultiTokenPredictor code path.
"""

import pytest
import torch


class TestSpecStepIdxModulo:
    """MST-01 ~ MST-02: spec_step_idx % num_mtp_layers cycling."""

    def test_modulo_zero(self, mtp_predictor, sample_batch):
        """MST-01: spec_step_idx=0, num_mtp_layers=2 → uses layer at mtp_start_layer_idx."""
        # With num_hidden_layers=2, num_mtp_layers=2 → mtp_start_layer_idx=2
        # spec_step_idx=0 → current_step_idx=0 → layers["2"]
        out = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=0,
        )
        assert out.shape[1] == sample_batch["hidden_states"].shape[1]

    def test_modulo_wrap(self, mtp_predictor, sample_batch):
        """MST-02: spec_step_idx=2, num_mtp_layers=2 → wraps back to layer 0."""
        # Step 0 → layer["2"], Step 1 → layer["3"], Step 2 → wraps → layer["2"]
        out = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=2,  # wraps to 0
        )
        assert out.shape[1] == sample_batch["hidden_states"].shape[1]

    def test_different_layers_different_outputs(self, mtp_predictor, sample_batch):
        """MST-03: Different spec_step_idx values use different layers → different outputs."""
        out0 = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=0,
        )
        out1 = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=1,
        )
        # Different layers have different weights → different outputs
        assert not torch.allclose(out0, out1), \
            "different MTP layers should produce different outputs"

    def test_same_layer_same_output(self, mtp_predictor, sample_batch):
        """MST-04: spec_step_idx=0 and spec_step_idx=2 (same layer) → same output."""
        out0 = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=0,
        )
        out2 = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=2,  # 2 % 2 = 0 → same layer
        )
        assert torch.allclose(out0, out2), \
            "same layer + same input → should produce same output"


class TestStatePropagation:
    """MST-05 ~ MST-06: previous_hidden_states flow between steps."""

    def test_prev_hidden_affects_output(self, mtp_predictor, sample_batch):
        """MST-05: Different previous_hidden_states → different outputs."""
        out1 = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=0,
        )
        # Perturb prev_hidden
        prev2 = sample_batch["hidden_states"].clone()
        prev2[0, 0] += 1.0
        out2 = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=prev2,
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=0,
        )
        assert not torch.allclose(out1, out2), \
            "different prev_hidden should change output via hnorm → eh_proj fusion"

    def test_sequential_token_generation_pattern(self, mtp_model, sample_batch, minimal_config):
        """MST-06: Simulate multi-step spec decode — state passes between steps."""
        num_steps = 3
        prev_hidden = sample_batch["hidden_states"]
        positions = sample_batch["positions"]
        prev_tokens = sample_batch["input_ids"]

        outputs = []
        for step in range(num_steps):
            # Simulate input_ids shift (as in propose_multi_mtp)
            shifted = torch.roll(prev_tokens, -1)
            shifted[-1] = shifted[-1]  # simplified: keep last same

            embeds = mtp_model.embed_input_ids(shifted)
            hidden = mtp_model(
                input_ids=shifted,
                positions=positions + step,
                hidden_states=prev_hidden,
                inputs_embeds=embeds,
                spec_step_idx=step,
            )
            outputs.append(hidden)
            prev_hidden = hidden
            prev_tokens = shifted

        # All steps should produce valid shapes
        for i, out in enumerate(outputs):
            assert out.shape == (sample_batch["input_ids"].shape[0], minimal_config.hidden_size), \
                f"step {i}: expected ({sample_batch['input_ids'].shape[0]}, {minimal_config.hidden_size}), got {out.shape}"

        # Steps with same spec_step_idx mod should use same layer
        # (step 0 → layer 0, step 2 → layer 0 again for num_mtp_layers=2)
        # Different inputs → different outputs even with same layer
        assert not torch.allclose(outputs[0], outputs[2]), \
            "same layer but different positions/prev_hidden → should differ"
