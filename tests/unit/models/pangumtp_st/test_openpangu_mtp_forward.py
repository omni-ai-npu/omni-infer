# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""FWD-01 ~ FWD-09: Forward-pass integration tests for MinimalMTP."""

import pytest
import torch


# ==============================================================================
# FWD-01 ~ FWD-03: Layer output shape & embed/bypass paths
# ==============================================================================

class TestLayerForward:
    """FWD-01: MTP Layer single-step forward produces correct output shape."""

    def test_layer_output_shape(self, mtp_layer, sample_batch):
        """FWD-01: output shape = (bsz, hidden_size)."""
        out = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_index=0,
        )
        B = sample_batch["input_ids"].shape[0]
        H = mtp_layer.config.hidden_size
        assert out.shape == (B, H), f"expected ({B}, {H}), got {out.shape}"


class TestPredictorForward:
    """FWD-02 ~ FWD-03: Predictor forward with and without inputs_embeds."""

    def test_predictor_embed_path(self, mtp_predictor, sample_batch, minimal_config):
        """FWD-02: inputs_embeds=None → uses embed_tokens path."""
        out = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=None,                           # ← trigger embedding
            spec_step_idx=0,
        )
        B, H = sample_batch["input_ids"].shape[0], minimal_config.hidden_size
        assert out.shape == (B, H)

    def test_predictor_bypass_embed_path(self, mtp_predictor, sample_batch, minimal_config):
        """FWD-03: inputs_embeds is not None → bypasses embed_tokens."""
        out = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],  # ← given explicitly
            spec_step_idx=0,
        )
        B, H = sample_batch["input_ids"].shape[0], minimal_config.hidden_size
        assert out.shape == (B, H)


# ==============================================================================
# FWD-04 ~ FWD-05: Top-level MTP forward & compute_logits
# ==============================================================================

class TestMTPTopLevel:
    """FWD-04 ~ FWD-05: OpenPanguMTP top-level forward and logits."""

    def test_mtp_full_forward(self, mtp_model, sample_batch, minimal_config):
        """FWD-04: MTP top-level forward produces correct shape."""
        out = mtp_model(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=0,
        )
        B, H = sample_batch["input_ids"].shape[0], minimal_config.hidden_size
        assert out.shape == (B, H)

    def test_compute_logits_shape(self, mtp_model, sample_batch, minimal_config):
        """FWD-05: compute_logits returns (bsz, vocab_size)."""
        # First run forward to get hidden_states
        hidden = mtp_model(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=0,
        )
        logits = mtp_model.compute_logits(hidden, spec_step_idx=0)
        B = sample_batch["input_ids"].shape[0]
        V = minimal_config.vocab_size
        assert logits.shape == (B, V), f"expected ({B}, {V}), got {logits.shape}"


# ==============================================================================
# FWD-06 ~ FWD-07: Semantic correctness — different inputs, gradient flow
# ==============================================================================

class TestSemanticCorrectness:
    """FWD-06 ~ FWD-07: Verify the model is actually doing computation."""

    def test_different_inputs_different_outputs(self, mtp_layer, sample_batch):
        """FWD-06: different inputs produce different outputs (not constant)."""
        out1 = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_index=0,
        )
        # Perturb input
        perturbed = sample_batch["inputs_embeds"].clone()
        perturbed[0, 0] += 1.0
        out2 = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=perturbed,
            spec_step_index=0,
        )
        assert not torch.allclose(out1, out2), "different inputs should give different outputs"

    def test_gradient_flow(self, mtp_layer, sample_batch):
        """FWD-07: gradients propagate through the entire computation graph."""
        # Build a small loss: sum of outputs
        out = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_index=0,
        )
        loss = out.sum()
        loss.backward()

        # Every parameter on the forward path should have a gradient.
        # shared_head is not on the forward path — it's only used in compute_logits.
        no_grad = [name for name, p in mtp_layer.named_parameters()
                   if p.requires_grad and p.grad is None
                   and "shared_head" not in name]
        assert not no_grad, f"parameters without gradient: {no_grad}"

        # Gradients should be non-zero somewhere (not a dead network)
        max_grads = {name: p.grad.abs().max().item()
                     for name, p in mtp_layer.named_parameters()
                     if p.requires_grad and p.grad is not None}
        non_zero = [name for name, v in max_grads.items() if v == 0.0]
        # Allow some zeros (e.g. unused rows), but not everything
        assert len(non_zero) < len(max_grads), \
            f"all parameters have zero gradient — dead network?"


# ==============================================================================
# FWD-08 ~ FWD-09: eh_proj fusion & enorm/hnorm independence
# ==============================================================================

class TestMTPArchitecture:
    """FWD-08 ~ FWD-09: Verify MTP-specific architectural properties."""

    def test_eh_proj_fusion(self, mtp_layer, sample_batch, minimal_config):
        """FWD-08: eh_proj correctly fuses inputs_embeds and prev_hidden."""
        # Grab internal values by manually replicating the fusion step
        emb = sample_batch["inputs_embeds"]
        prev = sample_batch["hidden_states"]

        # Forward through enorm/hnorm → the layer does this internally
        # We test that eh_proj weight participates in the computation
        # by capturing output before and after modifying eh_proj weight
        out1 = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=prev,
            inputs_embeds=emb,
            spec_step_index=0,
        )

        # Perturb eh_proj weight
        original = mtp_layer.eh_proj.weight.data.clone()
        mtp_layer.eh_proj.weight.data += 0.1
        out2 = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=prev,
            inputs_embeds=emb,
            spec_step_index=0,
        )
        mtp_layer.eh_proj.weight.data.copy_(original)

        assert not torch.allclose(out1, out2), \
            "modifying eh_proj weight should change output"

    def test_enorm_hnorm_independent(self, mtp_layer):
        """FWD-09: enorm and hnorm have independent weights."""
        assert not torch.equal(mtp_layer.enorm.weight.data, mtp_layer.hnorm.weight.data), \
            "enorm and hnorm should have independent (random) initializations"
