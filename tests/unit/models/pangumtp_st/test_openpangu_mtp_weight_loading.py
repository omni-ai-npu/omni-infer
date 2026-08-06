# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""WLD-01 ~ WLD-05: Weight-loading & determinism tests.

Verifies that weights actually participate in computation and that
deterministic seeding produces reproducible results.
"""

import pytest
import torch


class TestWeightAffectsOutput:
    """WLD-01, WLD-03, WLD-04: Weight changes → output changes."""

    def test_weight_change_affects_output(self, mtp_layer, sample_batch):
        """WLD-01: Modifying any weight changes the output."""
        def forward():
            return mtp_layer(
                input_ids=sample_batch["input_ids"],
                positions=sample_batch["positions"],
                previous_hidden_states=sample_batch["hidden_states"],
                inputs_embeds=sample_batch["inputs_embeds"],
                spec_step_index=0,
            )

        out_before = forward()

        # Perturb one weight
        original = mtp_layer.eh_proj.weight.data.clone()
        mtp_layer.eh_proj.weight.data += 0.5
        out_after = forward()
        mtp_layer.eh_proj.weight.data.copy_(original)

        assert not torch.allclose(out_before, out_after), \
            "weight change must affect output"

    def test_eh_proj_weight_affects_fusion(self, mtp_layer, sample_batch):
        """WLD-03: eh_proj weight directly affects the fusion projection."""
        out1 = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_index=0,
        )

        # Scale eh_proj weight
        with torch.no_grad():
            mtp_layer.eh_proj.weight.data *= 2.0
        out2 = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_index=0,
        )
        # Restore
        with torch.no_grad():
            mtp_layer.eh_proj.weight.data /= 2.0

        assert not torch.allclose(out1, out2), \
            "scaling eh_proj weight should scale fusion result"

    def test_enorm_weight_affects_normalization(self, mtp_layer, sample_batch):
        """WLD-04: enorm weight participates in input embedding normalization."""
        out1 = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_index=0,
        )

        with torch.no_grad():
            mtp_layer.enorm.weight.data *= 3.0
        out2 = mtp_layer(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_index=0,
        )
        with torch.no_grad():
            mtp_layer.enorm.weight.data /= 3.0

        assert not torch.allclose(out1, out2), \
            "enorm weight should affect normalization output"

    def test_embed_tokens_weight_affects_embedding(self, mtp_predictor, sample_batch):
        """WLD-05: VocabParallelEmbedding weight affects embedded representation."""
        # Use the embed path (inputs_embeds=None) so embed_tokens is exercised
        out1 = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=None,
            spec_step_idx=0,
        )

        with torch.no_grad():
            mtp_predictor.embed_tokens.weight.data += 0.5
        out2 = mtp_predictor(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=None,
            spec_step_idx=0,
        )
        with torch.no_grad():
            mtp_predictor.embed_tokens.weight.data -= 0.5

        assert not torch.allclose(out1, out2), \
            "embed_tokens weight change must affect output when inputs_embeds is None"


class TestDeterminism:
    """WLD-02: Seed determinism — same seed → identical models → identical outputs."""

    def test_seed_determinism(self, minimal_config, sample_batch):
        """WLD-02: torch.manual_seed(42) twice → identical weights → identical output."""
        from .decoder_layer_stub import TestMTPModel

        def make_and_run():
            torch.manual_seed(42)
            model = TestMTPModel(minimal_config)
            return model(
                input_ids=sample_batch["input_ids"],
                positions=sample_batch["positions"],
                hidden_states=sample_batch["hidden_states"],
                inputs_embeds=sample_batch["inputs_embeds"],
                spec_step_idx=0,
            )

        out1 = make_and_run()
        out2 = make_and_run()
        assert torch.equal(out1, out2), \
            "same seed must produce bit-exact identical forward outputs"


class TestMTPStructure:
    """Verify TestMTPModel has the correct architectural structure."""

    def test_has_required_components(self, mtp_layer):
        """MTP Layer must have enorm, hnorm, eh_proj, shared_head, mtp_block."""
        assert hasattr(mtp_layer, "enorm")
        assert hasattr(mtp_layer, "hnorm")
        assert hasattr(mtp_layer, "eh_proj")
        assert hasattr(mtp_layer, "shared_head")
        assert hasattr(mtp_layer, "mtp_block")

    def test_shared_head_has_norm_and_head(self, mtp_layer):
        """shared_head must have norm and head sub-modules."""
        assert hasattr(mtp_layer.shared_head, "norm")
        assert hasattr(mtp_layer.shared_head, "head")

    def test_predictor_has_embed_and_layers(self, mtp_predictor):
        """Predictor must have embed_tokens and layers ModuleDict."""
        assert hasattr(mtp_predictor, "embed_tokens")
        assert hasattr(mtp_predictor, "layers")
        assert len(mtp_predictor.layers) == 2  # num_nextn_predict_layers=2
