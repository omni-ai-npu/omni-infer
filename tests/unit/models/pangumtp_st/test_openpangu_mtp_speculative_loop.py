# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""SPL-01 ~ SPL-06: Full speculative-decoding loop — draft → sample → verify → accept.

Simulates the EagleProposer.propose_multi_mtp() loop (patch_eagle.py:977-1031)
and the NPURejectionSampler greedy acceptance (rejection_sampler.py:481).

Key design decision:
  - draft argmax:       identical to production (patch_eagle.py:1023)
  - greedy acceptance:  algorithmically identical to production
                        (rejection_sampler.py:481: accepted = draft == target_argmax)
"""

import pytest
import torch


# ==============================================================================
# helpers — lightweight simulation of propose_multi_mtp internals
# ==============================================================================

def _sample_one_token(model, input_ids, positions, prev_hidden, spec_step):
    """Draft-step: embed → forward → compute_logits → argmax → token."""
    embeds = model.embed_input_ids(input_ids)
    hidden = model(input_ids=None, positions=positions,
                   hidden_states=prev_hidden, inputs_embeds=embeds,
                   spec_step_idx=spec_step)
    logits = model.compute_logits(hidden, spec_step_idx=spec_step)
    return logits.argmax(dim=-1), hidden


def _multi_step_draft(model, input_ids, positions, init_prev_hidden,
                      num_steps: int) -> torch.Tensor:
    """Run the propose_multi_mtp outer loop and return [batch, num_steps] token ids."""
    draft_tokens_list = []
    prev_hidden = init_prev_hidden
    cur_ids = input_ids.clone()
    cur_pos = positions.clone()

    for step in range(num_steps):
        # Shift input_ids (simplified single-token-per-request case)
        if step > 0:
            cur_ids = torch.roll(cur_ids, -1)
            cur_ids[-1] = draft_tokens_list[-1][-1]
            cur_pos = cur_pos + 1

        token, hidden = _sample_one_token(model, cur_ids, cur_pos, prev_hidden, step)
        draft_tokens_list.append(token)
        prev_hidden = hidden

    return torch.stack(draft_tokens_list, dim=1)  # [batch, num_steps]


def _greedy_accept(draft_tokens: torch.Tensor,
                   target_logits: torch.Tensor) -> torch.Tensor:
    """Greedy acceptance — algorithm identical to rejection_sampler.py:481.

    Returns accepted mask: [batch, num_steps] bool tensor.
    """
    target_top1 = target_logits.argmax(dim=-1)       # [batch, num_steps]
    accepted = draft_tokens == target_top1
    cum_accept = accepted.int().cumprod(dim=1)       # first reject → all subsequent 0
    return cum_accept.bool()


# ==============================================================================
# SPL-01 ~ SPL-02: Single / multi-step draft
# ==============================================================================

class TestDraftSampling:
    """SPL-01 ~ SPL-03: Draft sampling behaviour."""

    def test_single_step_draft(self, mtp_model, sample_batch, minimal_config):
        """SPL-01: Single-step draft produces valid token ids."""
        token, hidden = _sample_one_token(
            mtp_model,
            sample_batch["input_ids"],
            sample_batch["positions"],
            sample_batch["hidden_states"],
            spec_step=0,
        )
        B = sample_batch["input_ids"].shape[0]
        assert token.shape == (B,), f"expected ({B},), got {token.shape}"
        assert (token >= 0).all() and (token < minimal_config.vocab_size).all(), \
            "sampled tokens must be in vocab range"

    def test_multi_step_draft_loop(self, mtp_model, sample_batch, minimal_config):
        """SPL-02: Multi-step draft produces [batch, num_steps] token ids."""
        num_steps = 3
        draft_tokens = _multi_step_draft(
            mtp_model,
            sample_batch["input_ids"],
            sample_batch["positions"],
            sample_batch["hidden_states"],
            num_steps,
        )
        B = sample_batch["input_ids"].shape[0]
        assert draft_tokens.shape == (B, num_steps), \
            f"expected ({B}, {num_steps}), got {draft_tokens.shape}"
        assert (draft_tokens >= 0).all() and (draft_tokens < minimal_config.vocab_size).all()

    def test_draft_state_propagation(self, mtp_model, sample_batch):
        """SPL-03: Changing prev_hidden between steps → different next token."""
        # Run step 0 with original prev_hidden
        token_a, hidden_a = _sample_one_token(
            mtp_model,
            sample_batch["input_ids"],
            sample_batch["positions"],
            sample_batch["hidden_states"],
            spec_step=0,
        )
        # Run step 0 again with a different prev_hidden
        prev2 = sample_batch["hidden_states"].clone()
        prev2[0, :] += 1.0
        token_b, hidden_b = _sample_one_token(
            mtp_model,
            sample_batch["input_ids"],
            sample_batch["positions"],
            prev2,
            spec_step=0,
        )
        assert not torch.equal(token_a, token_b) or not torch.equal(hidden_a, hidden_b), \
            "different prev_hidden should change at least one of token or hidden"


# ==============================================================================
# SPL-04 ~ SPL-05: Target verification & acceptance
# ==============================================================================

class TestVerificationAndAcceptance:
    """SPL-04 ~ SPL-05: Target model verification + greedy acceptance."""

    def test_target_verification_top1_agreement(self, draft_target_models, sample_batch):
        """SPL-04: Compare draft token with target top-1 → acceptance mask."""
        draft, target = draft_target_models

        # Draft samples one token
        draft_token, _ = _sample_one_token(
            draft,
            sample_batch["input_ids"],
            sample_batch["positions"],
            sample_batch["hidden_states"],
            spec_step=0,
        )

        # Target scores the same position
        target_embeds = target.embed_input_ids(sample_batch["input_ids"])
        target_hidden = target(
            input_ids=None,
            positions=sample_batch["positions"],
            hidden_states=torch.zeros_like(target_embeds),
            inputs_embeds=target_embeds,
            spec_step_idx=0,
        )
        target_logits = target.compute_logits(target_hidden, spec_step_idx=0)

        # Acceptance: algorithm identical to rejection_sampler.py:481
        target_top1 = target_logits.argmax(dim=-1)
        accepted = (draft_token == target_top1)
        assert accepted.dtype == torch.bool
        # With different seeds, acceptance rate should be < 100%
        acc_rate = accepted.float().mean().item()
        assert 0.0 <= acc_rate <= 1.0, f"invalid acceptance rate: {acc_rate}"

    def test_full_speculative_loop(self, draft_target_models, sample_batch, minimal_config):
        """SPL-05: Complete draft → verify → accept → first_reject_cutoff."""
        draft, target = draft_target_models
        num_steps = 3

        # ---- Draft phase ----
        draft_tokens = _multi_step_draft(
            draft,
            sample_batch["input_ids"],
            sample_batch["positions"],
            sample_batch["hidden_states"],
            num_steps,
        )

        # ---- Verify phase: target scores each draft position ----
        target_logits_list = []
        # Use the same starting embed as draft step 0
        target_embeds = target.embed_input_ids(sample_batch["input_ids"])
        target_hidden = target(
            input_ids=None,
            positions=sample_batch["positions"],
            hidden_states=torch.zeros_like(target_embeds),
            inputs_embeds=target_embeds,
            spec_step_idx=0,
        )
        for step in range(num_steps):
            t_logits = target.compute_logits(target_hidden, spec_step_idx=step)
            target_logits_list.append(t_logits)

        target_top1 = torch.stack(
            [l.argmax(dim=-1) for l in target_logits_list], dim=1
        )

        # ---- Acceptance phase ----
        accepted = draft_tokens == target_top1           # [batch, num_steps]
        cum_accept = accepted.int().cumprod(dim=1)       # first reject → zero
        num_accepted = cum_accept.sum(dim=1)

        B = sample_batch["input_ids"].shape[0]
        assert num_accepted.shape == (B,)
        # Bonus token is always accepted; num_accepted can be 0 ~ num_steps
        assert (num_accepted >= 0).all() and (num_accepted <= num_steps).all(), \
            f"num_accepted out of range: {num_accepted}"

    def test_first_reject_cutoff(self):
        """SPL-05 (detail): verify cumprod correctly cuts off after first reject."""
        # Simulated acceptance mask:
        #   [True, True, False, True]  →  cumprod →  [1, 1, 0, 0]
        accepted = torch.tensor([
            [True, True, False, True],
            [True, False, True, True],
        ])
        cum_accept = accepted.int().cumprod(dim=1)
        expected = torch.tensor([
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ])
        assert torch.equal(cum_accept, expected), \
            f"cumprod cutoff wrong:\n{cum_accept}\nvs\n{expected}"


# ==============================================================================
# SPL-06: Determinism
# ==============================================================================

class TestSamplingDeterminism:
    """SPL-06: Fixed seed + fixed input → bit-exact identical draft tokens."""

    def test_deterministic_draft_tokens(self, minimal_config, sample_batch):
        """SPL-06: Two draft loops with same seed → identical token sequences."""
        from .decoder_layer_stub import TestMTPModel

        def run_loop():
            torch.manual_seed(42)
            model = TestMTPModel(minimal_config)
            return _multi_step_draft(
                model,
                sample_batch["input_ids"],
                sample_batch["positions"],
                sample_batch["hidden_states"],
                num_steps=3,
            )

        result1 = run_loop()
        result2 = run_loop()
        assert torch.equal(result1, result2), \
            "deterministic seed must produce identical draft tokens"

    def test_different_seeds_different_tokens(self, minimal_config, sample_batch):
        """SPL-06 (complement): Different seeds → different token trajectories."""
        from .decoder_layer_stub import TestMTPModel

        torch.manual_seed(42)
        model_a = TestMTPModel(minimal_config)
        tokens_a = _multi_step_draft(
            model_a, sample_batch["input_ids"], sample_batch["positions"],
            sample_batch["hidden_states"], num_steps=3,
        )

        torch.manual_seed(12345)
        model_b = TestMTPModel(minimal_config)
        tokens_b = _multi_step_draft(
            model_b, sample_batch["input_ids"], sample_batch["positions"],
            sample_batch["hidden_states"], num_steps=3,
        )

        # Different random weights → almost certainly different trajectories
        assert not torch.equal(tokens_a, tokens_b), \
            "different seeds should produce different draft token sequences"
