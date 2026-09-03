"""Tests for the Pangu speculative-config patch."""

import inspect
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

import numpy as np
import pytest
import torch

from omni_npu.vllm_patches.usefull_patch.common.patch_eagle import EagleProposerPatch
from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_hybrid import patch_speculative as patch_mod


class HFConfig:
    def __init__(self, model_type, architectures=(), num_nextn_predict_layers=2):
        self.model_type = model_type
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.architectures = list(architectures)

    def update(self, values):
        self.__dict__.update(values)


@pytest.mark.parametrize(
    "model_type,architectures,patch_dirs,expected_type,expected_arch",
    [
        (
            "openpangu_v2",
            ["PanguUltraMoEForCausalLM"],
            "",
            "openpangu_mtp",
            "OpenPanguMTPModel",
        ),
        (
            "openpangu_v2",
            ["OpenPanguV2ForCausalLM"],
            "",
            "mtp",
            "OpenPanguV2MTPModel",
        ),
        ("openpangu_v2_vl_moe", [], "", "openpangu_mtp", "OpenPanguMTPModel"),
        (
            "openpangu_v2_omni_moe",
            [],
            "pangu_v2_moe",
            "mtp",
            "OpenPanguV2MTPModel",
        ),
    ],
)
def test_pangu_speculative_model_mapping(
    monkeypatch,
    model_type,
    architectures,
    patch_dirs,
    expected_type,
    expected_arch,
):
    monkeypatch.setattr(
        patch_mod.envs, "OMNI_VLLM_PATCHES_DIR", patch_dirs, raising=False
    )
    config = HFConfig(
        model_type,
        architectures=architectures,
        num_nextn_predict_layers=3,
    )

    result = patch_mod.PanguV2MoeSpeculativeConfigPatch.hf_config_override(config)

    assert result is config
    assert config.model_type == expected_type
    assert config.n_predict == 3
    assert config.architectures == [expected_arch]


def test_non_pangu_speculative_config_delegates_to_upstream(monkeypatch):
    sentinel = SimpleNamespace(source="upstream")
    calls = []

    def original(config):
        calls.append(config)
        return sentinel

    monkeypatch.setattr(patch_mod, "_origin_hf_config_override", original)
    monkeypatch.setattr(
        patch_mod.envs, "OMNI_VLLM_PATCHES_DIR", "pangu_v2_moe", raising=False
    )
    config = HFConfig("other_model")

    assert (
        patch_mod.PanguV2MoeSpeculativeConfigPatch.hf_config_override(config)
        is sentinel
    )
    assert calls == [config]


def test_pangu_v2_moe_model_type_falls_through(monkeypatch):
    sentinel = SimpleNamespace(source="upstream")

    def _upstream(_config):
        return sentinel

    monkeypatch.setattr(patch_mod, "_origin_hf_config_override", _upstream)
    config = HFConfig("pangu_v2_moe")
    assert (
        patch_mod.PanguV2MoeSpeculativeConfigPatch.hf_config_override(config)
        is sentinel
    )


def test_mtp_model_types_include_both_pangu_drafters():
    model_types = set(get_args(patch_mod.SpeculativePatch.MTPModelTypes))
    assert {"openpangu_mtp", "pangu_ultra_moe_mtp", "mtp"} <= model_types


def test_speculative_patch_registration_targets():
    assert patch_mod.SpeculativePatch._target is patch_mod.speculative
    assert patch_mod.PanguV2MoeSpeculativeConfigPatch._target is (
        patch_mod.SpeculativeConfig
    )


def test_prepare_inputs_padded_carries_cpu_metadata_shadows():
    proposer = SimpleNamespace(arange=torch.arange(16))
    seq_lens_cpu = torch.tensor([11, 22], dtype=torch.int32)
    num_computed_tokens_cpu = torch.tensor([10, 20], dtype=torch.int32)
    seq_lens_upper_bound = torch.tensor([12, 24], dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([11, 22], dtype=torch.int32),
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_upper_bound,
        num_reqs=2,
        num_actual_tokens=4,
        max_seq_len=24,
        block_table_tensor=torch.zeros(2, 2, dtype=torch.int32),
        slot_mapping=torch.arange(4),
        dcp_local_seq_lens=None,
    )
    spec_decode = SimpleNamespace(
        cu_num_draft_tokens=torch.tensor([1, 2], dtype=torch.int32)
    )
    captured = {}

    def capture_metadata(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    with patch(
        "omni_npu.vllm_patches.usefull_patch.common.patch_eagle.CommonAttentionMetadata",
        side_effect=capture_metadata,
    ):
        result, _, _ = EagleProposerPatch.prepare_inputs_padded(
            proposer,
            common,
            spec_decode,
            valid_sampled_tokens_count=torch.tensor([1, 1], dtype=torch.int32),
        )

    assert result._seq_lens_cpu is seq_lens_cpu
    assert result._num_computed_tokens_cpu is num_computed_tokens_cpu
    assert result.seq_lens_cpu_upper_bound is seq_lens_upper_bound
    assert captured["query_start_loc_cpu"] is common.query_start_loc_cpu


def test_eagle_load_model_sets_image_token_index_for_omni_v2(monkeypatch):
    from omni_npu.vllm_patches.usefull_patch.common import patch_eagle as eagle_mod

    proposer = eagle_mod.EagleProposerPatch.__new__(eagle_mod.EagleProposerPatch)
    proposer.vllm_config = object()
    proposer.supports_mm_inputs = False
    proposer.parallel_drafting = False
    proposer.pass_hidden_states_to_model = False
    proposer.parallel_drafting_hidden_state_tensor = None

    def _get_model():
        return SimpleNamespace(config=SimpleNamespace())

    def _model_name(_model):
        return "OpenPanguOmniV2ForConditionalGeneration"

    def _noop(*_args, **_kwargs):
        return None

    def _no_layers(*_args, **_kwargs):
        return {}

    def _is_multimodal(_model):
        return True

    proposer._get_model = _get_model
    proposer.get_model_name = _model_name
    proposer._maybe_share_embeddings = _noop
    proposer._maybe_share_lm_head = _noop
    monkeypatch.setattr(eagle_mod, "get_layers_from_vllm_config", _no_layers)
    monkeypatch.setattr(eagle_mod, "supports_multimodal", _is_multimodal)

    target = SimpleNamespace(config=SimpleNamespace(image_token_id=77))

    def _get_language_model():
        return target

    target.get_language_model = _get_language_model
    eagle_mod.EagleProposerPatch.load_model(proposer, target)

    assert proposer.model.config.image_token_index == 77


def test_prepare_next_token_ids_padded_counts_are_int32():
    proposer = SimpleNamespace(
        backup_next_token_ids=SimpleNamespace(
            np=np.zeros(4, dtype=np.int32),
            gpu=torch.tensor([70, 71, 72, 73], dtype=torch.int32),
            copy_to_gpu=lambda _n: None,
        )
    )
    batch = SimpleNamespace(
        num_reqs=3,
        req_ids=["a", "b", "c"],
        num_tokens_no_spec=[5, 6, 7],
        vocab_size=100,
    )
    requests = {
        rid: SimpleNamespace(get_token_id=lambda _i, v=v: v)
        for rid, v in (("a", 11), ("b", 22), ("c", 33))
    }
    sampled = torch.tensor([[5, 6, -1], [7, -1, -1], [8, 9, 10]], dtype=torch.int64)
    discard = torch.tensor([False, False, True])

    next_ids, counts = EagleProposerPatch.prepare_next_token_ids_padded(
        proposer, sampled, requests, batch, discard
    )

    assert counts.dtype == torch.int32
    assert torch.equal(counts, torch.tensor([2, 1, 0], dtype=torch.int32))
    # discarded request falls back to its backup token
    assert torch.equal(next_ids, torch.tensor([6, 7, 72]))


def test_propose_early_exit_returns_int64_draft_ids():
    source = inspect.getsource(EagleProposerPatch.propose)
    early_exit, _, rest = source.partition("if not use_multi_mtp:")
    assert "draft_token_ids.int()" not in early_exit
    assert "draft_token_ids = draft_token_ids.int()" in rest

