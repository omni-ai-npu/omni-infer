"""Tests for the Pangu speculative-config patch."""

from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

import pytest
import torch

from omni_npu.vllm_patches.usefull_patch.patch_eagle import EagleProposerPatch
from omni_npu.vllm_patches.usefull_patch import patch_speculative as patch_mod


class HFConfig:
    def __init__(self, model_type, num_nextn_predict_layers=2):
        self.model_type = model_type
        self.num_nextn_predict_layers = num_nextn_predict_layers

    def update(self, values):
        self.__dict__.update(values)


@pytest.mark.parametrize(
    "model_type,patch_dirs,expected_type,expected_arch",
    [
        ("openpangu_v2", "", "openpangu_mtp", "OpenPanguMTPModel"),
        ("openpangu_v2_vl_moe", "", "openpangu_mtp", "OpenPanguMTPModel"),
        ("openpangu_v2_omni_moe", "pangu_v2_moe", "mtp", "PanguV2MTPModel"),
        ("pangu_v2_moe", "", "mtp", "PanguV2MTPModel"),
    ],
)
def test_pangu_speculative_model_mapping(
    monkeypatch, model_type, patch_dirs, expected_type, expected_arch
):
    monkeypatch.setattr(
        patch_mod.envs, "OMNI_VLLM_PATCHES_DIR", patch_dirs, raising=False
    )
    config = HFConfig(model_type, num_nextn_predict_layers=3)

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
        "omni_npu.vllm_patches.usefull_patch.patch_eagle.CommonAttentionMetadata",
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
