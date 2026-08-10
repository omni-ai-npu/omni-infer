"""Tests for the Pangu speculative-config patch."""

from types import SimpleNamespace
from typing import get_args

import pytest

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
