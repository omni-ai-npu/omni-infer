# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from omni_npu.vllm_patches.patches.models.qwen3_5.patch_speculative import (
    SpeculativeConfigPatch,
)


class _Config(SimpleNamespace):
    def update(self, values):
        for key, value in values.items():
            setattr(self, key, value)


def test_hf_config_override_routes_qwen35_to_mtp():
    cfg = _Config(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(mtp_num_hidden_layers=2),
    )

    out = SpeculativeConfigPatch.hf_config_override(cfg)

    assert out is cfg
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.n_predict == 2
    assert cfg.architectures == ["Qwen3_5MTP"]


def test_hf_config_override_routes_qwen35_moe_to_mtp():
    cfg = _Config(
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=SimpleNamespace(mtp_num_hidden_layers=3),
    )

    out = SpeculativeConfigPatch.hf_config_override(cfg)

    assert out is cfg
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.n_predict == 3
    assert cfg.architectures == ["Qwen3_5MoeMTP"]


@pytest.mark.parametrize(
    "model_type,n_predict,expected_type,expected_arch",
    [
        ("deepseek_v3", 1, "deepseek_mtp", "DeepSeekMTPModel"),
        ("deepseek_v32", 2, "deepseek_mtp", "DeepSeekMTPModel"),
        ("pangu_ultra_moe", 3, "pangu_ultra_moe_mtp", "OpenPanguMTPModel"),
        ("ernie4_5_moe", 4, "ernie_mtp", "ErnieMTPModel"),
        ("qwen3_next", 5, "qwen3_next_mtp", "Qwen3NextMTP"),
        ("exaone_moe", 6, "exaone_moe_mtp", "ExaoneMoeMTP"),
        ("longcat_flash", 7, "longcat_flash_mtp", "LongCatFlashMTPModel"),
    ],
)
def test_hf_config_override_routes_model_type_aliases(
    model_type,
    n_predict,
    expected_type,
    expected_arch,
):
    cfg = _Config(
        model_type=model_type,
        architectures=["BaseModel"],
        num_nextn_predict_layers=n_predict,
    )

    out = SpeculativeConfigPatch.hf_config_override(cfg)

    assert out is cfg
    assert cfg.model_type == expected_type
    assert cfg.n_predict == n_predict
    assert cfg.architectures == [expected_arch]


@pytest.mark.parametrize(
    "initial_arch,expected_type,expected_arch",
    [
        ("MiMoForCausalLM", "mimo_mtp", "MiMoMTPModel"),
        ("Glm4MoeForCausalLM", "glm4_moe_mtp", "Glm4MoeMTPModel"),
    ],
)
def test_hf_config_override_routes_architecture_aliases(
    initial_arch,
    expected_type,
    expected_arch,
):
    cfg = _Config(
        model_type="base",
        architectures=[initial_arch],
        num_nextn_predict_layers=8,
    )

    SpeculativeConfigPatch.hf_config_override(cfg)

    assert cfg.model_type == expected_type
    assert cfg.n_predict == 8
    assert cfg.architectures == [expected_arch]


def test_hf_config_override_routes_mistral_large3_eagle():
    cfg = _Config(
        model_type="mistral",
        architectures=["MistralLarge3ForCausalLM"],
    )

    SpeculativeConfigPatch.hf_config_override(cfg)

    assert cfg.architectures == ["EagleMistralLarge3ForCausalLM"]
