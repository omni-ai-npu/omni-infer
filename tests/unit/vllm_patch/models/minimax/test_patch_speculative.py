# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import SimpleNamespace

import pytest

from omni_npu.vllm_patches.core import VLLMPatch
from omni_npu.vllm_patches.patches.models.minimax import patch_speculative


def _speculative_config(method="eagle3", model_type="minimax_m2"):
    hf_text_config = SimpleNamespace(model_type=model_type)
    target_model_config = SimpleNamespace(hf_text_config=hf_text_config)
    return SimpleNamespace(
        method=method,
        target_model_config=target_model_config,
    )


def test_minimax_m2_eagle3_whitelist_error_is_bypassed():
    config = _speculative_config()

    result = patch_speculative._verify_args_with_minimax_m2_allowlist(
        config,
        lambda: (_ for _ in ()).throw(
            ValueError(
                "Eagle3 is only supported for ['llama', 'qwen', 'minicpm', "
                "'gpt_oss'] models. Got "
                "self.target_model_config.hf_text_config.model_type='minimax_m2'"
            )
        ),
    )

    assert result is config


def test_original_validator_success_is_preserved():
    config = _speculative_config()

    result = patch_speculative._verify_args_with_minimax_m2_allowlist(
        config,
        lambda: config,
    )

    assert result is config


def test_non_minimax_m2_whitelist_error_is_not_bypassed():
    config = _speculative_config(model_type="not_minimax")

    with pytest.raises(ValueError, match="Eagle3 is only supported"):
        patch_speculative._verify_args_with_minimax_m2_allowlist(
            config,
            lambda: (_ for _ in ()).throw(
                ValueError("Eagle3 is only supported for ['llama'] models")
            ),
        )


def test_refresh_speculative_config_validator_rebuilds_dataclasses(monkeypatch):
    validator = SimpleNamespace(func=None)
    decorators = SimpleNamespace(model_validators={"_verify_args": validator})
    rebuild_calls = []

    monkeypatch.setattr(
        patch_speculative.SpeculativeConfig,
        "__pydantic_decorators__",
        decorators,
        raising=False,
    )
    monkeypatch.setattr(
        patch_speculative,
        "rebuild_dataclass",
        lambda cls, force: rebuild_calls.append((cls, force)),
    )

    patch_speculative._refresh_speculative_config_validator()

    assert patch_speculative.SpeculativeConfig._verify_args is patch_speculative._verify_args
    assert validator.func is patch_speculative._verify_args
    assert rebuild_calls == [
        (patch_speculative.SpeculativeConfig, True),
        (patch_speculative.VllmConfig, True),
    ]


def test_patch_has_no_apply_override():
    patch_class = patch_speculative.MiniMaxM2SpeculativeConfigValidatorPatch

    assert "apply" not in patch_class.__dict__
    assert patch_class._attr_names_to_apply == []
    assert patch_class.apply.__func__ is VLLMPatch.apply.__func__
