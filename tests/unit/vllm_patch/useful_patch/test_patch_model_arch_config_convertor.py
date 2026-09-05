from types import SimpleNamespace

import pytest

from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_base import (
    patch_model_arch_config_convertor as patch_mod,
)


def _convertor(model_type=None, kv_lora_rank=None):
    instance = object.__new__(patch_mod.PanguV2MoeModelArchConfigConvertorPatch)
    values = {}
    if model_type is not None:
        values["model_type"] = model_type
    if kv_lora_rank is not None:
        values["kv_lora_rank"] = kv_lora_rank
    instance.hf_text_config = SimpleNamespace(**values)
    return instance


@pytest.mark.parametrize(
    "model_type",
    ["openpangu_v2", "openpangu_v2_vl_moe", "openpangu_v2_omni_moe"],
)
def test_pangu_mla_detection_requires_kv_lora_rank(model_type):
    assert _convertor(model_type, 128).is_deepseek_mla() is True
    assert _convertor(model_type).is_deepseek_mla() is False


@pytest.mark.parametrize("model_type", ["deepseek_v3", None])
def test_non_pangu_models_delegate_to_upstream(monkeypatch, model_type):
    calls = []

    def original(self):
        calls.append(self)
        return "upstream-result"

    monkeypatch.setattr(patch_mod, "_ORIGINAL_IS_DEEPSEEK_MLA", original)
    convertor = _convertor(model_type, 128)

    assert convertor.is_deepseek_mla() == "upstream-result"
    assert calls == [convertor]


def test_model_arch_patch_registration():
    cls = patch_mod.PanguV2MoeModelArchConfigConvertorPatch
    assert cls._target is patch_mod.ModelArchConfigConvertorBase
    assert cls._attr_names_to_apply == ["is_deepseek_mla"]
