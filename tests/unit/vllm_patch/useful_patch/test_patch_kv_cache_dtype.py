from typing import get_args

import torch

from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_moe import patch_kv_cache_dtype as patch_mod


def test_extended_cache_dtype_contains_custom_values():
    dtype_values = set(get_args(patch_mod._EXTENDED_CACHE_DTYPE))
    assert set(patch_mod.NEW_CACHE_DTYPES) <= dtype_values
    assert patch_mod.NEW_CACHE_DTYPES == {
        "hif8_ds_mla": torch.uint8,
        "int8_ds_mla": torch.int8,
        "li_int8_ds_mla": torch.bfloat16,
    }


def test_cache_config_uses_placeholder_then_restores_custom_dtype(monkeypatch):
    calls = []

    def original(self, *args, **kwargs):
        calls.append((args, dict(kwargs)))
        self.cache_dtype = kwargs.get("cache_dtype")

    monkeypatch.setattr(
        patch_mod.CacheConfigPatch, "_original_cache_config_init", original
    )
    instance = object.__new__(patch_mod.CacheConfigPatch)

    patch_mod.CacheConfigPatch.__init__(
        instance, cache_dtype="int8_ds_mla", block_size=128
    )

    assert calls == [
        ((), {"cache_dtype": patch_mod._PLACEHOLDER_DTYPE, "block_size": 128})
    ]
    assert instance.cache_dtype == "int8_ds_mla"


def test_cache_config_leaves_upstream_dtype_unchanged(monkeypatch):
    calls = []

    def original(self, *args, **kwargs):
        calls.append(dict(kwargs))
        self.cache_dtype = kwargs.get("cache_dtype")

    monkeypatch.setattr(
        patch_mod.CacheConfigPatch, "_original_cache_config_init", original
    )
    instance = object.__new__(patch_mod.CacheConfigPatch)

    patch_mod.CacheConfigPatch.__init__(instance, cache_dtype="auto")

    assert calls == [{"cache_dtype": "auto"}]
    assert instance.cache_dtype == "auto"


def test_cache_config_field_patch_updates_cli_introspection():
    field = patch_mod.CacheConfig.__dataclass_fields__["cache_dtype"]
    original_annotation = patch_mod.CacheConfig.__annotations__["cache_dtype"]
    original_field_type = field.type
    try:
        patch_mod.CacheConfigFieldTypePatch.apply()
        assert (
            patch_mod.CacheConfig.__annotations__["cache_dtype"]
            is patch_mod._EXTENDED_CACHE_DTYPE
        )
        assert field.type is patch_mod._EXTENDED_CACHE_DTYPE
    finally:
        patch_mod.CacheConfig.__annotations__["cache_dtype"] = original_annotation
        field.type = original_field_type


def test_torch_dtype_mapping_preserves_upstream_and_adds_custom_values():
    mapping = patch_mod.ExtendStrDtypeToTorchDtypePatch.STR_DTYPE_TO_TORCH_DTYPE
    for name, dtype in patch_mod.NEW_CACHE_DTYPES.items():
        assert mapping[name] is dtype
    for name, dtype in patch_mod.torch_utils_module.STR_DTYPE_TO_TORCH_DTYPE.items():
        assert mapping[name] is dtype
