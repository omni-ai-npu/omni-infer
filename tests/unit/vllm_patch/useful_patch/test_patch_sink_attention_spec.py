import pytest
import torch

from omni_npu.vllm_patches.usefull_patch.models.high_throughout import (
    patch_sink_attention_spec as patch_mod,
)


def _sink_mla(sink_len=128, **overrides):
    kwargs = {
        "block_size": 4,
        "num_kv_heads": 1,
        "head_size": 576,
        "dtype": torch.bfloat16,
        "sink_len": sink_len,
    }
    kwargs.update(overrides)
    return patch_mod.SinkMLAAttentionSpec(**kwargs)


def test_sink_mla_spec_keeps_sink_len_and_merges_uniform_groups():
    spec = _sink_mla()
    assert spec.sink_len == 128
    merged = patch_mod.SinkMLAAttentionSpec.merge([_sink_mla(64), _sink_mla(64)])
    assert isinstance(merged, patch_mod.SinkMLAAttentionSpec)
    assert merged.sink_len == 64
    assert merged.block_size == 4
    assert merged.head_size == 576


def test_sink_mla_spec_merge_rejects_mismatched_sink_len():
    with pytest.raises(AssertionError, match="same sink_len"):
        patch_mod.SinkMLAAttentionSpec.merge([_sink_mla(64), _sink_mla(128)])


def test_sink_attention_spec_patch_registration():
    cls = patch_mod.SinkAttentionSpecPatch
    assert cls._target is patch_mod.kv_cache_interface
    assert cls.SinkMLAAttentionSpec is patch_mod.SinkMLAAttentionSpec
    assert "SinkMLAAttentionSpec" in cls._attr_names_to_apply
