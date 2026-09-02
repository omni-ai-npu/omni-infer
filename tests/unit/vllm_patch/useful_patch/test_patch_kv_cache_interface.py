from types import SimpleNamespace

import pytest
import torch

from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_hybrid import patch_kv_cache_interface as patch_mod


def _dsa(cache_dtype_str=None, **overrides):
    kwargs = {
        "block_size": 4,
        "num_kv_heads": 1,
        "head_size": 8,
        "dtype": torch.bfloat16,
        "cache_dtype_str": cache_dtype_str,
    }
    kwargs.update(overrides)
    return patch_mod.DSAAttentionSpec(**kwargs)


@pytest.mark.parametrize(
    "cache_dtype_str,bytes_per_token",
    [
        ("fp8_ds_mla", 656 + 128 + 4),
        ("int8_ds_mla", 656 + 128 + 2),
        ("hif8_ds_mla", 656 + 128 + 4),
        ("li_int8_ds_mla", 576 * 2 + 128 + 2),
    ],
)
def test_dsa_quantized_real_page_size(cache_dtype_str, bytes_per_token):
    assert _dsa(cache_dtype_str).real_page_size_bytes == 4 * bytes_per_token


def test_dsa_unquantized_page_size_and_validation():
    spec = _dsa()
    assert spec.real_page_size_bytes == 4 * 1 * 8 * 2
    assert spec.head_size_v == 8
    with pytest.raises(AssertionError, match="num_kv_heads"):
        _dsa(num_kv_heads=2)
    with pytest.raises(AssertionError, match="sliding window"):
        _dsa(sliding_window=16)


def test_dsa_merge_preserves_cache_dtype():
    merged = patch_mod.DSAAttentionSpec.merge(
        [_dsa("int8_ds_mla"), _dsa("int8_ds_mla")]
    )
    assert isinstance(merged, patch_mod.DSAAttentionSpec)
    assert merged.cache_dtype_str == "int8_ds_mla"

    with pytest.raises(AssertionError, match="same quantization method"):
        patch_mod.DSAAttentionSpec.merge(
            [_dsa("int8_ds_mla"), _dsa("hif8_ds_mla")]
        )


def test_share_kv_sliding_window_uses_single_shared_head_storage():
    spec = patch_mod.ShareKVSlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    assert spec.real_page_size_bytes == 4 * 1 * 512 * 2
    assert spec.head_size_v == 512

    with pytest.raises(AssertionError, match="single KV head"):
        patch_mod.ShareKVSlidingWindowSpec(
            block_size=4,
            num_kv_heads=2,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
        )
    with pytest.raises(AssertionError, match="512 or 576"):
        patch_mod.ShareKVSlidingWindowSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=128,
        )


def _mome(**overrides):
    kwargs = {
        "block_size": 4,
        "shapes": ((2,), (3,), (4,)),
        "dtypes": (torch.bfloat16, torch.bfloat16, torch.bfloat16),
        "kernel_size": 3,
        "num_spec_tokens": 2,
    }
    kwargs.update(overrides)
    return patch_mod.MomeSpec(**kwargs)


def test_mome_page_size_memory_and_uniformity():
    spec = _mome()
    assert spec.num_total_tokens == 4
    assert spec.page_size_bytes == (2 + 3 + 4) * 2 * 4
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=17),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=12),  # vllm 0.25.1 A-tier fix: N=12 (num_retained=2, num_tokens=14, cdiv(14,4)=4, max_blocks=4+1+0=5)
    )
    assert spec.max_memory_usage_bytes(config) == 5 * spec.page_size_bytes

    same_total = _mome(kernel_size=4, num_spec_tokens=1)
    different_total = _mome(kernel_size=5, num_spec_tokens=1)
    assert spec.is_uniform_with_collection({"a": spec, "b": same_total})
    assert not spec.is_uniform_with_collection({"a": spec, "b": different_total})


def test_mome_page_size_padding_and_validation():
    assert _mome(page_size_padded=100).page_size_bytes == 100
    with pytest.raises(AssertionError, match="must be >="):
        _mome(page_size_padded=10).page_size_bytes
    with pytest.raises(ValueError, match="3 components"):
        _mome(shapes=((1,), (2,)))
    with pytest.raises(ValueError, match="3 components"):
        _mome(dtypes=(torch.bfloat16, torch.bfloat16))
    with pytest.raises(ValueError, match="positive kernel_size"):
        _mome(kernel_size=0)


def test_new_kv_cache_specs_patch_registration():
    cls = patch_mod.PanguNewKVCacheSpecsPatch
    assert cls._target is patch_mod.kv_cache_interface
    assert cls.DSAAttentionSpec is patch_mod.DSAAttentionSpec
    assert cls.ShareKVSlidingWindowSpec is patch_mod.ShareKVSlidingWindowSpec
    assert cls.MomeSpec is patch_mod.MomeSpec


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
