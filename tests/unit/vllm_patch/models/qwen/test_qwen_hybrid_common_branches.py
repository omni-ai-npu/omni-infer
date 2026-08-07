# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, MambaSpec

from omni_npu.vllm_patches.patches.models.qwen import qwen_hybrid_common as hybrid


def _attention_spec(**kwargs):
    values = dict(block_size=128, num_kv_heads=1, head_size=16, dtype=torch.float16)
    values.update(kwargs)
    return AttentionSpec(**values)


def _page_cache(num_blocks=2, block_size=2, hidden_size=2):
    raw = torch.arange(
        num_blocks * 2 * block_size * hidden_size, dtype=torch.float32
    )
    page_stride = 2 * block_size * hidden_size
    key = torch.as_strided(
        raw,
        (num_blocks, block_size, hidden_size),
        (page_stride, hidden_size, 1),
    )
    value = torch.as_strided(
        raw,
        (num_blocks, block_size, hidden_size),
        (page_stride, hidden_size, 1),
        storage_offset=block_size * hidden_size,
    )
    return key, value


def test_attention_spec_validation_errors():
    with pytest.raises(ValueError, match="requires an attention group"):
        hybrid.get_attention_kv_cache_spec([])

    first = _attention_spec()
    second = _attention_spec(block_size=256)
    with pytest.raises(ValueError, match="Attention specs must match"):
        hybrid.get_attention_kv_cache_spec(
            [
                KVCacheGroupSpec(["a"], first),
                KVCacheGroupSpec(["b"], second),
            ]
        )


def test_native_mamba_reshape_uses_largest_state_first():
    spec = MambaSpec(
        block_size=1,
        shapes=((2, 2), (1, 8)),
        dtypes=(torch.float16, torch.float16),
        page_size_padded=None,
    )
    raw = torch.zeros(spec.page_size_bytes * 2, dtype=torch.uint8)

    states = hybrid.reshape_native_mamba_kv_cache(raw, spec, 2)

    assert [tuple(state.shape) for state in states] == [(2, 2, 2), (2, 1, 8)]
    assert states[0].stride(0) == spec.page_size_bytes // 2
    assert states[1].storage_offset() == 0
    assert states[0].storage_offset() > states[1].storage_offset()


def test_native_attention_reshape_returns_page_strided_key_value_views():
    spec = _attention_spec()
    raw = torch.zeros(spec.page_size_bytes * 2, dtype=torch.uint8)

    key, value = hybrid.reshape_native_attention_kv_cache(raw, 2, spec)

    assert key.shape == value.shape
    assert key.stride(0) == spec.page_size_bytes // spec.dtype.itemsize
    assert value.storage_offset() > key.storage_offset()


@pytest.mark.parametrize(
    "key_shape,value_shape",
    [
        ((2, 2), (2, 2, 2)),
        ((2, 2, 2), (2, 2, 3)),
    ],
)
def test_page_dense_helper_rejects_incompatible_cache_shapes(key_shape, value_shape):
    key = torch.zeros(*key_shape)
    value = torch.zeros(*value_shape)
    assert hybrid.maybe_get_page_dense_kv_cache(key, value) is None


def test_page_dense_helper_rejects_different_inner_strides():
    key, value = _page_cache()
    value = value.transpose(1, 2)
    assert hybrid.maybe_get_page_dense_kv_cache(key, value) is None


def test_scatter_native_attention_fallback_updates_regular_cache():
    key_cache = torch.zeros(2, 2, 2)
    value_cache = torch.zeros_like(key_cache)
    key = torch.tensor([[1.0, 2.0]])
    value = torch.tensor([[3.0, 4.0]])

    hybrid.scatter_native_attention_kv_cache(
        key_cache,
        value_cache,
        torch.tensor([3]),
        key,
        value,
    )

    assert torch.equal(key_cache[1, 1], key[0])
    assert torch.equal(value_cache[1, 1], value[0])


def test_scatter_native_attention_dense_path_uses_kernel_slots(monkeypatch):
    key_cache, value_cache = _page_cache()
    calls = []

    def scatter(target, indices, updates):
        calls.append((indices.clone(), updates.clone()))
        target.index_copy_(0, indices.reshape(-1).long(), updates)

    monkeypatch.setattr(hybrid.torch_npu, "npu_scatter_nd_update_", scatter)
    hybrid.scatter_native_attention_kv_cache(
        key_cache,
        value_cache,
        torch.tensor([0, 3]),
        torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
        torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
        num_actual_tokens=1,
    )

    assert len(calls) == 2
    assert calls[0][0].reshape(-1).tolist() == [0]


def test_reshape_non_attention_rejects_attention_spec():
    with pytest.raises(NotImplementedError, match="Unsupported kv_cache_spec"):
        hybrid.reshape_non_attention_kv_cache(
            torch.zeros(16, dtype=torch.uint8),
            _attention_spec(),
            1,
        )


def test_get_attention_group_block_size():
    attention = _attention_spec()
    config = SimpleNamespace(
        kv_cache_groups=[KVCacheGroupSpec(["layer0", "layer1"], attention)]
    )

    assert hybrid._get_attention_group_block_size(config) == 128
    assert hybrid._get_attention_group_block_size(
        SimpleNamespace(kv_cache_groups=[])
    ) is None


def test_reshape_native_hybrid_cache_handles_attention_and_mamba_groups():
    attention = _attention_spec()
    mamba = MambaSpec(
        block_size=1,
        shapes=((2, 2),),
        dtypes=(torch.float16,),
        page_size_padded=None,
    )
    groups = [
        SimpleNamespace(
            kv_cache_spec=attention,
            backend=None,
            layer_names=["attn"],
        ),
        SimpleNamespace(
            kv_cache_spec=mamba,
            backend=None,
            layer_names=["gdn"],
        ),
    ]
    runner = SimpleNamespace(
        runner_only_attn_layers=[],
        _kv_cache_spec_attn_group_iterator=lambda: iter(groups),
    )

    result = hybrid.reshape_native_hybrid_kv_cache_tensors(
        runner,
        {
            "attn": torch.zeros(attention.page_size_bytes, dtype=torch.uint8),
            "gdn": torch.zeros(mamba.page_size_bytes, dtype=torch.uint8),
        },
    )

    assert set(result) == {"attn", "gdn"}
    assert len(result["attn"]) == 2
    assert len(result["gdn"]) == 1


def test_reshape_native_hybrid_cache_rejects_bad_raw_size():
    attention = _attention_spec()
    runner = SimpleNamespace(
        runner_only_attn_layers=[],
        _kv_cache_spec_attn_group_iterator=lambda: iter(
            [
                SimpleNamespace(
                    kv_cache_spec=attention,
                    backend=None,
                    layer_names=["attn"],
                )
            ]
        ),
    )

    with pytest.raises(ValueError, match="raw_tensor"):
        hybrid.reshape_native_hybrid_kv_cache_tensors(
            runner,
            {"attn": torch.zeros(3, dtype=torch.uint8)},
        )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"kv_sharing_target_layer_name": "shared"}, False),
        ({"key": None, "value": torch.zeros(1, 1, 2)}, False),
        ({"attn_metadata": SimpleNamespace(slot_mapping=None)}, False),
    ],
)
def test_should_update_native_attention_kv_cache_rejects_invalid_cases(
    kwargs, expected, monkeypatch
):
    impl = SimpleNamespace(
        kv_sharing_target_layer_name=kwargs.get("kv_sharing_target_layer_name"),
    )
    key = kwargs.get("key", torch.zeros(1, 1, 2))
    value = kwargs.get("value", torch.zeros(1, 1, 2))
    attn_metadata = kwargs.get(
        "attn_metadata",
        SimpleNamespace(slot_mapping=torch.tensor([0])),
    )
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.setattr(
        hybrid.model_extra_config.operator_opt_config,
        "enable_kv_rmsnorm_rope_cache",
        False,
    )

    assert (
        hybrid.should_update_native_attention_kv_cache(
            impl,
            key,
            value,
            attn_metadata,
        )
        is expected
    )


def test_should_update_native_attention_kv_cache_rejects_kv_rmsnorm_plugin(
    monkeypatch,
):
    impl = SimpleNamespace(kv_sharing_target_layer_name=None)
    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
    monkeypatch.setattr(
        hybrid.model_extra_config.operator_opt_config,
        "enable_kv_rmsnorm_rope_cache",
        True,
    )

    assert not hybrid.should_update_native_attention_kv_cache(
        impl,
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        SimpleNamespace(slot_mapping=torch.tensor([0])),
    )


def test_should_update_native_attention_kv_cache_accepts_valid_case(monkeypatch):
    impl = SimpleNamespace(kv_sharing_target_layer_name=None)
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.setattr(
        hybrid.model_extra_config.operator_opt_config,
        "enable_kv_rmsnorm_rope_cache",
        False,
    )

    assert hybrid.should_update_native_attention_kv_cache(
        impl,
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        SimpleNamespace(slot_mapping=torch.tensor([0])),
    )


def _native_attn_impl(num_kv_heads=1):
    impl = hybrid.QwenNativeAttentionImplPatch.__new__(
        hybrid.QwenNativeAttentionImplPatch
    )
    impl.num_kv_heads = num_kv_heads
    impl.kv_sharing_target_layer_name = None
    return impl


def test_native_attention_forward_scatters_and_remaps_dense_kv(monkeypatch):
    key_cache, value_cache = _page_cache()
    attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0]),
        num_actual_tokens=1,
        block_tables=torch.tensor([[0, 1]]),
    )
    query = torch.zeros(1, 1, 2)
    key = torch.ones(1, 1, 2)
    value = torch.ones(1, 1, 2) * 2
    scatter_calls = []
    delegate_calls = []

    monkeypatch.setattr(
        hybrid.model_extra_config.operator_opt_config,
        "kv_nz",
        False,
    )
    monkeypatch.setattr(
        hybrid,
        "scatter_native_attention_kv_cache",
        lambda *args, **kwargs: scatter_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        hybrid,
        "_npu_attention_backend_forward",
        lambda *args, **kwargs: delegate_calls.append((args, kwargs)) or "ok",
    )

    result = _native_attn_impl().forward(
        layer=None,
        query=query,
        key=key,
        value=value,
        kv_cache=(key_cache, value_cache),
        attn_metadata=attn_metadata,
    )

    assert result == "ok"
    assert len(scatter_calls) == 1
    assert len(delegate_calls) == 1
    passed_kv_cache = delegate_calls[0][0][5]
    passed_metadata = delegate_calls[0][0][6]
    assert passed_kv_cache[0].shape[0] == key_cache.shape[0] * 2 - 1
    assert torch.equal(passed_metadata.block_tables, attn_metadata.block_tables * 2)
    assert delegate_calls[0][0][3] is None
    assert delegate_calls[0][0][4] is None


def test_native_attention_forward_skips_scatter_when_kv_nz_enabled(monkeypatch):
    key_cache, value_cache = _page_cache()
    attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0]),
        num_actual_tokens=1,
        block_tables=torch.tensor([[0, 1]]),
    )
    scatter_calls = []

    monkeypatch.setattr(
        hybrid.model_extra_config.operator_opt_config,
        "kv_nz",
        True,
    )
    monkeypatch.setattr(
        hybrid,
        "scatter_native_attention_kv_cache",
        lambda *args, **kwargs: scatter_calls.append(True),
    )
    monkeypatch.setattr(
        hybrid,
        "_npu_attention_backend_forward",
        lambda *args, **kwargs: "delegate",
    )

    result = _native_attn_impl().forward(
        layer=None,
        query=torch.zeros(1, 1, 2),
        key=torch.ones(1, 1, 2),
        value=torch.ones(1, 1, 2),
        kv_cache=(key_cache, value_cache),
        attn_metadata=attn_metadata,
    )

    assert result == "delegate"
    assert scatter_calls == []
