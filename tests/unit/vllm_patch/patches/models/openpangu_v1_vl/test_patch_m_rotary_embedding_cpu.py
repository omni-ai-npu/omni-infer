# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Host position arithmetic and wrapper contracts; no tensor kernels are mocked.

The torch/vLLM doubles supply only types and constructor interfaces. Numerical
RoPE and OOT dispatch are covered separately by the real-runtime tests.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")


@pytest.fixture
def rope(patch_env):
    default_dtype = object()
    patch_env.stub("torch", dtype=object, Tensor=object,
                   get_default_dtype=lambda: default_dtype)
    group = SimpleNamespace(world_size=1)
    patch_env.stub("vllm.distributed", get_pp_group=lambda: group)

    class Base:
        def __init__(self, *args):
            self.base_args = args
            self.cache_max_position_num = 8

    class Interleaved(Base):
        get_mrope_interleaved_id_list = Mock(return_value=[0, 1, 2, 2])

    original = Mock(return_value=object())
    target = patch_env.stub(
        "vllm.model_executor.layers.rotary_embedding", MRotaryEmbedding=Base,
        MRotaryEmbeddingInterleaved=Interleaved, RotaryEmbedding=Base,
        get_rope=original, _ROPE_DICT={},
    )
    mod = patch_env.load("patch_m_rotary_embedding")
    mod.RotaryEmbeddingModulePatch.apply()
    assert target.MRotaryEmbeddingInterleaved is Interleaved
    return SimpleNamespace(mod=mod, target=target, group=group, original=original,
                           dtype=default_dtype, base=Base, interleaved=Interleaved)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("count", [0, 1, 4])
@pytest.mark.parametrize("delta,context", [(7, 11), (-3, 11)])
def test_decode_position_write_matches_arange_and_preserves_surroundings(rope, dtype, count, delta, context):
    out = np.full((3, 9), -99, dtype=dtype)
    rope.base.get_next_input_positions_tensor(out, 2, delta, context, count)
    expected = np.full((3, 9), -99, dtype=dtype)
    expected[:, 2:2 + count] = np.arange(delta + context, delta + context + count, dtype=dtype)
    np.testing.assert_array_equal(out, expected)


def test_host_cache_reuses_grows_and_separates_dtypes(rope):
    patch = rope.mod.RotaryEmbeddingModulePatch
    dtype = np.dtype(np.int64)
    first = patch._get_np_position_slice(2, 5, dtype)
    cached = patch._position_cache[dtype]
    assert np.shares_memory(first, cached)
    patch._get_np_position_slice(1, 4, dtype)
    assert patch._position_cache[dtype] is cached
    grown = patch._get_np_position_slice(5, 8, dtype)
    assert patch._position_cache[dtype].size == 10
    np.testing.assert_array_equal(grown, [5, 6, 7])
    other = patch._get_np_position_slice(2, 5, np.dtype(np.int32))
    np.testing.assert_array_equal(other, first)
    assert other.dtype == np.int32 and not np.shares_memory(other, first)


def build(rope, **overrides):
    args = dict(head_size=8, rotary_dim=8, max_position=16, base=10000,
                rope_scaling={"mrope_interleaved": True, "mrope_section": [1, 1, 2]},
                num_hidden_layers_cache=3)
    args.update(overrides)
    return rope.target.get_rope_wrapper(**args)


@pytest.mark.parametrize("pp,expected_layers", [(1, 3), (2, 1)])
def test_wrapper_preserves_target_class_and_pipeline_cache_contract(rope, pp, expected_layers):
    rope.group.world_size = pp
    result = build(rope)
    assert isinstance(result, rope.interleaved)
    assert rope.target.MRotaryEmbeddingInterleaved is rope.interleaved
    assert result.base_args == (8, 8, 16, 10000, True, rope.dtype)
    assert result.num_hidden_layers_cache == expected_layers
    assert result.layer_cache is None and result.layer_counts == 0
    assert result.mrope_dim == [0, 1, 2, 2] * 2
    assert result.mrope_section_3d == [1] * 8
    result.get_mrope_interleaved_id_list.assert_called_once_with(1, 1, 2, force_last=True)
    assert rope.mod.RotaryEmbeddingModulePatch._position_cache[np.dtype(np.int64)].size == 72


def test_wrapper_cache_hit_and_isolation(rope):
    first = build(rope)
    assert build(rope) is first
    assert build(rope, base=20000) is not first
    assert build(rope, dtype=object()) is not first
    assert build(rope, num_hidden_layers_cache=2) is not first
    assert build(rope, rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 1, 1]}) is not first
    assert len(rope.target._ROPE_DICT) == 5


def test_partial_rotary_and_two_sections_keep_input_config_unchanged(rope):
    scaling = {"mrope_interleaved": True, "mrope_section": [1, 1], "rotary_mode": "interleave"}
    result = build(rope, partial_rotary_factor=.5, rope_scaling=scaling)
    assert result.base_args[1] == 4
    assert result.rotary_mode == "interleave"
    result.get_mrope_interleaved_id_list.assert_called_once_with(1, 1, 0)
    assert scaling == {"mrope_interleaved": True, "mrope_section": [1, 1], "rotary_mode": "interleave"}


@pytest.mark.parametrize("scaling", [None, {"rope_type": "linear", "factor": 2.0, "rope_theta": 1234}])
def test_language_rope_delegates_without_mutating_scaling(rope, scaling):
    original_scaling = None if scaling is None else dict(scaling)
    dual_chunk = {"chunk_size": 4}
    result = rope.target.get_rope_wrapper(8, 4, 16, 10000, False, scaling, rope.dtype, .5, dual_chunk)
    parameters = {"rope_theta": 10000, "rope_type": "default"} if scaling is None else dict(scaling)
    parameters.update(partial_rotary_factor=.5, rope_dim=4)
    rope.original.assert_called_once_with(8, 16, False, parameters, rope.dtype, dual_chunk)
    assert result is rope.original.return_value
    assert scaling == original_scaling
    assert rope.target._ROPE_DICT == {}


@pytest.mark.parametrize("changes,message", [
    ({"mrope_section": None}, "cannot be None"),
    ({"mrope_section": [1, 1]}, "must equal"),
    ({"mrope_interleaved": False}, "must be true"),
    ({"rotary_mode": "unknown"}, "half.*interleave"),
    ({"num_hidden_layers_cache": 0}, "at least 1"),
    ({"mrope_section": [1, 1, 1, 1]}, "two or three"),
])
def test_existing_constructor_validation(rope, changes, message):
    args = dict(head_size=8, rotary_dim=8, max_position_embeddings=16, base=10000,
                is_neox_style=True, dtype=rope.dtype, mrope_section=[1, 1, 2])
    args.update(changes)
    with pytest.raises(ValueError, match=message):
        rope.interleaved(**args)


def test_cos_sin_delegates_to_existing_rebuild_without_copying(rope):
    layer = build(rope)
    positions, cos, sin = object(), object(), object()
    layer._rebuild_pos_emb = Mock(return_value=(cos, sin))
    result = layer.get_cos_sin(positions)
    assert result[0] is cos and result[1] is sin
    layer._rebuild_pos_emb.assert_called_once_with(positions)
