# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch


@pytest.fixture
def merge(patch_env):
    def flatten(value):
        if isinstance(value, torch.Tensor):
            return value.flatten(0, -2)
        return torch.cat([flatten(item) for item in value])

    target = patch_env.stub(
        "vllm.model_executor.models.utils", _flatten_embeddings=flatten,
        _embedding_count_expression=lambda value: str(flatten(value).shape[0]),
    )
    patch_env.stub("vllm.multimodal", NestedTensors=object)
    mod = patch_env.load("patch_multimodal_embeddings")
    mod.NPU_MergeMultimodalEmbeddingsPatch.apply()
    return target._merge_multimodal_embeddings


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["sparse", "all", "batched", "strided"])
def test_merge_matches_masked_scatter_in_place(merge, dtype, layout):
    storage = torch.arange(48, dtype=torch.float32).reshape(6, 8).to(dtype)
    if layout == "batched":
        inputs = storage.reshape(2, 3, 8)
        mask = torch.tensor([[False, True, True], [True, False, False]])
    elif layout == "strided":
        inputs = storage[:, ::2]
        assert not inputs.is_contiguous()
        mask = torch.tensor([True, False, True, False, False, True])
    else:
        inputs = storage
        mask = torch.ones(6, dtype=torch.bool) if layout == "all" else torch.tensor(
            [False, True, True, False, True, False])
    count = int(mask.sum())
    features = torch.linspace(-3, 7, count * inputs.shape[-1], dtype=torch.float64).reshape(count, -1)
    before = inputs.clone()
    expected = inputs.clone().masked_scatter_(mask.unsqueeze(-1), features.to(dtype))
    result = merge(inputs, [features[:1], [features[1:]]], mask)
    assert result is inputs
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(result[~mask], before[~mask], rtol=0, atol=0)


@pytest.mark.parametrize("features", [[], [torch.empty(0, 4)]])
def test_no_placeholders_preserves_input(merge, features):
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    before = inputs.clone()
    assert merge(inputs, features, torch.zeros(3, dtype=torch.bool)) is inputs
    torch.testing.assert_close(inputs, before, rtol=0, atol=0)


@pytest.mark.parametrize("actual", [0, 1, 3])
def test_count_mismatch_does_not_broadcast_or_partially_write(merge, actual):
    inputs = torch.ones(4, 3)
    with pytest.raises(ValueError, match=f"{actual} multimodal tokens to 2 placeholders"):
        merge(inputs, [torch.zeros(actual, 3)], torch.tensor([False, True, True, False]))
    assert torch.equal(inputs, torch.ones(4, 3))


def test_index_put_error_keeps_the_original_cause(merge):
    inputs = torch.ones(2, 4)
    with pytest.raises(ValueError, match="Error during index_put operation") as caught:
        merge(inputs, [torch.zeros(2, 3)], torch.ones(2, dtype=torch.bool))
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert torch.equal(inputs, torch.ones(2, 4))
