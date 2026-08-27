# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
from torch import nn

from omni_npu.v1.layers.attention.weight_utils import (
    install_q_b_split_loaders,
    mark_split_q_up_params_loaded,
)


pytestmark = pytest.mark.unit


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))


class _AttentionWithSplitQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_b_proj = _Projection()
        self.q_b_nope_proj = _Projection()
        self.q_b_pe_proj = _Projection()


def test_marks_split_projection_params_when_source_was_loaded():
    module = nn.Module()
    module.add_module("self_attn", _AttentionWithSplitQ())
    loaded = {"self_attn.q_b_proj.weight"}

    result = mark_split_q_up_params_loaded(module, loaded)

    assert result is loaded
    assert loaded == {
        "self_attn.q_b_proj.weight",
        "self_attn.q_b_nope_proj.weight",
        "self_attn.q_b_pe_proj.weight",
    }


def test_does_not_mark_split_projection_params_without_loaded_source():
    module = nn.Module()
    module.add_module("self_attn", _AttentionWithSplitQ())
    loaded = {"some_other.weight"}

    mark_split_q_up_params_loaded(module, loaded)

    assert loaded == {"some_other.weight"}


def test_split_loader_forwards_args_and_splits_by_head():
    src = nn.Module()
    src.weight = nn.Parameter(torch.empty(6, 4))
    nope = nn.Module()
    nope.weight = nn.Parameter(torch.empty(4, 4))
    pe = nn.Module()
    pe.weight = nn.Parameter(torch.empty(2, 4))
    loaded = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    calls = []

    def original_loader(param, loaded_weight, *args, **kwargs):
        calls.append((args, kwargs))
        with torch.no_grad():
            param.copy_(loaded_weight)
        return "loaded"

    src.weight.weight_loader = original_loader
    install_q_b_split_loaders(src, nope, pe, 3, 2)
    result = src.weight.weight_loader(
        src.weight,
        loaded,
        "shard",
        shard_id=3,
    )

    expected = loaded.reshape(2, 3, 4)
    assert result == "loaded"
    assert calls == [(("shard",), {"shard_id": 3})]
    torch.testing.assert_close(nope.weight, expected[:, :2].reshape(4, 4))
    torch.testing.assert_close(pe.weight, expected[:, 2:].reshape(2, 4))


def test_split_loader_second_load_restores_transposed_layout():
    """RL weight sync reloads into weights that PWAL already transposed.

    The split loader must slice on the restored plain layout and re-apply
    the transposed layout on the split params, mirroring the linear
    weight_loader's veRL special case.
    """
    src = nn.Module()
    src.weight = nn.Parameter(torch.empty(6, 4))
    nope = nn.Module()
    nope.weight = nn.Parameter(torch.empty(4, 4))
    pe = nn.Module()
    pe.weight = nn.Parameter(torch.empty(2, 4))

    def original_loader(param, loaded_weight, *args, **kwargs):
        if getattr(param, "is_weight_transposed", False):
            param.data = param.data.t_()
        with torch.no_grad():
            param.data.copy_(loaded_weight)
        if getattr(param, "is_weight_transposed", False):
            param.data = param.data.t_()
        return "loaded"

    src.weight.weight_loader = original_loader
    install_q_b_split_loaders(src, nope, pe, 3, 2)

    first = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    src.weight.weight_loader(src.weight, first)
    first_by_head = first.reshape(2, 3, 4)
    torch.testing.assert_close(nope.weight, first_by_head[:, :2].reshape(4, 4))

    # Simulate process_weights_after_loading transposing all three weights.
    for param in (src.weight, nope.weight, pe.weight):
        param.data = param.data.t().contiguous()
        param.is_weight_transposed = True

    second = torch.arange(24, 48, dtype=torch.float32).reshape(6, 4)
    result = src.weight.weight_loader(src.weight, second)

    second_by_head = second.reshape(2, 3, 4)
    assert result == "loaded"
    # q_b_proj ends up transposed again with the new values.
    assert tuple(src.weight.shape) == (4, 6)
    torch.testing.assert_close(src.weight.data.t(), second)
    # Split params received the new slices and were restored to the
    # transposed layout as well.
    assert tuple(nope.weight.shape) == (4, 4)
    torch.testing.assert_close(
        nope.weight.data.t(), second_by_head[:, :2].reshape(4, 4)
    )
    assert tuple(pe.weight.shape) == (4, 2)
    torch.testing.assert_close(
        pe.weight.data.t(), second_by_head[:, 2:].reshape(2, 4)
    )
