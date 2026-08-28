# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from omni_npu.v1.layers.attention import weight_utils as wu_mod
from omni_npu.v1.layers.attention.weight_utils import (
    install_q_b_split_loaders,
    mark_split_q_up_params_loaded,
    release_q_b_proj_storage,
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


def test_release_q_b_proj_storage_requires_prior_load():
    module = nn.Module()
    module.weight = nn.Parameter(torch.empty(6, 4))

    # No split loader run yet (no plain shape recorded): release is a no-op,
    # e.g. the init-time post_weight_load before weights are loaded.
    release_q_b_proj_storage(module)
    assert module.weight.data.numel() == 24

    # Simulate a completed load, then release.
    module.weight._q_b_plain_shape = (6, 4)
    release_q_b_proj_storage(module)
    assert module.weight.data.numel() == 0
    assert module.weight._q_b_storage_released is True

    # Releasing again must not fail.
    release_q_b_proj_storage(module)
    assert module.weight.data.numel() == 0


def test_split_loader_reload_after_release_rematerializes_and_splits():
    """After release, reload must re-materialize storage, refresh split projections, and release again."""
    src = nn.Module()
    src.weight = nn.Parameter(torch.empty(6, 4))
    nope = nn.Module()
    nope.weight = nn.Parameter(torch.empty(4, 4))
    pe = nn.Module()
    pe.weight = nn.Parameter(torch.empty(2, 4))

    def original_loader(param, loaded_weight, *args, **kwargs):
        with torch.no_grad():
            param.copy_(loaded_weight)
        return "loaded"

    src.weight.weight_loader = original_loader
    install_q_b_split_loaders(src, nope, pe, 3, 2)

    first = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    src.weight.weight_loader(src.weight, first)
    # Before release (pre-PWAL) the source stays materialized.
    assert src.weight.data.numel() == 24

    release_q_b_proj_storage(src)
    assert src.weight.data.numel() == 0

    second = torch.arange(24, 48, dtype=torch.float32).reshape(6, 4)
    result = src.weight.weight_loader(src.weight, second)

    second_by_head = second.reshape(2, 3, 4)
    assert result == "loaded"
    torch.testing.assert_close(nope.weight, second_by_head[:, :2].reshape(4, 4))
    torch.testing.assert_close(pe.weight, second_by_head[:, 2:].reshape(2, 4))
    # The source storage is released again after the splits are refreshed.
    assert src.weight.data.numel() == 0


def _copy_loader(param, loaded_weight, *args, **kwargs):
    """Copy loaded_weight into param for split-loader tests."""
    with torch.no_grad():
        param.copy_(loaded_weight)
    return "loaded"


def test_plain_layout_casts_nz_then_transposes(monkeypatch):
    """_plain_layout ND-casts NZ weights before optionally transposing."""
    casts = []

    def fake_cast(tensor, fmt):
        casts.append(fmt)
        return tensor

    monkeypatch.setattr(
        wu_mod, "torch_npu",
        SimpleNamespace(npu_format_cast=fake_cast, Format=SimpleNamespace(ND=2)),
        raising=False,
    )
    import torch_npu
    monkeypatch.setattr(torch_npu, "npu_format_cast", fake_cast, raising=False)
    monkeypatch.setattr(
        torch_npu, "Format", SimpleNamespace(ND=2, FRACTAL_NZ=29), raising=False
    )
    param = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    param.is_weight_nz = True
    param.is_weight_transposed = True
    out = wu_mod._plain_layout(param)
    assert casts == [2]
    torch.testing.assert_close(out, param.data.t())


def test_store_post_pwal_restores_nz_and_transposed_layout(monkeypatch):
    """_store_post_pwal copies plain data then re-applies transpose and NZ."""
    casts = []
    recapture = []

    def fake_cast(tensor, fmt):
        casts.append(fmt)
        return tensor.clone()

    import torch_npu
    monkeypatch.setattr(torch_npu, "npu_format_cast", fake_cast, raising=False)
    monkeypatch.setattr(
        torch_npu, "Format", SimpleNamespace(ND=2, FRACTAL_NZ=29), raising=False
    )
    monkeypatch.setattr(
        "omni_npu.compilation.acl_graph.set_aclgraph_recapture",
        lambda flag: recapture.append(flag),
    )
    param = nn.Parameter(torch.zeros(3, 2))
    param.is_weight_nz = True
    param.is_weight_transposed = True
    plain = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    wu_mod._store_post_pwal(param, plain)
    assert 2 in casts and 29 in casts
    assert recapture == [True]


def test_split_loader_supports_1d_bias_layout():
    """1D q_b_proj weights split along heads the same way as 2D weights."""
    src = nn.Module()
    src.weight = nn.Parameter(torch.empty(6))
    nope = nn.Module()
    nope.weight = nn.Parameter(torch.empty(4))
    pe = nn.Module()
    pe.weight = nn.Parameter(torch.empty(2))
    src.weight.weight_loader = _copy_loader
    install_q_b_split_loaders(src, nope, pe, 3, 2)
    loaded = torch.arange(6, dtype=torch.float32)
    assert src.weight.weight_loader(src.weight, loaded) == "loaded"
    expected = loaded.reshape(2, 3)
    torch.testing.assert_close(nope.weight, expected[:, :2].reshape(4))
    torch.testing.assert_close(pe.weight, expected[:, 2:].reshape(2))


def test_split_loader_raises_on_shape_mismatch():
    """Split destinations must match the expected per-head shapes."""
    src = nn.Module()
    src.weight = nn.Parameter(torch.empty(6, 4))
    nope = nn.Module()
    nope.weight = nn.Parameter(torch.empty(2, 4))
    pe = nn.Module()
    pe.weight = nn.Parameter(torch.empty(2, 4))
    src.weight.weight_loader = _copy_loader
    install_q_b_split_loaders(src, nope, pe, 3, 2)
    with pytest.raises(ValueError, match="shape mismatch"):
        src.weight.weight_loader(
            src.weight, torch.arange(24, dtype=torch.float32).reshape(6, 4)
        )


def test_split_loader_raises_when_out_dim_not_divisible():
    """q_b_proj output dim must be divisible by qk_head_dim."""
    src = nn.Module()
    src.weight = nn.Parameter(torch.empty(5, 4))
    nope = nn.Module()
    nope.weight = nn.Parameter(torch.empty(4, 4))
    pe = nn.Module()
    pe.weight = nn.Parameter(torch.empty(2, 4))
    src.weight.weight_loader = _copy_loader
    install_q_b_split_loaders(src, nope, pe, 3, 2)
    with pytest.raises(ValueError, match="not divisible"):
        src.weight.weight_loader(
            src.weight, torch.arange(20, dtype=torch.float32).reshape(5, 4)
        )


def test_install_skips_missing_scale_or_loader():
    """install_q_b_split_loaders ignores attrs without a matching loader."""
    src = nn.Module()
    src.weight = nn.Parameter(torch.empty(6, 4))
    nope = nn.Module()
    nope.weight = nn.Parameter(torch.empty(4, 4))
    pe = nn.Module()
    pe.weight = nn.Parameter(torch.empty(2, 4))
    install_q_b_split_loaders(src, nope, pe, 3, 2)
    assert getattr(src.weight, "weight_loader", None) is None
    src.weight_scale = nn.Parameter(torch.empty(6, 1))
    nope.weight_scale = nn.Parameter(torch.empty(4, 1))
    src.weight_scale.weight_loader = _copy_loader
    install_q_b_split_loaders(src, nope, pe, 3, 2)
    assert src.weight_scale.weight_loader is _copy_loader
