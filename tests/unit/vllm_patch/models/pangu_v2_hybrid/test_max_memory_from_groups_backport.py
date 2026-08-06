# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for MaxMemoryUsageFromGroupsBackport36030 (backport of vllm #36030).

Requires a working ``vllm`` import (NPU / CI docker); skipped otherwise.

Validates that hybrid startup admission sums the per-request block need over
ALL kv-cache groups instead of only ``groups[0]`` — the v0.14.0 bug that lets a
config which cannot fit one ``max_model_len`` request start, then livelock
(see docs/discussions/vllm-hybrid-kv-admission-livelock-0.87x.md).
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")  # needs a real vllm environment (NPU / CI docker)

from vllm.v1.core import kv_cache_utils  # noqa: E402
from omni.vllm_patches.patches.models.pangu_v2_hybrid import (  # noqa: E402
    patch_kv_cache_utils as P,
)


def _make_group(n_layers, block_need, page):
    """A general (non-UniformType) group whose single layer needs `block_need`
    blocks, i.e. max_memory_usage_bytes == block_need * page."""
    spec = SimpleNamespace(max_memory_usage_bytes=lambda cfg: block_need * page)
    return SimpleNamespace(layer_names=["l"] * n_layers, kv_cache_spec=spec)


@pytest.mark.unit
def test_sums_block_need_over_all_groups(monkeypatch):
    """General case must sum ALL groups (the #36030 fix), not only groups[0]."""
    PAGE = 1000
    monkeypatch.setattr(kv_cache_utils, "get_uniform_page_size", lambda specs: PAGE)

    # DSA-like group needs 4096 blocks/layer, SWA-like needs 800.
    groups = [_make_group(16, 4096, PAGE), _make_group(16, 800, PAGE)]

    result = P._max_memory_usage_bytes_from_groups_patched(None, groups)

    # group_size(16) * PAGE * (4096 + 800)
    assert result == 16 * PAGE * (4096 + 800)
    # Regression guard: strictly greater than the buggy "groups[0] only" value
    # (16 * PAGE * 4096) that let the livelock config pass startup admission.
    assert result > 16 * PAGE * 4096


@pytest.mark.unit
def test_empty_groups_returns_zero():
    assert P._max_memory_usage_bytes_from_groups_patched(None, []) == 0


@pytest.mark.unit
def test_single_general_group_no_inflation(monkeypatch):
    """A single general group equals its own block need (sum of one)."""
    PAGE = 1000
    monkeypatch.setattr(kv_cache_utils, "get_uniform_page_size", lambda specs: PAGE)

    result = P._max_memory_usage_bytes_from_groups_patched(
        None, [_make_group(16, 4096, PAGE)]
    )
    assert result == 16 * PAGE * 4096
