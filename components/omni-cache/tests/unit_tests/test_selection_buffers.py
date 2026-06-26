# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch
import sys
import types

import pytest
import torch

from omni_cache.gather_selection.core.buffers import (
    SelectionBuffers,
    initialize_selection_buffers,
)


class FakeCache:
    def __init__(self):
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
        self.runner = SimpleNamespace(max_num_reqs=4)
        self.block_size = 16
        self.k_rope_size = 64
        self.kvcache_size = 512
        self.num_layers = 2
        self.device = torch.device("cpu")

    @property
    def selection_state_size(self):
        return 8192 * 2

    @property
    def s_max_block_num(self):
        return 512


def _cache():
    return FakeCache()


@pytest.fixture(autouse=True)
def _stub_custom_ops(monkeypatch):
    monkeypatch.setitem(sys.modules, "custom_ops", types.ModuleType("custom_ops"))


def test_legacy_path_creates_global_table_status_buffers(monkeypatch):
    monkeypatch.delenv("USE_OMNI_INPUT_BATCH", raising=False)
    cache = _cache()

    SelectionBuffers(cache)

    assert cache.s_max_block_num == 512
    assert cache.selection_state_size == 8192 * 2
    assert cache.selection_kv_cache.shape == (2, 512 * 4, 16, 512)
    assert cache.selection_k_rope.shape == (2, 512 * 4, 16, 1)
    assert cache.selection_kv_block_table.shape == (4, 512)
    assert cache.selection_kv_block_status.shape == (2, 4, 8192 * 2)
    assert cache.selection_kv_block_table_buffer.shape == (4, 512)
    assert cache.selection_kv_block_status_buffer.shape == (2, 4, 8192 * 2)
    assert cache.index_buffer.shape == (4,)


def test_input_batch_path_skips_global_table_status_buffers(monkeypatch):
    monkeypatch.setenv("USE_OMNI_INPUT_BATCH", "1")
    cache = _cache()

    SelectionBuffers(cache)

    assert cache.s_max_block_num == 512
    assert cache.selection_state_size == 8192 * 2
    assert cache.selection_kv_cache.shape == (2, 512 * 4, 16, 512)
    assert cache.selection_k_rope.shape == (2, 512 * 4, 16, 1)
    assert not hasattr(cache, "selection_kv_block_table")
    assert not hasattr(cache, "selection_kv_block_status")
    assert not hasattr(cache, "selection_kv_block_table_buffer")
    assert not hasattr(cache, "selection_kv_block_status_buffer")
    assert cache.index_buffer.shape == (4,)


def test_initialize_selection_buffers_constructs_selection_buffers():
    cache = _cache()

    with patch(
        "omni_cache.gather_selection.core.buffers.SelectionBuffers"
    ) as buffers_cls:
        initialize_selection_buffers(cache)

    buffers_cls.assert_called_once_with(cache)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
