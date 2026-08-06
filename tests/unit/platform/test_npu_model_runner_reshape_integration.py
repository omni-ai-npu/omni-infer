# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

"""Integration tests for NPUModelRunner KV cache reshape via public iterator."""

from unittest.mock import MagicMock, patch

import torch

from omni_npu.worker.npu_model_runner import NPUModelRunner
from tests.unit.platform.utils import create_vllm_config
from vllm.v1.kv_cache_interface import AttentionSpec


class TestNPUModelRunnerReshapeIntegration:
    def setup_method(self):
        self.vllm_cfg = create_vllm_config()
        with patch.object(NPUModelRunner, "_init_device_properties"):
            self.runner = NPUModelRunner(self.vllm_cfg, torch.device("cpu"))

    def test_reshape_kv_cache_tensors_uses_public_iterator(self, monkeypatch):
        kv_cache_spec = AttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float16,
        )

        class DummyBackend:
            def reshape_kv_cache(self, raw_tensor, num_blocks, kv_cache_spec, **kwargs):
                return torch.ones(2, 2, dtype=kv_cache_spec.dtype)

        class DummyGroup:
            def __init__(self):
                self.kv_cache_spec = kv_cache_spec
                self.backend = DummyBackend()
                self.layer_names = ["layer_0"]

        monkeypatch.setattr(
            self.runner,
            "iter_kv_cache_attn_groups",
            lambda: iter([DummyGroup()]),
        )
        self.runner.runner_only_attn_layers = set()

        page_bytes = kv_cache_spec.page_size_bytes
        raw = torch.zeros(page_bytes * 64, dtype=torch.uint8)

        result = self.runner._reshape_kv_cache_tensors(
            kv_cache_config=MagicMock(),
            kv_cache_raw_tensors={"layer_0": raw},
            kernel_block_sizes=[kv_cache_spec.block_size],
        )

        assert "layer_0" in result
        assert result["layer_0"].dtype == torch.float16

    def test_reshape_skips_runner_only_layers(self, monkeypatch):
        kv_cache_spec = AttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float16,
        )

        class DummyBackend:
            call_count = 0

            def reshape_kv_cache(self, raw_tensor, num_blocks, kv_cache_spec, **kwargs):
                DummyBackend.call_count += 1
                return torch.ones(1)

        class DummyGroup:
            def __init__(self):
                self.kv_cache_spec = kv_cache_spec
                self.backend = DummyBackend()
                self.layer_names = ["skip_me", "keep_me"]

        monkeypatch.setattr(
            self.runner,
            "iter_kv_cache_attn_groups",
            lambda: iter([DummyGroup()]),
        )
        self.runner.runner_only_attn_layers = {"skip_me"}
        page_bytes = kv_cache_spec.page_size_bytes
        raw_tensors = {
            "skip_me": torch.zeros(page_bytes * 2, dtype=torch.uint8),
            "keep_me": torch.zeros(page_bytes * 2, dtype=torch.uint8),
        }

        result = self.runner._reshape_kv_cache_tensors(
            kv_cache_config=MagicMock(),
            kv_cache_raw_tensors=raw_tensors,
            kernel_block_sizes=[2],
        )

        assert "keep_me" in result
        assert "skip_me" not in result
        assert DummyBackend.call_count == 1

    def test_reshape_invokes_hybrid_layout_when_mixed_cache_types(self, monkeypatch):
        kv_cache_spec = AttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float16,
        )
        update_called = {"value": False}

        def _mock_update(kv_caches):
            update_called["value"] = True

        monkeypatch.setattr(
            self.runner,
            "_update_hybrid_attention_mamba_layout",
            _mock_update,
        )
        self.runner.runner_only_attn_layers = set()

        class TensorBackend:
            def reshape_kv_cache(self, raw_tensor, num_blocks, spec, **kwargs):
                return torch.ones(2, 2, dtype=torch.float16)

        class TupleBackend:
            def reshape_kv_cache(self, raw_tensor, num_blocks, spec, **kwargs):
                return (
                    torch.ones(2, 4, 2, dtype=torch.float16),
                    torch.ones(2, 4, 2, dtype=torch.float16),
                )

        class Group:
            def __init__(self, name, backend):
                self.kv_cache_spec = kv_cache_spec
                self.backend = backend
                self.layer_names = [name]

        groups = [
            Group("attn_layer", TensorBackend()),
            Group("mamba_layer", TupleBackend()),
        ]
        monkeypatch.setattr(
            self.runner,
            "iter_kv_cache_attn_groups",
            lambda: iter(groups),
        )
        page_bytes = kv_cache_spec.page_size_bytes
        raw_tensors = {
            "attn_layer": torch.zeros(page_bytes * 2, dtype=torch.uint8),
            "mamba_layer": torch.zeros(page_bytes * 2, dtype=torch.uint8),
        }

        self.runner._reshape_kv_cache_tensors(
            kv_cache_config=MagicMock(),
            kv_cache_raw_tensors=raw_tensors,
            kernel_block_sizes=[2],
        )
        assert update_called["value"] is True
