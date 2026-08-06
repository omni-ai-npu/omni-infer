# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from omni.vllm_patches.patches.models.qwen import qwen_hybrid_common as hybrid


class TestUniformTypeKvCacheConfig:
    @patch(
        "omni.vllm_patches.patches.models.qwen.qwen_hybrid_common.kv_cache_utils.may_override_num_blocks",
        side_effect=lambda _cfg, n: n,
    )
    def test_builds_per_layer_tensors(self, _override):
        layer_spec = SimpleNamespace(page_size_bytes=128)
        uniform_spec = MagicMock(spec=UniformTypeKVCacheSpecs)
        uniform_spec.page_size_bytes = 128
        uniform_spec.kv_cache_specs = {"layer0": layer_spec, "layer1": layer_spec}
        groups = [KVCacheGroupSpec(["layer0", "layer1"], uniform_spec)]
        vllm_config = MagicMock()

        cfg = hybrid.uniform_type_kv_cache_config(
            vllm_config, groups, available_memory=1280
        )

        assert cfg.num_blocks == 10
        assert len(cfg.kv_cache_tensors) == 2
        assert cfg.kv_cache_tensors[0].size == 128 * 10
        assert cfg.kv_cache_tensors[0].shared_by == ["layer0"]


class TestHybridKvCacheHelpers:
    def test_attention_block_size_alignment_rounds_and_rejects(self):
        assert hybrid.get_supported_attention_block_size(1) == 128
        assert hybrid.get_supported_attention_block_size(128) == 128
        assert hybrid.get_supported_attention_block_size(129) == 256
        with pytest.raises(ValueError, match="supported FIA block size"):
            hybrid.get_supported_attention_block_size(1025)

    def test_align_hybrid_groups_pads_mamba_to_attention_page(self):
        attention = AttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
        )
        mamba = MambaSpec(
            block_size=1,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=None,
        )
        groups = hybrid.align_hybrid_kv_cache_groups_to_attention(
            [
                KVCacheGroupSpec(["attn"], attention),
                KVCacheGroupSpec(["mamba"], mamba),
            ]
        )

        assert groups[1].kv_cache_spec.page_size_padded == attention.page_size_bytes

    def test_align_hybrid_groups_rejects_oversized_mamba_page(self):
        attention = AttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
        )
        mamba = MambaSpec(
            block_size=1,
            shapes=((1024, 1024),),
            dtypes=(torch.float32,),
            page_size_padded=None,
        )
        with pytest.raises(ValueError, match="must not exceed"):
            hybrid.align_hybrid_kv_cache_groups_to_attention(
                [
                    KVCacheGroupSpec(["attn"], attention),
                    KVCacheGroupSpec(["mamba"], mamba),
                ]
            )

    def test_build_hybrid_tensors_use_unpadded_mamba_pages(self):
        mamba = MambaSpec(
            block_size=1,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
            page_size_padded=4096,
        )
        groups = [
            KVCacheGroupSpec(["mamba"], mamba),
            KVCacheGroupSpec(["attn"], SimpleNamespace(page_size_bytes=64)),
        ]

        tensors = hybrid.build_hybrid_kv_cache_tensors(groups, num_blocks=3)

        unpadded_mamba = hybrid.get_unpadded_kv_cache_spec(mamba)
        assert tensors[0].size == unpadded_mamba.page_size_bytes * 3
        assert tensors[0].shared_by == ["mamba"]
        assert tensors[1].size == 64 * 3
        assert hybrid.get_hybrid_bytes_per_block(groups) == (
            unpadded_mamba.page_size_bytes + 64
        )

    def test_hybrid_helpers_skip_groups_shorter_than_group_size(self):
        groups = [
            KVCacheGroupSpec(
                ["attn0", "attn1"],
                SimpleNamespace(page_size_bytes=32),
            ),
            KVCacheGroupSpec(["mamba0"], SimpleNamespace(page_size_bytes=64)),
        ]

        tensors = hybrid.build_hybrid_kv_cache_tensors(groups, num_blocks=2)

        assert hybrid.get_hybrid_bytes_per_block(groups) == 32 + 64 + 32
        assert [tensor.shared_by for tensor in tensors] == [
            ["attn0"],
            ["mamba0"],
            ["attn1"],
        ]

    def test_reshape_non_attention_handles_mamba_and_rejects_unknown(self):
        mamba = MambaSpec(
            block_size=1,
            shapes=((2, 2), (1, 4)),
            dtypes=(torch.float32, torch.float16),
        )
        raw = torch.arange(
            mamba.page_size_bytes * 2,
            dtype=torch.int64,
            device="cpu",
        ).to(torch.uint8)

        states = hybrid.reshape_non_attention_kv_cache(raw, mamba, num_blocks=2)

        assert len(states) == 2
        assert states[0].shape == (2, 2, 2)
        assert states[1].shape == (2, 1, 4)
        with pytest.raises(NotImplementedError, match="Unsupported kv_cache_spec"):
            hybrid.reshape_non_attention_kv_cache(raw, MagicMock(), num_blocks=2)


class TestQwenHybridInputBatchPatch:
    def test_apply_uses_local_patch_records_when_parent_has_record(self, monkeypatch):
        class Parent:
            _omni_npu_applied_patches = {
                "may_reinitialize_input_batch": "ParentPatch",
                "other": "ParentPatch",
            }

        class Target(Parent):
            local_marker = object()

        class PatchUnderTest(hybrid.QwenHybridInputBatchPatch):
            _target = Target

        monkeypatch.setattr(hybrid.VLLMPatch, "apply", MagicMock())

        PatchUnderTest.apply()

        assert Target._omni_npu_applied_patches == {"other": "ParentPatch"}
        hybrid.VLLMPatch.apply.assert_called_once()

    def test_check_cpu_offload_disabled_accepts_zero_and_rejects_nonzero(self):
        hybrid.QwenHybridInputBatchPatch._check_cpu_offload_disabled(0)

        with pytest.raises(RuntimeError, match="CPU weight offloading"):
            hybrid.QwenHybridInputBatchPatch._check_cpu_offload_disabled(1)

    def test_may_reinitialize_uses_cache_block_size_for_mamba(self):
        runner = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, cpu_offload_gb=0),
            _init_npu_input_batch=MagicMock(),
        )
        mamba = MambaSpec(
            block_size=8,
            shapes=((2, 2),),
            dtypes=(torch.float32,),
        )
        attention = SimpleNamespace(block_size=32)
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=mamba),
                SimpleNamespace(kv_cache_spec=attention),
            ]
        )

        hybrid.QwenHybridInputBatchPatch.may_reinitialize_input_batch(
            runner, kv_cache_config, [8, 32]
        )

        runner._init_npu_input_batch.assert_called_once_with([16, 32], [16, 32])


class TestKvCacheUtilsPatch:
    def test_empty_groups_returns_minimal_config(self):
        vllm_config = MagicMock()
        cfg = hybrid.QwenHybridKVCacheUtilsPatch.get_kv_cache_config_from_groups(
            vllm_config, [], available_memory=1024
        )
        assert cfg.num_blocks == 1
        assert cfg.kv_cache_tensors == []

    def test_unify_hybrid_kv_cache_specs_is_noop(self):
        spec = {"layer0": MagicMock()}
        # Same as vLLM upstream: in-place side effect only, return value unused.
        assert (
            hybrid.QwenHybridKVCacheUtilsPatch.unify_hybrid_kv_cache_specs(spec)
            is None
        )
        assert "layer0" in spec

    def test_hybrid_path_requires_registered_fn(self):
        vllm_config = MagicMock()
        groups = [
            KVCacheGroupSpec(["a"], MagicMock()),
            KVCacheGroupSpec(["b"], MagicMock()),
        ]
        original = hybrid._hybrid_kv_cache_config_fn
        try:
            hybrid.set_hybrid_kv_cache_config_fn(None)
            with pytest.raises(RuntimeError, match="set_hybrid_kv_cache_config_fn"):
                hybrid.QwenHybridKVCacheUtilsPatch.get_kv_cache_config_from_groups(
                    vllm_config, groups, available_memory=4096
                )
        finally:
            hybrid.set_hybrid_kv_cache_config_fn(original)

    @patch(
        "omni.vllm_patches.patches.models.qwen.qwen_hybrid_common.uniform_type_kv_cache_config",
        return_value="uniform_cfg",
    )
    def test_single_uniform_group_uses_uniform_helper(self, mock_uniform):
        uniform_spec = MagicMock(spec=UniformTypeKVCacheSpecs)
        groups = [KVCacheGroupSpec(["layer0"], uniform_spec)]
        vllm_config = MagicMock()

        out = hybrid.QwenHybridKVCacheUtilsPatch.get_kv_cache_config_from_groups(
            vllm_config, groups, available_memory=2048
        )

        mock_uniform.assert_called_once_with(vllm_config, groups, 2048)
        assert out == "uniform_cfg"


class TestRegistryHelpers:
    def test_register_local_qwen3_next_model(self):
        sentinel = object()
        original = hybrid.ModelRegistry.models.get("Qwen3NextForCausalLM")
        try:
            hybrid.register_local_qwen3_next_model(sentinel)
            entry = hybrid.ModelRegistry.models["Qwen3NextForCausalLM"]
            assert entry.model_cls is sentinel
            assert entry.interfaces is hybrid.QWEN3_NEXT_REGISTRY_MODEL_INFO
        finally:
            if original is None:
                hybrid.ModelRegistry.models.pop("Qwen3NextForCausalLM", None)
            else:
                hybrid.ModelRegistry.models["Qwen3NextForCausalLM"] = original

    @patch.object(hybrid, "register_local_qwen3_next_model")
    @patch.object(hybrid.logger, "info")
    def test_apply_local_qwen3_next_registry_success(
        self, mock_info, mock_register
    ):
        sentinel = object()
        hybrid.apply_local_qwen3_next_registry(
            sentinel,
            success_log="ok",
            failure_log="fail",
        )
        mock_register.assert_called_once_with(sentinel)
        mock_info.assert_called_once_with("ok")

    @patch.object(hybrid, "register_local_qwen3_next_model", side_effect=RuntimeError)
    @patch.object(hybrid.logger, "exception")
    def test_apply_local_qwen3_next_registry_logs_failure(
        self, mock_exc, _register
    ):
        hybrid.apply_local_qwen3_next_registry(
            object(),
            success_log="ok",
            failure_log="fail",
        )
        mock_exc.assert_called_once_with("fail")

    @patch.object(hybrid, "apply_local_qwen3_next_registry")
    def test_register_local_qwen3_next_for_hybrid_patch(
        self, mock_apply
    ):
        hybrid.register_local_qwen3_next_for_hybrid_patch(
            success_log="ok",
            failure_log="fail",
        )
        mock_apply.assert_called_once()
        model_cls = mock_apply.call_args[0][0]
        assert model_cls.__name__ == "Qwen3NextForCausalLM"
