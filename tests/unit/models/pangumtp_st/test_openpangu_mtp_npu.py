# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU smoke tests for MTP + real OpenPanguDecoderLayer.

Verifies that MTP integrates correctly with real NPU attention backends
(MLA / DSA / MOME).  Architecture logic is covered by CPU MinimalMTP tests;
these tests only confirm that the real components don't crash.

Requires: NPU hardware + CANN environment sourced + torch_npu + hccl.
"""

import os
import socket
import pytest
import torch
from types import SimpleNamespace
from transformers import PretrainedConfig

# ---------------------------------------------------------------------------
# NPU availability guard
# ---------------------------------------------------------------------------
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except (ImportError, AttributeError):
    _HAS_NPU = False

pytestmark = pytest.mark.skipif(not _HAS_NPU, reason="requires NPU hardware")


# ==============================================================================
# vLLM 0.14 workarounds for NPU tests (stub modules from conftest interfere)
# ==============================================================================
import sys as _sys

# 1. RoPE _compute_inv_freq CPU/NPU device mismatch
_dsr_key = "vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope"
if _dsr_key in _sys.modules:
    _dsr = _sys.modules[_dsr_key]
    def _patched_cif(self, scaling_factor):
        dim = self.rotary_dim
        inv_freq = 1.0 / (scaling_factor * self.base ** (
            torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to("npu") if torch.npu.is_available() else inv_freq
    _dsr.DeepseekScalingRotaryEmbedding._compute_inv_freq = _patched_cif


# ==============================================================================
# Helpers
# ==============================================================================

def _make_vllm_wrapper_config():
    """Create a SimpleNamespace for set_current_vllm_config with all fields
    needed by vLLM 0.14 code paths."""
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            mode="DISABLED", static_forward_context={},
            custom_ops=["none"], disabled_custom_ops=set(),
        ),
        model_config=SimpleNamespace(
            device=torch.device("npu"), max_model_len=512,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1,
            pipeline_parallel_size=1, enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=8, max_num_batched_tokens=2048,
        ),
        cache_config=SimpleNamespace(
            block_size=16, enable_prefix_caching=False,
            calculate_kv_scales=False,
        ),
        attention_config=SimpleNamespace(backend=None),
    )

def _init_hccl_dist(monkeypatch):
    """Single-rank hccl distributed init.

    Set the distributed env vars via monkeypatch.setenv (not os.environ.setdefault)
    so they are restored at test teardown. Otherwise RANK/WORLD_SIZE leak into the
    process env and later torchrun-only tests (e.g. test_communicator's
    TestNPUCommunicatorMultiDevice) mistake a leaked WORLD_SIZE=1 for a real
    distributed launch, self-init a single-rank group, and crash under reordered runs.
    """
    if torch.distributed.is_initialized():
        return
    torch.npu.set_device(0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(port))

    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    import torch.distributed as dist
    dist.init_process_group(backend="hccl", rank=0, world_size=1)
    init_distributed_environment(backend="hccl", world_size=1, rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1)


def _make_npu_hf_config():
    """Minimal HF config with MLA fields."""
    return PretrainedConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, vocab_size=256, max_position_embeddings=512,
        rms_norm_eps=1e-6, num_hidden_layers=2, num_nextn_predict_layers=2,
        rope_theta=10000.0,
        qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16,
        q_lora_rank=24, kv_lora_rank=16, rope_interleaved=True,
        index_topk=0, sliding_window_list=[512, 512], swa_layers=[0, 1],
        param_sink_number=0, first_k_dense_replace=2, n_routed_experts=None,
        hidden_act="silu", use_mome=False, is_moe=False,
        rope_parameters={"rope_type": "default"}, index_n_heads=1,
        index_head_dim=16, dsa_layers=[],
    )


def _make_npu_vllm_cfg(hf_config=None):
    """SimpleNamespace VllmConfig."""
    if hf_config is None:
        hf_config = _make_npu_hf_config()
    draft_model_cfg = SimpleNamespace(hf_config=hf_config)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, max_model_len=512),
        speculative_config=SimpleNamespace(
            method="openpangu_mtp", num_speculative_tokens=2,
            draft_model_config=draft_model_cfg,
        ),
        quant_config=None,
        compilation_config=SimpleNamespace(
            mode="DISABLED", static_forward_context={}, custom_ops=["none"],
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8, enable_chunked_prefill=False),
        cache_config=SimpleNamespace(block_size=16, gpu_memory_utilization=0.3,
                                      swap_space=0, cache_dtype="auto",
                                      enable_prefix_caching=False,
                                      calculate_kv_scales=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1,
            enable_expert_parallel=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        kv_transfer_config=None,
    )


def _patch_model_extra_config(monkeypatch):
    """Patch model_extra_config so OpenPanguMTP init doesn't need real config files."""
    ns = SimpleNamespace(
        parall_config=SimpleNamespace(
            ena_dp_lmhead_parallel=False, ena_seq_parallel=False,
            ena_context_parallel=False, enable_flashcomm2=False,
            sharded_o_proj=False,
        ),
        operator_opt_config=SimpleNamespace(
            use_noncontiguous_kv=True, merge_q_kv_conv=False,
            moe_comm_strategy="", optimize_first_chunk=False,
            use_mome_inplace_update=False, use_aicpu_fa_tiling=False,
            split_q_up_in_multistream=False, router_gating_in_fp32=False,
            disable_npu_top_k_top_p_sample=False, use_topk_topp_stream=False,
            num_extra_reserved_blocks=0, enable_prefill_mla_absorb_pa=False,
            enable_kv_rmsnorm_rope_cache=False, decode_moe_dispatch_combine=False,
            kv_nz=False, enable_super_kernel=False,
        ),
    )
    for path in [
        "omni.v1.models.pangu.pangu_ultra_moe_mtp.model_extra_config",
        "omni.v1.models.pangu.pangu_ultra_moe.model_extra_config",
        "omni.v1.layers.attention.npu_pangu.model_extra_config",
        "omni.model_config.config_loader.loader.model_extra_config",
    ]:
        try:
            monkeypatch.setattr(path, ns)
        except AttributeError:
            pass


# ==============================================================================
# NPU-01: Init
# ==============================================================================

class TestNPUMTPInit:
    """Verify OpenPanguMultiTokenPredictorLayer can be instantiated."""

    def test_layer_init(self, monkeypatch, default_vllm_config):
        """NPU-01: Import and instantiation succeeds on NPU."""
        _init_hccl_dist(monkeypatch)
        _patch_model_extra_config(monkeypatch)

        # vLLM 0.14: yarn_linear_ramp_mask returns CPU tensor but RoPE uses NPU
        import vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope as _dsr
        _orig_cif = _dsr.DeepseekScalingRotaryEmbedding._compute_inv_freq
        def _patched_cif(self, scaling_factor):
            r = _orig_cif(self, scaling_factor)
            return r.to("npu") if r.device.type != "npu" else r
        monkeypatch.setattr(_dsr.DeepseekScalingRotaryEmbedding,
                            "_compute_inv_freq", _patched_cif)

        # Ensure get_current_vllm_config returns a valid config
        # VllmConfig() has model_config=None in vLLM 0.14; provide explicit config
        from vllm.config import set_current_vllm_config
        _wrap = SimpleNamespace(
            compilation_config=SimpleNamespace(mode="DISABLED", static_forward_context={},
                custom_ops=["none"], disabled_custom_ops=set()),
            model_config=SimpleNamespace(device=torch.device("npu"), max_model_len=512),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1,
                enable_expert_parallel=False),
            scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=2048),
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=False,
                calculate_kv_scales=False),
            attention_config=SimpleNamespace(backend=None),
        )
        with set_current_vllm_config(_wrap):
            from omni.v1.models.pangu.pangu_ultra_moe_mtp import \
                OpenPanguMultiTokenPredictorLayer
            from omni.v1.models.pangu.pangu_ultra_moe import OpenPanguDecoderLayer

            vllm_cfg = _make_npu_vllm_cfg()
            layer = OpenPanguMultiTokenPredictorLayer(
            vllm_config=vllm_cfg, prefix="model.layers.10",
        )

        assert hasattr(layer, "mtp_block")
        assert isinstance(layer.mtp_block, OpenPanguDecoderLayer), \
            "NPU test must use real OpenPanguDecoderLayer"


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="class")
def _cleanup_dist():
    yield
    # This module's tests init a real hccl process group + vLLM parallel state via
    # _init_hccl_dist(); tear the vLLM wrappers down fully so nothing leaks into
    # later tests (e.g. test_rl_update_weight seeing stale state / test_communicator
    # hitting EADDRINUSE) under whole-suite / reordered runs. Destroy vLLM's wrappers
    # (destroy_model_parallel + destroy_distributed_environment) rather than the raw
    # torch PG, so no dangling _WORLD reference survives pointing at a dead group.
    try:
        from vllm.distributed import (destroy_distributed_environment,
                                      destroy_model_parallel)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
