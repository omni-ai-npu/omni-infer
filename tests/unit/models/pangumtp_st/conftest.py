# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Fixtures for MTP integration tests.

Strategy (following test_deepseek_mtp.py / test_gpt_oss_v1.py patterns):
  1. Minimal module-level stubs so ``import omni.v1.models.pangu.pangu_ultra_moe_mtp`` succeeds.
  2. ``patch.object`` to replace RMSNorm (→ avoids CustomOp chain) and
     OpenPanguDecoderLayer (→ DecoderLayerStub) inside the imported module.
  3. Call the **real** OpenPanguMultiTokenPredictorLayer.__init__ and .forward() —
     only the heavy NPU deps are swapped out.
"""

import sys
import types
import os
import socket
import pytest
import torch
from types import SimpleNamespace
from transformers import PretrainedConfig
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from unittest.mock import patch

# ==============================================================================
# Module-level stubs (minimal — just enough for "import pangu_ultra_moe_mtp")
# ==============================================================================

class _StubLoader(Loader):
    def create_module(self, spec): return None
    def exec_module(self, module): pass

_LOADER = _StubLoader()

def _real_class(name, **attrs):
    d = dict(attrs); d.setdefault('__module__', 'stub')
    # Many stubbed classes are used as generics: ClassName[Metadata].
    d.setdefault('__class_getitem__', classmethod(lambda cls, item: cls))
    return type(name, (), d)

def _real_mod(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []; m.__file__ = None
    m.__spec__ = ModuleSpec(name, _LOADER, origin='stub')
    for k, v in attrs.items(): setattr(m, k, v)
    return m

# -- torch_npu ---------------------------------------------------------------
for _tn in ["torch_npu", "torch_npu._C", "torch_npu.npu", "torch_npu.utils",
            "torch_npu.optim", "torch_npu.streams", "torch_npu.random",
            "torch_npu._C._ptc", "torchair"]:
    if _tn not in sys.modules:
        sys.modules[_tn] = _real_mod(_tn)

# -- torch.npu (many omni_npu modules access it) ----------------------------
import torch as _torch

class _AutoNPUModule(types.ModuleType):
    def __init__(self):
        super().__init__("torch.npu"); self.__path__ = []; self.__file__ = None
        self.__spec__ = ModuleSpec("torch.npu", _LOADER, origin="stub")
        self.config = types.ModuleType("torch.npu.config")
        self.config.allow_internal_format = True
        self.is_available = lambda: False
        self.device_count = lambda: 0
        for _cls in ["NPUGraph", "Stream", "Event", "ExternalEvent"]:
            setattr(self, _cls, type(_cls, (), {}))
    def __getattr__(self, name):
        if name.startswith("_"): raise AttributeError(name)
        sub = types.ModuleType(f"torch.npu.{name}")
        setattr(self, name, sub); return sub

if not hasattr(_torch, "npu"):
    _torch.npu = _AutoNPUModule()

# -- Missing vLLM 0.12.x modules ---------------------------------------------
# vLLM 0.14 provides all needed modules — no stubs needed.
# (vllm.attention, vllm.entrypoints.openai.protocol, etc. all exist.)

# -- omni_npu internals -------------------------------------------------------
# model_extra_config only — parsers load fine now that vLLM 0.14 protocol stub
# provides all needed classes (DeltaMessage, ChatCompletionRequest, etc.).
# Do NOT stub omni.v1.parsers or it breaks existing parser tests.

# model_extra_config patch — fixture-level, NOT module-level.
# Must NOT replace sys.modules["...loader"] because other tests import
# parse_hf_config etc. from the real module.
def _make_model_extra_ns():
    return SimpleNamespace(
        parall_config=SimpleNamespace(ena_dp_lmhead_parallel=False, ena_seq_parallel=False,
            ena_context_parallel=False, enable_flashcomm2=False, sharded_o_proj=False),
        operator_opt_config=SimpleNamespace(use_noncontiguous_kv=True, merge_q_kv_conv=False,
            moe_comm_strategy="", optimize_first_chunk=False, use_mome_inplace_update=False,
            use_aicpu_fa_tiling=False, split_q_up_in_multistream=False,
            router_gating_in_fp32=False, disable_npu_top_k_top_p_sample=False,
            use_topk_topp_stream=False, num_extra_reserved_blocks=0,
            enable_prefill_mla_absorb_pa=False, enable_kv_rmsnorm_rope_cache=False,
            decode_moe_dispatch_combine=False, kv_nz=False, enable_super_kernel=False,
            enable_kv_rms_norm_rope_cache=False, enable_mlaprolog=False,
            enable_unpad_mla_prefill=False, disable_kv_cache_manager=False,
            use_omni_cache=False),
    )

# ==============================================================================
# Regular fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(42); yield

@pytest.fixture(scope="session")
def hf_config():
    return _make_hf_config()

# Alias — test files use 'minimal_config'
@pytest.fixture(scope="session")
def minimal_config():
    return _make_hf_config()

def _make_hf_config():
    return PretrainedConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, vocab_size=256, max_position_embeddings=512,
        rms_norm_eps=1e-6, num_hidden_layers=2, num_nextn_predict_layers=2,
        rope_theta=10000.0,
    )

@pytest.fixture
def vllm_config(hf_config):
    draft_cfg = SimpleNamespace(hf_config=hf_config)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        speculative_config=SimpleNamespace(method="openpangu_mtp",
            num_speculative_tokens=2, draft_model_config=draft_cfg),
        quant_config=None,
        compilation_config=SimpleNamespace(mode="DISABLED",
            static_forward_context={}, custom_ops=[]),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1),
        kv_transfer_config=None,
    )

@pytest.fixture
def sample_batch(hf_config):
    B, H = 4, hf_config.hidden_size
    return {
        "input_ids": torch.tensor([10, 20, 30, 40], dtype=torch.int64),
        "positions": torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        "hidden_states": torch.arange(B * H, dtype=torch.float32).view(B, H) * 0.01,
        "inputs_embeds": torch.arange(B * H, dtype=torch.float32).view(B, H) * 0.01,
    }

# ==============================================================================
# Import the REAL module.  @support_torch_compile calls get_current_vllm_config()
# at class-definition time — make it a no-op before any import runs.
# ==============================================================================
from vllm.compilation.decorators import support_torch_compile as _orig_stc
def _noop_stc(cls=None, **kwargs):
    return cls if cls is not None else lambda c: c

# Patch in vllm AND in omni_npu (both reference it)
import vllm.compilation.decorators as _stc_mod
_stc_mod.support_torch_compile = _noop_stc

import omni.v1.models.pangu.pangu_ultra_moe_mtp as _mtp_mod

# ==============================================================================
# Lightweight fakes (same pattern as test_deepseek_mtp.py)
# ==============================================================================

class _FakeRMSNorm(torch.nn.Module):
    """Avoids vLLM RMSNorm → CustomOp → get_current_vllm_config chain."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(dim).normal_())
        self.eps = eps
    def forward(self, x, residual=None):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True))
        out = (x / (rms + self.eps)) * self.weight
        if residual is not None:
            out = out + residual
            return out, residual
        return out

class _FakeVocabParallelEmbedding(torch.nn.Module):
    """Avoids VocabParallelEmbedding → CustomOp chain."""
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings  # needed by LogitsProcessor
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim).normal_())
    def forward(self, input_ids, **kwargs):
        return torch.nn.functional.embedding(input_ids, self.weight)

class _FakeParallelLMHead(_FakeVocabParallelEmbedding):
    """Avoids ParallelLMHead → CustomOp chain."""
    pass

class _FakeLogitsProcessor(torch.nn.Module):
    """Avoids LogitsProcessor chain if needed."""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
    def forward(self, head, hidden_states):
        return torch.nn.functional.linear(hidden_states, head.weight)

# ==============================================================================
# MTP fixtures — patch.object to replace CustomOp subclasses + DecoderLayer
# ==============================================================================

from .decoder_layer_stub import DecoderLayerStub

def _make_vllm_cfg_for_mtp(hf_config):
    draft_cfg = SimpleNamespace(hf_config=hf_config)
    return SimpleNamespace(
        quant_config=None,
        compilation_config=SimpleNamespace(mode="DISABLED",
            static_forward_context={}, custom_ops=[]),
        model_config=SimpleNamespace(hf_config=hf_config),
        speculative_config=SimpleNamespace(method="openpangu_mtp",
            num_speculative_tokens=2, draft_model_config=draft_cfg),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1),
        kv_transfer_config=None,
    )

# Collect patches that all MTP fixtures share
def _mtp_patches():
    return [
        patch.object(_mtp_mod, "RMSNorm", _FakeRMSNorm),
        patch.object(_mtp_mod, "ParallelLMHead", _FakeParallelLMHead),
        patch.object(_mtp_mod, "VocabParallelEmbedding", _FakeVocabParallelEmbedding),
        patch.object(_mtp_mod, "LogitsProcessor", _FakeLogitsProcessor),
        patch.object(_mtp_mod, "OpenPanguDecoderLayer", DecoderLayerStub),
        patch.object(_mtp_mod, "model_extra_config", _make_model_extra_ns()),
        patch("omni.v1.models.pangu.pangu_ultra_moe_mtp.maybe_prefix",
              side_effect=lambda p, s: f"{p}.{s}"),
    ]

@pytest.fixture
def mtp_layer(hf_config):
    """Real OpenPanguMultiTokenPredictorLayer — CustomOp subclasses patched."""
    vllm_cfg = _make_vllm_cfg_for_mtp(hf_config)
    patches = _mtp_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        return _mtp_mod.OpenPanguMultiTokenPredictorLayer(
            vllm_config=vllm_cfg, prefix="model.layers.10")

@pytest.fixture
def mtp_predictor(hf_config):
    """Real OpenPanguMultiTokenPredictor — CustomOp subclasses patched."""
    vllm_cfg = _make_vllm_cfg_for_mtp(hf_config)
    patches = _mtp_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        return _mtp_mod.OpenPanguMultiTokenPredictor(
            vllm_config=vllm_cfg, prefix="model")

@pytest.fixture
def mtp_model(hf_config):
    """Real OpenPanguMTP — CustomOp subclasses patched."""
    vllm_cfg = _make_vllm_cfg_for_mtp(hf_config)
    patches = _mtp_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        return _mtp_mod.OpenPanguMTP(vllm_config=vllm_cfg, prefix="")

@pytest.fixture
def draft_target_models(hf_config):
    """Two OpenPanguMTP instances with different random weights."""
    vllm_cfg = _make_vllm_cfg_for_mtp(hf_config)
    patches = _mtp_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        torch.manual_seed(42)
        draft = _mtp_mod.OpenPanguMTP(vllm_config=vllm_cfg, prefix="")
        torch.manual_seed(999)
        target = _mtp_mod.OpenPanguMTP(vllm_config=vllm_cfg, prefix="")
    return draft, target
