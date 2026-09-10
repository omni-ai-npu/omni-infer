# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.mla import MLAModules

from omni_npu.layers.attention import npu_mla_wrapper as npu_mla_wrapper_module
from omni_npu.layers.attention.npu_mla_wrapper import NPUMultiHeadLatentAttentionWrapper


pytestmark = pytest.mark.unit


class _FakeMLAAttn:
    def __init__(self, layer_name: str = "layer0"):
        self.layer_name = layer_name
        self.kv_cache = ["kv_cache_0"]
        self.called_with = None

    def __call__(self, q, kv_c_normed, k_pe, output_shape):
        self.called_with = (q, kv_c_normed, k_pe, output_shape)
        return torch.zeros(output_shape, dtype=q.dtype, device=q.device)


def _build_wrapper(q_lora_rank):
    mla_modules = MLAModules(
        kv_a_layernorm=MagicMock(side_effect=lambda x: x),
        kv_b_proj=MagicMock(),
        rotary_emb=None,
        o_proj=MagicMock(side_effect=lambda x: (x + 1,)),
        fused_qkv_a_proj=MagicMock(),
        kv_a_proj_with_mqa=MagicMock(),
        q_a_layernorm=MagicMock(side_effect=lambda x: x),
        q_b_proj=MagicMock(),
        q_proj=MagicMock(),
        indexer=None,
        indexer_rotary_emb=None,
        is_sparse=False,
        topk_indices_buffer=None,
    )
    with patch(
        "vllm.model_executor.layers.mla.MLAAttention",
        return_value=_FakeMLAAttn(),
    ):
        wrapper = NPUMultiHeadLatentAttentionWrapper(
            hidden_size=16,
            num_heads=2,
            scale=1.0,
            qk_nope_head_dim=3,
            qk_rope_head_dim=2,
            v_head_dim=7,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=6,
            mla_modules=mla_modules,
        )
    return wrapper


def test_forward_oot_with_q_lora_rank(default_vllm_config):
    wrapper = _build_wrapper(q_lora_rank=4)

    tokens = 3
    hidden_states = torch.randn(tokens, 8)
    positions = torch.arange(tokens, dtype=torch.long)

    qkv_dim = wrapper.q_lora_rank + wrapper.kv_lora_rank + wrapper.qk_rope_head_dim
    qkv_lora = torch.randn(tokens, qkv_dim)
    wrapper.fused_qkv_a_proj.return_value = (qkv_lora,)
    wrapper.q_b_proj.return_value = (
        torch.ones(tokens, wrapper.num_heads * wrapper.qk_head_dim),
    )

    llama_4_scaling = torch.tensor(2.0)
    output = wrapper.forward_oot(positions, hidden_states, llama_4_scaling)

    wrapper.fused_qkv_a_proj.assert_called_once_with(hidden_states)
    wrapper.q_a_layernorm.assert_called_once()
    wrapper.q_b_proj.assert_called_once()
    wrapper.kv_a_layernorm.assert_called_once()

    q_used = wrapper.mla_attn.called_with[0]
    assert torch.allclose(q_used, torch.full_like(q_used, 2.0))
    assert output.shape == (tokens, wrapper.num_heads * wrapper.v_head_dim)
    assert torch.allclose(output, torch.ones_like(output))


def test_forward_oot_without_q_lora_rank(default_vllm_config):
    wrapper = _build_wrapper(q_lora_rank=None)

    tokens = 2
    hidden_states = torch.randn(tokens, 8)
    positions = torch.arange(tokens, dtype=torch.long)

    kv_dim = wrapper.kv_lora_rank + wrapper.qk_rope_head_dim
    wrapper.kv_a_proj_with_mqa.return_value = (torch.randn(tokens, kv_dim),)
    wrapper.q_proj.return_value = (
        torch.randn(tokens, wrapper.num_heads * wrapper.qk_head_dim),
    )

    output = wrapper.forward_oot(positions, hidden_states)

    wrapper.kv_a_proj_with_mqa.assert_called_once_with(hidden_states)
    wrapper.q_proj.assert_called_once_with(hidden_states)
    wrapper.fused_qkv_a_proj.assert_not_called()
    assert output.shape == (tokens, wrapper.num_heads * wrapper.v_head_dim)


def test_forward_oot_sparse_indexer_uses_context(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch
):
    wrapper = _build_wrapper(q_lora_rank=4)

    tokens = 1
    hidden_states = torch.randn(tokens, 8)
    positions = torch.arange(tokens, dtype=torch.long)

    qkv_dim = wrapper.q_lora_rank + wrapper.kv_lora_rank + wrapper.qk_rope_head_dim
    wrapper.fused_qkv_a_proj.return_value = (torch.randn(tokens, qkv_dim),)
    wrapper.q_b_proj.return_value = (
        torch.randn(tokens, wrapper.num_heads * wrapper.qk_head_dim),
    )

    wrapper.indexer = MagicMock(return_value=torch.zeros(1, dtype=torch.int64))
    wrapper.is_sparse = True
    wrapper.indexer_rope_emb = object()
    wrapper.mla_attn.layer_name = "layer0"
    wrapper.mla_attn.kv_cache = "kv_cache_0"

    monkeypatch.setattr(
        npu_mla_wrapper_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer0": "meta"}, virtual_engine=0),
    )

    wrapper.forward_oot(positions, hidden_states)

    wrapper.indexer.assert_called_once()
    args = wrapper.indexer.call_args.args
    assert args[0] is hidden_states
    assert args[1] is not None
    assert args[2] is positions
    assert args[3] is wrapper.indexer_rope_emb
    assert args[4] == "kv_cache_0"
    assert args[5] == "meta"

