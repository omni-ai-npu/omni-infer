# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""TestMTPModel — a pure-PyTorch model mirroring OpenPanguMTP architecture.

Zero omni_npu / vllm / torch_npu imports.
Designed for MTP integration testing on CPU.

Architecture (1:1 mapping to OpenPanguMTP):
  TestMTPModel
    └── model: TestMTPPredictor
          ├── embed_tokens: nn.Embedding
          ├── layers: ModuleDict
          │     └── TestMTPLayer[i]
          │           ├── enorm: SimpleRMSNorm
          │           ├── hnorm: SimpleRMSNorm
          │           ├── eh_proj: nn.Linear
          │           ├── shared_head: TestMTPSharedHead
          │           │     ├── norm: SimpleRMSNorm
          │           │     └── head: nn.Linear
          │           └── mtp_block: DecoderLayerStub
          │                 ├── self_attn: SimpleAttention (GQA + RoPE)
          │                 ├── mlp: SwiGLU MLP
          │                 ├── input_layernorm: SimpleRMSNorm
          │                 └── post_attention_layernorm: SimpleRMSNorm
          └── logits_processor: (inlined in compute_logits)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ==============================================================================
# Configuration
# ==============================================================================

class TestMTPConfig:
    """Tiny model config matching OpenPanguMTP requirements."""
    hidden_size: int = 64
    num_attention_heads: int = 4
    num_key_value_heads: int = 2          # GQA
    intermediate_size: int = 128
    vocab_size: int = 256
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    num_hidden_layers: int = 2            # main model layers
    num_nextn_predict_layers: int = 2     # MTP layers
    head_dim: int = 16                    # hidden_size // num_heads
    rope_theta: float = 10000.0


# ==============================================================================
# Base components
# ==============================================================================

class SimpleRMSNorm(nn.Module):
    """Equivalent to vLLM RMSNorm — no external dependency."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # Random init (not ones) so different RMSNorm instances are distinguishable
        self.weight = nn.Parameter(torch.empty(hidden_size).normal_())
        self.eps = eps

    def forward(self, x, residual=None):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True))
        out = (x / (rms + self.eps)) * self.weight
        if residual is not None:
            out = out + residual
            return out, residual
        return out


class SimpleRotaryEmbedding(nn.Module):
    """Equivalent to vLLM RotaryEmbedding — no external dependency."""

    def __init__(self, head_dim, max_position_embeddings, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(pos, freqs)
        self.register_buffer("cos_cached", torch.cos(freqs).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", torch.sin(freqs).unsqueeze(0).unsqueeze(0))

    def get_cos_sin(self, positions: torch.Tensor):
        """Returns (cos, sin) each of shape (bsz, 1, 1, head_dim//2).

        RoPE uses per-pair rotation: cos/sin are applied to half-dimension chunks.
        _apply_rotary splits x into (x1, x2) each of size head_dim//2, which
        matches the last dimension of the returned cos/sin tensors.
        """
        bsz = positions.numel()
        cos = self.cos_cached[:, :, positions, :]   # [1, 1, B, head_dim//2]
        sin = self.sin_cached[:, :, positions, :]
        half = self.head_dim // 2
        return cos.reshape(bsz, 1, 1, half), sin.reshape(bsz, 1, 1, half)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ==============================================================================
# Decoder layer components
# ==============================================================================

class SimpleAttention(nn.Module):
    """Standard Multi-Head Attention with GQA and RoPE."""

    def __init__(self, config: TestMTPConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = SimpleRotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self, hidden_states, cos, sin):
        B, seq_len, _ = hidden_states.shape
        N, K, D = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, seq_len, N, D)
        k = self.k_proj(hidden_states).view(B, seq_len, K, D)
        v = self.v_proj(hidden_states).view(B, seq_len, K, D)

        # cos/sin are [B, 1, 1, half]; keep 4D to broadcast correctly with q/k [B, seq, heads, half]
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k = k.repeat_interleave(N // K, dim=1)
        v = v.repeat_interleave(N // K, dim=1)

        scale = D ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.o_proj(out)


class DecoderLayerStub(nn.Module):
    """
    Equivalent to OpenPanguDecoderLayer.forward_normal().

    Pre-Norm Transformer Block:
      input_layernorm → self_attn → post_attention_layernorm → mlp

    Signature matches production code:
      forward(hidden_states, cos, sin, residual) → (hidden_states, residual)
    """

    # Match OpenPanguDecoderLayer.__init__(config, prefix, vllm_config) signature
    def __init__(self, config, prefix: str = "", vllm_config=None):
        super().__init__()
        # Adapt: PretrainedConfig doesn't have head_dim
        if not hasattr(config, 'head_dim'):
            config.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = SimpleAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )
        self.input_layernorm = SimpleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = SimpleRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, residual=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states.unsqueeze(1), cos, sin).squeeze(1)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ==============================================================================
# MTP-specific components (1:1 mapping to OpenPanguMTP)
# ==============================================================================

class TestMTPSharedHead(nn.Module):
    """Equivalent to OpenPanguMTP.SharedHead: RMSNorm + LM Head."""

    def __init__(self, config: TestMTPConfig):
        super().__init__()
        self.norm = SimpleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.norm(hidden_states)


class TestMTPLayer(nn.Module):
    """
    1:1 mirror of OpenPanguMultiTokenPredictorLayer.

    Flow:
      enorm(embeds) + hnorm(prev_hidden) → eh_proj(concat) → mtp_block → +residual
    """

    def __init__(self, config: TestMTPConfig):
        super().__init__()
        self.config = config
        self.enorm = SimpleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = SimpleRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = TestMTPSharedHead(config)
        self.mtp_block = DecoderLayerStub(config)

    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds, spec_step_index=0):
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        cos, sin = self.mtp_block.self_attn.rotary_emb.get_cos_sin(positions)
        hidden_states, residual = self.mtp_block(hidden_states, cos, sin, residual=None)
        hidden_states = residual + hidden_states
        return hidden_states


class TestMTPPredictor(nn.Module):
    """
    1:1 mirror of OpenPanguMultiTokenPredictor.

    Includes spec_step_idx % num_mtp_layers cycling and
    wrapped_layers (ACLGraph) placeholder.
    """

    def __init__(self, config: TestMTPConfig):
        super().__init__()
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleDict({
            str(idx): TestMTPLayer(config)
            for idx in range(
                self.mtp_start_layer_idx,
                self.mtp_start_layer_idx + self.num_mtp_layers,
            )
        })
        self.wrapped_layers = None  # ACLGraph placeholder

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds=None, spec_step_idx=0):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        current_step_idx = spec_step_idx % self.num_mtp_layers

        if self.wrapped_layers is not None:
            layer = self.wrapped_layers[
                str(self.mtp_start_layer_idx + current_step_idx)]
        else:
            layer = self.layers[
                str(self.mtp_start_layer_idx + current_step_idx)]

        return layer(input_ids, positions, previous_hidden_states,
                     inputs_embeds, current_step_idx)

    def compute_logits(self, hidden_states, spec_step_idx=0):
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[
            str(self.mtp_start_layer_idx + current_step_idx)]
        return mtp_layer.shared_head.head(
            mtp_layer.shared_head(hidden_states))


class TestMTPModel(nn.Module):
    """
    1:1 mirror of OpenPanguMTP — top-level wrapper.
    """

    def __init__(self, config: TestMTPConfig):
        super().__init__()
        self.config = config
        self.model = TestMTPPredictor(config)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, hidden_states,
                inputs_embeds=None, spec_step_idx=0):
        return self.model(input_ids, positions, hidden_states,
                         inputs_embeds, spec_step_idx)

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.model.compute_logits(hidden_states, spec_step_idx)
