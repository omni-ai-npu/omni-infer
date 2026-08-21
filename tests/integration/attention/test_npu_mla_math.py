# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NPU MLA math correctness tests against a pure PyTorch reference.

What is being tested
--------------------
These tests validate the mathematical correctness of the NPU MLA implementation
used by `NPUDeepseekMLAAttention` by comparing its outputs to a reference
implementation written in pure PyTorch ops.

We cover two core execution modes:
1) Prefill: causal attention over a full prompt for each sequence.
2) Decode: single-token attention against an existing KV-cache (prefix).

Why this test looks "low-level"
-------------------------------
`NPUDeepseekMLAAttention` integrates with vLLM's runtime via forward-context
metadata (slot mapping, block table, prefill/decode metadata) and uses NPU-
specific fused ops to:
  - write normalized+RoPE key/value representations into paged KV-cache
  - run attention with a block_table/slot_mapping indirection

For unit-level math validation, we:
  - construct deterministic synthetic metadata and KV-cache layouts
  - build strided/non-contiguous hidden_states to catch layout assumptions
  - implement an explicit PyTorch reference (einsum/softmax) per scenario
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import pytest
import torch

from transformers import DeepseekV3Config as _DeepseekConfig

from omni_npu.attention.backends.mla import NPUMLAImpl, NPUMLAPrefillBackend
from omni_npu.platform import NPUPlatform
from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention
from omni_npu.v1.layers.linear import (
    ColumnParallelFlashCommLinear,
    RowParallelFlashCommLinear,
)
from vllm.config import CacheConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.v1.attention.backends.mla.prefill.registry import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on environment
    raise RuntimeError("torch_npu must be available for NPU MLA tests") from exc

if not (hasattr(torch, "npu") and torch.npu.device_count() > 0):
    raise RuntimeError("NPU hardware is required for NPU MLA tests")

NPU_AVAILABLE = True


@pytest.fixture(scope="module", autouse=True)
def _register_npu_mla_prefill_backend() -> None:
    """Mirror vLLM's platform initialization for direct module construction."""
    backend = MLAPrefillBackendEnum.FLASH_ATTN
    was_overridden = backend.is_overridden()
    previous_path = backend.get_path()

    NPUPlatform.pre_register_and_update()
    try:
        assert backend.get_class() is NPUMLAPrefillBackend
        yield
    finally:
        if was_overridden:
            register_mla_prefill_backend(backend, previous_path)
        else:
            backend.clear_override()


@pytest.fixture(autouse=True)
def _clear_mla_static_masks() -> None:
    NPUMLAImpl.SHARE_MASK_TRIL_SPARSE = None
    yield
    NPUMLAImpl.SHARE_MASK_TRIL_SPARSE = None


QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
KV_LORA_RANK = 512
Q_LORA_RANK = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
NUM_HEADS = 4
HIDDEN_SIZE = 256
BLOCK_SIZE = 128
MAX_POS = 2048


@dataclass
class DummyPrefillMeta:
    seq_lens: list[int]
    query_cumlens: list[int]
    max_query_len: int
    block_table: torch.Tensor | None = None


@dataclass
class DummyDecodeMeta:
    block_table: torch.Tensor
    seq_lens: list[int]
    query_cumlens: list[int]


@dataclass
class DummyAttnMeta:
    prefill: DummyPrefillMeta | None
    decode: DummyDecodeMeta | None
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_prefills: int
    num_decodes: int
    num_decode_tokens: int
    max_query_len: int = 2


class DummyForwardContext:
    def __init__(self, attn_metadata: DummyAttnMeta):
        self.attn_metadata = attn_metadata
        self.virtual_engine = 0
        self.batch_descriptor = None
        self.no_compile_layers = {}
        self.capturing = False


class DummyCompilationConfig:
    def __init__(self) -> None:
        self.static_forward_context: dict[str, object] = {}


class DummyParallelConfig:
    pipeline_parallel_size = 1


class DummySchedulerConfig:
    max_num_seqs = 8


class DummyModelConfig:
    max_model_len = MAX_POS


class DummyAttentionConfig:
    def __init__(self):
        self.backend = "dummy"
        self.use_non_causal = False


class DummyVllmConfig:
    def __init__(self) -> None:
        self.compilation_config = DummyCompilationConfig()
        self.parallel_config = DummyParallelConfig()
        self.scheduler_config = DummySchedulerConfig()
        self.cache_config = CacheConfig(block_size=BLOCK_SIZE)
        self.model_config = DummyModelConfig()
        self.attention_config = DummyAttentionConfig()
        self.kv_transfer_config = None
        self.speculative_config = None


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # NPU rotary embedding returns cos/sin with extra singleton dims:
    # - common shapes are [T, 1, 1, D] or [T, 1, D]
    # vLLM's `ApplyRotaryEmb.forward_static` expects cos/sin shaped [T, D/2]
    # for GPT-J style.
    #
    # NPURotaryEmbedding builds interleaved cos/sin for GPT-J style (D, not D/2),
    # so we take every other element (even indices) to get the half-dim cache.
    if cos.dim() == 4:
        cos = cos.squeeze(2)
        sin = sin.squeeze(2)
    if cos.dim() == 3:
        cos = cos.squeeze(1)
        sin = sin.squeeze(1)
    if cos.shape[-1] == x.shape[-1]:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    elif cos.shape[-1] * 2 != x.shape[-1]:
        raise ValueError(
            f"Incompatible rope cache shape: cos={tuple(cos.shape)}, x={tuple(x.shape)}"
        )
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    if x.dim() == 2:
        x = x.unsqueeze(1)
        out = ApplyRotaryEmb.forward_static(x, cos, sin, is_neox_style=False).squeeze(1)
        return out
    return ApplyRotaryEmb.forward_static(x, cos, sin, is_neox_style=False)


def _assert_allclose(actual: torch.Tensor,
                     expected: torch.Tensor,
                     name: str,
                     atol: float = 2e-2,
                     rtol: float = 2e-2) -> None:
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    diff = (actual - expected).abs()
    flat_idx = diff.view(-1).argmax().item()
    idx = torch.unravel_index(torch.tensor(flat_idx, device="cpu"), diff.shape)
    actual_val = actual.view(-1)[flat_idx].item()
    expected_val = expected.view(-1)[flat_idx].item()
    raise AssertionError(
        f"{name} mismatch: max_diff={diff.max().item()} idx={tuple(idx)} "
        f"actual={actual_val} expected={expected_val} "
        f"shape={tuple(actual.shape)} dtype={actual.dtype}"
    )


def _build_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    # Positions for prefill are per-token:
    # shape [B*S], with each sequence being [0..S-1].
    positions = torch.arange(seq_len, device=device, dtype=torch.long)
    return positions.repeat(batch_size)


def _get_cos_sin_for_positions(
    rotary_emb: torch.nn.Module, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.flatten()
    seqlen = int(positions.max().item()) + 1

    def _gather_by_positions(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cos.shape[0] == positions.numel():
            return cos, sin
        if cos.shape[0] >= seqlen:
            cos = cos.index_select(0, positions)
            sin = sin.index_select(0, positions)
            return cos, sin
        raise ValueError(
            f"Cannot align rotary cache with positions: "
            f"cache_len={cos.shape[0]}, num_positions={positions.numel()}, seqlen={seqlen}"
        )

    try:
        # Prefer deterministic full-cache path, then gather exact positions.
        cos, sin = rotary_emb.get_cos_sin(seqlen)
        cos, sin = _gather_by_positions(cos, sin)
    except TypeError:
        # Fallback for rotary implementations that only accept position tensors.
        cos, sin = rotary_emb.get_cos_sin(positions)
        cos, sin = _gather_by_positions(cos, sin)
    # Keep cache dim aligned with NPU MLA rope dim.
    # Some rotary impls return half-dim cache [T, D/2].
    if cos.shape[-1] * 2 == QK_ROPE_HEAD_DIM:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    elif cos.shape[-1] != QK_ROPE_HEAD_DIM:
        raise ValueError(
            f"Unexpected rope dim from rotary cache: {cos.shape[-1]} "
            f"(expected {QK_ROPE_HEAD_DIM} or {QK_ROPE_HEAD_DIM // 2})"
        )

    # NPU kernels in npu_mla.py expect 4D cos/sin (B, 1, 1, D).
    if cos.dim() == 2:
        cos = cos.unsqueeze(1).unsqueeze(1)
        sin = sin.unsqueeze(1).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    elif cos.dim() != 4:
        raise ValueError(f"Unexpected rope cache rank: cos={cos.dim()}, sin={sin.dim()}")

    return cos, sin


def _build_query_cumlens(batch_size: int, seq_len: int) -> list[int]:
    # "cumlens" are cumulative lengths in TND layout.
    # For uniform-length sequences this is simply [S, 2S, 3S, ...].
    cumlens = []
    total = 0
    for _ in range(batch_size):
        total += seq_len
        cumlens.append(total)
    return cumlens


def _build_block_table(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    # Deterministic block table with per-sequence contiguous block IDs.
    # shape: [B, ceil(S / BLOCK_SIZE)]
    # value: block ids; block_table[b, j] is the physical block id of the j-th
    #        logical block for sequence b.
    num_blocks_per_seq = math.ceil(seq_len / BLOCK_SIZE)
    block_table = torch.empty(
        (batch_size, num_blocks_per_seq), dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        start = i * num_blocks_per_seq
        block_table[i] = torch.arange(
            start, start + num_blocks_per_seq, dtype=torch.int32, device=device
        )
    return block_table


def _build_slot_mapping(batch_size: int, seq_len: int, block_table: torch.Tensor) -> torch.Tensor:
    # Map every token position to its block slot for prefill writes.
    # shape: [B*S]
    # slot = block_id * BLOCK_SIZE + offset_within_block
    slot_list: list[int] = []
    for i in range(batch_size):
        for pos in range(seq_len):
            block_id = int(block_table[i, pos // BLOCK_SIZE].item())
            slot_list.append(block_id * BLOCK_SIZE + (pos % BLOCK_SIZE))
    return torch.tensor(slot_list, dtype=torch.long, device=block_table.device)


def _build_decode_slot_mapping(
    seq_len: int, block_table: torch.Tensor, device: torch.device
) -> torch.Tensor:
    # Map only the decode token (last position) for each sequence.
    # shape: [B]
    slot_list: list[int] = []
    pos = seq_len - 1
    for i in range(block_table.shape[0]):
        block_id = int(block_table[i, pos // BLOCK_SIZE].item())
        slot_list.append(block_id * BLOCK_SIZE + (pos % BLOCK_SIZE))
    return torch.tensor(slot_list, dtype=torch.long, device=device)


def _init_kv_cache(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # KV caches are shaped per-block; MLA uses separate rope and compressed KV caches.
    # `NPUDeepseekMLAAttention` expects:
    # - kv_cache[0]: [num_blocks, BLOCK_SIZE, KV_LORA_RANK]
    # - kv_cache[1]: [num_blocks, BLOCK_SIZE, QK_ROPE_HEAD_DIM]
    #
    # The module itself unsqueezes dim=2 before calling fused NPU ops.
    num_blocks_per_seq = math.ceil(seq_len / BLOCK_SIZE)
    num_blocks = batch_size * num_blocks_per_seq
    k_cache = torch.zeros(
        (num_blocks, BLOCK_SIZE, KV_LORA_RANK), dtype=dtype, device=device
    )
    k_rope_cache = torch.zeros(
        (num_blocks, BLOCK_SIZE, QK_ROPE_HEAD_DIM), dtype=dtype, device=device
    )
    return k_cache, k_rope_cache


def _fill_existing_cache(
    module: NPUDeepseekMLAAttention,
    k_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    rng: torch.Generator,
) -> None:
    # Populate cache for prefix tokens to emulate a real decode scenario.
    #
    # Important: decode reads from kv_cache populated during prefill (or previous decodes).
    # Those cached values are not arbitrary; they are produced by:
    #   kv_a_layernorm(kv_a) + rotary(k_pe)
    # and written into the paged cache via slot_mapping.
    batch_size = block_table.shape[0]
    device = k_cache.device
    num_tokens = batch_size * (seq_len - 1)
    if num_tokens <= 0:
        return

    positions = torch.arange(seq_len - 1, device=device, dtype=torch.long).repeat(batch_size)
    cos, sin = _get_cos_sin_for_positions(module.rotary_emb, positions)

    kv_a = torch.randn(
        num_tokens, KV_LORA_RANK, generator=rng, device=device, dtype=k_cache.dtype
    )
    k_pe = torch.randn(
        num_tokens, QK_ROPE_HEAD_DIM, generator=rng, device=device, dtype=k_cache.dtype
    )
    kv_a = module.kv_a_layernorm(kv_a)
    k_pe = _apply_rope(k_pe, cos, sin)

    slots: list[int] = []
    for i in range(batch_size):
        for pos in range(seq_len - 1):
            block_id = int(block_table[i, pos // BLOCK_SIZE].item())
            slots.append(block_id * BLOCK_SIZE + (pos % BLOCK_SIZE))
    slot_tensor = torch.tensor(slots, dtype=torch.long, device=device)

    flat_k = k_cache.view(-1, KV_LORA_RANK)
    flat_rope = k_rope_cache.view(-1, QK_ROPE_HEAD_DIM)
    flat_k[slot_tensor] = kv_a
    flat_rope[slot_tensor] = k_pe


def _transpose_flashcomm_weights(module: torch.nn.Module) -> None:
    # FlashCommLinear stores weights in a transposed layout for NPU matmul.
    # The reference path uses `torch.matmul(x, weight)` so we transpose once here.
    for layer in module.modules():
        if isinstance(layer, (ColumnParallelFlashCommLinear, RowParallelFlashCommLinear)):
            weight = layer.weight
            if weight is not None and weight.ndim == 2:
                weight.data = weight.data.t().contiguous()
                setattr(weight, "is_weight_transposed", True)


def _project_q(module: NPUDeepseekMLAAttention, hidden_states: torch.Tensor) -> torch.Tensor:
    # Test-only helper: match the module's q path (q_a_proj -> RMSNorm -> q_b_proj).
    q_lora = module.q_a_proj(hidden_states)[0]
    q_norm = module.q_a_layernorm(q_lora)
    return module.q_b_proj(q_norm)[0]


def _reset_mla_impl_weights(module: NPUDeepseekMLAAttention) -> None:
    # vLLM MLA implementation expects:
    # - W_UK_T: [N, P, L] used for q_nope -> q_latent (decode)
    # - W_UV:   [N, L, V] used for latent output -> value head dim
    # These are derived from kv_b_proj weights (concatenated per head).
    kv_weight = module.kv_b_proj.weight
    kv_weight = kv_weight.view(
        KV_LORA_RANK, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM
    )
    w_uk, w_uv = kv_weight.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    module.attn.impl.W_UK_T = w_uk.permute(1, 2, 0).contiguous()
    module.attn.impl.W_UV = w_uv.transpose(0, 1).contiguous()


def _build_module(
    device: torch.device,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
    attn_metadata: DummyAttnMeta,
) -> NPUDeepseekMLAAttention:
    # Build a minimal module instance without needing a full vLLM engine.
    # We monkeypatch:
    # - current vLLM config (so MLAAttention can pick backend/kv-cache settings)
    # - forward context (so `forward()` finds our synthetic attn_metadata)
    # - tensor parallel helpers (so TP group init is not required in unit tests)
    dummy_config = DummyVllmConfig()
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: dummy_config,
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: dummy_config,
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: dummy_config,
    )
    monkeypatch.setattr(
        "omni_npu.v1.layers.attention.npu_mla.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: type("TPGroup", (), {"rank_in_group": 0, "world_size": 1})(),
    )
    ctx = DummyForwardContext(attn_metadata)
    monkeypatch.setattr(
        "omni_npu.v1.layers.attention.npu_mla.get_forward_context",
        lambda: ctx,
    )
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: ctx,
    )

    # Production prefill quantizes q into a dict (`{'x_int8': ..., 'pertoken_scale': ...}`)
    # before feeding it into q_b_proj. In unit tests we don't exercise quantized matmul,
    # so we:
    # - make dynamic quant a no-op (keep float tensor)
    # - wrap q_b_proj.forward to accept that dict and unwrap it back to a tensor
    monkeypatch.setattr(
        "omni_npu.v1.layers.attention.npu_mla.torch_npu.npu_dynamic_quant",
        lambda x: (x, None),
    )

    config = _DeepseekConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        max_position_embeddings=MAX_POS,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        q_lora_rank=Q_LORA_RANK,
    )
    config.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}

    module = NPUDeepseekMLAAttention(
        vllm_config=dummy_config,
        config=config,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        max_position_embeddings=MAX_POS,
        cache_config=dummy_config.cache_config,
        quant_config=None,
        prefix=f"layers.1",
    ).to(device=device, dtype=dtype)

    orig_q_b_forward = module.q_b_proj.forward

    def _q_b_forward_accept_dict(input_: object, throw_dequant: bool = False):
        if isinstance(input_, dict):
            input_ = input_["x_int8"]
        return orig_q_b_forward(input_, throw_dequant=throw_dequant)

    module.q_b_proj.forward = _q_b_forward_accept_dict  # type: ignore[method-assign]

    torch.manual_seed(0)
    for param in module.parameters():
        if param is not None and param.ndim > 0:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)

    _transpose_flashcomm_weights(module)
    _reset_mla_impl_weights(module)
    module.eval()

    # Add module to the forward context's no_compile_layers for npu_mla_forward
    ctx.no_compile_layers[module.prefix] = module

    return module


def _reference_prefill(
    module: NPUDeepseekMLAAttention,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_lens: list[int],
) -> torch.Tensor:
    # Reference prefill: causal attention per sequence using pure PyTorch ops.
    #
    # Shapes (typical):
    # - hidden_states: [T, H] where T = sum(seq_lens)
    # - q:            [T, N, QK_HEAD_DIM]
    # - kv_a:         [T, KV_LORA_RANK] (compressed key/value)
    # - k_pe:         [T, QK_ROPE_HEAD_DIM] (RoPE part)
    # - k_nope / v:   [T, N, ...] (expanded per-head via kv_b_proj)
    q = _project_q(module, hidden_states).view(-1, NUM_HEADS, QK_HEAD_DIM)
    latent_cache = module.kv_a_proj_with_mqa(hidden_states)[0]
    kv_a, k_pe = latent_cache.split([KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
    kv_a = module.kv_a_layernorm(kv_a)

    q_nope, q_pe = q.split([QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)
    q_pe = _apply_rope(q_pe, cos, sin)
    k_pe = _apply_rope(k_pe, cos, sin)

    kv = module.kv_b_proj(kv_a)[0].view(
        -1, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM
    )
    k_nope, v = kv.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    k_rope = k_pe.unsqueeze(1).repeat(1, NUM_HEADS, 1)

    outputs = []
    start = 0
    for seq_len in seq_lens:
        end = start + seq_len
        # Slice per-sequence TND segments:
        # qn/qp shape [S, N, D], kn/kr/vv shape [S, N, D].
        qn = q_nope[start:end]
        qp = q_pe[start:end]
        kn = k_nope[start:end]
        kr = k_rope[start:end]
        vv = v[start:end]

        # Attention score decomposes into "NOPE" dot and "RoPE" dot:
        # score[h, t, s] = <q_nope[t,h], k_nope[s,h]> + <q_rope[t,h], k_rope[s,h]>
        # and then apply causal masking (upper-triangular) for prefill.
        scores = (
            torch.einsum("thd,shd->hts", qn.float(), kn.float())
            + torch.einsum("thd,shd->hts", qp.float(), kr.float())
        ) * module.scaling
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hts,shd->thd", probs, vv.float())
        outputs.append(out)
        start = end

    attn_out = torch.cat(outputs, dim=0).to(hidden_states.dtype)
    attn_out = attn_out.reshape(-1, NUM_HEADS * V_HEAD_DIM)
    return module.o_proj(attn_out)[0]


def _reference_decode(
    module: NPUDeepseekMLAAttention,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    slot_mapping: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    # Reference decode: attend a single new token against cached prefix per sequence.
    #
    # Shapes (typical):
    # - hidden_states: [B, H] (one token per sequence)
    # - q_nope:        [B, N, QK_NOPE_HEAD_DIM]
    # - q_pe:          [B, N, QK_ROPE_HEAD_DIM]
    # - kv_cache[0]:   [num_blocks, BLOCK_SIZE, KV_LORA_RANK]
    # - kv_cache[1]:   [num_blocks, BLOCK_SIZE, QK_ROPE_HEAD_DIM]
    #
    # Design: we emulate the paged KV-cache by indexing into flattened slots.
    q = _project_q(module, hidden_states)
    kv = module.kv_a_proj_with_mqa(hidden_states)[0]

    q = q.view(-1, NUM_HEADS, 1, QK_HEAD_DIM)
    q_nope, q_pe = q.split([QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)
    q_nope = q_nope.squeeze(2)
    q_pe = _apply_rope(q_pe.squeeze(2), cos, sin)
    q_nope = torch.einsum("bhd,hdv->bhv", q_nope.float(), module.attn.impl.W_UK_T.float())

    kv_a, k_pe = kv.split([KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
    kv_a = module.kv_a_layernorm(kv_a)
    k_pe = _apply_rope(k_pe, cos, sin)

    flat_k = kv_cache[0].view(-1, KV_LORA_RANK)
    flat_rope = kv_cache[1].view(-1, QK_ROPE_HEAD_DIM)
    # Insert the new token at its physical slot (per sequence) just like production.
    flat_k[slot_mapping] = kv_a
    flat_rope[slot_mapping] = k_pe

    outputs = []
    for i, seq_len in enumerate(seq_lens):
        # Gather this sequence's prefix+current slots using its block_table row.
        # This mirrors how the fused kernel uses `block_table` to map logical
        # positions to physical blocks.
        slots = []
        for pos in range(seq_len):
            block_id = int(block_table[i, pos // BLOCK_SIZE].item())
            slots.append(block_id * BLOCK_SIZE + (pos % BLOCK_SIZE))
        slots_t = torch.tensor(slots, dtype=torch.long, device=hidden_states.device)
        kn = flat_k[slots_t].unsqueeze(1).expand(-1, NUM_HEADS, -1)
        kr = flat_rope[slots_t].unsqueeze(1).expand(-1, NUM_HEADS, -1)
        qn = q_nope[i]
        qp = q_pe[i]

        # Decode has no causal mask for the single query token; it attends over the
        # full available KV length (seq_len) for that sequence.
        scores = (
            torch.einsum("hd,shd->hs", qn.float(), kn.float())
            + torch.einsum("hd,shd->hs", qp.float(), kr.float())
        ) * module.scaling
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hs,shd->hd", probs, kn.float())
        outputs.append(out)

    attn_out = torch.stack(outputs, dim=0)
    out_proj = torch.einsum("bhd,hdv->bhv", attn_out.float(), module.attn.impl.W_UV.float())
    out_proj = out_proj.reshape(hidden_states.shape[0], -1).to(hidden_states.dtype)
    return module.o_proj(out_proj)[0]


@pytest.mark.integration
@pytest.mark.parametrize("batch_size,seq_len", [(1, 16), (8, 160)])
def test_npu_mla_prefill_matches_reference(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    seq_len: int,
) -> None:
    # Prefill path with strided inputs; single/multi-block via seq_len.
    # Example: hidden_states shape [B * S, H], where B=batch_size, S=seq_len.
    # This matches the prefill branch in NPUDeepseekMLAAttention._forward_prefill.
    device = torch.device("npu")
    dtype = torch.bfloat16

    # Block table shape [B, ceil(S / BLOCK_SIZE)], slot_mapping shape [B * S].
    # Each entry in slot_mapping is a flattened slot index: block_id * BLOCK_SIZE + offset.
    block_table = _build_block_table(batch_size, seq_len, device)
    slot_mapping = _build_slot_mapping(batch_size, seq_len, block_table)
    query_cumlens = _build_query_cumlens(batch_size, seq_len)
    prefill_meta = DummyPrefillMeta(
        seq_lens=[seq_len] * batch_size,
        query_cumlens=query_cumlens,
        max_query_len=seq_len,
        block_table=block_table,
    )
    attn_metadata = DummyAttnMeta(
        prefill=prefill_meta,
        decode=None,
        slot_mapping=slot_mapping,
        num_actual_tokens=batch_size * seq_len,
        num_prefills=batch_size,
        num_decodes=0,
        num_decode_tokens=0,
    )

    module = _build_module(device, dtype, monkeypatch, attn_metadata)
    kv_cache = _init_kv_cache(batch_size, seq_len, device, dtype)
    # NPU runtime wraps each layer's cache for virtual-engine compatibility.
    module.attn.kv_cache = [kv_cache]

    # Cos/sin for every token position (length = B * S).
    # Example: positions = [0, 1, 2, ..., S-1] repeated for each sequence.
    positions = _build_positions(batch_size, seq_len, device)
    cos, sin = _get_cos_sin_for_positions(module.rotary_emb, positions)

    # Make a non-contiguous input: base [H, 2*B*S] -> take every other column.
    # This ensures the attention kernel handles strided inputs correctly.
    base = torch.randn(
        HIDDEN_SIZE, positions.numel() * 2, device=device, dtype=dtype
    )
    hidden_states = base[:, ::2].t()
    assert not hidden_states.is_contiguous()

    # Compare NPU output [B*S, H] against a pure PyTorch reference.
    # The reference expands q/k/v, applies rotary, and runs causal attention per sequence.
    actual = module(hidden_states, cos, sin)
    expected = _reference_prefill(
        module,
        hidden_states,
        cos,
        sin,
        [seq_len] * batch_size,
    )

    _assert_allclose(actual, expected, "prefill_output")


@pytest.mark.integration
@pytest.mark.parametrize("batch_size,seq_len", [(1, 16), (8, 160)])
def test_npu_mla_decode_matches_reference(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    seq_len: int,
) -> None:
    # Decode path with cached prefix and a new token per sequence.
    # Example: hidden_states shape [B, H], one new token per sequence.
    # This matches the decode branch in NPUDeepseekMLAAttention._forward_decode.
    device = torch.device("npu")
    dtype = torch.bfloat16

    # Block table shape [B, ceil(S / BLOCK_SIZE)], slot_mapping shape [B].
    # For decode, we only write the last position into the cache.
    block_table = _build_block_table(batch_size, seq_len, device)
    slot_mapping = _build_decode_slot_mapping(seq_len, block_table, device)
    decode_meta = DummyDecodeMeta(
        block_table=block_table,
        seq_lens=[seq_len] * batch_size,
        query_cumlens=list(range(1, batch_size + 1)),
    )
    attn_metadata = DummyAttnMeta(
        prefill=None,
        decode=decode_meta,
        slot_mapping=slot_mapping,
        num_actual_tokens=batch_size,
        num_prefills=0,
        num_decodes=batch_size,
        num_decode_tokens=batch_size,
    )

    module = _build_module(device, dtype, monkeypatch, attn_metadata)

    kv_cache = _init_kv_cache(batch_size, seq_len, device, dtype)
    ref_kv_cache = _init_kv_cache(batch_size, seq_len, device, dtype)
    rng = torch.Generator(device=device).manual_seed(1234)
    # Seed cache with prefix tokens (S-1) so decode attends over real context.
    # These cached values are produced with the same RMSNorm + RoPE path as production.
    _fill_existing_cache(module, ref_kv_cache[0], ref_kv_cache[1], block_table, seq_len, rng)
    kv_cache[0].copy_(ref_kv_cache[0])
    kv_cache[1].copy_(ref_kv_cache[1])

    module.attn.kv_cache = [kv_cache]

    # Cos/sin for the decode token position (last position for each sequence).
    # Positions shape [B], cos/sin shape [B, 1, 1, rope_dim] in NPU rotary emb.
    positions = torch.full(
        (batch_size,),
        seq_len - 1,
        device=device,
        dtype=torch.long,
    )
    cos, sin = _get_cos_sin_for_positions(module.rotary_emb, positions)

    # Non-contiguous [B, H] input by slicing a transposed base.
    # This mirrors a strided decode input in a fused pipeline.
    base = torch.randn(HIDDEN_SIZE, batch_size * 2, device=device, dtype=dtype)
    hidden_states = base[:, ::2].t()
    assert not hidden_states.is_contiguous()

    # Compare NPU output [B, H] against a pure PyTorch reference.
    # The reference uses cached KV for prefix and inserts the new token by slot mapping.
    actual = module(hidden_states, cos, sin)
    expected = _reference_decode(
        module,
        hidden_states,
        cos,
        sin,
        block_table,
        [seq_len] * batch_size,
        slot_mapping,
        ref_kv_cache,
    )

    _assert_allclose(actual, expected, "decode_output")
