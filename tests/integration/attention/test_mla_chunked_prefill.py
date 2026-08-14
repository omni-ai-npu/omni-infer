# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Chunked-prefill math test for the base-layer NPU MLA backend.

What path is being tested
------------------------
This file targets `omni_npu.attention.backends.mla.NPUMLAImpl._forward_prefill`
when `attn_metadata.prefill.chunked_context is not None` (a.k.a. "chunked prefill").

In this mode, prefill attention is computed as a merge of:
1) Context attention: queries attend to cached prefix KV gathered from paged KV-cache
   in one or more chunks (no causal mask required because all context tokens are
   strictly before the current query tokens).
2) Suffix attention: queries attend to the current chunk's KV (causal within the
   chunk) using the NPU fused attention kernel.

The backend merges these two attention results using log-sum-exp (LSE) merging,
which must be mathematically equivalent to running a single attention over the
concatenated KV sequence: [context_prefix, suffix_tokens] with causal masking.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch_npu must be available for NPU MLA tests") from exc

if not (hasattr(torch, "npu") and torch.npu.device_count() > 0):
    raise RuntimeError("NPU hardware is required for NPU MLA tests")

from types import SimpleNamespace

from omni_npu.attention.backends.mla import NPUMLAImpl, NPUMLAMetadataBuilder
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec


# Keep dimensions small enough for test runtime, but realistic for MLA.
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM

NUM_HEADS = 4
BLOCK_SIZE = 128


@pytest.fixture(autouse=True)
def _clear_mla_static_masks() -> None:
    NPUMLAImpl.SHARE_MASK_TRIL_SPARSE = None
    yield
    NPUMLAImpl.SHARE_MASK_TRIL_SPARSE = None


@contextmanager
def _default_device(device: str):
    prev_device = torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(prev_device)


@dataclass
class DummyChunkedContext:
    # Total number of gathered KV tokens for each chunk iteration.
    # Example (B=2): seq_tot=[64, 32] means first iter gathers 32 tokens/seq (64 total),
    # second iter gathers 16 tokens/seq (32 total).
    seq_tot: list[int]
    # Workspace to hold gathered [kv_c_normed | k_pe] contiguously.
    # Shape: [max(seq_tot), KV_LORA_RANK + QK_ROPE_HEAD_DIM] = [*, 576]
    workspace: torch.Tensor
    # Per-iter cu_seq_lens in TND format, shape [iters, B+1].
    cu_seq_lens: torch.Tensor
    # Per-iter start offsets (where to start gathering within each sequence), len=iters.
    starts: list[torch.Tensor]


def _assert_allclose(actual: torch.Tensor,
                     expected: torch.Tensor,
                     name: str,
                     atol: float = 3e-2,
                     rtol: float = 3e-2) -> None:
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return
    diff = (actual - expected).abs()
    flat_idx = diff.view(-1).argmax().item()
    idx = torch.unravel_index(torch.tensor(flat_idx, device="cpu"), diff.shape)
    raise AssertionError(
        f"{name} mismatch: max_diff={diff.max().item()} idx={tuple(idx)} "
        f"actual={actual.view(-1)[flat_idx].item()} expected={expected.view(-1)[flat_idx].item()} "
        f"shape={tuple(actual.shape)} dtype={actual.dtype}"
    )


def _build_contiguous_block_table(batch_size: int, total_seq_len: int, device: torch.device) -> torch.Tensor:
    num_blocks_per_seq = math.ceil(total_seq_len / BLOCK_SIZE)
    block_table = torch.empty((batch_size, num_blocks_per_seq), dtype=torch.int32, device=device)
    for i in range(batch_size):
        start = i * num_blocks_per_seq
        block_table[i] = torch.arange(start, start + num_blocks_per_seq, dtype=torch.int32, device=device)
    return block_table


def _slot(block_id: int, pos: int) -> int:
    return block_id * BLOCK_SIZE + (pos % BLOCK_SIZE)


def _slots_for_positions(block_table: torch.Tensor, seq_idx: int, positions: range) -> list[int]:
    slots = []
    for pos in positions:
        block_id = int(block_table[seq_idx, pos // BLOCK_SIZE].item())
        slots.append(_slot(block_id, pos))
    return slots


def _fill_cache_for_context(
    k_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fill the KV-cache for the prefix tokens [0..context_len-1] and return the
    contiguous per-token tensors used (for reference).

    Cache layout:
    - k_cache:    [num_blocks, BLOCK_SIZE, KV_LORA_RANK]
    - rope_cache: [num_blocks, BLOCK_SIZE, QK_ROPE_HEAD_DIM]
    """
    batch_size = block_table.shape[0]
    device = k_cache.device
    dtype = k_cache.dtype

    total_ctx_tokens = batch_size * context_len
    ctx_k = torch.randn(total_ctx_tokens, KV_LORA_RANK, generator=rng, device=device, dtype=dtype)
    ctx_rope = torch.randn(total_ctx_tokens, QK_ROPE_HEAD_DIM, generator=rng, device=device, dtype=dtype)

    slots = []
    for b in range(batch_size):
        slots.extend(_slots_for_positions(block_table, b, range(0, context_len)))
    slot_tensor = torch.tensor(slots, dtype=torch.long, device=device)

    flat_k = k_cache.view(-1, KV_LORA_RANK)
    flat_rope = rope_cache.view(-1, QK_ROPE_HEAD_DIM)
    flat_k[slot_tensor] = ctx_k
    flat_rope[slot_tensor] = ctx_rope
    return ctx_k.view(batch_size, context_len, -1), ctx_rope.view(batch_size, context_len, -1)


def _make_chunked_context(
    batch_size: int,
    context_len: int,
    chunk_lens: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> DummyChunkedContext:
    """
    Create a chunk schedule that covers the context prefix.

    Example (context_len=48, chunk_lens=[32,16]):
      iter0: gather positions [0..31]   for each seq
      iter1: gather positions [32..47]  for each seq
    """
    assert sum(chunk_lens) == context_len, "chunk_lens must sum to context_len"
    iters = len(chunk_lens)

    max_tot = batch_size * max(chunk_lens)
    workspace = torch.empty((max_tot, KV_LORA_RANK + QK_ROPE_HEAD_DIM), device=device, dtype=dtype)

    cu_rows = []
    starts: list[torch.Tensor] = []
    offset = 0
    for chunk_len in chunk_lens:
        cu = torch.arange(0, (batch_size + 1) * chunk_len, chunk_len, device=device, dtype=torch.int32)
        cu_rows.append(cu)
        starts.append(torch.full((batch_size,), offset, dtype=torch.int32, device="cpu"))
        offset += chunk_len

    cu_seq_lens = torch.stack(cu_rows, dim=0)
    seq_tot = [int(cu[-1].item()) for cu in cu_rows]
    return DummyChunkedContext(seq_tot=seq_tot, workspace=workspace, cu_seq_lens=cu_seq_lens, starts=starts)


def _reference_full_prefill_over_context_plus_suffix(
    impl: NPUMLAImpl,
    kv_b_proj: torch.nn.Module,
    q: torch.Tensor,
    suffix_kv_c_normed: torch.Tensor,
    suffix_k_pe: torch.Tensor,
    ctx_kv_c_normed: torch.Tensor,
    ctx_k_pe: torch.Tensor,
    context_len: int,
    suffix_len: int,
) -> torch.Tensor:
    """
    Compute the mathematically expected prefill output for the suffix queries:
    attention over keys/values formed by concatenating [context_prefix, suffix],
    with causal masking applied only within the suffix portion.

    Shapes:
    - q:                  [T, N, QK_HEAD_DIM] where T=B*suffix_len (queries only)
    - suffix_kv_c_normed: [T, KV_LORA_RANK]
    - suffix_k_pe:        [T, 1, QK_ROPE_HEAD_DIM]
    - ctx_kv_c_normed:    [B, context_len, KV_LORA_RANK]
    - ctx_k_pe:           [B, context_len, QK_ROPE_HEAD_DIM]
    """
    device = q.device
    dtype = q.dtype
    batch_size = ctx_kv_c_normed.shape[0]

    q_nope, q_pe = q.split([QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)
    q_pe = q_pe.to(torch.float32)
    q_nope = q_nope.to(torch.float32)

    # Expand KV for suffix.
    suffix_k_pe = suffix_k_pe.squeeze(1)
    suffix_kv = kv_b_proj(suffix_kv_c_normed)[0].view(-1, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
    suffix_k_nope, suffix_v = suffix_kv.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    suffix_k_nope = suffix_k_nope.to(torch.float32)
    suffix_v = suffix_v.to(torch.float32)
    suffix_k_rope = suffix_k_pe.view(-1, 1, QK_ROPE_HEAD_DIM).repeat(1, NUM_HEADS, 1).to(torch.float32)

    # Expand KV for context (cached prefix).
    ctx_flat = ctx_kv_c_normed.reshape(-1, KV_LORA_RANK)
    ctx_kv = kv_b_proj(ctx_flat)[0].view(-1, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
    ctx_k_nope, ctx_v = ctx_kv.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    ctx_k_nope = ctx_k_nope.to(torch.float32)
    ctx_v = ctx_v.to(torch.float32)
    ctx_k_rope = ctx_k_pe.reshape(-1, QK_ROPE_HEAD_DIM).view(-1, 1, QK_ROPE_HEAD_DIM).repeat(1, NUM_HEADS, 1).to(torch.float32)

    # Per-sequence reference: for each query token t in suffix, attend over:
    # keys = [context_len tokens] + [suffix tokens up to t].
    out = torch.empty((batch_size * suffix_len, NUM_HEADS, V_HEAD_DIM), device=device, dtype=torch.float32)
    for b in range(batch_size):
        q_b = q_nope[b * suffix_len:(b + 1) * suffix_len]
        qpe_b = q_pe[b * suffix_len:(b + 1) * suffix_len]
        kctx_nope = ctx_k_nope[b * context_len:(b + 1) * context_len]
        kctx_rope = ctx_k_rope[b * context_len:(b + 1) * context_len]
        vctx = ctx_v[b * context_len:(b + 1) * context_len]

        ks_nope = suffix_k_nope[b * suffix_len:(b + 1) * suffix_len]
        ks_rope = suffix_k_rope[b * suffix_len:(b + 1) * suffix_len]
        vs = suffix_v[b * suffix_len:(b + 1) * suffix_len]

        for t in range(suffix_len):
            kn = torch.cat([kctx_nope, ks_nope[: t + 1]], dim=0)  # [K, N, P]
            kr = torch.cat([kctx_rope, ks_rope[: t + 1]], dim=0)  # [K, N, R]
            vv = torch.cat([vctx, vs[: t + 1]], dim=0)            # [K, N, V]

            scores = (
                torch.einsum("hd,khd->hk", q_b[t], kn)
                + torch.einsum("hd,khd->hk", qpe_b[t], kr)
            ) * float(impl.scale)
            probs = torch.softmax(scores, dim=-1)
            out[b * suffix_len + t] = torch.einsum("hk,khd->hd", probs, vv)

    return out.to(dtype).reshape(batch_size * suffix_len, -1)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 8])
def test_mla_chunked_prefill_matches_full_attention(
        default_vllm_config, monkeypatch: pytest.MonkeyPatch,
        batch_size: int) -> None:
    """
    Scenario:
    - Each sequence has a cached prefix (context_len) already present in paged KV-cache.
    - We run a prefill for a new suffix (suffix_len) with chunked_context enabled.
    - The backend must merge context-attn and suffix-attn correctly (LSE merge).
    """
    device = torch.device("npu")
    dtype = torch.bfloat16

    context_len = 48
    suffix_len = 16
    total_seq_len = context_len + suffix_len

    # Chunk schedule for context prefix: two iterations, 32 then 16 tokens.
    chunk_lens = [32, 16]

    # Patch vLLM config lookups and workspace-size to keep the unit test self-contained.
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
        lambda *_args, **_kwargs: 32 * batch_size,
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: None,
    )
    ctx = type("Ctx", (), {"batch_descriptor": None, "virtual_engine": 0})()
    # vLLM's MLA code may use `get_forward_context` via different import paths
    # depending on vLLM version. Patch the likely targets, but don't hard-fail
    # if one doesn't exist.
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: ctx,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: ctx,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mla_attention.get_forward_context",
        lambda: ctx,
        raising=False,
    )

    # vLLM linear layers return `(out, bias)`; NPUMLAImpl expects that convention.
    linear = torch.nn.Linear(
        KV_LORA_RANK,
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        bias=False,
        device=device,
        dtype=dtype,
    )
    for p in linear.parameters():
        p.requires_grad_(False)

    class KVProjWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Linear):
            super().__init__()
            self.inner = inner

        @property
        def weight(self):
            return self.inner.weight

        def forward(self, x: torch.Tensor):
            return self.inner(x), None

    kv_b_proj = KVProjWrapper(linear)

    impl = NPUMLAImpl(
        num_heads=NUM_HEADS,
        head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        scale=1.0 / math.sqrt(QK_HEAD_DIM),
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kv_b_proj=kv_b_proj,
    )

    # Build paged KV-cache for compressed KV and RoPE parts.
    num_blocks_per_seq = math.ceil(total_seq_len / BLOCK_SIZE)
    num_blocks = batch_size * num_blocks_per_seq
    k_cache = torch.zeros((num_blocks, BLOCK_SIZE, KV_LORA_RANK), device=device, dtype=dtype)
    rope_cache = torch.zeros((num_blocks, BLOCK_SIZE, QK_ROPE_HEAD_DIM), device=device, dtype=dtype)
    kv_cache = (k_cache, rope_cache)

    block_table = _build_contiguous_block_table(batch_size, total_seq_len, device)
    rng = torch.Generator(device=device).manual_seed(123)
    ctx_kv_c_normed, ctx_k_pe = _fill_cache_for_context(k_cache, rope_cache, block_table, context_len, rng)

    # Suffix inputs are laid out in TND format: concatenate sequences.
    # q: [T, N, QK_HEAD_DIM], kv_c_normed: [T, KV_LORA_RANK], k_pe: [T, 1, R]
    total_q = batch_size * suffix_len
    q_nope = torch.randn((total_q, NUM_HEADS, QK_NOPE_HEAD_DIM), generator=rng, device=device, dtype=dtype)
    q_rope = torch.randn((total_q, NUM_HEADS, QK_ROPE_HEAD_DIM), generator=rng, device=device, dtype=dtype)
    q = torch.cat([q_nope, q_rope], dim=-1)

    suffix_kv_c_normed = torch.randn((total_q, KV_LORA_RANK), generator=rng, device=device, dtype=dtype)
    suffix_k_pe = torch.randn((total_q, 1, QK_ROPE_HEAD_DIM), generator=rng, device=device, dtype=dtype)

    # slot_mapping for writing suffix tokens into cache corresponds to positions
    # [context_len .. context_len+suffix_len-1] per sequence, in TND order.
    suffix_slots = []
    for b in range(batch_size):
        suffix_slots.extend(_slots_for_positions(block_table, b, range(context_len, total_seq_len)))
    slot_mapping = torch.tensor(suffix_slots, dtype=torch.long, device=device)

    class DummyModelConfig:
        def __init__(self):
            self.dtype = dtype
            self.hf_text_config = SimpleNamespace(
                q_lora_rank=None,
                kv_lora_rank=KV_LORA_RANK,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
            )

        def get_head_size(self):
            return KV_LORA_RANK + QK_ROPE_HEAD_DIM

        def get_num_attention_heads(self, _parallel_config):
            return NUM_HEADS

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=SimpleNamespace(use_non_causal=False),
        scheduler_config=SimpleNamespace(max_num_seqs=batch_size),
        model_config=DummyModelConfig(),
        parallel_config=SimpleNamespace(
            cp_kv_cache_interleave_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=0),
        kv_transfer_config=None,
        speculative_config=None,
    )

    kv_cache_spec = AttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=dtype,
    )
    metadata_builder = NPUMLAMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layer0"],
        vllm_config=vllm_config,
        device=device,
    )

    # Provide a safe wrapper for gather that tolerates CPU `starts` tensors by converting them to ints.
    import omni_npu.attention.ops as ops

    real_gather = ops.gather_and_maybe_dequant_cache
    real_merge = ops.merge_attn_states

    # --- Branch execution guards ---
    # We want to ensure we are actually exercising the "chunked prefill" path:
    # - `_compute_prefill_context` must be invoked
    # - `gather_and_maybe_dequant_cache` must be called once per chunk iteration
    # - `merge_attn_states` must be used to merge (a) chunk context outputs and (b) context+suffix outputs
    call_counts = {
        "compute_prefill_context": 0,
        "gather_cache": 0,
        "merge_states": 0,
    }

    real_compute_prefill_context = impl._compute_prefill_context

    def compute_prefill_context_wrapper(*args, **kwargs):
        call_counts["compute_prefill_context"] += 1
        return real_compute_prefill_context(*args, **kwargs)

    monkeypatch.setattr(impl, "_compute_prefill_context", compute_prefill_context_wrapper)

    def gather_wrapper(*, seq_starts=None, **kwargs):
        if seq_starts is not None and isinstance(seq_starts, torch.Tensor) and seq_starts.device.type == "cpu":
            seq_starts = torch.tensor([int(x) for x in seq_starts.tolist()], dtype=torch.int32, device=device)
        call_counts["gather_cache"] += 1
        return real_gather(seq_starts=seq_starts, **kwargs)

    monkeypatch.setattr(ops, "gather_and_maybe_dequant_cache", gather_wrapper)

    def merge_wrapper(*args, **kwargs):
        call_counts["merge_states"] += 1
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(ops, "merge_attn_states", merge_wrapper)

    query_start_loc_cpu = torch.tensor(
        [i * suffix_len for i in range(batch_size + 1)], dtype=torch.int32, device="cpu"
    )
    query_start_loc = query_start_loc_cpu.to(device)
    seq_lens_cpu = torch.full((batch_size,), total_seq_len, dtype=torch.int32, device="cpu")
    seq_lens = seq_lens_cpu.to(device)

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=batch_size,
        num_actual_tokens=total_q,
        max_query_len=suffix_len,
        max_seq_len=total_seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )
    with _default_device("cpu"):
        attn_metadata = metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
    chunked_context = attn_metadata.prefill.chunked_context

    layer = type("Layer", (), {})()
    layer._k_scale = None

    output = torch.empty((total_q, NUM_HEADS * V_HEAD_DIM), device=device, dtype=dtype)

    actual = impl.forward(
        layer=layer,
        q=q,
        k_c_normed=suffix_kv_c_normed,
        k_pe=suffix_k_pe,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )

    # Confirm the intended chunked-prefill branch executed.
    assert chunked_context is not None and len(chunk_lens) > 0
    assert call_counts["compute_prefill_context"] == 1, call_counts
    assert call_counts["gather_cache"] == len(chunk_lens), call_counts
    # merge_attn_states is used:
    # - (iters-1) times to merge context chunks
    # - +1 time to merge context-attn with suffix-attn
    assert call_counts["merge_states"] == len(chunk_lens), call_counts

    expected = _reference_full_prefill_over_context_plus_suffix(
        impl=impl,
        kv_b_proj=kv_b_proj,
        q=q,
        suffix_kv_c_normed=suffix_kv_c_normed,
        suffix_k_pe=suffix_k_pe,
        ctx_kv_c_normed=ctx_kv_c_normed,
        ctx_k_pe=ctx_k_pe,
        context_len=context_len,
        suffix_len=suffix_len,
    )

    _assert_allclose(actual, expected, "chunked_prefill_output")
