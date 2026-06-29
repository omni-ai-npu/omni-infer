# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Custom op wrappers for NPUPanguSparseAttention sub-paths.

These wrappers encapsulate kernel calls that mutate kv_cache / conv_states
or take view-of-input as parameters. By marking them as opaque custom ops,
we hide the in-place mutations from torch.compile / AOT autograd's
functionalization pass, while keeping the surrounding compute traceable so
it can be fused / kernel-controlled.

Importing this module (see the import at the bottom of npu_pangu.py)
triggers `direct_register_custom_op` for every op defined here. Callers
opt in by substituting `torch.ops.vllm.<op_name>(...)` for the
corresponding direct method call; nothing forces them to.

==============================================================================
How to use
==============================================================================

1. Import side-effect is enough — just `from omni_npu.v1.layers.attention
   import npu_pangu_custom_ops  # noqa: F401`. After that, every op below
   is reachable via `torch.ops.vllm.<op_name>(...)`.

2. At a call site that currently invokes a method which (a) writes into
   kv_cache / conv_states in place, and (b) is inside a region you want
   torch.compile to trace, replace the method call with the corresponding
   wrapper. Example:

       # before
       return self._apply_SWA_attention_decode(
           q_nope, q_pe, kv_cache, attn_metadata,
       )

       # after
       return torch.ops.vllm.npu_pangu_swa_decode(
           q_nope, q_pe, kv_cache[0], kv_cache[1], self.layer_name,
       )

   Pass `self.layer_name` (or the parent attention layer's prefix) as the
   last argument; the wrapper uses it to look up the live layer object
   and attn_metadata out of forward_context.

3. Current op catalog (op_name -> wrapped method / kernel):

     npu_pangu_swa_decode            -> _apply_SWA_attention_decode
     npu_pangu_indexer_cache_update  -> Indexer._update_indexer_cache
     npu_pangu_lightning_indexer     -> Indexer._apply_lightning_indexer
     npu_pangu_kv_cache_update       -> _npu_kvrmsnorm_rope_cache
                                        (DSA / absorb paths only)
     npu_pangu_mome_fc2_scatter_and_return
                                     -> npu_ai_infra_scatter_block_update_
                                        (FC2 cache scatter tail)
     npu_pangu_mome_conv             -> npu_ai_infra_fused_causal_conv1d
                                        (generic, modes 0/1)
     npu_pangu_kv_down_mome_inplace  -> npu_ai_infra_fused_causal_conv1d
                                        (kv-down specialization, see below)

==============================================================================
How to add a new op
==============================================================================

For each op you need three things, in order:

    def npu_pangu_<name>(<args>, layer_name: str) -> <Tensor | None>:
        layer, attn_metadata = _lookup_layer_and_attn_metadata(layer_name)
        ...                                  # do the real work
        return ...                           # or no return

    def npu_pangu_<name>_fake(<same args>, layer_name: str) -> <same type>:
        # same signature as the real op; return a tensor with the right
        # shape / dtype / device, or None. Keep the signature IDENTICAL
        # (including Optional and default values); torch infers the schema
        # from annotations.
        ...

    direct_register_custom_op(
        op_name="npu_pangu_<name>",
        op_func=npu_pangu_<name>,
        mutates_args=[],
        fake_impl=npu_pangu_<name>_fake,
        dispatch_key="PrivateUse1",
    )

==============================================================================
Hard constraints (learned the hard way — please respect)
==============================================================================

(a) `mutates_args` must stay `[]` for ops in this file, even when the body
    mutates kv_cache / conv_states in place. Declaring the cache tensors
    (especially any `Optional[Tensor]` slot) as mutated triggers
    autograd / AOT functionalization errors on this repo's paged-cache
    trace path. The mutation stays hidden inside the opaque op boundary.
    Ordering with downstream consumers is NOT preserved automatically by
    call order — FX / npugraph_ex can and will reorder back-to-back opaque
    `torch.ops.vllm.*` calls without an explicit SSA edge. To force the
    order, return the mutated tensor(s) from the op and have the caller
    consume the returns in the next op (see `npu_pangu_indexer_cache_update`
    → `npu_pangu_lightning_indexer` for the canonical pattern).

    Exception: the single-Tensor inplace pattern used by
    `npu_pangu_kv_down_mome_inplace` — it declares `mutates_args=["kv"]`
    AND returns the same `kv` tensor. This collapses to schema
    `Tensor(a!) -> Tensor(a!)`, which IS accepted. It only works because:
      * exactly one Tensor argument is mutated,
      * that argument is a leaf tensor (not a view / slice),
      * the return is that same tensor.
    Don't try to generalize this to multi-arg or Optional-arg cases.

(b) Supported return types: `Tensor`, `tuple[Tensor, ...]`, or `None`.
    Do NOT use
      * `list[Tensor]` whose elements alias `mutates_args` inputs
                                           — AOT functionalization rejects
        "output has alias annotations" at trace time.
    If you need to express "this op mutates several tensors and the next
    op consumes them", return those tensors (as a `tuple[Tensor, ...]`) and
    have the caller bind / forward the returns — that's the only reliable
    way to establish the SSA edge that prevents the two ops from being
    reordered (see (a)).

(c) The fake_impl signature must be byte-identical to the real op's
    signature (same arg names, same annotations, same defaults). Torch
    infers the op schema from the real op's annotations; any mismatch
    produces opaque registration failures.

(d) View-of-input as a mutated arg (`mutates_args=["x"]` where `x` is
    `big_tensor[:, :n]` at the call site) trips AOT's view-input-mutation
    assertion. Work around it the way `npu_pangu_kv_down_mome_inplace`
    does: take the WHOLE leaf tensor as input and do the slicing INSIDE
    the op body.
"""

from typing import Optional

import torch

from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_layer_and_attn_metadata(layer_name: str):
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata.get(f"{layer.prefix}.attn")
    return layer, attn_metadata


def _indexer_cache_tuple(
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    kv_cache_2: Optional[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if kv_cache_2 is None:
        return (kv_cache_0, kv_cache_1)
    return (kv_cache_0, kv_cache_1, kv_cache_2)


# ---------------------------------------------------------------------------
# npu_pangu_swa_decode: wraps _apply_SWA_attention_decode
# ---------------------------------------------------------------------------


def npu_pangu_swa_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer, attn_metadata = _lookup_layer_and_attn_metadata(layer_name)
    return layer._apply_SWA_attention_decode(
        q_nope, q_pe, (kv_cache_0, kv_cache_1), attn_metadata,
    )


def npu_pangu_swa_decode_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer, _ = _lookup_layer_and_attn_metadata(layer_name)
    num_tokens = q_nope.size(0)
    # Latent [T, N, L] — v_up (W_UV absorb) is deferred to _mla_epilog
    # so it can overlap with side-stream work fired by pre_epilog_callback.
    return torch.empty(
        (num_tokens, layer.num_local_heads, layer.kv_lora_rank),
        device=q_nope.device, dtype=q_nope.dtype,
    )


direct_register_custom_op(
    op_name="npu_pangu_swa_decode",
    op_func=npu_pangu_swa_decode,
    mutates_args=[],
    fake_impl=npu_pangu_swa_decode_fake,
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# npu_pangu_indexer_cache_update: wraps Indexer._update_indexer_cache.
# In-place writes into kv_cache; returns (kv_cache_0, kv_cache_1) so the
# downstream lightning_indexer call gets an explicit SSA edge — without
# it, FX / npugraph_ex may reorder the two opaque ops and lightning_indexer
# reads stale cache (precision bug). Callers must consume the returns:
# `kv_cache = torch.ops.vllm.npu_pangu_indexer_cache_update(...)` so the
# next `kv_cache[0]` / `kv_cache[1]` read picks up the dependency.
# (See the module docstring "Hard constraints" for why mutates_args=[].)
# The wrapper takes only (k, kv_cache..., layer_name) and forwards the
# live attn_metadata it pulls from forward_context to the underlying
# method, which derives slot_mapping / slot_mapping_2d itself. This keeps
# the wrapper compatible with both the master signature
# `_update_indexer_cache(k, attention_metadata, kv_cache)` and any future
# refactor that swaps in a 2D slot mapping internally.
# ---------------------------------------------------------------------------


def npu_pangu_indexer_cache_update(
    k: torch.Tensor,
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    kv_cache_2: Optional[torch.Tensor],
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer, attn_metadata = _lookup_layer_and_attn_metadata(layer_name)
    kv_cache = _indexer_cache_tuple(kv_cache_0, kv_cache_1, kv_cache_2)
    layer.indexer._update_indexer_cache(k, attn_metadata, kv_cache)
    return kv_cache_0, kv_cache_1


def npu_pangu_indexer_cache_update_fake(
    k: torch.Tensor,
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    kv_cache_2: Optional[torch.Tensor],
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(kv_cache_0), torch.empty_like(kv_cache_1)


direct_register_custom_op(
    op_name="npu_pangu_indexer_cache_update",
    op_func=npu_pangu_indexer_cache_update,
    mutates_args=[],
    fake_impl=npu_pangu_indexer_cache_update_fake,
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# npu_pangu_lightning_indexer: wraps Indexer._apply_lightning_indexer
# (pure compute over kv_cache; returns topk_indices)
# ---------------------------------------------------------------------------


def npu_pangu_lightning_indexer(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    kv_cache_2: Optional[torch.Tensor],
    layer_name: str,
) -> torch.Tensor:
    layer, attn_metadata = _lookup_layer_and_attn_metadata(layer_name)
    kv_cache = _indexer_cache_tuple(kv_cache_0, kv_cache_1, kv_cache_2)
    return layer.indexer._apply_lightning_indexer(
        q, weights, attn_metadata, kv_cache,
    )


def npu_pangu_lightning_indexer_fake(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    kv_cache_2: Optional[torch.Tensor],
    layer_name: str,
) -> torch.Tensor:
    layer, _ = _lookup_layer_and_attn_metadata(layer_name)
    return torch.empty(
        (q.size(0), 1, layer.indexer.index_topk),
        device=q.device,
        dtype=torch.int32,
    )


direct_register_custom_op(
    op_name="npu_pangu_lightning_indexer",
    op_func=npu_pangu_lightning_indexer,
    mutates_args=[],
    fake_impl=npu_pangu_lightning_indexer_fake,
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# npu_pangu_kv_cache_update: wraps _npu_kvrmsnorm_rope_cache (single kv input)
# Only applicable to paths that return a 2-tuple (kv_cache, topk_indices),
# i.e. DSA or absorb paths. Non-absorb prefill returns 3 tensors and is
# kept on the original direct call.
# ---------------------------------------------------------------------------


def npu_pangu_kv_cache_update(
    kv: torch.Tensor,
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer, attn_metadata = _lookup_layer_and_attn_metadata(layer_name)
    kv_cache, _ = layer._npu_kvrmsnorm_rope_cache(
        kv, (kv_cache_0, kv_cache_1), cos, sin, attn_metadata, None,
    )
    return kv_cache


def npu_pangu_kv_cache_update_fake(
    kv: torch.Tensor,
    kv_cache_0: torch.Tensor,
    kv_cache_1: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(kv_cache_0), torch.empty_like(kv_cache_1)


direct_register_custom_op(
    op_name="npu_pangu_kv_cache_update",
    op_func=npu_pangu_kv_cache_update,
    mutates_args=[],
    fake_impl=npu_pangu_kv_cache_update_fake,
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# npu_pangu_mome_fc2_scatter_and_return: wraps the FC2 mome scatter step
# ---------------------------------------------------------------------------


def npu_pangu_mome_fc2_scatter_and_return(
    gathered_results: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_indices_rearranged: torch.Tensor,
    all_caches: torch.Tensor,
    rearrange_ratio: int,
) -> torch.Tensor:
    _, state_len, dim = all_caches.shape
    torch.ops.custom.npu_ai_infra_scatter_block_update_(
        kv_cache.view(-1, state_len * rearrange_ratio, dim // rearrange_ratio),
        cache_indices_rearranged,
        all_caches.view(-1, dim // rearrange_ratio),
    )
    return gathered_results


def npu_pangu_mome_fc2_scatter_and_return_fake(
    gathered_results: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_indices_rearranged: torch.Tensor,
    all_caches: torch.Tensor,
    rearrange_ratio: int,
) -> torch.Tensor:
    return torch.empty_like(gathered_results)


direct_register_custom_op(
    op_name="npu_pangu_mome_fc2_scatter_and_return",
    op_func=npu_pangu_mome_fc2_scatter_and_return,
    mutates_args=[],
    fake_impl=npu_pangu_mome_fc2_scatter_and_return_fake,
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# npu_pangu_mome_conv: wraps npu_ai_infra_fused_causal_conv1d (mode 0 / mode 1)
# Replaces direct conv1d calls inside _apply_MOME's SP branch and the
# layer.forward call on non-Ascend950 path.
# ---------------------------------------------------------------------------


def npu_pangu_mome_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    num_computed_tokens: torch.Tensor,
    block_idx_first_scheduled_token: Optional[torch.Tensor],
    block_idx_last_scheduled_token: Optional[torch.Tensor],
    initial_state_idx: Optional[torch.Tensor],
    pad_slot_id: int,
    max_query_len: int,
    block_size: int,
    mode: int,
    inplace: bool,
) -> torch.Tensor:
    if mode == 0:
        return torch.ops.custom.npu_ai_infra_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=None,
            num_accepted_tokens=None,
            num_computed_tokens=num_computed_tokens,
            residual_connection=1,
            conv_mode=1,
            inplace=inplace,
        )

    return torch.ops.custom.npu_ai_infra_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        pad_slot_id=pad_slot_id,
        max_query_len=max_query_len,
        residual_connection=1,
        block_size=block_size,
        conv_mode=1,
        inplace=inplace,
    )


def npu_pangu_mome_conv_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    num_computed_tokens: torch.Tensor,
    block_idx_first_scheduled_token: Optional[torch.Tensor],
    block_idx_last_scheduled_token: Optional[torch.Tensor],
    initial_state_idx: Optional[torch.Tensor],
    pad_slot_id: int,
    max_query_len: int,
    block_size: int,
    mode: int,
    inplace: bool,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="npu_pangu_mome_conv",
    op_func=npu_pangu_mome_conv,
    mutates_args=[],
    fake_impl=npu_pangu_mome_conv_fake,
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# npu_pangu_kv_down_mome_inplace: KV-down MoME with whole-kv input
#
# Use case: _kv_down_mome's non-Ascend950 path needs to apply MoME conv
# inplace on kv[:, :kv_lora_rank] without an explicit cat afterwards.
# Declaring mutates_args=["x"] on npu_pangu_mome_conv when x is a slice
# triggers AOT's view-input-mutation assertion. This specialized wrapper
# takes the WHOLE kv (a leaf tensor from kv_a_proj) so mutates_args=["kv"]
# is safe — kv is not a view. Slice + inplace happen inside the opaque op.
# ---------------------------------------------------------------------------


def npu_pangu_kv_down_mome_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    num_computed_tokens: torch.Tensor,
    block_idx_first_scheduled_token: Optional[torch.Tensor],
    block_idx_last_scheduled_token: Optional[torch.Tensor],
    initial_state_idx: Optional[torch.Tensor],
    pad_slot_id: int,
    max_query_len: int,
    block_size: int,
    kv_lora_rank: int,
    num_actual_tokens: int,
) -> torch.Tensor:
    torch.ops.custom.npu_ai_infra_fused_causal_conv1d(
        kv[:num_actual_tokens, :kv_lora_rank],
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        pad_slot_id=pad_slot_id,
        max_query_len=max_query_len,
        residual_connection=1,
        block_size=block_size,
        conv_mode=1,
        inplace=True,
    )
    return kv


def npu_pangu_kv_down_mome_inplace_fake(
    kv: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    num_computed_tokens: torch.Tensor,
    block_idx_first_scheduled_token: Optional[torch.Tensor],
    block_idx_last_scheduled_token: Optional[torch.Tensor],
    initial_state_idx: Optional[torch.Tensor],
    pad_slot_id: int,
    max_query_len: int,
    block_size: int,
    kv_lora_rank: int,
    num_actual_tokens: int,
) -> torch.Tensor:
    return kv


direct_register_custom_op(
    op_name="npu_pangu_kv_down_mome_inplace",
    op_func=npu_pangu_kv_down_mome_inplace,
    mutates_args=["kv"],
    fake_impl=npu_pangu_kv_down_mome_inplace_fake,
    dispatch_key="PrivateUse1",
)
