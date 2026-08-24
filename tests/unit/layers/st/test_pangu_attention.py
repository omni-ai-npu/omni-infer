# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Single-layer Pangu attention precision tests (real NPU, TP=2).

Guards the PD-disaggregation + APC + chunked-prefill combination via in-test
invariants (no golden file), on each implementation's TP=2 communication path.
APC and chunked prefill run on the prefill (kv_producer) node:

  - chunked prefill (>= 3 chunks) == single full prefill
  - APC prefix-cache hit == full prefill (suffix positions)
  - chunked prefill + APC combined == full prefill (suffix)

with chunk / prefix boundaries on / around block_size multiples (a known
edge-case for accuracy drift). On the low-latency sequence-parallel path, token
counts not divisible by TP (the ±1 cases) are padded to a TP multiple before
sharding, matching the model runner.

Covers DSA (layer 0, index_topk sparse) and SWA (layer 1, sliding-window MLA)
for both low-latency and high-performance implementations, with MoME enabled.
The low-latency path uses li_int8_ds_mla KV cache; the high-performance path
uses its native bf16 cache. Every test runs in a 2-process
distributed_worker_pool worker (skipped on < 2 NPUs).
"""

from __future__ import annotations

import pytest
import torch

try:
    import torch_npu  # noqa: F401
    _NPU_COUNT = torch.npu.device_count() if hasattr(torch, "npu") else 0
except Exception:
    _NPU_COUNT = 0

pytestmark = pytest.mark.skipif(
    _NPU_COUNT < 2, reason="requires >= 2 NPUs with torch_npu (default TP=2)"
)


def _mome_available() -> bool:
    """MomeAttention is injected into vllm.model_executor.layers.npumome by
    patch_mome_hybrid (patch_mome.py / patch_mome_hybrid.py), which has not
    been migrated to the usefull_patch directory yet. Without it, npu_dsa.py /
    npu_mla.py raise NameError on the high-performance construction path."""
    try:
        from vllm.model_executor.layers.npumome import MomeAttention  # noqa: F401
        return True
    except Exception:
        return False


# Conditional skip so the high-performance cases auto-recover once the
# underlying product patch (patch_mome / patch_mome_hybrid) is in place. The
# low-latency cases run unconditionally: bootstrap_worker triggers
# NPUPlatform.pre_register_and_update(), which activates the MLA prefill
# backend override (omni/platform.py).
MOME_MISSING = pytest.mark.skipif(
    not _mome_available(),
    reason="MomeAttention not registered: patch_mome / patch_mome_hybrid not yet "
           "migrated to usefull_patch",
)

from . import pangu_attention_st_common as H  # noqa: E402
from .distributed_test_common import distributed_worker_pool  # noqa: E402,F401

BLOCK = H.BLOCK_SIZE
CHUNK = H.CHUNK_SIZE  # 256 = 2 * block_size
NUM_BLOCKS = 64
C = H.PrefillCase

# Default runtime config: tensor-parallel size 2.
TP2 = {"world_size": 2, "tp_size": 2, "dp_size": 1,
       "enable_expert_parallel": False}

LAYER_CASES = [
    pytest.param(H.DSA_LAYER_IDX, id="dsa"),
    pytest.param(H.SWA_LAYER_IDX, id="swa"),
]

# Context-Parallel (DSA zigzag) only applies to the DSA layer; the SWA layer
# never takes the CP forward path (is_cp_layer requires is_dsa_layer).
DSA_ONLY = [pytest.param(H.DSA_LAYER_IDX, id="dsa")]

# FlashComm2 only applies to the SWA layer (the DSA layer never takes the FC2
# forward path: enable_flashcomm2 requires not is_dsa_layer).
SWA_ONLY = [pytest.param(H.SWA_LAYER_IDX, id="swa")]

# Edge cases, each fully described by (total tokens, chunk size, APC hit blocks).
# To guard a newly-found edge case, just append one PrefillCase here.
CASES = [
    # chunked prefill, no APC (hit_blocks=0):
    # 3 full chunks, ends mid-block (tail=1): chunked accumulation + partial tail.
    C(total=2 * CHUNK + 1, chunk=CHUNK, hit_blocks=0),  # [256,256,1]
    # last chunk's query crosses a block boundary (fresh multi-block write).
    C(total=CHUNK + BLOCK + 1, chunk=CHUNK, hit_blocks=0),  # [256,129]
    # finer granularity: one block per chunk, every boundary block-aligned.
    C(total=4 * BLOCK, chunk=BLOCK, hit_blocks=0),  # [128,128,128,128]

    # real APC prefix-cache hit (hit_blocks>0):
    # shallow hit (1 cached block), suffix stays within one block (single forward).
    C(total=BLOCK + 64, chunk=CHUNK, hit_blocks=1),  # ctx=128, suffix=[128,192)
    # deeper hit (2 cached blocks), suffix crosses a block boundary.
    C(total=CHUNK + BLOCK + 1, chunk=CHUNK, hit_blocks=2),  # ctx=256, suffix=129
    # hit + chunked suffix: the suffix is re-split into >1 forward (chunk+apc).
    C(total=2 * CHUNK + 1, chunk=CHUNK, hit_blocks=2),  # ctx=256, suffix=[256,1]
]

# PD-disaggregation path: prefill (APC + chunked prefill) on the kv_producer
# node (is_pd_disagg=True).
@pytest.mark.parametrize("layer_idx", LAYER_CASES)
def test_pd_producer_invariants(distributed_worker_pool, layer_idx):
    """TP=2 prefill (kv_producer) node, is_pd_disagg=True: each CASE's chunked /
    APC / chunk+APC invariant == full prefill."""
    distributed_worker_pool(
        H.mc_invariants_worker, layer_idx, CASES,
        NUM_BLOCKS, "kv_producer", config={}, runtime_config=TP2,
    )


# High-throughput implementation of the same Pangu DSA/SWA layers.  Reuse the
# exact cases and invariants above, but construct NPUDeepseekSparseAttention
# (npu_dsa.py) or NPUDeepseekMLAAttention (npu_mla.py) in the shared harness.
# These layers use conventional tensor-parallel full-token inputs/outputs;
# pangu_attention_st_common hides that contract difference from the cases.
@MOME_MISSING
@pytest.mark.parametrize("layer_idx", LAYER_CASES)
def test_pd_producer_high_performance_invariants(
    distributed_worker_pool, layer_idx,
):
    """TP=2 high-performance prefill producer: chunked/APC invariants equal
    a full prefill for both DSA and sliding-window MLA."""
    distributed_worker_pool(
        H.mc_invariants_worker, layer_idx, CASES,
        NUM_BLOCKS, "kv_producer", False, False, "high_performance",
        config={}, runtime_config=TP2,
    )


# Context-Parallel (CP) DSA prefill path (_forward_prefill_cp): the DSA-layer
# zig-zag sequence-parallel path the kv_producer node uses (ena_seq_parallel +
# ena_context_parallel). Same _sp_shard_forward contract as standard TP; the
# layer does the zig-zag all_to_all / all_gather internally. DSA-only
# (is_cp_layer requires is_dsa_layer).
@pytest.mark.parametrize("layer_idx", DSA_ONLY)
def test_pd_producer_cp_invariants(distributed_worker_pool, layer_idx):
    """TP=2 prefill (kv_producer) node, Context-Parallel DSA: each CASE's
    chunked / APC / chunk+APC invariant == full prefill on the CP zig-zag path."""
    distributed_worker_pool(
        H.mc_invariants_worker, layer_idx, CASES,
        NUM_BLOCKS, "kv_producer", True, config={}, runtime_config=TP2,
    )


# High-performance DSA has a separate _forward_prefill_cp implementation in
# npu_dsa.py. Keep this DSA-only case distinct so CP cannot silently fall back
# to the ordinary high-performance _forward_prefill path.
@MOME_MISSING
@pytest.mark.parametrize("layer_idx", DSA_ONLY)
def test_pd_producer_high_performance_cp_invariants(
    distributed_worker_pool, layer_idx
):
    """TP=2 high-performance Context-Parallel DSA producer: each CASE's
    chunked/APC invariant equals full prefill on _forward_prefill_cp."""
    distributed_worker_pool(
        H.mc_invariants_worker, layer_idx, CASES,
        NUM_BLOCKS, "kv_producer", True, False, "high_performance",
        config={}, runtime_config=TP2,
    )


# FlashComm2 (FC2) SWA prefill path (_forward_prefill_FC2), a TP-communication
# optimization for SWA-layer prefill (enable_flashcomm2). Input arrives ungathered
# (TP-local) and the global all_gather is skipped; the layer all-gathers the
# smaller q_lora / kv internally and a replicated o_proj emits an SP-layout output
# -- the same SP-sharded contract _sp_shard_forward already drives, so no special
# wiring is needed. With prefix caching enabled (as the rest of the suite uses),
# the FC2 mome sequence-parallel branch (maybe_enable_sp_for_mome) is disabled, so
# the layer takes the all-gather-heads path that does not touch fc2_metadata.
# SWA-only (the DSA layer never takes the FC2 path); guards all three invariants
# on the FC2 forward, mirroring the standard SWA test.
@pytest.mark.parametrize("layer_idx", SWA_ONLY)
def test_pd_producer_fc2_invariants(distributed_worker_pool, layer_idx):
    """TP=2 prefill (kv_producer) node, FlashComm2 SWA: each CASE's chunked /
    APC / chunk+APC invariant == full prefill on the FC2 forward."""
    distributed_worker_pool(
        H.mc_invariants_worker, layer_idx, CASES,
        NUM_BLOCKS, "kv_producer", False, True, config={}, runtime_config=TP2,
    )
