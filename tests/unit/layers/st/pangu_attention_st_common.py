# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared harness for single-layer Pangu attention precision tests.

Drives either the low-latency NPUPanguSparseAttention or the high-performance
NPUDeepseekSparseAttention/NPUDeepseekMLAAttention on a real NPU through the
production metadata builders. Checks invariants of the PD-disaggregation / APC
/ chunked-prefill combination: chunked prefill and an APC prefix-cache hit must
each match a single full prefill numerically, for the same token positions. The
reference is computed in-test, so no golden file is needed. Edge cases place
chunk / prefix boundaries on block_size multiples.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import torch
import torch.nn.functional as F

# Mirrors the Pangu V2 505B config; dims are real so kernels see the same
# shapes. Layer 0 is DSA, layer 1 is SWA.
PANGU_HF_CONFIG = dict(
    dtype=torch.bfloat16,
    hidden_size=5120,
    num_attention_heads=64,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=512,
    q_lora_rank=1536,
    index_topk=2048,
    index_head_dim=128,
    index_n_heads=32,
    rms_norm_eps=1e-5,
    rope_theta=6400000,
    rope_interleaved=False,
    param_sink_number=128,
    router_sliding_window=3,
    use_mome=True,
    num_hidden_layers=53,
    max_position_embeddings=4096,
    swa_layers=[1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25,
                26, 28, 29, 31, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46, 47, 50,
                51, 52],
    dsa_layers=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
                49],
    sliding_window_list=[512] * 32 + [2048] * 3,
)

DSA_LAYER_IDX = 0
SWA_LAYER_IDX = 1
BLOCK_SIZE = 128
# Chunked-prefill granularity. A multiple of block_size so every chunk boundary
# except the tail lands on a block multiple (the historical accuracy edge case);
# only the total prompt length is varied to sweep the tail boundary conditions.
CHUNK_SIZE = 2 * BLOCK_SIZE
# Fixed blocks per request so a request occupies identical physical blocks
# across a standalone (reference) run and a tested run.
BLOCKS_PER_REQ = 16
MAX_MODEL_LEN = 4096
NPU = "npu"

_CURRENT_CFG_CM = None


def _set_model_extra_config(enable_cp: bool = False,
                            enable_fc2: bool = False,
                            implementation: str = "low_latency") -> None:
    from omni_npu.model_config.config_loader.loader import model_extra_config
    op = model_extra_config.operator_opt_config
    pc = model_extra_config.parall_config
    op.use_noncontiguous_kv = True
    # The HP sink-attention AICPU metadata kernel rejects TP=2 synthetic
    # single-layer metadata on A3.  That tiling optimisation is orthogonal to
    # the attention/APC invariants guarded here, so use the regular HP kernel;
    # retain the production low-latency setting for the existing suite.
    op.use_aicpu_fa_tiling = implementation == "low_latency"
    op.optimize_first_chunk = False  # -> first_chunk_pa = True (PA path)
    pc.ena_context_parallel = enable_cp
    pc.enable_flashcomm2 = enable_fc2
    # Only CP needs ena_seq_parallel: the DSA builder attaches a real SPManager to
    # prefill metadata under this flag. The FC2 SWA path does not read it.
    pc.ena_seq_parallel = enable_cp
    # HP DSA CP may produce uneven per-rank token counts for tail chunks.
    # Production uses the sharded o_proj variant for this contract; unlike a
    # token-dimension ReduceScatter it also supports non-TP-divisible tails.
    pc.sharded_o_proj = enable_cp and implementation == "high_performance"


def bootstrap_worker(kv_role: Optional[str] = None,
                     enable_cp: bool = False, enable_fc2: bool = False,
                     implementation: str = "low_latency"):
    """Bootstrap inside a distributed_worker_pool worker (distributed already
    initialized by the worker loop): apply patches, set the model extra config
    (optionally enabling CP / FC2) and a persistent current VllmConfig.

    Returns a VllmConfig whose TP matches the live tensor-parallel group.
    """
    global _CURRENT_CFG_CM
    import torch_npu  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_tp_group
    from omni_npu.vllm_patches import apply_patches

    # This is a manual multi-directory selection, so the patch loader does not
    # expand pangu_v2_hybrid's dependency mapping. List every dependency
    # explicitly; pangu_sink_swa_mla provides SinkMLAAttentionSpec used by the
    # high-performance DSA/MLA StaticSinkMLAAttention cache spec.
    os.environ["OMNI_NPU_PATCHES_DIR"] = (
        "pangu_v2_base,pangu_sink_swa_mla,pangu_v2_hybrid,pangu_v2_moe"
    )
    os.environ.setdefault("OMNI_NPU_VLLM_PATCHES", "ALL")
    os.environ.setdefault("OMNI_REUSE_PREFILLED_TOKENS", "1")
    # Single-host multi-process: give each rank's HCCL socket a distinct port.
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "60000-60100")
    apply_patches()
    # The engine boot path (EngineArgs.create_engine_config / add_cli_args /
    # NPUWorker) invokes NPUPlatform.pre_register_and_update(), which is where
    # the MLA prefill backend override is registered (omni/platform.py). This
    # harness builds VllmConfig directly, so trigger the same pre-registration
    # explicitly -- otherwise FLASH_ATTN resolves to the vLLM CUDA-only
    # implementation and prefill fails with an ImportError on NPU.
    from vllm.platforms import current_platform
    current_platform.pre_register_and_update()
    torch.set_default_dtype(torch.bfloat16)
    _set_model_extra_config(
        enable_cp=enable_cp, enable_fc2=enable_fc2,
        implementation=implementation,
    )
    tp_size = get_tp_group().world_size
    cfg = build_vllm_config(
        kv_role, tp_size=tp_size, implementation=implementation,
    )
    if _CURRENT_CFG_CM is None:
        _CURRENT_CFG_CM = set_current_vllm_config(cfg)
        _CURRENT_CFG_CM.__enter__()
    return cfg


def build_hf_config() -> SimpleNamespace:
    hf = SimpleNamespace(**PANGU_HF_CONFIG)
    hf.rope_parameters = {
        "rope_theta": hf.rope_theta,
        "rope_type": "deepseek_yarn",
        "factor": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": hf.max_position_embeddings,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    return hf


def build_vllm_config(kv_role: Optional[str], tp_size: int = 1,
                      implementation: str = "low_latency"):
    from vllm.config import VllmConfig
    cfg = VllmConfig()
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.cache_config.block_size = BLOCK_SIZE
    cfg.cache_config.enable_prefix_caching = True
    # li_int8_ds_mla is a low-latency Pangu cache format.  The standalone
    # high-throughput DSA/MLA classes use their native bf16 cache path.
    cfg.cache_config.cache_dtype = (
        "li_int8_ds_mla" if implementation == "low_latency" else "auto"
    )
    # The high-performance attention classes use the patched vLLM
    # MomeAttention, whose cache spec reads these values.  Production derives
    # the padded size while verifying the model config; this single-layer
    # harness builds VllmConfig directly, so provide the equivalent minimum.
    cfg.cache_config.mamba_block_size = BLOCK_SIZE
    mome_tokens = PANGU_HF_CONFIG["router_sliding_window"] - 1
    if kv_role is not None:
        mome_tokens += 1  # fake speculative slot used by PD disaggregation
    mome_page_size = (
        PANGU_HF_CONFIG["q_lora_rank"]
        + PANGU_HF_CONFIG["kv_lora_rank"]
        + PANGU_HF_CONFIG["num_attention_heads"]
        * PANGU_HF_CONFIG["v_head_dim"]
    ) * torch.tensor([], dtype=torch.bfloat16).element_size() * mome_tokens
    dsa_page_size = (
        PANGU_HF_CONFIG["kv_lora_rank"]
        + PANGU_HF_CONFIG["qk_rope_head_dim"]
        + PANGU_HF_CONFIG["index_head_dim"]
    ) * torch.tensor([], dtype=torch.bfloat16).element_size() * BLOCK_SIZE
    cfg.cache_config.mamba_page_size_padded = max(
        mome_page_size, dsa_page_size,
    )
    cfg.scheduler_config.enable_chunked_prefill = True
    cfg.scheduler_config.max_num_seqs = 8
    cfg.scheduler_config.max_num_batched_tokens = 8192
    if kv_role is not None:
        cfg.kv_transfer_config = SimpleNamespace(
            kv_role=kv_role,
            is_kv_producer=(kv_role == "kv_producer"),
            is_kv_consumer=(kv_role == "kv_consumer"),
            # vLLM 0.25 selector.py reads is_kv_transfer_instance directly to
            # decide whether to use the KV-connector backend. A PD producer /
            # consumer role simulates an active KV transfer instance.
            is_kv_transfer_instance=(kv_role in ("kv_producer", "kv_consumer")),
        )
    hf = build_hf_config()

    class _ModelConfig:
        def __init__(self):
            self.dtype = torch.bfloat16
            self.max_model_len = MAX_MODEL_LEN
            self.hf_config = hf
            self.hf_text_config = hf
            self.is_moe = True
            self.use_mla = True

        def get_head_size(self):
            return hf.kv_lora_rank + hf.qk_rope_head_dim

        def get_num_attention_heads(self, parallel_config=None):
            return hf.num_attention_heads

        def get_num_kv_heads(self, parallel_config=None):
            return 1

    cfg.model_config = _ModelConfig()
    return cfg


def build_layer(vllm_config, layer_idx: int, seed: int = 0,
                implementation: str = "low_latency"):
    from vllm.config import set_current_vllm_config
    hf = build_hf_config()
    # Build under this config so attn/mome register into its (per-config)
    # static_forward_context, avoiding "Duplicate layer name" collisions between
    # PD / non-PD variants sharing the same layer prefix.
    with torch.device(NPU), set_current_vllm_config(vllm_config):
        common_kwargs = dict(
            vllm_config=vllm_config, config=hf,
            hidden_size=hf.hidden_size, num_heads=hf.num_attention_heads,
            qk_nope_head_dim=hf.qk_nope_head_dim,
            qk_rope_head_dim=hf.qk_rope_head_dim,
            v_head_dim=hf.v_head_dim, q_lora_rank=hf.q_lora_rank,
            kv_lora_rank=hf.kv_lora_rank,
            max_position_embeddings=hf.max_position_embeddings,
            cache_config=vllm_config.cache_config, quant_config=None,
            prefix=f"model.layers.{layer_idx}.self_attn",
        )
        if implementation == "low_latency":
            from omni_npu.v1.layers.attention.npu_pangu import (
                NPUPanguSparseAttention,
            )
            layer = NPUPanguSparseAttention(
                **common_kwargs, rope_theta=hf.rope_theta,
                swa_layers=hf.swa_layers,
                param_sink_number=hf.param_sink_number,
                sliding_window_list=hf.sliding_window_list,
            )
        elif implementation == "high_performance":
            if layer_idx in hf.dsa_layers:
                from omni_npu.v1.layers.attention.npu_dsa import (
                    NPUDeepseekSparseAttention,
                )
                layer_cls = NPUDeepseekSparseAttention
            else:
                from omni_npu.v1.layers.attention.npu_mla import (
                    NPUDeepseekMLAAttention,
                )
                layer_cls = NPUDeepseekMLAAttention
                # The 505B HP sink kernel is tied to its production TP=4 head
                # partition, while this shared ST fixture deliberately exposes
                # two devices.  Keep this TP=2 test focused on the HP SWA/MLA,
                # MoME, chunked-prefill and APC paths; sink-kernel behavior has
                # dedicated coverage in test_npu_mla.py.
                hf.param_sink_number = 0
            layer = layer_cls(**common_kwargs)
            # Normalise the two production implementations behind the small
            # interface used by this shared test harness.
            layer.is_dsa_layer = layer_idx in hf.dsa_layers
            layer.layer_name = layer.prefix
        else:
            raise ValueError(f"unknown attention implementation: {implementation}")
    _init_weights(layer, seed)
    _transpose_flashcomm_weights(layer)
    if implementation == "low_latency":
        layer.process_weights_after_loading()
    else:
        _prepare_high_performance_weights(layer)
    layer._test_implementation = implementation
    return layer


def _init_weights(layer, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for _, p in layer.named_parameters():
            if p.is_floating_point():
                p.data.normal_(0, 0.02)


def _transpose_flashcomm_weights(layer) -> None:
    # FlashComm linears use matmul(x, weight) and expect [in, out] layout. Do
    # the transpose directly to avoid the vLLM base PWAL JIT-compile init path.
    from omni_npu.v1.layers.linear import FlashCommLinearBase
    with torch.no_grad():
        for m in layer.modules():
            if isinstance(m, FlashCommLinearBase) and hasattr(m, "weight"):
                if getattr(m.weight, "is_weight_transposed", False):
                    continue
                m.weight.data = m.weight.data.t().contiguous()
                m.weight.is_weight_transposed = True


def _prepare_high_performance_weights(layer) -> None:
    """Complete the model-loader-only setup for a standalone HP layer.

    The normal loader transposes FlashComm weights and derives the MLA absorb
    matrices in separate module callbacks.  This harness initialises weights
    directly, so reproduce that deterministic derivation here.
    """
    with torch.no_grad():
        if getattr(layer, "sharded_o_proj", False):
            # ShardedLinear's loader stores one byte shard per rank and installs
            # the gather callback used by prefetch().  The normal model loader
            # invokes it with the full checkpoint tensor; this standalone test
            # initialises parameters directly, so reproduce that step here.
            loaded = torch.empty(
                layer.hidden_size,
                layer.num_heads * layer.v_head_dim,
                dtype=layer.default_cfg["dtype"],
                device=layer.o_proj.weight.device,
            )
            loaded.normal_(0, 0.02)
            layer.o_proj.weight.weight_loader(layer.o_proj.weight, loaded)

        kv_weight = layer.kv_b_proj.weight.view(
            layer.kv_lora_rank,
            layer.num_local_heads,
            layer.qk_nope_head_dim + layer.v_head_dim,
        )
        w_uk, w_uv = kv_weight.split(
            [layer.qk_nope_head_dim, layer.v_head_dim], dim=-1,
        )
        layer.attn.impl.W_UK_T = w_uk.permute(1, 2, 0).contiguous()
        layer.attn.impl.W_UV = w_uv.transpose(0, 1).contiguous()

        if layer.param_sink_number > 0:
            sink_compressed_kv = layer.kv_a_layernorm(
                layer.param_sink_compressed_kv
            )
            layer.attn.update_sink_kv(
                layer.param_sink_k_pe, sink_compressed_kv,
            )


def alloc_caches(layer, vllm_config, num_blocks: int) -> None:
    """Allocate attn + mome caches via the production reshape_kv_cache."""
    from omni_npu.attention.backends.mla import NPUMLABackend
    from omni_npu.attention.backends.dsa import NPUDSABackend
    from omni_npu.attention.backends.mome import NPUPanguMomeBackend

    spec = layer.attn.get_kv_cache_spec(vllm_config)
    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8,
                      device=NPU)
    backend = NPUDSABackend if layer.is_dsa_layer else NPUMLABackend
    attn_kv_cache = backend.reshape_kv_cache(raw, num_blocks, spec)

    mome_layer, _ = _mome_layer_and_name(layer)
    mspec = mome_layer.get_kv_cache_spec(vllm_config)
    mraw = torch.zeros(num_blocks * mspec.page_size_bytes, dtype=torch.int8,
                       device=NPU)
    mome_kv_cache = NPUPanguMomeBackend.reshape_kv_cache(
        mraw, num_blocks, mspec,
    )

    if getattr(layer, "_test_implementation", "low_latency") == "high_performance":
        # HP layers follow the vLLM bind convention: one entry per virtual
        # engine, read as layer.kv_cache[forward_context.virtual_engine].
        layer.attn.kv_cache = [attn_kv_cache]
        mome_layer.kv_cache = [mome_kv_cache]
    else:
        # Low-latency NPUPanguSparseAttention indexes the reshape tuple
        # directly (attn.kv_cache[i] / mome_attn.kv_cache[mome_cache_index]
        # must be individual tensors), mirroring AttentionLayerBase.
        # bind_kv_cache in vLLM 0.25. A list would hand the whole cache
        # tuple to the mome/attn kernels as a single "state".
        layer.attn.kv_cache = attn_kv_cache
        mome_layer.kv_cache = mome_kv_cache


def _mome_layer_and_name(layer):
    """Return the MoME cache owner and its production metadata key."""
    if getattr(layer, "_test_implementation", "low_latency") == "low_latency":
        return layer.mome_attn, f"{layer.layer_name}.mome"
    return layer.conv, f"{layer.layer_name}.conv"


def build_metadata(layer, vllm_config, reqs, num_blocks: int,
                   block_tables=None):
    """Build (attn_meta, mome_meta) via the production builders.

    reqs: list of (context_len, query_len).
    block_tables: optional per-req explicit physical block-id lists, overriding
    the default contiguous ``base = r * BLOCKS_PER_REQ`` layout. Used by the
    real-APC tests to make a follow-up request reuse a previous request's
    physically non-contiguous prefix blocks, as prefix caching does in prod.
    """
    device = torch.device(NPU)
    from omni_npu.attention.backends.mla import NPUMLAMetadataBuilder
    from omni_npu.attention.backends.dsa import NPUDSAMetadataBuilder
    from omni_npu.attention.backends.mome import NPUMomeAttentionMetadataBuilder
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

    prefix = layer.layer_name
    context_lens = [c for c, _ in reqs]
    query_lens = [q for _, q in reqs]
    seq_lens_list = [c + q for c, q in reqs]
    num_reqs = len(reqs)
    total_q = sum(query_lens)

    # query_start_loc: cumulative per-request query offsets, e.g. [0, q0, q0+q1].
    query_offsets = [0]
    for q in query_lens:
        query_offsets.append(query_offsets[-1] + q)
    qsl = torch.tensor(query_offsets, dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    num_computed_cpu = torch.tensor(context_lens, dtype=torch.int32)

    assert num_reqs * BLOCKS_PER_REQ <= num_blocks, "not enough blocks"
    # Build each request's physical block_table, then map its query token
    # positions [c, c+q) to flat KV-cache slot indices via the block_table.
    block_table = torch.zeros((num_reqs, BLOCKS_PER_REQ), dtype=torch.int32)
    slots = []
    for r, (c, q) in enumerate(reqs):
        if block_tables is not None:
            row = block_tables[r]
            for b in range(BLOCKS_PER_REQ):
                block_table[r, b] = row[b] if b < len(row) else row[-1]
        else:
            base = r * BLOCKS_PER_REQ  # default: contiguous physical blocks
            for b in range(BLOCKS_PER_REQ):
                block_table[r, b] = base + b
        for pos in range(c, c + q):
            blk = int(block_table[r, pos // BLOCK_SIZE].item())
            slots.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
    slot_mapping = torch.tensor(slots, dtype=torch.long)

    common = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens.to(device),
        num_reqs=num_reqs,
        num_actual_tokens=total_q,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens_list),
        block_table_tensor=block_table.to(device),
        slot_mapping=slot_mapping.to(device),
        causal=True,
        # vLLM 0.25.x: the MLA metadata builder reads the CPU seq-len upper
        # bound directly (mla_attention.py build asserts it for prefills);
        # the deprecated _seq_lens_cpu fallback no longer satisfies it.
        seq_lens_cpu_upper_bound=seq_lens,
    )
    common._seq_lens_cpu = seq_lens
    common._num_computed_tokens_cpu = num_computed_cpu

    attn_spec = layer.attn.get_kv_cache_spec(vllm_config)
    builder_cls = NPUDSAMetadataBuilder if layer.is_dsa_layer else NPUMLAMetadataBuilder
    abuilder = builder_cls(attn_spec, [f"{prefix}.attn"], vllm_config, device)
    if (getattr(layer, "_test_implementation", "low_latency") ==
            "high_performance" and getattr(layer, "ena_cp", False)):
        # CP is a prefill-only path. The generic MLA metadata builder normally
        # classifies query_len=1 as decode, including a one-token chunked-
        # prefill tail. Keep every CASE on _forward_prefill_cp instead.
        abuilder.reorder_batch_threshold = 0
    attn_meta = abuilder.build(0, common)
    if getattr(layer, "_test_implementation", "low_latency") == "high_performance":
        # NPUMLAMetadataBuilder may retain Python-list cumlens for eager
        # prefill, while the HP custom-op schema in this environment requires
        # tensors.  Production model-runner preparation performs this
        # normalisation before the layer is invoked.
        for phase in (attn_meta.prefill, attn_meta.decode):
            if phase is None:
                continue
            for field in ("query_cumlens", "seq_lens"):
                value = getattr(phase, field, None)
                if isinstance(value, list):
                    setattr(
                        phase, field,
                        torch.tensor(value, dtype=torch.int32, device=device),
                    )

    mome_layer, mome_name = _mome_layer_and_name(layer)
    mspec = mome_layer.get_kv_cache_spec(vllm_config)
    mbuilder = NPUMomeAttentionMetadataBuilder(
        mspec, [mome_name], vllm_config, device,
    )
    num_prompt = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    mome_meta = mbuilder.build(0, common, num_prompt_tokens=num_prompt)
    return attn_meta, mome_meta


def run_forward(layer, hidden, cos, sin, attn_meta, mome_meta):
    """Forward driver: patches the forward-context so the layer sees our meta."""
    prefix = layer.layer_name
    _, mome_name = _mome_layer_and_name(layer)
    ctx = SimpleNamespace(
        attn_metadata={f"{prefix}.attn": attn_meta, mome_name: mome_meta},
        no_compile_layers={prefix: layer},
        virtual_engine=0, batch_descriptor=None, capturing=False,
        num_tokens=hidden.shape[0],
    )
    from omni_npu.v1.layers.attention import (
        npu_dsa, npu_mla, npu_pangu, npu_pangu_custom_ops,
    )
    modules = (npu_pangu, npu_pangu_custom_ops, npu_dsa, npu_mla)
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(
                patch.object(module, "get_forward_context", return_value=ctx)
            )
        return layer.forward(hidden, cos, sin)


def cos_sin_for_positions(layer, positions: torch.Tensor):
    cos = layer.rotary_emb.cos_cached.index_select(0, positions)
    sin = layer.rotary_emb.sin_cached.index_select(0, positions)
    if getattr(layer, "_test_implementation", "low_latency") == "high_performance":
        cos = cos.view(-1, 1, 1, cos.shape[-1])
        sin = sin.view(-1, 1, 1, sin.shape[-1])
    return cos, sin


def cos_sin_for(layer, start: int, length: int):
    positions = torch.arange(start, start + length, dtype=torch.long, device=NPU)
    return cos_sin_for_positions(layer, positions)


def make_hidden(num_tokens: int, hidden_size: int, seed: int = 123):
    g = torch.Generator(device=NPU).manual_seed(seed)
    return torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16,
                       device=NPU, generator=g) * 0.05


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _sp_shard_forward(layer, hidden_full_tokens, cos, sin, attn_meta, mome_meta):
    """One TP forward over a sequence-parallel sharded token block.

    The layer expects 1/TP tokens per rank (all-gathered internally) and emits
    a reduce-scattered 1/TP output. Token counts not divisible by TP are padded
    up to a TP multiple before sharding (as the production model runner does)
    and sliced off after gathering; cos/sin stay full length since the layer
    indexes them by num_actual_tokens. Returns the gathered output [qlen, H].
    """
    from vllm.distributed import get_tp_group
    tp = get_tp_group()
    if (getattr(layer, "_test_implementation", "low_latency") ==
            "high_performance" and not getattr(layer, "ena_cp", False)):
        # The high-throughput path uses the conventional TP contract: every
        # rank receives all tokens, heads are sharded, and o_proj all-reduces a
        # full-token output. High-throughput DSA CP instead consumes and emits
        # the sequence-parallel token shards driven below.
        return run_forward(layer, hidden_full_tokens.clone(), cos, sin,
                           attn_meta, mome_meta)
    r, w = tp.rank_in_group, tp.world_size
    qlen = hidden_full_tokens.shape[0]
    padded = _round_up(qlen, w)
    h = hidden_full_tokens
    if padded != qlen:
        h = F.pad(h, (0, 0, 0, padded - qlen))
    per = padded // w
    shard = h[r * per:(r + 1) * per].clone()
    out_shard = run_forward(layer, shard, cos, sin, attn_meta, mome_meta)
    out_full = tp.all_gather(out_shard, dim=0)
    return out_full[:qlen]


def prefill_in_chunks_tp(layer, vllm_config, hidden_full, chunks, num_blocks):
    """Run a single sequence with FRESH caches in `chunks` = list of
    (context_len, query_len) sequence-parallel forwards. Returns concatenated
    [S, H]; chunks not divisible by the TP world size are padded internally.
    """
    alloc_caches(layer, vllm_config, num_blocks)
    outs = []
    for (ctx, qlen) in chunks:
        attn_meta, mome_meta = build_metadata(layer, vllm_config,
                                              [(ctx, qlen)], num_blocks)
        cos, sin = cos_sin_for(layer, ctx, qlen)
        chunk_h = hidden_full[ctx:ctx + qlen]
        outs.append(_sp_shard_forward(layer, chunk_h, cos, sin,
                                      attn_meta, mome_meta))
    return torch.cat(outs, dim=0)


def _hidden_size():
    return PANGU_HF_CONFIG["hidden_size"]


def _chunked(start, end, chunk=CHUNK_SIZE):
    """Split [start, end) into `chunk`-sized forwards as (context_len, query_len).

    e.g. _chunked(0, 513, 256) -> [(0, 256), (256, 256), (512, 1)].
    """
    out, ctx = [], start
    while ctx < end:
        q = min(chunk, end - ctx)
        out.append((ctx, q))
        ctx += q
    return out


# Physical block layout for the real-APC tests: request A caches its prefix in
# A_BLOCKS, and follow-up request B reuses those blocks (block-aligned hit)
# while appending its own physically non-contiguous suffix blocks
# (B_SUFFIX_BLOCKS), as prefix caching does in production.
A_BLOCKS = list(range(BLOCKS_PER_REQ))                       # [0 .. 15]
B_SUFFIX_BLOCKS = list(range(2 * BLOCKS_PER_REQ,             # [32 .. 47]
                             3 * BLOCKS_PER_REQ))


def _apc_block_table(context_len):
    """Request B's block_table: A's prefix blocks (block-aligned hit) followed
    by B's own non-contiguous suffix blocks."""
    nblk = context_len // BLOCK_SIZE
    return (A_BLOCKS[:nblk] + B_SUFFIX_BLOCKS)[:BLOCKS_PER_REQ]


def _apc_real_suffix_tp(layer, vllm_config, hidden, context_len, suffix_chunks,
                        num_blocks):
    """Real cross-request APC on the TP path.

    Request A full-prefills the block-aligned prefix [0, context_len) into
    A_BLOCKS (cached). A separate request B reuses A's prefix blocks and queries
    the suffix in `suffix_chunks` forwards, writing into its own non-contiguous
    suffix blocks. Returns B's concatenated suffix output [sum(query), H].
    """
    assert context_len % BLOCK_SIZE == 0, "APC hits are block-aligned"
    alloc_caches(layer, vllm_config, num_blocks)
    a_meta = build_metadata(layer, vllm_config, [(0, context_len)], num_blocks,
                            block_tables=[A_BLOCKS])
    cos, sin = cos_sin_for(layer, 0, context_len)
    _sp_shard_forward(layer, hidden[:context_len], cos, sin, *a_meta)
    b_table = _apc_block_table(context_len)
    outs = []
    for (ctx, qlen) in suffix_chunks:
        b_meta = build_metadata(layer, vllm_config, [(ctx, qlen)], num_blocks,
                                block_tables=[b_table])
        cos, sin = cos_sin_for(layer, ctx, qlen)
        outs.append(_sp_shard_forward(layer, hidden[ctx:ctx + qlen], cos, sin,
                                      *b_meta))
    return torch.cat(outs, dim=0)


def _worker_layer(device, layer_idx, kv_role=None,
                  enable_cp=False, enable_fc2=False,
                  implementation="low_latency"):
    torch.npu.set_device(device)
    cfg = bootstrap_worker(kv_role=kv_role, enable_cp=enable_cp,
                           enable_fc2=enable_fc2,
                           implementation=implementation)
    layer = build_layer(cfg, layer_idx, implementation=implementation)
    return cfg, layer


@dataclass(frozen=True)
class PrefillCase:
    """One self-describing prefill edge case.

    total:      S, total prompt tokens.
    chunk:      chunked-prefill granularity for the (non-cached) suffix; >= total
                means a single forward. Should be a block_size multiple to keep
                internal chunk boundaries block-aligned.
    hit_blocks: APC cached-prefix depth in blocks (cached prefix = hit_blocks *
                block_size). 0 disables APC (pure chunked prefill).

    The invariant collapses to chunked / apc / chunk+apc depending on the args:
      hit_blocks=0                  -> chunked == full
      hit_blocks>0, chunk>=suffix   -> APC hit (single suffix forward) == full
      hit_blocks>0, chunk<suffix    -> APC hit + chunked suffix == full
    """
    total: int
    chunk: int = CHUNK_SIZE
    hit_blocks: int = 0


def _run_case(layer, cfg, case: PrefillCase, num_blocks):
    """Run one PrefillCase and assert its invariant against a full prefill.

    Reference = full prefill [0, total). Tested = optionally cache a block-aligned
    prefix [0, ctx) (ctx = hit_blocks * block_size) as a real cross-request APC
    hit, then prefill the remaining [ctx, total) in `chunk`-sized forwards.
    Compares the tested output to the reference at the suffix positions.
    """
    total, chunk, ctx = case.total, case.chunk, case.hit_blocks * BLOCK_SIZE
    assert ctx < total, f"hit prefix {ctx} must be < total {total}"
    hidden = make_hidden(total, _hidden_size())
    out_full = prefill_in_chunks_tp(layer, cfg, hidden, [(0, total)], num_blocks)
    suffix_chunks = _chunked(ctx, total, chunk)
    if ctx == 0:
        tested = prefill_in_chunks_tp(layer, cfg, hidden, suffix_chunks,
                                      num_blocks)
    else:
        tested = _apc_real_suffix_tp(layer, cfg, hidden, ctx, suffix_chunks,
                                     num_blocks)
    name = f"S{total}_chunk{chunk}_hit{case.hit_blocks}"
    assert_close(tested, out_full[ctx:], name)


# A worker builds the layer once and loops over its cases, so a single 2-process
# spawn amortises the (heavy) HCCL init + layer build. All asserts run locally on
# every rank.
def mc_invariants_worker(device, rank, world_size, layer_idx, cases, num_blocks,
                         kv_role=None, enable_cp=False, enable_fc2=False,
                         implementation="low_latency"):
    """Run each PrefillCase in `cases` on the live (TP) layer, asserting its
    chunked / apc / chunk+apc invariant against a full prefill (see PrefillCase
    and _run_case). kv_role / enable_cp / enable_fc2 select the layer variant."""
    cfg, layer = _worker_layer(device, layer_idx, kv_role=kv_role,
                               enable_cp=enable_cp, enable_fc2=enable_fc2,
                               implementation=implementation)
    for case in cases:
        _run_case(layer, cfg, case, num_blocks)


def assert_close(actual, expected, name, atol=3e-2, rtol=3e-2):
    a = actual.float()
    e = expected.float()
    if a.shape != e.shape:
        raise AssertionError(f"{name}: shape {tuple(a.shape)} != {tuple(e.shape)}")
    if torch.allclose(a, e, atol=atol, rtol=rtol):
        return
    diff = (a - e).abs()
    idx = int(diff.view(-1).argmax().item())
    raise AssertionError(
        f"{name} mismatch: max_abs_diff={diff.max().item():.4e} "
        f"mean_abs_diff={diff.mean().item():.4e} "
        f"flat_idx={idx} shape={tuple(a.shape)}"
    )
