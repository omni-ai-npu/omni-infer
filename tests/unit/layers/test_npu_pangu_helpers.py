# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import omni_npu.v1.layers.attention.npu_pangu as pangu_mod
from omni_npu.attention.backends.dsa import NPUDSAMetadataBuilder
from omni_npu.v1.layers.attention.npu_pangu import (
    NPUPanguSparseAttention,
    _get_slot_mapping_2d,
    npu_pangu_forward_fake,
)
from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod
from omni_npu.v1.models.pangu import pangu_v2_moe_mtp as mtp_mod


_FA_CALLERS = (
    "decode",
    "decode_mla",
    "prefill_absorb",
    "prefill_absorb_mla",
    "prefill",
    "prefill_mla",
)


def _build_sparse_attention(
    *,
    layer_idx,
    num_hidden_layers=4,
    swa_layers=None,
    sliding_window_list=None,
    index_topk=None,
    index_head_dim=8,
    indexer_types=None,
    rope_interleave=None,
    rope_interleaved=None,
    enable_attn_sp=False,
    tp_size=1,
):
    if swa_layers is None:
        swa_layers = [0, 1]
    if sliding_window_list is None:
        sliding_window_list = [512, 512]
    config_kwargs = {
        "num_hidden_layers": num_hidden_layers,
        "index_head_dim": index_head_dim,
        "use_mome": False,
    }
    if index_topk is not None:
        config_kwargs["index_topk"] = index_topk
    if indexer_types is not None:
        config_kwargs["indexer_types"] = indexer_types
    if rope_interleave is not None:
        config_kwargs["rope_interleave"] = rope_interleave
    if rope_interleaved is not None:
        config_kwargs["rope_interleaved"] = rope_interleaved
    config = SimpleNamespace(**config_kwargs)
    cache_config = SimpleNamespace(block_size=16, cache_dtype="auto")
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    compilation_config = SimpleNamespace(static_forward_context={})
    original_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    patches = [
        patch.object(NPUPanguSparseAttention, "_init_MLA_weights"),
        patch.object(NPUPanguSparseAttention, "_init_rotary_emb"),
        patch.object(NPUPanguSparseAttention, "_init_param_sinks"),
        patch.object(NPUPanguSparseAttention, "_align_pagesize"),
        patch.object(NPUPanguSparseAttention, "_init_attention_layers"),
        patch.object(NPUPanguSparseAttention, "_init_mome_layer"),
        patch.object(NPUPanguSparseAttention, "_init_cross_layer_shared_ops"),
        patch.object(
            pangu_mod,
            "get_tp_group",
            return_value=SimpleNamespace(world_size=tp_size),
        ),
        patch.object(pangu_mod, "on_ascend950", return_value=False),
        patch.object(
            pangu_mod,
            "get_current_vllm_config",
            return_value=SimpleNamespace(compilation_config=compilation_config),
        ),
        patch.object(
            pangu_mod.model_extra_config.operator_opt_config,
            "use_noncontiguous_kv",
            True,
        ),
        patch.object(
            pangu_mod.model_extra_config.parall_config,
            "ena_swa_attn_seq_parallel",
            enable_attn_sp,
        ),
        patch.object(torch, "zeros", side_effect=cpu_zeros),
    ]
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        return NPUPanguSparseAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=16,
            num_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=8,
            kv_lora_rank=8,
            rope_theta=10000,
            swa_layers=swa_layers,
            param_sink_number=1,
            sliding_window_list=sliding_window_list,
            cache_config=cache_config,
            prefix=f"model.layers.{layer_idx}.self_attn",
        )


class TestGetSlotMapping2d(unittest.TestCase):
    def test_fast_path_returns_existing_slot_mapping_2d(self):
        cached = torch.tensor([[0, 0], [1, 1]])

        def _should_not_be_called(*_args, **_kwargs):
            raise AssertionError("get_slot_mapping_2d must not be called when attribute is set")

        meta = SimpleNamespace(slot_mapping_2d=cached, get_slot_mapping_2d=_should_not_be_called)
        self.assertIs(_get_slot_mapping_2d(meta), cached)

    def test_default_layer_idx_invokes_zero_arg_callback(self):
        """Mirrors MLA's lambda which accepts no arguments."""
        sentinel = torch.tensor([[1, 2]])

        def _zero_arg_slot_mapping():
            return sentinel

        meta = SimpleNamespace(
            slot_mapping_2d=None, get_slot_mapping_2d=_zero_arg_slot_mapping
        )
        self.assertIs(_get_slot_mapping_2d(meta), sentinel)

    def test_explicit_layer_idx_is_passed_through(self):
        """DSA's closure expects layer_idx; ensure it gets forwarded."""
        seen = {}

        def cb(layer_idx):
            seen["layer_idx"] = layer_idx
            return torch.tensor([[3, 4]])

        meta = SimpleNamespace(slot_mapping_2d=None, get_slot_mapping_2d=cb)
        out = _get_slot_mapping_2d(meta, layer_idx=7)
        self.assertEqual(seen["layer_idx"], 7)
        self.assertTrue(torch.equal(out, torch.tensor([[3, 4]])))

    def test_returns_none_when_metadata_has_neither_attribute(self):
        meta = SimpleNamespace()
        self.assertIsNone(_get_slot_mapping_2d(meta))


class TestDSALazySlotMapping2d(unittest.TestCase):
    """Direct coverage for NPUDSAMetadataBuilder._lazy_slot_mapping_2d."""

    @staticmethod
    def _minimal_builder(block_size=16):
        b = NPUDSAMetadataBuilder.__new__(NPUDSAMetadataBuilder)
        b.kv_cache_spec = SimpleNamespace(block_size=block_size)
        return b

    @staticmethod
    def _metadata(slots, first_layer_idx=-1):
        class Metadata(SimpleNamespace):
            pass

        return Metadata(
            slot_mapping=slots,
            first_layer_idx=first_layer_idx,
            slot_mapping_cache=None,
        )

    def test_default_layer_idx_recomputes_into_cache(self):
        # MLA's zero-arg lambda path: layer_idx defaults to -1, closure must
        # recompute and write through the cache; returned tensor is the cache.
        b = self._minimal_builder(block_size=16)
        meta = self._metadata(torch.tensor([0, 5, 16, 17], dtype=torch.long))
        inner = b._lazy_slot_mapping_2d(meta)
        out = inner()
        expect = torch.stack([meta.slot_mapping // 16, meta.slot_mapping % 16], dim=-1)
        self.assertTrue(torch.equal(out, expect))
        self.assertIs(out, meta.slot_mapping_cache)

    def test_first_layer_idx_populates_cache(self):
        # DSA first-layer path: writes cache and returns it.
        b = self._minimal_builder(block_size=16)
        meta = self._metadata(torch.tensor([0, 5, 16, 17], dtype=torch.long), first_layer_idx=0)
        inner = b._lazy_slot_mapping_2d(meta)
        out = inner(0)
        expect = torch.stack([meta.slot_mapping // 16, meta.slot_mapping % 16], dim=-1)
        self.assertTrue(torch.equal(out, expect))
        self.assertIs(out, meta.slot_mapping_cache)

    def test_non_first_layer_idx_returns_existing_cache(self):
        # Non-first DSA layer reads back the cache populated by the first layer.
        b = self._minimal_builder(block_size=16)
        meta = self._metadata(torch.tensor([0, 5, 16, 17], dtype=torch.long), first_layer_idx=0)
        inner = b._lazy_slot_mapping_2d(meta)
        inner(0)  # populate cache
        seeded = meta.slot_mapping_cache
        out = inner(3)
        self.assertIs(out, seeded)


class TestPanguIndexShare(unittest.TestCase):
    def test_shared_indexer_skips_topk(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        config = SimpleNamespace(indexer_types=["none", "full", "shared"])

        attention.layer_idx = 1
        self.assertFalse(attention._skip_topk(config))
        attention.layer_idx = 2
        self.assertTrue(attention._skip_topk(config))

    def test_full_and_shared_layers_use_same_metadata_topk(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.is_dsa_layer = True
        attention.skip_topk = False
        attention.use_mome = False
        attention.prefix = "model.layers.0.self_attn"
        attention.qk_head_dim = 4
        attention.qk_nope_head_dim = 2
        attention.qk_rope_head_dim = 2
        attention.num_local_heads = 1
        attention.first_chunk_pa = False
        attention.q_a_proj = MagicMock(side_effect=lambda value: value)
        attention.q_a_layernorm = MagicMock(side_effect=lambda value: value)
        attention.q_b_proj = MagicMock(
            side_effect=lambda value: torch.zeros(value.shape[0], 4)
        )
        attention._w_uk_t_absorb = MagicMock(side_effect=lambda value: value)
        attention._q_rope = MagicMock(side_effect=lambda value, *_args: value)
        attention._kv_down_mome = MagicMock(side_effect=lambda value, *_args: value)
        topk_indices = torch.arange(12).view(4, 1, 3)
        attention.indexer = MagicMock(return_value=topk_indices)
        metadata = SimpleNamespace(
            prefill=SimpleNamespace(topk_indices_buffer=None),
            decode=None,
        )
        hidden_states = torch.randn(4, 4)
        cos = torch.randn(4, 2)
        kv_cache = (torch.empty(0), torch.empty(0))

        with patch(
            "torch.ops.vllm.npu_pangu_kv_cache_update",
            return_value=kv_cache,
        ):
            full_output = attention._mla_prolog_sequential(
                hidden_states, cos, cos, kv_cache, metadata, None
            )

        self.assertIs(metadata.prefill.topk_indices_buffer, topk_indices)
        self.assertIs(full_output[-1], topk_indices)

        attention.skip_topk = True
        attention.indexer.reset_mock()
        with patch(
            "torch.ops.vllm.npu_pangu_kv_cache_update",
            return_value=kv_cache,
        ):
            shared_output = attention._mla_prolog_sequential(
                hidden_states, cos, cos, kv_cache, metadata, None
            )

        attention.indexer.assert_not_called()
        self.assertIs(shared_output[-1], topk_indices)


class TestPanguFAMetadataIsolation(unittest.TestCase):
    def test_hybrid_swa_and_full_mla_use_separate_producers(self):
        cases = (
            (0, True, "", False, False),
            (1, False, "", False, False),
            (2, True, "_mla", False, False),
            (3, False, "_mla", False, False),
            (4, True, "", False, False),
        )
        for layer_idx, producer, suffix, is_dsa, skip_topk in cases:
            with self.subTest(layer_idx=layer_idx):
                attention = _build_sparse_attention(layer_idx=layer_idx)
                self.assertEqual(attention.is_fa_metadata_producer, producer)
                self.assertEqual(attention._fa_meta_suffix, suffix)
                self.assertEqual(attention.is_dsa_layer, is_dsa)
                self.assertEqual(attention.skip_topk, skip_topk)

    def test_dsa_skips_full_mla_producer_and_uses_mla_suffix(self):
        attention = _build_sparse_attention(
            layer_idx=2,
            swa_layers=[0, 1],
            sliding_window_list=[512, 512],
            index_topk=4,
            indexer_types=["unique", "unique", "unique", "unique"],
        )
        self.assertTrue(attention.is_dsa_layer)
        self.assertFalse(attention.is_fa_metadata_producer)
        self.assertEqual(attention._fa_meta_suffix, "_mla")

    def test_index_topk_none_or_zero_falls_back_to_full_mla(self):
        for index_topk in (None, 0):
            with self.subTest(index_topk=index_topk):
                kwargs = {"layer_idx": 2}
                if index_topk is not None:
                    kwargs["index_topk"] = index_topk
                attention = _build_sparse_attention(**kwargs)
                self.assertFalse(attention.is_dsa_layer)
                self.assertTrue(attention.is_fa_metadata_producer)
                self.assertEqual(attention._fa_meta_suffix, "_mla")

    def test_rope_interleave_prefers_explicit_flag(self):
        interleaved = _build_sparse_attention(
            layer_idx=0, rope_interleave=True, rope_interleaved=False
        )
        self.assertTrue(interleaved.rope_interleave)
        fallback = _build_sparse_attention(
            layer_idx=0, rope_interleaved=True
        )
        self.assertTrue(fallback.rope_interleave)
        disabled = _build_sparse_attention(layer_idx=0)
        self.assertFalse(disabled.rope_interleave)

    def test_shared_ops_register_swa_and_mla_callers(self):
        created = []

        class FakeSharedOp:
            def __init__(self, **kwargs):
                created.append(kwargs)

        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.on_ascend950 = True
        fake_ops = SimpleNamespace(
            _npu_fused_infer_attention_sink_metadata=object(),
            npu_ai_infra_attention_pioneer_metadata=object(),
        )
        with patch.object(pangu_mod, "CrossLayerSharedOp", FakeSharedOp), \
                patch.object(
                    pangu_mod, "npu_fused_infer_attention_sink_metadata", None
                ), \
                patch.object(
                    pangu_mod, "npu_ai_infra_attention_pioneer_metadata", None
                ), \
                patch.object(pangu_mod.torch.ops, "custom", fake_ops, create=True):
            attention._init_cross_layer_shared_ops()

        self.assertEqual(len(created), 2)
        self.assertEqual(created[0]["callers"], _FA_CALLERS)
        self.assertEqual(created[1]["callers"], _FA_CALLERS)
        self.assertEqual(created[1]["shape"], (1024,))

    def test_kv_rmsnorm_rope_cache_kwargs_honor_rotary_mode(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.kv_lora_rank = 8
        attention.qk_rope_head_dim = 4
        attention.kv_a_layernorm = SimpleNamespace(
            weight=torch.ones(8), variance_epsilon=1e-6
        )
        kv = torch.zeros(2, 12)
        cos = torch.zeros(2, 4)
        sin = torch.zeros(2, 4)
        metadata = SimpleNamespace(slot_mapping=torch.arange(2))

        attention.rope_interleave = False
        half = attention._kv_rmsnorm_rope_cache_v2_kwargs(
            kv, cos, sin, metadata, k_cache=None, ckv_cache=torch.zeros(2, 1, 8)
        )
        self.assertEqual(half["rotary_mode"], "half")
        self.assertIsNone(half["k_cache"])

        attention.rope_interleave = True
        interleave = attention._kv_rmsnorm_rope_cache_v2_kwargs(
            kv, cos, sin, metadata
        )
        self.assertEqual(interleave["rotary_mode"], "interleave-half")

    def test_page_size_without_dsa_uses_mla_page(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.kv_lora_rank = 8
        attention.qk_rope_head_dim = 4
        attention.use_mome = False
        cache_config = SimpleNamespace(block_size=16)
        page = attention._calculate_page_size_padded(
            cache_config, "auto", SimpleNamespace()
        )
        self.assertEqual(page, 16 * (8 + 4) * 2)


class TestOpenPanguV2DecoderAndMoE(unittest.TestCase):
    def test_decoder_init_sets_default_rope_theta(self):
        captured = {}

        def fake_attn(*_args, **kwargs):
            captured["rope_theta"] = kwargs["rope_theta"]
            return SimpleNamespace(o_proj=SimpleNamespace(prefix="o_proj"))

        config = SimpleNamespace(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            kv_lora_rank=8,
            param_sink_number=1,
            first_k_dense_replace=99,
            rope_parameters={"rope_theta": 10000},
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            intermediate_size=32,
            hidden_act="silu",
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            cache_config=SimpleNamespace(),
            quant_config=None,
            parallel_config=SimpleNamespace(),
        )
        with patch.object(model_mod, "NPUPanguSparseAttention", fake_attn), \
                patch.object(model_mod, "OpenPanguV2MLP", MagicMock()), \
                patch.object(model_mod, "RMSNorm", MagicMock()), \
                patch.object(model_mod, "_normalize_rope_parameters"):
            layer = model_mod.OpenPanguV2DecoderLayer(
                config, "model.layers.0", vllm_config
            )

        self.assertEqual(captured["rope_theta"], 10000)
        self.assertNotIsInstance(layer.mlp, model_mod.OpenPanguV2MOE)

    def test_set_side_stream_forwards_to_moe(self):
        layer = model_mod.OpenPanguV2DecoderLayer.__new__(
            model_mod.OpenPanguV2DecoderLayer
        )
        torch.nn.Module.__init__(layer)
        moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
        torch.nn.Module.__init__(moe)
        layer.mlp = moe
        layer.self_attn = SimpleNamespace()
        side = object()
        fetch = object()
        layer.set_side_stream(side, fetch)
        self.assertIs(moe.side_stream, side)
        self.assertIs(moe.fetch_stream, fetch)
        self.assertIs(layer.self_attn.side_stream, side)

    def test_dispatch_combine_splits_over_configured_max_batch(self):
        moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
        moe.ep_comm_name = "ep"
        moe.side_stream = None
        moe.gate = MagicMock(return_value=(torch.zeros(4, 2), None))
        moe.experts = SimpleNamespace(
            top_k=1, topk_group=1, num_expert_group=1
        )
        moe.e_score_correction_bias = None
        moe.routed_scaling_factor = 1.0
        moe.use_moe_force_load_balance = False
        moe.enable_eplb = False
        moe._is_quant = False
        moe.moe_dispatch_combine_max_batch_size = 2
        moe.shared_experts = MagicMock(return_value=torch.ones(4, 3))
        chunks = []

        def fake_single(hidden, *_args, **_kwargs):
            chunks.append(hidden.shape[0])
            return torch.zeros(hidden.shape[0], 3)

        moe._dispatch_combine_single_batch = fake_single
        moe._get_mc2_mask = MagicMock(return_value=None)
        hidden = torch.zeros(4, 3)
        with patch.object(
            model_mod.torch_npu,
            "npu_moe_gating_top_k",
            return_value=(
                torch.ones(4, 1),
                torch.zeros(4, 1, dtype=torch.int32),
                None,
            ),
        ):
            out = moe._forward_dispatch_combine(hidden)

        self.assertEqual(chunks, [2, 2])
        self.assertEqual(tuple(out.shape), (4, 3))


def _bare_swa_attention(**attrs):
    """Build an uninitialized NPUPanguSparseAttention with SWA SP defaults."""
    attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
    defaults = dict(
        on_ascend950=False,
        use_mome=False,
        enable_mome_sp=False,
        use_mome_inplace_update=False,
        sharded_o_proj=False,
        is_cp_layer=False,
        is_attn_sp_layer=True,
        is_dsa_layer=False,
        enable_flashcomm2=False,
        tp_size=1,
        moe_comm_strategy="agrs",
        prefix="model.layers.0.self_attn",
        num_heads=2,
        num_local_heads=2,
        v_head_dim=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        kv_lora_rank=8,
        sliding_window=512,
        block_size=16,
        scaling=1.0,
        use_aicpu_fa_tiling=False,
        sink_k_nope=None,
        sink_k_pe=None,
        W_UV=torch.zeros(2, 8, 4),
        attn=SimpleNamespace(
            kv_cache=[(torch.zeros(1), torch.zeros(1))],
            impl=SimpleNamespace(SHARE_MASK_TRIL_SPARSE=None),
        ),
    )
    defaults.update(attrs)
    for name, value in defaults.items():
        setattr(attention, name, value)
    return attention


def _fwctx_patch():
    """Stub vLLM forward context for SWA SP unit tests."""
    return patch.object(
        pangu_mod,
        "get_forward_context",
        return_value=SimpleNamespace(virtual_engine=0),
    )


def _prefill_sp_forward_ctx(attention, tokens=4):
    """Forward context used by npu_pangu_forward SWA SP dispatch tests."""
    meta = SimpleNamespace(
        num_actual_tokens=tokens,
        num_decode_tokens=0,
        num_decodes=0,
        num_prefills=1,
        prefill=SimpleNamespace(),
        decode=None,
    )
    return SimpleNamespace(
        no_compile_layers={"layer": attention}, attn_metadata=meta
    )


def _stub_prefill_sp_projections(attention):
    """CPU stubs for Q/KV projections used by _forward_prefill_sp tests."""
    attention.q_a_proj = MagicMock(
        side_effect=lambda value: torch.zeros(value.size(0), 8)
    )
    attention.q_a_layernorm = MagicMock(side_effect=lambda value: value)
    attention.q_b_proj = MagicMock(
        side_effect=lambda value: torch.zeros(value.size(0), 16)
    )
    attention._q_rope = MagicMock(side_effect=lambda value, *_a: value)
    attention._w_uk_t_absorb = MagicMock(
        side_effect=lambda value: torch.zeros(value.size(0), 2, 8)
    )
    attention.kv_a_proj_with_mqa = MagicMock(
        side_effect=lambda value: torch.zeros(value.size(0), 12)
    )


def _stub_prefill_sp_kernels(attention, absorb_tokens):
    """Stub kv-cache, absorb, and o_proj used by _forward_prefill_sp tests."""
    attention._npu_kvrmsnorm_rope_cache = MagicMock(
        return_value=((torch.zeros(4, 1, 8), torch.zeros(4, 1, 4)),)
    )
    attention._apply_SWA_attention_prefill_absorb = MagicMock(
        return_value=torch.ones(absorb_tokens, 8)
    )
    attention._apply_o_proj = MagicMock(side_effect=lambda value: value)


def _prefill_sp_manager(local, valid):
    """SP manager that keeps the first `local` tokens."""
    return SimpleNamespace(
        sp_len=local,
        valid_token_count=valid,
        slice_tokens=lambda tensor, cached=None: tensor[:local],
        ag_tokens=lambda tensor: tensor,
    )


def _call_npu_pangu_forward_sp(attention, hidden):
    """Invoke npu_pangu_forward under a stubbed prefill-SP context."""
    ctx = _prefill_sp_forward_ctx(attention)
    with patch.object(pangu_mod, "get_forward_context", return_value=ctx):
        return pangu_mod.npu_pangu_forward(
            hidden, torch.zeros(4, 2), torch.zeros(4, 2), "layer"
        )


class TestPanguSWASeqParallel(unittest.TestCase):
    def test_constructor_replicates_heads_on_swa_sp_layer(self):
        """SWA SP layers keep full head counts instead of TP-sharding Q/KV."""
        sharded = _build_sparse_attention(layer_idx=0, tp_size=2)
        self.assertFalse(sharded.is_attn_sp_layer)
        self.assertEqual(sharded.num_local_heads, 1)
        sp_attn = _build_sparse_attention(layer_idx=0, enable_attn_sp=True, tp_size=2)
        self.assertTrue(sp_attn.is_attn_sp_layer)
        self.assertEqual(sp_attn.num_local_heads, sp_attn.num_heads)
        self.assertTrue(sp_attn.disable_o_conv_tp)

    def test_npu_pangu_forward_dispatches_prefill_sp(self):
        """Pure prefill on an SP layer must call _forward_prefill_sp."""
        attention = _bare_swa_attention()
        hidden = torch.zeros(4, 8)
        attention._forward_prefill_sp = MagicMock(return_value=hidden)
        out_h = _call_npu_pangu_forward_sp(attention, hidden)
        attention._forward_prefill_sp.assert_called_once()
        self.assertIs(out_h, hidden)

    def test_npu_pangu_forward_sp_rejects_allreduce_moe(self):
        """SWA SP cannot run with the allreduce MoE communication strategy."""
        attention = _bare_swa_attention(moe_comm_strategy="allreduce")
        hidden = torch.zeros(4, 8)
        with self.assertRaises(AssertionError):
            _call_npu_pangu_forward_sp(attention, hidden)

    def test_absorb_returns_empty_on_zero_sp_shard(self):
        """Empty SP shards skip the FA kernel but still return a 2D tensor."""
        attention = _bare_swa_attention()
        q_nope = torch.zeros(3, 2, 8)
        q_pe = torch.zeros(3, 2, 4)
        kv = (torch.zeros(4, 1, 8), torch.zeros(4, 1, 4))
        sp_manager = SimpleNamespace(
            valid_token_count=0,
            sp_attn_meta=lambda: (None, None, None),
        )
        meta = SimpleNamespace(prefill=SimpleNamespace())
        out = attention._apply_SWA_attention_prefill_absorb(
            q_nope, q_pe, kv, attn_metadata=meta, sp_manager=sp_manager
        )
        self.assertEqual(tuple(out.shape), (0, 8))

    def test_forward_prefill_sp_pads_short_shards(self):
        """_forward_prefill_sp pads FA output back to the local SP length."""
        attention = _bare_swa_attention()
        local = 4
        valid = 2
        hidden = torch.ones(local, 8)
        _stub_prefill_sp_projections(attention)
        _stub_prefill_sp_kernels(attention, absorb_tokens=valid)
        meta = SimpleNamespace(
            num_actual_tokens=8,
            prefill=SimpleNamespace(sp_manager=_prefill_sp_manager(local, valid)),
        )
        with _fwctx_patch():
            out = attention._forward_prefill_sp(
                hidden, torch.zeros(8, 4), torch.zeros(8, 4), meta, None
            )
        self.assertEqual(tuple(out.shape), (local, 8))
        self.assertTrue(torch.equal(out[valid:], torch.zeros(local - valid, 8)))

    def test_forward_prefill_sp_rejects_ascend950(self):
        """SWA SP is A3-only."""
        attention = _bare_swa_attention(on_ascend950=True)
        meta = SimpleNamespace(
            num_actual_tokens=4, prefill=SimpleNamespace(sp_manager=object())
        )
        with self.assertRaises(AssertionError):
            attention._forward_prefill_sp(
                torch.zeros(4, 8), torch.zeros(4, 2), torch.zeros(4, 2), meta
            )

    def test_forward_prefill_sp_mome_inplace_and_prefetch(self):
        """MoME SP + sharded o_proj prefetch still returns local-token output."""
        attention = _bare_swa_attention(
            use_mome=True, enable_mome_sp=True,
            use_mome_inplace_update=True, sharded_o_proj=True,
        )
        local = 3
        hidden = torch.ones(local, 8)
        _stub_prefill_sp_projections(attention)
        _stub_prefill_sp_kernels(attention, absorb_tokens=local)
        attention._apply_MOME = MagicMock()
        attention.o_proj = SimpleNamespace(prefetch=MagicMock())
        attention.qa_conv = object()
        attention.compresskv_conv = object()
        attention.o_conv = object()
        meta = SimpleNamespace(
            num_actual_tokens=6,
            prefill=SimpleNamespace(sp_manager=_prefill_sp_manager(local, local)),
        )
        prefetch = SimpleNamespace(wait_stream=MagicMock())
        with _fwctx_patch(), patch.object(
            pangu_mod.torch.npu, "current_stream", return_value=object()
        ), patch.object(
            pangu_mod, "named_stream", return_value=prefetch
        ), patch.object(
            pangu_mod.torch.npu, "stream", return_value=nullcontext()
        ):
            out = attention._forward_prefill_sp(
                hidden, torch.zeros(6, 4), torch.zeros(6, 4), meta, object()
            )
        self.assertEqual(tuple(out.shape), (local, 8))
        attention.o_proj.prefetch.assert_called_once()
        attention._apply_MOME.assert_called()

    def test_absorb_uses_sp_metadata_and_runs_sink_kernel(self):
        """Non-empty SP shards pass sp_attn_meta lengths into the sink kernel."""
        attention = _bare_swa_attention()
        q_nope = torch.zeros(2, 2, 8)
        q_pe = torch.zeros(2, 2, 4)
        kv = (torch.zeros(4, 1, 8), torch.zeros(4, 1, 4))
        query_cumlens = torch.tensor([2], dtype=torch.int32)
        seq_lens = torch.tensor([4], dtype=torch.int32)
        block_table = torch.zeros(1, 2, dtype=torch.int32)
        sp_manager = SimpleNamespace(
            valid_token_count=2,
            sp_attn_meta=lambda: (query_cumlens, seq_lens, block_table),
        )
        meta = SimpleNamespace(prefill=SimpleNamespace())
        sink_out = torch.zeros(2, 2, 8)
        bmm_out = torch.zeros(2, 2, 4)
        with patch(
            "torch.ops.custom.npu_fused_infer_attention_sink",
            return_value=(sink_out,),
            create=True,
        ), patch.object(
            pangu_mod.torch_npu, "npu_transpose_batchmatmul", return_value=bmm_out
        ):
            out = attention._apply_SWA_attention_prefill_absorb(
                q_nope, q_pe, kv, attn_metadata=meta, sp_manager=sp_manager
            )
        self.assertEqual(tuple(out.shape), (2, 8))


if __name__ == "__main__":
    unittest.main()
