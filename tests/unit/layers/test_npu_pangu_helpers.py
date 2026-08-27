# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
from contextlib import ExitStack
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
            return_value=SimpleNamespace(world_size=1),
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


class TestPanguIndexShare(unittest.TestCase):
    def test_forward_fake_preserves_hidden_and_topk_shapes(self):
        hidden_states = torch.zeros(3, 16)
        topk_buffer = torch.zeros(3, 1, 8, dtype=torch.int32)

        output, output_topk = npu_pangu_forward_fake(
            hidden_states,
            torch.zeros(3, 1, 64),
            torch.zeros(3, 1, 64),
            "model.layers.0.self_attn",
            topk_buffer,
        )

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(output.dtype, hidden_states.dtype)
        self.assertEqual(output_topk.shape, topk_buffer.shape)
        self.assertEqual(output_topk.dtype, topk_buffer.dtype)

    def test_constructor_marks_shared_indexer_layer(self):
        attention = _build_sparse_attention(
            layer_idx=1,
            num_hidden_layers=2,
            swa_layers=[0],
            sliding_window_list=[128],
            index_topk=4,
            indexer_types=["unique", "shared"],
            rope_interleaved=True,
        )

        self.assertTrue(attention.is_dsa_layer)
        self.assertTrue(attention.skip_topk)

    def test_prepare_phase_inputs_slices_shared_topk_buffer(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)

        def _no_slot_mapping():
            return None

        metadata = SimpleNamespace(
            num_decode_tokens=2,
            num_actual_tokens=5,
            slot_mapping=torch.arange(5),
            slot_mapping_2d=None,
            get_slot_mapping_2d=_no_slot_mapping,
            prefill=SimpleNamespace(),
            decode=SimpleNamespace(),
        )
        hidden_states = torch.arange(15).view(5, 3)
        cos = torch.arange(10).view(5, 2)
        sin = cos + 100
        topk_buffer = torch.arange(20, dtype=torch.int32).view(5, 1, 4)

        prefill_inputs = attention._prepare_phase_inputs(
            hidden_states, cos, sin, metadata, "prefill", topk_buffer
        )
        self.assertTrue(torch.equal(prefill_inputs[0], hidden_states[2:]))
        self.assertTrue(torch.equal(prefill_inputs[3], topk_buffer[2:]))

        decode_inputs = attention._prepare_phase_inputs(
            hidden_states, cos, sin, metadata, "decode", topk_buffer
        )
        self.assertTrue(torch.equal(decode_inputs[0], hidden_states[:2]))
        self.assertTrue(torch.equal(decode_inputs[3], topk_buffer[:2]))

    def test_sequential_prolog_shared_layer_skips_indexer(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.first_chunk_pa = False
        attention.use_mome = False
        attention.is_dsa_layer = True
        attention.skip_topk = True
        attention.num_local_heads = 1
        attention.qk_head_dim = 4
        attention.qk_nope_head_dim = 2
        attention.qk_rope_head_dim = 2
        attention.prefix = "model.layers.1.self_attn"
        attention.q_a_proj = MagicMock(side_effect=lambda value: value)
        attention.q_a_layernorm = MagicMock(side_effect=lambda value: value)
        attention.q_b_proj = MagicMock(
            side_effect=lambda value: torch.cat([value, value], dim=-1)
        )
        attention._w_uk_t_absorb = MagicMock(side_effect=lambda value: value)
        attention._q_rope = MagicMock(side_effect=lambda value, *_: value)
        attention._kv_down_mome = MagicMock(
            return_value=torch.zeros(2, 4, dtype=torch.float32)
        )
        attention.indexer = MagicMock(
            side_effect=AssertionError("shared layer must not execute indexer")
        )

        hidden_states = torch.ones(2, 2)
        cos = torch.ones(2, 2)
        sin = torch.ones(2, 2)
        kv_cache = (torch.zeros(1), torch.zeros(1))
        metadata = SimpleNamespace(
            decode=SimpleNamespace(), prefill=SimpleNamespace(chunked_context=None)
        )
        topk_buffer = torch.arange(8, dtype=torch.int32).view(2, 1, 4)
        updated_cache = (torch.ones(1), torch.ones(1))

        with patch(
            "torch.ops.vllm.npu_pangu_kv_cache_update",
            return_value=updated_cache,
            create=True,
        ):
            result = attention._mla_prolog_sequential(
                hidden_states,
                cos,
                sin,
                kv_cache,
                metadata,
                None,
                topk_buffer,
            )

        attention.indexer.assert_not_called()
        self.assertIs(result[2], updated_cache)
        self.assertIs(result[3], topk_buffer)


def _pp_group():
    return SimpleNamespace(is_first_rank=True, is_last_rank=True)


class TestPanguModelTopkBuffer(unittest.TestCase):
    def test_resolve_decode_mc2_mask_supports_metadata_dict(self):
        mask = torch.tensor([True, False, True, False])
        metadata = SimpleNamespace(
            decode=SimpleNamespace(mc2_mask=mask),
            num_decode_tokens=3,
            num_actual_tokens=3,
        )
        context = SimpleNamespace(
            attn_metadata={"mla": SimpleNamespace(), "mome": metadata}
        )

        with patch.object(model_mod, "get_forward_context", return_value=context):
            resolved = model_mod._resolve_decode_mc2_mask(3)

        self.assertTrue(torch.equal(resolved, mask[:3]))

    def test_get_mc2_mask_returns_empty_tensor_for_mixed_batch(self):
        metadata = SimpleNamespace(
            decode=SimpleNamespace(mc2_mask=torch.ones(4, dtype=torch.bool)),
            num_decode_tokens=2,
            num_actual_tokens=3,
        )
        context = SimpleNamespace(attn_metadata=metadata)

        with patch.object(model_mod, "get_forward_context", return_value=context):
            result = model_mod.npu_get_mc2_mask(torch.zeros(3, 8))

        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.numel(), 0)

    def test_main_model_threads_topk_buffer_between_layers(self):
        class FakeLayer:
            def __init__(self, replacement_buffer=None):
                self.replacement_buffer = replacement_buffer
                self.seen_buffers = []

            def mhc_head(self, hidden_states):
                return hidden_states, None, None, None, None

            def __call__(self, hidden_states, residual, h_post, h_res,
                         cos, sin, sk_event, topk_indices_buffer):
                self.seen_buffers.append(topk_indices_buffer)
                output_buffer = (
                    self.replacement_buffer
                    if self.replacement_buffer is not None
                    else topk_indices_buffer
                )
                return model_mod.OpenPanguV2DecoderLayerOutput(
                    hidden_states, residual, h_post, h_res, sk_event,
                    output_buffer,
                )

        model = model_mod.OpenPanguV2Model.__new__(model_mod.OpenPanguV2Model)
        torch.nn.Module.__init__(model)
        replacement = torch.full((3, 1, 4), 7, dtype=torch.int32)
        first_layer = FakeLayer(replacement)
        second_layer = FakeLayer()
        model.layers = [first_layer, second_layer]
        model.start_layer = 0
        model.end_layer = 2
        model.need_tp_padding = False
        model.use_mhc = False
        model.config = SimpleNamespace(index_topk=4)

        def embed_tokens(input_ids, **_kwargs):
            return input_ids.float().unsqueeze(-1)

        model.embed_tokens = embed_tokens
        model.cos_cached = torch.zeros(8, 2)
        model.sin_cached = torch.zeros(8, 2)

        with patch.object(model_mod, "get_pp_group", return_value=_pp_group()):
            model.forward(
                input_ids=torch.tensor([1, 2, 3]),
                positions=torch.tensor([0, 1, 2]),
                intermediate_tensors=None,
            )

        initial_buffer = first_layer.seen_buffers[0]
        self.assertEqual(initial_buffer.shape, (3, 1, 4))
        self.assertEqual(initial_buffer.dtype, torch.int32)
        self.assertIs(second_layer.seen_buffers[0], replacement)

    def test_mtp_layer_allocates_and_passes_topk_buffer(self):
        class FakeMTPBlock:
            def __init__(self):
                def _get_cos_sin(positions):
                    zeros = torch.zeros(positions.shape[0], 2)
                    return zeros, zeros

                self.self_attn = SimpleNamespace(
                    rotary_emb=SimpleNamespace(get_cos_sin=_get_cos_sin)
                )
                self.seen_buffer = None

            def mhc_head(self, hidden_states):
                return hidden_states, None, None, None, None

            def __call__(self, hidden_states, residual, *args):
                self.seen_buffer = args[-1]
                return model_mod.OpenPanguV2DecoderLayerOutput(
                    hidden_states, residual, None, None, None,
                    self.seen_buffer,
                )

        layer = mtp_mod.OpenPanguV2MultiTokenPredictorLayer.__new__(
            mtp_mod.OpenPanguV2MultiTokenPredictorLayer
        )
        torch.nn.Module.__init__(layer)
        layer.enorm = torch.nn.Identity()
        layer.hnorm = torch.nn.Identity()

        def eh_proj(value):
            return value[:, :2], None

        layer.eh_proj = eh_proj
        layer.need_tp_padding = False
        layer.config = SimpleNamespace(index_topk=5)
        layer.mtp_block = FakeMTPBlock()

        hidden = layer.forward(
            input_ids=torch.tensor([1, 2, 3]),
            positions=torch.tensor([0, 1, 2]),
            previous_hidden_states=torch.ones(3, 2),
            inputs_embeds=torch.ones(3, 2),
        )

        self.assertEqual(hidden.shape, (3, 2))
        self.assertEqual(layer.mtp_block.seen_buffer.shape, (3, 1, 5))
        self.assertEqual(layer.mtp_block.seen_buffer.dtype, torch.int32)


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
                patch.object(pangu_mod.torch.ops, "custom", fake_ops, raising=False):
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
        moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
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


if __name__ == "__main__":
    unittest.main()
