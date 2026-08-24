# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
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
        meta = SimpleNamespace(slot_mapping_2d=None, get_slot_mapping_2d=lambda: sentinel)
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

    @patch.object(NPUPanguSparseAttention, "_init_MLA_weights")
    @patch.object(NPUPanguSparseAttention, "_init_rotary_emb")
    @patch.object(NPUPanguSparseAttention, "_init_param_sinks")
    @patch.object(NPUPanguSparseAttention, "_align_pagesize")
    @patch.object(NPUPanguSparseAttention, "_init_attention_layers")
    @patch.object(NPUPanguSparseAttention, "_init_mome_layer")
    @patch.object(NPUPanguSparseAttention, "_init_cross_layer_shared_ops")
    def test_constructor_marks_shared_indexer_layer(self, *_init_mocks):
        config = SimpleNamespace(
            num_hidden_layers=2,
            index_topk=4,
            index_head_dim=8,
            indexer_types=["unique", "shared"],
            rope_interleaved=True,
            use_mome=False,
        )
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

        with (
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
        ):
            attention = NPUPanguSparseAttention(
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
                swa_layers=[0],
                param_sink_number=1,
                sliding_window_list=[128],
                cache_config=cache_config,
                prefix="model.layers.1.self_attn",
            )

        self.assertTrue(attention.is_dsa_layer)
        self.assertTrue(attention.skip_topk)

    def test_prepare_phase_inputs_slices_shared_topk_buffer(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        metadata = SimpleNamespace(
            num_decode_tokens=2,
            num_actual_tokens=5,
            slot_mapping=torch.arange(5),
            slot_mapping_2d=None,
            get_slot_mapping_2d=lambda: None,
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
                return model_mod.PanguV2DecoderLayerOutput(
                    hidden_states, residual, h_post, h_res, sk_event,
                    output_buffer,
                )

        model = model_mod.PanguV2Model.__new__(model_mod.PanguV2Model)
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
                self.self_attn = SimpleNamespace(
                    rotary_emb=SimpleNamespace(
                        get_cos_sin=lambda positions: (
                            torch.zeros(positions.shape[0], 2),
                            torch.zeros(positions.shape[0], 2),
                        )
                    )
                )
                self.seen_buffer = None

            def mhc_head(self, hidden_states):
                return hidden_states, None, None, None, None

            def __call__(self, hidden_states, residual, *args):
                self.seen_buffer = args[-1]
                return model_mod.PanguV2DecoderLayerOutput(
                    hidden_states, residual, None, None, None,
                    self.seen_buffer,
                )

        layer = mtp_mod.PanguV2MultiTokenPredictorLayer.__new__(
            mtp_mod.PanguV2MultiTokenPredictorLayer
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


if __name__ == "__main__":
    unittest.main()
