# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

import omni_npu.v1.layers.attention.npu_pangu as pangu_mod
from omni_npu.attention.backends.dsa import NPUDSAMetadataBuilder
from omni_npu.compilation.utils import OP_FIA_SINK, OP_FIA_V2
from omni_npu.v1.layers.attention import npu_pangu_custom_ops as custom_ops_mod
from omni_npu.v1.layers.attention.npu_pangu import (
    NPUPanguSparseAttention,
    PanguAttentionOutputGate,
    _get_slot_mapping_2d,
    npu_pangu_forward_fake,
    npu_pangu_forward,
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


class TestDSparkAttention(unittest.TestCase):
    def test_decode_graph_padding(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.scaling = 0.125
        attention.use_gpt_oss_sink = False
        attention.num_local_heads = 2
        attention.kv_lora_rank = 8
        q_nope = torch.zeros(32, 2, 8)
        q_pe = torch.zeros(32, 2, 4)
        ori_kv_range = torch.tensor([[0, 2048]], dtype=torch.int32)
        dmtp_token_post = torch.tensor([[2048]], dtype=torch.int32)
        section = SimpleNamespace(
            num_tokens=16,
            seq_lens=torch.tensor([2064], dtype=torch.int32),
            query_cumlens=torch.tensor([16], dtype=torch.int32),
            block_table=torch.zeros(1, 4, dtype=torch.int32),
            ori_kv_range=ori_kv_range,
            dmtp_token_post=dmtp_token_post,
        )
        metadata = SimpleNamespace(
            causal=False,
            max_query_len=16,
            decode=section,
        )
        captured = {}

        def fake_dmtp(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return (torch.full((16, 2, 8), 3.0),)

        custom_ops = SimpleNamespace(npu_ai_infra_diffusion_mtp_attention=fake_dmtp)
        with patch.object(pangu_mod.torch.ops, "custom", custom_ops, create=True):
            output = attention._apply_SWA_attention_decode(
                q_nope,
                q_pe,
                (torch.zeros(1), torch.zeros(1)),
                metadata,
            )

        self.assertEqual(output.shape, (32, 2, 8))
        self.assertTrue(torch.all(output[:16] == 3))
        self.assertTrue(torch.all(output[16:] == 0))
        self.assertTrue(output.is_contiguous())
        self.assertEqual(captured["args"][0].shape[0], 16)
        self.assertIs(captured["kwargs"]["ori_kv_range"], ori_kv_range)
        self.assertIs(captured["kwargs"]["dmtp_token_post"], dmtp_token_post)


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


class TestPanguCustomOps(unittest.TestCase):
    def test_cla_swa_kv_cache_update_real_and_fake(self):
        kv = torch.randn(3, 6)
        cos = torch.randn(3, 2)
        sin = torch.randn(3, 2)
        kv_cache_0 = torch.randn(1, 4, 6)
        kv_cache_1 = torch.randn(1, 4, 2)
        k_nope = torch.randn(3, 4)
        k_pe = torch.randn(3, 2)
        prepare_kv = Mock(return_value=(k_nope, k_pe))
        layer = SimpleNamespace(
            cla_swa_attn_name="model.cla_swa",
            kv_lora_rank=4,
            qk_rope_head_dim=2,
            _prepare_cla_swa_kv=prepare_kv,
        )
        context = SimpleNamespace(no_compile_layers={"layer": layer}, attn_metadata={"model.cla_swa": object()})

        with patch.object(custom_ops_mod, "get_forward_context", return_value=context):
            output = custom_ops_mod.npu_pangu_cla_swa_kv_cache_update(
                kv, cos, sin, kv_cache_0, kv_cache_1, "layer",
            )
            fake_output = custom_ops_mod.npu_pangu_cla_swa_kv_cache_update_fake(
                kv, cos, sin, kv_cache_0, kv_cache_1, "layer",
            )

        self.assertIs(output[0], k_nope)
        self.assertIs(output[1], k_pe)
        self.assertIs(output[2], kv_cache_0)
        self.assertIs(output[3], kv_cache_1)
        args = prepare_kv.call_args.args
        self.assertIs(args[0], kv)
        self.assertIs(args[1], cos)
        self.assertIs(args[2], sin)
        self.assertIs(args[3][0], kv_cache_0)
        self.assertIs(args[3][1], kv_cache_1)
        self.assertIs(args[4], context.attn_metadata["model.cla_swa"])
        self.assertEqual(fake_output[0].shape, (3, 4))
        self.assertEqual(fake_output[1].shape, (3, 2))
        self.assertEqual(fake_output[2].shape, kv_cache_0.shape)
        self.assertEqual(fake_output[3].shape, kv_cache_1.shape)


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

    def test_mixed_batch_slices_inputs_with_phase(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.on_ascend950 = False
        metadata = SimpleNamespace(
            num_decode_tokens=2,
            num_actual_tokens=5,
            slot_mapping=torch.arange(5),
            slot_mapping_2d=torch.arange(10).view(5, 2),
            prefill=SimpleNamespace(),
            decode=SimpleNamespace(),
        )
        hidden = torch.arange(15).view(5, 3)
        cos = torch.arange(10).view(5, 2)
        sin = cos + 100

        prefill = attention._prepare_phase_inputs(hidden, cos, sin, metadata, "prefill")
        torch.testing.assert_close(prefill[0], hidden[2:5])
        torch.testing.assert_close(prefill[1], cos[2:5])
        torch.testing.assert_close(prefill[2], sin[2:5])

        decode = attention._prepare_phase_inputs(hidden, cos, sin, metadata, "decode")
        torch.testing.assert_close(decode[0], hidden[:2])
        torch.testing.assert_close(decode[1], cos[:2])
        torch.testing.assert_close(decode[2], sin[:2])
        with self.assertRaisesRegex(ValueError, "Unsupported attention phase"):
            attention._prepare_phase_inputs(hidden, cos, sin, metadata, "invalid")

    def test_cla_cp_dispatch(self):
        hidden = torch.ones(2, 3)

        layer = SimpleNamespace(
            prefix="model.layers.7.self_attn",
            cla_swa_attn_name="model.cla_swa_layers.40.attn",
            is_cla_reuse_layer=True,
            is_cp_layer=True,
            is_attn_sp_layer=False,
            enable_flashcomm2=False,
            is_dsa_layer=True,
            tp_size=2,
            moe_comm_strategy="allgather_reducescatter",
            _forward_cla_prefill_cp=Mock(return_value=hidden + 1),
        )
        global_metadata = SimpleNamespace(
            num_actual_tokens=16,
            num_decode_tokens=0,
            num_decodes=0,
            num_prefills=1,
            prefill=SimpleNamespace(),
        )
        local_metadata = SimpleNamespace(prefill=SimpleNamespace())
        context = SimpleNamespace(
            no_compile_layers={"layer": layer},
            attn_metadata={
                "model.layers.7.self_attn.attn": global_metadata,
                "model.cla_swa_layers.40.attn": local_metadata,
            },
        )

        with patch("omni_npu.v1.layers.attention.npu_pangu.get_forward_context", return_value=context):
            output = npu_pangu_forward(hidden, torch.empty(0), torch.empty(0), "layer")

        torch.testing.assert_close(output, hidden + 1)
        self.assertIs(layer._forward_cla_prefill_cp.call_args.args[3], global_metadata)
        self.assertIs(layer._forward_cla_prefill_cp.call_args.args[4], local_metadata)

class TestPanguMomeDisabled(unittest.TestCase):
    def test_swa_pa_policy_for_decode_first_and_later_chunks(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        for prefill, first_chunk_pa, expected in (
            (None, False, True),
            (SimpleNamespace(chunked_context=None), False, False),
            (SimpleNamespace(chunked_context=None), True, True),
            (SimpleNamespace(chunked_context=object()), False, True),
        ):
            with self.subTest(prefill=prefill, first_chunk_pa=first_chunk_pa):
                attention.first_chunk_pa = first_chunk_pa
                self.assertEqual(attention._use_swa_prefill_pa(SimpleNamespace(prefill=prefill)), expected)

    def test_flashcomm2_prefill_without_mome_uses_absorb_attention(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.use_mome = False
        attention.enable_mome_sp = False
        attention.tp_size = 1
        attention.moe_comm_strategy = "allreduce"
        attention.num_heads = 1
        attention.num_local_heads = 1
        attention.qk_head_dim = 5
        attention.qk_nope_head_dim = 3
        attention.qk_rope_head_dim = 2
        attention.kv_lora_rank = 3
        attention.v_head_dim = 4
        attention.rope_interleave = True
        attention.W_UK_T = object()
        attention.sharded_o_proj = False
        attention._use_swa_prefill_pa = Mock(return_value=True)
        attention.q_a_proj = Mock(return_value=torch.randn(2, 3))
        attention.q_a_layernorm = Mock(side_effect=lambda value: value)
        attention.q_b_proj = Mock(return_value=torch.randn(2, 5))
        attention.kv_a_proj_with_mqa = Mock(return_value=torch.randn(2, 5))
        kv_cache = (torch.randn(1), torch.randn(1))
        attention.attn = SimpleNamespace(kv_cache=kv_cache)
        attention._npu_kvrmsnorm_rope_cache = Mock(return_value=(kv_cache, None))
        attn_output = torch.randn(2, 4)
        attention._apply_SWA_attention_prefill_absorb = Mock(return_value=attn_output)
        projected = torch.randn(2, 6)
        attention._apply_o_proj = Mock(return_value=projected)
        metadata = SimpleNamespace(num_actual_tokens=2, num_decode_tokens=0)
        hidden_states = torch.randn(2, 4)
        cos = torch.randn(2, 2)

        with (
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(virtual_engine=0),
            ),
            patch("torch_npu.npu_rotary_mul", side_effect=lambda value, *_args, **_kwargs: value),
            patch("torch_npu.npu_transpose_batchmatmul", side_effect=lambda value, *_args, **_kwargs: value),
        ):
            output = attention._forward_prefill_FC2(hidden_states, cos, cos, metadata)

        self.assertIs(output, projected)
        attention._apply_SWA_attention_prefill_absorb.assert_called_once()
        attention._apply_o_proj.assert_called_once_with(attn_output)

    def test_flashcomm2_without_mome_gathers_heads_for_decode(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.use_mome = False
        attention.enable_flashcomm2 = True
        attention.is_dsa_layer = False
        attention.num_heads = 4
        attention.num_local_heads = 1
        attention.v_head_dim = 2
        attn_output = torch.randn(1, 2)
        gathered = torch.randn(1, 8)
        projected = torch.randn(1, 3)
        tp_group = SimpleNamespace(all_gather=Mock(return_value=gathered))

        with (
            patch("omni_npu.v1.layers.attention.npu_pangu.get_tp_group", return_value=tp_group),
            patch.object(attention, "_apply_o_proj", return_value=projected) as mock_o_proj,
        ):
            output = attention._mla_epilog(attn_output)

        self.assertIs(output, projected)
        tp_group.all_gather.assert_called_once_with(attn_output, dim=1)
        mock_o_proj.assert_called_once_with(gathered)


class TestPanguGptOssSink(unittest.TestCase):
    def test_gpt_oss_sinks_are_initialized_in_fp32(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.hf_config = SimpleNamespace(use_gpt_oss_sink=True, torch_dtype=torch.bfloat16)
        attention.param_sink_number = 0
        attention.num_heads = 4
        attention.num_local_heads = 2
        attention.is_cla_reuse_layer = True
        empty_tensors = [torch.empty(2, dtype=torch.float32), torch.empty(2, dtype=torch.float32)]

        with patch("torch.empty", side_effect=empty_tensors) as mock_empty:
            attention._init_gpt_oss_sink()

        self.assertEqual(attention.gpt_oss_sink.dtype, torch.float32)
        self.assertEqual(attention.gpt_oss_sink_swa.dtype, torch.float32)
        self.assertEqual(mock_empty.call_args_list[0].kwargs["dtype"], torch.float32)
        self.assertEqual(mock_empty.call_args_list[1].kwargs["dtype"], torch.float32)

        loaded_weight = torch.arange(4, dtype=torch.float32)
        with patch(
            "omni_npu.v1.layers.attention.npu_pangu.get_tp_group",
            return_value=SimpleNamespace(rank_in_group=1),
        ):
            attention.gpt_oss_sink.weight_loader(attention.gpt_oss_sink, loaded_weight)
        torch.testing.assert_close(attention.gpt_oss_sink, loaded_weight[2:])

        with self.assertRaisesRegex(ValueError, "gpt_oss_sink must have shape"):
            attention.gpt_oss_sink.weight_loader(attention.gpt_oss_sink, loaded_weight[:3])
        invalid_param = torch.nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "local gpt_oss_sink parameter must have shape"):
            attention.gpt_oss_sink.weight_loader(invalid_param, loaded_weight)

        attention.num_local_heads = attention.num_heads
        full_param = torch.nn.Parameter(torch.empty(4, dtype=torch.float32), requires_grad=False)
        with patch("omni_npu.v1.layers.attention.npu_pangu.get_tp_group") as mock_tp_group:
            attention._load_gpt_oss_sink_weight(full_param, loaded_weight)
        torch.testing.assert_close(full_param, loaded_weight)
        mock_tp_group.assert_not_called()

    def test_gpt_oss_fia_runtime_capture_and_padding(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.num_local_heads = 2
        attention.kv_lora_rank = 4
        query = torch.randn(5, 2, 4)
        op_output = torch.randn(2, 3, 4)

        with (
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(capturing=False),
            ),
            patch.object(
                NPUPanguSparseAttention, "_apply_gpt_oss_fia", return_value=op_output
            ) as mock_fia,
        ):
            output = attention._run_gpt_oss_fia({"query": query}, num_tokens=5, num_actual_tokens=3)

        torch.testing.assert_close(output[:, :3], op_output)
        torch.testing.assert_close(output[:, 3:], torch.zeros(2, 2, 4))
        mock_fia.assert_called_once()

        attention.use_gpt_oss_sink_rescale = False
        attention.gpt_oss_sink = torch.tensor([0.25, -0.5], dtype=torch.float32)
        attention.attn = SimpleNamespace(layer_name="global_swa")
        kwargs = {"query": query, "key_sink": torch.empty(0)}
        native_output = torch.randn(2, 5, 4)
        with (
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(capturing=False),
            ),
            patch(
                "torch_npu.npu_fused_infer_attention_score_v2",
                return_value=(native_output, torch.empty(0)),
            ) as mock_native_fia,
        ):
            runtime_output = attention._apply_gpt_oss_fia(dict(kwargs))

        self.assertIs(runtime_output, native_output)
        native_kwargs = mock_native_fia.call_args.kwargs
        self.assertIs(native_kwargs["query"], query)
        torch.testing.assert_close(native_kwargs["learnable_sink"], attention.gpt_oss_sink)
        self.assertNotIn("key_sink", native_kwargs)

        with (
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(capturing=True),
            ),
            patch("omni_npu.v1.layers.attention.npu_pangu.capture_graph_task") as mock_capture,
        ):
            captured_output = attention._apply_gpt_oss_fia(dict(kwargs), output_shape=(2, 5, 4), num_tokens=5)

        torch.testing.assert_close(captured_output, torch.zeros_like(captured_output))
        self.assertIs(mock_capture.call_args.kwargs["op_desc"], OP_FIA_V2)
        self.assertEqual(mock_capture.call_args.kwargs["num_tokens"], 5)

    def test_fia_rescale_uses_ordinary_sink_attention_and_lse(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.num_local_heads = 2
        attention.use_aicpu_fa_tiling = False
        attention.use_gpt_oss_sink_rescale = True
        attention.gpt_oss_sink = torch.tensor([0.25, -0.5], dtype=torch.float32)
        query = torch.randn(3, 2, 4)
        op_output = torch.ones(2, 3, 3)
        softmax_lse = torch.tensor([[[0.5], [1.0]], [[-0.25], [0.75]], [[0.0], [0.25]]])
        kwargs = {
            "query": query,
            "key": torch.randn(3, 2, 4),
            "value": torch.randn(3, 2, 3),
            "query_rope": torch.randn(3, 2, 2),
            "key_rope": torch.randn(3, 2, 2),
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "input_layout": "TND_NTD",
            "actual_seq_qlen": torch.tensor([3]),
            "actual_seq_kvlen": torch.tensor([3]),
            "key_sink": torch.empty(0),
        }

        with (
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(capturing=False),
            ),
            patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(op_output, softmax_lse),
            ) as mock_fia,
        ):
            output = attention._apply_gpt_oss_fia(kwargs)

        expected_scale = torch.sigmoid(softmax_lse.squeeze(-1) - attention.gpt_oss_sink.view(1, 2))
        torch.testing.assert_close(output, op_output * expected_scale.transpose(0, 1).unsqueeze(-1))
        op_kwargs = mock_fia.call_args.kwargs
        self.assertTrue(op_kwargs["return_softmax_lse"])
        self.assertNotIn("learnable_sink", op_kwargs)
        self.assertNotIn("key_sink", op_kwargs)

    def test_fia_rescale_capture_uses_graph_only_without_aicpu_tiling(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.num_local_heads = 2
        attention.use_aicpu_fa_tiling = True
        query = torch.randn(3, 2, 4)
        op_output = torch.ones(2, 3, 3)
        softmax_lse = torch.zeros(3, 2, 1)
        kwargs = {"query": query, "input_layout": "TND_NTD"}

        with (
            patch.object(attention, "_add_fia_metadata", side_effect=lambda value, _caller, _recompute: value),
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(capturing=True),
            ),
            patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(op_output, softmax_lse),
            ) as mock_fia,
            patch("omni_npu.v1.layers.attention.npu_pangu.capture_graph_task") as mock_capture,
        ):
            output = attention._apply_gpt_oss_fia_rescale(kwargs, torch.zeros(2), op_output.shape, 3, None, "decode")

        torch.testing.assert_close(output, op_output * 0.5)
        mock_fia.assert_called_once()
        mock_capture.assert_not_called()

        attention.use_aicpu_fa_tiling = False
        with (
            patch.object(attention, "_add_fia_metadata", side_effect=lambda value, _caller, _recompute: value),
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(capturing=True),
            ),
            patch("torch.ops.custom.npu_fused_infer_attention_sink") as mock_fia,
            patch("omni_npu.v1.layers.attention.npu_pangu.capture_graph_task") as mock_capture,
        ):
            captured_output = attention._apply_gpt_oss_fia_rescale(
                dict(kwargs), torch.zeros(2), op_output.shape, 3, "cla_swa", "cla_decode",
            )

        torch.testing.assert_close(captured_output, torch.zeros_like(captured_output))
        mock_fia.assert_not_called()
        self.assertIs(mock_capture.call_args.kwargs["op_desc"], OP_FIA_SINK)
        self.assertEqual(mock_capture.call_args.kwargs["num_tokens"], 3)

    def test_direct_swa_prefill_forwards_cla_fia_metadata_policy(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.use_gpt_oss_sink = True
        attention.num_local_heads = 2
        attention.v_head_dim = 3
        attention.scaling = 0.5
        attention.is_fa_metadata_producer = False
        attention.attn = SimpleNamespace(
            layer_name="global_swa",
            impl=SimpleNamespace(SHARE_MASK_TRIL_SPARSE=torch.empty(0)),
        )
        local_attention = SimpleNamespace(
            layer_name="cla_swa",
            impl=SimpleNamespace(SHARE_MASK_TRIL_SPARSE=torch.empty(0)),
        )
        metadata = SimpleNamespace(prefill=SimpleNamespace(query_cumlens=torch.tensor([2])))
        output = torch.randn(2, 2, 3)
        tensors = [torch.randn(2, 2, 4), torch.randn(2, 2, 2), torch.randn(2, 2, 4),
                   torch.randn(2, 2, 2), torch.randn(2, 2, 3)]

        with patch.object(attention, "_apply_gpt_oss_fia", return_value=output) as mock_fia:
            attention._apply_SWA_attention_prefill(
                *tensors,
                attn_metadata=metadata,
                attention_layer=local_attention,
                sliding_window=512,
                learnable_sink=torch.zeros(2),
                metadata_caller="cla_prefill",
                recompute_metadata=True,
            )

        self.assertEqual(mock_fia.call_args.kwargs["layer_name"], "cla_swa")
        self.assertEqual(mock_fia.call_args.kwargs["metadata_caller"], "cla_prefill")
        self.assertTrue(mock_fia.call_args.kwargs["recompute_metadata"])

    def test_cla_reuse_layer_uses_separate_shared_fia_metadata(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.num_local_heads = 2
        attention.use_aicpu_fa_tiling = True
        attention.is_cla_fa_metadata_producer = True
        attention._fia_aic_core_num = 24
        attention._fia_aiv_core_num = 48
        kwargs = {
            "query": torch.randn(3, 2, 4),
            "key": torch.randn(1, 16, 4),
            "value": torch.randn(1, 16, 4),
            "query_rope": torch.randn(3, 2, 2),
            "num_key_value_heads": 1,
            "actual_seq_qlen": torch.tensor([3]),
            "actual_seq_kvlen": torch.tensor([3]),
            "block_table": torch.zeros(1, 1, dtype=torch.int32),
            "block_size": 16,
            "sparse_mode": 4,
            "pre_tokens": 511,
            "next_tokens": 0,
        }
        fia_metadata = torch.empty(1024, dtype=torch.int32)

        with (
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.npu_fused_infer_attention_sink_metadata",
                return_value=fia_metadata,
            ) as mock_metadata,
        ):
            output = attention._add_fia_metadata(kwargs, "cla_prefill_absorb", attention.is_cla_fa_metadata_producer)

        self.assertIs(output["meta_data"], fia_metadata)
        _, recompute, caller = mock_metadata.call_args.args
        self.assertTrue(recompute)
        self.assertEqual(caller, "cla_prefill_absorb")

        with self.assertRaisesRegex(ValueError, "metadata caller is required"):
            attention._add_fia_metadata(dict(kwargs), None)

    def test_sfa_rescale_uses_pioneer_softmax_statistics(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.num_local_heads = 2
        attention.use_gpt_oss_sink_rescale = True
        attention.gpt_oss_sink = torch.tensor([0.5, -0.25], dtype=torch.float32)
        attention.dummy_value_cache = torch.empty(1, 2, 1, 4)
        attention.scaling = 0.125
        q_nope = torch.randn(2, 2, 4)
        q_pe = torch.randn(2, 2, 2)
        kv_cache = (torch.randn(1, 2, 4), torch.randn(1, 2, 2))
        op_output = torch.ones(2, 2, 4)
        softmax_max = torch.tensor([[[0.5, 1.0], [-0.5, 0.25]]])
        softmax_sum = torch.tensor([[[2.0, 4.0], [1.5, 3.0]]])

        with (
            patch.object(
                torch.ops.custom,
                "npu_ai_infra_sparse_flash_attention_pioneer",
                return_value=(op_output, softmax_max, softmax_sum),
                create=True,
            ) as mock_pioneer,
            patch.object(
                torch.ops.custom,
                "npu_ai_infra_sparse_flash_attention",
                create=True,
            ) as mock_legacy,
        ):
            output = attention._apply_gpt_oss_sfa(
                q_nope, q_pe, kv_cache, torch.zeros(2, 1, 1, dtype=torch.int32),
                torch.tensor([2]), torch.tensor([2]), torch.zeros(1, 1, dtype=torch.int32),
            )

        softmax_lse = (softmax_max + torch.log(softmax_sum)).squeeze(0)
        expected_scale = torch.sigmoid(softmax_lse - attention.gpt_oss_sink.view(1, 2))
        torch.testing.assert_close(output, op_output * expected_scale.unsqueeze(-1))
        op_kwargs = mock_pioneer.call_args.kwargs
        self.assertIs(op_kwargs["query"], q_nope)
        self.assertIs(op_kwargs["query_rope"], q_pe)
        self.assertTrue(op_kwargs["return_softmax_lse"])
        self.assertNotIn("batch_invariant", op_kwargs)
        self.assertEqual(op_kwargs["layout_kv"], "PA_BSND")
        self.assertNotIn("sinks", op_kwargs)
        self.assertNotIn("key_sink", op_kwargs)
        mock_legacy.assert_not_called()

    def test_sfa_without_rescale_keeps_native_sink_operator(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.use_gpt_oss_sink_rescale = False
        attention.gpt_oss_sink = torch.tensor([0.5, -0.25], dtype=torch.float32)
        attention.scaling = 0.125
        q_nope = torch.randn(2, 2, 4)
        q_pe = torch.randn(2, 2, 2)
        kv_cache = (torch.randn(1, 2, 4), torch.randn(1, 2, 2))
        op_output = torch.ones(2, 2, 4)

        with (
            patch.object(
                torch.ops.custom,
                "npu_ai_infra_sparse_flash_attention",
                return_value=(op_output,),
                create=True,
            ) as mock_legacy,
            patch.object(
                torch.ops.custom,
                "npu_ai_infra_sparse_flash_attention_pioneer",
                create=True,
            ) as mock_pioneer,
        ):
            output = attention._apply_gpt_oss_sfa(
                q_nope, q_pe, kv_cache, torch.zeros(2, 1, 1, dtype=torch.int32),
                torch.tensor([2]), torch.tensor([2]), torch.zeros(1, 1, dtype=torch.int32),
            )

        self.assertIs(output, op_output)
        op_kwargs = mock_legacy.call_args.kwargs
        torch.testing.assert_close(op_kwargs["query"], torch.cat([q_nope, q_pe], dim=-1))
        self.assertIs(op_kwargs["sinks"], attention.gpt_oss_sink)
        mock_pioneer.assert_not_called()

    def test_dsa_cp_prefill_and_decode_route_gpt_oss_sink_through_sfa(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.use_gpt_oss_sink = True
        attention.param_sink_number = 0
        attention.num_local_heads = 1
        attention.kv_lora_rank = 2
        attention.v_head_dim = 3
        attention.W_UV = object()
        attention.use_mome = False
        attention.enable_flashcomm2 = False
        q_nope = torch.randn(2, 1, 2)
        q_pe = torch.randn(2, 1, 1)
        kv_cache = (torch.randn(1, 4, 2), torch.randn(1, 4, 1))
        topk_indices = torch.zeros(4, 1, dtype=torch.int32)
        metadata = SimpleNamespace(
            query_cumlens=torch.tensor([2]),
            seq_lens=torch.tensor([2]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
        )
        attn_metadata = SimpleNamespace(prefill=metadata, decode=None)
        sp_manager = SimpleNamespace(
            cp_attn_meta=Mock(
                return_value=(metadata.query_cumlens, metadata.seq_lens, None, metadata.block_table)
            )
        )
        latent = torch.randn(2, 1, 2)
        projected = torch.randn(2, 1, 3)

        with (
            patch.object(attention, "_apply_gpt_oss_sfa", return_value=latent) as mock_sfa,
            patch("torch_npu.npu_transpose_batchmatmul", return_value=projected) as mock_v_up,
            patch.object(attention, "_apply_o_proj", side_effect=lambda value: value),
        ):
            cp_output = attention._apply_DSA_attention_cp(
                q_nope, q_pe, kv_cache, topk_indices, sp_manager, attn_metadata,
            )
            prefill_output = attention._apply_DSA_attention(q_nope, q_pe, kv_cache, topk_indices, attn_metadata)
            attn_metadata.prefill = None
            attn_metadata.decode = metadata
            decode_output = attention._apply_DSA_attention(q_nope, q_pe, kv_cache, topk_indices, attn_metadata)
            self.assertEqual(mock_v_up.call_count, 2)
            decode_projected = attention._mla_epilog(decode_output, attn_metadata)
            prefill_projected = attention._mla_epilog(prefill_output)

        torch.testing.assert_close(cp_output, projected.reshape(2, 3))
        torch.testing.assert_close(prefill_output, projected.reshape(2, 3))
        torch.testing.assert_close(decode_output, latent)
        torch.testing.assert_close(decode_projected, projected.reshape(2, 3))
        self.assertIs(prefill_projected, prefill_output)
        self.assertEqual(mock_v_up.call_count, 3)
        self.assertEqual(mock_v_up.call_args.kwargs, {"perm_x1": (1, 0, 2), "perm_y": (1, 0, 2)})
        self.assertEqual(mock_sfa.call_count, 3)
        self.assertIs(mock_sfa.call_args_list[0].args[3], topk_indices)
        torch.testing.assert_close(mock_sfa.call_args_list[1].args[3], topk_indices[:2])
        torch.testing.assert_close(mock_sfa.call_args_list[2].args[3], topk_indices[:2])

    def test_dsa_cp_and_normal_paths_keep_legacy_kernel_arguments(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.use_gpt_oss_sink = False
        attention.param_sink_number = 1
        attention.num_local_heads = 1
        attention.kv_lora_rank = 2
        attention.v_head_dim = 3
        attention.W_UV = object()
        attention.scaling = 0.125
        attention.on_ascend950 = True
        attention.sink_kv = torch.randn(1, 3)
        attention.sink_k_nope = torch.randn(1, 2)
        attention.dummy_value_cache = torch.empty(0)
        q_nope, q_pe = torch.randn(2, 1, 2), torch.randn(2, 1, 1)
        kv_cache = (torch.randn(1, 16, 2), torch.randn(1, 16, 1))
        topk = torch.zeros(4, 1, dtype=torch.int32)
        metadata = SimpleNamespace(query_cumlens=torch.tensor([2]), seq_lens=torch.tensor([5]), block_table=object())
        cp_query, cp_lens, cp_table = torch.tensor([1, 2]), torch.tensor([3, 5]), object()
        manager = SimpleNamespace(cp_attn_meta=lambda: (cp_query, cp_lens, None, cp_table))
        for use_cp in (False, True):
            for dtype in ("auto", "int8_ds_mla", "fp8_ds_mla"):
                attention.cache_config = SimpleNamespace(cache_dtype=dtype)
                attention.dummy_value_cache_hif8_fp8 = torch.empty(0)
                latent, projected = torch.randn(2, 1, 2), torch.randn(2, 1, 3)
                quantized = dtype == "int8_ds_mla" or (dtype == "fp8_ds_mla" and not use_cp)
                op_name = "npu_ai_infra_kv_quant_sparse_flash_attention" if quantized else (
                    "npu_ai_infra_sparse_flash_attention_pioneer"
                )
                with (
                    self.subTest(use_cp=use_cp, dtype=dtype),
                    patch.object(
                        torch.ops.custom, op_name, return_value=latent if quantized else (latent,)
                    ) as mock_op,
                    patch("torch_npu.npu_transpose_batchmatmul", return_value=projected),
                ):
                    args = (q_nope, q_pe, kv_cache, topk)
                    attn_metadata = SimpleNamespace(prefill=metadata, decode=None)
                    if use_cp:
                        output = attention._apply_DSA_attention_cp(*args, manager, attn_metadata)
                    else:
                        output = attention._apply_DSA_attention(*args, attn_metadata)
                torch.testing.assert_close(output, projected.reshape(2, 3))
                kwargs = mock_op.call_args.kwargs
                expected_query_lens = cp_query if use_cp else metadata.query_cumlens
                torch.testing.assert_close(kwargs["actual_seq_lengths_query"], expected_query_lens)
                torch.testing.assert_close(kwargs["actual_seq_lengths_kv"], cp_lens if use_cp else metadata.seq_lens)
                self.assertIs(kwargs["block_table"], cp_table if use_cp else metadata.block_table)
                self.assertIs(kwargs["key_sink"], attention.sink_kv)
                self.assertIs(kwargs["value_sink"], attention.sink_k_nope)
                torch.testing.assert_close(kwargs["sparse_indices"], topk if use_cp else topk[:2])
                torch.testing.assert_close(kwargs["query"], torch.cat([q_nope, q_pe], dim=-1))


class TestPanguCLA(unittest.TestCase):
    def test_cla_initializes_branch_modules_and_cache_aliases(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.hidden_size = 16
        attention.q_lora_rank = 4
        attention.kv_lora_rank = 3
        attention.qk_nope_head_dim = 5
        attention.qk_rope_head_dim = 2
        attention.qk_head_dim = 7
        attention.v_head_dim = 6
        attention.num_heads = 4
        attention.num_local_heads = 2
        attention.quant_config = None
        attention.layer_name = "model.layers.7.self_attn"
        attention.layer_idx = 7
        attention.hf_config = SimpleNamespace(
            rms_norm_eps=1e-6,
            num_hidden_layers=32,
            num_nextn_predict_layers=1,
            cla_explicit_mapping=[[7, 3]],
        )
        attention.is_cp_layer = False
        attention.is_attn_sp_layer = False
        attention.is_cla_reuse_layer = True
        attention.cla_source_layer_idx = 3
        attention.cla_swa_gate_window = 512
        attention.sliding_window = 1024
        attention.aligned_window_size = 512
        attention.split_q_up_in_multistream = False
        attention.sharded_o_proj = False
        attention.enable_flashcomm2 = True
        attention.is_dsa_layer = False
        attention.skip_topk = True
        attention.index_head_dim = 32
        attention.scaling = 0.125
        attention.cache_config = object()
        attention.cache_dtype_str = "auto"
        attention.page_size_padded = 128

        def module(*_args, **kwargs):
            return SimpleNamespace(prefix=kwargs.get("prefix"))

        with (
            patch("omni_npu.v1.layers.attention.npu_pangu.ReplicatedLinear", side_effect=module),
            patch("omni_npu.v1.layers.attention.npu_pangu.RMSNorm", side_effect=module),
            patch("omni_npu.v1.layers.attention.npu_pangu.ColumnParallelFlashCommLinear", side_effect=module),
            patch("omni_npu.v1.layers.attention.npu_pangu.RowParallelFlashCommLinear", side_effect=module),
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.PanguAttentionOutputGate",
                side_effect=module,
            ) as gate_cls,
            patch("omni_npu.v1.layers.attention.npu_pangu.DSAAttention", side_effect=module),
            patch("omni_npu.v1.layers.attention.npu_pangu.MLASWAAttention", side_effect=module),
        ):
            attention._init_MLA_weights()
            attention._init_attention_layers()

            attention.is_cla_reuse_layer = False
            attention._init_MLA_weights()
            self.assertTrue(hasattr(attention, "kv_a_proj_with_mqa"))
            self.assertTrue(hasattr(attention, "kv_a_layernorm"))
            del attention.kv_a_proj_with_mqa
            del attention.kv_a_layernorm
            attention.is_cla_reuse_layer = True

        self.assertFalse(hasattr(attention, "kv_a_proj_with_mqa"))
        self.assertFalse(hasattr(attention, "kv_a_layernorm"))
        self.assertEqual(attention.kv_a_proj_with_mqa_swa.prefix, "model.layers.7.self_attn.kv_a_proj_with_mqa_swa")
        self.assertEqual(attention.attn.kv_sharing_target_layer_name, "model.layers.3.self_attn.attn")
        self.assertEqual(attention.cla_swa_attn_name, "model.cla_swa_layers.40.attn")
        self.assertEqual(attention.attn_cla_swa.prefix, attention.cla_swa_attn_name)
        self.assertTrue(gate_cls.call_args_list[0].args[-1])
        self.assertTrue(gate_cls.call_args_list[1].args[-1])

        with (
            patch("omni_npu.v1.layers.attention.npu_pangu.npu_fused_infer_attention_sink_metadata", None),
            patch("omni_npu.v1.layers.attention.npu_pangu.CrossLayerSharedOp") as shared_op,
        ):
            attention.on_ascend950 = False
            attention._init_cross_layer_shared_ops()
        callers = shared_op.call_args.kwargs["callers"]
        self.assertIn("cla_prefill_absorb_cp", callers)

    def test_cla_process_weights_builds_global_and_local_absorb_matrices(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.num_local_heads = 2
        attention.qk_nope_head_dim = 2
        attention.v_head_dim = 1
        attention.kv_lora_rank = 2
        attention.is_cla_reuse_layer = True
        attention.param_sink_number = 0
        global_weight = torch.arange(12, dtype=torch.float32).view(2, 6)
        local_weight = global_weight + 20
        attention.kv_b_proj = SimpleNamespace(weight=global_weight)
        attention.kv_b_proj_swa = SimpleNamespace(weight=local_weight)

        attention.process_weights_after_loading()

        global_heads = global_weight.t().view(2, 3, 2)
        local_heads = local_weight.t().view(2, 3, 2)
        torch.testing.assert_close(attention.W_UK_T, global_heads[:, :2])
        torch.testing.assert_close(attention.W_UV, global_heads[:, 2:].transpose(1, 2))
        torch.testing.assert_close(attention.W_UK_T_cla_swa, local_heads[:, :2])
        torch.testing.assert_close(attention.W_UV_cla_swa, local_heads[:, 2:].transpose(1, 2))

        attention.param_sink_number = 1
        with self.assertRaisesRegex(ValueError, "Token KV sinks are unsupported"):
            attention.process_weights_after_loading()
        attention.kv_a_layernorm = lambda value: value + 1
        attention.param_sink_compressed_kv = torch.tensor([[2.0, 3.0]])
        attention.param_sink_k_pe = torch.tensor([[4.0]])
        attention.process_weights_after_loading()
        torch.testing.assert_close(attention.sink_kv, torch.tensor([[[3.0, 4.0, 4.0]]]))

    def test_cla_mla_prefill_flashcomm2_dispatch_and_mome_fallback(self):
        hidden = torch.randn(2, 4)
        global_metadata = SimpleNamespace(
            num_actual_tokens=2,
            num_decode_tokens=0,
            num_decodes=0,
            num_prefills=1,
            prefill=SimpleNamespace(),
        )
        local_metadata = SimpleNamespace(prefill=SimpleNamespace())

        for use_mome in (False, True):
            with self.subTest(use_mome=use_mome):
                fc2_output = hidden + 1
                regular_fc2_output = hidden + 2
                layer = SimpleNamespace(
                    prefix="model.layers.7.self_attn",
                    cla_swa_attn_name="model.cla_swa_layers.40.attn",
                    is_cla_reuse_layer=True,
                    is_cp_layer=False,
                    is_attn_sp_layer=False,
                    enable_flashcomm2=True,
                    is_dsa_layer=False,
                    use_mome=use_mome,
                    tp_size=1,
                    moe_comm_strategy="allgather_reducescatter",
                    _forward_cla_prefill_FC2=Mock(return_value=fc2_output),
                    _forward_prefill_FC2=Mock(return_value=regular_fc2_output),
                )
                context = SimpleNamespace(
                    no_compile_layers={"layer": layer},
                    attn_metadata={
                        "model.layers.7.self_attn.attn": global_metadata,
                        "model.cla_swa_layers.40.attn": local_metadata,
                    },
                )

                with patch("omni_npu.v1.layers.attention.npu_pangu.get_forward_context", return_value=context):
                    output = npu_pangu_forward(hidden, torch.empty(0), torch.empty(0), "layer")

                if use_mome:
                    torch.testing.assert_close(output, regular_fc2_output)
                    layer._forward_prefill_FC2.assert_called_once()
                    layer._forward_cla_prefill_FC2.assert_not_called()
                else:
                    self.assertIs(output, fc2_output)
                    layer._forward_cla_prefill_FC2.assert_called_once_with(
                        hidden,
                        unittest.mock.ANY,
                        unittest.mock.ANY,
                        global_metadata,
                        local_metadata,
                    )
                    layer._forward_prefill_FC2.assert_not_called()

    def test_cla_mla_prefill_flashcomm2_combines_branches_before_all_to_all(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.use_mome = False
        attention.tp_size = 2
        attention.moe_comm_strategy = "allgather_reducescatter"
        attention.num_heads = 4
        attention.num_local_heads = 2
        attention.qk_head_dim = 3
        attention.qk_nope_head_dim = 2
        attention.qk_rope_head_dim = 1
        attention.q_lora_rank = 4
        attention.kv_lora_rank = 2
        attention.v_head_dim = 3
        attention.layer_name = "model.layers.7.self_attn"
        attention.W_UK_T = object()
        attention.W_UK_T_cla_swa = object()
        attention.W_UV = object()
        attention.W_UV_cla_swa = object()
        attention.cla_swa_gate_window = 512
        attention.is_cla_fa_metadata_producer = True
        attention.use_gpt_oss_sink = False
        attention.gpt_oss_sink_swa = None
        attention.sharded_o_proj = False

        hidden_states = torch.randn(2, 5)
        cos = torch.randn(3, 1)
        q_lora_local = torch.randn(2, 4)
        q_lora = torch.randn(4, 4)
        q = torch.randn(3, 2, 3)
        q_nope_global = torch.randn(3, 2, 2)
        q_nope_local = torch.randn(3, 2, 2)
        q_pe = torch.randn(3, 2, 1)
        local_kv_local = torch.randn(2, 3)
        local_kv = torch.randn(4, 3)
        k_nope_local = torch.randn(3, 2)
        k_pe_local = torch.randn(3, 1)
        global_output = torch.randn(3, 6)
        local_latent = torch.randn(3, 2, 2)
        local_output = torch.randn(3, 6)
        gated_global = torch.randn(3, 6)
        gated_local = torch.randn(3, 6)
        gate_global_local = torch.randn(2, 4)
        gate_local_local = torch.randn(2, 4)
        gate_local = torch.cat((gate_global_local, gate_local_local), dim=-1)
        gate_global = torch.randn(4, 4)
        gate_local_global = torch.randn(4, 4)
        gate = torch.cat((gate_global, gate_local_global), dim=-1)
        projected_local = torch.cat((q_lora_local, local_kv_local, gate_local), dim=-1)
        projected = torch.cat((q_lora, local_kv, gate), dim=-1)
        projected_output = torch.randn(2, 5)
        global_cache = (torch.randn(1), torch.randn(1))
        local_cache = (torch.randn(1), torch.randn(1))

        attention.attn = SimpleNamespace(kv_cache=global_cache)
        attention.attn_cla_swa = SimpleNamespace(kv_cache=local_cache)
        attention.q_a_proj = Mock(return_value=q_lora_local)
        attention.q_a_layernorm = Mock(side_effect=lambda value: value)
        attention.q_b_proj = Mock(return_value=q.flatten(1))
        attention.kv_a_proj_with_mqa_swa = Mock(return_value=local_kv_local)
        attention.swa_gate_global = SimpleNamespace(
            w_gate=Mock(return_value=gate_global_local),
            _apply_gate=Mock(return_value=gated_global),
        )
        attention.swa_gate_local = SimpleNamespace(
            w_gate=Mock(return_value=gate_local_local),
            _apply_gate=Mock(return_value=gated_local),
        )
        attention._apply_o_proj = Mock(return_value=projected_output)

        attention.sharded_o_proj = True
        attention.o_proj = SimpleNamespace(prefetch=Mock())
        prefetch_stream = Mock()

        tp_group = SimpleNamespace(all_gather=Mock(return_value=projected), device_group=object())
        global_metadata = SimpleNamespace(
            num_actual_tokens=3,
            num_decode_tokens=0,
            prefill=SimpleNamespace(chunked_context=None),
        )
        local_metadata = SimpleNamespace(num_actual_tokens=3, prefill=SimpleNamespace(chunked_context=None))

        def copy_all_to_all(output, value, **_kwargs):
            output.copy_(value)

        with (
            patch.object(torch.npu, "current_stream", return_value=prefetch_stream),
            patch.object(torch.npu, "stream", return_value=nullcontext()),
            patch.object(pangu_mod, "named_stream", return_value=prefetch_stream),
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(virtual_engine=0),
            ),
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_tp_group",
                return_value=tp_group,
            ),
            patch("torch.distributed.all_to_all_single", side_effect=copy_all_to_all) as mock_all_to_all,
            patch.object(attention, "_q_rope", return_value=q_pe),
            patch.object(
                attention,
                "_w_uk_t_absorb",
                side_effect=lambda _q, weight=None: q_nope_global if weight is None else q_nope_local,
            ),
            patch.object(attention, "_use_swa_prefill_pa", return_value=True),
            patch(
                "torch.ops.vllm.npu_pangu_cla_swa_kv_cache_update",
                return_value=(k_nope_local, k_pe_local, *local_cache),
            ) as mock_cache_update,
            patch.object(
                attention, "_apply_SWA_attention_prefill_absorb", return_value=global_output,
            ) as mock_global_attention,
            patch.object(
                attention, "_apply_swa_pa_attention", return_value=local_latent,
            ) as mock_local_attention,
            patch("torch_npu.npu_transpose_batchmatmul", return_value=local_output) as mock_v_up,
        ):
            output = attention._forward_cla_prefill_FC2(hidden_states, cos, cos, global_metadata, local_metadata)

        self.assertIs(output, projected_output)
        tp_group.all_gather.assert_called_once()
        torch.testing.assert_close(tp_group.all_gather.call_args.args[0], projected_local)
        torch.testing.assert_close(attention.q_b_proj.call_args.args[0], q_lora[:3])
        torch.testing.assert_close(mock_cache_update.call_args.args[0], local_kv[:3])
        mock_global_attention.assert_called_once()
        mock_v_up.assert_called_once_with(
            local_latent, attention.W_UV_cla_swa, perm_x1=(1, 0, 2), perm_y=(1, 0, 2),
        )
        self.assertIs(mock_local_attention.call_args.args[2][0], local_cache[0])
        self.assertIs(mock_local_attention.call_args.args[2][1], local_cache[1])
        expected_gate = torch.sigmoid(gate[:3])
        torch.testing.assert_close(attention.swa_gate_global._apply_gate.call_args.args[0], expected_gate[:, :4])
        torch.testing.assert_close(attention.swa_gate_local._apply_gate.call_args.args[0], expected_gate[:, 4:])
        mock_all_to_all.assert_called_once()
        self.assertEqual(attention._apply_o_proj.call_args.args[0].shape, (2, 12))
        attention.o_proj.prefetch.assert_called_once_with(prefetch_stream)

    def test_resolves_converted_checkpoint_mapping(self):
        config = SimpleNamespace(num_hidden_layers=32, cla_explicit_mapping=[[7, 3], [15, 11], [23, 19], [31, 27]])
        self.assertEqual(NPUPanguSparseAttention._get_cla_mapping(config), {7: 3, 15: 11, 23: 19, 31: 27})

    def test_rejects_invalid_cla_mapping(self):
        cases = (
            ([[3]], "pairs"),
            ([[3, 1], [3, 2]], "Duplicate CLA reuse layer"),
            ([[3, 3]], "must precede"),
            ([[8, 1]], "must be in"),
        )
        for raw_mapping, message in cases:
            with self.subTest(raw_mapping=raw_mapping):
                config = SimpleNamespace(num_hidden_layers=8, cla_explicit_mapping=raw_mapping)
                with self.assertRaisesRegex(ValueError, message):
                    NPUPanguSparseAttention._get_cla_mapping(config)

    def test_cla_mla_routes_global_prefill_and_decode_to_mla_attention(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.is_dsa_layer = False
        attention.use_gpt_oss_sink = False
        attention.gpt_oss_sink_swa = None
        attention.cla_swa_gate_window = 512
        attention.is_cla_fa_metadata_producer = True
        attention.attn_cla_swa = object()
        attention.num_local_heads = 2
        attention.v_head_dim = 4
        attention.W_UV = torch.randn(2, 3, 4)
        attention.W_UV_cla_swa = torch.randn(2, 3, 4)
        attention.swa_gate_global = lambda _hidden_states, output: output
        attention.swa_gate_local = lambda _hidden_states, output: output

        hidden_states = torch.randn(2, 8)
        q_nope = torch.randn(2, 2, 3)
        q_nope_global = torch.randn(2, 2, 4)
        q_pe = torch.randn(2, 2, 2)
        global_kv_cache = (torch.randn(1), torch.randn(1))
        local_kv_cache = (torch.randn(1), torch.randn(1))
        global_latent = torch.randn(2, 2, 3)
        local_latent = torch.randn(2, 2, 3)
        global_output = torch.einsum("tnl,nlv->tnv", global_latent, attention.W_UV).reshape(2, 8)
        local_output = torch.einsum("tnl,nlv->tnv", local_latent, attention.W_UV_cla_swa).reshape(2, 8)
        local_metadata = SimpleNamespace(prefill=SimpleNamespace(), decode=None, num_actual_tokens=2)
        global_metadata = SimpleNamespace(prefill=SimpleNamespace(), decode=None)

        with (
            patch.object(
                attention, "_prepare_cla_queries", return_value=(q_nope, q_nope_global, q_pe),
            ),
            patch.object(
                attention,
                "_prepare_cla_kv",
                return_value=(global_kv_cache, local_kv_cache, (object(), object())),
            ),
            patch.object(
                attention,
                "_prepare_cla_swa_inputs",
                return_value=(q_nope, local_kv_cache),
            ),
            patch.object(attention, "_get_topk_indices") as mock_topk,
            patch.object(attention, "_apply_DSA_attention") as mock_dsa,
            patch.object(
                attention, "_apply_SWA_attention_prefill_absorb", return_value=global_output,
            ) as mock_prefill,
            patch.object(
                attention, "_apply_SWA_attention_decode", return_value=global_latent,
            ) as mock_decode,
            patch.object(attention, "_use_swa_prefill_pa", side_effect=lambda metadata: metadata is local_metadata),
            patch.object(attention, "_apply_swa_pa_attention", return_value=local_latent),
            patch(
                "torch_npu.npu_transpose_batchmatmul",
                side_effect=lambda value, weight, **_kwargs: (value.transpose(0, 1) @ weight).transpose(0, 1),
            ) as mock_v_up,
            patch.object(attention, "_mla_epilog", side_effect=lambda output, *_args: output),
        ):
            prefill_output = attention._forward_cla(
                hidden_states, torch.empty(0), torch.empty(0), global_metadata, local_metadata, None,
            )
            global_metadata.prefill = None
            global_metadata.decode = SimpleNamespace()
            local_metadata.prefill = None
            local_metadata.decode = SimpleNamespace()
            decode_output = attention._forward_cla(
                hidden_states, torch.empty(0), torch.empty(0), global_metadata, local_metadata, None,
            )

        torch.testing.assert_close(prefill_output, global_output + local_output)
        torch.testing.assert_close(decode_output, global_output + local_output)
        self.assertEqual(mock_v_up.call_count, 3)
        self.assertIs(mock_v_up.call_args_list[0].args[1], attention.W_UV_cla_swa)
        self.assertIs(mock_v_up.call_args_list[1].args[1], attention.W_UV)
        self.assertIs(mock_v_up.call_args_list[2].args[1], attention.W_UV_cla_swa)
        mock_prefill.assert_called_once_with(
            q_nope_global,
            q_pe,
            global_kv_cache,
            attn_metadata=global_metadata,
            recompute_metadata=True,
        )
        mock_decode.assert_called_once_with(q_nope_global, q_pe, global_kv_cache, attn_metadata=global_metadata)
        mock_topk.assert_not_called()
        mock_dsa.assert_not_called()

    def test_per_head_sigmoid_gate_selects_local_heads_and_broadcasts(self):
        class FixedProjection(torch.nn.Module):
            def forward(self, hidden_states):
                return hidden_states

        projection = FixedProjection()
        with patch(
            "omni_npu.v1.layers.attention.npu_pangu.ColumnParallelFlashCommLinear",
            return_value=projection,
        ) as mock_linear:
            gate = PanguAttentionOutputGate(6, 4, 2, None, "gate", True)
        gate_input = torch.tensor([[1.0, -1.0, 0.0, 2.0], [3.0, 1.0, -2.0, 0.0]])
        attention_output = torch.ones(2, 6)

        with patch(
            "omni_npu.v1.layers.attention.npu_pangu.get_tp_group",
            return_value=SimpleNamespace(rank_in_group=1),
        ):
            output = gate(gate_input, attention_output).view(2, 2, 3)
        expected = torch.sigmoid(gate_input[:, 2:]).unsqueeze(-1).expand_as(output)
        torch.testing.assert_close(output, expected)
        gate.num_local_heads = 4
        manager = SimpleNamespace(sp_to_cp=Mock(side_effect=lambda value: value.flip(0)))
        cp_output = gate.forward_cp(gate_input, torch.ones(2, 12), manager)
        cp_expected = torch.sigmoid(gate_input.flip(0)).unsqueeze(-1).expand(2, 4, 3)
        torch.testing.assert_close(cp_output.view(2, 4, 3), cp_expected)
        manager.sp_to_cp.assert_called_once_with(gate_input)
        mock_linear.assert_called_once_with(
            6, 4, bias=False, quant_config=None, prefix="gate.w_gate", return_bias=False, disable_tp=True,
        )

    def test_prepare_cla_swa_kv_uses_noncontiguous_scatter(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.on_ascend950 = False
        attention.kv_lora_rank = 4
        attention.qk_rope_head_dim = 2
        attention.rope_interleave = True
        attention.kv_a_layernorm_swa = lambda value: value
        kv = torch.randn(3, 6)
        cos = torch.randn(3, 2)
        sin = torch.randn(3, 2)
        kv_cache = (torch.zeros(2, 16, 4), torch.zeros(2, 16, 2))
        slot_mapping_2d = torch.tensor([[0, 0], [0, 1], [0, 2]], dtype=torch.int32)
        metadata = SimpleNamespace(slot_mapping=torch.arange(3), slot_mapping_2d=slot_mapping_2d)

        with (
            patch("torch_npu.npu_rotary_mul", side_effect=lambda value, *_args, **_kwargs: value),
            patch("torch.ops.custom.npu_ai_infra_scatter_block_update_") as mock_scatter,
            patch("torch_npu.npu_scatter_nd_update_") as mock_scatter_nd,
        ):
            k_nope, k_pe = attention._prepare_cla_swa_kv(kv, cos, sin, kv_cache, metadata)

        torch.testing.assert_close(k_nope, kv[:, :4])
        torch.testing.assert_close(k_pe, kv[:, 4:])
        self.assertEqual(mock_scatter.call_count, 2)
        self.assertIs(mock_scatter.call_args_list[0].args[0], kv_cache[0])
        self.assertIs(mock_scatter.call_args_list[1].args[0], kv_cache[1])
        self.assertIs(mock_scatter.call_args_list[0].args[1], slot_mapping_2d)
        self.assertEqual(mock_scatter.call_args_list[0].args[2].shape, (3, 4))
        self.assertEqual(mock_scatter.call_args_list[1].args[2].shape, (3, 2))
        mock_scatter_nd.assert_not_called()

    def test_cla_cp_swa_uses_local_cache_block_table(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.use_gpt_oss_sink = False
        attention.gpt_oss_sink_swa = None
        attention.cla_swa_gate_window = 512
        attention.is_cla_fa_metadata_producer = True
        attention.attn_cla_swa = SimpleNamespace(layer_name="cla_swa.7")
        q_nope = torch.randn(6, 2, 4)
        q_pe = torch.randn(6, 2, 2)
        kv_cache = (torch.randn(4, 16, 4), torch.randn(4, 16, 2))
        local_block_table = torch.tensor([[3, 7], [5, 9]], dtype=torch.int32)
        local_metadata = SimpleNamespace(prefill=SimpleNamespace(block_table=local_block_table))
        query_cumlens = torch.tensor([2, 4, 5, 6], dtype=torch.int32)
        seq_lens = torch.tensor([2, 8, 3, 9], dtype=torch.int32)
        global_block_table = torch.tensor([[1, 2], [1, 2], [4, 6], [4, 6]], dtype=torch.int32)
        sp_manager = SimpleNamespace(
            cp_attn_meta=Mock(return_value=(query_cumlens, seq_lens, None, global_block_table)),
        )
        output = torch.randn(6, 2, 4)

        with patch.object(attention, "_apply_swa_pa_attention", return_value=output) as mock_attention:
            result = attention._apply_cla_swa_attention_cp(q_nope, q_pe, kv_cache, local_metadata, sp_manager)

        self.assertIs(result, output)
        mock_attention.assert_called_once()
        args, kwargs = mock_attention.call_args
        self.assertIs(args[0], q_nope)
        self.assertIs(args[1], q_pe)
        self.assertIs(args[2], kv_cache)
        self.assertIs(args[3], local_metadata.prefill)
        self.assertIs(kwargs["query_cumlens"], query_cumlens)
        self.assertIs(kwargs["seq_lens"], seq_lens)
        torch.testing.assert_close(
            kwargs["block_table"], torch.tensor([[3, 7], [3, 7], [5, 9], [5, 9]], dtype=torch.int32)
        )
        self.assertEqual(kwargs["num_tokens"], 6)
        self.assertEqual(kwargs["num_actual_tokens"], 6)
        self.assertEqual(kwargs["metadata_caller"], "cla_prefill_absorb_cp")
        self.assertEqual(kwargs["sliding_window"], 512)
        self.assertTrue(kwargs["recompute_metadata"])

    def test_cla_prefill_cp_updates_local_cache_and_combines_both_branches(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.num_local_heads = 2
        attention.v_head_dim = 4
        attention.qk_head_dim = 5
        attention.qk_nope_head_dim = 3
        attention.qk_rope_head_dim = 2
        attention.W_UK_T = object()
        attention.W_UK_T_cla_swa = object()
        attention.W_UV_cla_swa = object()
        attention.layer_name = "model.layers.0.self_attn"
        attention.first_chunk_pa = True
        hidden_states = torch.randn(4, 6)
        cos = torch.randn(4, 2)
        q_lora = torch.randn(4, 4)
        q = torch.randn(4, 2, 5)
        q_nope_global = torch.randn(4, 2, 4)
        q_nope_cla_swa = torch.randn(4, 2, 4)
        q_pe = torch.randn(4, 2, 2)
        local_kv_sp = torch.randn(4, 6)
        local_kv = torch.randn(4, 6)
        global_output = torch.randn(4, 8)
        local_latent = torch.randn(4, 2, 4)
        local_output = torch.randn(4, 8)
        projected_output = torch.randn(4, 6)
        global_cache = (torch.randn(1), torch.randn(1))
        local_cache = (torch.randn(1), torch.randn(1))
        attention.attn = SimpleNamespace(kv_cache=global_cache)
        attention.attn_cla_swa = SimpleNamespace(kv_cache=local_cache)
        attention.q_a_proj = Mock(return_value=q_lora)
        attention.q_a_layernorm = Mock(side_effect=lambda value: value)
        attention.q_b_proj = Mock(return_value=q)
        attention.kv_a_proj_with_mqa_swa = Mock(return_value=local_kv_sp)
        attention._apply_o_proj = Mock(return_value=projected_output)
        attention.swa_gate_global = SimpleNamespace(forward_cp=Mock(return_value=global_output))
        attention.swa_gate_local = SimpleNamespace(forward_cp=Mock(return_value=local_output))
        sp_manager = SimpleNamespace(
            cp_slice=Mock(side_effect=lambda value, **_kwargs: value),
            sp_to_cp=Mock(side_effect=lambda value: value),
            ag_tokens=Mock(return_value=local_kv),
            cp_to_sp=Mock(side_effect=lambda value: value),
        )
        cp_topk = torch.ones(4, 1, dtype=torch.int32)
        global_metadata = SimpleNamespace(
            prefill=SimpleNamespace(sp_manager=sp_manager, topk_indices_buffer=cp_topk),
            decode=None,
            num_actual_tokens=4,
        )
        local_metadata = SimpleNamespace(prefill=SimpleNamespace())

        with (
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(virtual_engine=0),
            ),
            patch.object(attention, "_w_uk_t_absorb", side_effect=[q_nope_global, q_nope_cla_swa]),
            patch.object(attention, "_q_rope", return_value=q_pe),
            patch(
                "torch.ops.vllm.npu_pangu_cla_swa_kv_cache_update",
                return_value=(torch.randn(4, 4), torch.randn(4, 2), *local_cache),
            ) as mock_update_cache,
            patch.object(attention, "_apply_DSA_attention_cp", return_value=global_output) as mock_global_attention,
            patch.object(attention, "_apply_cla_swa_attention_cp", return_value=local_latent),
            patch("torch_npu.npu_transpose_batchmatmul", return_value=local_output) as mock_v_up,
        ):
            output = attention._forward_cla_prefill_cp(hidden_states, cos, cos, global_metadata, local_metadata)

        self.assertIs(output, projected_output)
        self.assertIs(mock_global_attention.call_args.args[3], cp_topk)
        mock_v_up.assert_called_once_with(
            local_latent, attention.W_UV_cla_swa, perm_x1=(1, 0, 2), perm_y=(1, 0, 2),
        )
        attention.kv_a_proj_with_mqa_swa.assert_called_once_with(hidden_states)
        self.assertIs(sp_manager.ag_tokens.call_args.args[0], local_kv_sp)
        self.assertEqual(mock_update_cache.call_count, 1)
        update_args = mock_update_cache.call_args.args
        self.assertIs(update_args[0], local_kv)
        torch.testing.assert_close(update_args[1], cos)
        torch.testing.assert_close(update_args[2], cos)
        self.assertIs(update_args[3], local_cache[0])
        self.assertIs(update_args[4], local_cache[1])
        self.assertEqual(update_args[5], attention.layer_name)
        attention.swa_gate_global.forward_cp.assert_called_once_with(hidden_states, global_output, sp_manager)
        attention.swa_gate_local.forward_cp.assert_called_once()
        local_gate_args = attention.swa_gate_local.forward_cp.call_args.args
        self.assertIs(local_gate_args[0], hidden_states)
        torch.testing.assert_close(local_gate_args[1], local_output)
        self.assertIs(local_gate_args[2], sp_manager)
        torch.testing.assert_close(attention._apply_o_proj.call_args.args[0], global_output + local_output)

    def test_cla_cp_first_prefill_uses_direct_swa_attention(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        torch.nn.Module.__init__(attention)
        attention.num_local_heads = 2
        attention.qk_head_dim = 5
        attention.qk_nope_head_dim = 3
        attention.qk_rope_head_dim = 2
        attention.v_head_dim = 4
        attention.W_UK_T = object()
        attention.W_UK_T_cla_swa = object()
        attention.W_UV_cla_swa = object()
        attention.layer_name = "model.layers.0.self_attn"
        attention.first_chunk_pa = False
        attention.use_gpt_oss_sink = False
        attention.gpt_oss_sink_swa = None
        attention.cla_swa_gate_window = 512
        attention.is_cla_fa_metadata_producer = True

        hidden_states = torch.randn(2, 6)
        cos = torch.randn(4, 2)
        sin = torch.randn(4, 2)
        cp_cos = torch.randn(2, 2)
        cp_sin = torch.randn(2, 2)
        q_lora_sp = torch.randn(2, 4)
        q_lora = torch.randn(4, 4)
        q = torch.randn(4, 2, 5)
        q_nope_global = torch.randn(2, 2, 4)
        q_pe_cla_swa = torch.randn(4, 2, 2)
        q_pe_global = torch.randn(2, 2, 2)
        local_kv_sp = torch.randn(2, 6)
        local_kv = torch.randn(4, 6)
        k_nope_cla_swa = torch.randn(4, 4)
        k_pe_cla_swa = torch.randn(4, 2)
        kv_up_cla_swa = torch.randn(4, 2, 7)
        global_output = torch.randn(2, 8)
        local_output_full = torch.randn(4, 8)
        local_output = torch.randn(2, 8)
        projected_output = torch.randn(2, 6)
        global_cache = (torch.randn(1), torch.randn(1))
        local_cache = (torch.randn(1), torch.randn(1))
        attention.attn = SimpleNamespace(kv_cache=global_cache)
        attention.attn_cla_swa = SimpleNamespace(kv_cache=local_cache)
        attention.q_a_proj = Mock(return_value=q_lora_sp)
        attention.q_a_layernorm = Mock(side_effect=lambda value: value)
        attention.q_b_proj = Mock(return_value=q)
        attention.kv_a_proj_with_mqa_swa = Mock(return_value=local_kv_sp)
        attention.kv_b_proj_swa = Mock(return_value=kv_up_cla_swa)
        attention._apply_o_proj = Mock(return_value=projected_output)
        attention.swa_gate_global = SimpleNamespace(forward_cp=Mock(return_value=global_output))
        attention.swa_gate_local = SimpleNamespace(forward_cp=Mock(return_value=local_output))
        cp_slices = iter([cp_cos, cp_sin, q[:2, :, :3], q_pe_global, local_output])
        sp_manager = SimpleNamespace(
            cp_slice=Mock(side_effect=lambda *_args, **_kwargs: next(cp_slices)),
            sp_to_cp=Mock(),
            ag_tokens=Mock(side_effect=[q_lora, local_kv]),
            cp_to_sp=Mock(side_effect=lambda value: value),
        )
        topk_indices = torch.ones(2, 1, dtype=torch.int32)
        global_metadata = SimpleNamespace(
            prefill=SimpleNamespace(sp_manager=sp_manager, topk_indices_buffer=topk_indices),
            decode=None,
            num_actual_tokens=4,
        )
        local_metadata = SimpleNamespace(
            prefill=SimpleNamespace(chunked_context=None), decode=None, num_actual_tokens=4
        )

        with (
            patch(
                "omni_npu.v1.layers.attention.npu_pangu.get_forward_context",
                return_value=SimpleNamespace(virtual_engine=0),
            ),
            patch.object(attention, "_w_uk_t_absorb", return_value=q_nope_global),
            patch.object(attention, "_q_rope", return_value=q_pe_cla_swa),
            patch(
                "torch.ops.vllm.npu_pangu_cla_swa_kv_cache_update",
                return_value=(k_nope_cla_swa, k_pe_cla_swa, *local_cache),
            ),
            patch.object(attention, "_apply_DSA_attention_cp", return_value=global_output),
            patch.object(attention, "_apply_cla_swa_attention_cp") as mock_pa,
            patch.object(
                attention, "_apply_SWA_attention_prefill", return_value=local_output_full
            ) as mock_direct,
            patch("torch_npu.npu_transpose_batchmatmul") as mock_v_up,
        ):
            output = attention._forward_cla_prefill_cp(hidden_states, cos, sin, global_metadata, local_metadata)

        self.assertIs(output, projected_output)
        sp_manager.sp_to_cp.assert_not_called()
        self.assertIs(sp_manager.ag_tokens.call_args_list[0].args[0], q_lora_sp)
        self.assertIs(sp_manager.ag_tokens.call_args_list[1].args[0], local_kv_sp)
        mock_pa.assert_not_called()
        mock_v_up.assert_not_called()
        direct_args = mock_direct.call_args.args
        torch.testing.assert_close(direct_args[0], q[:, :, :3])
        self.assertIs(direct_args[1], q_pe_cla_swa)
        torch.testing.assert_close(direct_args[2], kv_up_cla_swa[:, :, :3])
        self.assertEqual(mock_direct.call_args.kwargs["metadata_caller"], "cla_prefill")
        self.assertIs(sp_manager.cp_slice.call_args_list[-1].args[0], local_output_full)

    def test_mixed_batch_slices_both_cla_metadata_objects(self):
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.on_ascend950 = False

        def make_metadata(offset):
            metadata = SimpleNamespace(
                num_decode_tokens=2,
                num_actual_tokens=5,
                slot_mapping=torch.arange(5) + offset,
                slot_mapping_2d=None,
                prefill=SimpleNamespace(),
                decode=SimpleNamespace(),
            )
            metadata.get_slot_mapping_2d = lambda: torch.stack(
                [metadata.slot_mapping // 16, metadata.slot_mapping % 16], dim=-1
            )
            return metadata

        global_metadata = make_metadata(0)
        local_metadata = make_metadata(100)
        hidden = torch.arange(15).view(5, 3)
        cos = torch.arange(10).view(5, 2)

        local_metadata.num_actual_tokens = 4
        with self.assertRaisesRegex(ValueError, "token counts differ"):
            attention._prepare_phase_inputs(hidden, cos, cos, global_metadata, "prefill", local_metadata)
        self.assertEqual(global_metadata.num_actual_tokens, 5)
        self.assertIsNotNone(global_metadata.decode)
        torch.testing.assert_close(global_metadata.slot_mapping, torch.arange(5))
        local_metadata.num_actual_tokens = 5

        attention._prepare_phase_inputs(
            hidden, cos, cos, global_metadata, "prefill",
            attn_metadata_cla_swa=local_metadata,
        )
        torch.testing.assert_close(local_metadata.slot_mapping, torch.arange(102, 105))
        torch.testing.assert_close(local_metadata.slot_mapping_2d, torch.tensor([[6, 6], [6, 7], [6, 8]]))
        attention._prepare_phase_inputs(
            hidden, cos, cos, global_metadata, "decode",
            attn_metadata_cla_swa=local_metadata,
        )
        torch.testing.assert_close(local_metadata.slot_mapping, torch.arange(100, 102))
        torch.testing.assert_close(local_metadata.slot_mapping_2d, torch.tensor([[6, 4], [6, 5]]))
        attention._restore_phase_metadata(global_metadata, local_metadata)
        torch.testing.assert_close(local_metadata.slot_mapping, torch.arange(100, 105))
        torch.testing.assert_close(
            local_metadata.slot_mapping_2d,
            torch.tensor([[6, 4], [6, 5], [6, 6], [6, 7], [6, 8]]),
        )


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

    def test_aicpu_tiling_records_current_stream_limits(self):
        stream = object()
        with (
            patch.object(pangu_mod.model_extra_config.operator_opt_config, "use_aicpu_fa_tiling", True),
            patch.object(torch.npu, "current_stream", return_value=stream),
            patch.object(torch.npu, "get_stream_limit", return_value={"cube_core_num": 12, "vector_core_num": 24})
            as limits,
        ):
            attention = _build_sparse_attention(layer_idx=0)
        limits.assert_called_once_with(stream)
        self.assertEqual((attention._fia_aic_core_num, attention._fia_aiv_core_num), (12, 24))

    def test_shared_ops_register_swa_and_mla_callers(self):
        created = []

        class FakeSharedOp:
            def __init__(self, **kwargs):
                created.append(kwargs)

        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.on_ascend950 = True
        attention.hf_config = SimpleNamespace()
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

    def test_page_size_dsa_covers_every_cache_dtype(self):
        """Pin the per-page byte layout of each DSA cache format."""
        attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
        attention.kv_lora_rank = 8
        attention.qk_rope_head_dim = 4
        attention.use_mome = False
        attention.block_size_c8 = 8
        cache_config = SimpleNamespace(block_size=16)
        config = SimpleNamespace(index_topk=2048, index_head_dim=8)

        cases = {
            # 788 bytes per token: 512 fp8, 64 bf16, 4 fp32, 128 int8, 1 fp32
            "fp8_ds_mla": 8 * 788,
            "hif8_ds_mla": 8 * 788,
            # 786 bytes per token: 512 int8, 64 bf16, 4 fp32, 128 int8, 1 bf16
            "int8_ds_mla": 8 * 786,
            # 1282 bytes per token: 576 bf16, 128 int8, 1 bf16
            "li_int8_ds_mla": 16 * 1282,
            # Non-quant: (kv_lora_rank + qk_rope_head_dim + index_head_dim) bf16
            "auto": 16 * (8 + 4 + 8) * 2,
        }
        # Bound by name: the helper is private to the attention class.
        calculate = getattr(attention, "_calculate_page_size_padded")
        for cache_dtype_str, expected in cases.items():
            with self.subTest(cache_dtype=cache_dtype_str):
                page = calculate(cache_config, cache_dtype_str, config)
                self.assertEqual(page, expected)


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

    def test_dispatch_combine_force_load_balance_overrides_topk_ids(self):
        """use_moe_force_load_balance replaces gating ids with round-robin ids."""
        moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
        moe.ep_comm_name = "ep"
        moe.side_stream = None
        moe.gate = MagicMock(return_value=(torch.zeros(4, 2), None))
        moe.experts = SimpleNamespace(top_k=1, topk_group=1, num_expert_group=1)
        moe.e_score_correction_bias = None
        moe.routed_scaling_factor = 1.0
        moe.n_routed_experts = 2
        moe.use_moe_force_load_balance = True
        moe.aux_load_balance_tensor = torch.arange(2, dtype=torch.int32).unsqueeze(0)
        moe.enable_eplb = False
        moe._is_quant = False
        moe.moe_dispatch_combine_max_batch_size = 8
        moe.shared_experts = MagicMock(return_value=torch.ones(4, 3))

        seen = {}

        def fake_single(hidden, topk_weights, topk_ids, *_args, **_kwargs):
            seen["topk_ids"] = topk_ids
            return torch.zeros(hidden.shape[0], 3)

        moe._dispatch_combine_single_batch = fake_single
        moe._get_mc2_mask = MagicMock(return_value=None)
        with patch.object(
            model_mod.torch_npu,
            "npu_moe_gating_top_k",
            return_value=(
                torch.ones(4, 1),
                torch.full((4, 1), 9, dtype=torch.int32),
                None,
            ),
        ):
            out = moe._forward_dispatch_combine(torch.zeros(4, 3))

        ids = seen["topk_ids"]
        self.assertEqual(tuple(ids.shape), (4, 1))
        # The gating stub returned expert 9 everywhere; the override wins.
        self.assertTrue(torch.equal(ids, torch.tensor([[0], [1], [0], [1]], dtype=torch.int32)))
        self.assertEqual(tuple(out.shape), (4, 3))


class TestOpenPanguV2LoadWeightsSkip(unittest.TestCase):
    def test_unknown_weight_is_reported_through_the_logger(self):
        """A checkpoint key with no matching param must warn, not print."""
        model = model_mod.OpenPanguV2ForCausalLM.__new__(
            model_mod.OpenPanguV2ForCausalLM
        )
        model.model = SimpleNamespace()
        model.config = SimpleNamespace(n_routed_experts=2, tie_word_embeddings=False)
        model.num_redundant_experts = 0
        model.named_parameters = lambda: []

        with (
            patch.object(
                model_mod, "fused_moe_make_expert_params_mapping", return_value=[]
            ),
            patch.object(
                model_mod, "get_spec_layer_idx_from_weight_name", return_value=None
            ),
            patch.object(model_mod, "is_pp_missing_parameter", return_value=False),
            patch.object(
                model_mod, "maybe_remap_kv_scale_name", side_effect=lambda name, _: name
            ),
            patch.object(model_mod, "high_throughout", return_value=False),
            patch.object(model_mod, "logger") as mock_logger,
        ):
            loaded = model.load_weights(
                [("model.layers.0.nowhere.weight", torch.zeros(1))]
            )

        self.assertEqual(loaded, set())
        mock_logger.warning.assert_called_once_with(
            "Skip loading model.layers.0.nowhere.weight."
        )


class TestOpenPanguV2MtpSharedWeight(unittest.TestCase):
    """set_shared_weight rebinds real submodules, so _modules is the thing to check."""

    @staticmethod
    def _bare_mtp():
        mtp = mtp_mod.OpenPanguV2MTP.__new__(mtp_mod.OpenPanguV2MTP)
        torch.nn.Module.__init__(mtp)
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(4, 2)
        model.norm = torch.nn.LayerNorm(2)
        layer = torch.nn.Module()
        layer.shared_head = torch.nn.Module()
        layer.shared_head.head = torch.nn.Linear(2, 4)
        model.layers = {0: layer}
        mtp.model = model
        return mtp

    def test_embed_tokens_is_rebound_without_leaving_a_stale_module(self):
        mtp = self._bare_mtp()
        stale = mtp.model.embed_tokens
        order_before = [name for name, _ in mtp.model.named_parameters()]
        target = SimpleNamespace(embed_tokens=torch.nn.Embedding(4, 2))

        mtp.set_shared_weight(target)

        self.assertIs(mtp.model.embed_tokens, target.embed_tokens)
        children = dict(mtp.model.named_children())
        self.assertIs(children["embed_tokens"], target.embed_tokens)
        self.assertNotIn(stale, list(mtp.model.modules()))
        shared = dict(mtp.model.named_parameters())["embed_tokens.weight"]
        self.assertIs(shared, target.embed_tokens.weight)
        # Plain assignment keeps registration order; del + reassign would not.
        self.assertEqual([name for name, _ in mtp.model.named_parameters()], order_before)

    def test_lm_head_is_rebound_on_every_layer(self):
        mtp = self._bare_mtp()
        head = mtp.model.layers[0].shared_head
        stale = head.head
        target = SimpleNamespace(lm_head=torch.nn.Linear(2, 4))

        mtp.set_shared_weight(target)

        self.assertIs(head.head, target.lm_head)
        self.assertIs(dict(head.named_children())["head"], target.lm_head)
        self.assertNotIn(stale, list(head.modules()))

    def test_target_without_shared_attrs_leaves_the_model_untouched(self):
        mtp = self._bare_mtp()
        embed = mtp.model.embed_tokens
        head = mtp.model.layers[0].shared_head.head

        mtp.set_shared_weight(SimpleNamespace())

        self.assertIs(mtp.model.embed_tokens, embed)
        self.assertIs(mtp.model.layers[0].shared_head.head, head)


def _bare_swa_attention(**attrs):
    """Build an uninitialized NPUPanguSparseAttention with SWA SP defaults."""
    attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
    defaults = dict(
        on_ascend950=False,
        _fa_meta_suffix="",
        is_fa_metadata_producer=True,
        use_gpt_oss_sink=False,
        param_sink_number=0,
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
            kv_cache=(torch.zeros(1), torch.zeros(1)),
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
        causal=True,
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


class TestPanguDSADecodeCustomOp(unittest.TestCase):
    def test_real_op_forwards_live_layer_and_metadata(self):
        """The real wrapper must resolve and call the live DSA layer."""
        q_nope = torch.zeros(3, 2, 4)
        q_pe = torch.zeros(3, 2, 4)
        kv_cache = torch.zeros(2, 8)
        topk_indices = torch.zeros(3, 1, 2, dtype=torch.int32)
        attn_metadata = object()
        expected = torch.ones(3, 2, 8)
        layer = SimpleNamespace(
            prefix="model.layers.0.self_attn",
            _apply_DSA_attention=MagicMock(return_value=expected),
        )
        context = SimpleNamespace(
            no_compile_layers={"layer": layer},
            attn_metadata=attn_metadata,
        )

        with patch.object(custom_ops_mod, "get_forward_context", return_value=context):
            result = custom_ops_mod.npu_pangu_dsa_decode(
                q_nope, q_pe, kv_cache, topk_indices, "layer"
            )

        self.assertIs(result, expected)
        layer._apply_DSA_attention.assert_called_once_with(
            q_nope=q_nope,
            q_pe=q_pe,
            kv_cache=(kv_cache,),
            topk_indices=topk_indices,
            attn_metadata=attn_metadata,
        )

    def test_fake_op_preserves_dynamic_token_dimension(self):
        """The fake output must remain symbolic across capture gears."""
        layer = SimpleNamespace(num_local_heads=2, kv_lora_rank=8)
        kv_cache = torch.zeros(2, 8)
        topk_indices = torch.zeros(1, 1, 2, dtype=torch.int32)

        with patch.object(
            custom_ops_mod,
            "_lookup_layer_and_attn_metadata",
            return_value=(layer, None),
        ):
            for num_tokens in (4, 16):
                q_nope = torch.zeros(num_tokens, 2, 4)
                q_pe = torch.zeros(num_tokens, 2, 4)
                result = custom_ops_mod.npu_pangu_dsa_decode_fake(
                    q_nope, q_pe, kv_cache, topk_indices, "layer"
                )
                self.assertEqual(result.shape, (num_tokens, 2, 8))
                self.assertEqual(result.dtype, q_nope.dtype)
                self.assertEqual(result.device, q_nope.device)

    def test_forward_decode_dispatches_dsa_through_custom_op(self):
        """DSA decode must cross the opaque custom-op boundary."""
        attention = _bare_swa_attention(is_dsa_layer=True, layer_idx=3)
        attention.is_cla_reuse_layer = False
        q_nope = torch.zeros(4, 2, 4)
        q_pe = torch.zeros(4, 2, 4)
        kv_cache = (torch.zeros(2, 8), torch.zeros(2, 8))
        topk_indices = torch.zeros(4, 1, 2, dtype=torch.int32)
        latent = torch.ones(4, 2, 8)
        expected = torch.ones(4, 8)
        attention._mla_prolog = MagicMock(
            return_value=(q_nope, q_pe, kv_cache, topk_indices)
        )
        attention._mla_epilog = MagicMock(return_value=expected)
        attention.pre_epilog_callback = None
        attn_metadata = object()

        with patch.object(
            pangu_mod,
            "sk_scope",
            side_effect=lambda _name: nullcontext(),
        ), patch(
            "torch.ops.vllm.npu_pangu_dsa_decode",
            return_value=latent,
        ) as decode_op:
            result = attention._forward_decode(
                torch.zeros(4, 8),
                torch.zeros(4, 4),
                torch.zeros(4, 4),
                attn_metadata,
                None,
            )

        self.assertIs(result, expected)
        decode_op.assert_called_once_with(
            q_nope, q_pe, kv_cache[0], topk_indices, attention.prefix
        )
        attention._mla_epilog.assert_called_once_with(latent, attn_metadata, None)

    def test_decode_slices_tp_padding_by_metadata(self):
        """Decode must exclude TP padding without a tensor-size branch."""
        attention = _bare_swa_attention(
            is_attn_sp_layer=False,
            is_dsa_layer=True,
        )
        hidden = torch.zeros(4, 8)
        cos = torch.zeros(4, 4)
        sin = torch.zeros(4, 4)
        decoded = torch.ones(2, 8)
        attention._forward_decode = MagicMock(return_value=decoded)
        metadata = SimpleNamespace(
            num_actual_tokens=4,
            num_decode_tokens=2,
            num_decodes=2,
            num_prefills=0,
            prefill=None,
            decode=SimpleNamespace(),
        )
        context = SimpleNamespace(
            no_compile_layers={"layer": attention},
            attn_metadata=metadata,
        )

        with patch.object(pangu_mod, "get_forward_context", return_value=context):
            result = pangu_mod.npu_pangu_forward(hidden, cos, sin, "layer")

        self.assertIs(result, hidden)
        self.assertTrue(torch.equal(result[:2], decoded))
        self.assertTrue(torch.equal(result[2:], torch.zeros(2, 8)))
        attention._forward_decode.assert_called_once()
        call_args = attention._forward_decode.call_args.args
        self.assertTrue(torch.equal(call_args[0], hidden[:2]))
        self.assertTrue(torch.equal(call_args[1], cos[:2]))
        self.assertTrue(torch.equal(call_args[2], sin[:2]))
        self.assertIs(call_args[3], metadata)
        self.assertIsNone(call_args[4])


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

    def test_npu_pangu_forward_rejects_noncausal_prefill(self):
        attention = _bare_swa_attention()
        hidden = torch.zeros(4, 8)
        ctx = _prefill_sp_forward_ctx(attention)
        ctx.attn_metadata.causal = False
        with patch.object(pangu_mod, "get_forward_context", return_value=ctx):
            with self.assertRaisesRegex(
                NotImplementedError, "Non-causal MLA prefill is not supported"
            ):
                pangu_mod.npu_pangu_forward(
                    hidden,
                    torch.zeros(4, 2),
                    torch.zeros(4, 2),
                    "layer",
                )

    def test_npu_pangu_forward_keeps_dsa_prefill(self):
        attention = _bare_swa_attention(
            is_dsa_layer=True,
            is_attn_sp_layer=False,
        )
        hidden = torch.zeros(4, 8)
        ctx = _prefill_sp_forward_ctx(attention)
        del ctx.attn_metadata.causal
        attention._forward_prefill = MagicMock(return_value=hidden)
        with patch.object(pangu_mod, "get_forward_context", return_value=ctx):
            out_h = pangu_mod.npu_pangu_forward(
                hidden,
                torch.zeros(4, 2),
                torch.zeros(4, 2),
                "layer",
            )

        attention._forward_prefill.assert_called_once()
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
        with patch.object(pangu_mod, "get_forward_context", return_value=SimpleNamespace(capturing=False)), patch(
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
