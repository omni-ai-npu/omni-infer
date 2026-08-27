# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from omni_npu.v1.models.pangu import pangu_ultra_moe as ultra_mod
from vllm.config import CompilationMode


def _mhc_vllm_config(cfg):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg),
        quant_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )


@contextmanager
def _patch_open_pangu_model_init(pp_group, layers, start, end):
    with patch.object(
        ultra_mod, "get_pp_group", return_value=pp_group
    ), patch.object(
        ultra_mod,
        "NPUVocabParallelEmbedding",
        return_value=nn.Identity(),
    ), patch.object(
        ultra_mod,
        "make_layers",
        return_value=(start, end, layers),
    ), patch.object(
        ultra_mod,
        "make_empty_intermediate_tensors_factory",
        return_value=MagicMock(),
    ):
        yield


def _new_open_pangu_model_base():
    model = ultra_mod.OpenPanguModelBase.__new__(ultra_mod.OpenPanguModelBase)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=False)
    return model


def _load_weights_with_split_mark(model, loaded_params):
    loader = MagicMock()
    loader.load_weights.return_value = loaded_params
    with patch.object(
        ultra_mod,
        "AutoWeightsLoader",
        return_value=loader,
    ), patch.object(
        ultra_mod,
        "mark_split_q_up_params_loaded",
    ) as mark_split:
        result = model.load_weights([])
    return result, mark_split


class TestPanguUltraMoeDecoderLayer(unittest.TestCase):
    @staticmethod
    def _base_cfg(**kwargs):
        cfg = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            num_hidden_layers=2,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            kv_lora_rank=4,
            use_mhc=True,
            mhc_num_stream=2,
            first_k_dense_replace=99,
            max_position_embeddings=8192,
        )
        for key, value in kwargs.items():
            setattr(cfg, key, value)
        return cfg

    @staticmethod
    def _vllm_cfg(cfg):
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_config=cfg),
            cache_config=None,
            quant_config=None,
            parallel_config=SimpleNamespace(),
        )

    @patch.object(ultra_mod, "NPUmHCRL")
    @patch.object(ultra_mod, "OpenPanguMLP")
    @patch.object(ultra_mod, "RMSNorm")
    @patch.object(ultra_mod, "NPUDeepseekMLAAttention")
    @patch.object(ultra_mod, "get_tp_group")
    def test_init_use_mhc_enabled_when_not_mtp(
        self, mock_get_tp, mock_attn, mock_norm, mock_mlp, mock_mhc
    ):
        mock_get_tp.return_value = SimpleNamespace(device_group=None)
        mock_attn.return_value = nn.Identity()
        mock_norm.side_effect = lambda *args, **kwargs: nn.Identity()
        mock_mlp.return_value = nn.Identity()
        mock_mhc.return_value = nn.Identity()

        cfg = self._base_cfg(num_hidden_layers=2, use_mhc=True)
        layer = ultra_mod.OpenPanguDecoderLayer(
            config=cfg,
            prefix="model.layers.0",
            vllm_config=self._vllm_cfg(cfg),
        )

        self.assertFalse(layer.is_mtp_layer)
        self.assertTrue(layer.use_mhc)
        self.assertEqual(mock_mhc.call_count, 2)

    @patch.object(ultra_mod, "NPUmHCRL")
    @patch.object(ultra_mod, "OpenPanguMLP")
    @patch.object(ultra_mod, "RMSNorm")
    @patch.object(ultra_mod, "NPUDeepseekMLAAttention")
    @patch.object(ultra_mod, "get_tp_group")
    def test_init_use_mhc_disabled_for_mtp_layer(
        self, mock_get_tp, mock_attn, mock_norm, mock_mlp, mock_mhc
    ):
        mock_get_tp.return_value = SimpleNamespace(device_group=None)
        mock_attn.return_value = nn.Identity()
        mock_norm.side_effect = lambda *args, **kwargs: nn.Identity()
        mock_mlp.return_value = nn.Identity()
        mock_mhc.return_value = nn.Identity()

        cfg = self._base_cfg(num_hidden_layers=2, use_mhc=True)
        layer = ultra_mod.OpenPanguDecoderLayer(
            config=cfg,
            prefix="model.layers.2",
            vllm_config=self._vllm_cfg(cfg),
        )

        self.assertTrue(layer.is_mtp_layer)
        self.assertFalse(layer.use_mhc)
        mock_mhc.assert_not_called()


    def test_forward_mhc_fused_uses_custom_op_side_work(self):
        layer = ultra_mod.OpenPanguDecoderLayer.__new__(
            ultra_mod.OpenPanguDecoderLayer
        )
        nn.Module.__init__(layer)

        hidden_states = object()
        residual = object()
        h_post = object()
        h_res = object()
        resolved_attn_h_res = object()
        cos = object()
        sin = object()
        attn_output = object()
        mlp_input = object()
        mlp_residual = object()
        mlp_h_post = object()
        mlp_h_res = object()
        resolved_mlp_h_post = object()
        resolved_mlp_h_res = object()
        mlp_output = object()
        next_hidden = object()
        next_residual = object()
        next_h_post = object()
        next_h_res = object()
        post_attention_norm = object()
        pre_mlp_norm = object()
        post_mlp_norm = object()
        next_norm = object()

        layer.self_attn = MagicMock(return_value=attn_output)
        layer.mlp = MagicMock(return_value=mlp_output)
        layer.post_attention_layernorm = post_attention_norm
        layer.pre_mlp_layernorm = pre_mlp_norm
        layer.post_mlp_layernorm = post_mlp_norm
        layer.has_block_post_layernorm = False
        layer.attn_mhc_task_key = "attn_key"
        layer.mlp_mhc_task_key = "mlp_key"

        layer.attn_mhc_module = MagicMock()
        layer.attn_mhc_module.resolve_sinkhorn.return_value = resolved_attn_h_res
        layer.attn_mhc_module.mhc_sandwich_norm_post_preonly.return_value = (
            mlp_input,
            mlp_residual,
        )

        layer.mlp_mhc_module = MagicMock()
        layer.mlp_mhc_module.launch_fused_split_sinkhorn.return_value = (
            mlp_h_post,
            mlp_h_res,
        )
        layer.mlp_mhc_module.resolve_fused_split_sinkhorn.return_value = (
            resolved_mlp_h_post,
            resolved_mlp_h_res,
        )
        layer.mlp_mhc_module.mhc_sandwich_norm_post_preonly.return_value = (
            next_hidden,
            next_residual,
        )

        next_mhc = MagicMock()
        next_mhc.enable_mhc_multistream = False
        next_mhc.launch_fused_split_sinkhorn.return_value = (
            next_h_post,
            next_h_res,
        )
        layer._mhc_tail_refs = (next_mhc, next_norm, "next_key", False)

        result = layer.forward_mhc_fused(
            hidden_states,
            residual,
            h_post,
            h_res,
            cos,
            sin,
        )

        self.assertEqual(
            result,
            (next_hidden, next_residual, next_h_post, next_h_res),
        )
        layer.self_attn.assert_called_once_with(hidden_states, cos, sin)
        layer.attn_mhc_module.resolve_sinkhorn.assert_called_once_with(
            h_res, "attn_key"
        )
        layer.attn_mhc_module.mhc_sandwich_norm_post_preonly.assert_called_once_with(
            attn_output,
            residual,
            h_post,
            resolved_attn_h_res,
            post_attention_norm,
            layer.mlp_mhc_module,
            pre_mlp_norm,
            return_h_in_f32=False,
        )
        layer.mlp_mhc_module.launch_fused_split_sinkhorn.assert_called_once_with(
            mlp_residual, "mlp_key"
        )
        layer.mlp.assert_called_once_with(mlp_input)
        layer.mlp_mhc_module.resolve_fused_split_sinkhorn.assert_called_once_with(
            mlp_h_post, mlp_h_res, "mlp_key"
        )
        layer.mlp_mhc_module.mhc_sandwich_norm_post_preonly.assert_called_once_with(
            mlp_output,
            mlp_residual,
            resolved_mlp_h_post,
            resolved_mlp_h_res,
            post_mlp_norm,
            next_mhc,
            next_norm,
            None,
        )
        next_mhc.launch_fused_split_sinkhorn.assert_called_once_with(
            next_residual, "next_key"
        )

        layer.attn_mhc_module.resolve_sinkhorn.reset_mock()
        layer.attn_mhc_module.resolve_fused_split_sinkhorn.return_value = (
            h_post,
            resolved_attn_h_res,
        )
        layer.attn_mhc_module.mhc_sandwich_norm_post_preonly.reset_mock()

        layer.forward_mhc_fused(
            hidden_states,
            residual,
            h_post,
            h_res,
            cos,
            sin,
            h_res_from_fused_split=True,
        )

        layer.attn_mhc_module.resolve_sinkhorn.assert_not_called()
        layer.attn_mhc_module.resolve_fused_split_sinkhorn.assert_called_once_with(
            h_post, h_res, "attn_key"
        )
        sandwich_args = (
            layer.attn_mhc_module.mhc_sandwich_norm_post_preonly
            .call_args.args
        )
        self.assertIs(sandwich_args[2], h_post)
        self.assertIs(sandwich_args[3], resolved_attn_h_res)

        layer.self_attn.reset_mock()
        layer.self_attn.forward_mhc_deferred.return_value = (
            attn_output,
            h_post,
            h_res,
        )
        layer.attn_mhc_module.resolve_fused_split_sinkhorn.reset_mock()
        layer.attn_mhc_module.resolve_fused_split_sinkhorn.return_value = (
            h_post,
            resolved_attn_h_res,
        )

        layer.forward_mhc_fused(
            hidden_states,
            residual,
            None,
            None,
            cos,
            sin,
            h_res_from_fused_split=True,
        )

        layer.self_attn.assert_not_called()
        layer.self_attn.forward_mhc_deferred.assert_called_once_with(
            hidden_states,
            cos,
            sin,
            residual,
            layer.attn_mhc_module.prefix,
            "attn_key",
        )
        layer.attn_mhc_module.resolve_fused_split_sinkhorn.assert_called_once_with(
            h_post,
            h_res,
            "attn_key",
        )

    def test_forward_mhc_uses_registered_sinkhorn_results(self):
        layer = ultra_mod.OpenPanguDecoderLayer.__new__(
            ultra_mod.OpenPanguDecoderLayer
        )
        nn.Module.__init__(layer)
        hidden_states = object()
        cos = object()
        sin = object()
        attn_pre = object()
        attn_h_post = object()
        attn_h_res = object()
        attn_registered = object()
        attn_normalized = object()
        attn_output = object()
        attn_post_norm = object()
        attn_resolved = object()
        attn_post = object()
        mlp_pre = object()
        mlp_h_post = object()
        mlp_h_res = object()
        mlp_registered = object()
        mlp_normalized = object()
        mlp_output = object()
        mlp_post_norm = object()
        mlp_resolved = object()
        final_output = object()

        layer.sandwich_norm = True
        layer.has_block_post_layernorm = False
        layer.attn_mhc_task_key = "attn_key"
        layer.mlp_mhc_task_key = "mlp_key"
        layer.attn_mhc_module = MagicMock()
        layer.attn_mhc_module.mhc_pre.return_value = (
            attn_pre,
            attn_h_post,
            attn_h_res,
        )
        layer.attn_mhc_module.maybe_register_sinkhorn.return_value = (
            attn_registered
        )
        layer.attn_mhc_module.resolve_sinkhorn.return_value = attn_resolved
        layer.attn_mhc_module.mhc_post.return_value = attn_post
        layer.input_layernorm = MagicMock(return_value=attn_normalized)
        layer.self_attn = MagicMock(return_value=attn_output)
        layer.post_attention_layernorm = MagicMock(
            return_value=attn_post_norm
        )
        layer.mlp_mhc_module = MagicMock()
        layer.mlp_mhc_module.mhc_pre.return_value = (
            mlp_pre,
            mlp_h_post,
            mlp_h_res,
        )
        layer.mlp_mhc_module.maybe_register_sinkhorn.return_value = (
            mlp_registered
        )
        layer.mlp_mhc_module.resolve_sinkhorn.return_value = mlp_resolved
        layer.mlp_mhc_module.mhc_post.return_value = final_output
        layer.pre_mlp_layernorm = MagicMock(return_value=mlp_normalized)
        layer.mlp = MagicMock(return_value=mlp_output)
        layer.post_mlp_layernorm = MagicMock(return_value=mlp_post_norm)

        result = layer.forward_mhc(hidden_states, cos, sin, None)

        self.assertEqual(result, (final_output, None))
        layer.attn_mhc_module.maybe_register_sinkhorn.assert_called_once_with(
            attn_h_res, "attn_key"
        )
        layer.attn_mhc_module.resolve_sinkhorn.assert_called_once_with(
            attn_registered, "attn_key"
        )
        layer.attn_mhc_module.mhc_post.assert_called_once_with(
            attn_post_norm, attn_h_post, hidden_states, attn_resolved
        )
        layer.mlp_mhc_module.maybe_register_sinkhorn.assert_called_once_with(
            mlp_h_res, "mlp_key"
        )
        layer.mlp_mhc_module.resolve_sinkhorn.assert_called_once_with(
            mlp_registered, "mlp_key"
        )
        layer.mlp_mhc_module.mhc_post.assert_called_once_with(
            mlp_post_norm, mlp_h_post, attn_post, mlp_resolved
        )

    def test_mhc_head_registers_sinkhorn_via_compile_safe_custom_op(self):
        layer = ultra_mod.OpenPanguDecoderLayer.__new__(
            ultra_mod.OpenPanguDecoderLayer
        )
        nn.Module.__init__(layer)

        hidden_states = torch.randn(2, 4)
        mixed_hidden = object()
        normalized_hidden = object()
        h_post = object()
        h_res = object()
        registered_h_res = object()

        layer.attn_mhc_task_key = "attn_key"
        layer.attn_mhc_module = MagicMock()
        layer.attn_mhc_module.mhc_pre.return_value = (
            mixed_hidden,
            h_post,
            h_res,
        )
        layer.attn_mhc_module.maybe_register_sinkhorn.return_value = (
            registered_h_res
        )
        layer.input_layernorm = MagicMock(return_value=normalized_hidden)

        result = layer.mhc_head(hidden_states)

        self.assertIs(result[0], normalized_hidden)
        self.assertTrue(torch.equal(result[1], hidden_states))
        self.assertNotEqual(result[1].data_ptr(), hidden_states.data_ptr())
        self.assertIs(result[2], h_post)
        self.assertIs(result[3], registered_h_res)
        layer.attn_mhc_module.maybe_register_sinkhorn.assert_called_once_with(
            h_res, "attn_key"
        )

    def test_moe_forward_uses_fp32_activation_for_fused_router(self):
        moe = ultra_mod.OpenPanguMoE.__new__(ultra_mod.OpenPanguMoE)
        nn.Module.__init__(moe)
        moe.is_sequence_parallel = False
        moe.gate = MagicMock()
        moe.experts = MagicMock()

        hidden_states_bf16 = torch.randn(2, 4, dtype=torch.bfloat16)
        hidden_states_fp32 = torch.randn(2, 4, dtype=torch.float32)
        router_logits = torch.randn(2, 3, dtype=torch.float32)
        final_hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        moe.gate.return_value = (router_logits, None)
        moe.experts.return_value = (
            torch.empty_like(final_hidden_states),
            final_hidden_states,
        )

        result = moe(
            {
                "hidden_states_bf16": hidden_states_bf16,
                "hidden_states_fp32": hidden_states_fp32,
            }
        )

        self.assertEqual(moe.gate.call_count, 1)
        self.assertEqual(moe.experts.call_count, 1)
        torch.testing.assert_close(
            moe.gate.call_args.args[0], hidden_states_fp32
        )
        torch.testing.assert_close(
            moe.experts.call_args.kwargs["hidden_states"],
            hidden_states_bf16,
        )
        torch.testing.assert_close(
            moe.experts.call_args.kwargs["router_logits"],
            router_logits,
        )
        self.assertTrue(torch.equal(result, final_hidden_states))


class TestOpenPanguModelMHCWiring(unittest.TestCase):
    @staticmethod
    def _config(num_hidden_layers=2):
        return SimpleNamespace(
            pad_token_id=0,
            vocab_size=128,
            hidden_size=16,
            num_hidden_layers=num_hidden_layers,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            use_mhc=True,
            mhc_num_stream=2,
        )

    @staticmethod
    def _layer(index):
        layer = nn.Module()
        layer.use_mhc_fusion_op = True
        layer.attn_mhc_module = nn.Identity()
        layer.input_layernorm = nn.Identity()
        layer.attn_mhc_task_key = f"attn_{index}"
        layer._mhc_tail_refs = None
        return layer

    def test_init_links_local_layers_and_model_tail(self):
        cfg = self._config()
        layers = nn.ModuleList([self._layer(0), self._layer(1)])
        merge_mhc = nn.Identity()
        final_norm = nn.Identity()
        pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

        with _patch_open_pangu_model_init(pp_group, layers, 0, 2), patch.object(
            ultra_mod, "RMSNorm", return_value=final_norm
        ), patch.object(
            ultra_mod, "NPUmHCRL", return_value=merge_mhc
        ):
            model = ultra_mod.OpenPanguModel(vllm_config=_mhc_vllm_config(cfg))

        self.assertTrue(model.use_mhc_fusion_op)
        first_tail = layers[0]._mhc_tail_refs
        self.assertIs(first_tail[0], layers[1].attn_mhc_module)
        self.assertIs(first_tail[1], layers[1].input_layernorm)
        self.assertEqual(first_tail[2:], ("attn_1", False))
        last_tail = layers[1]._mhc_tail_refs
        self.assertIs(last_tail[0], merge_mhc)
        self.assertIs(last_tail[1], final_norm)
        self.assertEqual(last_tail[2:], (None, True))

    def test_init_stops_fused_state_at_pipeline_boundary(self):
        cfg = self._config()
        layer = self._layer(0)
        layers = nn.ModuleList([layer])
        pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=False)

        with _patch_open_pangu_model_init(pp_group, layers, 0, 1), patch.object(
            ultra_mod, "PPMissingLayer", return_value=nn.Identity()
        ), patch.object(
            ultra_mod, "NPUmHCRL", return_value=nn.Identity()
        ):
            model = ultra_mod.OpenPanguModel(vllm_config=_mhc_vllm_config(cfg))

        self.assertTrue(model.use_mhc_fusion_op)
        self.assertEqual(layer._mhc_tail_refs, (None, None, None, False))

    def test_forward_dispatches_fused_mhc_across_local_layers(self):
        model = ultra_mod.OpenPanguModel.__new__(ultra_mod.OpenPanguModel)
        nn.Module.__init__(model)
        layer_0 = nn.Module()
        layer_1 = nn.Module()
        rotary = MagicMock()
        cos = object()
        sin = object()
        rotary.get_cos_sin.return_value = (cos, sin)
        layer_0.self_attn = SimpleNamespace(rotary_emb=rotary)
        layer_0.attn_mhc_module = SimpleNamespace(
            can_use_fusion=MagicMock(return_value=True)
        )
        layer_0.mhc_head = MagicMock()
        layer_0.forward_mhc_fused = MagicMock()
        layer_1.forward_mhc_fused = MagicMock()
        model.layers = nn.ModuleList([layer_0, layer_1])
        model.start_layer = 0
        model.end_layer = 2
        model.use_mhc = True
        model.use_mhc_fusion_op = True
        model.num_stream = 2

        inputs_embeds = object()
        head_hidden = object()
        head_residual = object()
        head_h_post = object()
        head_h_res = object()
        middle_hidden = object()
        middle_residual = object()
        middle_h_post = object()
        middle_h_res = object()
        final_hidden = object()
        layer_0.mhc_head.return_value = (
            head_hidden,
            head_residual,
            head_h_post,
            head_h_res,
        )
        layer_0.forward_mhc_fused.return_value = (
            middle_hidden,
            middle_residual,
            middle_h_post,
            middle_h_res,
        )
        layer_1.forward_mhc_fused.return_value = (
            final_hidden,
            None,
            None,
            None,
        )
        pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)

        with patch.object(
            ultra_mod, "get_pp_group", return_value=pp_group
        ), patch.object(
            ultra_mod.model_extra_config.parall_config,
            "ena_seq_parallel",
            False,
        ):
            result = model.forward(
                input_ids=object(),
                positions=object(),
                intermediate_tensors=None,
                inputs_embeds=inputs_embeds,
            )

        self.assertIs(result, final_hidden)
        layer_0.mhc_head.assert_called_once_with(inputs_embeds)
        first_call = layer_0.forward_mhc_fused.call_args
        self.assertEqual(
            first_call.args,
            (
                head_hidden,
                head_residual,
                head_h_post,
                head_h_res,
                cos,
                sin,
            ),
        )
        self.assertFalse(first_call.kwargs["h_res_from_fused_split"])
        second_call = layer_1.forward_mhc_fused.call_args
        self.assertEqual(
            second_call.args,
            (
                middle_hidden,
                middle_residual,
                middle_h_post,
                middle_h_res,
                cos,
                sin,
            ),
        )
        self.assertTrue(second_call.kwargs["h_res_from_fused_split"])


class TestOpenPanguMHCWeightPostProcessing(unittest.TestCase):
    def test_process_mhc_weights_initializes_mhc_modules(self):
        model = ultra_mod.OpenPanguModelBase.__new__(
            ultra_mod.OpenPanguModelBase
        )
        nn.Module.__init__(model)
        mhc = ultra_mod.NPUmHCRL.__new__(ultra_mod.NPUmHCRL)
        nn.Module.__init__(mhc)
        mhc.process_weights_after_loading = MagicMock()
        model.add_module("mhc", mhc)

        model._process_mhc_weights_after_loading()

        mhc.process_weights_after_loading.assert_called_once_with()

    def test_moe_post_weight_load_processes_mhc_weights(self):
        model = ultra_mod.OpenPanguMoEModel.__new__(
            ultra_mod.OpenPanguMoEModel
        )
        nn.Module.__init__(model)
        model._process_mhc_weights_after_loading = MagicMock()

        model.post_weight_load()

        model._process_mhc_weights_after_loading.assert_called_once_with()

    def test_base_load_weights_runs_split_mark_and_mhc_processing(self):
        model = _new_open_pangu_model_base()
        model._process_mhc_weights_after_loading = MagicMock()
        loaded_params = {"model.layers.0.self_attn.q_b_proj.weight"}
        result, mark_split = _load_weights_with_split_mark(model, loaded_params)

        self.assertIs(result, loaded_params)
        mark_split.assert_called_once_with(model, loaded_params)
        model._process_mhc_weights_after_loading.assert_called_once_with()


class TestOpenPanguMoERouterGatingInFp32(unittest.TestCase):
    """Tests for router_gating_in_fp32 flag in OpenPanguMoE.__init__."""

    @staticmethod
    def _make_config():
        return SimpleNamespace(
            hidden_act="silu",
            n_routed_experts=4,
            n_shared_experts=1,
            hidden_size=16,
            moe_intermediate_size=32,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            model_type="openpangu_v2",
        )

    @staticmethod
    def _make_parallel_config():
        return SimpleNamespace(
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        )

    @patch.object(ultra_mod, "NPUSharedFusedMoE", MagicMock())
    @patch.object(ultra_mod, "OpenPanguMLP", MagicMock())
    @patch.object(ultra_mod, "ReplicatedLinear", MagicMock(return_value=nn.Identity()))
    @patch.object(
        ultra_mod,
        "get_ep_group",
        return_value=SimpleNamespace(
            device_group=SimpleNamespace(rank=lambda: 0, size=lambda: 2)
        ),
    )
    @patch.object(ultra_mod, "get_tp_group", return_value=SimpleNamespace(rank_in_group=0))
    @patch.object(ultra_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_gate_uses_params_dtype_fp32_when_router_gating_in_fp32_true(self, *_):
        """When router_gating_in_fp32=True, gate should be constructed with params_dtype=torch.float32."""
        cfg = self._make_config()
        parallel_cfg = self._make_parallel_config()

        with patch.object(
            ultra_mod.model_extra_config.operator_opt_config,
            "router_gating_in_fp32",
            True,
        ):
            ultra_mod.OpenPanguMoE(cfg, parallel_cfg, quant_config=None, prefix="moe")

        # ReplicatedLinear should be called with params_dtype=torch.float32
        gate_call_kwargs = ultra_mod.ReplicatedLinear.call_args.kwargs
        assert gate_call_kwargs.get("params_dtype") == torch.float32, (
            f"Expected params_dtype=torch.float32, got {gate_call_kwargs.get('params_dtype')}"
        )

    @patch.object(ultra_mod, "NPUSharedFusedMoE", MagicMock())
    @patch.object(ultra_mod, "OpenPanguMLP", MagicMock())
    @patch.object(ultra_mod, "ReplicatedLinear", MagicMock(return_value=nn.Identity()))
    @patch.object(
        ultra_mod,
        "get_ep_group",
        return_value=SimpleNamespace(
            device_group=SimpleNamespace(rank=lambda: 0, size=lambda: 2)
        ),
    )
    @patch.object(ultra_mod, "get_tp_group", return_value=SimpleNamespace(rank_in_group=0))
    @patch.object(ultra_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_gate_no_params_dtype_when_router_gating_in_fp32_false(self, *_):
        """When router_gating_in_fp32=False, gate should be constructed without params_dtype."""
        cfg = self._make_config()
        parallel_cfg = self._make_parallel_config()

        with patch.object(
            ultra_mod.model_extra_config.operator_opt_config,
            "router_gating_in_fp32",
            False,
        ):
            ultra_mod.OpenPanguMoE(cfg, parallel_cfg, quant_config=None, prefix="moe")

        # ReplicatedLinear should NOT be called with params_dtype
        gate_call_kwargs = ultra_mod.ReplicatedLinear.call_args.kwargs
        assert "params_dtype" not in gate_call_kwargs, (
            f"Expected no params_dtype in kwargs, but got params_dtype={gate_call_kwargs.get('params_dtype')}"
        )


class TestOpenPanguModelBaseComputeLogits(unittest.TestCase):
    """Tests for compute_logits dtype alignment with lm_head weight."""

    def test_compute_logits_casts_hidden_states_to_lm_head_dtype(self):
        base = ultra_mod.OpenPanguModelBase.__new__(ultra_mod.OpenPanguModelBase)
        base.lm_head = SimpleNamespace(
            weight=SimpleNamespace(dtype=torch.float32),
        )
        captured = {}

        def _logits_processor(lm_head, hidden_states):
            captured["hidden_dtype"] = hidden_states.dtype
            return torch.randn(2, 16)

        base.logits_processor = _logits_processor
        hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)

        logits = ultra_mod.OpenPanguModelBase.compute_logits(base, hidden_states)

        self.assertEqual(captured["hidden_dtype"], torch.float32)
        self.assertEqual(tuple(logits.shape), (2, 16))


class TestOpenPanguModelBaseLoadWeights(unittest.TestCase):
    def test_marks_split_q_params_after_loading(self):
        model = _new_open_pangu_model_base()
        loaded_params = {"model.layers.0.self_attn.q_b_proj.weight"}
        result, mark_split = _load_weights_with_split_mark(model, loaded_params)

        self.assertIs(result, loaded_params)
        mark_split.assert_called_once_with(model, loaded_params)
