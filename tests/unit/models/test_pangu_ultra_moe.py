# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from omni.v1.models.pangu import pangu_ultra_moe as ultra_mod


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

    @patch.object(ultra_mod, "SharedFusedMoE", MagicMock())
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

    @patch.object(ultra_mod, "SharedFusedMoE", MagicMock())
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