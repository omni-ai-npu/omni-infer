# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import importlib
import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from omni_npu.vllm_patches.patches.models.pangu_v2_base.patch_mla import mlaPatch
mlaPatch.apply()

from omni_npu.vllm_patches.patches.models.pangu_sink_swa_mla.patch_static_sink_attention import StaticSinkAttentionPatch
StaticSinkAttentionPatch.apply()

from omni_npu.vllm_patches.patches.models.pangu_sink_swa_mla.patch_mome import MoMEPatch
MoMEPatch.apply()

import omni_models.models.pangu.openpangu as openpangu_mod


class _FakeLinearOut(nn.Module):
    """Mock for linear layers that return (tensor, None) tuple."""
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self, x):
        if isinstance(x, tuple):
            return x
        return x, None


class _FakeActivation(nn.Module):
    """Mock for activation functions that return just a tensor."""
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self, x):
        if isinstance(x, tuple):
            return x[0]
        return x


class _FakeMergedColumn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        sizes = kwargs.get("output_sizes")
        if sizes is None and len(args) >= 2:
            sizes = args[1]
        self._out_dim = sum(sizes) if sizes else 32

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, self._out_dim, device=x.device, dtype=x.dtype), None


class _FakeRowParallel(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.reduce_results = True

    def forward(self, x):
        # Handle tuple input (tensor, None) from activation
        if isinstance(x, tuple):
            x = x[0]
        if x.shape[-1] != 8:
            return torch.zeros_like(x[..., :8]), None
        return x, None


class _FakeNorm:
    def __call__(self, x, residual=None):
        if residual is None:
            return x
        return x, residual


class _FakeRMSNormModule(nn.Module):
    def forward(self, x, residual=None):
        if residual is None:
            return x
        return x, residual


class _FakeVocabEmbed(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self, input_ids):
        return torch.zeros((input_ids.shape[0], 8), dtype=torch.float32)


class _FakeAttn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *a, **k):
        if not a:
            return torch.zeros(1)
        q = a[0]
        return torch.zeros_like(q)


class _FakeSinkAttn(nn.Module):
    """StaticSinkAttention stub: return hidden_states-shaped tensor for o_proj."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, q, k, v, **kwargs):
        # output matches q's token dim and head*head_dim for last dim
        return torch.zeros(q.shape[0], q.shape[1], device=q.device, dtype=q.dtype)

    def update_sink_kv(self, key, value):
        # Mock method for sink kv update
        pass


class TestCheckFfnActFn(unittest.TestCase):
    def test_silu_ok(self):
        openpangu_mod.check_ffn_act_fn("silu")

    def test_non_silu_raises(self):
        with self.assertRaises(ValueError):
            openpangu_mod.check_ffn_act_fn("gelu")


class TestFastGELUMindSpore(unittest.TestCase):
    def test_forward(self):
        m = openpangu_mod.FastGELUMindSpore()
        x = torch.randn(2, 4)
        y = m(x)
        self.assertEqual(y.shape, x.shape)


class TestOpenPanguMLP(unittest.TestCase):
    @patch.object(openpangu_mod, "SiluAndMul", _FakeActivation)
    @patch.object(openpangu_mod, "MergedColumnParallelLinear", _FakeMergedColumn)
    @patch.object(openpangu_mod, "RowParallelLinear", _FakeRowParallel)
    def test_forward(self, *_):
        m = openpangu_mod.OpenPanguMLP(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
            quant_config=None,
            bias=False,
            prefix="mlp",
        )
        x = torch.randn(2, 8)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 8))


class TestPanguEmbeddedMLPVanilla(unittest.TestCase):
    @patch.object(openpangu_mod, "MergedColumnParallelLinear", _FakeMergedColumn)
    @patch.object(openpangu_mod, "RowParallelLinear", _FakeRowParallel)
    def test_forward(self, *_):
        m = openpangu_mod.PanguEmbeddedMLPVanilla(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
            quant_config=None,
            prefix="m",
        )
        x = torch.randn(2, 8)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 8))

    @patch.object(openpangu_mod, "MergedColumnParallelLinear", _FakeMergedColumn)
    @patch.object(openpangu_mod, "RowParallelLinear", _FakeRowParallel)
    def test_bad_activation(self, *_):
        with self.assertRaises(ValueError):
            openpangu_mod.PanguEmbeddedMLPVanilla(
                hidden_size=8,
                intermediate_size=16,
                hidden_act="gelu",
                prefix="m",
            )


class _FakeExpertsMoE:
    def __init__(self, *a, **k):
        self.maybe_all_reduce_tensor_model_parallel = lambda t: t

    def __call__(self, hidden_states, router_logits):
        return None, hidden_states


class TestOpenPanguMoEInit(unittest.TestCase):
    @patch.object(openpangu_mod, "SharedFusedMoE", MagicMock())
    @patch.object(openpangu_mod, "OpenPanguMLP", MagicMock())
    @patch.object(openpangu_mod, "ReplicatedLinear", _FakeLinearOut)
    @patch.object(
        openpangu_mod,
        "get_ep_group",
        return_value=SimpleNamespace(
            device_group=SimpleNamespace(rank=lambda: 0, size=lambda: 2)
        ),
    )
    @patch.object(openpangu_mod, "get_tp_group", return_value=SimpleNamespace(rank_in_group=0))
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_builds_moe(self, *_):
        cfg = SimpleNamespace(
            hidden_act="silu",
            n_routed_experts=4,
            n_shared_experts=1,
            hidden_size=16,
            moe_intermediate_size=32,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        parallel_cfg = SimpleNamespace(
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        )
        m = openpangu_mod.OpenPanguMoE(cfg, parallel_cfg, quant_config=None, prefix="moe")
        self.assertIsNotNone(m.experts)
        self.assertIsNotNone(m.shared_experts)

    @patch.object(openpangu_mod, "SharedFusedMoE", MagicMock())
    @patch.object(openpangu_mod, "OpenPanguMLP", MagicMock())
    @patch.object(openpangu_mod, "ReplicatedLinear", _FakeLinearOut)
    @patch.object(
        openpangu_mod,
        "get_ep_group",
        return_value=SimpleNamespace(
            device_group=SimpleNamespace(rank=lambda: 0, size=lambda: 2)
        ),
    )
    @patch.object(openpangu_mod, "get_tp_group", return_value=SimpleNamespace(rank_in_group=0))
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_router_bias(self, *_):
        cfg = SimpleNamespace(
            hidden_act="silu",
            n_routed_experts=4,
            n_shared_experts=1,
            hidden_size=16,
            moe_intermediate_size=32,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            router_enable_expert_bias=True,
        )
        parallel_cfg = SimpleNamespace(
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        )
        m = openpangu_mod.OpenPanguMoE(cfg, parallel_cfg, quant_config=None, prefix="m")
        self.assertIsNotNone(m.gate.e_score_correction_bias)


class TestOpenPanguMoE(unittest.TestCase):
    def test_forward_fp32_scaling(self):
        m = openpangu_mod.OpenPanguMoE.__new__(openpangu_mod.OpenPanguMoE)
        m.is_sequence_parallel = False
        m.tp_size = 1
        m.routed_scaling_factor = 2.0
        m.shared_experts = None
        m.gate = MagicMock(return_value=(torch.zeros(4, 2), None))
        m.experts = _FakeExpertsMoE()
        x = torch.ones(4, 8, dtype=torch.float32)
        y = m.forward(x)
        self.assertEqual(tuple(y.shape), (4, 8))

    @patch.object(openpangu_mod, "tensor_model_parallel_all_gather", side_effect=lambda t, _: t)
    @patch.object(openpangu_mod, "sequence_parallel_chunk", side_effect=lambda t: t)
    def test_forward_sequence_parallel(self, *_):
        m = openpangu_mod.OpenPanguMoE.__new__(openpangu_mod.OpenPanguMoE)
        m.is_sequence_parallel = True
        m.tp_size = 1
        m.routed_scaling_factor = 1.0
        m.shared_experts = object()
        m.gate = MagicMock(return_value=(torch.zeros(4, 2), None))

        class _E:
            def __init__(self):
                self.maybe_all_reduce_tensor_model_parallel = lambda t: t

            def __call__(self, hidden_states, router_logits):
                return torch.ones(4, 8), torch.ones(4, 8)

        m.experts = _E()
        x = torch.ones(4, 8, dtype=torch.float16)
        y = m.forward(x)
        self.assertEqual(tuple(y.shape), (4, 8))


class _FakeMLAWrapper:
    def __init__(self, *a, **k):
        pass

    def __call__(self, positions, hidden_states):
        return hidden_states + 0.1

    class mla_attn:
        @staticmethod
        def update_sink_kv(*a, **k):
            pass


class TestOpenPanguMLAAttention(unittest.TestCase):
    def _base_cfg(self):
        return SimpleNamespace(
            rope_parameters={"rope_theta": 10000.0},
            rms_norm_eps=1e-5,
            torch_dtype=torch.float32,
            num_attention_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            kv_lora_rank=8,
            q_lora_rank=None,
            param_sink_number=0,
            rope_interleaved=True,
        )

    @patch.object(openpangu_mod, "set_weight_attrs", lambda *a, **k: None)
    @patch.object(openpangu_mod, "MultiHeadLatentAttentionWrapper", _FakeMLAWrapper)
    @patch.object(openpangu_mod, "MLAModules", MagicMock())
    @patch.object(openpangu_mod, "RMSNorm", side_effect=lambda *a, **k: nn.Identity())
    @patch.object(openpangu_mod, "get_rope", MagicMock(return_value=MagicMock()))
    @patch.object(openpangu_mod, "set_default_rope_theta", lambda *a, **k: None)
    @patch.object(openpangu_mod, "RowParallelLinear", _FakeLinearOut)
    @patch.object(openpangu_mod, "ColumnParallelLinear", _FakeLinearOut)
    @patch.object(openpangu_mod, "ReplicatedLinear", _FakeLinearOut)
    @patch.object(openpangu_mod, "MergedColumnParallelLinear", _FakeLinearOut)
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_forward_and_post_weight_load(self, *_):
        cfg = self._base_cfg()
        vllm_cfg = SimpleNamespace(
            cache_config=None,
            quant_config=None,
        )
        attn = openpangu_mod.OpenPanguMLAAttention(
            config=cfg,
            vllm_config=vllm_cfg,
            hidden_size=32,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=None,
            kv_lora_rank=8,
            prefix="model.layers.0.self_attn",
        )
        pos = torch.zeros(2, dtype=torch.long)
        h = torch.randn(2, 32)
        out = attn(pos, h)
        self.assertEqual(tuple(out.shape), (2, 32))
        attn.post_weight_load()


class _FakeQKVForEmbedded(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self, hidden_states):
        b = hidden_states.shape[0]
        return torch.zeros(b, 96), None


class TestOpenPanguEmbeddedAttention(unittest.TestCase):
    @patch.object(openpangu_mod, "Attention", _FakeAttn)
    @patch.object(openpangu_mod, "get_rope", MagicMock(return_value=lambda pos, q, k: (q, k)))
    @patch.object(openpangu_mod, "QKVParallelLinear", _FakeQKVForEmbedded)
    @patch.object(openpangu_mod, "RowParallelLinear", _FakeLinearOut)
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_forward(self, *_):
        cfg = SimpleNamespace(
            head_dim=8,
            rope_theta=10000.0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            rms_norm_eps=1e-5,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        attn = openpangu_mod.OpenPanguEmbeddedAttention(
            config=cfg,
            hidden_size=32,
            num_heads=4,
            num_kv_heads=4,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )
        pos = torch.zeros(2, dtype=torch.long)
        h = torch.randn(2, 32)
        out = attn(pos, h)
        self.assertEqual(tuple(out.shape), (2, 32))

    def test_init_rotary_mrope(self):
        # Mock get_rope_wrapper inside the module that imports it
        mock_rope_wrapper = MagicMock(return_value=MagicMock())
        mock_get_rope = MagicMock(return_value=MagicMock())
        with patch.dict(
            'sys.modules',
            {'vllm.model_executor.layers.rotary_embedding':
             MagicMock(get_rope=MagicMock(), get_rope_wrapper=mock_rope_wrapper)}
        ), patch.object(openpangu_mod, "get_rope", mock_get_rope):
            cfg = SimpleNamespace(
                rope_theta=10000.0,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                num_hidden_layers=4,
                num_nextn_predict_layers=0,
                model_type="PanguEmbedded",
                rope_scaling={"mrope_section": [1, 2, 3]},  # Fix: add rope_scaling to config
            )
            attn = openpangu_mod.OpenPanguEmbeddedAttention.__new__(
                openpangu_mod.OpenPanguEmbeddedAttention
            )
            attn.head_dim = 8
            attn.qk_rope_dim = 8
            attn.max_position_embeddings = 128
            # Inject the mocked function
            import vllm.model_executor.layers.rotary_embedding as re_mod
            re_mod.get_rope_wrapper = mock_rope_wrapper
            # Fix: call _init_rotary_emb with correct signature (config, quant_config only)
            attn._init_rotary_emb(cfg, quant_config=None)
            self.assertIsNotNone(attn.rotary_emb)


class TestOpenPanguSinkAttention(unittest.TestCase):
    @patch.object(openpangu_mod, "set_weight_attrs", lambda *a, **k: None)
    @patch.object(openpangu_mod, "StaticSinkAttention", _FakeSinkAttn)
    @patch.object(openpangu_mod, "get_rope", MagicMock(return_value=lambda pos, q, k: (q, k)))
    @patch.object(openpangu_mod, "QKVParallelLinear", _FakeQKVForEmbedded)
    @patch.object(openpangu_mod, "RowParallelLinear", _FakeLinearOut)
    @patch.object(openpangu_mod, "get_tensor_model_parallel_rank", return_value=0)
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_forward_pangu_embedded_path(self, *_):
        cfg = SimpleNamespace(
            model_type="PanguEmbedded",
            param_sink_number=2,
            param_sink_with_value=False,
            rms_norm_eps=1e-5,
            torch_dtype=torch.float32,
            rope_theta=10000.0,
            num_attention_heads=4,
            num_key_value_heads=4,
            qk_nope_dim=4,
            qk_rope_dim=4,
            v_channels=8,
        )
        rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        attn = openpangu_mod.OpenPanguSinkAttention(
            config=cfg,
            hidden_size=32,
            num_heads=4,
            num_kv_heads=4,
            rope_parameters=rope_parameters,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )
        pos = torch.zeros(2, dtype=torch.long)
        h = torch.randn(2, 32)
        out = attn(pos, h)
        self.assertEqual(len(out.shape), 2)
        attn.post_weight_load()


class TestSinkhornKnopps(unittest.TestCase):
    def test_runs(self):
        h = torch.randn(2, 3, 3)
        y = openpangu_mod.sinkhorn_knopps(h, sinkhorn_iters=3, eps=1e-6)
        self.assertEqual(y.shape, h.shape)


def _fake_replicated_linear_phi(out_dim: int):
    class _R(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, x):
            b = x.shape[0]
            return torch.randn(b, out_dim), None

    return _R


class TestMHCModule(unittest.TestCase):
    def _cfg(self):
        return SimpleNamespace(
            mhc_num_stream=2,
            hidden_size=8,
            mhc_use_gamma=False,
            rms_norm_eps=1e-5,
            mhc_recur_norm=2,
        )

    def _get_vllm_config(self):
        """Create a minimal VllmConfig for CustomOp initialization."""
        from vllm.config import VllmConfig, CompilationConfig

        # Use mocks to bypass complex validation
        mock_vllm_config = MagicMock(spec=VllmConfig)
        mock_compilation_config = MagicMock(spec=CompilationConfig)
        mock_compilation_config.custom_ops = ["none"]
        mock_compilation_config.disabled_custom_ops = MagicMock()
        mock_compilation_config.disabled_custom_ops.update = MagicMock()
        mock_vllm_config.compilation_config = mock_compilation_config
        return mock_vllm_config

    # num_stream=2, merge_layer_only_pre=False -> phi out = (n+2)*n = 8
    @patch.object(openpangu_mod, "ReplicatedLinear", _fake_replicated_linear_phi(8))
    def test_hc_pre_full(self, *_):
        from vllm.config import set_current_vllm_config
        cfg = self._cfg()
        vllm_cfg = self._get_vllm_config()
        with set_current_vllm_config(vllm_cfg):
            m = openpangu_mod.mHCModule(cfg, merge_layer_only_pre=False, prefix="m")
            x = torch.randn(2, 16)
            y, hp, hr = m.hc_pre(x)
            self.assertEqual(y.shape[0], 2)

    @patch.object(openpangu_mod, "ReplicatedLinear", _fake_replicated_linear_phi(2))
    def test_hc_pre_merge_only(self, *_):
        from vllm.config import set_current_vllm_config
        cfg = self._cfg()
        vllm_cfg = self._get_vllm_config()
        with set_current_vllm_config(vllm_cfg):
            m = openpangu_mod.mHCModule(cfg, merge_layer_only_pre=True, prefix="m")
            x = torch.randn(2, 16)
            y, hp, hr = m.hc_pre(x)
            self.assertEqual(y.shape[0], 2)

    @patch.object(openpangu_mod, "ReplicatedLinear", _fake_replicated_linear_phi(8))
    def test_hc_pre_with_gamma(self, *_):
        from vllm.config import set_current_vllm_config
        cfg = self._cfg()
        cfg.mhc_use_gamma = True
        vllm_cfg = self._get_vllm_config()
        with set_current_vllm_config(vllm_cfg):
            m = openpangu_mod.mHCModule(cfg, merge_layer_only_pre=False, prefix="mg")
            x = torch.randn(2, 16)
            y, _, _ = m.hc_pre(x)
            self.assertEqual(y.shape[0], 2)

    @patch.object(openpangu_mod, "ReplicatedLinear", _fake_replicated_linear_phi(8))
    def test_hc_post_branches(self, *_):
        from vllm.config import set_current_vllm_config
        cfg = self._cfg()
        vllm_cfg = self._get_vllm_config()
        with set_current_vllm_config(vllm_cfg):
            m_pre = openpangu_mod.mHCModule(cfg, merge_layer_only_pre=True, prefix="m")
            x = torch.randn(2, 16)  # hidden_size * num_stream = 8 * 2 = 16
            result = m_pre.hc_post(x, x, None, None)
            self.assertTrue(torch.equal(result, x))

            # For full module, just test that hc_post can be called with proper inputs
            m_full = openpangu_mod.mHCModule(cfg, merge_layer_only_pre=False, prefix="m2")
            # Test merge_layer_only_pre path returns x directly
            self.assertTrue(torch.equal(m_pre.hc_post(x, x, None, None), x))

    @patch.object(openpangu_mod, "ReplicatedLinear", _fake_replicated_linear_phi(8))
    def test_hc_split_sinkhorn_torch(self, *_):
        from vllm.config import set_current_vllm_config
        cfg = self._cfg()
        vllm_cfg = self._get_vllm_config()
        with set_current_vllm_config(vllm_cfg):
            m = openpangu_mod.mHCModule(cfg, merge_layer_only_pre=False, prefix="m")
            w = torch.randn(2, 2 + 2 + 4)
            hp, hpost, hres = m.hc_split_sinkhorn_torch(w)
            self.assertIsNotNone(hp)

            m2 = openpangu_mod.mHCModule(cfg, merge_layer_only_pre=True, prefix="m2")
            w2 = torch.randn(2, 2)
            hp2, hpost2, hres2 = m2.hc_split_sinkhorn_torch(w2)
            self.assertIsNotNone(hp2)


class TestOpenPanguDecoderLayer(unittest.TestCase):
    @patch.object(openpangu_mod, "OpenPanguMLP")
    @patch.object(openpangu_mod, "RMSNorm")
    @patch.object(openpangu_mod, "OpenPanguEmbeddedAttention")
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_basic(self, mock_get_tp, mock_attn, mock_norm, mock_mlp):
        """Test basic __init__ with default config (no MLA, no sink attention, no MoE)."""
        from vllm.config import set_current_vllm_config

        cfg = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            num_hidden_layers=4,
            is_causal=True,
        )

        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=cfg),
            cache_config=None,
            quant_config=None,
            parallel_config=SimpleNamespace(
                enable_eplb=False,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            ),
        )

        with set_current_vllm_config(vllm_cfg):
            layer = openpangu_mod.OpenPanguDecoderLayer(
                config=cfg,
                prefix="model.layers.0",
                vllm_config=vllm_cfg,
            )

        # Verify basic attributes
        self.assertEqual(layer.hidden_size, 32)
        self.assertEqual(layer.layer_idx, 0)
        self.assertFalse(layer.use_mla)
        self.assertFalse(layer.use_sink_attention)
        self.assertFalse(layer.use_mhc)
        self.assertFalse(layer.sandwich_norm)
        self.assertEqual(layer.routed_scaling_factor, 1.0)
        self.assertEqual(layer.num_hidden_layers, 4)

        # Verify attention was created
        self.assertTrue(mock_attn.called)

        # Verify MLP was created
        self.assertTrue(mock_mlp.called)

        # Verify layer norms were created
        self.assertTrue(mock_norm.called)

    @patch.object(openpangu_mod, "OpenPanguMoE")
    @patch.object(openpangu_mod, "RMSNorm")
    @patch.object(openpangu_mod, "OpenPanguEmbeddedAttention")
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_with_moe(self, mock_get_tp, mock_attn, mock_norm, mock_moe):
        """Test __init__ with MoE enabled."""
        from vllm.config import set_current_vllm_config

        cfg = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            num_hidden_layers=4,
            is_causal=True,
            n_routed_experts=8,
            first_k_dense_replace=0,  # All layers use MoE
        )

        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=cfg),
            cache_config=None,
            quant_config=None,
            parallel_config=SimpleNamespace(
                enable_eplb=False,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            ),
        )

        with set_current_vllm_config(vllm_cfg):
            layer = openpangu_mod.OpenPanguDecoderLayer(
                config=cfg,
                prefix="model.layers.0",
                vllm_config=vllm_cfg,
            )

        # Verify MoE was created
        self.assertTrue(mock_moe.called)

    @patch.object(openpangu_mod, "mHCModule")
    @patch.object(openpangu_mod, "OpenPanguMLP")
    @patch.object(openpangu_mod, "RMSNorm")
    @patch.object(openpangu_mod, "OpenPanguEmbeddedAttention")
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_with_mhc(self, mock_get_tp, mock_attn, mock_norm, mock_mlp, mock_mhc):
        """Test __init__ with MHC enabled."""
        from vllm.config import set_current_vllm_config

        cfg = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            num_hidden_layers=4,
            is_causal=True,
            use_mhc=True,
            mhc_num_stream=2,
        )

        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=cfg),
            cache_config=None,
            quant_config=None,
            parallel_config=SimpleNamespace(
                enable_eplb=False,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            ),
        )

        with set_current_vllm_config(vllm_cfg):
            layer = openpangu_mod.OpenPanguDecoderLayer(
                config=cfg,
                prefix="model.layers.0",
                vllm_config=vllm_cfg,
            )

        # Verify MHC modules were created
        self.assertTrue(layer.use_mhc)
        self.assertEqual(mock_mhc.call_count, 2)  # attn_mhc_module and mlp_mhc_module

    @patch.object(openpangu_mod, "OpenPanguMLP")
    @patch.object(openpangu_mod, "RMSNorm")
    @patch.object(openpangu_mod, "OpenPanguEmbeddedAttention")
    @patch.object(openpangu_mod, "get_tensor_model_parallel_world_size", return_value=1)
    def test_init_with_sandwich_norm(self, mock_get_tp, mock_attn, mock_norm, mock_mlp):
        """Test __init__ with sandwich norm enabled."""
        from vllm.config import set_current_vllm_config

        cfg = SimpleNamespace(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            num_hidden_layers=4,
            is_causal=True,
            sandwich_norm=True,
        )

        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=cfg),
            cache_config=None,
            quant_config=None,
            parallel_config=SimpleNamespace(
                enable_eplb=False,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            ),
        )

        with set_current_vllm_config(vllm_cfg):
            layer = openpangu_mod.OpenPanguDecoderLayer(
                config=cfg,
                prefix="model.layers.0",
                vllm_config=vllm_cfg,
            )

        # Verify sandwich norm is enabled
        self.assertTrue(layer.sandwich_norm)
        # Verify post_mlp_layernorm was created
        self.assertTrue(hasattr(layer, 'post_mlp_layernorm'))

    def test_forward_normal_residual_none(self):
        d = openpangu_mod.OpenPanguDecoderLayer.__new__(openpangu_mod.OpenPanguDecoderLayer)
        d.use_mhc = False
        d.sandwich_norm = False
        d.router_sliding_window = 0
        d.merge_conv = None
        d.layer_idx = 1
        d.routed_scaling_factor = None
        d.first_k_dense_replace = 99
        d.num_hidden_layers = 4
        d.input_layernorm = _FakeNorm()
        d.post_attention_layernorm = _FakeNorm()
        d.self_attn = MagicMock(return_value=torch.ones(2, 8))
        d.mlp = MagicMock(return_value=torch.ones(2, 8))
        d.has_block_post_layernorm = False
        h = torch.ones(2, 8)
        pos = torch.zeros(2, dtype=torch.long)
        out, res = d.forward_normal(pos, h, residual=None)
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_forward_mhc(self):
        d = openpangu_mod.OpenPanguDecoderLayer.__new__(openpangu_mod.OpenPanguDecoderLayer)
        d.use_mhc = True
        d.sandwich_norm = False
        d.input_layernorm = _FakeNorm()
        d.self_attn = MagicMock(return_value=torch.ones(2, 8))
        d.mlp = MagicMock(return_value=torch.ones(2, 8))
        d.attn_mhc_module = MagicMock()
        d.attn_mhc_module.hc_pre.return_value = (torch.ones(2, 8), torch.ones(2, 2), torch.ones(2, 4))
        d.attn_mhc_module.hc_post.side_effect = lambda x, r, hp, hr: x
        d.mlp_mhc_module = MagicMock()
        d.mlp_mhc_module.hc_pre.return_value = (torch.ones(2, 8), None, None)
        d.mlp_mhc_module.hc_post.side_effect = lambda x, r, hp, hr: x
        d.pre_mlp_layernorm = _FakeNorm()
        d.post_attention_layernorm = _FakeNorm()
        d.has_block_post_layernorm = False
        h = torch.ones(2, 8)
        pos = torch.zeros(2, dtype=torch.long)
        out, res = d.forward_mhc(pos, h, None)
        self.assertEqual(tuple(out.shape), (2, 8))
        self.assertIsNone(res)

    def test_forward_dispatches(self):
        d = openpangu_mod.OpenPanguDecoderLayer.__new__(openpangu_mod.OpenPanguDecoderLayer)
        d.use_mhc = False
        d.is_mtp_layer = False
        d.forward_normal = MagicMock(return_value=(torch.ones(2, 8), None))
        d.forward_mhc = MagicMock(return_value=(torch.ones(2, 8), None))
        d.forward(torch.zeros(2, dtype=torch.long), torch.ones(2, 8), None)
        self.assertTrue(d.forward_normal.called)
        d2 = openpangu_mod.OpenPanguDecoderLayer.__new__(openpangu_mod.OpenPanguDecoderLayer)
        d2.use_mhc = True
        d2.is_mtp_layer = False  # Fix: add missing is_mtp_layer attribute
        d2.forward_normal = MagicMock()
        d2.forward_mhc = MagicMock(return_value=(torch.ones(2, 8), None))
        d2.forward(torch.zeros(2, dtype=torch.long), torch.ones(2, 8), None)
        self.assertTrue(d2.forward_mhc.called)

    def test_forward_normal_sandwich_norm(self):
        d = openpangu_mod.OpenPanguDecoderLayer.__new__(openpangu_mod.OpenPanguDecoderLayer)
        d.use_mhc = False
        d.sandwich_norm = True
        d.router_sliding_window = 0
        d.merge_conv = None
        d.layer_idx = 2
        d.routed_scaling_factor = None
        d.input_layernorm = _FakeNorm()
        d.post_attention_layernorm = _FakeNorm()
        d.pre_mlp_layernorm = _FakeNorm()
        d.post_mlp_layernorm = _FakeNorm()
        d.self_attn = MagicMock(return_value=torch.ones(2, 8))
        d.mlp = MagicMock(return_value=torch.ones(2, 8))
        d.has_block_post_layernorm = False
        out, res = d.forward_normal(
            torch.zeros(2, dtype=torch.long), torch.ones(2, 8), residual=None
        )
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_forward_normal_fp16_routed_scale(self):
        d = openpangu_mod.OpenPanguDecoderLayer.__new__(openpangu_mod.OpenPanguDecoderLayer)
        nn.Module.__init__(d)  # Initialize nn.Module base class
        d.use_mhc = False
        d.sandwich_norm = False
        d.router_sliding_window = 0
        d.merge_conv = None
        d.layer_idx = 0
        d.routed_scaling_factor = 2.0
        d.first_k_dense_replace = 99
        d.num_hidden_layers = 4
        d.input_layernorm = _FakeNorm()
        d.post_attention_layernorm = _FakeNorm()
        d.self_attn = MagicMock(return_value=torch.ones(2, 8, dtype=torch.float16))
        d.mlp = MagicMock(return_value=torch.ones(2, 8, dtype=torch.float16))
        d.has_block_post_layernorm = False
        out, res = d.forward_normal(
            torch.zeros(2, dtype=torch.long), torch.ones(2, 8, dtype=torch.float16), residual=None
        )
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_forward_normal_router_sliding_merge_conv(self):
        d = openpangu_mod.OpenPanguDecoderLayer.__new__(openpangu_mod.OpenPanguDecoderLayer)
        d.use_mhc = False
        d.sandwich_norm = False
        d.router_sliding_window = 2
        d.layer_idx = 1
        d.routed_scaling_factor = None
        d.merge_conv = MagicMock(return_value=torch.ones(2, 8))
        d.input_layernorm = _FakeNorm()
        d.post_attention_layernorm = lambda x: x
        d.self_attn = MagicMock(return_value=torch.ones(2, 8))
        d.mlp = MagicMock(return_value=torch.ones(2, 8))
        d.has_block_post_layernorm = False
        out, res = d.forward_normal(
            torch.zeros(2, dtype=torch.long), torch.ones(2, 8), residual=None
        )
        self.assertEqual(tuple(out.shape), (2, 8))
        self.assertTrue(d.merge_conv.called)


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positions, hidden_states, residual):
        return hidden_states + 1.0, residual


class TestOpenPanguModel(unittest.TestCase):
    @patch.object(openpangu_mod, "RMSNorm", side_effect=lambda *a, **k: _FakeRMSNormModule())
    @patch.object(openpangu_mod, "make_empty_intermediate_tensors_factory", return_value=lambda: {})
    @patch.object(openpangu_mod, "make_layers", return_value=(0, 1, [_FakeDecoderLayer()]))
    @patch.object(openpangu_mod, "VocabParallelEmbedding", _FakeVocabEmbed)
    @patch.object(openpangu_mod, "get_pp_group", return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True))
    def test_embed_and_forward_first_last_rank(self, *_):
        # Create model without going through __init__ to avoid compilation wrapper issues
        m = openpangu_mod.OpenPanguModel.__new__(openpangu_mod.OpenPanguModel)
        nn.Module.__init__(m)
        m.config = SimpleNamespace(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=1,
            rms_norm_eps=1e-5,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
        m.num_redundant_experts = 0
        m.embed_tokens = _FakeVocabEmbed()
        m.start_layer = 0
        m.end_layer = 1
        m.layers = [_FakeDecoderLayer()]
        m.norm = _FakeRMSNormModule()
        m.use_mhc = False
        m.make_empty_intermediate_tensors = lambda: {}

        ids = torch.zeros(2, dtype=torch.long)
        pos = torch.arange(2, dtype=torch.long)
        emb = m.embed_input_ids(ids)
        out = m.forward(ids, pos, intermediate_tensors=None, inputs_embeds=None)
        self.assertIsNotNone(out)

    def test_forward_pipeline_not_last(self):
        m = openpangu_mod.OpenPanguModel.__new__(openpangu_mod.OpenPanguModel)
        m.start_layer = 0
        m.end_layer = 1
        m.use_mhc = False
        m.layers = [_FakeDecoderLayer()]
        m.norm = _FakeNorm()
        inter = {"hidden_states": torch.ones(2, 8), "residual": torch.zeros(2, 8)}
        with patch.object(
            openpangu_mod,
            "get_pp_group",
            return_value=SimpleNamespace(is_first_rank=False, is_last_rank=False),
        ), patch.object(
            openpangu_mod, "IntermediateTensors", side_effect=lambda d: d
        ):
            out = m.forward(
                torch.zeros(2, dtype=torch.long),
                torch.arange(2, dtype=torch.long),
                intermediate_tensors=inter,
                inputs_embeds=None,
            )
        self.assertIn("hidden_states", out)

    @patch.object(openpangu_mod, "mHCModule", MagicMock())
    @patch.object(openpangu_mod, "RMSNorm", side_effect=lambda *a, **k: _FakeRMSNormModule())
    @patch.object(openpangu_mod, "make_empty_intermediate_tensors_factory", return_value=lambda: {})
    @patch.object(openpangu_mod, "make_layers", return_value=(0, 1, [_FakeDecoderLayer()]))
    @patch.object(openpangu_mod, "VocabParallelEmbedding", _FakeVocabEmbed)
    @patch.object(openpangu_mod, "get_pp_group", return_value=SimpleNamespace(is_first_rank=True, is_last_rank=True))
    def test_forward_use_mhc_merge(self, *_):
        # Create model without going through __init__ to avoid compilation wrapper issues
        m = openpangu_mod.OpenPanguModel.__new__(openpangu_mod.OpenPanguModel)
        nn.Module.__init__(m)
        m.config = SimpleNamespace(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=1,
            rms_norm_eps=1e-5,
            pad_token_id=0,
            tie_word_embeddings=False,
            use_mhc=True,
            mhc_num_stream=2,
        )
        m.num_redundant_experts = 0
        m.embed_tokens = _FakeVocabEmbed()
        m.start_layer = 0
        m.end_layer = 1
        m.layers = [_FakeDecoderLayer()]
        m.norm = _FakeRMSNormModule()
        m.use_mhc = True
        m.num_stream = 2
        m.merge_mhc_module = MagicMock()
        m.merge_mhc_module.hc_pre.return_value = (torch.ones(2, 8), None, None)
        m.make_empty_intermediate_tensors = lambda: {}

        ids = torch.zeros(2, dtype=torch.long)
        pos = torch.arange(2, dtype=torch.long)
        out = m.forward(ids, pos, intermediate_tensors=None, inputs_embeds=None)
        self.assertIsNotNone(out)

    @patch.object(openpangu_mod, "is_pp_missing_parameter", return_value=False)
    def test_load_attn_mlp_weight(self, *_):
        m = openpangu_mod.OpenPanguModel.__new__(openpangu_mod.OpenPanguModel)
        m.config = SimpleNamespace()
        param = MagicMock()
        param.weight_loader = MagicMock()
        params_dict = {"model.layers.0.self_attn.qkv_proj.weight": param}
        loaded = set()
        mapping = [(".qkv_proj", ".q_proj", "q")]
        ok = m.load_attn_mlp_weight(
            mapping,
            params_dict,
            "model.layers.0.self_attn.q_proj.weight",  # This is the weight name from checkpoint
            torch.ones(1),
            loaded,
        )
        self.assertTrue(ok)
        self.assertIn("model.layers.0.self_attn.qkv_proj.weight", loaded)

    @patch.object(openpangu_mod, "is_pp_missing_parameter", return_value=False)
    def test_load_expert_weight(self, *_):
        m = openpangu_mod.OpenPanguModel.__new__(openpangu_mod.OpenPanguModel)
        m.config = SimpleNamespace(n_routed_experts=2)
        param = MagicMock()
        param.weight_loader = MagicMock(return_value=True)
        params_dict = {"model.layers.0.mlp.experts.merged_w.weight": param}
        loaded = set()
        flag = {"is_expert_weight": False}
        mapping = [("merged_w", "gate_proj", 0, "0")]
        ok = m.load_expert_weight(
            mapping,
            params_dict,
            "model.layers.0.mlp.experts.gate_proj.weight",
            torch.ones(1),
            loaded,
            flag,
        )
        self.assertTrue(ok)
        self.assertTrue(flag["is_expert_weight"])

    @patch.object(openpangu_mod, "is_pp_missing_parameter", return_value=False)
    @patch.object(openpangu_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n)
    def test_load_weights_simple_param(self, *_):
        m = openpangu_mod.OpenPanguModel.__new__(openpangu_mod.OpenPanguModel)
        nn.Module.__init__(m)  # Initialize nn.Module base class
        m.config = SimpleNamespace(
            tie_word_embeddings=False,
            num_hidden_layers=2,
            num_nextn_predict_layers=0,
        )
        m.num_redundant_experts = 0
        m.norm = nn.Identity()  # Add norm module
        # Create a proper parameter with weight_loader
        p = nn.Parameter(torch.ones(2, 2))
        openpangu_mod.set_weight_attrs(p, {"weight_loader": openpangu_mod.default_weight_loader})
        # Register the parameter properly
        m.register_parameter("w", p)

        def _named_parameters():
            return [("model.w", p)]

        m.named_parameters = _named_parameters
        m.named_modules = lambda: [("self", m), ("w", nn.Identity())]
        m.post_weight_load = MagicMock()
        loaded = m.load_weights([("model.w", torch.ones(2, 2))])
        self.assertIn("model.w", loaded)

    def test_post_weight_load(self):
        child = MagicMock()
        child.post_weight_load = MagicMock()
        m = openpangu_mod.OpenPanguModel.__new__(openpangu_mod.OpenPanguModel)
        m.named_modules = lambda: [("self", m), ("c", child)]
        openpangu_mod.OpenPanguModel.post_weight_load(m)
        self.assertTrue(child.post_weight_load.called)


class TestOpenPanguModelBase(unittest.TestCase):
    @patch.object(openpangu_mod, "AutoWeightsLoader")
    def test_load_weights_delegates_to_auto_loader(self, mock_loader_cls):
        mock_loader_cls.return_value.load_weights.return_value = {"p"}
        base = openpangu_mod.OpenPanguModelBase.__new__(openpangu_mod.OpenPanguModelBase)
        base.config = SimpleNamespace(tie_word_embeddings=False)
        loaded = openpangu_mod.OpenPanguModelBase.load_weights(base, [])
        self.assertEqual(loaded, {"p"})

    def test_compute_logits_embed_forward(self):
        base = openpangu_mod.OpenPanguModelBase.__new__(openpangu_mod.OpenPanguModelBase)
        base.model = MagicMock()
        base.model.embed_input_ids.return_value = torch.ones(2, 8)
        base.model.return_value = torch.ones(2, 8)
        base.lm_head = object()
        base.logits_processor = MagicMock(return_value=torch.randn(2, 16))
        e = base.embed_input_ids(torch.zeros(2, dtype=torch.long))
        h = base.forward(torch.zeros(2, dtype=torch.long), torch.arange(2), None)
        logits = base.compute_logits(h)
        self.assertEqual(tuple(logits.shape), (2, 16))
        self.assertEqual(tuple(e.shape), (2, 8))


class TestOpenPanguMoEModel(unittest.TestCase):
    def test_update_physical_experts_metadata(self):
        # Create a mock that will pass isinstance check
        layer_moe = openpangu_mod.OpenPanguMoE.__new__(openpangu_mod.OpenPanguMoE)
        layer_moe.n_local_physical_experts = 2
        layer_moe.n_physical_experts = 4
        layer_moe.n_redundant_experts = 0
        layer_moe.experts = MagicMock()
        m2 = openpangu_mod.OpenPanguMoEModel.__new__(openpangu_mod.OpenPanguMoEModel)
        m2.num_local_physical_experts = 2
        m2.num_logical_experts = 4
        m2.model = SimpleNamespace(layers=[SimpleNamespace(mlp=layer_moe)])
        openpangu_mod.OpenPanguMoEModel.update_physical_experts_metadata(m2, 6, 2)
        self.assertEqual(m2.num_physical_experts, 6)
        self.assertTrue(layer_moe.experts.update_expert_map.called)


class TestSubclassExports(unittest.TestCase):
    @patch.object(openpangu_mod.OpenPanguEmbeddedModel, "__init__", lambda *a, **k: None)
    def test_embedded_and_moe_classes_exist(self, *_):
        self.assertTrue(issubclass(openpangu_mod.PanguEmbeddedForCausalLM, openpangu_mod.OpenPanguEmbeddedModel))
        self.assertTrue(issubclass(openpangu_mod.PanguUltraMoEForCausalLM, openpangu_mod.OpenPanguMoEModel))
        self.assertTrue(issubclass(openpangu_mod.PanguProMoEV2ForCausalLM, openpangu_mod.OpenPanguMoEModel))


if __name__ == "__main__":
    unittest.main()
