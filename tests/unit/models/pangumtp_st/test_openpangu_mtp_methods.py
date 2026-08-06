# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Coverage completion tests for OpenPanguMTP.

Covers previously uncovered methods:
  - get_spec_layer()          (lines 203-213)
  - _rewrite_spec_layer_name() (lines 341-371)
  - set_shared_weight()       (lines 215-222)
  - load_weights()            (lines 224-332)
  - post_weight_load()        (lines 334-339)
  - insert_conv_before()      (lines 375-380)
  - wrapped_layers forward path (line 142)
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

# ==============================================================================
# Helpers: create OpenPanguMTP instances without full __init__
# ==============================================================================

def _make_minimal_mtp():
    """Create OpenPanguMTP with __new__ + manual setup (same pattern as test_deepseek_mtp.py)."""
    import omni_npu.v1.models.pangu.pangu_ultra_moe_mtp as mtp_mod

    m = mtp_mod.OpenPanguMTP.__new__(mtp_mod.OpenPanguMTP)
    nn.Module.__init__(m)  # sets up _modules etc.
    m.config = SimpleNamespace(
        num_hidden_layers=2,
        num_nextn_predict_layers=2,
        n_routed_experts=2,       # must be int for FusedMoE.make_expert_params_mapping
        n_shared_experts=0,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=256,
        rms_norm_eps=1e-6,
    )
    # Create a minimal model.predictor for methods that need it
    m.model = SimpleNamespace(
        mtp_start_layer_idx=2,
        embed_tokens=nn.Embedding(256, 64),
        layers={"2": SimpleNamespace(shared_head=SimpleNamespace(head=nn.Linear(64, 256)))},
    )
    return m, mtp_mod


# ==============================================================================
# compute_logits dtype alignment
# ==============================================================================

class TestComputeLogitsDtype:
    def test_compute_logits_casts_hidden_states_to_head_dtype(self):
        import omni_npu.v1.models.pangu.pangu_ultra_moe_mtp as mtp_mod

        pred = mtp_mod.OpenPanguMultiTokenPredictor.__new__(
            mtp_mod.OpenPanguMultiTokenPredictor
        )
        pred.num_mtp_layers = 1
        pred.mtp_start_layer_idx = 2
        pred.logits_processor = MagicMock(return_value=torch.randn(2, 256))

        shared_head = MagicMock()
        shared_head.head.weight = SimpleNamespace(dtype=torch.float32)
        shared_head.return_value = torch.randn(2, 64, dtype=torch.bfloat16)
        pred.layers = {"2": SimpleNamespace(shared_head=shared_head)}

        hidden_states = torch.randn(2, 64, dtype=torch.bfloat16)
        pred.compute_logits(hidden_states, spec_step_idx=0)

        _, hidden_arg = pred.logits_processor.call_args[0]
        assert hidden_arg.dtype == torch.float32


# ==============================================================================
# get_spec_layer
# ==============================================================================

class TestGetSpecLayer:
    def test_normal_mtp_layer(self):
        m, _ = _make_minimal_mtp()
        assert m.get_spec_layer("model.layers.2.mlp.weight") == 2
        assert m.get_spec_layer("model.layers.3.self_attn.weight") == 3

    def test_non_mtp_layer_returns_none(self):
        m, _ = _make_minimal_mtp()
        assert m.get_spec_layer("model.layers.1.weight") is None
        assert m.get_spec_layer("model.embed_tokens.weight") is None

    def test_no_layers_in_name(self):
        m, _ = _make_minimal_mtp()
        assert m.get_spec_layer("model.norm.weight") is None

    def test_no_num_nextn(self):
        m, _ = _make_minimal_mtp()
        m.config.num_nextn_predict_layers = 0
        # hasattr returns True but condition check fails due to > 0
        assert m.get_spec_layer("model.layers.2.weight") is None

    def test_no_num_nextn_attr(self):
        m, _ = _make_minimal_mtp()
        del m.config.num_nextn_predict_layers
        assert m.get_spec_layer("model.layers.2.weight") is None

    def test_out_of_range_mtp_idx(self):
        m, _ = _make_minimal_mtp()
        # layer 10 -> mtp_idx=8 >= num_nextn_predict_layers=2
        assert m.get_spec_layer("model.layers.10.weight") is None


# ==============================================================================
# _rewrite_spec_layer_name
# ==============================================================================

class TestRewriteSpecLayerName:
    def test_transformer_block_adds_mtp_block(self):
        m, _ = _make_minimal_mtp()
        out = m._rewrite_spec_layer_name(2, "model.layers.2.self_attn.q_proj.weight")
        assert out == "model.layers.2.mtp_block.self_attn.q_proj.weight"

    def test_spec_layer_weight_keeps_no_mtp_block(self):
        m, _ = _make_minimal_mtp()
        out = m._rewrite_spec_layer_name(2, "model.layers.2.enorm.weight")
        assert out == "model.layers.2.enorm.weight"

    def test_embed_tokens_moves_to_top_level(self):
        m, _ = _make_minimal_mtp()
        out = m._rewrite_spec_layer_name(2, "model.layers.2.embed_tokens.weight")
        assert out == "model.embed_tokens.weight"

    def test_eh_proj_kept_as_spec(self):
        m, _ = _make_minimal_mtp()
        out = m._rewrite_spec_layer_name(2, "model.layers.2.eh_proj.weight")
        assert out == "model.layers.2.eh_proj.weight"

    def test_hnorm_kept_as_spec(self):
        m, _ = _make_minimal_mtp()
        out = m._rewrite_spec_layer_name(2, "model.layers.2.hnorm.weight")
        assert out == "model.layers.2.hnorm.weight"

    def test_shared_head_kept_as_spec(self):
        m, _ = _make_minimal_mtp()
        out = m._rewrite_spec_layer_name(2, "model.layers.2.shared_head.weight")
        assert out == "model.layers.2.shared_head.weight"

    def test_non_spec_layer(self):
        m, _ = _make_minimal_mtp()
        out = m._rewrite_spec_layer_name(3, "model.layers.3.mlp.down_proj.weight")
        assert out == "model.layers.3.mtp_block.mlp.down_proj.weight"


# ==============================================================================
# set_shared_weight
# ==============================================================================

class TestSetSharedWeight:
    def test_shares_embed_tokens(self):
        m, _ = _make_minimal_mtp()
        target = SimpleNamespace(embed_tokens=nn.Embedding(256, 64))
        original = m.model.embed_tokens
        m.set_shared_weight(target)
        assert m.model.embed_tokens is target.embed_tokens
        assert m.model.embed_tokens is not original

    def test_shares_lm_head(self):
        m, _ = _make_minimal_mtp()
        new_head = nn.Linear(64, 256)
        target = SimpleNamespace(lm_head=new_head)
        old_head = m.model.layers["2"].shared_head.head
        m.set_shared_weight(target)
        assert m.model.layers["2"].shared_head.head is new_head
        assert m.model.layers["2"].shared_head.head is not old_head

    def test_no_embed_tokens_on_target(self):
        m, _ = _make_minimal_mtp()
        target = SimpleNamespace()  # no embed_tokens attr
        original = m.model.embed_tokens
        m.set_shared_weight(target)
        assert m.model.embed_tokens is original  # unchanged

    def test_no_lm_head_on_target(self):
        m, _ = _make_minimal_mtp()
        target = SimpleNamespace(embed_tokens=nn.Embedding(256, 64))
        old_head = m.model.layers["2"].shared_head.head
        m.set_shared_weight(target)
        assert m.model.layers["2"].shared_head.head is old_head  # unchanged


# ==============================================================================
# load_weights — follows test_deepseek_mtp.py pattern
# ==============================================================================

class TestLoadWeights:
    def test_skips_rotary_inv_freq(self):
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        m.named_parameters = lambda: []
        with patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n), \
             patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]):
            loaded = m.load_weights([("model.layers.2.rotary_emb.inv_freq", torch.ones(1))])
        assert len(loaded) == 0

    def test_skips_non_spec_layer(self):
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        m.named_parameters = lambda: []
        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n):
            # get_spec_layer returns None for non-MTP layers
            loaded = m.load_weights([("model.layers.1.self_attn.weight", torch.ones(1))])
        assert len(loaded) == 0

    def test_loads_normal_weight(self):
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2

        param = nn.Parameter(torch.zeros(4, 4))
        m.named_parameters = lambda: [("model.layers.2.mtp_block.self_attn.q_proj.weight", param)]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n), \
             patch.object(mtp_mod, "default_weight_loader") as mock_loader:
            loaded = m.load_weights([("model.layers.2.self_attn.q_proj.weight", torch.ones(4, 4))])
        assert "model.layers.2.mtp_block.self_attn.q_proj.weight" in loaded
        mock_loader.assert_called_once()

    def test_loads_stacked_gate_up_proj(self):
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2

        param = nn.Parameter(torch.zeros(4, 4))
        param.weight_loader = lambda p, w, shard_id: p.data.copy_(w)
        m.named_parameters = lambda: [("model.layers.2.mtp_block.mlp.gate_up_proj.weight", param)]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n):
            loaded = m.load_weights([("model.layers.2.mlp.gate_proj.weight", torch.ones(4, 4))])
        assert "model.layers.2.mtp_block.mlp.gate_up_proj.weight" in loaded

    def test_shared_embed_only_loaded_for_first_spec_layer(self):
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2

        param = nn.Parameter(torch.zeros(8, 8))
        param.weight_loader = lambda p, w: p.data.copy_(w)
        m.named_parameters = lambda: [("model.embed_tokens.weight", param)]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n):
            loaded = m.load_weights([
                ("model.layers.2.embed_tokens.weight", torch.ones(8, 8)),
                ("model.layers.3.embed_tokens.weight", torch.ones(8, 8)),
            ])
        assert "model.embed_tokens.weight" in loaded

    def test_mlp_experts_skip_in_stacked_params(self):
        """Line 260: skip mlp.experts when name not in params_dict."""
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        m.config.n_routed_experts = 2

        param = nn.Parameter(torch.zeros(4, 4))
        m.named_parameters = lambda: [("model.layers.2.mtp_block.mlp.down_proj.weight", param)]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n):
            loaded = m.load_weights([("model.layers.2.mlp.down_proj.weight", torch.ones(4, 4))])
        assert "model.layers.2.mtp_block.mlp.down_proj.weight" in loaded

    def test_conv_insert_before(self):
        """Line 314: insert_conv_before when use_noncontiguous_kv=True."""
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        param = nn.Parameter(torch.zeros(4, 4))
        # After rewrite: .mtp_block.compresskv_conv.weight
        # With insert_conv_before: .mtp_block.conv.compresskv_conv.weight
        m.named_parameters = lambda: [
            ("model.layers.2.mtp_block.conv.compresskv_conv.weight", param),
        ]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n), \
             patch.object(mtp_mod.model_extra_config.operator_opt_config, "use_noncontiguous_kv", True), \
             patch.object(mtp_mod.model_extra_config.operator_opt_config, "merge_q_kv_conv", False):
            loaded = m.load_weights([("model.layers.2.compresskv_conv.weight", torch.ones(4, 4))])
        assert any("compresskv_conv" in k for k in loaded)

    def test_conv_merge_q_kv(self):
        """Lines 318-321: merge_q_kv_conv path for qa_conv."""
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        param = nn.Parameter(torch.zeros(4, 4))
        # _conv → _conv.merge_conv: qa_conv.merge_conv (NOT merge_conv.merge_conv — name stays)
        m.named_parameters = lambda: [
            ("model.layers.2.mtp_block.self_attn.qa_conv.merge_conv.weight", param),
        ]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n), \
             patch.object(mtp_mod.model_extra_config.operator_opt_config, "use_noncontiguous_kv", False), \
             patch.object(mtp_mod.model_extra_config.operator_opt_config, "merge_q_kv_conv", True):
            loaded = m.load_weights([("model.layers.2.self_attn.qa_conv.weight", torch.ones(4, 4))])
        assert any("merge_conv" in k for k in loaded)

    def test_bias_skip_in_stacked(self):
        """Line 274: skip .bias when not in params_dict."""
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        param = nn.Parameter(torch.zeros(4, 4))
        m.named_parameters = lambda: [("model.layers.2.mtp_block.mlp.down_proj.weight", param)]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n):
            loaded = m.load_weights([("model.layers.2.mlp.down_proj.bias", torch.ones(4))])
        # bias not in params_dict → falls through to else → name.endswith('.bias') → skip
        assert len(loaded) == 0

    def test_expert_weight_mapping(self):
        """Lines 282-296: expert_params_mapping path."""
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        m.config.n_routed_experts = 2

        param = nn.Parameter(torch.zeros(4, 4))
        param.weight_loader = lambda p, w, name, shard_id=0, expert_id=0: p.data.copy_(w)
        m.named_parameters = lambda: [
            ("model.layers.2.mtp_block.mlp.experts.0.gate_up_proj.weight", param),
        ]

        expert_map = [("gate_up_proj", "gate_proj", 0, 0)]
        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=expert_map), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n):
            loaded = m.load_weights([("model.layers.2.mlp.experts.0.gate_proj.weight", torch.ones(4, 4))])
        assert "model.layers.2.mtp_block.mlp.experts.0.gate_up_proj.weight" in loaded

    def test_bias_skip_in_expert_else(self):
        """Line 300: skip .bias in else branch when not in params_dict."""
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        param = nn.Parameter(torch.zeros(4))
        m.named_parameters = lambda: [("model.layers.2.mtp_block.self_attn.q_proj.weight", param)]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n):
            loaded = m.load_weights([("model.layers.2.self_attn.q_proj.bias", torch.ones(4))])
        # .bias not in params → skip
        assert len(loaded) == 0

    def test_e_score_correction_bias_rename(self):
        """Line 309: e_score_correction_bias → gate.e_score_correction_bias."""
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2
        param = nn.Parameter(torch.zeros(2))
        # After _rewrite_spec_layer_name: .mtp_block. is added
        # Then name.replace("e_score_correction_bias", "gate.e_score_correction_bias")
        m.named_parameters = lambda: [
            ("model.layers.2.mtp_block.mlp.gate.e_score_correction_bias", param),
        ]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n):
            loaded = m.load_weights([("model.layers.2.mlp.e_score_correction_bias", torch.ones(2))])
        assert "model.layers.2.mtp_block.mlp.gate.e_score_correction_bias" in loaded

    def test_conv_name_merge_conv(self):
        """Lines 313-316: _conv → _conv.merge_conv when use_noncontiguous_kv=False."""
        m, mtp_mod = _make_minimal_mtp()
        m.model.mtp_start_layer_idx = 2

        param = nn.Parameter(torch.zeros(4, 4))
        # After _rewrite_spec_layer_name: model.layers.2.compresskv_conv.weight
        # → model.layers.2.mtp_block.compresskv_conv.weight (compresskv_conv not in spec list)
        # Then _conv handling: replace("_conv", "_conv.merge_conv")
        # → model.layers.2.mtp_block.compresskv_conv.merge_conv.weight
        m.named_parameters = lambda: [
            ("model.layers.2.mtp_block.compresskv_conv.merge_conv.weight", param),
        ]

        with patch.object(mtp_mod.FusedMoE, "make_expert_params_mapping", return_value=[]), \
             patch.object(mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n), \
             patch.object(mtp_mod.model_extra_config.operator_opt_config, "use_noncontiguous_kv", False), \
             patch.object(mtp_mod.model_extra_config.operator_opt_config, "merge_q_kv_conv", False):
            loaded = m.load_weights([("model.layers.2.compresskv_conv.weight", torch.ones(4, 4))])
        assert any("compresskv_conv" in k for k in loaded)


# ==============================================================================
# post_weight_load
# ==============================================================================

class TestPostWeightLoad:
    def test_calls_post_weight_load_on_submodules(self):
        m, mtp_mod = _make_minimal_mtp()

        called = []
        class ModWithPost(nn.Module):
            def post_weight_load(self):
                called.append(1)

        sub = SimpleNamespace(post_weight_load=ModWithPost().post_weight_load)
        m.model.layers["2"] = sub

        m.named_modules = lambda: [("", m), ("model.layers.2", sub)]
        m.post_weight_load()
        assert len(called) == 1, "post_weight_load should be called on submodules"

    def test_skips_self(self):
        m, mtp_mod = _make_minimal_mtp()
        m.named_modules = lambda: [("", m)]
        # Should not recurse — self is skipped
        m.post_weight_load()  # no error = pass


# ==============================================================================
# insert_conv_before
# ==============================================================================

class TestInsertConvBefore:
    def test_inserts_conv_before_conv_suffix(self):
        from omni_npu.v1.models.pangu.pangu_ultra_moe_mtp import insert_conv_before
        # "model.layers.2.qa_conv.weight" -> "model.layers.2.conv.qa_conv.weight"
        result = insert_conv_before("model.layers.2.qa_conv.weight")
        assert "conv.qa_conv" in result
        assert result == "model.layers.2.conv.qa_conv.weight"

    def test_compresskv_conv(self):
        from omni_npu.v1.models.pangu.pangu_ultra_moe_mtp import insert_conv_before
        result = insert_conv_before("model.layers.3.compresskv_conv.weight")
        assert "conv.compresskv_conv" in result

    def test_o_conv(self):
        from omni_npu.v1.models.pangu.pangu_ultra_moe_mtp import insert_conv_before
        result = insert_conv_before("model.layers.2.o_conv.bias")
        assert "conv.o_conv" in result


# ==============================================================================
# wrapped_layers forward path
# ==============================================================================

class TestWrappedLayers:
    def test_wrapped_layers_used_when_set(self, mtp_predictor, sample_batch, minimal_config):
        """Line 142: wrapped_layers path in OpenPanguMultiTokenPredictor.forward."""
        pred = mtp_predictor
        # Set wrapped_layers to a copy of layers dict
        pred.wrapped_layers = dict(pred.layers)

        B = sample_batch["input_ids"].shape[0]
        out = pred(
            input_ids=sample_batch["input_ids"],
            positions=sample_batch["positions"],
            previous_hidden_states=sample_batch["hidden_states"],
            inputs_embeds=sample_batch["inputs_embeds"],
            spec_step_idx=0,
        )
        assert out.shape == (B, minimal_config.hidden_size)

        # Clean up
        pred.wrapped_layers = None
