# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for Pangu V2 high-throughout attention backend selection."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention
from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention
from omni_npu.v1.layers.attention.npu_pangu import NPUPanguSparseAttention
from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


def _pp_group():
    return SimpleNamespace(is_first_rank=True, is_last_rank=True)


class TestHighThroughoutHelpers:
    def test_high_throughout_reads_env(self):
        with patch.object(model_mod.envs, "OMNI_PANGU_V2_HIGH_THROUGHOUT", True):
            assert model_mod.high_throughout() is True
        with patch.object(model_mod.envs, "OMNI_PANGU_V2_HIGH_THROUGHOUT", False):
            assert model_mod.high_throughout() is False

    def test_select_attn_cls_defaults_to_npu_pangu(self):
        config = SimpleNamespace(index_topk=2048, dsa_layers=[0, 2])
        with patch.object(model_mod, "high_throughout", return_value=False):
            assert (
                model_mod.select_pangu_v2_attn_cls(config, 0)
                is NPUPanguSparseAttention
            )

    def test_select_attn_cls_uses_dsa_when_index_topk_and_no_dsa_layers(self):
        config = SimpleNamespace(index_topk=2048)
        with patch.object(model_mod, "high_throughout", return_value=True):
            assert (
                model_mod.select_pangu_v2_attn_cls(config, 3)
                is NPUDeepseekSparseAttention
            )

    def test_select_attn_cls_respects_dsa_layers(self):
        config = SimpleNamespace(index_topk=2048, dsa_layers=[0, 2])
        with patch.object(model_mod, "high_throughout", return_value=True):
            assert (
                model_mod.select_pangu_v2_attn_cls(config, 0)
                is NPUDeepseekSparseAttention
            )
            assert (
                model_mod.select_pangu_v2_attn_cls(config, 1)
                is NPUDeepseekMLAAttention
            )

    def test_select_attn_cls_falls_back_to_mla_without_index_topk(self):
        config = SimpleNamespace()
        with patch.object(model_mod, "high_throughout", return_value=True):
            assert (
                model_mod.select_pangu_v2_attn_cls(config, 0)
                is NPUDeepseekMLAAttention
            )


class TestRewriteMomeConvName:
    def test_leaves_non_conv_names_unchanged(self):
        assert model_mod.rewrite_mome_conv_name(
            "model.layers.0.self_attn.q_b_proj.weight"
        ) == "model.layers.0.self_attn.q_b_proj.weight"

    def test_contiguous_kv_maps_to_merge_conv(self):
        with patch.object(
            model_mod.model_extra_config.operator_opt_config,
            "use_noncontiguous_kv",
            False,
        ):
            assert model_mod.rewrite_mome_conv_name(
                "model.layers.0.self_attn.qa_conv.weight"
            ) == "model.layers.0.self_attn.qa_conv.merge_conv.weight"

    def test_noncontiguous_kv_inserts_conv_module(self):
        with patch.object(
            model_mod.model_extra_config.operator_opt_config,
            "use_noncontiguous_kv",
            True,
        ):
            assert model_mod.rewrite_mome_conv_name(
                "model.layers.0.self_attn.qa_conv.weight"
            ) == "model.layers.0.self_attn.conv.qa_conv.weight"


class TestHighThroughoutModelForward:
    def test_model_forward_uses_rotary_emb_when_high_throughout(self):
        class FakeLayer:
            def __init__(self):
                self.seen_cos = None
                self.self_attn = SimpleNamespace(
                    rotary_emb=SimpleNamespace(
                        get_cos_sin=lambda positions: (
                            torch.full((positions.shape[0], 2), 3.0),
                            torch.full((positions.shape[0], 2), 4.0),
                        )
                    )
                )

            def mhc_head(self, hidden_states):
                return hidden_states, None, None, None, None

            def __call__(
                self,
                hidden_states,
                residual,
                h_post,
                h_res,
                cos,
                sin,
                sk_event,
                topk_indices_buffer,
            ):
                self.seen_cos = cos
                return model_mod.OpenPanguV2DecoderLayerOutput(
                    hidden_states,
                    residual,
                    h_post,
                    h_res,
                    sk_event,
                    topk_indices_buffer,
                )

        model = model_mod.OpenPanguV2Model.__new__(model_mod.OpenPanguV2Model)
        torch.nn.Module.__init__(model)
        layer = FakeLayer()
        model.layers = [layer]
        model.start_layer = 0
        model.end_layer = 1
        model.need_tp_padding = False
        model.use_mhc = False
        model.config = SimpleNamespace(index_topk=4)

        def _embed_as_float(input_ids, **_kwargs):
            return input_ids.float().unsqueeze(-1)

        model.embed_tokens = _embed_as_float
        model.cos_cached = torch.zeros(8, 2)
        model.sin_cached = torch.zeros(8, 2)

        with (
            patch.object(model_mod, "get_pp_group", return_value=_pp_group()),
            patch.object(model_mod, "high_throughout", return_value=True),
        ):
            model.forward(
                input_ids=torch.tensor([1, 2, 3]),
                positions=torch.tensor([0, 1, 2]),
                intermediate_tensors=None,
            )

        assert torch.equal(layer.seen_cos, torch.full((3, 2), 3.0))


class TestHighThroughoutDecoderLayerInit:
    def _config(self, **extra):
        cfg = dict(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            kv_lora_rank=8,
            q_lora_rank=8,
            param_sink_number=1,
            first_k_dense_replace=99,
            rope_parameters={"rope_theta": 10000},
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            intermediate_size=32,
            hidden_act="silu",
            index_topk=8,
            dsa_layers=[0],
        )
        cfg.update(extra)
        return SimpleNamespace(**cfg)

    def _vllm_config(self, config):
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            cache_config=SimpleNamespace(),
            quant_config=None,
            parallel_config=SimpleNamespace(),
        )

    def test_layer0_builds_dsa_and_disables_flashcomm2(self, monkeypatch):
        captured = {}

        class FakeDSA:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

        monkeypatch.setattr(model_mod, "high_throughout", lambda: True)
        monkeypatch.setattr(model_mod, "NPUDeepseekSparseAttention", FakeDSA)
        monkeypatch.setattr(model_mod, "OpenPanguV2MLP", MagicMock())
        monkeypatch.setattr(model_mod, "RMSNorm", MagicMock())
        monkeypatch.setattr(model_mod, "_normalize_rope_parameters", lambda *a, **k: None)
        monkeypatch.setattr(
            model_mod.model_extra_config.parall_config, "enable_flashcomm2", True
        )
        config = self._config()
        layer = model_mod.OpenPanguV2DecoderLayer(
            config, "model.layers.0", self._vllm_config(config)
        )
        assert isinstance(layer.self_attn, FakeDSA)
        assert "rope_theta" not in captured["kwargs"]
        assert layer.attn_supports_pre_epilog is False
        assert model_mod.model_extra_config.parall_config.enable_flashcomm2 is False

    def test_non_dsa_layer_builds_mla(self, monkeypatch):
        class FakeMLA:
            pre_epilog_callback = None

            def __init__(self, **_kwargs):
                pass

        monkeypatch.setattr(model_mod, "high_throughout", lambda: True)
        monkeypatch.setattr(model_mod, "NPUDeepseekMLAAttention", FakeMLA)
        monkeypatch.setattr(model_mod, "OpenPanguV2MLP", MagicMock())
        monkeypatch.setattr(model_mod, "RMSNorm", MagicMock())
        monkeypatch.setattr(model_mod, "_normalize_rope_parameters", lambda *a, **k: None)
        monkeypatch.setattr(
            model_mod.model_extra_config.parall_config, "enable_flashcomm2", False
        )
        config = self._config()
        layer = model_mod.OpenPanguV2DecoderLayer(
            config, "model.layers.1", self._vllm_config(config)
        )
        assert isinstance(layer.self_attn, FakeMLA)
        assert layer.attn_supports_pre_epilog is True


class TestHighThroughoutDecoderForward:
    def test_forward_skips_mhc_hook_without_pre_epilog(self):
        layer = model_mod.OpenPanguV2DecoderLayer.__new__(
            model_mod.OpenPanguV2DecoderLayer
        )
        layer.use_mhc = True
        layer.use_post_norm = False
        layer.side_stream = object()
        layer.attn_supports_pre_epilog = False
        layer.layer_idx = 0
        layer.first_k_dense_replace = 99
        layer.hidden_size = 4
        layer.mhc_num_stream = 1
        layer.input_layernorm = lambda x: x
        layer.post_attention_layernorm = lambda x: x
        layer.pre_mlp_layernorm = lambda x: x
        layer.post_mlp_layernorm = lambda x: x
        layer.mlp = lambda x: x
        layer.attn_mhc_module = MagicMock()
        layer.mlp_mhc_module = MagicMock()
        layer._tail_refs = (None, MagicMock(), True)

        class _Attn:
            prefix = "attn"
            pre_epilog_callback = "stale"

            def __call__(self, hs, _cos, _sin, topk):
                return hs, topk

        layer.self_attn = _Attn()
        layer.mhc_sandwich_norm_post_pre = MagicMock(
            side_effect=lambda hs, residual, *a, **k: (hs, residual, None, None, None)
        )
        hs = torch.ones(2, 4)
        out = layer.forward(
            hs,
            torch.ones(2, 4),
            None,
            None,
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            None,
            torch.zeros(2, 1, dtype=torch.int32),
        )
        assert layer.self_attn.pre_epilog_callback == "stale"
        kwargs = layer.mhc_sandwich_norm_post_pre.call_args_list[-1].kwargs
        assert kwargs["defer_side_launch"] is False
        torch.testing.assert_close(out.hidden_states, hs)

    def test_forward_clears_and_installs_pre_epilog_hook(self):
        layer = model_mod.OpenPanguV2DecoderLayer.__new__(
            model_mod.OpenPanguV2DecoderLayer
        )
        layer.use_mhc = True
        layer.use_post_norm = False
        layer.side_stream = object()
        layer.attn_supports_pre_epilog = True
        layer.layer_idx = 0
        layer.first_k_dense_replace = 99
        layer.hidden_size = 4
        layer.mhc_num_stream = 1
        layer.post_attention_layernorm = lambda x: x
        layer.pre_mlp_layernorm = lambda x: x
        layer.post_mlp_layernorm = lambda x: x
        layer.mlp = lambda x: x
        layer.attn_mhc_module = MagicMock()
        layer.mlp_mhc_module = MagicMock()
        layer._tail_refs = (None, MagicMock(), True)
        layer._launch_split_post_res_sinkhorn = MagicMock(
            return_value=("hp", "hr", "ev")
        )

        class _Attn:
            prefix = "attn"
            pre_epilog_callback = "stale"

            def __call__(self, hs, _cos, _sin, topk):
                return hs, topk

        layer.self_attn = _Attn()
        layer.mhc_sandwich_norm_post_pre = MagicMock(
            side_effect=lambda hs, residual, *a, **k: (hs, residual, None, None, None)
        )
        layer.forward(
            torch.ones(2, 4),
            torch.ones(2, 4),
            None,
            None,
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            None,
            torch.zeros(2, 1, dtype=torch.int32),
        )
        layer._launch_split_post_res_sinkhorn.assert_called_once()
        assert layer.self_attn.pre_epilog_callback is None


def _bare_v2_mtp():
    import torch.nn as nn
    import omni_npu.v1.models.pangu.pangu_v2_moe_mtp as mtp_mod

    m = mtp_mod.OpenPanguV2MTP.__new__(mtp_mod.OpenPanguV2MTP)
    nn.Module.__init__(m)
    m.config = SimpleNamespace(
        num_hidden_layers=2, num_nextn_predict_layers=1, n_routed_experts=2,
    )
    m.model = SimpleNamespace(mtp_start_layer_idx=2)
    return m, mtp_mod


class TestHighThroughoutMtpLoadWeights:
    def test_rewrites_mome_conv_name_when_high_throughout(self):
        import torch.nn as nn

        m, mtp_mod = _bare_v2_mtp()
        param = nn.Parameter(torch.zeros(2, 2))
        rewritten = "model.layers.2.mtp_block.self_attn.conv.qa_conv.weight"
        m.named_parameters = lambda: [(rewritten, param)]
        with (
            patch.object(mtp_mod, "high_throughout", return_value=True),
            patch.object(
                mtp_mod, "fused_moe_make_expert_params_mapping", return_value=[]
            ),
            patch.object(
                model_mod.model_extra_config.operator_opt_config,
                "use_noncontiguous_kv",
                True,
            ),
            patch.object(mtp_mod, "default_weight_loader") as loader,
        ):
            loaded = m.load_weights(
                [("model.layers.2.self_attn.qa_conv.weight", torch.ones(2, 2))]
            )
        assert rewritten in loaded
        loader.assert_called_once()

    def test_skips_unknown_param_name(self):
        m, mtp_mod = _bare_v2_mtp()
        m.named_parameters = lambda: []
        with (
            patch.object(mtp_mod, "high_throughout", return_value=False),
            patch.object(
                mtp_mod, "fused_moe_make_expert_params_mapping", return_value=[]
            ),
        ):
            loaded = m.load_weights(
                [("model.layers.2.self_attn.missing.weight", torch.ones(1))]
            )
        assert loaded == set()
