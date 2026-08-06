# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omni.layers.npu_rms_norm import NPUMiniMaxText01RMSNormTP, NPURMSNorm


@pytest.mark.usefixtures("default_vllm_config")
class TestNPURMSNorm(unittest.TestCase):
    def setUp(self):
        """initialize the test environment"""
        self.hidden_size = 1024
        self.batch_size = 4
        self.tp_size = 8

        self.mock_tp_group = MagicMock()
        def mock_all_gather_func(tensor, dim=0):
            return torch.cat([tensor] * self.tp_size, dim=dim)
        self.mock_tp_group.all_gather = MagicMock(side_effect=mock_all_gather_func)
        self.tp_group_patch = patch("vllm.distributed.get_tp_group", return_value=self.mock_tp_group)
        self.tp_group_patch.start()

        self.mock_config = MagicMock()
        self.mock_config.operator_opt_config = MagicMock()
        self.mock_config.operator_opt_config.omni_disable_npu_add_rms_norm = False
        self.config_patch = patch("omni.model_config.config_loader.loader.model_extra_config", self.mock_config)
        self.config_patch_loader = patch("omni.layers.npu_rms_norm.model_extra_config", self.mock_config)
        self.config_patch.start()
        self.config_patch_loader.start()

        self.npu_add_rms_norm_mock = MagicMock(side_effect=lambda x, r, w, e: (x.clone(), None, r.clone()))
        self.npu_rms_norm_mock = MagicMock(side_effect=lambda x, w, e: (x.clone(),))
        self.npu_dynamic_quant_mock = MagicMock(side_effect=lambda x: (x.clone().to(torch.int8), torch.ones(x.shape[0])))

        self.patch_add = patch("torch_npu.npu_add_rms_norm", new=self.npu_add_rms_norm_mock)
        self.patch_rms = patch("torch_npu.npu_rms_norm", new=self.npu_rms_norm_mock)
        self.patch_quant = patch("torch_npu.npu_dynamic_quant", new=self.npu_dynamic_quant_mock)

        self.patch_add.start()
        self.patch_rms.start()
        self.patch_quant.start()

        import vllm.distributed.parallel_state as ps
        ps._TP = self.mock_tp_group
        self.rms_norm = NPURMSNorm(self.hidden_size, eps=1e-6)

    def tearDown(self):
        """clear test environment"""
        self.tp_group_patch.stop()
        self.config_patch.stop()
        self.config_patch_loader.stop()
        self.patch_add.stop()
        self.patch_rms.stop()
        self.patch_quant.stop()

        import vllm.distributed.parallel_state as ps
        ps._TP = None

    def test_forward_oot_no_residual_basic(self):
        """Case: residual=None -> Only RMSNorm"""
        x = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=None, quant_symbol=True, y_transform="AG")

        self.npu_rms_norm_mock.assert_called_once()
        self.npu_add_rms_norm_mock.assert_not_called()
        self.mock_tp_group.all_gather.assert_not_called()
        self.npu_dynamic_quant_mock.assert_not_called()

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_with_residual_without_AG_without_quant(self):
        """Case: residual=Tensor, No AG, No Quant -> AddRMSNorm"""
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=res, y_transform="", quant_symbol=False)

        self.npu_add_rms_norm_mock.assert_called_once()
        self.mock_tp_group.all_gather.assert_not_called()
        self.npu_dynamic_quant_mock.assert_not_called()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        normed_x, new_res = result
        self.assertEqual(normed_x.shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_with_residual_with_AG_without_quant(self):
        """Case: residual=Tensor, AG -> AddRMSNorm + AllGather"""
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=res, y_transform="AG", quant_symbol=False)

        self.npu_add_rms_norm_mock.assert_called_once()
        self.mock_tp_group.all_gather.assert_called_once()
        self.npu_dynamic_quant_mock.assert_not_called()

        normed_x, new_res = result
        expected_rows = self.batch_size * self.tp_size
        self.assertEqual(normed_x.shape, (expected_rows, self.hidden_size))
        self.assertEqual(new_res.shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_with_residual_without_AG_with_quant(self):
        """Case: residual=Tensor, Quant -> AddRMSNorm + DynamicQuant"""
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=res, y_transform="", quant_symbol=True)

        self.npu_add_rms_norm_mock.assert_called_once()
        self.mock_tp_group.all_gather.assert_not_called()
        self.npu_dynamic_quant_mock.assert_called_once()

        output_dict, new_res = result
        self.assertIsInstance(output_dict, dict)
        self.assertIn("x_int8", output_dict)
        self.assertIn("pertoken_scale", output_dict)
        self.assertEqual(output_dict["x_int8"].shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_with_residual_with_AG_with_quant(self):
        """Case: residual=Tensor, AG + Quant -> AddRMSNorm + AllGather + DynamicQuant"""
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=res, y_transform="AG", quant_symbol=True)

        self.npu_add_rms_norm_mock.assert_called_once()
        self.mock_tp_group.all_gather.assert_called_once()
        self.npu_dynamic_quant_mock.assert_called_once()

        quant_args = self.npu_dynamic_quant_mock.call_args[0][0]
        expected_rows = self.batch_size * self.tp_size
        self.assertEqual(quant_args.shape[0], expected_rows)
        output_dict, new_res = result
        self.assertEqual(output_dict["x_int8"].shape, (expected_rows, self.hidden_size))
        self.assertEqual(new_res.shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_disabled_with_residual(self):
        """Case: disable fused add+rms path -> fallback to x + residual + rms_norm."""
        self.mock_config.operator_opt_config.omni_disable_npu_add_rms_norm = True
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        normed_x, new_res = self.rms_norm(
            x, residual=res, quant_symbol=True, y_transform="AG"
        )

        self.npu_add_rms_norm_mock.assert_not_called()
        self.mock_tp_group.all_gather.assert_not_called()
        self.npu_dynamic_quant_mock.assert_not_called()
        self.npu_rms_norm_mock.assert_called_once()
        fallback_input = self.npu_rms_norm_mock.call_args[0][0]
        expected = x + res
        self.assertTrue(torch.equal(fallback_input, expected))
        self.assertTrue(torch.equal(normed_x, expected))
        self.assertTrue(torch.equal(new_res, expected))

    def test_forward_oot_disabled_without_residual(self):
        """Case: disable fused add+rms path -> fallback uses plain rms_norm."""
        self.mock_config.operator_opt_config.omni_disable_npu_add_rms_norm = True
        x = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=None, quant_symbol=True, y_transform="AG")

        self.npu_add_rms_norm_mock.assert_not_called()
        self.mock_tp_group.all_gather.assert_not_called()
        self.npu_dynamic_quant_mock.assert_not_called()
        self.npu_rms_norm_mock.assert_called_once()
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (self.batch_size, self.hidden_size))


@pytest.mark.usefixtures("default_vllm_config")
class TestNPUMiniMaxText01RMSNormTP:
    @patch("omni.layers.npu_rms_norm.get_tensor_model_parallel_world_size", return_value=4)
    def test_init_sets_full_weight_and_loader(self, mock_world):
        norm = NPUMiniMaxText01RMSNormTP(16, eps=1e-5)

        assert norm.weight.shape == (16,)
        assert norm.tp_world == 4
        assert norm.variance_epsilon == pytest.approx(1e-5)

        loaded = torch.arange(16, dtype=torch.float32)
        norm.weight.weight_loader(norm.weight, loaded)
        assert torch.equal(norm.weight.data, loaded)

    def test_local_rms_sq_from_rstd_keeps_singleton_dim(self):
        rstd = torch.tensor([[0.5], [0.25]], dtype=torch.float32)

        rms_sq = NPUMiniMaxText01RMSNormTP.local_rms_sq_from_rstd(rstd)

        expected = torch.tensor([[4.0], [16.0]], dtype=torch.float32)
        assert torch.equal(rms_sq, expected)

    def test_local_rms_sq_from_rstd_reduces_last_dim(self):
        rstd = torch.tensor([[0.5, 1.0], [0.25, 0.5]], dtype=torch.float32)

        rms_sq = NPUMiniMaxText01RMSNormTP.local_rms_sq_from_rstd(rstd)

        expected = torch.tensor(
            [[(1 / 0.75) ** 2], [(1 / 0.375) ** 2]], dtype=torch.float32
        )
        assert torch.allclose(rms_sq, expected)

    @patch("omni.layers.npu_rms_norm.tensor_model_parallel_all_reduce")
    @patch("omni.layers.npu_rms_norm.torch_npu.npu_rms_norm")
    def test_npu_tp_rms_norm_qk_skips_rescale_when_tp_world_is_one(
        self, mock_rms_norm, mock_all_reduce
    ):
        q = torch.ones(2, 4, dtype=torch.float32)
        k = torch.ones(2, 4, dtype=torch.float32) * 2
        q_normed = q + 1
        k_normed = k + 2
        mock_rms_norm.side_effect = [
            (q_normed.clone(), torch.full((2, 1), 0.5)),
            (k_normed.clone(), torch.full((2, 1), 0.25)),
        ]

        out_q, out_k = NPUMiniMaxText01RMSNormTP.npu_tp_rms_norm_qk(
            q,
            torch.ones(4),
            k,
            torch.ones(4),
            1e-6,
            tp_world=1,
        )

        assert torch.equal(out_q, q_normed)
        assert torch.equal(out_k, k_normed)
        mock_all_reduce.assert_not_called()

    @patch("omni.layers.npu_rms_norm.tensor_model_parallel_all_reduce")
    @patch("omni.layers.npu_rms_norm.torch_npu.npu_rms_norm")
    def test_npu_tp_rms_norm_qk_rescales_with_global_rms(
        self, mock_rms_norm, mock_all_reduce
    ):
        q = torch.ones(1, 2, dtype=torch.float32)
        k = torch.ones(1, 2, dtype=torch.float32) * 2
        q_normed = torch.tensor([[3.0, 6.0]], dtype=torch.float32)
        k_normed = torch.tensor([[4.0, 8.0]], dtype=torch.float32)
        mock_rms_norm.side_effect = [
            (q_normed.clone(), torch.tensor([[0.5]], dtype=torch.float32)),
            (k_normed.clone(), torch.tensor([[0.25]], dtype=torch.float32)),
        ]
        mock_all_reduce.return_value = torch.tensor([[18.0, 32.0]], dtype=torch.float32)

        out_q, out_k = NPUMiniMaxText01RMSNormTP.npu_tp_rms_norm_qk(
            q,
            torch.ones(2),
            k,
            torch.ones(2),
            1e-6,
            tp_world=2,
        )

        expected_q = q_normed * (2.0 / 3.0)
        expected_k = k_normed
        assert torch.allclose(out_q, expected_q)
        assert torch.allclose(out_k, expected_k)
        mock_all_reduce.assert_called_once()

    @patch("omni.layers.npu_rms_norm.get_tensor_model_parallel_world_size", return_value=4)
    @patch("omni.layers.npu_rms_norm.get_tensor_model_parallel_rank", return_value=1)
    def test_get_shard_index(self, mock_rank, mock_world):
        shard_index = NPUMiniMaxText01RMSNormTP.get_shard_index(torch.ones(16))

        assert shard_index == slice(4, 8)

    @patch("omni.layers.npu_rms_norm.get_tensor_model_parallel_world_size", return_value=2)
    @patch("omni.layers.npu_rms_norm.get_tensor_model_parallel_rank", return_value=1)
    @patch.object(NPUMiniMaxText01RMSNormTP, "npu_tp_rms_norm_qk")
    def test_forward_qk_prefill_uses_local_weight_shards(
        self, mock_norm_qk, mock_rank, mock_world
    ):
        q_norm = SimpleNamespace(
            weight=torch.arange(8, dtype=torch.float32),
            variance_epsilon=1e-6,
            tp_world=2,
        )
        k_norm = SimpleNamespace(
            weight=torch.arange(8, 16, dtype=torch.float32),
            variance_epsilon=1e-6,
            tp_world=2,
        )
        q = torch.randn(2, 4)
        k = torch.randn(2, 4)

        NPUMiniMaxText01RMSNormTP.forward_qk_prefill(q_norm, k_norm, q, k)

        call_args = mock_norm_qk.call_args[0]
        assert torch.equal(call_args[0], q)
        assert torch.equal(call_args[1], q_norm.weight[4:8])
        assert torch.equal(call_args[2], k)
        assert torch.equal(call_args[3], k_norm.weight[4:8])
        assert call_args[4] == q_norm.variance_epsilon
        assert call_args[5] == q_norm.tp_world

    @patch("omni.layers.npu_rms_norm.get_tensor_model_parallel_world_size", return_value=2)
    @patch("omni.layers.npu_rms_norm.get_tp_group")
    def test_all_gather_qk_heads_gathers_across_tp_ranks(
        self, mock_get_tp_group, mock_world
    ):
        tp_group = MagicMock()
        tp_group.all_gather.side_effect = lambda tensor, dim=0: torch.cat(
            [tensor, tensor + 1000], dim=dim
        )
        mock_get_tp_group.return_value = tp_group

        q = torch.arange(8, dtype=torch.float32).view(2, 4)
        k = torch.arange(4, dtype=torch.float32).view(2, 2) + 100

        q_ag, k_ag = NPUMiniMaxText01RMSNormTP.all_gather_qk_heads(q, k, head_dim=2)

        assert q_ag.shape == (2, 2, 2, 2)
        assert k_ag.shape == (2, 2, 1, 2)
        assert torch.equal(q_ag[:, 0], q.view(2, 2, 2))
        assert torch.equal(q_ag[:, 1], q.view(2, 2, 2) + 1000)
        assert torch.equal(k_ag[:, 0], k.view(2, 1, 2))
        assert torch.equal(k_ag[:, 1], k.view(2, 1, 2) + 1000)

    @patch("omni.layers.npu_rms_norm.get_tensor_model_parallel_world_size", return_value=1)
    @patch("omni.layers.npu_rms_norm.get_tp_group")
    def test_all_gather_qk_heads_keeps_single_rank_layout(
        self, mock_get_tp_group, mock_world
    ):
        mock_get_tp_group.return_value = MagicMock()
        q = torch.arange(8, dtype=torch.float32).view(2, 4)
        k = torch.arange(4, dtype=torch.float32).view(2, 2)

        q_ag, k_ag = NPUMiniMaxText01RMSNormTP.all_gather_qk_heads(q, k, head_dim=2)

        assert q_ag.shape == (2, 1, 2, 2)
        assert k_ag.shape == (2, 1, 1, 2)
        assert torch.equal(q_ag[:, 0], q.view(2, 2, 2))
        assert torch.equal(k_ag[:, 0], k.view(2, 1, 2))

    @patch("omni.layers.npu_rms_norm.get_tensor_model_parallel_world_size", return_value=2)
    @patch("omni.layers.npu_rms_norm.get_tp_group")
    @patch("omni.layers.npu_rms_norm.torch_npu.npu_rms_norm")
    @patch.object(NPUMiniMaxText01RMSNormTP, "all_gather_qk_heads")
    def test_forward_qk_decoder_returns_current_rank_slice(
        self, mock_all_gather_qk, mock_rms_norm, mock_get_tp_group, mock_world
    ):
        mock_get_tp_group.return_value = SimpleNamespace(rank_in_group=1)
        q_ag = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.float32)
        k_ag = torch.tensor([[[[5.0, 6.0]], [[7.0, 8.0]]]], dtype=torch.float32)
        mock_all_gather_qk.return_value = (q_ag, k_ag)
        mock_rms_norm.side_effect = [
            (q_ag + 10.0,),
            (k_ag + 20.0,),
        ]
        q_norm = SimpleNamespace(
            head_dim=2,
            weight=torch.arange(4, dtype=torch.float32),
            variance_epsilon=1e-6,
        )
        k_norm = SimpleNamespace(
            head_dim=2,
            weight=torch.arange(4, dtype=torch.float32),
            variance_epsilon=1e-6,
        )

        out_q, out_k = NPUMiniMaxText01RMSNormTP.forward_qk_decoder(
            q_norm, k_norm, torch.randn(1, 2), torch.randn(1, 2)
        )

        assert torch.equal(out_q, torch.tensor([[13.0, 14.0]], dtype=torch.float32))
        assert torch.equal(out_k, torch.tensor([[27.0, 28.0]], dtype=torch.float32))

    @patch("omni.layers.npu_rms_norm.get_forward_context")
    @patch.object(NPUMiniMaxText01RMSNormTP, "forward_qk_prefill", return_value=("q_prefill", "k_prefill"))
    @patch.object(NPUMiniMaxText01RMSNormTP, "forward_qk_decoder", return_value=("q_decoder", "k_decoder"))
    def test_forward_qk_dispatches_to_prefill(
        self, mock_decoder, mock_prefill, mock_forward_context
    ):
        mock_forward_context.return_value = SimpleNamespace(attn_metadata=None)

        out = NPUMiniMaxText01RMSNormTP.forward_qk(
            SimpleNamespace(), SimpleNamespace(), torch.randn(1, 2), torch.randn(1, 2)
        )

        assert out == ("q_prefill", "k_prefill")
        mock_prefill.assert_called_once()
        mock_decoder.assert_not_called()

    @patch("omni.layers.npu_rms_norm.get_forward_context")
    @patch.object(NPUMiniMaxText01RMSNormTP, "forward_qk_prefill", return_value=("q_prefill", "k_prefill"))
    @patch.object(NPUMiniMaxText01RMSNormTP, "forward_qk_decoder", return_value=("q_decoder", "k_decoder"))
    def test_forward_qk_dispatches_to_decoder_when_decode_only(
        self, mock_decoder, mock_prefill, mock_forward_context
    ):
        mock_forward_context.return_value = SimpleNamespace(
            attn_metadata={"req": SimpleNamespace(num_prefills=0)}
        )

        out = NPUMiniMaxText01RMSNormTP.forward_qk(
            SimpleNamespace(), SimpleNamespace(), torch.randn(1, 2), torch.randn(1, 2)
        )

        assert out == ("q_decoder", "k_decoder")
        mock_decoder.assert_called_once()
        mock_prefill.assert_not_called()


if __name__ == "__main__":
    unittest.main()