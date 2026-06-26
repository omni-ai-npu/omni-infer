# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import unittest
from unittest.mock import patch, MagicMock
import torch
from omni_npu.layers.activation import NPUSiluAndMul


@pytest.mark.usefixtures("default_vllm_config")
class TestNPUSiluAndMul(unittest.TestCase):
    def setUp(self):
        """Create unit test environment"""
        self.silu_and_mul = NPUSiluAndMul()

        self.mock_input = torch.randn(8, 128, dtype=torch.bfloat16)
        self.mock_output = torch.randn(8, 128, dtype=torch.bfloat16)
        self.scale = torch.randn(8, dtype=torch.bfloat16)
        self.npu_dequant_swiglu_quant_mock = MagicMock()
        self.npu_dequant_swiglu_quant_mock.return_value = (self.mock_input, self.scale)
        self.npu_swiglu_mock = MagicMock()
        self.npu_swiglu_mock.return_value = self.mock_output

        self.patch1 = patch("torch_npu.npu_dequant_swiglu_quant", new=self.npu_dequant_swiglu_quant_mock)
        self.patch2 = patch("torch_npu.npu_swiglu", new=self.npu_swiglu_mock)

        self.patch1.start()
        self.patch2.start()

    def tearDown(self):
        """clear unit test environment"""
        self.patch1.stop()
        self.patch2.stop()

    def test_forward_with_quant_symbol_false_tensor_input(self):
        """Test the scenario when quant_symbol=False and the input is a Tensor"""
        self.npu_swiglu_mock.reset_mock()
        x = self.mock_input

        result = self.silu_and_mul(x, quant_symbol=False)

        self.npu_swiglu_mock.assert_called_once_with(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_forward_with_quant_symbol_false_dict_input(self):
        """Test the scenario when quant_symbol=False and the input is a dictionary"""
        self.npu_swiglu_mock.reset_mock()
        x = {"x_int8": self.mock_input}

        result = self.silu_and_mul(x, quant_symbol=False)

        self.npu_swiglu_mock.assert_called_once_with(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_forward_with_quant_symbol_true_dict_input(self):
        """Test the scenario when quant_symbol=True and the input is a dictionary."""
        self.npu_dequant_swiglu_quant_mock.reset_mock()
        x = {
            "x_int8": self.mock_input,
            "out_scale": torch.randn(128, dtype=torch.bfloat16),
            "in_scale": torch.randn(128, dtype=torch.bfloat16),
            "pertoken_scale": self.scale,
        }

        result = self.silu_and_mul(x, quant_symbol=True)

        self.npu_dequant_swiglu_quant_mock.assert_called_once()
        self.assertIsInstance(result, dict)
        self.assertIn("x_int8", result)
        self.assertIn("pertoken_scale", result)

    def test_forward_with_quant_symbol_true_dict_missing_keys(self):
        """Test the scenario when quant_symbol=True and the input dictionary is missing certain keys."""
        self.npu_dequant_swiglu_quant_mock.reset_mock()
        x = {"x_int8": self.mock_input, "out_scale": self.scale}

        result = self.silu_and_mul(x, quant_symbol=True)

        self.npu_dequant_swiglu_quant_mock.assert_called_once()
        self.assertIsInstance(result, dict)

    def test_forward_with_quant_symbol_true_tensor_input(self):
        """Test the scenario when quant_symbol=True but the input is a Tensor."""
        self.npu_swiglu_mock.reset_mock()
        x = self.mock_input

        result = self.silu_and_mul(x, quant_symbol=True)

        self.npu_swiglu_mock.assert_called_once_with(x)
        self.assertIsInstance(result, torch.Tensor)


if __name__ == "__main__":
    unittest.main()