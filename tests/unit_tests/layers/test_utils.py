import os
import unittest

import torch
import torch_npu

from omni.layers.attention.backend.utils import create_aligend_tensor


class TestAlignedTensor(unittest.TestCase):

    def setUp(self):
        self.original_env = os.getenv("VLLM_ENABLE_KV_CACHE_TENSOR_2MB_ALIGNMENT")

    def tearDown(self):
        if self.original_env is not None:
            os.environ["VLLM_ENABLE_KV_CACHE_TENSOR_2MB_ALIGNMENT"] = self.original_env
        elif "VLLM_ENABLE_KV_CACHE_TENSOR_2MB_ALIGNMENT" in os.environ:
            del os.environ["VLLM_ENABLE_KV_CACHE_TENSOR_2MB_ALIGNMENT"]

    def test_aligned_creation_when_enable(self):
        os.environ["VLLM_ENABLE_KV_CACHE_TENSOR_2MB_ALIGNMENT"] = "1"

        device = "npu"  
        dummy = torch.zeros(123, dtype=torch.uint8, device=device)

        shape = (16, 20)
        dtype = torch.float16

        tensor = create_aligend_tensor(shape, dtype, device)

        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, dtype)
        self.assertEqual(tensor.device.type, 'npu')

        alignment = 2 * 1024 * 1024
        ptr = tensor.data_ptr()
        is_aligned = (ptr & (alignment - 1)) == 0
        self.assertTrue(is_aligned, f"Tensor address {hex(ptr)} is not 2MB aligned.")