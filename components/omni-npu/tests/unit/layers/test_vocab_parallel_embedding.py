# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from unittest.mock import Mock, patch

from omni_npu.v1.layers.vocab_parallel_embedding import (
    NPUVocabParallelEmbedding,
    NPUParallelLMHead,
    get_masked_input_and_mask,
)


class _mock_group:
    def __init__(self, size: int, rank: int=0):
        self.world_size = size
        self.rank_in_group = rank
    def all_reduce(self, x: torch.Tensor):
        return x.clone()


tp_group = "vllm.distributed.parallel_state._TP"
local_world = "omni_npu.v1.distributed.parallel_state_ext._LOCAL_WORLD"
dispatch_forward = "vllm.model_executor.custom_op.CustomOp.dispatch_forward"
test_file = "omni_npu.v1.layers.vocab_parallel_embedding"


class TestGetMaskedInputAndMask:

    def test_no_added_vocab(monkeypatch):
        input_ids = torch.tensor([0, 3, 4, 7, 9])
        masked_input, mask = get_masked_input_and_mask(
            input_ids,
            org_vocab_start_index=0,
            org_vocab_end_index=5,
            num_org_vocab_padding=2,
            added_vocab_start_index=5,
            added_vocab_end_index=5,
        )

        assert torch.equal(masked_input, torch.tensor([0, 3, 4, 0, 0]))
        assert torch.equal(mask, torch.tensor([False, False, False, True, True]))

    def test_with_added_vocab(monkeypatch):
        input_ids = torch.tensor([0, 2, 6, 7, 9])
        masked_input, mask = get_masked_input_and_mask(
            input_ids,
            org_vocab_start_index=0,
            org_vocab_end_index=4,
            num_org_vocab_padding=0,
            added_vocab_start_index=6,
            added_vocab_end_index=8,
        )

        assert torch.equal(masked_input, torch.tensor([0, 2, 4, 5, 0]))
        assert torch.equal(mask, torch.tensor([False, False, False, False, True]))


# @patch(dispatch_forward, _mock_dispatch_forward)
@patch(tp_group, _mock_group(4, 0))
@patch(local_world, _mock_group(4, 0))
class TestNPUVocabParallelEmbedding:

    def test_init_with_org_num_embeddings(self):
        embedding = NPUVocabParallelEmbedding(
            num_embeddings=2000,
            embedding_dim=256,
            org_num_embeddings=1500
        )
        assert embedding.num_embeddings == 2000
        assert embedding.org_vocab_size == 1500
        assert embedding.num_added_embeddings == 500

    @patch(test_file + ".get_masked_input_and_mask")
    @patch(test_file + ".tensor_model_parallel_all_reduce")
    def test_forward_default(self, mock_all_reduce, mock_get_mask):
        embedding = NPUVocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=128
        )
        mock_get_mask.return_value = (torch.tensor([1, 2, 3]), torch.tensor([False, False, False]))
        embedding.quant_method = Mock()
        mock_output = torch.randn(3, 128)
        embedding.quant_method.embedding.return_value = mock_output
        mock_all_reduce.return_value = mock_output

        input_tensor = torch.tensor([10, 20, 30])
        result = embedding.forward(input_tensor)

        mock_all_reduce.assert_called_once()
        assert torch.all(result == mock_output)

    @patch(test_file + ".get_masked_input_and_mask")
    @patch(test_file + ".tensor_model_parallel_reduce_scatter")
    def test_forward_flash_comm_1(self, mock_reduce_scatter, mock_get_mask):
        embedding = NPUVocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=128
        )
        mock_get_mask.return_value = (torch.tensor([1, 2, 3]), torch.tensor([False, False, False]))
        embedding.quant_method = Mock()
        mock_output = torch.randn(3, 128)
        embedding.quant_method.embedding.return_value = mock_output
        mock_reduce_scatter.return_value = mock_output
        input_tensor = torch.tensor([10, 20, 30])
        result = embedding.forward(input_tensor, enable_scatter=True)
        mock_reduce_scatter.assert_called_once_with(mock_output, dim=0)
        assert torch.all(result == mock_output)

    def test_forward_tp_size_1(self):
        embedding = NPUVocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=128,
        )
        embedding.quant_method = Mock()
        mock_output = torch.randn(3, 128)
        embedding.quant_method.embedding.return_value = mock_output
        
        input_tensor = torch.tensor([10, 20, 30])
        result = embedding.forward(input_tensor)
        assert torch.all(result == mock_output)

    @patch(test_file + ".get_masked_input_and_mask")
    def test_forward_with_input_mask(self, mock_get_mask):
        embedding = NPUVocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=128,
        )
        input_mask = torch.tensor([False, True, False])
        mock_get_mask.return_value = (torch.tensor([1, 0, 3]), input_mask)
        embedding.quant_method = Mock()
        mock_output = torch.ones(3, 128)
        embedding.quant_method.embedding.return_value = mock_output
        input_tensor = torch.tensor([10, 20, 30])
        result = embedding.forward(input_tensor)
        assert torch.all(result[1] == 0)


# @patch(dispatch_forward, _mock_dispatch_forward)
@patch(tp_group, _mock_group(4, 0))
@patch(local_world, _mock_group(4, 0))
class TestNPUParallelLMHead:

    def test_init_without_bias(self):
        lm_head = NPUParallelLMHead(
            num_embeddings=1000,
            embedding_dim=128,
            bias=False
        )
        assert lm_head.bias is None

    def test_init_with_bias(self):
        lm_head = NPUParallelLMHead(
            num_embeddings=1000,
            embedding_dim=128,
            bias=True,
            params_dtype=torch.float32
        )
        assert lm_head.bias is not None
        assert lm_head.bias.shape == (256,)

    def test_init_default_dp_parallel_is_false(self):
        lm_head = NPUParallelLMHead(
            num_embeddings=1000,
            embedding_dim=128,
        )
        assert lm_head.dp_parallel is False
        # Class-level pad target default
        assert NPUParallelLMHead._dp_pad_n == 0

    def test_init_dp_parallel_rebuilds_sharding_with_dp_group(self):
        """With dp_parallel=True the lm_head re-initializes its sharding to
        use the DP group's size instead of the TP group's."""
        dp_group = _mock_group(8, 3)
        with patch("omni_npu.v1.layers.vocab_parallel_embedding.get_dp_group",
                   return_value=dp_group):
            lm_head = NPUParallelLMHead(
                num_embeddings=1000,
                embedding_dim=128,
                dp_parallel=True,
            )
        assert lm_head.dp_parallel is True
        # DP group size (8) took the place of the TP size (4) used by the
        # default @patch decorators above.
        assert lm_head.tp_size == 8
        # Weight was re-allocated in the second create_weights call, so per-
        # partition count matches DP sharding.
        assert lm_head.num_embeddings_per_partition == 1024 // 8

    def test_init_local_lmhead_parallel_rebuilds_sharding_with_local_group(self):
        """With local_lmhead_parallel=True the lm_head re-initializes its
        sharding to use the local_world_group instead of the TP group."""
        local_group = _mock_group(16, 5)
        with patch(
            "omni_npu.v1.layers.vocab_parallel_embedding.get_local_world_group",
            return_value=local_group,
        ):
            lm_head = NPUParallelLMHead(
                num_embeddings=1000,
                embedding_dim=128,
                local_lmhead_parallel=True,
            )
        assert lm_head.local_lmhead_parallel is True
        assert lm_head.dp_parallel is False
        assert lm_head.tp_size == 16
        assert lm_head.num_embeddings_per_partition == 1024 // 16
