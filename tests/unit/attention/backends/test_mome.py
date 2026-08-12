# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Covers NPUMomeAttentionMetadataBuilder.build non-prefix-cache path (seq diff + masked_fill).
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import omni_npu.attention.backends.mome as mome_mod
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from omni_npu.vllm_patches.usefull_patch.patch_kv_cache_interface import MomeSpec


class TestNPUMomeAttentionMetadataBuilder(unittest.TestCase):
    def test_build_masks_zero_len_slots_when_prefix_caching_disabled(self):
        """
        When enable_prefix_caching is False, cache_indices come from block_table[:, 0];
        zero-length query segments get PAD_SLOT_ID (mome.py ~255-256).
        """
        b = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        b.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
            speculative_config=None,
        )
        b.compilation_config = SimpleNamespace(
            max_cudagraph_capture_size=None,
            cudagraph_mode=MagicMock(
                has_full_cudagraphs=MagicMock(return_value=False)
            ),
        )
        b.mome_block_size = 16
        b.reorder_batch_threshold = 1
        b.num_spec = 0
        b.fake_num_spec = 0
        b.is_decode_node = False
        b.use_spec_decode = False
        b.is_decode_node = False
        b.decode_cudagraph_max_bs = 16
        b.kv_cache_spec = MagicMock(block_size=16)
        b.cache_indices_tensor = torch.zeros(16, dtype=torch.int32)
        b.num_computed_tokens = torch.zeros(16, dtype=torch.int32)
        b.num_accepted_tokens = torch.zeros(16, dtype=torch.int32)
        b.num_prompt_tokens = torch.full((16, ), 100, dtype=torch.int32)
        b.block_idx_last_computed_token = None
        b.block_idx_first_scheduled_token = None
        b.block_idx_last_scheduled_token = None

        # Two decode rows; first request has 0 new tokens (diff segment 0)
        common = SimpleNamespace(
            num_reqs=2,
            query_start_loc=torch.tensor([0, 1, 1], dtype=torch.int32),
            seq_lens=torch.tensor([1, 0], dtype=torch.int32),
            block_table_tensor=torch.tensor([[42], [43]], dtype=torch.int32),
            max_query_len=1,
        )

        def _compute_num_computed_tokens():
            q_lens = common.query_start_loc[1:] - common.query_start_loc[:-1]
            return common.seq_lens - q_lens

        common.compute_num_computed_tokens = _compute_num_computed_tokens

        with patch.object(
            mome_mod,
            "split_decodes_and_prefills",
            return_value=(2, 0, 2, 0),
        ):
            out = b.build(0, common, fast_build=False)

        self.assertEqual(int(out.cache_indices[0].item()), 42)
        self.assertEqual(int(out.cache_indices[1].item()), int(PAD_SLOT_ID))

    def test_update_block_table_prefix_caching_full_cudagraph_copies_block_indices(self):
        b = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        b.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True),
        )
        b.compilation_config = SimpleNamespace(
            cudagraph_mode=MagicMock(
                has_full_cudagraphs=MagicMock(return_value=True)
            ),
        )
        b.decode_cudagraph_max_bs = 8
        b.cache_indices_tensor = torch.full((8, 4), -1, dtype=torch.int32)
        b.num_computed_tokens = torch.full((8,), -1, dtype=torch.int32)
        b.num_accepted_tokens = torch.full((8,), -1, dtype=torch.int32)
        b.block_idx_last_computed_token = torch.full((8,), -1, dtype=torch.int32)
        b.block_idx_first_scheduled_token = torch.full((8,), -1, dtype=torch.int32)
        b.block_idx_last_scheduled_token = torch.full((8,), -1, dtype=torch.int32)

        decode_meta = SimpleNamespace(
            cache_indices=torch.tensor([[99, 99, 99, 99], [88, 88, 88, 88]], dtype=torch.int32),
            num_computed_tokens=torch.tensor([9, 9], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([7, 7], dtype=torch.int32),
            block_idx_last_computed_token=torch.tensor([5, 5], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([6, 6], dtype=torch.int32),
            block_idx_last_scheduled_token=torch.tensor([7, 7], dtype=torch.int32),
        )
        metadata = SimpleNamespace(
            num_prefills=0,
            num_decodes=2,
            cache_indices=torch.tensor([[99, 99, 99, 99], [88, 88, 88, 88]], dtype=torch.int32),
            num_computed_tokens=torch.tensor([9, 9], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([7, 7], dtype=torch.int32),
            block_idx_last_computed_token=torch.tensor([0, 1], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([2, 3], dtype=torch.int32),
            block_idx_last_scheduled_token=torch.tensor([4, 5], dtype=torch.int32),
            prefill=None,
            decode=decode_meta,
        )
        blk_table = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32)

        out = b.update_block_table(metadata, blk_table, slot_mapping=torch.empty(0, dtype=torch.int32))

        self.assertTrue(torch.equal(out.cache_indices, blk_table))
        self.assertTrue(torch.equal(b.cache_indices_tensor[:2], blk_table))

        self.assertTrue(torch.equal(out.block_idx_last_computed_token, torch.tensor([0, 1], dtype=torch.int32)))
        self.assertTrue(torch.equal(out.block_idx_first_scheduled_token, torch.tensor([2, 3], dtype=torch.int32)))
        self.assertTrue(torch.equal(out.block_idx_last_scheduled_token, torch.tensor([4, 5], dtype=torch.int32)))

        self.assertEqual(
            out.block_idx_last_computed_token.storage().data_ptr(),
            b.block_idx_last_computed_token.storage().data_ptr(),
        )
        self.assertEqual(
            out.block_idx_first_scheduled_token.storage().data_ptr(),
            b.block_idx_first_scheduled_token.storage().data_ptr(),
        )
        self.assertEqual(
            out.block_idx_last_scheduled_token.storage().data_ptr(),
            b.block_idx_last_scheduled_token.storage().data_ptr(),
        )

        self.assertTrue(torch.equal(out.decode.cache_indices, blk_table))
        self.assertTrue(torch.equal(out.decode.block_idx_last_computed_token, torch.tensor([0, 1], dtype=torch.int32)))
        self.assertTrue(torch.equal(out.decode.block_idx_first_scheduled_token, torch.tensor([2, 3], dtype=torch.int32)))
        self.assertTrue(torch.equal(out.decode.block_idx_last_scheduled_token, torch.tensor([4, 5], dtype=torch.int32)))

    # ---- Tests for _update_cache_indices_for_flashcomm2 ----

    def test_update_cache_indices_for_flashcomm2_shape_and_values(self):
        """
        _update_cache_indices_for_flashcomm2 should produce cache_indices_rearranged
        of shape (bsz * state_len * rearrange_ratio, 2) where:
          - column 0 broadcasts the per-request cache index
          - column 1 is arange(state_len * rearrange_ratio) tiled per request
        """
        b = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        b.state_len = 3  # e.g. kernel_width - 1 + num_spec = 2 + 1

        fc2_metadata = SimpleNamespace(rearrange_ratio=None, cache_indices_rearranged=None)
        cache_indices = torch.tensor([10, 20, 30], dtype=torch.int32)  # bsz=3

        b._update_cache_indices_for_flashcomm2(fc2_metadata, cache_indices)

        rearrange_ratio = 4
        bsz = 3
        state_len = 3
        expected_shape = (bsz * state_len * rearrange_ratio, 2)
        self.assertEqual(fc2_metadata.cache_indices_rearranged.shape, expected_shape)
        self.assertEqual(fc2_metadata.rearrange_ratio, rearrange_ratio)

        # Verify column 0: each block of state_len*rearrange_ratio rows has the same cache index
        col0 = fc2_metadata.cache_indices_rearranged[:, 0].view(bsz, -1)
        for i, idx in enumerate([10, 20, 30]):
            self.assertTrue(torch.all(col0[i] == idx))

        # Verify column 1: repeats arange(state_len * rearrange_ratio) for each request
        col1 = fc2_metadata.cache_indices_rearranged[:, 1].view(bsz, -1)
        expected_col1 = torch.arange(state_len * rearrange_ratio, dtype=torch.int32)
        for i in range(bsz):
            self.assertTrue(torch.equal(col1[i], expected_col1))

    # ---- Tests for _build_for_flashcomm2 ----

    def _make_builder_for_flashcomm2(self, state_len=3):
        """Helper to create a builder with minimal attributes for _build_for_flashcomm2."""
        b = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        b.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=False),
        )
        b.state_len = state_len
        return b

    def _make_common_for_flashcomm2(self, query_start_loc, seq_lens, num_reqs):
        """Helper to build a CommonAttentionMetadata-like namespace for flashcomm2 tests."""
        qsl = torch.tensor(query_start_loc, dtype=torch.int32)
        common = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=qsl,
            query_start_loc_cpu=qsl,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            num_computed_tokens_cpu=torch.zeros(num_reqs, dtype=torch.int32),
        )
        return common

    @patch("omni_npu.attention.backends.mome.get_tp_group")
    def test_build_for_flashcomm2_single_rank(self, mock_get_tp_group):
        """With tp_size=1, all tokens go to rank 0."""
        mock_tp = MagicMock()
        mock_tp.world_size = 1
        mock_tp.rank_in_group = 0
        mock_get_tp_group.return_value = mock_tp

        b = self._make_builder_for_flashcomm2(state_len=3)
        # 2 requests: 3 tokens and 2 tokens = 5 total tokens
        common = self._make_common_for_flashcomm2(
            query_start_loc=[0, 3, 5], seq_lens=[3, 2], num_reqs=2,
        )
        cache_indices = torch.tensor([100, 200], dtype=torch.int32)

        result = b._build_for_flashcomm2(common, cache_indices)

        self.assertEqual(result.padded_local_size, 5)
        self.assertEqual(result.local_size, [5])
        self.assertEqual(result.token_range, [[0, 5]])
        self.assertEqual(result.req_idx_start, [0])
        self.assertEqual(result.req_idx_end, [2])
        self.assertEqual(result.max_local_reqs, 2)
        # qsl_local should match original qsl
        self.assertTrue(torch.equal(
            result.qsl_local, torch.tensor([0, 3, 5], dtype=torch.int32)
        ))
        # cache_indices_rearranged should be populated
        self.assertIsNotNone(result.cache_indices_rearranged)

    @patch("omni_npu.attention.backends.mome.get_tp_group")
    def test_build_for_flashcomm2_two_ranks_even_split(self, mock_get_tp_group):
        """
        With tp_size=2 and 4 tokens from a single request,
        tokens split evenly: rank 0 gets [0,2), rank 1 gets [2,4).
        """
        mock_tp = MagicMock()
        mock_tp.world_size = 2
        mock_tp.rank_in_group = 0
        mock_get_tp_group.return_value = mock_tp

        b = self._make_builder_for_flashcomm2(state_len=3)
        # 1 request with 4 tokens
        common = self._make_common_for_flashcomm2(
            query_start_loc=[0, 4], seq_lens=[4], num_reqs=1,
        )
        cache_indices = torch.tensor([50], dtype=torch.int32)

        result = b._build_for_flashcomm2(common, cache_indices)

        self.assertEqual(result.padded_local_size, 2)
        self.assertEqual(result.local_size, [2, 2])
        self.assertEqual(result.token_range, [[0, 2], [2, 4]])
        # rank 0 owns the single request (it spans both ranks)
        self.assertEqual(result.req_idx_start[0], 0)
        self.assertEqual(result.req_idx_end[0], 1)
        # qsl_local for rank 0 should be [0, 2]
        self.assertTrue(torch.equal(
            result.qsl_local, torch.tensor([0, 2], dtype=torch.int32)
        ))

    @patch.dict("vllm.v1.kv_cache_interface.__dict__", {"MomeSpec": MomeSpec})
    @patch.object(mome_mod.GDNAttentionMetadataBuilder, "__init__", return_value=None)
    @patch.object(mome_mod.NPUMomeAttentionMetadataBuilder, "_init_reorder_batch_threshold", return_value=None)
    def test_init_sets_pd_disagg_and_fake_num_spec(self, mock_reorder, mock_super_init):
        from omni_npu.attention.backends.mome import NPUMomeAttentionMetadataBuilder

        spec_cfg = SimpleNamespace(num_speculative_tokens=2)
        kv_trans = SimpleNamespace(kv_role="kv_consumer")
        vllm_cfg = SimpleNamespace(
            speculative_config=spec_cfg,
            kv_transfer_config=kv_trans,
            compilation_config=SimpleNamespace(
                cudagraph_mode=MagicMock(has_full_cudagraphs=MagicMock(return_value=False)),
                max_cudagraph_capture_size=None,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(
                max_model_len=4096,
                hf_config=SimpleNamespace(router_sliding_window=5),  # 自定义值
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
        )

        fake_spec = MomeSpec(
            block_size=8,
            shapes=((512,), (512,), (1024,)),
            dtypes=(torch.bfloat16, torch.bfloat16, torch.bfloat16),
            kernel_size=4,
            num_spec_tokens=0,
        )

        builder = NPUMomeAttentionMetadataBuilder(
            kv_cache_spec=fake_spec,
            layer_names=["layer"],
            vllm_config=vllm_cfg,
            device=torch.device("cpu"),
        )

        self.assertTrue(builder.is_pd_disagg)
        self.assertTrue(builder.is_decode_node)
        self.assertEqual(builder.num_spec, 2)
        self.assertEqual(builder.fake_num_spec, 2)
        self.assertEqual(builder.kernel_width, 5)
        self.assertEqual(builder.state_len, 5 - 1 + 2)

    @patch("omni_npu.attention.backends.mome.split_decodes_and_prefills")
    def test_build_prefix_caching_spec_prompt_tokens(self, mock_split):
        builder = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        builder.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            speculative_config=SimpleNamespace(num_speculative_tokens=2),
            kv_transfer_config=None,
            compilation_config=SimpleNamespace(
                cudagraph_mode=MagicMock(has_full_cudagraphs=MagicMock(return_value=False))
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
        )
        builder.compilation_config = builder.vllm_config.compilation_config
        builder.mome_block_size = 8
        builder.num_spec = 2
        builder.use_spec_decode = True
        builder.is_pd_disagg = False
        builder.is_decode_node = False
        builder.reuse_prefilled_tokens = False
        builder.fake_num_spec = 2
        builder.reorder_batch_threshold = 1
        builder.kv_cache_spec = MagicMock(block_size=8)
        builder.decode_cudagraph_max_bs = 16
        builder.cache_indices_tensor = torch.empty(16, dtype=torch.int32)
        builder.num_computed_tokens = torch.empty(16, dtype=torch.int32)
        builder.num_accepted_tokens = torch.empty(16, dtype=torch.int32)
        builder.block_idx_last_computed_token = torch.empty(16, dtype=torch.int32)
        builder.block_idx_first_scheduled_token = torch.empty(16, dtype=torch.int32)
        builder.block_idx_last_scheduled_token = torch.empty(16, dtype=torch.int32)
        builder.enable_flashcomm2 = False

        num_reqs = 2
        seq_lens = torch.tensor([10, 12], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
        block_table = torch.randint(0, 100, (num_reqs, 4), dtype=torch.int32)

        num_comp = torch.tensor([8, 9], dtype=torch.int32)
        num_prompt = torch.tensor([7, 10], dtype=torch.int32)

        common = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table_tensor=block_table,
            max_query_len=3,
            compute_num_computed_tokens=lambda: num_comp,
        )
        mock_split.return_value = (0, num_reqs, 0, 5)

        num_acc = torch.tensor([2, 2], dtype=torch.int32)

        meta = builder.build(0, common,
                            num_accepted_tokens=num_acc,
                            num_prompt_tokens=num_prompt)

        self.assertIsNotNone(meta.block_idx_last_computed_token)
        self.assertEqual(int(meta.block_idx_last_computed_token[0]), 1)
        self.assertEqual(int(meta.block_idx_last_computed_token[1]), 1)

    @patch("omni_npu.attention.backends.mome.split_decodes_and_prefills")
    def test_build_masked_fill_applied(self, mock_split):
        builder = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        builder.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            speculative_config=SimpleNamespace(num_speculative_tokens=1),
            kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
            compilation_config=SimpleNamespace(
                cudagraph_mode=MagicMock(has_full_cudagraphs=MagicMock(return_value=False))
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
        )
        builder.compilation_config = builder.vllm_config.compilation_config
        builder.mome_block_size = 8
        builder.num_spec = 1
        builder.use_spec_decode = True
        builder.is_pd_disagg = True
        builder.is_decode_node = True
        builder.reuse_prefilled_tokens = False
        builder.fake_num_spec = 1
        builder.reorder_batch_threshold = 1
        builder.kv_cache_spec = MagicMock(block_size=8)
        builder.decode_cudagraph_max_bs = 16
        builder.cache_indices_tensor = torch.empty(16, dtype=torch.int32)
        builder.num_computed_tokens = torch.empty(16, dtype=torch.int32)
        builder.num_accepted_tokens = torch.empty(16, dtype=torch.int32)
        builder.enable_flashcomm2 = False

        num_reqs = 2
        seq_lens = torch.tensor([8, 6], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 3, 4], dtype=torch.int32)
        block_table = torch.randint(0, 100, (num_reqs, 2), dtype=torch.int32)

        num_comp = torch.tensor([6, 4], dtype=torch.int32)
        num_prompt = torch.tensor([5, 5], dtype=torch.int32)

        common = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table_tensor=block_table,
            max_query_len=3,
            compute_num_computed_tokens=lambda: num_comp,
        )
        mock_split.return_value = (0, num_reqs, 0, 7)

        num_acc = torch.tensor([3, 3], dtype=torch.int32)

        meta = builder.build(0, common,
                            num_accepted_tokens=num_acc,
                            num_prompt_tokens=num_prompt)

        self.assertFalse(torch.equal(meta.num_accepted_tokens, num_acc),
                        "masked_fill_ should have modified num_accepted_tokens")

    @patch("omni_npu.attention.backends.mome.split_decodes_and_prefills")
    def test_build_for_cudagraph_capture(self, mock_split):
        builder = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        builder.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            speculative_config=None,
            kv_transfer_config=None,
            compilation_config=SimpleNamespace(
                cudagraph_mode=MagicMock(has_full_cudagraphs=MagicMock(return_value=False))
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
        )
        builder.compilation_config = builder.vllm_config.compilation_config
        builder.mome_block_size = 8
        builder.num_spec = 0
        builder.use_spec_decode = False
        builder.fake_num_spec = 0
        builder.reorder_batch_threshold = 1
        builder.kv_cache_spec = MagicMock(block_size=8)
        builder.decode_cudagraph_max_bs = 4
        builder.is_pd_disagg = False
        builder.is_decode_node = False          # 补充缺失属性
        builder.reuse_prefilled_tokens = False

        common = SimpleNamespace(
            num_reqs=2,
            num_actual_tokens=2,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            block_table_tensor=torch.randint(0, 10, (2, 2), dtype=torch.int32),
            max_query_len=1,
            compute_num_computed_tokens=lambda: torch.tensor([2, 4]),
        )
        mock_split.return_value = (2, 0, 2, 0)  # all decodes

        meta = builder.build_for_cudagraph_capture(common)
        self.assertEqual(meta.num_decodes, 2)
        self.assertTrue(torch.equal(meta.query_start_loc, common.query_start_loc))

    def test_update_block_table_with_prefill_present(self):
        from omni_npu.attention.backends.mome import NPUMomeAttentionMetadata

        builder = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        builder.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True),
            compilation_config=SimpleNamespace(
                cudagraph_mode=MagicMock(has_full_cudagraphs=MagicMock(return_value=False))
            ),
        )
        builder.compilation_config = builder.vllm_config.compilation_config
        builder.decode_cudagraph_max_bs = 8
        builder.cache_indices_tensor = torch.full((8, 4), -1, dtype=torch.int32)
        builder.num_computed_tokens = torch.full((8,), -1, dtype=torch.int32)
        builder.num_accepted_tokens = torch.full((8,), -1, dtype=torch.int32)
        builder.block_idx_last_computed_token = torch.full((8,), -1, dtype=torch.int32)
        builder.block_idx_first_scheduled_token = torch.full((8,), -1, dtype=torch.int32)
        builder.block_idx_last_scheduled_token = torch.full((8,), -1, dtype=torch.int32)

        cum_len = torch.tensor([0, 2, 5], dtype=torch.int32)
        base_meta = NPUMomeAttentionMetadata(
            num_prefills=1, num_prefill_tokens=3,
            num_decodes=1, num_decode_tokens=2,
            query_start_loc=cum_len,
            cache_indices=torch.zeros(2, 4, dtype=torch.int32),
            block_idx_last_computed_token=torch.tensor([1, 0], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([1, 0], dtype=torch.int32),
            block_idx_last_scheduled_token=torch.tensor([2, 0], dtype=torch.int32),
            num_computed_tokens=torch.tensor([4, 5], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
            num_reqs=2,
        )
        base_meta.prefill = NPUMomeAttentionMetadata(
            num_prefills=1, num_prefill_tokens=3,
            num_decodes=0, num_decode_tokens=0, num_reqs=1,
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            cache_indices=torch.zeros(1, 4, dtype=torch.int32),
            block_idx_last_computed_token=torch.tensor([1], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([1], dtype=torch.int32),
            block_idx_last_scheduled_token=torch.tensor([2], dtype=torch.int32),
            num_computed_tokens=torch.tensor([4], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
        )
        base_meta.decode = NPUMomeAttentionMetadata(
            num_prefills=0, num_prefill_tokens=0,
            num_decodes=1, num_decode_tokens=2, num_reqs=1,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            cache_indices=torch.zeros(1, 4, dtype=torch.int32),
            block_idx_last_computed_token=torch.tensor([0], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([0], dtype=torch.int32),
            block_idx_last_scheduled_token=torch.tensor([0], dtype=torch.int32),
            num_computed_tokens=torch.tensor([5], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
        )

        new_blk = torch.tensor([[11,12,13,14],[21,22,23,24]], dtype=torch.int32)
        result = builder.update_block_table(base_meta, new_blk, slot_mapping=torch.empty(0, dtype=torch.int32))

        self.assertIsNotNone(result.prefill)
        self.assertTrue(torch.equal(result.prefill.cache_indices, new_blk[1:]))
        self.assertTrue(torch.equal(result.prefill.block_idx_last_computed_token,
                                    torch.tensor([0], dtype=torch.int32)))


class TestComputePrefixCachingBlockIndices(unittest.TestCase):
    """Unit tests for NPUMomeAttentionMetadataBuilder._compute_prefix_caching_block_indices."""

    def _make_builder(self, num_spec=0, is_decode_node=False):
        b = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        b.num_spec = num_spec
        b.is_decode_node = is_decode_node
        b.reuse_prefilled_tokens = False
        return b

    def _make_common(self, num_computed, seq_lens):
        return SimpleNamespace(
            compute_num_computed_tokens=lambda: torch.tensor(num_computed, dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        )

    def test_no_spec_computes_simple_block_indices(self):
        """
        When num_spec=0 (or num_prompt_tokens=None), all three outputs use
        the simple ceil-division formula without any spec-decode adjustment.
        Verifies block_idx_last_computed_token, first_scheduled, and last_scheduled.
        """
        b = self._make_builder(num_spec=0)
        # num_computed=[8,16], seq_lens=[9,24], block_size=8
        # lct:  ceil(8/8)-1=0,  ceil(16/8)-1=1
        # fst:  ceil(9/8)-1=1,  ceil(17/8)-1=2
        # lst:  ceil(9/8)-1=1,  ceil(24/8)-1=2
        common = self._make_common(num_computed=[8, 16], seq_lens=[9, 24])
        num_accepted = torch.tensor([1, 1], dtype=torch.int32)

        lct, fst, lst = b._compute_prefix_caching_block_indices(
            common, mome_block_size=8, num_accepted_tokens=num_accepted
        )

        self.assertTrue(torch.equal(lct, torch.tensor([0, 1], dtype=torch.int32)))
        self.assertTrue(torch.equal(fst, torch.tensor([1, 2], dtype=torch.int32)))
        self.assertTrue(torch.equal(lst, torch.tensor([1, 2], dtype=torch.int32)))

    def test_spec_decode_mtp_branch_when_computed_exceeds_prompt(self):
        """
        With num_spec>0 and num_computed > num_prompt_tokens, the MTP formula
        ceil((num_computed - num_accepted + num_spec + 1) / B) - 1 is used.
        """
        b = self._make_builder(num_spec=2)
        # num_computed=17, num_prompt=10, num_accepted=2, block_size=8
        # 17 > 10, so MTP: ceil((17-2+2+1)/8)-1 = ceil(18/8)-1 = 3-1 = 2
        common = self._make_common(num_computed=[17], seq_lens=[20])
        num_accepted = torch.tensor([2], dtype=torch.int32)
        num_prompt = torch.tensor([10], dtype=torch.int32)

        lct, _, _ = b._compute_prefix_caching_block_indices(
            common, mome_block_size=8,
            num_accepted_tokens=num_accepted, num_prompt_tokens=num_prompt,
        )

        self.assertEqual(int(lct[0].item()), 2)

    def test_spec_decode_prefill_branch_when_computed_not_exceeds_prompt(self):
        """
        With num_spec>0 and num_computed <= num_prompt_tokens, falls back to
        the simple formula ceil(num_computed / B) - 1 (no MTP adjustment).
        """
        b = self._make_builder(num_spec=2)
        # num_computed=8 <= num_prompt=10 → simple: ceil(8/8)-1 = 0
        common = self._make_common(num_computed=[8], seq_lens=[12])
        num_accepted = torch.tensor([2], dtype=torch.int32)
        num_prompt = torch.tensor([10], dtype=torch.int32)

        lct, _, _ = b._compute_prefix_caching_block_indices(
            common, mome_block_size=8,
            num_accepted_tokens=num_accepted, num_prompt_tokens=num_prompt,
        )

        self.assertEqual(int(lct[0].item()), 0)

    def test_decode_node_recompute_overrides_lct_when_computed_below_prompt(self):
        """
        On a decode node, when num_computed < num_prompt_tokens (decode recompute),
        the override ceil((num_computed+1)/B)-1 applies, which can differ from
        the spec-decode formula when num_computed is block-aligned.
        """
        b = self._make_builder(num_spec=2, is_decode_node=True)
        # num_computed=8 (block-aligned), num_prompt=10, block_size=8
        # Without decode-node override: ceil(8/8)-1 = 0  (num_computed<=num_prompt path)
        # With decode-node override:    ceil(9/8)-1 = 1  (num_computed < num_prompt path)
        common = self._make_common(num_computed=[8], seq_lens=[12])
        num_accepted = torch.tensor([2], dtype=torch.int32)
        num_prompt = torch.tensor([10], dtype=torch.int32)

        lct, _, _ = b._compute_prefix_caching_block_indices(
            common, mome_block_size=8,
            num_accepted_tokens=num_accepted, num_prompt_tokens=num_prompt,
        )

        self.assertEqual(int(lct[0].item()), 1)

    def test_mixed_batch_mtp_and_prefill_reqs(self):
        """
        Mixed batch: req0 has num_computed > num_prompt (MTP branch),
        req1 has num_computed <= num_prompt (simple branch). Verifies torch.where
        selects the correct formula per request independently.
        """
        b = self._make_builder(num_spec=2)
        # req0: 16>10 → ceil((16-2+2+1)/8)-1 = ceil(17/8)-1 = 3-1 = 2
        # req1:  8<=10 → ceil(8/8)-1 = 0
        common = self._make_common(num_computed=[16, 8], seq_lens=[20, 12])
        num_accepted = torch.tensor([2, 2], dtype=torch.int32)
        num_prompt = torch.tensor([10, 10], dtype=torch.int32)

        lct, _, _ = b._compute_prefix_caching_block_indices(
            common, mome_block_size=8,
            num_accepted_tokens=num_accepted, num_prompt_tokens=num_prompt,
        )

        self.assertEqual(int(lct[0].item()), 2)
        self.assertEqual(int(lct[1].item()), 0)


def _manual_builder(*, reuse_prefilled_tokens=False):
    """Builder via __new__ skips __init__; set attrs that build() expects."""
    b = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
        mome_mod.NPUMomeAttentionMetadataBuilder
    )
    b.reuse_prefilled_tokens = reuse_prefilled_tokens
    return b