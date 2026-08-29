# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import os
import weakref

import torch

from vllm.v1.kv_cache_interface import AttentionSpec
import omni_npu.attention.backends.dsa as mla_mod
from omni_npu.attention.backends.utils import get_batch_desc


class TestNPUDSABackend(unittest.TestCase):
    def test_metadata_default_slot_mapping_2d_can_be_overridden(self):
        metadata = mla_mod.NPUDSAMetadata(
            prefill=None,
            decode=None,
            num_actual_tokens=0,
            num_prefills=0,
            num_decodes=0,
            num_decode_tokens=0,
            slot_mapping=None,
            num_reqs=0,
            max_query_len=0,
            max_seq_len=0,
            query_start_loc=None,
        )

        self.assertIsNone(metadata.get_slot_mapping_2d())

        expected = torch.tensor([[1, 2]], dtype=torch.long)

        def get_expected_slot_mapping_2d():
            return expected

        metadata.get_slot_mapping_2d = get_expected_slot_mapping_2d
        self.assertIs(metadata.get_slot_mapping_2d(), expected)

    def test_reshape_kv_cache_splits_into_three_expected_shapes(self):
        num_blocks = 2
        block_size = 3
        shapes = [
            (num_blocks, block_size, 1, 512),
            (num_blocks, block_size, 1, 64),
            (num_blocks, block_size, 1, 128),
        ]
        total = sum(int(torch.tensor(s).prod().item()) for s in shapes)

        raw = torch.zeros((total,), dtype=torch.bfloat16)

        mock_kv_transfer_config = MagicMock()
        mock_kv_transfer_config.kv_role = "kv_producer"
        
        mock_vllm_config = MagicMock()
        mock_vllm_config.kv_transfer_config = mock_kv_transfer_config

        with patch("omni_npu.attention.backends.dsa.get_current_vllm_config",
                return_value=mock_vllm_config):
            kv_cache_spec = AttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
            )
            out = mla_mod.NPUDSABackend.reshape_kv_cache(
                raw_tensor=raw,
                num_blocks=num_blocks,
                kv_cache_spec=kv_cache_spec,
            )
        
        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[0].shape), shapes[0])
        self.assertEqual(tuple(out[1].shape), shapes[1])
        self.assertEqual(tuple(out[2].shape), shapes[2])

    def test_reshape_kv_cache_hif8_uses_float32_scale_dtype(self):
        num_blocks = 1
        block_size = 1
        raw = torch.zeros((656 + 128 + 4,), dtype=torch.bfloat16)

        mock_vllm_config = MagicMock()
        mock_vllm_config.cache_config.cache_dtype = "hif8_ds_mla"
        kv_cache_spec = AttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )

        with patch("omni_npu.attention.backends.dsa.get_current_vllm_config",
                   return_value=mock_vllm_config):
            out = mla_mod.NPUDSABackend.reshape_kv_cache(
                raw_tensor=raw,
                num_blocks=num_blocks,
                kv_cache_spec=kv_cache_spec,
            )

        self.assertEqual(tuple(out[0].shape), (1, 1, 1, 656))
        self.assertEqual(tuple(out[1].shape), (1, 1, 1, 128))
        self.assertEqual(out[2].dtype, torch.float32)

    def test_reshape_kv_cache_raises_when_numel_mismatch(self):
        raw = torch.zeros((123,), dtype=torch.bfloat16)
        
        # mock get_current_vllm_config
        mock_kv_transfer_config = MagicMock()
        mock_kv_transfer_config.kv_role = "kv_producer" 
        
        mock_vllm_config = MagicMock()
        mock_vllm_config.kv_transfer_config = mock_kv_transfer_config

        kv_cache_spec = AttentionSpec(
            block_size=1,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )

        with patch("omni_npu.attention.backends.dsa.get_current_vllm_config",
                return_value=mock_vllm_config), \
            self.assertRaises(RuntimeError):
            _ = mla_mod.NPUDSABackend.reshape_kv_cache(
                raw_tensor=raw,
                num_blocks=1,
                kv_cache_spec=kv_cache_spec,
            )


class TestNPUDSABackendReshapeNoncontiguous(unittest.TestCase):
    """Tests for _reshape_kv_cache_noncontiguous covering the new dtype branches."""

    def _call_with_dtype(self, cache_dtype_str):
        mock_spec = MagicMock()
        mock_spec.cache_dtype_str = cache_dtype_str
        mock_spec.block_size = 1
        mock_spec.page_size_bytes = 4096

        sentinel = object()
        with patch.object(
            mla_mod,
            "_maybe_padded_raw_tensor_to_strided_caches",
            return_value=sentinel,
        ) as mock_fn:
            result = mla_mod.NPUDSABackend._reshape_kv_cache_noncontiguous(
                raw_tensor=MagicMock(),
                num_blocks=2,
                kv_cache_spec=mock_spec,
            )
        self.assertIs(result, sentinel)
        _, kwargs = mock_fn.call_args
        return kwargs["shapes"], kwargs["dtypes"]

    def test_int8_ds_mla_shapes_and_dtypes(self):
        shapes, dtypes = self._call_with_dtype("int8_ds_mla")
        self.assertEqual(shapes, ((656,), (128,), (1,)))
        self.assertEqual(dtypes, (torch.int8, torch.int8, torch.float16))

    def test_hif8_ds_mla_shapes_and_dtypes(self):
        shapes, dtypes = self._call_with_dtype("hif8_ds_mla")
        self.assertEqual(shapes, ((656,), (128,), (1,)))
        self.assertEqual(dtypes, (torch.uint8, torch.uint8, torch.float32))

    def test_li_int8_ds_mla_shapes_and_dtypes(self):
        shapes, dtypes = self._call_with_dtype("li_int8_ds_mla")
        self.assertEqual(shapes, ((576,), (128,), (1,)))
        self.assertEqual(dtypes, (torch.bfloat16, torch.int8, torch.float16))

    def test_fp8_ds_mla_shapes_and_dtypes(self):
        """Existing fp8_ds_mla branch: regression guard."""
        shapes, dtypes = self._call_with_dtype("fp8_ds_mla")
        self.assertEqual(shapes, ((656,), (128,), (1,)))
        self.assertEqual(dtypes, (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float32))

    def test_default_branch_shapes_and_dtypes(self):
        """Default branch uses model_extra_config.dtype (bf16 by default)."""
        from omni_npu.model_config.config_loader.loader import model_extra_config

        shapes, dtypes = self._call_with_dtype("bf16")
        self.assertEqual(shapes, ((576,), (128,)))
        self.assertEqual(dtypes, (model_extra_config.dtype, model_extra_config.dtype))

    def test_default_branch_respects_fp16_dtype(self):
        """Default branch follows model_extra_config.dtype when set to float16."""
        with patch.object(mla_mod.model_extra_config, "dtype", torch.float16):
            shapes, dtypes = self._call_with_dtype("bf16")
        self.assertEqual(shapes, ((576,), (128,)))
        self.assertEqual(dtypes, (torch.float16, torch.float16))


class TestNPUDSAMetadataBuilder(unittest.TestCase):
    def test_lazy_slot_mapping_2d_reuses_first_layer_result(self):
        builder = mla_mod.NPUDSAMetadataBuilder.__new__(
            mla_mod.NPUDSAMetadataBuilder
        )
        builder.kv_cache_spec = SimpleNamespace(block_size=16)

        class Metadata(SimpleNamespace):
            pass

        metadata = Metadata(
            slot_mapping=torch.tensor([0, 17, 34], dtype=torch.long),
            slot_mapping_cache=None,
            first_layer_idx=3,
        )
        callback = builder._lazy_slot_mapping_2d(metadata)

        with patch.object(torch, "stack", wraps=torch.stack) as stack_mock:
            first = callback(3)
            reused = callback(4)

        self.assertIs(first, reused)
        self.assertEqual(first.data_ptr(), reused.data_ptr())
        stack_mock.assert_called_once()

    def test_lazy_slot_mapping_2d_does_not_retain_metadata(self):
        builder = mla_mod.NPUDSAMetadataBuilder.__new__(
            mla_mod.NPUDSAMetadataBuilder
        )
        builder.kv_cache_spec = SimpleNamespace(block_size=16)

        class Metadata:
            slot_mapping = torch.tensor([0, 17, 34], dtype=torch.long)
            slot_mapping_cache = None
            first_layer_idx = 3

        metadata = Metadata()
        metadata.get_slot_mapping_2d = builder._lazy_slot_mapping_2d(metadata)
        metadata_ref = weakref.ref(metadata)

        del metadata
        import gc; gc.collect()

        self.assertIsNone(metadata_ref())

    def test_get_batch_desc_passes_layer_idx_and_slices_once(self):
        slot_mapping = torch.tensor([0, 1, 17, 18], dtype=torch.long)
        slot_mapping_2d = torch.tensor(
            [[0, 0], [0, 1], [1, 1], [1, 2]], dtype=torch.long
        )
        callback = MagicMock(return_value=slot_mapping_2d)
        metadata = SimpleNamespace(
            num_actual_tokens=4,
            num_decode_tokens=2,
            num_decodes=2,
            num_prefills=1,
            slot_mapping=slot_mapping,
            slot_mapping_2d=None,
            get_slot_mapping_2d=callback,
            prefill=SimpleNamespace(),
            decode=SimpleNamespace(),
        )

        get_batch_desc(metadata, layer_idx=3)

        callback.assert_called_once_with(3)
        self.assertTrue(
            torch.equal(metadata.decode.slot_mapping_2d, slot_mapping_2d[:2])
        )
        self.assertTrue(
            torch.equal(metadata.prefill.slot_mapping_2d, slot_mapping_2d[2:])
        )

    def test_get_batch_desc_reuses_existing_slot_mapping_2d(self):
        slot_mapping = torch.tensor([0, 1], dtype=torch.long)
        slot_mapping_2d = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
        callback = MagicMock()
        metadata = SimpleNamespace(
            num_actual_tokens=2,
            num_decode_tokens=2,
            num_decodes=2,
            num_prefills=0,
            slot_mapping=slot_mapping,
            slot_mapping_2d=slot_mapping_2d,
            get_slot_mapping_2d=callback,
            prefill=None,
            decode=SimpleNamespace(),
        )

        get_batch_desc(metadata, layer_idx=3)

        callback.assert_not_called()
        self.assertEqual(
            metadata.decode.slot_mapping_2d.data_ptr(), slot_mapping_2d.data_ptr()
        )

    def test_init_success_with_valid_config(self):
        """Test NPUDSAMetadataBuilder.__init__ succeeds with valid config."""
        max_batched = 8192
        mock_vllm_config = MagicMock()
        mock_vllm_config.speculative_config = None
        mock_vllm_config.scheduler_config.max_num_seqs = 16
        mock_vllm_config.scheduler_config.max_num_batched_tokens = max_batched
        mock_vllm_config.kv_transfer_config = None
        mock_vllm_config.model_config.hf_config = MagicMock()
        mock_vllm_config.model_config.hf_config.param_sink_number = 128

        mock_kv_cache_spec = MagicMock()
        mock_kv_cache_spec.block_size = 16

        cudagraph_mode = MagicMock()
        cudagraph_mode.has_full_cudagraphs = MagicMock(return_value=False)

        with patch.object(mla_mod.MLACommonMetadataBuilder, "__init__", return_value=None), \
             patch.object(mla_mod, "current_platform") as mock_platform:
            mock_platform.device_type = "npu"

            # Create via __new__ and set parent-initialized attributes
            builder = mla_mod.NPUDSAMetadataBuilder.__new__(mla_mod.NPUDSAMetadataBuilder)
            builder._use_fi_prefill = False
            builder._use_cudnn_prefill = False
            builder.dcp_world_size = 1
            builder.aot_schedule = False
            builder.vllm_config = mock_vllm_config
            # super().__init__ is mocked; real builder reads compilation_config and kv_cache_spec from self
            builder.compilation_config = SimpleNamespace(
                max_cudagraph_capture_size=None,
                cudagraph_mode=cudagraph_mode,
            )
            builder.kv_cache_spec = mock_kv_cache_spec

            builder.__init__(
                kv_cache_spec=mock_kv_cache_spec,
                layer_names=["model.layers.0.self_attn"],
                vllm_config=mock_vllm_config,
                device=torch.device("npu"),
            )

            # Verify parent __init__ was called
            mla_mod.MLACommonMetadataBuilder.__init__.assert_called_once()

            # Verify attributes set by __init__
            self.assertEqual(builder.prefill_metadata_cls, mla_mod.NPUDSAPrefillMetadata)
            self.assertEqual(builder.uniform_decode_query_len, 1)
            self.assertEqual(builder.mc2_mask.shape, (16,))
            self.assertEqual(builder.mc2_mask.dtype, torch.bool)
            self.assertEqual(builder.sink_len, 128)
            self.assertEqual(builder.first_layer_idx, 0)

    def _new_builder_minimal(self):
        """
        Avoid calling real MLACommonMetadataBuilder.__init__.
        We'll create via __new__ and fill only attributes used by methods we test.
        """
        cudagraph_mode = MagicMock()
        cudagraph_mode.has_full_cudagraphs = MagicMock(return_value=False)
        b = mla_mod.NPUDSAMetadataBuilder.__new__(mla_mod.NPUDSAMetadataBuilder)
        b.uniform_decode_query_len = 1
        b.mc2_mask = torch.zeros(16, dtype=torch.bool)
        b.vllm_config = MagicMock()
        b.vllm_config.kv_transfer_config = None
        b.vllm_config.speculative_config = None
        b.vllm_config.scheduler_config.max_num_seqs = 16
        b.dcp_world_size = 1
        b._use_fi_prefill = False
        b._use_cudnn_prefill = False
        b.aot_schedule = False
        b.kv_cache_spec = MagicMock()
        b.kv_cache_spec.block_size = 16
        b.compilation_config = SimpleNamespace(
            max_cudagraph_capture_size=None,
            cudagraph_mode=cudagraph_mode,
        )
        b.decode_cudagraph_max_bs = 16
        b.slot_mapping = torch.zeros(32, dtype=torch.int64, device="cpu")
        b.ena_kvsp = False
        b.first_layer_idx = -1
        b.force_first_chunk_context = False
        return b

    def test_generate_activate_mask_sets_prefix_true(self):
        b = self._new_builder_minimal()
        mask = b._generate_activate_mask(5)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(torch.all(mask[:5]))
        self.assertTrue(torch.all(~mask[5:]))

    def test_generate_activate_mask_reuses_builder_buffer(self):
        b = self._new_builder_minimal()
        mask = b._generate_activate_mask(3)
        self.assertIs(mask, b.mc2_mask)
        self.assertTrue(torch.equal(mask[:3], torch.tensor([True, True, True])))
        self.assertTrue(torch.equal(mask[3:8], torch.tensor([False, False, False, False, False])))

    def test_build_prefill_populates_seq_lens_and_query_cumlens(self):
        b = self._new_builder_minimal()

        class _Prefill:
            def __init__(self):
                self.query_start_loc = torch.tensor([0, 2, 5], dtype=torch.long)
                self.chunked_context = None
                self.seq_lens = None
                self.query_cumlens = None

        class _Meta:
            def __init__(self):
                self.prefill = _Prefill()
                self.decode = None
                self.num_actual_tokens = 0
                self.num_reqs = 0
                self.num_prefills = 2
                self.num_decodes = 0
                self.slot_mapping = torch.tensor([], dtype=torch.long)
                self.seq_lens = torch.tensor([2, 3], dtype=torch.long)
                self.slot_mapping_cache = None

        fake_meta = _Meta()

        with patch.object(mla_mod.MLACommonMetadataBuilder, "build", return_value=fake_meta), \
             patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=False): 
            out = b.build(common_prefix_len=0, common_attn_metadata=fake_meta, fast_build=False)

        self.assertIs(out, fake_meta)
        self.assertTrue(torch.equal(out.prefill.seq_lens, torch.tensor([2, 3], dtype=torch.long)))
        self.assertTrue(torch.equal(out.prefill.query_cumlens, torch.tensor([2, 5], dtype=torch.long)))
        expect_2d = torch.stack(
            [fake_meta.slot_mapping // 16, fake_meta.slot_mapping % 16], dim=-1
        )
        self.assertTrue(torch.equal(out.get_slot_mapping_2d(), expect_2d))

    def test_build_decode_sets_num_actual_tokens_from_query_start_loc(self):
        b = self._new_builder_minimal()
        block_table = torch.tensor([[0, 1]], dtype=torch.int32)
        seq_lens = torch.tensor([8], dtype=torch.int32)
        query_start_loc_cpu = torch.tensor([0, 3], dtype=torch.int32)
        query_start_loc_device = torch.tensor([0, 3], dtype=torch.int32)

        result = b._build_decode(
            block_table_tensor=block_table,
            seq_lens_device=seq_lens,
            max_seq_len=8,
            query_start_loc_cpu=query_start_loc_cpu,
            query_start_loc_device=query_start_loc_device,
            num_decode_tokens=3,
            dcp_tot_seq_lens_device=None,
        )

        self.assertIs(result.block_table, block_table)
        self.assertIs(result.seq_lens, seq_lens)
        self.assertTrue(torch.equal(result.query_cumlens, torch.tensor([3], dtype=torch.int32)))
        self.assertEqual(int(result.num_actual_tokens), 3)

    def test_build_prefill_calls_init_cp_when_context_parallel_enabled(self):
        """Context-parallel prefill attaches an SPManager from init_cp."""
        b = self._new_builder_minimal()
        b.vllm_config.model_config = MagicMock()
        b.vllm_config.model_config.hf_config = MagicMock()
        b.vllm_config.model_config.hf_config.router_sliding_window = 0

        class _Prefill:
            def __init__(self):
                self.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
                self.chunked_context = None
                self.query_cumlens = None
                self.seq_lens = None
                self.block_table = torch.zeros(1, 2, dtype=torch.int32)

        class _Meta:
            def __init__(self):
                self.prefill = _Prefill()
                self.decode = None
                self.num_actual_tokens = 4
                self.num_reqs = 1
                self.num_prefills = 1
                self.num_decodes = 0
                self.slot_mapping = torch.zeros(4, dtype=torch.int64)
                self.seq_lens = torch.tensor([4], dtype=torch.int64)
                self.slot_mapping_cache = None
                # build() hands SPManager.init_cp a numpy copy of the CPU
                # cumulative query lengths; CommonAttentionMetadata carries it
                # alongside the device tensor.
                self.query_start_loc_cpu = torch.tensor([0, 4], dtype=torch.int32)

        fake_meta = _Meta()
        fake_sp = object()
        me = SimpleNamespace(
            parall_config=SimpleNamespace(ena_context_parallel=True),
        )
        init_cp_m = MagicMock(return_value=fake_sp)
        sp_cls = MagicMock()
        sp_cls.init_cp = init_cp_m

        with patch.object(
            mla_mod.MLACommonMetadataBuilder, "build", return_value=fake_meta
        ), \
             patch.object(mla_mod, "model_extra_config", me), \
             patch.object(mla_mod, "get_tp_group", return_value=MagicMock()), \
             patch.object(mla_mod, "SPManager", sp_cls):
            out = b.build(
                common_prefix_len=0, common_attn_metadata=fake_meta, fast_build=False
            )

        self.assertIs(out.prefill.sp_manager, fake_sp)
        init_cp_m.assert_called_once()
        ckwargs = init_cp_m.call_args.kwargs
        self.assertTrue(
            torch.equal(ckwargs["cumlens"], out.prefill.query_start_loc)
        )
        cl = ckwargs["computed_lens"]
        self.assertEqual(
            cl.detach().cpu().view(-1).tolist(),
            [0],
        )
        # mome_kernel_width is no longer passed to init_cp; a width of 0
        # (router_sliding_window unset) is what installs the paged cache_fn.
        self.assertIsNotNone(out.prefill.cache_fn)

    def test_build_decode_sets_mc2_mask_without_aligning_slot_mapping(self):
        b = self._new_builder_minimal()
        mock_kv_transfer_config = MagicMock()
        mock_kv_transfer_config.kv_role = "kv_consumer"
        b.vllm_config.kv_transfer_config = mock_kv_transfer_config
        b.vllm_config.scheduler_config.max_num_seqs = 16

        class _Decode:
            def __init__(self):
                self.mc2_mask = None
                self.num_actual_tokens = 12

        class _Meta:
            def __init__(self):
                self.decode = _Decode()
                self.prefill = None
                self.num_actual_tokens = 12
                self.num_reqs = 4
                self.num_prefills = 0
                self.num_decodes = 3
                self.slot_mapping = torch.tensor([1, 2, 3], dtype=torch.long)
                self.slot_mapping_cache = None

        fake_meta = _Meta()
        orig_slot_mapping = fake_meta.slot_mapping.clone()
        fake_common_attn_metadata = SimpleNamespace(
            seq_lens=torch.tensor([], dtype=torch.long)
        )

        fake_mask = torch.tensor(
            [True] * 12 + [False] * 4,
            dtype=torch.bool,
        )
        with patch.object(
            mla_mod.MLACommonMetadataBuilder,
            "build",
            return_value=fake_meta,
        ) as mock_super_build, \
            patch.object(
                b,
                "_generate_activate_mask",
                return_value=fake_mask,
            ) as mock_gen_mask:

            out = b.build(
                common_prefix_len=0,
                common_attn_metadata=fake_common_attn_metadata,
                fast_build=False,
            )       

        self.assertIs(out, fake_meta)
        mock_super_build.assert_called_once()
        mock_gen_mask.assert_called_once_with(12)

        self.assertIs(fake_meta.decode.mc2_mask, fake_mask)
        self.assertTrue(torch.equal(fake_meta.slot_mapping, orig_slot_mapping))
        expect_2d = torch.stack(
            [fake_meta.slot_mapping // 16, fake_meta.slot_mapping % 16], dim=-1
        )
        self.assertTrue(torch.equal(fake_meta.get_slot_mapping_2d(), expect_2d))

    def test_build_decode_no_mc2_mask_when_kv_role_is_producer(self):
        """mc2_mask should NOT be set when kv_role is kv_producer (pd-mixed uses TP)."""
        b = self._new_builder_minimal()
        mock_kv_transfer_config = MagicMock()
        mock_kv_transfer_config.kv_role = "kv_producer"
        b.vllm_config.kv_transfer_config = mock_kv_transfer_config
        b.vllm_config.scheduler_config.max_num_seqs = 16

        class _Decode:
            def __init__(self):
                self.mc2_mask = None
                self.num_actual_tokens = 12

        class _Meta:
            def __init__(self):
                self.decode = _Decode()
                self.prefill = None
                self.num_actual_tokens = 12
                self.num_reqs = 4
                self.num_prefills = 0
                self.num_decodes = 3
                self.slot_mapping = torch.tensor([1, 2, 3], dtype=torch.long)
                self.slot_mapping_cache = None

        fake_meta = _Meta()
        fake_common_attn_metadata = SimpleNamespace(
            seq_lens=torch.tensor([], dtype=torch.long)
        )

        with patch.object(
            mla_mod.MLACommonMetadataBuilder,
            "build",
            return_value=fake_meta,
        ), patch.object(
            b,
            "_generate_activate_mask",
        ) as mock_gen_mask:
            out = b.build(
                common_prefix_len=0,
                common_attn_metadata=fake_common_attn_metadata,
                fast_build=False,
            )

        # mc2_mask should remain None since kv_role is kv_producer
        self.assertIsNone(out.decode.mc2_mask)
        # _generate_activate_mask should NOT be called
        mock_gen_mask.assert_not_called()
    def test_build_decode_mc2_mask_uses_decode_num_actual_tokens(self):
        """mc2_mask should use metadata.num_actual_tokens (not decode.num_actual_tokens)."""
        b = self._new_builder_minimal()
        mock_kv_transfer_config = MagicMock()
        mock_kv_transfer_config.kv_role = "kv_consumer"
        b.vllm_config.kv_transfer_config = mock_kv_transfer_config
        b.vllm_config.scheduler_config.max_num_seqs = 16

        class _Decode:
            def __init__(self):
                self.mc2_mask = None
                self.num_actual_tokens = 5  # different from metadata.num_actual_tokens

        class _Meta:
            def __init__(self):
                self.decode = _Decode()
                self.prefill = None
                self.num_actual_tokens = 12  # this is the value that should be used
                self.num_reqs = 4
                self.num_prefills = 0
                self.num_decodes = 3
                self.slot_mapping = torch.tensor([1, 2, 3], dtype=torch.long)
                self.slot_mapping_cache = None

        fake_meta = _Meta()
        fake_common_attn_metadata = SimpleNamespace(
            seq_lens=torch.tensor([], dtype=torch.long)
        )

        fake_mask = torch.tensor(
            [True] * 12 + [False] * 4,
            dtype=torch.bool,
        )
        with patch.object(
            mla_mod.MLACommonMetadataBuilder,
            "build",
            return_value=fake_meta,
        ), patch.object(
            b,
            "_generate_activate_mask",
            return_value=fake_mask,
        ) as mock_gen_mask:
            out = b.build(
                common_prefix_len=0,
                common_attn_metadata=fake_common_attn_metadata,
                fast_build=False,
            )
        # _generate_activate_mask should be called with metadata.num_actual_tokens (12),
        # NOT with decode.num_actual_tokens (5)
        mock_gen_mask.assert_called_once_with(5)
        self.assertIs(out.decode.mc2_mask, fake_mask)

    def test_build_decode_no_mc2_mask_when_kv_transfer_config_is_none(self):
        """mc2_mask should NOT be set when kv_transfer_config is None."""
        b = self._new_builder_minimal()
        b.vllm_config.kv_transfer_config = None

        class _Decode:
            def __init__(self):
                self.mc2_mask = None
                self.num_actual_tokens = 8

        class _Meta:
            def __init__(self):
                self.decode = _Decode()
                self.prefill = None
                self.num_actual_tokens = 8
                self.num_reqs = 2
                self.num_prefills = 0
                self.num_decodes = 2
                self.slot_mapping = torch.tensor([0, 1], dtype=torch.long)
                self.slot_mapping_cache = None

        fake_meta = _Meta()
        fake_common_attn_metadata = SimpleNamespace(
            seq_lens=torch.tensor([], dtype=torch.long)
        )

        with patch.object(
            mla_mod.MLACommonMetadataBuilder,
            "build",
            return_value=fake_meta,
        ), patch.object(
            b,
            "_generate_activate_mask",
        ) as mock_gen_mask:
            out = b.build(
                common_prefix_len=0,
                common_attn_metadata=fake_common_attn_metadata,
                fast_build=False,
            )

        self.assertIsNone(out.decode.mc2_mask)
        mock_gen_mask.assert_not_called()

    def test_init_clamps_decode_cudagraph_max_bs_when_max_capture_size_set(self):
        """Covers min(decode_cudagraph_max_bs, max_cudagraph_capture_size) in dsa __init__."""
        mock_vllm_config = MagicMock()
        mock_vllm_config.speculative_config = None
        mock_vllm_config.scheduler_config.max_num_seqs = 16
        mock_vllm_config.scheduler_config.max_num_batched_tokens = 4096
        mock_vllm_config.kv_transfer_config = None
        mock_vllm_config.model_config.hf_config = MagicMock()
        mock_vllm_config.model_config.hf_config.param_sink_number = 0

        mock_kv_cache_spec = MagicMock()
        mock_kv_cache_spec.block_size = 16

        cudagraph_mode = MagicMock()
        cudagraph_mode.has_full_cudagraphs = MagicMock(return_value=False)

        with patch.object(mla_mod.MLACommonMetadataBuilder, "__init__", return_value=None), \
             patch.object(mla_mod, "current_platform") as mock_platform:
            mock_platform.device_type = "npu"

            builder = mla_mod.NPUDSAMetadataBuilder.__new__(mla_mod.NPUDSAMetadataBuilder)
            builder._use_fi_prefill = False
            builder._use_cudnn_prefill = False
            builder.dcp_world_size = 1
            builder.aot_schedule = False
            builder.vllm_config = mock_vllm_config
            builder.compilation_config = SimpleNamespace(
                max_cudagraph_capture_size=4,
                cudagraph_mode=cudagraph_mode,
            )
            builder.kv_cache_spec = mock_kv_cache_spec

            builder.__init__(
                kv_cache_spec=mock_kv_cache_spec,
                layer_names=["model.layers.0.self_attn"],
                vllm_config=mock_vllm_config,
                device=torch.device("npu"),
            )

        self.assertEqual(builder.decode_cudagraph_max_bs, 4)

    def test_build_decode_only_copies_slot_mapping_2d_into_cudagraph_buffer(self):
        """Covers cudagraph copy path when has_full_cudagraphs and decode-only batch."""
        cudagraph_mode = MagicMock()
        cudagraph_mode.has_full_cudagraphs = MagicMock(return_value=True)
        b = mla_mod.NPUDSAMetadataBuilder.__new__(mla_mod.NPUDSAMetadataBuilder)
        b.uniform_decode_query_len = 1
        b.mc2_mask = torch.zeros(8, dtype=torch.bool)
        b.vllm_config = MagicMock()
        b.vllm_config.kv_transfer_config = None
        b.dcp_world_size = 1
        b._use_fi_prefill = False
        b._use_cudnn_prefill = False
        b.aot_schedule = False
        b.kv_cache_spec = MagicMock()
        b.kv_cache_spec.block_size = 16
        b.compilation_config = SimpleNamespace(
            max_cudagraph_capture_size=None,
            cudagraph_mode=cudagraph_mode,
        )
        b.decode_cudagraph_max_bs = 8
        b.slot_mapping = torch.zeros(8, dtype=torch.int64)
        b.first_layer_idx = -1

        class _Decode:
            def __init__(self):
                self.mc2_mask = None

        class _Meta:
            def __init__(self):
                self.decode = _Decode()
                self.prefill = None
                self.num_actual_tokens = 2
                self.num_reqs = 1
                self.num_prefills = 0
                self.num_decodes = 2
                self.slot_mapping = torch.tensor([0, 1], dtype=torch.long)
                self.slot_mapping_cache = None

        fake_meta = _Meta()
        fake_common = SimpleNamespace(seq_lens=torch.tensor([], dtype=torch.long))

        with patch.object(
            mla_mod.MLACommonMetadataBuilder, "build", return_value=fake_meta
        ):
            out = b.build(
                common_prefix_len=0,
                common_attn_metadata=fake_common,
                fast_build=False,
            )

        expect = torch.stack(
            [fake_meta.slot_mapping // 16, fake_meta.slot_mapping % 16], dim=-1
        )
        self.assertIs(out, fake_meta)
        self.assertTrue(torch.equal(out.get_slot_mapping_2d(), expect))


class TestNPUDSAImplUpdateSinkKV(unittest.TestCase):
    def test_update_sink_kv_sets_sink_len_correctly(self):
        """Test update_sink_kv correctly sets sink_len from sink_compressed_kv shape."""
        # Create minimal impl stub
        impl = SimpleNamespace()
        impl.qk_nope_head_dim = 4
        impl.qk_rope_head_dim = 2
        impl.v_head_dim = 4
        sink_len = 128

        # Create test sink data (2D k_pe: update_sink_kv unsqueezes on dim=1; 3D would break torch.cat)
        sink_k_pe = torch.randn((sink_len, impl.qk_rope_head_dim), dtype=torch.float32)
        sink_compressed_kv = torch.randn((sink_len, impl.qk_nope_head_dim + impl.v_head_dim), dtype=torch.float32)

        # Call update_sink_kv
        mla_mod.NPUDSAImpl.update_sink_kv(impl, sink_k_pe, sink_compressed_kv)

        # Verify sink_len is set correctly
        self.assertEqual(impl.sink_len, sink_len, "sink_len should be set to the first dimension of sink_compressed_kv")


class TestNPUDSAImplApplySparseAttention(unittest.TestCase):
    def test_apply_sparse_attention(self):
        """Test _apply_sparse_attention with basic functionality including sink."""
        # Add indexer to impl
        indexer = SimpleNamespace()
        indexer.topk_tokens = 8
        indexer.topk_indices_buffer = torch.zeros((16, indexer.topk_tokens), dtype=torch.int32)

        impl = SimpleNamespace()
        impl.num_heads = 2
        impl.scale = 1.0
        impl.sink_len = 128
        impl.qk_nope_head_dim = 4
        impl.qk_lora_rank = 3
        impl.kv_lora_rank = 3
        impl.indexer = indexer
        # Add static method reference
        impl.get_args_from_attn_metadata = mla_mod.NPUDSAImpl.get_args_from_attn_metadata.__get__(None, mla_mod.NPUDSAImpl)

        # Create test inputs
        num_tokens = 4
        q_nope = torch.randn((num_tokens, impl.num_heads, impl.qk_nope_head_dim), dtype=torch.float32)
        q_pe = torch.randn((num_tokens, 1, 4), dtype=torch.float32)
        k_nope = torch.randn((16, impl.qk_nope_head_dim), dtype=torch.float32)
        k_rope = torch.randn((16, 4), dtype=torch.float32)
        kv_cache = (k_nope, k_rope)

        # Create metadata
        class _Decode:
            def __init__(self):
                self.query_cumulens = torch.tensor([1, 1], dtype=torch.int32)
                self.seq_lens = torch.tensor([5, 6], dtype=torch.int32)
                self.block_table = torch.zeros((2, 4), dtype=torch.int32)
                self.dcp_tot_seq_lens = None
                self.mc2_mask = None

        class _Prefill:
            def __init__(self):
                self.query_cumulens = torch.tensor([2], dtype=torch.int32)
                self.seq_lens = torch.tensor([7], dtype=torch.int32)
                self.block_table = torch.zeros((1, 4), dtype=torch.int32)
                self.query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
                self.chunked_context = None

        meta = SimpleNamespace()
        meta.num_actual_tokens = 4
        meta.num_decodes = 2
        meta.num_prefills = 1
        meta.num_decode_tokens = 2
        meta.query_start_loc = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
        meta.decode = _Decode()
        meta.prefill = _Prefill()

        # Verify get_args_from_attn_metadata works correctly
        block_table, actual_seq_lens_key, actual_seq_lens_query = \
            impl.get_args_from_attn_metadata(meta)

        # Verify block_table concatenation (decode + prefill)
        self.assertEqual(block_table.shape[0], 3, "block_table should have 3 rows (2 decode + 1 prefill)")
        self.assertEqual(block_table.shape[1], 4, "block_table should have 4 columns")

        # Verify seq_lens concatenation
        self.assertEqual(len(actual_seq_lens_key), 3, "actual_seq_lens_key should have 3 elements")

        # Verify query_start_loc slicing [1:]
        self.assertEqual(len(actual_seq_lens_query), 3, "actual_seq_lens_query should have 3 elements")

        # Now test _apply_sparse_attention by mocking the custom operation
        mock_attn_func = MagicMock()

        def mock_sparse_flash_attention_enhance(**kwargs):
            # Verify block_table and seq_lens are passed correctly
            self.assertEqual(kwargs["block_table"].shape[0], 3, "block_table should have 3 rows")
            self.assertEqual(len(kwargs["actual_seq_lengths_kv"]), 3, "actual_seq_lengths_kv should have 3 elements")
            self.assertEqual(len(kwargs["actual_seq_lengths_query"]), 3, "actual_seq_lengths_query should have 3 elements")
            # Verify sparse_indices includes sink tokens
            self.assertEqual(kwargs["sparse_indices"].shape[2], 8 + 128,
                           "sparse_indices should have sink length (128) + original topk tokens (8)")
            return [torch.randn((num_tokens, impl.num_heads, impl.kv_lora_rank), dtype=torch.float32)]

        mock_attn_func.side_effect = mock_sparse_flash_attention_enhance

        with patch.object(mla_mod.torch.ops, "custom", create=True):
            mla_mod.torch.ops.custom.npu_sparse_flash_attention_enhance = mock_attn_func

            result = mla_mod.NPUDSAImpl._apply_sparse_attention(
                impl, q_nope, q_pe, kv_cache, meta
            )

        # Verify output shape
        expected_shape = (num_tokens, impl.num_heads, impl.kv_lora_rank)
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(mock_attn_func.called, "custom attention function should be called")


class TestNPUDSAImplGetArgsFromAttnMetadata(unittest.TestCase):
    """Test the static get_args_from_attn_metadata method."""

    def test_get_args_both_decode_and_prefill(self):
        """Test get_args_from_attn_metadata with both decode and prefill."""
        # Create metadata
        class _Decode:
            def __init__(self):
                self.query_cumulens = torch.tensor([1, 1], dtype=torch.int32)
                self.seq_lens = torch.tensor([5, 6], dtype=torch.int32)
                self.block_table = torch.zeros((2, 4), dtype=torch.int32)
                self.dcp_tot_seq_lens = None
                self.mc2_mask = None

        class _Prefill:
            def __init__(self):
                self.query_cumulens = torch.tensor([2], dtype=torch.int32)
                self.seq_lens = torch.tensor([7], dtype=torch.int32)
                self.block_table = torch.zeros((1, 4), dtype=torch.int32)
                self.query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
                self.chunked_context = None

        meta = SimpleNamespace()
        meta.num_decodes = 2
        meta.num_prefills = 1
        meta.query_start_loc = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
        meta.decode = _Decode()
        meta.prefill = _Prefill()

        # Call the static method
        block_table, actual_seq_lens_key, actual_seq_lens_query = \
            mla_mod.NPUDSAImpl.get_args_from_attn_metadata(meta)

        # Verify block_table concatenation (decode + prefill)
        self.assertEqual(block_table.shape[0], 3, "block_table should have 3 rows (2 decode + 1 prefill)")
        self.assertEqual(block_table.shape[1], 4, "block_table should have 4 columns")

        # Verify seq_lens concatenation
        self.assertEqual(len(actual_seq_lens_key), 3, "actual_seq_lens_key should have 3 elements")
        self.assertTrue(torch.equal(actual_seq_lens_key[:2], torch.tensor([5, 6], dtype=torch.int32)),
                       "first two elements should be decode seq_lens")
        self.assertTrue(torch.equal(actual_seq_lens_key[2:], torch.tensor([7], dtype=torch.int32)),
                       "last element should be prefill seq_lens")

        # Verify query_start_loc slicing [1:]
        self.assertEqual(len(actual_seq_lens_query), 3, "actual_seq_lens_query should have 3 elements")
        expected_query_lens = torch.tensor([1, 2, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(actual_seq_lens_query, expected_query_lens),
                       "actual_seq_lens_query should be query_start_loc[1:]")


if __name__ == "__main__":
    unittest.main()
