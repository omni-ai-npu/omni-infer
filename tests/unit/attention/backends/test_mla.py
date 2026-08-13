# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import types
import unittest
from unittest.mock import MagicMock, patch
from typing import Generic, TypeVar
import pytest
import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@pytest.fixture(scope="module")
def mla_setup():
    """Module-scoped fixture for MLA tests - properly manages mock lifecycle."""
    MetadataT = TypeVar("MetadataT")

    class AttentionMetadataBuilder(Generic[MetadataT]):
        def __init__(self, *args, **kwargs):
            if len(args) == 4:
                kv_cache_spec, layer_names, vllm_config, device = args
            else:
                kv_cache_spec = kwargs["kv_cache_spec"]
                layer_names = kwargs["layer_names"]
                vllm_config = kwargs["vllm_config"]
                device = kwargs["device"]
            self.kv_cache_spec = kv_cache_spec
            self.layer_names = layer_names
            self.vllm_config = vllm_config
            self.device = device

        def _init_reorder_batch_threshold(self, *args, **kwargs):
            self.reorder_batch_threshold = 0

    utils_mod = types.ModuleType("vllm.v1.attention.backends.utils")
    utils_mod.PAD_SLOT_ID = -1
    utils_mod.split_decodes_and_prefills = MagicMock(return_value=(1, 0, 1, 0))
    utils_mod.compute_causal_conv1d_metadata = MagicMock()
    utils_mod.dcp_local_seq_lens = MagicMock()
    utils_mod.get_dcp_local_seq_lens = MagicMock()
    utils_mod.get_per_layer_parameters = MagicMock()
    utils_mod.infer_global_hyperparameters = MagicMock()
    attn_backend_mod = types.ModuleType("vllm.v1.attention.backend")
    attn_backend_mod.AttentionMetadataBuilder = AttentionMetadataBuilder
    attn_backend_mod.CommonAttentionMetadata = MagicMock()
    attn_backend_mod.AttentionCGSupport = MagicMock()

    attn_backend_mod.AttentionBackend = AttentionBackend
    attn_backend_mod.AttentionImpl = AttentionImpl
    attn_backend_mod.AttentionLayer = AttentionLayer
    attn_backend_mod.AttentionType = AttentionType
    attn_backend_mod.MultipleOf = MultipleOf

    # Create a real class for AttentionMetadata to avoid metaclass conflicts.
    class AttentionMetadata:
        pass
    attn_backend_mod.AttentionMetadata = AttentionMetadata

    # Create a real class for MLAAttentionImpl to avoid metaclass conflicts.
    A = TypeVar("A")
    class MLAAttentionImpl(Generic[A]):
        pass
    attn_backend_mod.MLAAttentionImpl = MLAAttentionImpl

    # vLLM 0.25.1: copy missing public attrs from real modules so transitive
    # imports still resolve. Test overrides above remain in place.
    import importlib as _importlib

    try:
        _real = _importlib.import_module("vllm.v1.attention.backends.utils")
        for _attr in dir(_real):
            if _attr.startswith("_"):
                continue
            if not hasattr(utils_mod, _attr):
                setattr(utils_mod, _attr, getattr(_real, _attr))
    except Exception:
        pass
    try:
        _real = _importlib.import_module("vllm.v1.attention.backend")
        for _attr in dir(_real):
            if _attr.startswith("_"):
                continue
            if not hasattr(attn_backend_mod, _attr):
                setattr(attn_backend_mod, _attr, getattr(_real, _attr))
    except Exception:
        pass

    utils_mod_patcher = patch.dict(
        "sys.modules",
        {
            "vllm.v1.attention.backends.utils": utils_mod,
            "vllm.v1.attention.backend": attn_backend_mod,
        },
    )
    utils_mod_patcher.start()

    # Mock forward context with capturing=False
    mock_forward_ctx = MagicMock()
    mock_forward_ctx.capturing = False
    mock_forward_ctx.batch_descriptor = None
    mock_forward_ctx.no_compile_layers = {}

    forward_ctx_mod = types.ModuleType("vllm.forward_context")
    forward_ctx_mod.ForwardContext = MagicMock()
    forward_ctx_mod.get_forward_context = MagicMock(return_value=mock_forward_ctx)
    forward_ctx_mod.is_forward_context_available = MagicMock(return_value=True)
    forward_ctx_mod.set_forward_context = MagicMock(return_value=MagicMock())
    forward_ctx_mod.BatchDescriptor = MagicMock()
    forward_ctx_mod.capturing = False
    try:
        _real = _importlib.import_module("vllm.forward_context")
        for _attr in dir(_real):
            if _attr.startswith("_"):
                continue
            if not hasattr(forward_ctx_mod, _attr):
                setattr(forward_ctx_mod, _attr, getattr(_real, _attr))
    except Exception:
        pass
    forward_context_mod_patcher = patch.dict("sys.modules", {
        "vllm.forward_context": forward_ctx_mod
    })
    forward_context_mod_patcher.start()

    try:
        from omni_npu.attention.backends.mla import (
            NPUMLAImpl,
            NPUMLAMetadata,
            NPUMLADecodeMetadata,
            NPUMLAMetadataBuilder,
            NPUMLABackend,
        )
    except Exception as e:
        print(f"❌ FAILED to import omni_npu classes: {e}")
        import traceback

        traceback.print_exc()
        forward_context_mod_patcher.stop()
        utils_mod_patcher.stop()
        raise


    # Yield the imported classes and helper functions to tests
    yield {
        "impl": NPUMLAImpl,
        "metadata": NPUMLAMetadata,
        "decode_metadata": NPUMLADecodeMetadata,
        "backend": NPUMLABackend,
        "builder": NPUMLAMetadataBuilder,
    }

    # Cleanup after all tests in module
    forward_context_mod_patcher.stop()
    utils_mod_patcher.stop()


@pytest.mark.unit
class TestNPUAttentionBackendMLAUtilsFunc(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup_fixture(self, mla_setup):
        """Auto-use fixture that stores data for unittest.TestCase methods."""
        self.mla_setup = mla_setup

    def test_lazy_slot_mapping_2d_reuses_cached_tensor(self):
        builder_cls = self.mla_setup["builder"]
        builder = builder_cls.__new__(builder_cls)
        builder.kv_cache_spec = types.SimpleNamespace(block_size=16)
        metadata = types.SimpleNamespace(
            slot_mapping=torch.tensor([0, 17, 34], dtype=torch.long),
            slot_mapping_2d=None,
        )
        callback = builder._lazy_slot_mapping_2d(metadata)

        with patch.object(torch, "stack", wraps=torch.stack) as stack_mock:
            first = callback()
            reused = callback()

        self.assertIs(first, reused)
        self.assertIs(first, metadata.slot_mapping_2d)
        self.assertEqual(first.data_ptr(), reused.data_ptr())
        stack_mock.assert_called_once()

    def test_metadata_default_slot_mapping_2d_can_be_overridden(self):
        metadata = self.mla_setup["metadata"](
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

    def test_npu_mla_reshape_kv_cache(self):
        backend = self.mla_setup["backend"]

        self.assertEqual(backend.get_name(), "NPUMLA")
        self.assertEqual(backend.get_builder_cls(), self.mla_setup["builder"])
        self.assertEqual(backend.get_impl_cls(), self.mla_setup["impl"])

        num_blocks = 10
        block_size = 128
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        dtype = torch.bfloat16

        total_bf16_elements = (
            num_blocks * block_size * (kv_lora_rank + qk_rope_head_dim)
        )
        total_bytes = total_bf16_elements * 2
        raw_tensor = torch.empty(total_bytes, dtype=torch.uint8)

        kv_cache_spec = AttentionSpec(
            block_size=block_size,
            num_kv_heads=8,
            head_size=128,
            dtype=dtype,
        )
        result = backend.reshape_kv_cache(
            raw_tensor=raw_tensor,
            num_blocks=num_blocks,
            kv_cache_spec=kv_cache_spec,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        nope_cache, rope_cache = result
        self.assertEqual(nope_cache.shape, (num_blocks, block_size, kv_lora_rank))
        self.assertEqual(rope_cache.shape, (num_blocks, block_size, qk_rope_head_dim))
        self.assertEqual(nope_cache.dtype, dtype)
        self.assertEqual(rope_cache.dtype, dtype)

        print("Backend contract test passed!")



@pytest.mark.unit
class TestNPUAttentionBackendMLANpuMlaImpl(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup_fixture(self, mla_setup):
        """Auto-use fixture that stores data for unittest.TestCase methods."""
        self.mla_setup = mla_setup
        import omni_npu.attention.backends.mla as mla_mod

        with patch.object(
            mla_mod.model_extra_config.operator_opt_config,
            "use_aicpu_fa_tiling",
            False,
        ):
            yield

    def _new_builder_for_current_build(
        self,
        *,
        kv_transfer_config=None,
        sink_len: int = 0,
    ):
        import omni_npu.attention.backends.mla as mla_mod

        class FakePrefillMetadata:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeMetadata:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        builder = self.mla_setup["builder"].__new__(self.mla_setup["builder"])
        builder.device = torch.device("cpu")
        builder.reorder_batch_threshold = 0
        builder.vllm_config = MagicMock()
        builder.vllm_config.kv_transfer_config = kv_transfer_config
        builder.dcp_world_size = 1
        builder.query_len_support = mla_mod.QueryLenSupport.VARLEN
        builder._use_fi_prefill = False
        builder._use_cudnn_prefill = False
        builder._use_trtllm_ragged_prefill = False
        builder.aot_schedule = False
        builder.page_size = 16
        builder.chunked_prefill_workspace_size = 0
        builder.chunked_prefill_workspace = None
        builder.kv_cache_spec = MagicMock(block_size=16)
        builder.model_config = MagicMock()
        builder.model_config.get_head_size.return_value = 128
        builder.mc2_mask = torch.zeros(256, dtype=torch.bool)
        builder.prefill_metadata_cls = FakePrefillMetadata
        builder.metadata_cls = FakeMetadata
        if sink_len:
            builder.sink_len = sink_len
        return builder, mla_mod

    def _make_common_for_current_build(
        self,
        *,
        seq_lens,
        query_start_loc,
        block_table=None,
    ):
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        query_start_loc = torch.tensor(query_start_loc, dtype=torch.int32)
        num_reqs = len(seq_lens)
        if block_table is None:
            block_table = torch.zeros(num_reqs, 2, dtype=torch.int32)
        return MagicMock(
            num_reqs=num_reqs,
            num_actual_tokens=int(query_start_loc[-1].item()),
            max_query_len=1,
            max_seq_len=int(seq_lens.max().item()) if num_reqs else 0,
            block_table_tensor=block_table,
            slot_mapping=torch.arange(int(query_start_loc[-1].item()), dtype=torch.int64),
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            dcp_local_seq_lens=seq_lens,
        )


    def test_builder_build_prefill_with_sink_len_updates_seq_lens(self):
        builder, mla_mod = self._new_builder_for_current_build(sink_len=128)
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[0],
            query_start_loc=[0, 3],
        )

        with (
            patch.object(
                mla_mod,
                "split_decodes_and_prefills",
                return_value=(0, 1, 0, 3),
            ),
            patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                False,
            ),
        ):
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        self.assertEqual(result.prefill.sink_len, 128)
        self.assertEqual(result.prefill.query_cumlens, [3])
        self.assertEqual(result.prefill.seq_lens, [128])

    def test_builder_build_prefill_creates_chunked_context_metadata(self):
        builder, mla_mod = self._new_builder_for_current_build()
        builder.chunked_prefill_workspace_size = 2
        builder.chunked_prefill_workspace = object()
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[6],
            query_start_loc=[0, 1],
        )

        with torch.device("cpu"), patch.object(
            mla_mod,
            "split_decodes_and_prefills",
            return_value=(0, 1, 0, 1),
        ):
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        chunked = result.prefill.chunked_context
        self.assertIsNotNone(chunked)
        self.assertEqual(chunked.max_seq_lens, [2, 2, 1])
        self.assertEqual(chunked.seq_tot, [2, 2, 1])
        self.assertEqual(chunked.chunk_total_token.tolist(), [2, 2, 1])
        self.assertTrue(torch.equal(
            chunked.seq_lens,
            torch.tensor([[2], [2], [1]], dtype=torch.int32, device="cpu"),
        ))
        self.assertIs(chunked.workspace, builder.chunked_prefill_workspace)

    def test_update_sink_kv(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16
        sink_len = 128
        qk_rope_head_dim = 64
        kv_lora_rank = 512
        sink_k_pe = torch.randn(sink_len, qk_rope_head_dim, dtype=dtype, device=device)
        sink_compressed_kv = torch.randn(
            sink_len, kv_lora_rank, dtype=dtype, device=device
        )
        impl = self.mla_setup["impl"](
            num_heads=32,
            head_size=128,
            scale=1.0 / (128**0.5),
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            logits_soft_cap=None,
            kv_sharing_target_layer_name=None,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            kv_lora_rank=512,
            q_lora_rank=256,
            qk_head_dim=192,
            kv_b_proj=torch.nn.Linear(512, 8 * 256, bias=False).to(dtype),
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )
        impl.update_sink_kv(sink_k_pe, sink_compressed_kv)
        self.assertEqual(impl.sink_len, sink_len)
        self.assertEqual(impl.sink_k_pe.shape, (sink_len, 1, qk_rope_head_dim))
        self.assertTrue(
            torch.equal(impl.sink_compressed_kv, sink_compressed_kv.unsqueeze(1))
        )

    def test_build_decode_with_aicpu_fa_tiling(self):
        import omni_npu.attention.backends.mla as mla_mod

        builder = self.mla_setup["builder"].__new__(self.mla_setup["builder"])
        builder.reorder_batch_threshold = 0
        builder.vllm_config = MagicMock()
        builder.vllm_config.kv_transfer_config = None
        builder.dcp_world_size = 1

        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        seq_lens_device = torch.tensor([10, 20], dtype=torch.int32)
        query_start_loc_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)
        query_start_loc_device = torch.tensor([0, 1, 2], dtype=torch.int32)

        with patch.object(
            mla_mod.model_extra_config.operator_opt_config,
            "use_aicpu_fa_tiling",
            True,
        ):
            result = builder._build_decode(
                block_table_tensor=block_table,
                seq_lens_device=seq_lens_device,
                max_seq_len=20,
                query_start_loc_cpu=query_start_loc_cpu,
                query_start_loc_device=query_start_loc_device,
                num_decode_tokens=2,
                dcp_tot_seq_lens_device=None,
            )

        self.assertIs(result.seq_lens, seq_lens_device)
        self.assertTrue(torch.equal(result.query_cumlens, query_start_loc_device[1:]))
        self.assertEqual(result.num_tokens, 2)

    def test_build_decode_without_aicpu_fa_tiling(self):
        import omni_npu.attention.backends.mla as mla_mod

        builder = self.mla_setup["builder"].__new__(self.mla_setup["builder"])
        builder.reorder_batch_threshold = 0
        builder.vllm_config = MagicMock()
        builder.vllm_config.kv_transfer_config = None
        builder.dcp_world_size = 1

        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        seq_lens_device = torch.tensor([10, 20], dtype=torch.int32)
        query_start_loc_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)
        query_start_loc_device = torch.tensor([0, 1, 2], dtype=torch.int32)

        with patch.object(
            mla_mod.model_extra_config.operator_opt_config,
            "use_aicpu_fa_tiling",
            False,
        ):
            result = builder._build_decode(
                block_table_tensor=block_table,
                seq_lens_device=seq_lens_device,
                max_seq_len=20,
                query_start_loc_cpu=query_start_loc_cpu,
                query_start_loc_device=query_start_loc_device,
                num_decode_tokens=2,
                dcp_tot_seq_lens_device=None,
            )

        self.assertEqual(result.seq_lens, [10, 20])
        self.assertEqual(result.query_cumlens, [1, 2])
        self.assertEqual(result.num_tokens, 2)

    def test_build_decode_with_zero_sink_len_keeps_seq_lens(self):
        import omni_npu.attention.backends.mla as mla_mod

        builder = self.mla_setup["builder"].__new__(self.mla_setup["builder"])
        builder.reorder_batch_threshold = 0
        builder.vllm_config = MagicMock()
        builder.vllm_config.kv_transfer_config = None
        builder.dcp_world_size = 1
        builder.sink_len = 0

        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        seq_lens_device = torch.tensor([0, 20], dtype=torch.int32)
        query_start_loc_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)
        query_start_loc_device = torch.tensor([0, 1, 2], dtype=torch.int32)

        with patch.object(
            mla_mod.model_extra_config.operator_opt_config,
            "use_aicpu_fa_tiling",
            False,
        ):
            result = builder._build_decode(
                block_table_tensor=block_table,
                seq_lens_device=seq_lens_device,
                max_seq_len=20,
                query_start_loc_cpu=query_start_loc_cpu,
                query_start_loc_device=query_start_loc_device,
                num_decode_tokens=2,
                dcp_tot_seq_lens_device=None,
            )

        self.assertEqual(result.seq_lens, [0, 20])
        self.assertEqual(result.query_cumlens, [1, 2])
        self.assertEqual(result.num_tokens, 2)

    def test_build_decode_with_kv_transfer_config(self):
        mock_kv_transfer_config = MagicMock()
        mock_kv_transfer_config.kv_role = "kv_consumer"
        builder, mla_mod = self._new_builder_for_current_build(
            kv_transfer_config=mock_kv_transfer_config,
        )
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[10],
            query_start_loc=[0, 1],
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        )

        with patch.object(
            mla_mod,
            "split_decodes_and_prefills",
            return_value=(1, 0, 1, 0),
        ):
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        self.assertIsNotNone(result.decode.mc2_mask)
        self.assertTrue(result.decode.mc2_mask[0].item())

    def test_build_decode_with_sink_len(self):
        builder, mla_mod = self._new_builder_for_current_build(sink_len=64)
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[0, 5],
            query_start_loc=[0, 1, 2],
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        )

        with (
            patch.object(
                mla_mod,
                "split_decodes_and_prefills",
                return_value=(2, 0, 2, 0),
            ),
            patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                False,
            ),
        ):
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        self.assertEqual(result.decode.sink_len, 64)
        self.assertEqual(result.decode.seq_lens, [64, 5])

    def test_build_prefill_with_sink_len_and_fa_tiling(self):
        builder, mla_mod = self._new_builder_for_current_build(sink_len=64)
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[0],
            query_start_loc=[0, 5],
        )

        with (
            patch.object(
                mla_mod,
                "split_decodes_and_prefills",
                return_value=(0, 1, 0, 5),
            ),
            patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                True,
            ),
        ):
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        self.assertEqual(result.prefill.sink_len, 64)
        self.assertTrue(torch.equal(
            result.prefill.query_cumlens, 
            torch.tensor([5])
        ))
        self.assertTrue(torch.equal(
            result.prefill.seq_lens, 
            torch.tensor([64])
        ))

    def test_build_decode_with_sink_len_and_fa_tiling(self):
        builder, mla_mod = self._new_builder_for_current_build(sink_len=64)
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[0, 1],
            query_start_loc=[0, 1, 2],
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        )

        with (
            patch.object(
                mla_mod,
                "split_decodes_and_prefills",
                return_value=(2, 0, 2, 0),
            ),
            patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                True,
            ),
        ):
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        self.assertEqual(result.decode.sink_len, 64)
        self.assertTrue(torch.equal(
            result.decode.seq_lens, 
            torch.tensor([64, 1])
        ))

    def test_build_decode_no_mc2_mask_when_kv_role_is_producer(self):
        """mc2_mask should NOT be set when kv_role is kv_producer (pd-mixed uses TP)."""
        mock_kv_transfer_config = MagicMock()
        mock_kv_transfer_config.kv_role = "kv_producer"
        builder, mla_mod = self._new_builder_for_current_build(
            kv_transfer_config=mock_kv_transfer_config,
        )
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[10],
            query_start_loc=[0, 1],
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        )

        with patch.object(
            mla_mod,
            "split_decodes_and_prefills",
            return_value=(1, 0, 1, 0),
        ), patch.object(
            builder,
            "generate_activate_mask",
        ) as mock_gen_mask:
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        # mc2_mask should remain None since kv_role is kv_producer
        self.assertIsNone(result.decode.mc2_mask)
        # generate_activate_mask should NOT be called
        mock_gen_mask.assert_not_called()

    def test_build_decode_mc2_mask_uses_query_cumlens_last(self):
        """mc2_mask should use decode.query_cumlens[-1] as the token count."""
        mock_kv_transfer_config = MagicMock()
        mock_kv_transfer_config.kv_role = "kv_consumer"
        builder, mla_mod = self._new_builder_for_current_build(
            kv_transfer_config=mock_kv_transfer_config,
        )
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[10, 5],
            query_start_loc=[0, 1, 3],
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        )

        with patch.object(
            mla_mod,
            "split_decodes_and_prefills",
            return_value=(2, 0, 3, 0),
        ), patch.object(
            builder,
            "generate_activate_mask",
            wraps=builder.generate_activate_mask,
        ) as mock_gen_mask:
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        # generate_activate_mask should be called with query_cumlens[-1] (=3)
        mock_gen_mask.assert_called_once_with(3)
        self.assertIsNotNone(result.decode.mc2_mask)

    def test_build_decode_no_mc2_mask_when_kv_transfer_config_is_none(self):
        """mc2_mask should NOT be set when kv_transfer_config is None."""
        builder, mla_mod = self._new_builder_for_current_build(
            kv_transfer_config=None,
        )
        common_attn_metadata = self._make_common_for_current_build(
            seq_lens=[10],
            query_start_loc=[0, 1],
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        )

        with patch.object(
            mla_mod,
            "split_decodes_and_prefills",
            return_value=(1, 0, 1, 0),
        ), patch.object(
            builder,
            "generate_activate_mask",
        ) as mock_gen_mask:
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        # mc2_mask should remain None since kv_transfer_config is None
        self.assertIsNone(result.decode.mc2_mask)
        mock_gen_mask.assert_not_called()
