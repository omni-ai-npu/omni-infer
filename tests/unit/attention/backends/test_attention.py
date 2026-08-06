# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NPUAttentionBackendImpl that do NOT require actual NPU hardware.
These tests use mocking to verify the logic and API contracts.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib
import torch
import pytest
from typing import Generic, TypeVar
import types


import torch_npu

from vllm.v1.attention.backend import AttentionBackend, AttentionImpl, AttentionLayer, AttentionType
from vllm.v1.kv_cache_interface import AttentionSpec


def create_mock_modules(name: str, pure_mock: bool = False):
    if pure_mock:
        res_mod = MagicMock()
    else:
        res_mod = types.ModuleType(name)
    res_mod.__path__ = []
    res_mod.__spec__ = None

    return res_mod


def make_model_extra_config(
    kv_nz: bool = False,
    enable_kv_rmsnorm_rope_cache: bool = False,
    use_aicpu_fa_tiling: bool = False
):
    return types.SimpleNamespace(
        operator_opt_config=types.SimpleNamespace(
            kv_nz=kv_nz,
            enable_kv_rmsnorm_rope_cache=enable_kv_rmsnorm_rope_cache,
            use_aicpu_fa_tiling=use_aicpu_fa_tiling
        )
    )


class SinkLenZeroThenNone:

    def __init__(self, impl):
        self.impl = impl

    def __eq__(self, other):
        self.impl.sink_len = None
        return other == 0


@pytest.fixture
def npu_attention_classes(monkeypatch):
    """
    Module-scoped fixture that sets up NPU attention backend classes.
    Uses patches to mock vLLM dependencies and provides clean isolation.
    """
    # Define a TypeVar for the metadata type
    MetadataT = TypeVar('MetadataT')

    # Make the mock class generic
    class AttentionMetadataBuilder(Generic[MetadataT]):
        def __init__(self, *args, **kwargs):
            if len(args) == 4:
                kv_cache_spec, layer_names, vllm_config, device = args
            else:
                kv_cache_spec = kwargs['kv_cache_spe']
                layer_names = kwargs['layer_names']
                vllm_config = kwargs['vllm_config']
                device = kwargs['device']
            self.kv_cache_spec = kv_cache_spec
            self.layer_names = layer_names
            self.vllm_config = vllm_config
            self.device = device
        def _init_reorder_batch_threshold(self, reorder_batch_threshold, default_threshold):
            self.reorder_batch_threshold = max(reorder_batch_threshold, 0)

    attn_backend_mod = types.ModuleType('vllm.v1.attention.backend')
    attn_backend_mod.AttentionMetadataBuilder = AttentionMetadataBuilder
    attn_backend_mod.CommonAttentionMetadata = MagicMock()
    attn_backend_mod.AttentionCGSupport = MagicMock()

    # Add missing imports to attn_backend_mod
    attn_backend_mod.AttentionBackend = AttentionBackend
    attn_backend_mod.AttentionImpl = AttentionImpl
    attn_backend_mod.AttentionLayer = AttentionLayer
    attn_backend_mod.AttentionType = AttentionType

    # Create a real class for AttentionMetadata to avoid metaclass conflict
    # When vllm.v1.attention.backends.mla.common.MLACommonMetadata inherits from it.
    class AttentionMetadata:
        pass

    attn_backend_mod.AttentionMetadata = AttentionMetadata

    utils_mod = types.ModuleType('vllm.v1.attention.backends.utils')
    utils_mod.split_decodes_and_prefills = MagicMock(return_value=(1, 0, 1, 0))
    utils_mod.PAD_SLOT_ID = -1

    monkeypatch.setattr("vllm.v1.attention.backend", attn_backend_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.utils",
                        utils_mod)

    vllm_distributed_mod = create_mock_modules("vllm.distributed")
    fake_tp = MagicMock()
    fake_tp.world_size = 1
    fake_tp.rank_in_group = 0
    vllm_distributed_mod.GroupCoordinator = MagicMock
    vllm_distributed_mod.get_tp_group = lambda: fake_tp
    monkeypatch.setitem(sys.modules, "vllm.distributed", vllm_distributed_mod)
    monkeypatch.setitem(sys.modules, "vllm.distributed.eplb", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm.distributed.eplb.eplb_state",
                        MagicMock())
    monkeypatch.setitem(sys.modules,
                        "vllm.distributed.get_tensor_model_parallel_rank",
                        MagicMock())
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.get_tensor_model_parallel_world_size",
        MagicMock())

    vllm_device_comm_mod = create_mock_modules(
        "vllm.distributed.device_communicators")
    monkeypatch.setitem(sys.modules, "vllm.distributed.device_communicators",
                        vllm_device_comm_mod)

    vllm_device_comm_shm_mod = create_mock_modules(
        "vllm.distributed.device_communicators.shm_object_storage", True)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.shm_object_storage",
        vllm_device_comm_shm_mod)

    vllm_pynccl_allocator_mod = create_mock_modules(
        "vllm.distributed.device_communicators.pynccl_allocator", True)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.pynccl_allocator",
        vllm_pynccl_allocator_mod)

    vllm_parallel_state_mod = create_mock_modules(
        "vllm.distributed.parallel_state", True)
    fake_dcp = MagicMock()
    fake_dcp.world_size = 1
    fake_dcp.rank_in_group = 0
    fake_pcp = MagicMock()
    fake_pcp.world_size = 1
    fake_pcp.rank_in_group = 0
    vllm_parallel_state_mod.get_tp_group = lambda: fake_tp
    vllm_parallel_state_mod.get_dcp_group = lambda: fake_dcp
    vllm_parallel_state_mod.get_pcp_group = lambda: fake_pcp
    monkeypatch.setitem(sys.modules,
                        "vllm.distributed.parallel_state.get_tp_group",
                        lambda: fake_tp)
    monkeypatch.setitem(sys.modules,
                        "vllm.distributed.parallel_state.get_dcp_group",
                        lambda: fake_dcp)
    monkeypatch.setitem(sys.modules,
                        "vllm.distributed.parallel_state.get_pcp_group",
                        lambda: fake_pcp)

    mock_forward_ctx = MagicMock()
    mock_forward_ctx.capturing = False
    mock_forward_ctx.batch_descriptor = None

    forward_ctx_mod = types.ModuleType('vllm.forward_context')
    forward_ctx_mod.get_forward_context = MagicMock(return_value=mock_forward_ctx)
    forward_ctx_mod.BatchDescriptor = MagicMock()
    forward_ctx_mod.capturing = False
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_ctx_mod)

    model_utils_mod = types.ModuleType("vllm.model_executor.models.utils")
    model_utils_mod.extract_layer_index = MagicMock(return_value=0)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.utils", model_utils_mod)

    dsa_mod = types.ModuleType("omni.attention.backends.dsa")
    dsa_mod.NPUDSABackend = MagicMock()
    monkeypatch.setitem(sys.modules, "omni.attention.backends.dsa", dsa_mod)

    mla_mod = types.ModuleType("omni.attention.backends.mla")
    mla_mod.NPUMLABackend = MagicMock()
    monkeypatch.setitem(sys.modules, "omni.attention.backends.mla", mla_mod)

    mome_mod = types.ModuleType("omni.attention.backends.mome")
    mome_mod.NPUPanguMomeBackend = MagicMock()
    monkeypatch.setitem(sys.modules, "omni.attention.backends.mome", mome_mod)

    try:
        import omni.attention.backends.attention as attn_mod
        import omni.attention.backends as backends_mod
        importlib.reload(attn_mod)
        importlib.reload(backends_mod)

        # Now it's safe to import omni_npu — its backend will inherit from REAL base classes
        from omni.attention.backends import (
            NPUAttentionBackendImpl as _impl,
            NPUMetadata as _meta,
            NPUAttentionBackend as _backend,
            NPUAttentionMetadataBuilder as _builder,
        )

        _builder._init_reorder_batch_threshold = lambda *args, **kwargs: None

        # Yield a dictionary with all the classes
        yield {
            'attention_module': attn_mod,
            'NPUAttentionBackendImpl': _impl,
            'NPUMetadata': _meta,
            'NPUAttentionBackend': _backend,
            'NPUAttentionMetadataBuilder': _builder,
            'AttentionType': AttentionType,
        }
    except Exception as e:
        print(f"❌ FAILED to import omni_npu classes: {e}")
        import traceback
        traceback.print_exc()
        raise


@pytest.mark.unit
class TestNPUAttentionBackendDefault(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_classes(self, npu_attention_classes):
        """Inject the attention classes from the module-scoped fixture."""
        self.impl_interface_cls = npu_attention_classes['NPUAttentionBackend']
        self.impl_cls = npu_attention_classes['NPUAttentionBackendImpl']
        self.AttentionType = npu_attention_classes['AttentionType']
        self.attention_module = npu_attention_classes['attention_module']
        self.npu_attention_classes = npu_attention_classes
        with patch.object(
            self.attention_module.model_extra_config.operator_opt_config,
            "use_aicpu_fa_tiling",
            False,
        ):
            yield

    def test_backend_properties(self):
        backend = self.impl_interface_cls()
        self.assertIn(torch.float16, backend.get_supported_dtypes())
        self.assertEqual(backend.get_name(), "VLLM_NPU_ATTN")
        self.assertIs(backend.get_impl_cls(), self.npu_attention_classes['NPUAttentionBackendImpl'])
        self.assertIs(backend.get_metadata_cls(), self.npu_attention_classes['NPUMetadata'])
        self.assertIs(backend.get_builder_cls(), self.npu_attention_classes['NPUAttentionMetadataBuilder'])
        self.assertTrue(backend.supports_attn_type(AttentionType.DECODER))
        self.assertTrue(backend.supports_attn_type(AttentionType.ENCODER_DECODER))
        self.assertFalse(backend.supports_attn_type(AttentionType.ENCODER))
        self.assertFalse(backend.supports_attn_type(AttentionType.ENCODER_ONLY))

    def test_kv_cache_shape_and_reshape(self):
        shape = self.impl_interface_cls.get_kv_cache_shape(
            num_blocks=10,
            block_size=16,
            num_kv_heads=4,
            head_size=128
        )
        self.assertEqual(shape, (10, 16, 512))  # 4 * 128 = 512

        raw = torch.randn(2 * 10 * 16 * 512, dtype=torch.bfloat16)
        kv_cache_spec = AttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
        )
        k_cache, v_cache = self.impl_interface_cls.reshape_kv_cache(
            raw, num_blocks=10, kv_cache_spec=kv_cache_spec,
        )
        self.assertEqual(k_cache.shape, (10, 16, 512))
        self.assertEqual(v_cache.shape, (10, 16, 512))
        self.assertTrue(torch.equal(raw[:10*16*512].view(10,16,512), k_cache))

        kv_cache_spec_with_value_head = types.SimpleNamespace(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            head_size_v=64,
            dtype=torch.bfloat16,
        )
        raw = torch.zeros(10 * 16 * 512 + 10 * 16 * 256, dtype=torch.bfloat16)
        k_cache, v_cache = self.impl_interface_cls.reshape_kv_cache(
            raw, num_blocks=10, kv_cache_spec=kv_cache_spec_with_value_head,
        )
        self.assertEqual(k_cache.shape, (10, 16, 512))
        self.assertEqual(v_cache.shape, (10, 16, 256))

    def test_kv_nz_cache_shape_and_reshape(self):
        self.assertEqual(
            self.attention_module.get_nz_dim(cache_dtype=torch.bfloat16),
            16,
        )
        self.assertEqual(
            self.attention_module.get_nz_dim(cache_dtype=torch.int8),
            32,
        )
        with patch.object(
            self.attention_module,
            "model_extra_config",
            make_model_extra_config(kv_nz=True),
        ):
            self.assertEqual(
                self.impl_interface_cls.get_kv_cache_shape(
                    num_blocks=10,
                    block_size=16,
                    num_kv_heads=4,
                    head_size=128,
                ),
                (10, 32, 16, 16),
            )
            self.assertEqual(
                self.impl_interface_cls.get_kv_cache_shape(
                    num_blocks=10,
                    block_size=16,
                    num_kv_heads=4,
                    head_size=128,
                    cache_dtype_str="int8",
                ),
                (10, 16, 16, 32),
            )
            self.assertEqual(
                self.impl_interface_cls.get_kv_cache_shape(
                    num_blocks=10,
                    block_size=16,
                    num_kv_heads=4,
                    head_size=256,
                ),
                (10, 64, 16, 16),
            )

            spec = types.SimpleNamespace(
                block_size=16,
                num_kv_heads=4,
                head_size=128,
                head_size_v=64,
                dtype=torch.bfloat16,
                cache_dtype_str=None,
            )
            key_shape = (10, 32, 16, 16)
            value_shape = (10, 16, 16, 16)
            raw = torch.zeros(
                10 * 32 * 16 * 16 + 10 * 16 * 16 * 16,
                dtype=torch.bfloat16,
            )
            k_cache, v_cache = self.impl_interface_cls.reshape_kv_cache(
                raw,
                num_blocks=10,
                kv_cache_spec=spec,
            )
            self.assertEqual(k_cache.shape, key_shape)
            self.assertEqual(v_cache.shape, value_shape)

@pytest.mark.unit
class TestNPUAttentionBackendDefaultMetadataBuilder(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_classes(self, npu_attention_classes):
        """Inject the attention classes from the module-scoped fixture."""
        self.metadata_builder_cls = npu_attention_classes['NPUAttentionMetadataBuilder']
        self.attention_module = npu_attention_classes['attention_module']
        self.npu_attention_classes = npu_attention_classes

    def test_metadata_builder(self):
        # Define a minimal CommonAttentionMetadata (normally from vLLm)
        class CommonAttentionMetadata:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def compute_num_computed_tokens(self) -> torch.Tensor:
                query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
                return self.seq_lens - query_lens

        spec = MagicMock()
        spec.block_size = 16
        vllm_config = MagicMock()
        vllm_config.reorder_batch_threshold = 0
        vllm_config.compilation_config = None
        builder = self.metadata_builder_cls(
            kv_cache_spec=spec,
            layer_names=["test"],
            vllm_config=vllm_config,
            device=torch.device("npu")
        )

        common_meta = CommonAttentionMetadata(
            num_actual_tokens=20,
            query_start_loc=torch.tensor([0, 10, 20]),
            seq_lens=torch.tensor([10, 10]),
            max_query_len=10,
            block_table_tensor=torch.randint(0, 100, (2, 10)),
            slot_mapping=torch.arange(20),
            context_lens=None,
            max_context_len=None,
            qkv_format="TND",
        )

        with patch(
                'vllm.v1.attention.backends.utils.split_decodes_and_prefills',
                return_value=(0, 2, 0, 20)
        ), patch(
                'omni.attention.backends.attention.split_decodes_and_prefills',
                return_value=(0, 2, 0, 20)
        ), patch.object(
                self.attention_module,
                "model_extra_config",
                make_model_extra_config(kv_nz=True),
        ):
            meta = builder.build(common_prefix_len=0,
                                 common_attn_metadata=common_meta)

        self.assertIsInstance(meta, self.npu_attention_classes['NPUMetadata'])
        self.assertEqual(meta.num_actual_tokens, 20)
        self.assertEqual(meta.num_prefills, 2)
        self.assertEqual(meta.query_start_loc, [0, 10, 20])
        self.assertEqual(meta.seq_lens, [10, 10])
        self.assertEqual(meta.max_query_len, 10)
        self.assertTrue(meta.is_pure_prefill_without_prefix_and_use_kvnz)

    def test_metadata_builder_marks_prefill_with_prefix_false(self):
        class CommonAttentionMetadata:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def compute_num_computed_tokens(self) -> torch.Tensor:
                query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
                return self.seq_lens - query_lens

        spec = MagicMock()
        spec.block_size = 16
        spec.head_size = 128
        vllm_config = MagicMock()
        vllm_config.reorder_batch_threshold = 0
        vllm_config.compilation_config = None
        builder = self.metadata_builder_cls(
            kv_cache_spec=spec,
            layer_names=["test"],
            vllm_config=vllm_config,
            device=torch.device("npu"),
        )

        common_meta = CommonAttentionMetadata(
            num_actual_tokens=4,
            query_start_loc=torch.tensor([0, 4]),
            seq_lens=torch.tensor([10]),
            max_query_len=4,
            block_table_tensor=torch.randint(0, 100, (1, 10)),
            slot_mapping=torch.arange(4),
        )

        with patch(
                'omni.attention.backends.attention.split_decodes_and_prefills',
                return_value=(0, 1, 0, 4)):
            meta = builder.build(common_prefix_len=0,
                                 common_attn_metadata=common_meta)

        self.assertFalse(meta.is_pure_prefill_without_prefix_and_use_kvnz)


@pytest.mark.unit
class TestNPUAttentionBackendDefaultImpl(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_classes(self, npu_attention_classes):
        """Inject the attention classes from the module-scoped fixture."""
        self.impl_cls = npu_attention_classes['NPUAttentionBackendImpl']
        self.metadata_cls = npu_attention_classes['NPUMetadata']
        self.AttentionType = npu_attention_classes['AttentionType']
        self.attention_module = npu_attention_classes['attention_module']

    def test_init_success(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
        )
        self.assertEqual(impl.num_heads, 8)
        self.assertEqual(impl.num_kv_heads, 4)
        self.assertEqual(impl.head_size, 128)

    def test_init_passes_required_args_to_attention_impl(self):
        with patch.object(AttentionImpl, "__init__", return_value=None) as mock_init:
            self.impl_cls(
                num_heads=8,
                head_size=128,
                scale=1.0,
                num_kv_heads=4,
                attn_type=self.AttentionType.DECODER,
            )

        args = mock_init.call_args.args
        required_args = args[:3] if args and args[0] == 8 else args[1:4]
        self.assertEqual(required_args, (8, 128, 1.0))

    def test_init_tolerates_abstract_attention_impl(self):
        with patch.object(
            AttentionImpl, "__init__", side_effect=NotImplementedError
        ):
            impl = self.impl_cls(
                num_heads=8,
                head_size=128,
                scale=1.0,
                num_kv_heads=4,
                attn_type=self.AttentionType.DECODER,
            )

        self.assertEqual(impl.num_heads, 8)
        self.assertEqual(impl.head_size, 128)

    def test_init_invalid_attn_type_raises(self):
        with self.assertRaises(NotImplementedError):
            self.impl_cls(
                num_heads=8,
                head_size=128,
                scale=1.0,
                attn_type="ENCODER",
            )

    def test_init_num_heads_not_divisible_by_kv_heads_raises(self):
        with self.assertRaises(RuntimeError):
            self.impl_cls(
                num_heads=7,
                head_size=128,
                scale=1.0,
                num_kv_heads=3,
                attn_type=self.AttentionType.DECODER,
            )

    def test_forward_calls_npu_fused_infer_attention_score_v2(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        query = torch.randn(batch_size, 8 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        key = torch.randn(batch_size, 4 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        value = torch.randn(batch_size, 4 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        kv_cache = (torch.zeros(batch_size ** 2, 16, 4 * 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device), torch.zeros(100, 16, 4 * 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device))

        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10)).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device),
            query_start_loc=[0, 10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        attn_output = torch.randn(batch_size, 8, 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        output = torch.empty_like(attn_output).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        prefill_output=output.clone()

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            elif indices.ndim > 1:
                raise NotImplementedError("Only 1D or [N,1] indices supported in mock")

            num_indices = indices.shape[0]
            if updates.shape[0] != num_indices:
                updates = updates[:num_indices]

            tensor[indices] = updates
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch_npu.npu_fused_infer_attention_score_v2', return_value=(prefill_output,)) as mock_decode:
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            # self.assertEqual(mock_scatter.call_count, 2)
            mock_decode.assert_called_once()
            args, kwargs = mock_decode.call_args
            self.assertEqual(kwargs['num_query_heads'], 8)
            self.assertEqual(kwargs['num_key_value_heads'], 4)
            self.assertEqual(kwargs['input_layout'], "TND")
            self.assertAlmostEqual(kwargs['softmax_scale'], 0.125)
            self.assertIs(result, output)

    def test_forward_calls_npu_fused_infer_attention_sink_full_attention(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
            sink_len=0,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        query = torch.randn(batch_size, 8 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        key = torch.randn(batch_size, 4 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        value = torch.randn(batch_size, 4 , 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        kv_cache = (torch.zeros(batch_size ** 2, 16, 4 * 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device), torch.zeros(100, 16, 4 * 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device))

        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10)).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device),
            query_start_loc=[0, 10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        attn_output = torch.randn(batch_size, 8, 128).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        output = torch.empty_like(attn_output).to(self.impl_cls.SHARE_MASK_TRIL_SPARSE.device)
        prefill_output=output.clone()

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            elif indices.ndim > 1:
                raise NotImplementedError("Only 1D or [N,1] indices supported in mock")

            num_indices = indices.shape[0]
            if updates.shape[0] != num_indices:
                updates = updates[:num_indices]

            tensor[indices] = updates
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch.ops.custom.npu_fused_infer_attention_sink', return_value=(prefill_output,)) as mock_decode, \
         patch.object(
             self.attention_module.model_extra_config.operator_opt_config,
             "use_aicpu_fa_tiling",
             False,
         ):
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            # self.assertEqual(mock_scatter.call_count, 2)
            mock_decode.assert_called_once()
            args, kwargs = mock_decode.call_args
            self.assertEqual(kwargs["sparse_mode"], 3)
            self.assertEqual(kwargs["sink_number"], 0)
            self.assertEqual(kwargs["actual_seq_qlen"], [10])
            self.assertEqual(kwargs['num_query_heads'], 8)
            self.assertEqual(kwargs['num_key_value_heads'], 4)
            self.assertEqual(kwargs['input_layout'], "TND")
            self.assertAlmostEqual(kwargs['softmax_scale'], 0.125)
            self.assertIs(result, output)

    def test_forward_calls_npu_fused_infer_attention_sink(self):
        sink = torch.randn(8)
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            sliding_window=256,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name="mock_layer",
            sinks=sink,
            head_size_v=128,
            sink_len=128,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(batch_size, 8, 128, device=device)
        key = torch.randn(batch_size, 4, 128, device=device)
        value = torch.randn(batch_size, 4, 128, device=device)
        kv_cache = (
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        prefill_output = output.clone()
        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10],
            seq_lens=[10 + 128],
            max_query_len=1,
            slot_mapping=torch.arange(10, device=device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        with patch('torch.ops.custom.npu_fused_infer_attention_sink', return_value=(prefill_output,)) as mock_decode, \
             patch.object(
                 self.attention_module.model_extra_config.operator_opt_config,
                 "use_aicpu_fa_tiling",
                 False,
             ):
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            kwargs = mock_decode.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["pre_tokens"], 256)
            self.assertEqual(kwargs["next_tokens"], 0)
            self.assertEqual(kwargs["sink_number"], 128)
            self.assertEqual(kwargs["actual_seq_qlen"], [10])

    def test_forward_passes_sink_and_sliding_kwargs(self):
        sink = torch.randn(8)
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            sliding_window=256,
            attn_type=self.AttentionType.DECODER,
            sinks=sink,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(batch_size, 8, 128, device=device)
        key = torch.randn(batch_size, 4, 128, device=device)
        value = torch.randn(batch_size, 4, 128, device=device)
        kv_cache = (
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        prefill_output = output.clone()
        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10, device=device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            tensor[indices] = updates[:indices.shape[0]]
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch_npu.npu_fused_infer_attention_score_v2', return_value=(prefill_output,)) as mock_decode:
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            kwargs = mock_decode.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["pre_tokens"], 256)
            self.assertEqual(kwargs["next_tokens"], 0)
            self.assertTrue(torch.equal(kwargs["learnable_sink"], sink.view(8)))

    def test_forward_default_sink_none_and_non_sliding_kwargs(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(batch_size, 8, 128, device=device)
        key = torch.randn(batch_size, 4, 128, device=device)
        value = torch.randn(batch_size, 4, 128, device=device)
        kv_cache = (
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        prefill_output = output.clone()
        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10],
            seq_lens=[10],
            max_query_len=1,
            slot_mapping=torch.arange(10, device=device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            tensor[indices] = updates[:indices.shape[0]]
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch_npu.npu_fused_infer_attention_score_v2', return_value=(prefill_output,)) as mock_decode:
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            kwargs = mock_decode.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 3)
            self.assertNotIn("pre_tokens", kwargs)
            self.assertNotIn("next_tokens", kwargs)
            self.assertNotIn("learnable_sink", kwargs)

    def test_forward_kv_nz_updates_cache_and_uses_nz_cache_shape(self):
        with patch.object(
            self.attention_module,
            "model_extra_config",
            make_model_extra_config(kv_nz=True),
        ):
            impl = self.impl_cls(
                num_heads=8,
                head_size=128,
                scale=0.125,
                num_kv_heads=4,
                attn_type=self.AttentionType.DECODER,
            )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        num_tokens = 20
        block_num = 8
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(num_tokens, 8, 128, device=device)
        key = torch.randn(num_tokens, 4, 128, device=device)
        value = torch.randn(num_tokens, 4, 128, device=device)
        kv_cache = (
            torch.zeros(block_num, 32, 16, 16, device=device),
            torch.zeros(block_num, 32, 16, 16, device=device),
        )
        output = torch.empty_like(query)
        attn_output = torch.zeros_like(query)
        softmax_lse = torch.zeros(num_tokens, device=device)

        metadata = self.metadata_cls(
            num_actual_tokens=num_tokens,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10, 20],
            seq_lens=[12, 12],
            max_query_len=10,
            slot_mapping=torch.arange(num_tokens, device=device),
            num_prefills=2,
            num_decodes=0,
            num_decode_tokens=0,
            is_pure_prefill_without_prefix_and_use_kvnz=False,
        )

        with patch.object(
            self.attention_module,
            "model_extra_config",
            make_model_extra_config(kv_nz=True),
        ), patch(
            "torch_npu.npu_scatter_pa_kv_cache",
            create=True,
        ) as mock_scatter, patch(
            "torch_npu._npu_fused_infer_attention_score_v2_infer_output",
            return_value=(attn_output, softmax_lse),
            create=True,
        ) as mock_infer_output, patch(
            "torch_npu.npu_fused_infer_attention_score_v2",
            return_value=(attn_output,),
            create=True,
        ) as mock_decode:
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            mock_scatter.assert_called_once()
            scatter_args = mock_scatter.call_args.args
            self.assertEqual(scatter_args[0].shape, (num_tokens, 4, 128))
            self.assertEqual(scatter_args[1].shape, (num_tokens, 4, 128))
            self.assertTrue(torch.equal(scatter_args[4], metadata.slot_mapping))

            mock_infer_output.assert_called_once()
            infer_kwargs = mock_infer_output.call_args.kwargs
            self.assertEqual(infer_kwargs["value"].shape, (block_num, 4, 8, 16, 16))
            self.assertIs(infer_kwargs["block_table"], metadata.block_tables)

            mock_decode.assert_called_once()
            kwargs = mock_decode.call_args.kwargs
            self.assertEqual(kwargs["key"].shape, (block_num, 4, 8, 16, 16))
            self.assertEqual(kwargs["value"].shape, (block_num, 4, 8, 16, 16))
            self.assertEqual(kwargs["block_size"], 16)
            self.assertEqual(kwargs["actual_seq_kvlen"], metadata.seq_lens)
            self.assertIs(result, output)

    def _run_forward_kv_nz_deferred_prefill_cache_update(self, capturing):
        sink = torch.randn(8)
        with patch.object(
            self.attention_module,
            "model_extra_config",
            make_model_extra_config(kv_nz=True),
        ):
            impl = self.impl_cls(
                num_heads=8,
                head_size=128,
                scale=0.125,
                num_kv_heads=4,
                sliding_window=256,
                attn_type=self.AttentionType.DECODER,
                sinks=sink,
            )
        impl.sink_len = SinkLenZeroThenNone(impl)

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0
        layer.layer_name = "test_layer"

        num_tokens = 20
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(num_tokens, 8, 128, device=device)
        key = torch.randn(num_tokens, 4, 128, device=device)
        value = torch.randn(num_tokens, 4, 128, device=device)
        kv_cache = (
            torch.zeros(8, 32, 16, 16, device=device),
            torch.zeros(8, 32, 16, 16, device=device),
        )
        output = torch.empty_like(query)
        prefill_output = torch.zeros_like(query)
        decode_output = torch.ones_like(query)
        softmax_lse = torch.zeros(num_tokens, device=device)

        metadata = self.metadata_cls(
            num_actual_tokens=num_tokens,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 8, 20],
            seq_lens=[8, 12],
            max_query_len=12,
            slot_mapping=torch.arange(num_tokens, device=device),
            num_prefills=2,
            num_decodes=0,
            num_decode_tokens=0,
            is_pure_prefill_without_prefix_and_use_kvnz=True,
        )
        forward_context = MagicMock()
        forward_context.capturing = capturing

        with patch.object(
            self.attention_module,
            "model_extra_config",
            make_model_extra_config(kv_nz=True),
        ), patch.object(
            self.attention_module,
            "get_forward_context",
            MagicMock(return_value=forward_context),
        ), patch.object(
            self.attention_module,
            "capture_graph_task",
        ) as mock_capture, patch(
            "torch_npu.npu_scatter_pa_kv_cache",
            create=True,
        ) as mock_scatter, patch(
            "torch_npu._npu_fused_infer_attention_score_v2_infer_output",
            return_value=(prefill_output, softmax_lse),
            create=True,
        ) as mock_infer_output, patch(
            "torch_npu.npu_fused_infer_attention_score_v2",
            return_value=(decode_output,),
            create=True,
        ) as mock_decode:
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            mock_infer_output.assert_called_once()
            infer_kwargs = mock_infer_output.call_args.kwargs
            self.assertEqual(infer_kwargs["query"].shape, (num_tokens, 8, 128))
            self.assertEqual(infer_kwargs["value"].shape, (num_tokens, 4, 128))
            self.assertEqual(infer_kwargs["input_layout"], "TND")
            self.assertIsNone(infer_kwargs["block_table"])

            mock_scatter.assert_called_once()
            scatter_args = mock_scatter.call_args.args
            self.assertEqual(scatter_args[0].shape, (num_tokens, 4, 128))
            self.assertEqual(scatter_args[1].shape, (num_tokens, 4, 128))
            self.assertIs(scatter_args[2], kv_cache[0])
            self.assertIs(scatter_args[3], kv_cache[1])
            self.assertTrue(torch.equal(scatter_args[4], metadata.slot_mapping))

            if capturing:
                mock_decode.assert_not_called()
                mock_capture.assert_called_once()
                kwargs = mock_capture.call_args.kwargs["op_kwargs"]
                self.assertEqual(mock_capture.call_args.kwargs["num_tokens"],
                                 num_tokens)
                self.assertEqual(mock_capture.call_args.kwargs["layer_name"],
                                 "test_layer")
            else:
                mock_capture.assert_not_called()
                mock_decode.assert_called_once()
                kwargs = mock_decode.call_args.kwargs

            self.assertEqual(kwargs["key"].shape, (num_tokens, 4, 128))
            self.assertEqual(kwargs["value"].shape, (num_tokens, 4, 128))
            self.assertEqual(kwargs["block_size"], 0)
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["pre_tokens"], 256)
            self.assertEqual(kwargs["next_tokens"], 0)
            self.assertEqual(kwargs["actual_seq_qlen"], [8, 20])
            self.assertEqual(kwargs["actual_seq_kvlen"], [8, 12])
            self.assertTrue(torch.equal(kwargs["learnable_sink"], sink.view(8)))
            self.assertIs(result, output)

    def test_forward_kv_nz_deferred_prefill_cache_update(self):
        self._run_forward_kv_nz_deferred_prefill_cache_update(capturing=False)

    def test_forward_kv_nz_deferred_prefill_cache_update_capturing(self):
        self._run_forward_kv_nz_deferred_prefill_cache_update(capturing=True)

    def test_forward_calls_npu_fused_infer_attention_score_bsnd(self):
        head_size = 256
        impl = self.impl_cls(
            num_heads=8,
            head_size=head_size,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        query = torch.randn(20, 8, head_size)
        key = torch.randn(20, 4, head_size)
        value = torch.randn(20, 4, head_size)
        kv_cache = (torch.zeros(100, 16, 4 * head_size), torch.zeros(100, 16, 4 * head_size))
        output = torch.empty_like(query)

        metadata = self.metadata_cls(
            num_actual_tokens=20,
            block_tables=torch.randint(0, 100, (2, 10)),
            query_start_loc=[0, 10, 20],
            seq_lens=[10, 10],
            max_query_len=10,
            slot_mapping=torch.arange(20),
            num_prefills=2,
        )

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            elif indices.ndim > 1:
                raise NotImplementedError("Only 1D or [N,1] indices supported in mock")

            num_indices = indices.shape[0]
            if updates.shape[0] != num_indices:
                updates = updates[:num_indices]

            tensor[indices] = updates
            return tensor
        def fake_fused_infer_attention_score(**kwargs):
            q = kwargs.get('query')
            return (torch.zeros_like(q),)
        def fake_fused_infer_attention_score_infer_output(**kwargs):
            q = kwargs.get('query')
            return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[1])

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
            patch('torch_npu._npu_fused_infer_attention_score_v2_infer_output', side_effect=fake_fused_infer_attention_score_infer_output), \
            patch('torch_npu.npu_fused_infer_attention_score_v2', side_effect=fake_fused_infer_attention_score) as mock_decode:
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            mock_decode.assert_called_once()
            self.assertIs(result, output)

    def test_forward_requires_output_tensor(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            attn_type=self.AttentionType.DECODER,
        )
        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        query = torch.randn(1, 1024)
        key = value = torch.randn(1, 512)
        kv_cache = (torch.zeros(10, 16, 512), torch.zeros(10, 16, 512))
        metadata = self.metadata_cls(
            num_actual_tokens=1,
            block_tables=torch.zeros(1, 1, dtype=torch.int32),
            query_start_loc=[0, 1],
            seq_lens=[1],
            slot_mapping=torch.tensor([0], dtype=torch.int64),
            num_prefills=0,
        )

        with self.assertRaises(AssertionError):
            impl.forward(layer, query, key, value, kv_cache, metadata, output=None)

    def test_forward_k_v_scale_not_one_raises(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=1.0,
            attn_type=self.AttentionType.DECODER,
        )
        layer = MagicMock()
        layer._k_scale_float = 0.5
        layer._v_scale_float = 1.0

        query = torch.randn(1, 1024)
        key = value = torch.randn(1, 512)
        kv_cache = (torch.zeros(10, 16, 512), torch.zeros(10, 16, 512))
        output = torch.empty_like(query)
        metadata = self.metadata_cls(
            num_actual_tokens=1,
            block_tables=torch.zeros(1, 1, dtype=torch.int32),
            query_start_loc=[0, 1],
            seq_lens=[1],
            slot_mapping=torch.tensor([0], dtype=torch.int64),
            num_prefills=0,
        )

        with self.assertRaises(RuntimeError):
            impl.forward(layer, query, key, value, kv_cache, metadata, output=output)

    def test_forward_bsnd_all_paths_for_future_removal(self):
        """
        SINGLE TEST FOR BSND BRANCH (head_size == 256).
        This test covers pure decode (with graph capturing) and hybrid batches 
        (with 0-length prefills).
        TODO: Delete this entire test when npu_fused_infer_attention_score_v2 supports head_size=256.
        """
        head_size = 256
        impl = self.impl_cls(num_heads=8, head_size=head_size, scale=0.125, num_kv_heads=4, attn_type=self.AttentionType.DECODER)
        layer = MagicMock()
        layer._k_scale_float = 1.0; layer._v_scale_float = 1.0; layer.layer_name = "test_layer"

        query = torch.randn(20, 8, head_size)
        key = torch.randn(20, 4, head_size)
        value = torch.randn(20, 4, head_size)
        kv_cache = (torch.zeros(100, 16, 4 * head_size), torch.zeros(100, 16, 4 * head_size))
        output = torch.empty_like(query)

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1: indices = indices.squeeze(1)
            tensor[indices] = updates[:indices.shape[0]]
            return tensor

        def fake_fused_infer_attention_score(**kwargs):
            q = kwargs.get('query')
            return (torch.zeros_like(q),)
        def fake_fused_infer_attention_score_infer_output(**kwargs):
            q = kwargs.get('query')
            return torch.zeros_like(q), torch.zeros(q.shape[0], q.shape[1])

        metadata_decode = self.metadata_cls(
            num_actual_tokens=20, block_tables=torch.randint(0, 100, (2, 10)),
            query_start_loc=[0, 10, 20], seq_lens=[10, 10], max_query_len=1,
            slot_mapping=torch.arange(20), num_prefills=0, num_decodes=2, num_decode_tokens=20
        )
        forward_ctx_mod = sys.modules["vllm.forward_context"]
        original_ctx = forward_ctx_mod.get_forward_context()
        original_ctx.capturing = True

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
             patch('torch_npu._npu_fused_infer_attention_score_v2_infer_output', side_effect=fake_fused_infer_attention_score_infer_output), \
             patch('omni.attention.backends.attention.capture_graph_task') as mock_capture:
            impl.forward(layer=layer, query=query, key=key, value=value, kv_cache=kv_cache, attn_metadata=metadata_decode, output=output)
            mock_capture.assert_called_once()
        
        original_ctx.capturing = False
        metadata_hybrid = self.metadata_cls(
            num_actual_tokens=12, block_tables=torch.randint(0, 100, (2, 10)),
            query_start_loc=[0, 2, 12, 12], seq_lens=[2, 10, 0], max_query_len=10,
            slot_mapping=torch.arange(12), num_prefills=2, num_decodes=1, num_decode_tokens=2
        )

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
             patch('torch_npu._npu_fused_infer_attention_score_v2_infer_output', side_effect=fake_fused_infer_attention_score_infer_output), \
             patch('torch_npu.npu_fused_infer_attention_score_v2', side_effect=fake_fused_infer_attention_score) as mock_decode:
            impl.forward(layer=layer, query=query, key=key, value=value, kv_cache=kv_cache, attn_metadata=metadata_hybrid, output=output)
            mock_decode.assert_called_once()

@pytest.mark.unit
class TestNPUAttentionBackendCrossAttention(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_classes(self, npu_attention_classes, monkeypatch):
        self.backend_cls = npu_attention_classes['NPUAttentionBackend']
        self.impl_cls = npu_attention_classes['NPUAttentionBackendImpl']
        self.metadata_cls = npu_attention_classes['NPUMetadata']
        self.AttentionType = npu_attention_classes['AttentionType']
        self.attention_module = npu_attention_classes['attention_module']
        mock_forward_ctx = MagicMock()
        mock_forward_ctx.capturing = False
        mock_forward_ctx.batch_descriptor = None
        monkeypatch.setattr(
            self.attention_module,
            "get_forward_context",
            MagicMock(return_value=mock_forward_ctx),
        )

    def test_backend_supports_cross_attention(self):
        backend = self.backend_cls()
        self.assertTrue(backend.supports_attn_type(self.AttentionType.ENCODER_DECODER))

    def test_init_encoder_only_raises(self):
        with self.assertRaises(NotImplementedError):
            self.impl_cls(
                num_heads=8,
                head_size=128,
                scale=0.125,
                num_kv_heads=4,
                attn_type=self.AttentionType.ENCODER_ONLY,
            )

    def test_cross_attention_first_step_uses_non_causal_settings(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.ENCODER_DECODER,
        )
        self.assertTrue(impl.is_cross_attention)

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(batch_size, 8, 128, device=device)
        key = torch.randn(batch_size, 4, 128, device=device)
        value = torch.randn(batch_size, 4, 128, device=device)
        kv_cache = (
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        prefill_output = output.clone()
        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10],
            seq_lens=[1500],
            max_query_len=1,
            slot_mapping=torch.arange(10, device=device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        def fake_scatter_nd_update_(tensor, indices, updates):
            if indices.ndim == 2 and indices.shape[1] == 1:
                indices = indices.squeeze(1)
            tensor[indices] = updates[:indices.shape[0]]
            return tensor

        with patch('torch_npu.npu_scatter_nd_update_', side_effect=fake_scatter_nd_update_), \
         patch('torch_npu.npu_fused_infer_attention_score_v2', return_value=(prefill_output,)) as mock_decode:
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            kwargs = mock_decode.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 0)
            self.assertIsNone(kwargs["atten_mask"])
            self.assertEqual(kwargs["actual_seq_kvlen"], [1500])

    def test_cross_attention_later_step_skips_cache_write(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.ENCODER_DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        batch_size = 10
        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(batch_size, 8, 128, device=device)
        kv_cache = (
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
            torch.zeros(batch_size ** 2, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        decode_output = output.clone()
        metadata = self.metadata_cls(
            num_actual_tokens=10,
            block_tables=torch.randint(0, 100, (2, 10), device=device),
            query_start_loc=[0, 10],
            seq_lens=[1500],
            max_query_len=1,
            slot_mapping=torch.empty(0, dtype=torch.int64, device=device),
            num_prefills=0,
            num_decode_tokens=8,
            num_decodes=2,
        )

        with patch('torch_npu.npu_scatter_nd_update_') as mock_scatter, \
         patch('torch_npu.npu_fused_infer_attention_score_v2', return_value=(decode_output,)) as mock_decode:
            result = impl.forward(
                layer=layer,
                query=query,
                key=None,
                value=None,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
            mock_scatter.assert_not_called()
            mock_decode.assert_called_once()
            self.assertIs(result, output)

    def test_cross_attention_rejects_mismatched_key_value(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.ENCODER_DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(1, 8, 128, device=device)
        key = torch.randn(1, 4, 128, device=device)
        output = torch.empty_like(query)
        metadata = self.metadata_cls(
            num_actual_tokens=1,
            block_tables=torch.randint(0, 16, (1, 1), device=device),
            query_start_loc=[0, 1],
            seq_lens=[1],
            max_query_len=1,
            slot_mapping=torch.empty(0, dtype=torch.int64, device=device),
            num_prefills=0,
            num_decode_tokens=1,
            num_decodes=1,
        )

        with self.assertRaisesRegex(NotImplementedError, "key and value"):
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=None,
                kv_cache=(
                    torch.empty(1, device=device),
                    torch.empty(1, device=device),
                ),
                attn_metadata=metadata,
                output=output,
            )

    def test_cross_attention_unsupported_input_fails_hard(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=self.AttentionType.ENCODER_DECODER,
        )

        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        device = self.impl_cls.SHARE_MASK_TRIL_SPARSE.device
        query = torch.randn(4, 8, 128, device=device)
        key = torch.randn(4, 4, 128, device=device)
        value = torch.randn(4, 4, 128, device=device)
        kv_cache = (
            torch.zeros(16, 16, 4 * 128, device=device),
            torch.zeros(16, 16, 4 * 128, device=device),
        )
        output = torch.empty_like(query)
        metadata = self.metadata_cls(
            num_actual_tokens=4,
            block_tables=torch.randint(0, 16, (1, 4), device=device),
            query_start_loc=[0, 4],
            seq_lens=[4],
            max_query_len=4,
            slot_mapping=torch.empty(0, dtype=torch.int64, device=device),
            num_prefills=0,
            num_decode_tokens=0,
            num_decodes=1,
        )

        with self.assertRaises(NotImplementedError):
            impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )
