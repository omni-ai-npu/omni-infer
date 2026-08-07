# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import types
import unittest
from unittest.mock import MagicMock, patch
from typing import Generic, TypeVar
import pytest
import torch

from vllm.v1.attention.backend import AttentionBackend, AttentionImpl, AttentionLayer, AttentionType
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

    # Create a real class for AttentionMetadata to avoid metaclass conflict
    # When vllm.v1.attention.backends.mla.common.MLACommonMetadata inherits from it.
    class AttentionMetadata:
        pass
    attn_backend_mod.AttentionMetadata = AttentionMetadata

    # Create a real class for MLAAttentionImpl to avoid metaclass conflict
    # When vllm.v1.attention.backends.mla.common.MLACommonBaseImpl inherits from it.
    A = TypeVar("A")
    class MLAAttentionImpl(Generic[A]):
        pass
    attn_backend_mod.MLAAttentionImpl = MLAAttentionImpl

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

    def make_fake_metadata(
        num_prefills: int,
        num_decode_tokens: int,
        seq_lens: list[int],
        num_reqs: int,
        max_query_len: int,
        max_seq_len: int,
        query_start_loc: torch.Tensor,
        block_size: int = 128,
        device: torch.device = torch.device("cpu"),
    ):
        total_tokens = sum(seq_lens)
        query_start_loc = [0]
        cumsum = 0
        for i, slen in enumerate(seq_lens):
            if i < len(seq_lens) - num_decode_tokens:
                cumsum += slen
            else:
                cumsum += 1
            query_start_loc.append(cumsum)

        query_start_loc = torch.tensor(
            query_start_loc, dtype=torch.int32, device=device
        )
        max_blocks_per_seq = max((s + block_size - 1) // block_size for s in seq_lens)
        block_table = torch.arange(
            len(seq_lens) * max_blocks_per_seq, dtype=torch.int32, device=device
        ).view(len(seq_lens), max_blocks_per_seq)
        slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device=device)

        class FakePrefillMeta:
            def __init__(self):
                self.query_start_loc = query_start_loc

        prefill_meta = FakePrefillMeta() if num_prefills > 0 else None
        decode_meta = None
        if num_decode_tokens > 0:
            decode_meta = NPUMLADecodeMetadata(
                block_table=block_table[-num_decode_tokens:],
                seq_lens=seq_lens[-num_decode_tokens:],
                query_cumlens=query_start_loc[1:].tolist(),
                dcp_tot_seq_lens=None,
            )
        batch_size = 1
        prompt_len = 1
        total_prefill_tokens = batch_size * prompt_len
        query_start_loc = torch.tensor(
            [0, total_prefill_tokens], dtype=torch.int32, device=device
        )

        metadata = NPUMLAMetadata(
            prefill=prefill_meta,
            decode=decode_meta,
            num_actual_tokens=total_tokens,
            num_prefills=num_prefills,
            num_decodes=len(seq_lens) - (num_prefills > 0),
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            num_reqs=batch_size,
            max_query_len=prompt_len,
            max_seq_len=prompt_len,
            query_start_loc=query_start_loc,
        )
        return metadata

    # Yield the imported classes and helper functions to tests
    yield {
        "impl": NPUMLAImpl,
        "metadata": NPUMLAMetadata,
        "decode_metadata": NPUMLADecodeMetadata,
        "backend": NPUMLABackend,
        "builder": NPUMLAMetadataBuilder,
        "make_fake_metadata": make_fake_metadata,
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
        self.assertEqual(backend.get_metadata_cls(), self.mla_setup["metadata"])
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

    def test_v_up_proj(self):
        device = torch.device("npu:0")
        dtype = torch.bfloat16

        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        hidden_size = num_heads * v_head_dim
        kv_lora_rank = 512
        num_kv_heads = 8
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )

        batch_size = 1
        prompt_len = 1

        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )

            B = 2
            x = torch.randn(num_heads * B, kv_lora_rank, dtype=dtype, device=device)

            impl.W_UV = torch.zeros(
                num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device
            )
            for h in range(num_heads):
                impl.W_UV[h, :v_head_dim, :] = torch.eye(
                    v_head_dim, dtype=dtype, device=device
                )

            out = impl._v_up_proj(x)

            expected = torch.zeros(B, hidden_size, dtype=dtype, device=device)
            for b in range(B):
                for h in range(num_heads):
                    token_idx = h * B + b
                    expected[b, h * v_head_dim : (h + 1) * v_head_dim] = x[
                        token_idx, :v_head_dim
                    ]

            self.assertTrue(torch.allclose(out, expected, atol=1e-3))
            print("_v_up_proj test passed!")


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

    def _make_impl_for_attention_call(
        self,
        *,
        num_heads=32,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    ):
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                8 * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(dtype)
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=v_head_dim,
                scale=1.0 / (v_head_dim**0.5),
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )
        impl.W_UK_T = torch.randn(
            num_heads, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device=device
        )
        impl.W_UV = torch.randn(
            num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device
        )
        impl.kv_b_proj = lambda x: (
            torch.randn(
                x.shape[0],
                num_heads * (qk_nope_head_dim + v_head_dim),
                dtype=dtype,
                device=device,
            ),
        )
        return impl

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

    def test_npu_mla_impl(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16

        num_heads = 32
        head_size = 128
        num_kv_heads = 8
        scale = 1.0 / (head_size**0.5)
        kv_cache_dtype = "auto"
        attn_type = AttentionType.DECODER

        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        hidden_size = num_heads * head_size
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )

        batch_size = 1
        prompt_len = 1
        mock_ctx = MagicMock()
        mock_ctx.batch_descriptor = MagicMock(
            num_reqs=batch_size,
            max_q_len=prompt_len,
            max_seq_len=prompt_len,
            uniform=True,
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=head_size,
                scale=scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype=kv_cache_dtype,
                attn_type=attn_type,
            )

            impl.W_UK_T = torch.randn(
                num_heads, qk_nope_head_dim, kv_lora_rank, device=device, dtype=dtype
            )
            impl.W_UV = torch.randn(
                num_heads, kv_lora_rank, v_head_dim, device=device, dtype=dtype
            )
            impl.kv_b_proj = lambda x: (
                torch.empty(
                    x.shape[0],
                    num_heads * (qk_nope_head_dim + v_head_dim),
                    dtype=x.dtype,
                    device=x.device,
                ),
            )

            impl.dcp_world_size = 1

            num_prefills = 1
            num_decode_tokens = 2
            seq_lens = [8, 1, 1]

            batch_size = 1
            prompt_len = 1
            total_prefill_tokens = batch_size * prompt_len
            query_start_loc = torch.tensor(
                [0, total_prefill_tokens], dtype=torch.int32, device=device
            )

            metadata = self.mla_setup["make_fake_metadata"](
                num_prefills=num_prefills,
                num_decode_tokens=num_decode_tokens,
                seq_lens=seq_lens,
                device=device,
                num_reqs=batch_size,
                max_query_len=prompt_len,
                max_seq_len=prompt_len,
                query_start_loc=query_start_loc,
            )

            total_tokens = sum(seq_lens)
            output = torch.empty(total_tokens, hidden_size, dtype=dtype, device=device)

            q = torch.randn(
                total_tokens,
                num_heads,
                qk_nope_head_dim + qk_rope_head_dim,
                dtype=dtype,
                device=device,
            )
            k_c_normed = torch.randn(
                total_tokens, kv_lora_rank, dtype=dtype, device=device
            )
            k_pe = torch.randn(
                total_tokens, qk_rope_head_dim, dtype=dtype, device=device
            )

            num_blocks = 10
            block_size = 128
            nope_cache = torch.empty(
                num_blocks, block_size, 512, dtype=torch.uint8, device=device
            )
            rope_cache = torch.empty(
                num_blocks, block_size, 64, dtype=torch.uint8, device=device
            )
            kv_cache = (nope_cache, rope_cache)

            layer = MagicMock()

            def mock_forward_prefill(*args, **kwargs):
                q_tensor = args[1]
                return torch.zeros(
                    q_tensor.shape[0], hidden_size, dtype=dtype, device=device
                )

            def mock_forward_decode(*args, **kwargs):
                q_tensor = args[1]
                return torch.zeros(
                    q_tensor.shape[0], hidden_size, dtype=dtype, device=device
                )

            def mock_v_up_proj(x, **kwargs):
                return torch.zeros(
                    x.shape[0], num_heads * v_head_dim, dtype=x.dtype, device=x.device
                )

            def fake_scatter_nd_update_(tensor, indices, updates):
                idx = indices.squeeze(-1)  # [N]
                max_idx = idx.max().item()
                current_size = tensor.size(0)

                if max_idx >= current_size:
                    new_size = max_idx + 1
                    new_tensor = torch.zeros(
                        (new_size,) + tensor.shape[1:],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    new_tensor[:current_size] = tensor
                    tensor.resize_(new_tensor.shape)
                    tensor.copy_(new_tensor)

                update_dim = updates.shape[-1]
                tensor[idx, :update_dim] = updates.to(tensor.dtype)
                return tensor

            with (
                patch(
                    "torch_npu.npu_scatter_nd_update_",
                    side_effect=fake_scatter_nd_update_,
                ),
                patch.object(
                    self.mla_setup["impl"],
                    "_forward_prefill",
                    side_effect=mock_forward_prefill,
                ),
                patch.object(
                    self.mla_setup["impl"],
                    "_forward_decode",
                    side_effect=mock_forward_decode,
                ),
                patch.object(
                    self.mla_setup["impl"], "_v_up_proj", side_effect=mock_v_up_proj
                ),
            ):
                out = impl.forward(
                    layer=layer,
                    q=q,
                    k_c_normed=k_c_normed,
                    k_pe=k_pe,
                    kv_cache=kv_cache,
                    attn_metadata=metadata,
                    output=output,
                )

            self.assertEqual(out.shape, (total_tokens, hidden_size))
            print("Impl test passed!")

    def test_forward_prefill(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_kv_heads = 8
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_lora_rank = 512
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )

        batch_size = 1
        prompt_len = 1
        mock_ctx = MagicMock()
        mock_ctx.batch_descriptor = MagicMock(
            num_reqs=batch_size,
            max_q_len=prompt_len,
            max_seq_len=prompt_len,
            uniform=True,
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
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
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )

            impl.kv_b_proj = lambda x: (
                torch.randn(x.shape[0], 32 * (128 + 128), dtype=dtype, device=device),
            )

            num_prefill_tokens = 8
            q = torch.randn(
                num_prefill_tokens, 32, 128 + 64, dtype=dtype, device=device
            )
            kv_c_normed = torch.randn(
                num_prefill_tokens, 512, dtype=dtype, device=device
            )
            k_pe = torch.randn(num_prefill_tokens, 64, dtype=dtype, device=device)
            k_scale = torch.tensor(1.0, dtype=dtype, device=device)

            metadata = MagicMock(
                prefill=type("PrefillMeta", (), {"query_cumlens": [8]})(),
                decode=None,
                num_actual_tokens=8,
                num_prefills=1,
                num_decodes=0,
                num_decode_tokens=0,
                slot_mapping=None,
            )
            metadata.prefill.chunked_context = None
            kv_cache = (torch.empty(0), torch.empty(0))

            mock_output = torch.randn(
                num_prefill_tokens, 32, 128, dtype=dtype, device=device
            )
            mock_lse = torch.randn(8, 32).npu()
            with patch(
                "torch.ops.npu.npu_fused_infer_attention_score",
                return_value=(mock_output, mock_lse),
            ) as mock_op:
                result = impl._forward_prefill(
                    q, kv_c_normed, k_pe, kv_cache, metadata, k_scale
                )

            mock_op.assert_called_once()
            args, kwargs = mock_op.call_args
            self.assertEqual(args[0].shape, (8, 32, 128))
            self.assertEqual(args[1].shape, (8, 32, 128))
            self.assertEqual(args[2].shape, (8, 32, 128))
            self.assertEqual(kwargs["actual_seq_lengths"], [8])
            self.assertEqual(kwargs["scale"], impl.scale)

            self.assertEqual(result.shape, (8, 4096))
            print("_forward_prefill test passed!")

    def test_insert_tensor_by_start_loc(self):
        raw = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.int32)
        insert = torch.tensor([[9], [8]], dtype=torch.int32)
        start_loc = [0, 2, 5]
        out = self.mla_setup["impl"]._insert_tensor_by_start_loc(raw, insert, start_loc)
        expected = torch.tensor(
            [[9], [8], [1], [2], [9], [8], [3], [4], [5]], dtype=torch.int32
        )
        self.assertTrue(torch.equal(out, expected))

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
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
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
        self.assertTrue(torch.equal(impl.sink_compressed_kv, sink_compressed_kv.unsqueeze(1)))

    def test_forward_prefill_with_sink_and_sliding_window(self):
        import omni_npu.attention.backends.mla as mla_mod

        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_kv_heads = 8
        sliding_window = 512
        sink_len = 128
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=sliding_window,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )
            impl.kv_b_proj = lambda x: (
                torch.randn(
                    x.shape[0],
                    num_heads * (qk_nope_head_dim + v_head_dim),
                    dtype=dtype,
                    device=device,
                ),
            )
            impl.sink_k_pe = torch.randn(
                sink_len, qk_rope_head_dim, dtype=dtype, device=device
            )
            impl.sink_compressed_kv = torch.randn(
                sink_len, kv_lora_rank, dtype=dtype, device=device
            )
            impl.sink_len = sink_len
            impl.W_UK_T = torch.randn(
                num_heads, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device=device
            )
            impl.W_UV = torch.randn(
                num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device
            )
            T = 8
            q = torch.randn(T, num_heads, qk_head_dim, dtype=dtype, device=device)
            kv_c_normed = torch.randn(T, kv_lora_rank, dtype=dtype, device=device)
            k_pe = torch.randn(T, qk_rope_head_dim, dtype=dtype, device=device)
            k_scale = torch.tensor(1.0, dtype=dtype, device=device)
            num_blocks, block_size, table_len = 10, 128, 10
            metadata = MagicMock(
                prefill=type(
                    "PrefillMeta",
                    (),
                    {
                        "query_start_loc": [0, T],
                        "query_cumlens": [T],
                        "seq_lens": [T + sink_len],
                        "block_table": torch.randint(
                            0, num_blocks, (1, table_len), dtype=torch.int32, device=device
                        ),
                        "chunked_context": None,
                    },
                )(),
                decode=None,
                num_actual_tokens=T,
                num_prefills=1,
                num_decodes=0,
                num_decode_tokens=0,
                slot_mapping=None,
            )
            kv_cache = (
                torch.randn(num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device),
                torch.randn(
                    num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device
                ),
            )
            sink_out = torch.randn(T, num_heads, kv_lora_rank, dtype=dtype, device=device)
            sink_lse = torch.randn(T, num_heads, 1, dtype=torch.float32, device=device)
            projected_out = torch.randn(
                T, num_heads * v_head_dim, dtype=dtype, device=device
            )

            with patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(sink_out, sink_lse),
            ) as mock_sink_op, patch.object(
                impl, "_v_up_proj", return_value=projected_out
            ) as mock_v_up_proj, patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                False,
            ):
                result = impl._forward_prefill(
                    q, kv_c_normed, k_pe, kv_cache, metadata, k_scale
                )

            mock_sink_op.assert_called_once()
            mock_v_up_proj.assert_called_once()
            kwargs = mock_sink_op.call_args.kwargs
            self.assertEqual(kwargs["actual_seq_qlen"], [T])
            self.assertEqual(kwargs["actual_seq_kvlen"], [T + sink_len])
            self.assertEqual(kwargs["sink_number"], sink_len)
            self.assertEqual(kwargs["pre_tokens"], sliding_window - 1)
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["num_key_value_heads"], 1)
            self.assertEqual(kwargs["block_table"].shape, (1, table_len))
            self.assertEqual(result.shape, (T, num_heads * v_head_dim))

    def test_forward_prefill_with_sink_and_sliding_window_and_fa_tiling(self):
        import omni_npu.attention.backends.mla as mla_mod

        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_kv_heads = 8
        sliding_window = 512
        sink_len = 128
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=sliding_window,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )
            impl.kv_b_proj = lambda x: (
                torch.randn(
                    x.shape[0],
                    num_heads * (qk_nope_head_dim + v_head_dim),
                    dtype=dtype,
                    device=device,
                ),
            )
            impl.sink_k_pe = torch.randn(
                sink_len, qk_rope_head_dim, dtype=dtype, device=device
            )
            impl.sink_compressed_kv = torch.randn(
                sink_len, kv_lora_rank, dtype=dtype, device=device
            )
            impl.sink_len = sink_len
            impl.W_UK_T = torch.randn(
                num_heads, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device=device
            )
            impl.W_UV = torch.randn(
                num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device
            )
            T = 8
            q = torch.randn(T, num_heads, qk_head_dim, dtype=dtype, device=device)
            kv_c_normed = torch.randn(T, kv_lora_rank, dtype=dtype, device=device)
            k_pe = torch.randn(T, qk_rope_head_dim, dtype=dtype, device=device)
            k_scale = torch.tensor(1.0, dtype=dtype, device=device)
            num_blocks, block_size, table_len = 10, 128, 10
            metadata = MagicMock(
                prefill=type(
                    "PrefillMeta",
                    (),
                    {
                        "query_start_loc": [0, T],
                        "query_cumlens": torch.tensor([T], dtype=dtype, device=device),
                        "seq_lens": torch.tensor([T + sink_len], dtype=dtype, device=device),
                        "block_table": torch.randint(
                            0, num_blocks, (1, table_len), dtype=torch.int32, device=device
                        ),
                        "chunked_context": None,
                        "num_tokens": T,
                    },
                )(),
                decode=None,
                num_actual_tokens=T,
                num_prefills=1,
                num_decodes=0,
                num_decode_tokens=0,
                slot_mapping=None,
            )
            kv_cache = (
                torch.randn(num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device),
                torch.randn(
                    num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device
                ),
            )
            sink_out = torch.randn(T, num_heads, kv_lora_rank, dtype=dtype, device=device)
            sink_lse = torch.randn(T, num_heads, 1, dtype=torch.float32, device=device)
            projected_out = torch.randn(
                T, num_heads * v_head_dim, dtype=dtype, device=device
            )

            with patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(sink_out, sink_lse),
            ) as mock_sink_op, patch.object(
                impl, "_v_up_proj", return_value=projected_out
            ) as mock_v_up_proj, patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                True,
            ), patch(
                "torch.ops.custom._npu_fused_infer_attention_sink_metadata",
                create=True,
            ) as mock_sink_metadata:
                result = impl._forward_prefill(
                    q, kv_c_normed, k_pe, kv_cache, metadata, k_scale
                )

            mock_sink_op.assert_called_once()
            mock_v_up_proj.assert_called_once()
            mock_sink_metadata.assert_called_once()
            kwargs = mock_sink_op.call_args.kwargs
            self.assertEqual(kwargs["actual_seq_qlen"].tolist(), [T])
            self.assertEqual(kwargs["actual_seq_kvlen"].tolist(), [T + sink_len])
            self.assertEqual(kwargs["sink_number"], sink_len)
            self.assertEqual(kwargs["pre_tokens"], sliding_window - 1)
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["num_key_value_heads"], 1)
            self.assertEqual(kwargs["block_table"].shape, (1, table_len))
            self.assertEqual(result.shape, (T, num_heads * v_head_dim))

    def test_forward_prefill_without_sink_uses_sink_op_when_fa_tiling_enabled(self):
        import omni_npu.attention.backends.mla as mla_mod

        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        T = 8
        impl = self._make_impl_for_attention_call(
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        q = torch.randn(T, num_heads, qk_nope_head_dim + qk_rope_head_dim, dtype=dtype, device=device)
        kv_c_normed = torch.randn(T, kv_lora_rank, dtype=dtype, device=device)
        k_pe = torch.randn(T, qk_rope_head_dim, dtype=dtype, device=device)
        k_scale = torch.tensor(1.0, dtype=dtype, device=device)
        metadata = MagicMock(
            prefill=type(
                "PrefillMeta",
                (),
                {
                    "query_cumlens": torch.tensor([T], dtype=torch.int32, device=device),
                    "seq_lens": torch.tensor([T], dtype=torch.int32, device=device),
                    "chunked_context": None,
                },
            )(),
            decode=None,
        )
        kv_cache = (
            torch.randn(10, 128, kv_lora_rank, dtype=dtype, device=device),
            torch.randn(10, 128, qk_rope_head_dim, dtype=dtype, device=device),
        )
        sink_out = torch.randn(T, num_heads, v_head_dim, dtype=dtype, device=device)
        sink_lse = torch.randn(T, num_heads, 1, dtype=torch.float32, device=device)
        meta_data = torch.arange(1024, dtype=torch.int32, device=device)
        fake_npu = MagicMock()
        fake_npu.current_stream.return_value = object()
        fake_npu.get_stream_limit.return_value = {"cube_core_num": 20, "vector_core_num": 40}

        with patch.object(
            mla_mod.model_extra_config.operator_opt_config,
            "use_aicpu_fa_tiling",
            True,
        ), patch.object(
            torch,
            "npu",
            fake_npu,
            create=True,
        ), patch(
            "torch.ops.custom._npu_fused_infer_attention_sink_metadata",
            return_value=meta_data,
            create=True,
        ) as mock_metadata_op, patch(
            "torch.ops.custom.npu_fused_infer_attention_sink",
            return_value=(sink_out, sink_lse),
            create=True,
        ) as mock_sink_op, patch(
            "torch.ops.npu.npu_fused_infer_attention_score",
            create=True,
        ) as mock_fia:
            result = impl._forward_prefill(
                q, kv_c_normed, k_pe, kv_cache, metadata, k_scale
            )

        mock_metadata_op.assert_called_once()
        mock_sink_op.assert_called_once()
        mock_fia.assert_not_called()
        metadata_kwargs = mock_metadata_op.call_args.kwargs
        self.assertEqual(metadata_kwargs["input_layout"], "TND")
        self.assertEqual(metadata_kwargs["input_layout_kv"], "TND")
        self.assertEqual(metadata_kwargs["sink_num"], 0)
        self.assertEqual(metadata_kwargs["k_sink_num"], 0)
        self.assertEqual(metadata_kwargs["pre_tokens"], (1 << 31) - 1)
        self.assertEqual(metadata_kwargs["next_tokens"], 0)
        self.assertEqual(metadata_kwargs["aic_core_num"], 20)
        self.assertEqual(metadata_kwargs["aiv_core_num"], 40)
        kwargs = mock_sink_op.call_args.kwargs
        self.assertIs(kwargs["meta_data"], meta_data)
        self.assertEqual(kwargs["actual_seq_qlen"].dtype, torch.int64)
        self.assertEqual(kwargs["actual_seq_kvlen"].dtype, torch.int64)
        self.assertEqual(kwargs["actual_seq_qlen"].tolist(), [T])
        self.assertEqual(kwargs["actual_seq_kvlen"].tolist(), [T])
        self.assertEqual(kwargs["input_layout"], "TND")
        self.assertEqual(kwargs["sparse_mode"], 3)
        self.assertEqual(kwargs["sink_number"], 0)
        self.assertTrue(kwargs["query"].is_contiguous())
        self.assertTrue(kwargs["key"].is_contiguous())
        self.assertTrue(kwargs["value"].is_contiguous())
        self.assertTrue(kwargs["query_rope"].is_contiguous())
        self.assertTrue(kwargs["key_rope"].is_contiguous())
        self.assertEqual(result.shape, (T, num_heads * v_head_dim))

    def test_forward_decode_without_sink_uses_sink_op_when_fa_tiling_enabled(self):
        import omni_npu.attention.backends.mla as mla_mod

        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        kv_lora_rank = 512
        T = 2
        impl = self._make_impl_for_attention_call(
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        decode_ql_nope = torch.randn(T, num_heads, qk_nope_head_dim, dtype=dtype, device=device)
        decode_q_pe = torch.randn(T, num_heads, qk_rope_head_dim, dtype=dtype, device=device)
        num_blocks, block_size, table_len = 10, 128, 10
        kv_cache = (
            torch.randn(num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device),
            torch.randn(num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device),
        )
        decode_meta = self.mla_setup["decode_metadata"](
            block_table=torch.randint(0, num_blocks, (T, table_len), dtype=torch.int32, device=device),
            seq_lens=torch.tensor([5, 3], dtype=torch.int32, device=device),
            query_cumlens=torch.tensor([1, 2], dtype=torch.int32, device=device),
            num_tokens=T,
            dcp_tot_seq_lens=None,
        )
        metadata = self.mla_setup["metadata"](
            prefill=None,
            decode=decode_meta,
            num_actual_tokens=T,
            num_prefills=0,
            num_decodes=T,
            num_decode_tokens=T,
            slot_mapping=None,
            num_reqs=T,
            max_query_len=1,
            max_seq_len=5,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        )
        sink_out = torch.randn(num_heads, T, kv_lora_rank, dtype=dtype, device=device)
        meta_data = torch.arange(1024, dtype=torch.int32, device=device)
        fake_npu = MagicMock()
        fake_npu.current_stream.return_value = object()
        fake_npu.get_stream_limit.return_value = {"cube_core_num": 20, "vector_core_num": 40}
        mock_ctx = MagicMock(capturing=False)

        with patch.object(
            mla_mod.model_extra_config.operator_opt_config,
            "use_aicpu_fa_tiling",
            True,
        ), patch.object(
            torch,
            "npu",
            fake_npu,
            create=True,
        ), patch(
            "omni_npu.attention.backends.mla.get_forward_context",
            return_value=mock_ctx,
        ), patch(
            "torch.ops.custom._npu_fused_infer_attention_sink_metadata",
            return_value=meta_data,
            create=True,
        ) as mock_metadata_op, patch(
            "torch.ops.custom.npu_fused_infer_attention_sink",
            return_value=(sink_out,),
            create=True,
        ) as mock_sink_op, patch(
            "torch.ops.npu.npu_fused_infer_attention_score",
            create=True,
        ) as mock_fia:
            output = impl._forward_decode(
                decode_ql_nope, decode_q_pe, kv_cache, metadata, MagicMock()
            )

        mock_metadata_op.assert_called_once()
        mock_sink_op.assert_called_once()
        mock_fia.assert_not_called()
        metadata_kwargs = mock_metadata_op.call_args.kwargs
        self.assertEqual(metadata_kwargs["input_layout"], "TND")
        self.assertEqual(metadata_kwargs["input_layout_kv"], "BnBsH")
        self.assertEqual(metadata_kwargs["sink_num"], 0)
        self.assertEqual(metadata_kwargs["k_sink_num"], 0)
        self.assertEqual(metadata_kwargs["pre_tokens"], (1 << 31) - 1)
        self.assertEqual(metadata_kwargs["next_tokens"], (1 << 31) - 1)
        self.assertEqual(metadata_kwargs["block_size"], block_size)
        kwargs = mock_sink_op.call_args.kwargs
        self.assertIs(kwargs["meta_data"], meta_data)
        self.assertEqual(kwargs["actual_seq_qlen"].dtype, torch.int64)
        self.assertEqual(kwargs["actual_seq_kvlen"].dtype, torch.int64)
        self.assertEqual(kwargs["actual_seq_qlen"].tolist(), [1, 2])
        self.assertEqual(kwargs["actual_seq_kvlen"].tolist(), [5, 3])
        self.assertEqual(kwargs["input_layout"], "TND_NTD")
        self.assertEqual(kwargs["sparse_mode"], 3)
        self.assertEqual(kwargs["sink_number"], 0)
        self.assertEqual(kwargs["block_table"].shape, (T, table_len))
        self.assertEqual(output.shape, sink_out.shape)

    def test_forward_decode_without_fa_tiling_uses_query_cumlens_for_num_tokens(self):
        import omni_npu.attention.backends.mla as mla_mod

        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        kv_lora_rank = 512
        query_tokens = 8
        actual_tokens = 2
        impl = self._make_impl_for_attention_call(
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            device=device,
        )
        decode_ql_nope = torch.randn(
            query_tokens, num_heads, qk_nope_head_dim, dtype=dtype, device=device
        )
        decode_q_pe = torch.randn(
            query_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device
        )
        num_blocks, block_size, table_len = 10, 128, 10
        kv_cache = (
            torch.randn(num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device),
            torch.randn(num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device),
        )
        decode_meta = self.mla_setup["decode_metadata"](
            block_table=torch.randint(
                0, num_blocks, (actual_tokens, table_len), dtype=torch.int32, device=device
            ),
            seq_lens=[5, 3],
            query_cumlens=[1, actual_tokens],
            num_tokens=query_tokens,
            dcp_tot_seq_lens=None,
        )
        metadata = self.mla_setup["metadata"](
            prefill=None,
            decode=decode_meta,
            num_actual_tokens=actual_tokens,
            num_prefills=0,
            num_decodes=actual_tokens,
            num_decode_tokens=actual_tokens,
            slot_mapping=None,
            num_reqs=actual_tokens,
            max_query_len=1,
            max_seq_len=5,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        )
        mock_ctx = MagicMock(capturing=False)
        mock_attn_out = torch.randn(
            actual_tokens, 1, num_heads, 128, dtype=dtype, device=device
        )

        with patch.object(
            mla_mod.model_extra_config.operator_opt_config,
            "use_aicpu_fa_tiling",
            False,
        ), patch(
            "omni_npu.attention.backends.mla.get_forward_context",
            return_value=mock_ctx,
        ), patch(
            "torch.ops.npu.npu_fused_infer_attention_score",
            return_value=(mock_attn_out,),
            create=True,
        ) as mock_fia, patch(
            "torch.ops.custom.npu_fused_infer_attention_sink",
            create=True,
        ) as mock_sink_op:
            impl._forward_decode(
                decode_ql_nope, decode_q_pe, kv_cache, metadata, MagicMock()
            )

        mock_fia.assert_called_once()
        mock_sink_op.assert_not_called()
        kwargs = mock_fia.call_args.kwargs
        self.assertEqual(kwargs["query"].shape[0], actual_tokens)
        self.assertEqual(kwargs["query_rope"].shape[0], actual_tokens)

    def test_forward_prefill_with_sink_pads_query_heads_to_power_of_two(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 30
        query_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_kv_heads = 8
        sink_len = 128
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )
            impl.W_UK_T = torch.randn(
                num_heads, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device=device
            )
            impl.sink_len = sink_len
            T = 8
            q = torch.randn(T, num_heads, qk_head_dim, dtype=dtype, device=device)
            kv_c_normed = torch.randn(T, kv_lora_rank, dtype=dtype, device=device)
            k_pe = torch.randn(T, qk_rope_head_dim, dtype=dtype, device=device)
            k_scale = torch.tensor(1.0, dtype=dtype, device=device)
            num_blocks, block_size, table_len = 10, 128, 10
            metadata = MagicMock(
                prefill=type(
                    "PrefillMeta",
                    (),
                    {
                        "query_start_loc": [0, T],
                        "query_cumlens": [T],
                        "seq_lens": [T + sink_len],
                        "block_table": torch.randint(
                            0, num_blocks, (1, table_len), dtype=torch.int32, device=device
                        ),
                        "chunked_context": None,
                    },
                )(),
                decode=None,
                num_actual_tokens=T,
                num_prefills=1,
                num_decodes=0,
                num_decode_tokens=0,
                slot_mapping=None,
            )
            kv_cache = (
                torch.randn(num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device),
                torch.randn(
                    num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device
                ),
            )
            sink_out = torch.randn(
                T, query_heads, kv_lora_rank, dtype=dtype, device=device
            )
            sink_lse = torch.randn(T, query_heads, 1, dtype=torch.float32, device=device)
            projected_out = torch.randn(
                T, num_heads * v_head_dim, dtype=dtype, device=device
            )

            with patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(sink_out, sink_lse),
            ) as mock_sink_op, patch.object(
                impl, "_v_up_proj", return_value=projected_out
            ) as mock_v_up_proj:
                result = impl._forward_prefill(
                    q, kv_c_normed, k_pe, kv_cache, metadata, k_scale
                )

            mock_sink_op.assert_called_once()
            mock_v_up_proj.assert_called_once()
            kwargs = mock_sink_op.call_args.kwargs
            self.assertEqual(kwargs["query"].shape, (T, query_heads, kv_lora_rank))
            self.assertEqual(kwargs["query_rope"].shape, (T, query_heads, qk_rope_head_dim))
            self.assertEqual(kwargs["num_query_heads"], query_heads)
            self.assertEqual(result.shape, (T, num_heads * v_head_dim))

    def test_forward_decode_with_sink_and_sliding_window(self):
        import omni_npu.attention.backends.mla as mla_mod

        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_kv_heads = 8
        sink_len = 128
        sliding_window = 512
        T = 2
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=sliding_window,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )
            impl.sink_len = sink_len
            decode_ql_nope = torch.randn(
                T, num_heads, qk_nope_head_dim, dtype=dtype, device=device
            )
            decode_q_pe = torch.randn(
                T, num_heads, qk_rope_head_dim, dtype=dtype, device=device
            )
            num_blocks, block_size, table_len = 10, 128, 10
            kv_cache = (
                torch.randn(
                    num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device
                ),
                torch.randn(
                    num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device
                ),
            )
            # seq_lens includes sink: builder adds sink_len to each seq
            decode_meta = self.mla_setup["decode_metadata"](
                block_table=torch.randint(
                    0, num_blocks, (T, table_len), dtype=torch.int32, device=device
                ),
                seq_lens=[5 + sink_len, 3 + sink_len],
                query_cumlens=[1, 2],
                num_tokens=2,
                dcp_tot_seq_lens=None,
            )
            metadata = self.mla_setup["metadata"](
                prefill=None,
                decode=decode_meta,
                num_actual_tokens=8,
                num_prefills=0,
                num_decodes=T,
                num_decode_tokens=T,
                slot_mapping=None,
                num_reqs=1,
                max_query_len=1,
                max_seq_len=1,
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            )
            sink_out = torch.randn(num_heads, T, v_head_dim, dtype=dtype, device=device)
            mock_ctx = MagicMock()
            mock_ctx.capturing = False
            with patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(sink_out.transpose(0, 1).contiguous(),),
            ) as mock_sink_op, patch(
                "omni_npu.attention.backends.mla.get_forward_context",
                return_value=mock_ctx
            ), patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                False,
            ):
                o = impl._forward_decode(
                    decode_ql_nope, decode_q_pe, kv_cache, metadata, MagicMock()
                )
            mock_sink_op.assert_called_once()
            kwargs = mock_sink_op.call_args.kwargs
            self.assertEqual(kwargs["actual_seq_qlen"], [1, 2])
            self.assertEqual(kwargs["actual_seq_kvlen"], [5 + sink_len, 3 + sink_len])
            self.assertEqual(kwargs["sink_number"], sink_len)
            self.assertEqual(kwargs["pre_tokens"], sliding_window - 1)
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["block_table"].shape, (T, table_len))
            self.assertEqual(o.shape, (num_heads, T, v_head_dim))

    def test_forward_decode_with_sink_and_sliding_window_and_fa_tiling(self):
        import omni_npu.attention.backends.mla as mla_mod

        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_kv_heads = 8
        sink_len = 128
        sliding_window = 512
        T = 2
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=sliding_window,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )
            impl.sink_len = sink_len
            decode_ql_nope = torch.randn(
                T, num_heads, qk_nope_head_dim, dtype=dtype, device=device
            )
            decode_q_pe = torch.randn(
                T, num_heads, qk_rope_head_dim, dtype=dtype, device=device
            )
            num_blocks, block_size, table_len = 10, 128, 10
            kv_cache = (
                torch.randn(
                    num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device
                ),
                torch.randn(
                    num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device
                ),
            )
            # seq_lens includes sink: builder adds sink_len to each seq
            decode_meta = self.mla_setup["decode_metadata"](
                block_table=torch.randint(
                    0, num_blocks, (T, table_len), dtype=torch.int32, device=device
                ),
                seq_lens=torch.tensor([5 + sink_len, 3 + sink_len], dtype=dtype, device=device),
                query_cumlens=torch.tensor([1, 2], dtype=dtype, device=device),
                num_tokens=2,
                dcp_tot_seq_lens=None,
            )
            metadata = self.mla_setup["metadata"](
                prefill=None,
                decode=decode_meta,
                num_actual_tokens=8,
                num_prefills=0,
                num_decodes=T,
                num_decode_tokens=T,
                slot_mapping=None,
                num_reqs=1,
                max_query_len=1,
                max_seq_len=1,
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            )
            sink_out = torch.randn(num_heads, T, v_head_dim, dtype=dtype, device=device)
            mock_ctx = MagicMock()
            mock_ctx.capturing = False
            with patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(sink_out.transpose(0, 1).contiguous(),),
            ) as mock_sink_op, patch(
                "omni_npu.attention.backends.mla.get_forward_context",
                return_value=mock_ctx
            ), patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                True,
            ), patch(
                "torch.ops.custom._npu_fused_infer_attention_sink_metadata",
                create=True,
            ) as mock_sink_metadata:
                o = impl._forward_decode(
                    decode_ql_nope, decode_q_pe, kv_cache, metadata, MagicMock()
                )
            mock_sink_op.assert_called_once()
            mock_sink_metadata.assert_called_once()
            kwargs = mock_sink_op.call_args.kwargs
            self.assertEqual(kwargs["actual_seq_qlen"].tolist(), [1, 2])
            self.assertEqual(kwargs["actual_seq_kvlen"].tolist(), [5 + sink_len, 3 + sink_len])
            self.assertEqual(kwargs["sink_number"], sink_len)
            self.assertEqual(kwargs["pre_tokens"], sliding_window - 1)
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["block_table"].shape, (T, table_len))
            self.assertEqual(o.shape, (num_heads, T, v_head_dim))

    def test_forward_prefill_with_sink_without_sliding_window(self):
        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_kv_heads = 8
        sink_len = 128
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )
            impl.kv_b_proj = lambda x: (
                torch.randn(
                    x.shape[0],
                    num_heads * (qk_nope_head_dim + v_head_dim),
                    dtype=dtype,
                    device=device,
                ),
            )
            impl.sink_k_pe = torch.randn(
                sink_len, qk_rope_head_dim, dtype=dtype, device=device
            )
            impl.sink_compressed_kv = torch.randn(
                sink_len, kv_lora_rank, dtype=dtype, device=device
            )
            impl.sink_len = sink_len
            impl.W_UK_T = torch.randn(
                num_heads, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device=device
            )
            impl.W_UV = torch.randn(
                num_heads, kv_lora_rank, v_head_dim, dtype=dtype, device=device
            )
            T = 8
            q = torch.randn(T, num_heads, qk_head_dim, dtype=dtype, device=device)
            kv_c_normed = torch.randn(T, kv_lora_rank, dtype=dtype, device=device)
            k_pe = torch.randn(T, qk_rope_head_dim, dtype=dtype, device=device)
            k_scale = torch.tensor(1.0, dtype=dtype, device=device)
            num_blocks, block_size, table_len = 10, 128, 10
            metadata = MagicMock(
                prefill=type(
                    "PrefillMeta",
                    (),
                    {
                        "query_start_loc": [0, T],
                        "query_cumlens": [T],
                        "seq_lens": [T + sink_len],
                        "block_table": torch.randint(
                            0, num_blocks, (1, table_len), dtype=torch.int32, device=device
                        ),
                        "chunked_context": None,
                    },
                )(),
                decode=None,
                num_actual_tokens=T,
                num_prefills=1,
                num_decodes=0,
                num_decode_tokens=0,
                slot_mapping=None,
            )
            kv_cache = (
                torch.randn(num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device),
                torch.randn(
                    num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device
                ),
            )
            out = torch.randn(T, num_heads, kv_lora_rank, dtype=dtype, device=device)
            out_lse = torch.randn(T, num_heads, 1, dtype=torch.float32, device=device)
            projected_out = torch.randn(
                T, num_heads * v_head_dim, dtype=dtype, device=device
            )
            # When sink_len > 0, prefill uses paged sink attention with matrix absorption.
            with patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(out, out_lse),
            ) as mock_sink_op, patch.object(
                impl, "_v_up_proj", return_value=projected_out
            ) as mock_v_up_proj:
                result = impl._forward_prefill(
                    q, kv_c_normed, k_pe, kv_cache, metadata, k_scale
                )
            mock_sink_op.assert_called_once()
            mock_v_up_proj.assert_called_once()
            kwargs = mock_sink_op.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(kwargs["actual_seq_qlen"], [T])
            self.assertEqual(kwargs["actual_seq_kvlen"], [T + sink_len])
            self.assertEqual(kwargs["sink_number"], sink_len)
            self.assertEqual(
                kwargs["pre_tokens"], self.mla_setup["impl"].MAX_WINDOW_SIZE
            )
            self.assertEqual(kwargs["num_key_value_heads"], 1)
            self.assertEqual(kwargs["block_table"].shape, (1, table_len))
            self.assertEqual(result.shape, (T, num_heads * v_head_dim))

    def test_forward_decode_with_sink_without_sliding_window(self):
        import omni_npu.attention.backends.mla as mla_mod

        device = torch.device("cpu")
        dtype = torch.bfloat16
        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        kv_lora_rank = 512
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_kv_heads = 8
        sink_len = 128
        T = 2
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )
            impl.sink_len = sink_len
            decode_ql_nope = torch.randn(
                T, num_heads, qk_nope_head_dim, dtype=dtype, device=device
            )
            decode_q_pe = torch.randn(
                T, num_heads, qk_rope_head_dim, dtype=dtype, device=device
            )
            num_blocks, block_size, table_len = 10, 128, 10
            kv_cache = (
                torch.randn(
                    num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device
                ),
                torch.randn(
                    num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device
                ),
            )
            # seq_lens includes sink: builder adds sink_len to each seq
            decode_meta = self.mla_setup["decode_metadata"](
                block_table=torch.randint(
                    0, num_blocks, (T, table_len), dtype=torch.int32, device=device
                ),
                seq_lens=[5 + sink_len, 3 + sink_len],
                query_cumlens=[1, 2],
                num_tokens=2,
                dcp_tot_seq_lens=None,
            )
            metadata = self.mla_setup["metadata"](
                prefill=None,
                decode=decode_meta,
                num_actual_tokens=8,
                num_prefills=0,
                num_decodes=T,
                num_decode_tokens=T,
                slot_mapping=None,
                num_reqs=1,
                max_query_len=1,
                max_seq_len=1,
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            )
            # npu_fused_infer_attention_sink returns (T, N, D), impl transposes to (N, T, D)
            sink_out = torch.randn(num_heads, T, v_head_dim, dtype=dtype, device=device)

            mock_ctx = MagicMock()
            mock_ctx.capturing = False
            with patch(
                "torch.ops.custom.npu_fused_infer_attention_sink",
                return_value=(sink_out.transpose(0, 1).contiguous(),),
            ) as mock_sink_op, patch(
                "omni_npu.attention.backends.mla.get_forward_context",
                return_value=mock_ctx
            ), patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling",
                False,
            ):
                o = impl._forward_decode(
                    decode_ql_nope, decode_q_pe, kv_cache, metadata, MagicMock()
                )
            mock_sink_op.assert_called_once()
            kwargs = mock_sink_op.call_args.kwargs
            self.assertEqual(kwargs["sparse_mode"], 4)
            self.assertEqual(
                kwargs["pre_tokens"], self.mla_setup["impl"].MAX_WINDOW_SIZE
            )
            self.assertEqual(kwargs["actual_seq_kvlen"], [5 + sink_len, 3 + sink_len])
            self.assertEqual(kwargs["sink_number"], sink_len)
            self.assertEqual(o.shape, (num_heads, T, v_head_dim))

    def test_forward_decode(self):
        mock_context = MagicMock()
        mock_context.batch_descriptor = MagicMock()
        device = torch.device("npu:0")
        dtype = torch.bfloat16

        num_heads = 32
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        hidden_size = num_heads * v_head_dim  # 4096
        q_lora_rank = 256
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_lora_rank = 128
        num_kv_heads = 8
        kv_b_proj = (
            torch.nn.Linear(
                kv_lora_rank,
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,
            )
            .to(device)
            .to(torch.bfloat16)
        )

        batch_size = 1
        prompt_len = 1
        mock_ctx = MagicMock()
        mock_ctx.batch_descriptor = MagicMock(
            num_reqs=batch_size,
            max_q_len=prompt_len,
            max_seq_len=prompt_len,
            uniform=True,
        )
        mock_ctx.capturing = False
        with (
            patch(
                "vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size",
                return_value=64,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_current_vllm_config",
                return_value=None,
            ),
            patch(
                "omni_npu.attention.backends.mla.get_forward_context",
                return_value=mock_ctx
            )
        ):
            impl = self.mla_setup["impl"](
                num_heads=num_heads,
                head_size=128,
                scale=1.0 / (128**0.5),
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                logits_soft_cap=None,
                kv_sharing_target_layer_name=None,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=512,
                q_lora_rank=q_lora_rank,
                qk_head_dim=qk_head_dim,
                kv_b_proj=kv_b_proj,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )

            T = 2  # number of decode tokens (batch size for decode)

            #  FIX: decode_ql_nope is (T, num_heads, qk_nope_head_dim)
            decode_ql_nope = torch.randn(
                T, num_heads, qk_nope_head_dim, dtype=dtype, device=device
            )

            #  FIX: decode_q_pe MUST have enough elements to be reshaped to (T, 1, num_heads, qk_rope_head_dim)
            # So its shape should be (T, num_heads, qk_rope_head_dim) — i.e., per-head RoPE
            decode_q_pe = torch.randn(
                T, num_heads, qk_rope_head_dim, dtype=dtype, device=device
            )

            num_blocks, block_size = 10, 128
            nope_cache = torch.randn(
                num_blocks, block_size, 512, dtype=dtype, device=device
            )
            rope_cache = torch.randn(
                num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device
            )
            kv_cache = (nope_cache, rope_cache)

            decode_meta = self.mla_setup["decode_metadata"](
                block_table=torch.randint(
                    0, num_blocks, (T, 10), dtype=torch.int32, device=device
                ),
                seq_lens=[5, 3],
                query_cumlens=[5, 8],
                num_tokens=8,
                dcp_tot_seq_lens=None,
            )
            batch_size = 1
            prompt_len = 1
            total_prefill_tokens = batch_size * prompt_len
            query_start_loc = torch.tensor(
                [0, total_prefill_tokens], dtype=torch.int32, device=device
            )

            metadata = self.mla_setup["metadata"](
                prefill=None,
                decode=decode_meta,
                num_actual_tokens=8,
                num_prefills=0,
                num_decodes=T,
                num_decode_tokens=T,
                slot_mapping=None,
                num_reqs=batch_size,
                max_query_len=prompt_len,
                max_seq_len=prompt_len,
                query_start_loc=query_start_loc,
            )

            layer = MagicMock()

            # Mock the NPU op
            mock_attn_out = torch.randn(
                T, 1, num_heads, v_head_dim, dtype=dtype, device=device
            )
            with patch(
                "torch.ops.npu.npu_fused_infer_attention_score",
                return_value=(mock_attn_out,),
            ) as mock_op:
                o = impl._forward_decode(
                    decode_ql_nope, decode_q_pe, kv_cache, metadata, layer
                )

            mock_op.assert_called_once()
            kwargs = mock_op.call_args.kwargs
            self.assertEqual(kwargs["block_table"].shape, (2, 10))
            self.assertEqual(kwargs["actual_seq_lengths_kv"], [5, 3])
            self.assertEqual(kwargs["input_layout"], "TND_NTD")
            self.assertEqual(kwargs["num_key_value_heads"], 1)

            # Output from _forward_decode is (T, 1, N, D) after transpose
            self.assertEqual(
                o.shape, (T, 1, num_heads, v_head_dim)
            )  # e.g., (2, 1, 32, 128)
            # self.assertIsNone(extra)
            print("_forward_decode test passed!")

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