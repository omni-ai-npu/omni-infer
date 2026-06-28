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
            # Stub out the real vllm gdn_attn backend so its module body
            # (which imports compute_causal_conv1d_metadata and other utils
            # names not provided by the stub above) is never executed during
            # unit tests. mome only needs GDNAttentionMetadataBuilder at
            # import time, which the MagicMock satisfies.
            "vllm.v1.attention.backends.gdn_attn": MagicMock(),
        },
    )
    utils_mod_patcher.start()

    # Mock forward context with capturing=False
    mock_forward_ctx = MagicMock()
    mock_forward_ctx.capturing = False
    mock_forward_ctx.batch_descriptor = None
    mock_forward_ctx.no_compile_layers = {}

    forward_ctx_mod = types.ModuleType("vllm.forward_context")
    forward_ctx_mod.get_forward_context = MagicMock(return_value=mock_forward_ctx)
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

    def test_builder_build_prefill_with_sink_len_updates_seq_lens(self):
        import omni_npu.attention.backends.mla as mla_mod

        builder = self.mla_setup["builder"].__new__(self.mla_setup["builder"])
        builder.reorder_batch_threshold = 0
        builder.vllm_config = MagicMock()
        builder.vllm_config.kv_transfer_config = None
        builder.dcp_world_size = 1
        builder.sink_len = 128
        builder.device = torch.device("cpu")
        builder.model_config = MagicMock()
        builder.model_config.get_head_size.return_value = None
        # Attributes normally set by MLACommonMetadataBuilder.__init__, which is
        # bypassed via __new__ above. Only the ones touched on the prefill path
        # (no chunked context, dcp_world_size == 1) are populated.
        builder.prefill_metadata_cls = mla_mod.NPUMLAPrefillMetadata
        builder.metadata_cls = mla_mod.NPUMLAMetadata
        builder._use_cudnn_prefill = False
        builder._use_trtllm_ragged_prefill = False
        builder._use_fi_prefill = False

        # One prefill request, 3 query tokens, no already-computed context.
        query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
        common_attn_metadata = MagicMock(
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=3,
            max_seq_len=3,
            block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
            slot_mapping=torch.zeros((1,), dtype=torch.int32),
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.tensor([0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([0], dtype=torch.int32),
            dcp_local_seq_lens=None,
        )

        # Drive the prefill branch: num_decodes=0, num_prefills=1.
        with (
            patch.object(
                mla_mod, "split_decodes_and_prefills",
                return_value=(0, 1, 0, 1),
            ),
            patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling", False,
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
            ) as mock_v_up_proj:
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

    def test_build_decode_with_kv_transfer_config(self):
        import omni_npu.attention.backends.mla as mla_mod

        builder = self.mla_setup["builder"].__new__(self.mla_setup["builder"])
        builder.reorder_batch_threshold = 0
        builder.vllm_config = MagicMock()
        builder.vllm_config.kv_transfer_config = MagicMock()
        builder.dcp_world_size = 1
        builder.mc2_mask = torch.zeros(256, dtype=torch.bool)
        builder.device = torch.device("cpu")
        builder.model_config = MagicMock()
        builder.model_config.get_head_size.return_value = None
        builder.metadata_cls = mla_mod.NPUMLAMetadata
        builder._use_fi_prefill = False

        # One decode request, one decode token.
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        common_attn_metadata = MagicMock(
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=10,
            block_table_tensor=torch.tensor([[0, 1]], dtype=torch.int32),
            slot_mapping=torch.zeros((1,), dtype=torch.int32),
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.tensor([10], dtype=torch.int32),
            dcp_local_seq_lens=None,
        )

        # Drive the decode branch: num_decodes=1, num_prefills=0.
        with (
            patch.object(
                mla_mod, "split_decodes_and_prefills",
                return_value=(1, 0, 1, 0),
            ),
            patch.object(
                mla_mod.model_extra_config.operator_opt_config,
                "use_aicpu_fa_tiling", False,
            ),
        ):
            result = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=False,
            )

        self.assertIsNotNone(result.decode.mc2_mask)
        self.assertTrue(result.decode.mc2_mask[0].item())

    def test_build_decode_with_sink_len(self):
        import omni_npu.attention.backends.mla as mla_mod

        builder = self.mla_setup["builder"].__new__(self.mla_setup["builder"])
        builder.reorder_batch_threshold = 0
        builder.vllm_config = MagicMock()
        builder.vllm_config.kv_transfer_config = None
        builder.dcp_world_size = 1
        builder.sink_len = 64
        builder.device = torch.device("cpu")
        builder.model_config = MagicMock()
        builder.model_config.get_head_size.return_value = None
        builder.metadata_cls = mla_mod.NPUMLAMetadata
        builder._use_fi_prefill = False

        # Two decode requests, one decode token each. seq_lens=[0, 5] so that
        # after applying sink_len the decode seq_lens become [64, 5].
        query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
        common_attn_metadata = MagicMock(
            num_reqs=2,
            num_actual_tokens=2,
            max_query_len=1,
            max_seq_len=5,
            block_table_tensor=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            slot_mapping=torch.zeros((2,), dtype=torch.int32),
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.tensor([0, 5], dtype=torch.int32),
            dcp_local_seq_lens=None,
        )

        # Drive the decode branch: num_decodes=2, num_prefills=0.
        with (
            patch.object(
                mla_mod, "split_decodes_and_prefills",
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

    def test_build_prefill_with_dcp_chunked_context(self):
        """Cover the DCP (dcp_world_size > 1) chunked-prefill branch in build().

        This path is the part copied from upstream super().build() that the
        other unit tests skip (they use dcp_world_size == 1). One prefill
        request with a non-zero context so max_context_len_cpu > 0 drives the
        chunked-context + DCP-local-sharding code.
        """
        import omni_npu.attention.backends.mla as mla_mod

        builder = self.mla_setup["builder"].__new__(self.mla_setup["builder"])
        builder.reorder_batch_threshold = 0
        builder.vllm_config = MagicMock()
        builder.vllm_config.kv_transfer_config = None
        builder.dcp_world_size = 2
        builder.sink_len = 0
        builder.device = torch.device("cpu")
        builder.model_config = MagicMock()
        builder.model_config.get_head_size.return_value = None
        builder.prefill_metadata_cls = mla_mod.NPUMLAPrefillMetadata
        builder.metadata_cls = mla_mod.NPUMLAMetadata
        # DCP-related attrs normally set in MLACommonMetadataBuilder.__init__.
        builder.dcp_local_block_size = 1
        builder.dcp_virtual_block_size = 2  # dcp_local_block_size * dcp_world_size
        builder.cp_kv_cache_interleave_size = 1
        builder.chunked_prefill_workspace_size = 128
        builder.aot_schedule = False  # NPU path, skips page_size alignment
        builder._use_cudnn_prefill = False
        builder._use_trtllm_ragged_prefill = False
        builder._use_fi_prefill = False

        # Force CPU as the default device for EVERY tensor in this test: the
        # mock input tensors below AND the torch.arange/torch.zeros allocated
        # inside build(). On real NPU CI the default device is npu:0, so
        # device-less torch.tensor/zeros would land on NPU. That both breaks
        # the device-matched torch.min (mla.py:300, needs both operands on the
        # same device) and the .pin_memory() calls (mla.py:390, only works on
        # dense CPU tensors -> "cannot pin 'npuIntType'"). Keeping the whole
        # branch on CPU reproduces the local (CPU-default) behaviour the test
        # was written against.
        with torch.device("cpu"):
            # 1 prefill req: query_len=3, seq_len=10 -> context=7 (>0, chunked).
            builder.chunked_prefill_workspace = torch.zeros(
                128, dtype=torch.float32)
            query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
            common_attn_metadata = MagicMock(
                num_reqs=1,
                num_actual_tokens=1,
                max_query_len=3,
                max_seq_len=10,
                block_table_tensor=torch.zeros((1, 16), dtype=torch.int32),
                slot_mapping=torch.zeros((3,), dtype=torch.int32),
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc,
                seq_lens=torch.tensor([10], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
                dcp_local_seq_lens=torch.tensor([10], dtype=torch.int32),
            )

            # Drive prefill branch: num_decodes=0, num_prefills=1.
            with (
                patch.object(
                    mla_mod, "split_decodes_and_prefills",
                    return_value=(0, 1, 0, 1),
                ),
                patch.object(
                    mla_mod.model_extra_config.operator_opt_config,
                    "use_aicpu_fa_tiling", False,
                ),
                patch.object(
                    mla_mod, "get_dcp_local_seq_lens",
                    return_value=torch.zeros((2, 1), dtype=torch.int32),
                ),
                # DCP post-processing calls real distributed helpers
                # (TP_Convertor) which need an initialized TP group; stub them
                # out for the unit test.
                patch.object(mla_mod.TP_Convertor, "do_scheduled_kv_reorg"),
                patch.object(builder, "prepare_dcp_slots"),
                patch.object(builder, "prepare_dcp_ag_reorg"),
            ):
                result = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    fast_build=False,
                )

            # The DCP branch produced a chunked context on the prefill metadata.
            self.assertIsNotNone(result.prefill)
            self.assertIsNotNone(result.prefill.chunked_context)
