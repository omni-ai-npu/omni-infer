# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import pytest
import torch

from tests.unit.layers.test_attn_unit_helpers import (
    mock_torch_npu_stream as _mock_torch_npu_stream,
    run_maybe_mome_out_partition_case,
)

MLA_MODULE = "omni_npu.v1.layers.attention.npu_mla"

cfg_i32 = {"device": "cpu", "dtype": torch.int32}
cfg_i64 = {"device": "cpu", "dtype": torch.int64}
cfg_bf16 = {"device": "cpu", "dtype": torch.bfloat16}


@pytest.mark.unit
def test_cross_layer_shared_op_reuses_buffers_and_isolates_callers():
    from omni_npu.attention.backends.utils import CrossLayerSharedOp

    # Keep buffers and expected tensors on CPU so the test is device-agnostic
    # (default torch device may be NPU in CI containers).
    expected_decode = torch.ones(4, device="cpu")
    expected_prefill = torch.full((4,), 2.0, device="cpu")
    op = MagicMock(side_effect=[expected_decode.clone(), expected_prefill.clone()])
    shared_op = CrossLayerSharedOp(
        op=op,
        shape=(4,),
        dtype=torch.float32,
        callers=("decode", "prefill"),
        device="cpu",
    )

    decode = shared_op({}, recompute=True, caller="decode")
    cached_decode = shared_op({}, recompute=False, caller="decode")
    prefill = shared_op({}, recompute=True, caller="prefill")

    assert decode.data_ptr() == cached_decode.data_ptr()
    assert decode.data_ptr() != prefill.data_ptr()
    assert torch.equal(cached_decode, expected_decode)
    assert torch.equal(prefill, expected_prefill)
    assert op.call_count == 2


@pytest.mark.unit
def test_cross_layer_shared_op_isolates_composite_keys_and_recomputes_unknown():
    from omni_npu.attention.backends.utils import CrossLayerSharedOp

    # Pin everything to CPU: earlier tests in the suite may leave the default
    # device on NPU, which would otherwise leak into tensor creation here.
    op = MagicMock(
        side_effect=[
            torch.ones(4, device="cpu"),
            torch.full((4,), 2.0, device="cpu"),
            torch.full((4,), 3.0, device="cpu"),
            torch.full((4,), 4.0, device="cpu"),
        ]
    )
    shared_op = CrossLayerSharedOp(
        op=op,
        shape=(4,),
        dtype=torch.float32,
        callers=(("decode", 511), ("decode", 1023), ("prefill", 511)),
        device="cpu",
    )

    decode_511 = shared_op({}, recompute=True, caller=("decode", 511))
    cached_decode_511 = shared_op({}, recompute=False, caller=("decode", 511))
    decode_1023 = shared_op({}, recompute=True, caller=("decode", 1023))
    prefill_511 = shared_op({}, recompute=True, caller=("prefill", 511))
    unknown = shared_op({}, recompute=False, caller=("decode", 2047))

    assert decode_511.data_ptr() == cached_decode_511.data_ptr()
    assert decode_511.data_ptr() != decode_1023.data_ptr()
    assert decode_511.data_ptr() != prefill_511.data_ptr()
    assert torch.equal(cached_decode_511, torch.ones(4, device="cpu"))
    assert torch.equal(decode_1023, torch.full((4,), 2.0, device="cpu"))
    assert torch.equal(prefill_511, torch.full((4,), 3.0, device="cpu"))
    assert torch.equal(unknown, torch.full((4,), 4.0, device="cpu"))
    assert op.call_count == 4


@pytest.mark.unit
def test_init_cross_layer_shared_ops_uses_expected_buffers(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    constructor = MagicMock(side_effect=[MagicMock(), MagicMock()])
    monkeypatch.setattr(mla_mod, "CrossLayerSharedOp", constructor)
    monkeypatch.setattr(mla_mod, "npu_fused_infer_attention_sink_metadata", None)
    monkeypatch.setattr(mla_mod, "npu_ai_infra_attention_pioneer_metadata", None)
    monkeypatch.setattr(
        torch.ops.custom,
        "_npu_fused_infer_attention_sink_metadata",
        MagicMock(),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_ai_infra_attention_pioneer_metadata",
        MagicMock(),
        raising=False,
    )
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.on_ascend950 = True
    fake._metadata_pre_tokens = (511, 1023)

    fake._init_cross_layer_shared_ops()

    assert constructor.call_count == 2
    assert constructor.call_args_list[0].kwargs["shape"] == (1024,)
    assert constructor.call_args_list[1].kwargs["shape"] == (1024,)
    expected_callers = tuple(
        (caller, pre_tokens)
        for caller in ("decode", "prefill_absorb", "prefill")
        for pre_tokens in (511, 1023)
    )
    for call in constructor.call_args_list:
        assert call.kwargs["dtype"] is torch.int32
        assert call.kwargs["callers"] == expected_callers


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layer_idx", "sliding_window", "expected_producer"),
    [
        (0, 512, True),
        (1, 512, False),
        (3, 1024, True),
        (4, 1024, False),
        (8, 2048, True),
        (9, 2048, True),
        (10, 2048, True),
    ],
)
def test_init_metadata_sharing_selects_main_producers_and_refreshes_all_mtp(
    layer_idx, sliding_window, expected_producer
):
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    config = SimpleNamespace(
        num_hidden_layers=8,
        num_nextn_predict_layers=3,
        swa_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        sliding_window_list=[
            512, 512, 512, 1024, 1024, 1024, 1024, 1024,
            2048, 2048, 2048,
        ],
    )
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.layer_idx = layer_idx
    fake.sliding_window = sliding_window

    fake._init_metadata_sharing(config)

    assert fake.is_fa_metadata_producer is expected_producer
    assert {511, 1023, 2047}.issubset(fake._metadata_pre_tokens)


@pytest.mark.unit
def test_init_metadata_sharing_recomputes_unregistered_window():
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    config = SimpleNamespace(
        num_hidden_layers=8,
        swa_layers=[1, 2],
        sliding_window_list=[512, 512],
    )
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.layer_idx = 6
    fake.sliding_window = 4096

    fake._init_metadata_sharing(config)

    assert 4095 not in fake._metadata_pre_tokens
    assert fake.is_fa_metadata_producer is True


@pytest.mark.unit
def test_init_metadata_sharing_uses_first_swa_layer_for_scalar_window():
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    config = SimpleNamespace(
        num_hidden_layers=8,
        swa_layers=[1, 2],
        sliding_window=512,
    )
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.layer_idx = 1
    fake.sliding_window = 512

    fake._init_metadata_sharing(config)

    assert fake.is_fa_metadata_producer is True
    assert 511 in fake._metadata_pre_tokens


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layer_idx", "expected_producer"),
    [(7, True), (8, False)],
)
def test_init_metadata_sharing_reuses_full_mla_after_skipping_dsa(
    layer_idx, expected_producer
):
    from omni_npu.attention.backends.mla import NPUMLAImpl
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    config = SimpleNamespace(
        num_hidden_layers=9,
        dsa_layers=[0, 3, 6],
        swa_layers=[1, 2, 4, 5],
        sliding_window_list=[512, 512, 1024, 1024],
    )
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.layer_idx = layer_idx
    fake.sliding_window = None

    fake._init_metadata_sharing(config)

    assert fake.is_fa_metadata_producer is expected_producer
    assert NPUMLAImpl.MAX_WINDOW_SIZE in fake._metadata_pre_tokens


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layer_idx", "expected_producer"),
    [(0, True), (2, False)],
)
def test_init_metadata_sharing_reuses_all_full_mla_layers(
    layer_idx, expected_producer
):
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    config = SimpleNamespace(num_hidden_layers=4)
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.layer_idx = layer_idx
    fake.sliding_window = None

    fake._init_metadata_sharing(config)

    assert fake.is_fa_metadata_producer is expected_producer


@pytest.mark.unit
@pytest.mark.parametrize("requires_partition", [True, False])
def test_mla_mome_out_partitions_only_when_o_proj_requires_it(
    monkeypatch, requires_partition
):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    run_maybe_mome_out_partition_case(
        NPUDeepseekMLAAttention,
        mla_mod,
        monkeypatch,
        requires_partition,
    )


# =========================
# Basic / no-effect mocks
# =========================


@contextmanager
def _mock_misc(yarn_get_mscale_ret: float = 1.0):
    with (
        patch("vllm.logger.init_logger", return_value=MagicMock()),
        patch(f"{MLA_MODULE}.current_platform", MagicMock(device_type="cpu")),
        patch(f"{MLA_MODULE}.current_stream", MagicMock(), create=True),
        patch(f"{MLA_MODULE}.extract_layer_index", return_value=0),
        patch(f"{MLA_MODULE}.get_rope", return_value=None),
        patch(f"{MLA_MODULE}.yarn_get_mscale", return_value=yarn_get_mscale_ret),
        patch("vllm.model_executor.layers.rotary_embedding.get_rope_wrapper", MagicMock(return_value=None), create=True),
    ):
        yield


# =========================
# NPU operator mocks (adapted for MLA)
# =========================

@contextmanager
def _mock_torch_npu():
    """
    Mock torch_npu and torch.ops.custom operators used by MLA.
    Differences from DSA:
      - Uses npu_fused_infer_attention_score / npu_fused_infer_attention_sink (dense)
      - Uses npu_ai_infra_kv_rmsnorm_rope_cache_v2 for non-contiguous KV
    """
    def fused_infer_attention_score(query, key, value, query_rope=None, key_rope=None,
                                    num_heads=0, num_key_value_heads=0, input_layout="TND",
                                    atten_mask=None, sparse_mode=0, actual_seq_lengths=None,
                                    actual_seq_lengths_kv=None, scale=1.0, next_tokens=0,
                                    **kwargs):
        assert input_layout in ["BSND", "TND", "TND_NTD"]
        assert query.dim() == 3
        assert value.dim() in [3, 4]
        T, N, _ = query.shape
        D = value.size(-1)
        if input_layout == "TND_NTD":
            return query.new_zeros(N, T, D), None
        return query.new_zeros(T, N, D), None

    def fused_infer_attention_sink(query, query_rope, key, value, key_rope,
                                   num_query_heads=0, num_key_value_heads=0,
                                   input_layout="TND", softmax_scale=1.0,
                                   sparse_mode=4, atten_mask=None,
                                   actual_seq_qlen=None, actual_seq_kvlen=None,
                                   pre_tokens=0, next_tokens=0, sink_number=0,
                                   key_sink=None, value_sink=None, key_rope_sink=None,
                                   block_table=None, block_size=0, **kwargs):
        assert input_layout in ["BSND", "TND", "TND_NTD"]
        assert query.dim() == 3
        assert value.dim() in [3, 4]
        T, N, _ = query.shape
        D = value.size(-1)
        if input_layout == "TND_NTD":
            return query.new_zeros(N, T, D), None
        return query.new_zeros(T, N, D), None

    def kv_rmsnorm_rope_cache_v2(latent_kv, weight, cos, sin, slot_mapping,
                                 k_cache, ckv_cache, k_rope_scale=None, k_rope_offset=None,
                                 epsilon=1e-6, cache_mode="PA", rotary_mode="interleave",
                                 quant_mode="none", is_output_kv=True):
        # latent_kv: [T, 1, 1, L+R]
        T = latent_kv.size(0)
        L = ckv_cache.size(-1)
        R = cos.size(-1)
        k_pe = latent_kv.new_zeros(T, 1, 1, R)
        k_nope = latent_kv.new_zeros(T, 1, 1, L)
        return k_pe, k_nope

    def kv_rmsnorm_rope_cache(latent_kv, weight, cos, sin, slot_mapping,
                              rope_cache, nope_cache, epsilon=1e-6, cache_mode="PA",
                              is_output_kv=True, **kwargs):
        T = latent_kv.size(0)
        L = nope_cache.size(-1)
        R = rope_cache.size(-1)
        k_pe = latent_kv.new_zeros(T, 1, 1, R)
        k_nope = latent_kv.new_zeros(T, 1, 1, L)
        return rope_cache, nope_cache, k_pe, k_nope

    def rotary_mul(x, cos, sin):
        return x

    def interleave_rope(x, cos, sin):
        return x

    def transpose_batchmatmul(
        input, weight=None, perm_x1=None, perm_x2=None, perm_y=None
    ):
        if weight is not None:
            x = input.permute(*perm_x1) if perm_x1 else input
            w = weight.permute(*perm_x2) if perm_x2 else weight
            if x.dim() == 3:
                out = torch.matmul(x, w)
                if perm_y is not None:
                    out = out.permute(*perm_y)
                return out
        return input

    def scatter_nd_update_(x, indices, updates):
        return x

    with (
        patch.multiple(
            "torch_npu",
            npu_rotary_mul=MagicMock(side_effect=rotary_mul),
            npu_interleave_rope=MagicMock(side_effect=interleave_rope),
            npu_transpose_batchmatmul=MagicMock(side_effect=transpose_batchmatmul),
            npu_scatter_nd_update_=MagicMock(side_effect=scatter_nd_update_),
            npu_kv_rmsnorm_rope_cache=MagicMock(side_effect=kv_rmsnorm_rope_cache),
        ),
        patch(
            "torch.ops.npu",
            npu_fused_infer_attention_score=MagicMock(side_effect=fused_infer_attention_score),
        ),
        patch(
            "torch.ops.custom",
            npu_fused_infer_attention_sink=MagicMock(side_effect=fused_infer_attention_sink),
            _npu_fused_infer_attention_sink_metadata=MagicMock(
                return_value=torch.zeros(1024, dtype=torch.int32)
            ),
            npu_ai_infra_kv_rmsnorm_rope_cache_v2=MagicMock(side_effect=kv_rmsnorm_rope_cache_v2),
        ),
    ):
        yield


# =========================
# Distributed mocks
# =========================

@contextmanager
def _mock_vllm_distributed(cards: int = 4, rank: int = 0):
    class MockGroupCoordinator:
        def __init__(self, world_size: int, rank_in_group: int):
            self.world_size = world_size
            self.rank_in_group = rank_in_group
            self.device_group = None

        def all_reduce(self, x):
            return x

        def all_gather(self, x, dim=-1):
            return torch.cat([x] * self.world_size, dim=dim)

        def reduce_scatter(self, x, dim=-1):
            return torch.chunk(x, self.world_size, dim=dim)[self.rank_in_group]

    coord = MockGroupCoordinator(world_size=cards, rank_in_group=rank)
    with (
        patch.multiple(
            "vllm.distributed.parallel_state",
            _WORLD=coord,
            _DP=MockGroupCoordinator(world_size=1, rank_in_group=0),
            _TP=coord,
        ),
    ):
        yield


# =========================
# Mock Layer Modules
# =========================

@contextmanager
def _mock_mome():
    class MockAggregateConv:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, x, **kwargs):
            return x

    class MockMomeAttention:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, x, *args, **kwargs):
            return x

    import omni_npu.v1.layers.attention.npu_mla as mla_mod
    mla_mod.AggregateConv = MockAggregateConv
    mla_mod.MomeAttention = MockMomeAttention
    yield
    mla_mod.AggregateConv = None
    mla_mod.MomeAttention = None


@contextmanager
def _mock_mla_attention(
    num_slots: int = 64,
    pg: int = 128,
    use_omni_cache: bool = False,
):
    class MockMLAAttention(torch.nn.Module):
        def __init__(
            self,
            num_heads: int,
            scale: float,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int | None,
            kv_lora_rank: int,
            kv_b_proj: torch.nn.Module,
            cache_config=None,
            quant_config=None,
            prefix: str = "",
            use_sparse: bool = False,
            indexer=None,
            sink_len: int = 0,
            sliding_window: int | None = None,
            page_size_padded: int | None = None,
            block_size_padded: int | None = None,
            **kw, # args for sink
        ):
            super().__init__()
            impl = SimpleNamespace()
            w = kv_b_proj.weight
            w = w.T.contiguous().view(
                kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim
            )
            w_uk, w_uv = w.split([qk_nope_head_dim, v_head_dim], dim=-1)
            impl.W_UK_T = w_uk.permute(1, 2, 0).contiguous()
            impl.W_UV = w_uv.transpose(0, 1).contiguous()
            impl.SHARE_MASK_TRIL_SPARSE = None
            self.impl = impl
            self.sink_k_pe = torch.zeros((128, qk_rope_head_dim), dtype=torch.bfloat16)
            self.sink_compressed_kv = torch.zeros((128, kv_lora_rank), dtype=torch.bfloat16)
            impl.sink_k_pe = self.sink_k_pe
            impl.sink_compressed_kv = self.sink_compressed_kv
            self.sink_populated = False

            # MLA KV cache has only two tensors (k_nope, k_rope), unlike DSA's three
            if use_omni_cache:
                self.kv_cache = [None]
            else:
                kv0 = torch.zeros(num_slots, pg, 1, kv_lora_rank, **cfg_bf16)
                kv1 = torch.zeros(num_slots, pg, 1, qk_rope_head_dim, **cfg_bf16)
                self.kv_cache = [(kv0, kv1)]

        def populate_sink_kv(self, k_nope_cache: torch.Tensor, k_pe_cache: torch.Tensor):
            self.sink_populated = True

        def update_sink_kv(self, k_pe: torch.Tensor, compressed_kv: torch.Tensor):
            self.sink_k_pe = k_pe
            self.sink_compressed_kv = compressed_kv
            self.impl.sink_k_pe = k_pe
            self.impl.sink_compressed_kv = compressed_kv

    with (
        patch(f"{MLA_MODULE}.MLAAttention", MockMLAAttention),
        # patch(f"vllm.model_executor.layers.attention.static_sink_attention.StaticSinkMLAAttention", MockMLAAttention),
    ):
        import omni_npu.v1.layers.attention.npu_mla as mla_mod
        mla_mod.StaticSinkMLAAttention = MockMLAAttention
        yield
        mla_mod.StaticSinkMLAAttention = None


@contextmanager
def _mock_flash_comm_linear(init_comm=None):
    def init_comm_0(linear):
        from vllm.distributed import get_tp_group
        linear.x_transform = lambda x: x
        linear.y_transform = lambda x: x
        linear.tp_size = get_tp_group().world_size
        linear.tp_rank = get_tp_group().rank_in_group
        if init_comm is not None:
            init_comm(linear)

    class MockReplicatedLinear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=False, quant_config=None, prefix=""):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.weight = torch.zeros(out_features, in_features, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = self.x_transform(x)
            y = x.new_zeros(*x.shape[:-1], self.out_features)
            return self.y_transform(y), None

    class MockColumnParallelFlashCommLinear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=False, quant_config=None, prefix="", **kwargs):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.out_per_part = max(1, out_features // self.tp_size)
            self.weight = torch.zeros(self.out_per_part, in_features, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = self.x_transform(x)
            y = x.new_zeros(*x.shape[:-1], self.out_per_part)
            return self.y_transform(y), None

    class MockRowParallelFlashCommLinear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=False, quant_config=None, prefix=""):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.in_per_part = max(1, in_features // self.tp_size)
            self.weight = torch.zeros(out_features, self.in_per_part, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = self.x_transform(x)
            y = x.new_zeros(*x.shape[:-1], self.out_features)
            return self.y_transform(y), None

        def requires_input_partition(self):
            return self.tp_size > 1

    with (
        patch(f"{MLA_MODULE}.ReplicatedLinear", MockReplicatedLinear),
        patch(f"{MLA_MODULE}.ColumnParallelFlashCommLinear", MockColumnParallelFlashCommLinear),
        patch(f"{MLA_MODULE}.RowParallelFlashCommLinear", MockRowParallelFlashCommLinear),
    ):
        yield


@contextmanager
def _mock_layernorm_rmsnorm():
    class MockRMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6, dtype=None):
            super().__init__()
            weight_cfg = dict(cfg_bf16)
            if dtype is not None:
                weight_cfg["dtype"] = dtype
            self.weight = torch.nn.Parameter(torch.ones(dim, **weight_cfg))
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor):
            assert x.size(-1) == self.weight.size(0)
            return x

    with patch(f"{MLA_MODULE}.RMSNorm", MockRMSNorm):
        yield


# =========================
# Config mocks
# =========================

def _make_mla_config(
    qk_rope_head_dim=64,
    rms_norm_eps=1e-6,
    rope_type="default",
    factor=2.0,
    mscale_all_dim=False,
    apply_yarn_scaling=True,
    rope_interleaved=True,
    kv_lora_rank=512,
    use_mome=False,
    dtype=torch.bfloat16,
    rope_scaling=None,
    num_hidden_layers=60,
    is_mtp_layer=False,
    num_nextn_predict_layers=0,
):
    return SimpleNamespace(
        qk_rope_head_dim=qk_rope_head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_parameters={
            "rope_type": rope_type,
            "factor": factor,
            "mscale_all_dim": mscale_all_dim,
            "apply_yarn_scaling": apply_yarn_scaling,
            "rope_theta": 10000.0,
        },
        rope_interleaved=rope_interleaved,
        kv_lora_rank=kv_lora_rank,
        use_mome=use_mome,
        param_sink_number=0,  # default no sink; sink tests will override
        param_sink_with_value=False,
        dtype=dtype,
        rope_scaling=rope_scaling,
        num_hidden_layers=num_hidden_layers,
        is_mtp_layer=is_mtp_layer,
        num_nextn_predict_layers=num_nextn_predict_layers,
    )


def _make_vllm_config(
    speculative_config=None,
    kv_transfer_config=None,
):
    return SimpleNamespace(
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
        compilation_config=SimpleNamespace(static_forward_context={}),
        cache_config=SimpleNamespace(block_size=128),
    )


@contextmanager
def _mock_model_extra_config(
    seq_parallel=False,
    kv_nz=False,
    prefill_absorb=True,
    use_noncontiguous_kv=False,
    merge_q_kv_conv=False,
    use_batch_invariant_op=False,
    use_aicpu_fa_tiling=False,
    enable_multi_stream=False,
    split_q_up_in_multistream=False,
    dtype=torch.bfloat16,
):
    with patch(
        f"{MLA_MODULE}.model_extra_config",
        MagicMock(
            dtype=dtype,
            parall_config=MagicMock(ena_seq_parallel=seq_parallel),
            operator_opt_config=MagicMock(
                kv_nz=kv_nz,
                enable_prefill_mla_absorb_pa=prefill_absorb,
                use_noncontiguous_kv=use_noncontiguous_kv,
                merge_q_kv_conv=merge_q_kv_conv,
                use_batch_invariant_op=use_batch_invariant_op,
                use_aicpu_fa_tiling=use_aicpu_fa_tiling,
                enable_precision_strong_consistency=False,
                enable_multi_stream=enable_multi_stream,
                split_q_up_in_multistream=split_q_up_in_multistream,
            ),
        ),
    ):
        yield


@contextmanager
def _mock_forward_context(
    seq_lens: list,
    prefill: bool = True,
    pd_mixed: bool = False,
):
    ctx = MagicMock()
    ctx.attn_metadata = MagicMock()
    ctx.virtual_engine = 0
    ctx.capturing = False
    ctx.no_compile_layers = {}

    class _StageMetadata:
        def __init__(self, q_lens: list, kv_lens: list, pg: int = 128):
            cu_q_lens = torch.cumsum(torch.tensor(q_lens, **cfg_i32), dim=0)
            self.query_cumlens = cu_q_lens
            self.query_start_loc = torch.tensor([0] + cu_q_lens.tolist(), **cfg_i32)
            self.seq_lens = torch.tensor(kv_lens, **cfg_i32)
            self.block_table = torch.zeros(sum(q_lens), pg, **cfg_i32)
            self.num_tokens = 0

    if pd_mixed:
        ctx.attn_metadata.prefill = _StageMetadata(seq_lens, seq_lens)
        ctx.attn_metadata.decode = _StageMetadata([1] * len(seq_lens), seq_lens)
        ctx.attn_metadata.slot_mapping = torch.arange(len(seq_lens) + sum(seq_lens), **cfg_i64)
        ctx.attn_metadata.num_decodes = len(seq_lens)
        ctx.attn_metadata.num_prefills = len(seq_lens)
        ctx.attn_metadata.num_actual_tokens = len(seq_lens) + sum(seq_lens)
        ctx.attn_metadata.num_decode_tokens = len(seq_lens)
        ctx.attn_metadata.max_query_len = max(seq_lens)
    elif prefill:
        ctx.attn_metadata.prefill = _StageMetadata(seq_lens, seq_lens)
        ctx.attn_metadata.decode = None
        ctx.attn_metadata.slot_mapping = torch.arange(sum(seq_lens), **cfg_i64)
        ctx.attn_metadata.num_decodes = 0
        ctx.attn_metadata.num_prefills = len(seq_lens)
        ctx.attn_metadata.num_actual_tokens = sum(seq_lens)
        ctx.attn_metadata.num_decode_tokens = 0
        ctx.attn_metadata.max_query_len = max(seq_lens)
    else:
        ctx.attn_metadata.decode = _StageMetadata([1] * len(seq_lens), seq_lens)
        ctx.attn_metadata.prefill = None
        ctx.attn_metadata.slot_mapping = torch.arange(len(seq_lens), **cfg_i64)
        ctx.attn_metadata.num_decodes = len(seq_lens)
        ctx.attn_metadata.num_prefills = 0
        ctx.attn_metadata.num_actual_tokens = len(seq_lens)
        ctx.attn_metadata.num_decode_tokens = len(seq_lens)
        ctx.attn_metadata.max_query_len = 1

    with patch(f"{MLA_MODULE}.get_forward_context", return_value=ctx):
        yield


# =========================
# Main patch context
# =========================

@contextmanager
def _patch_and_gen_configs(
    prefill: bool = True,
    ena_seq_parallel: bool = False,
    kv_nz: bool = False,
    prefill_absorb: bool = True,
    use_noncontiguous_kv: bool = False,
    use_batch_invariant_op: bool = False,
    use_aicpu_fa_tiling: bool = False,
    enable_multi_stream: bool = False,
    split_q_up_in_multistream: bool = False,
    use_mome: bool = False,
    init_flash_comm=None,
    seq_lens: list = [32, 47],
    hidden_size: int = 2048,
    q_lora_rank: int = 1536,
    num_heads: int = 32,
    qk_nope_head_dim: int = 64,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    kv_lora_rank: int = 512,
    tp_size: int = 4,
    tp_rank: int = 0,
    pg: int = 128,
    rope_type: str = "default",
    rope_interleaved: bool = True,
    factor: float = 2.0,
    mscale_all_dim: bool = False,
    apply_yarn_scaling: bool = True,
    pd_mixed: bool = False,
    param_sink_number: int = 0,
    sliding_window: int = 0,
    rope_scaling=None,
    num_hidden_layers: int = 60,
    is_mtp_layer: bool = False,
    num_nextn_predict_layers: int = 0,
):
    env = SimpleNamespace(
        seq_lens=seq_lens,
        tp_size=tp_size,
        cfg_bf16=dict(**cfg_bf16),
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        pg=pg,
        param_sink_number=param_sink_number,
    )
    cfg = _make_mla_config(
        qk_rope_head_dim=qk_rope_head_dim,
        rope_type=rope_type,
        factor=factor,
        mscale_all_dim=mscale_all_dim,
        apply_yarn_scaling=apply_yarn_scaling,
        rope_interleaved=rope_interleaved,
        kv_lora_rank=kv_lora_rank,
        use_mome=use_mome,
        rope_scaling=rope_scaling,
        num_hidden_layers=num_hidden_layers,
        is_mtp_layer=is_mtp_layer,
        num_nextn_predict_layers=num_nextn_predict_layers,
    )
    # Override param_sink_number if set
    if param_sink_number > 0:
        cfg.param_sink_number = param_sink_number
        cfg.param_sink_with_value = True
        cfg.sliding_window = sliding_window

    vllm_cfg = _make_vllm_config(
        speculative_config=SimpleNamespace(num_speculative_tokens=0) if not use_mome else None,
    )

    with (
        _mock_torch_npu_stream(),
        _mock_torch_npu(),
        _mock_misc(),
        _mock_vllm_distributed(cards=tp_size, rank=tp_rank),
        _mock_model_extra_config(
            seq_parallel=ena_seq_parallel,
            kv_nz=kv_nz,
            prefill_absorb=prefill_absorb,
            use_noncontiguous_kv=use_noncontiguous_kv,
            use_batch_invariant_op=use_batch_invariant_op,
            use_aicpu_fa_tiling=use_aicpu_fa_tiling,
            enable_multi_stream=enable_multi_stream,
            split_q_up_in_multistream=split_q_up_in_multistream,
        ),
        _mock_flash_comm_linear(init_comm=init_flash_comm),
        _mock_layernorm_rmsnorm(),
        _mock_mla_attention(num_slots=64, pg=pg),
        _mock_mome(),
        _mock_forward_context(seq_lens=seq_lens, prefill=prefill, pd_mixed=pd_mixed),
        patch(f"{MLA_MODULE}.get_current_vllm_config", return_value=vllm_cfg),
    ):
        yield cfg, vllm_cfg, env


class TestNPUDeepseekMLAAttention:

    @dataclass
    class _PrefillRouteMeta:
        chunked_context: object | None = None

    def test_forward_prefill_routes_absorb_for_absorb_or_chunked_context(self):
        with _patch_and_gen_configs(prefill_absorb=True) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

            module = NPUDeepseekMLAAttention(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                num_heads=env.num_heads,
                qk_nope_head_dim=env.qk_nope_head_dim,
                qk_rope_head_dim=env.qk_rope_head_dim,
                v_head_dim=env.v_head_dim,
                q_lora_rank=env.q_lora_rank,
                kv_lora_rank=env.kv_lora_rank,
                cache_config=None,
                quant_config=None,
                prefix="test_layer",
            )
            hidden_states = torch.zeros(2, env.hidden_size, **env.cfg_bf16)
            cos = torch.zeros(2, 1, 1, env.qk_rope_head_dim, **env.cfg_bf16)
            sin = torch.zeros_like(cos)

            absorb_ret = torch.ones_like(hidden_states)
            standard_ret = torch.zeros_like(hidden_states)
            with (
                patch.object(module, "_forward_prefill_absorb_pa", return_value=absorb_ret) as absorb_mock,
                patch.object(module, "_forward_prefill_standard", return_value=standard_ret) as standard_mock,
            ):
                out = module._forward_prefill(
                    hidden_states,
                    cos,
                    sin,
                    attn_metadata=self._PrefillRouteMeta(),
                )

            assert out is absorb_ret
            absorb_mock.assert_called_once()
            standard_mock.assert_not_called()

        with _patch_and_gen_configs(prefill_absorb=False) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

            module = NPUDeepseekMLAAttention(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                num_heads=env.num_heads,
                qk_nope_head_dim=env.qk_nope_head_dim,
                qk_rope_head_dim=env.qk_rope_head_dim,
                v_head_dim=env.v_head_dim,
                q_lora_rank=env.q_lora_rank,
                kv_lora_rank=env.kv_lora_rank,
                cache_config=None,
                quant_config=None,
                prefix="test_layer",
            )
            hidden_states = torch.zeros(2, env.hidden_size, **env.cfg_bf16)
            cos = torch.zeros(2, 1, 1, env.qk_rope_head_dim, **env.cfg_bf16)
            sin = torch.zeros_like(cos)

            absorb_ret = torch.ones_like(hidden_states)
            standard_ret = torch.zeros_like(hidden_states)
            with (
                patch.object(module, "_forward_prefill_absorb_pa", return_value=absorb_ret) as absorb_mock,
                patch.object(module, "_forward_prefill_standard", return_value=standard_ret) as standard_mock,
            ):
                out = module._forward_prefill(
                    hidden_states,
                    cos,
                    sin,
                    attn_metadata=self._PrefillRouteMeta(chunked_context=object()),
                )

            assert out is absorb_ret
            absorb_mock.assert_called_once()
            standard_mock.assert_not_called()

    def _test_with_cfg(
        self,
        prefill: bool = True,
        pd_mixed: bool = False,
        ena_seq_parallel: bool = False,
        use_noncontiguous_kv: bool = False,
        use_batch_invariant_op: bool = False,
        use_aicpu_fa_tiling: bool = False,
        enable_multi_stream: bool = False,
        split_q_up_in_multistream: bool = False,
        use_mome: bool = False,
        prefill_absorb: bool = False,
        rope_type: str = "default",
        param_sink_number: int = 0,
        sliding_window: int = 0,
        rope_interleaved: bool = True,
        rope_scaling=None,
        num_hidden_layers: int = 60,
        is_mtp_layer: bool = False,
        num_nextn_predict_layers: int = 0,
        o_proj_tp: int = None,
    ):
        def init_flash_comm(linear):
            def reduce_scatter(x):
                T, tp = x.size(0), linear.tp_size
                if T % tp != 0:
                    # pad to multiple
                    pad = tp - (T % tp)
                    x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])], dim=0)
                chunks = torch.chunk(x, tp, dim=0)
                return chunks[linear.tp_rank].contiguous()

            if ena_seq_parallel:
                if "o_proj" in linear.prefix and not pd_mixed:
                    if o_proj_tp is not None:
                        linear.tp_rank = linear.tp_rank % o_proj_tp
                        linear.tp_size = o_proj_tp
                    linear.y_transform = reduce_scatter

        with _patch_and_gen_configs(
            prefill=prefill,
            ena_seq_parallel=ena_seq_parallel,
            prefill_absorb=prefill_absorb,
            use_noncontiguous_kv=use_noncontiguous_kv,
            use_batch_invariant_op=use_batch_invariant_op,
            use_aicpu_fa_tiling=use_aicpu_fa_tiling,
            enable_multi_stream=enable_multi_stream,
            split_q_up_in_multistream=split_q_up_in_multistream,
            use_mome=use_mome,
            init_flash_comm=init_flash_comm,
            pd_mixed=pd_mixed,
            rope_type=rope_type,
            rope_interleaved=rope_interleaved,
            param_sink_number=param_sink_number,
            sliding_window=sliding_window,
            rope_scaling=rope_scaling,
            num_hidden_layers=num_hidden_layers,
            is_mtp_layer=is_mtp_layer,
            num_nextn_predict_layers=num_nextn_predict_layers,
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_mla import (
                NPUDeepseekMLAAttention,
                get_forward_context,
                npu_mla_forward,
            )

            # Ensure the layer is registered in static_forward_context (required by npu_mla_forward)
            vllm_cfg.compilation_config.static_forward_context["test_layer.attn"] = None

            m = NPUDeepseekMLAAttention(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                num_heads=env.num_heads,
                qk_nope_head_dim=env.qk_nope_head_dim,
                qk_rope_head_dim=env.qk_rope_head_dim,
                v_head_dim=env.v_head_dim,
                q_lora_rank=env.q_lora_rank,
                kv_lora_rank=env.kv_lora_rank,
                cache_config=None,
                quant_config=None,
                prefix="test_layer",
            )
            # Override prefix to match registration
            m.prefix = "test_layer"
            # Register the layer in forward context so that npu_mla_forward can find it
            ctx = get_forward_context()
            ctx.no_compile_layers["test_layer"] = m

            D, R = env.hidden_size, env.qk_rope_head_dim
            attn_metadata = ctx.attn_metadata
            T = 0
            for meta in [attn_metadata.prefill, attn_metadata.decode]:
                if meta is not None:
                    T += meta.query_cumlens.flatten()[-1].item()
            T0 = T  # for cos/sin

            if ena_seq_parallel:
                T = -(-T // env.tp_size) # ceil_div

            # Call via the torch.ops entry point to exercise the full dispatch logic
            out = npu_mla_forward(
                torch.zeros(T, D, **env.cfg_bf16),
                torch.zeros(T0, 1, 1, R, **env.cfg_bf16),
                torch.zeros(T0, 1, 1, R, **env.cfg_bf16),
                "test_layer",
            )
            assert out.shape == (T, D)

    def test_yarn_rope(self):
        self._test_with_cfg(prefill=True, rope_type="deepseek_yarn")

    def test_prefill_standard(self):
        self._test_with_cfg(prefill_absorb=False)

    def test_prefill_absorb(self):
        self._test_with_cfg(prefill_absorb=True)

    def test_prefill_sp(self):
        self._test_with_cfg(ena_seq_parallel=True)

    def test_decode(self):
        self._test_with_cfg(prefill=False)

    def test_decode_multistream_split_q(self):
        self._test_with_cfg(
            prefill=False,
            enable_multi_stream=True,
            split_q_up_in_multistream=True,
        )

    def test_pd_mixed(self):
        self._test_with_cfg(pd_mixed=True)

    def test_pd_mixed_sp(self):
        self._test_with_cfg(pd_mixed=True, ena_seq_parallel=True)

    def test_prefill_with_mome(self):
        self._test_with_cfg(use_mome=True)

    def test_decode_with_mome(self):
        self._test_with_cfg(prefill=False, use_mome=True)

    def test_decode_noncontiguous_kv(self):
        self._test_with_cfg(use_noncontiguous_kv=True, use_mome=True)

    def test_prefill_noncontiguous_kv(self):
        self._test_with_cfg(use_noncontiguous_kv=True, use_mome=True)

    def test_prefill_sink(self):
        self._test_with_cfg(param_sink_number=128, sliding_window=512)

    def test_decode_sink(self):
        self._test_with_cfg(prefill=False, param_sink_number=128, sliding_window=512)

    def test_decode_sink_noncontiguous_batch_invariant(self):
        self._test_with_cfg(
            prefill=False,
            use_noncontiguous_kv=True,
            use_batch_invariant_op=True,
            param_sink_number=128,
            sliding_window=512,
        )

    def test_decode_sink_noncontiguous_aicpu_tiling(self):
        self._test_with_cfg(
            prefill=False,
            use_noncontiguous_kv=True,
            use_aicpu_fa_tiling=True,
            param_sink_number=128,
            sliding_window=512,
        )

    def test_prefill_o_proj_tp1(self):
        self._test_with_cfg(o_proj_tp=1)

    def test_prefill_sp_o_proj_tp1(self):
        self._test_with_cfg(o_proj_tp=1, ena_seq_parallel=True)

    # ---- mrope / get_rope_wrapper tests ----

    def test_no_mrope_interleaved_true(self):
        """Without mrope_section, get_rope is used with is_neox_style=False (interleaved)."""
        self._test_with_cfg(rope_interleaved=True)

    def test_no_mrope_interleaved_false(self):
        """Without mrope_section, get_rope is used with is_neox_style=True (neox)."""
        self._test_with_cfg(rope_interleaved=False)

    def test_mrope_without_rope_scaling(self):
        """When rope_scaling is None, the non-mrope path is taken."""
        self._test_with_cfg(rope_scaling=None)

    def test_mrope_rope_scaling_no_mrope_section(self):
        """When rope_scaling exists but has no mrope_section, the non-mrope path is taken."""
        self._test_with_cfg(rope_scaling={"factor": 2.0})

    def test_mrope_with_mrope_section(self):
        """When rope_scaling has mrope_section, get_rope_wrapper is used instead of get_rope."""
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [0, 0, 32]},
            num_hidden_layers=60,
        )

    def test_mrope_with_mrope_section_mtp_layer(self):
        """When is_mtp_layer=True, cache_layer uses num_nextn_predict_layers."""
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [0, 0, 32]},
            num_hidden_layers=60,
            is_mtp_layer=True,
            num_nextn_predict_layers=3,
        )

    def test_mrope_with_mrope_section_not_mtp_layer(self):
        """When is_mtp_layer=False (default), cache_layer uses num_hidden_layers."""
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [0, 0, 32]},
            num_hidden_layers=60,
            is_mtp_layer=False,
            num_nextn_predict_layers=3,
        )


@pytest.mark.unit
def test_kv_norm_rope_cache_truncates_when_slots_shorter_than_latent_kv():
    """pd-mixed + SP: slots size < latent_kv size -> truncate latent_kv/cos/sin."""
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    R, L = 64, 512
    T_full, T_short = 8, 5

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.qk_rope_head_dim = R
    fake.kv_lora_rank = L
    fake.on_ascend950 = False
    fake.is_fa_metadata_producer = True
    fake.noncontiguous_kv = False
    fake.rope_interleaved = False
    fake.kv_a_layernorm = lambda x: x
    fake._apply_rope = lambda k_pe, cos, sin: k_pe

    latent_kv = torch.zeros(T_full, R + L)
    cos = torch.zeros(T_full)
    sin = torch.zeros(T_full)
    slots = torch.arange(T_short, dtype=torch.int64)

    k_nope, k_pe = NPUDeepseekMLAAttention._kv_norm_rope_cache(
        fake, latent_kv, cos, sin, slots, kv_cache=None, fused_op=False,
    )
    assert k_nope.shape == (T_short, 1, L)
    assert k_pe.shape == (T_short, 1, R)


@pytest.mark.unit
def test_kv_norm_rope_cache_a5_scatters_nope_and_pe_separately(monkeypatch):
    """A5: npu_scatter_nd_update_ called twice — once for nope into nope_cache, once for pe into rope_cache."""
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    R, L, T = 2, 4, 3

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.qk_rope_head_dim = R
    fake.kv_lora_rank = L
    fake.on_ascend950 = True
    fake.noncontiguous_kv = False
    fake.rope_interleaved = False
    fake.kv_a_layernorm = lambda x: x
    fake._apply_rope = lambda k_pe, cos, sin: k_pe

    slots_2d = torch.arange(T * 2, dtype=torch.int64).view(T, 2)
    slots = SimpleNamespace(
        slot_mapping=torch.arange(T, dtype=torch.int64),
        slot_mapping_2d=slots_2d,
    )
    nope_cache = torch.zeros(4, 8, 1, L)
    rope_cache = torch.zeros(4, 8, 1, R)

    calls = []
    scatter_mock = MagicMock(side_effect=lambda cache, idx, x: calls.append((cache, idx, x)))
    monkeypatch.setattr(mla_mod.torch_npu, "npu_scatter_nd_update_", scatter_mock)

    k_nope, k_pe = NPUDeepseekMLAAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(T, R + L),
        cos=torch.zeros(T, 1, 1, R),
        sin=torch.zeros(T, 1, 1, R),
        slots=slots,
        kv_cache=(nope_cache, rope_cache),
    )

    assert k_nope.shape == (T, 1, L)
    assert k_pe.shape == (T, 1, R)
    assert scatter_mock.call_count == 2

    nope_cache_arg, slots_2d_arg, nope_data = calls[0]
    assert nope_cache_arg is nope_cache
    assert torch.equal(slots_2d_arg, slots_2d)
    assert nope_data.shape == (T, L)

    rope_cache_arg, slots_2d_arg, pe_data = calls[1]
    assert rope_cache_arg is rope_cache
    assert torch.equal(slots_2d_arg, slots_2d)
    assert pe_data.shape == (T, R)


@pytest.mark.unit
def test_kv_norm_rope_cache_a5_uses_v2_for_noncontiguous_kv(monkeypatch):
    """A5 + noncontiguous_kv: npu_ai_infra_kv_rmsnorm_rope_cache_v2 is used instead of scatter."""
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    R, L, T = 2, 4, 3

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.qk_rope_head_dim = R
    fake.kv_lora_rank = L
    fake.on_ascend950 = True
    fake.noncontiguous_kv = True
    fake.rope_interleaved = False
    fake.kv_nz = False
    fake.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(L),
        variance_epsilon=1e-6,
    )
    monkeypatch.setattr(
        mla_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_kv_rmsnorm_rope_cache=True)
        ),
    )
    slots = torch.arange(T, dtype=torch.int64)
    nope_cache = torch.zeros(4, 8, 1, L)
    rope_cache = torch.zeros(4, 8, 1, R)

    v2_mock = MagicMock(
        return_value=(
            torch.zeros(T, 1, 1, R),
            torch.zeros(T, 1, 1, L),
        )
    )
    scatter_mock = MagicMock()
    monkeypatch.setattr(
        "torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2",
        v2_mock,
    )
    monkeypatch.setattr(mla_mod.torch_npu, "npu_scatter_nd_update_", scatter_mock)

    k_nope, k_pe = NPUDeepseekMLAAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(T, R + L),
        cos=torch.zeros(T, 1, 1, R),
        sin=torch.zeros(T, 1, 1, R),
        slots=slots,
        kv_cache=(nope_cache, rope_cache),
    )

    assert k_nope.shape == (T, 1, L)
    assert k_pe.shape == (T, 1, R)
    v2_mock.assert_called_once()
    scatter_mock.assert_not_called()


@pytest.mark.unit
def test_apply_standard_attention_resolves_dict_attn_metadata():
    """attn_metadata can be a dict keyed by f'{prefix}.attn'; resolve before reading max_query_len."""
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.prefix = "test_layer"
    fake.scaling = 1.0
    fake.attn = SimpleNamespace(impl=SimpleNamespace(SHARE_MASK_TRIL_SPARSE=None))

    inner_meta = SimpleNamespace(max_query_len=5)
    ctx = SimpleNamespace(
        attn_metadata={"test_layer.attn": inner_meta},
        capturing=False,
    )

    captured = {}

    def fake_score(q, k, v, **kw):
        captured.update(kw)
        return (q.new_zeros(*q.shape[:2], v.size(-1)),)

    T, N, D, R = 4, 2, 16, 8
    with (
        patch.object(mla_mod, "get_forward_context", return_value=ctx),
        patch(
            "torch.ops.npu",
            npu_fused_infer_attention_score=MagicMock(side_effect=fake_score),
        ),
    ):
        out = NPUDeepseekMLAAttention._apply_standard_attention(
            fake,
            q_nope=torch.zeros(T, N, D),
            q_pe=torch.zeros(T, N, R),
            keys=(torch.zeros(T, N, D), torch.zeros(T, N, R)),
            values=None,
            q_cumlens=torch.tensor([T]),
            kv_lens=torch.tensor([T]),
            block_table=None,
            num_tokens=T,
            layer_name="test_layer",
        )

    assert captured["sparse_mode"] == 3, "max_query_len from dict-resolved meta should drive sparse_mode=3"
    assert out.shape == (T, N, D)

@pytest.mark.unit
def test_apply_sink_attention_non_pa_basic():
    """覆盖 block_table is None 且 noncontiguous_kv=False 的 sink 注意力路径"""
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention
    import torch

    R = 64          # qk_rope_head_dim
    L = 512         # kv_lora_rank
    N = 8           # num_local_heads
    V = 128         # v_head_dim
    T = 4           # tokens (valid_tok)

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.prefix = "test"
    fake.scaling = 1.0
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.qk_rope_head_dim = R
    fake.v_head_dim = V
    fake.param_sink_number = 128
    fake.noncontiguous_kv = False
    fake.on_ascend950 = False
    fake.attn = SimpleNamespace(impl=SimpleNamespace(SHARE_MASK_TRIL_SPARSE=None))
    fake.sliding_window = 512  # 使 window_size 进入

    q_nope = torch.zeros(T, N, L)
    q_pe = torch.zeros(T, N, R)
    k_nope = torch.zeros(T, N, L)
    k_pe = torch.zeros(T, N, R)

    q_cumlens = torch.tensor([T], dtype=torch.int32)
    kv_lens = torch.tensor([T], dtype=torch.int32)

    # 确保走非 PA 路径：block_table 传 None
    block_table = None
    num_tokens = T

    captured = {}
    def fake_sink(query, query_rope, key, value, key_rope, **kwargs):
        captured.update(kwargs)
        return (torch.zeros_like(query),)

    with patch("torch.ops.custom.npu_fused_infer_attention_sink",
               side_effect=fake_sink) as mock_sink:
        out = NPUDeepseekMLAAttention._apply_sink_attention(
            fake,
            q_nope=q_nope,
            q_pe=q_pe,
            keys=(k_nope, k_pe),
            values=None,            # absorb 模式
            q_cumlens=q_cumlens,
            kv_lens=kv_lens,
            block_table=block_table,
            num_tokens=num_tokens,
            num_actual_tokens=num_tokens,
            layer_name="test_layer",
        )

    # 验证算子被调用且传入了正确的参数
    mock_sink.assert_called_once()
    assert captured.get("actual_seq_kvlen") is kv_lens  # 非 PA 分支应使用 kv_lens
    # 输出形状应为 TND（原始文档中要求 TND 返回）
    assert out.shape == (T, N, L)  # absorb 时 value 维度为 L

@pytest.mark.unit
def test_apply_sink_attention_non_pa_noncontiguous():
    """覆盖 block_table is None 且 noncontiguous_kv=True 的 sink 注意力"""
    from omni_npu.v1.layers.attention import npu_mla as layer_mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention
    import torch

    R, L, N, V, T = 64, 512, 8, 128, 4
    S = 128   # sink 长度

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.prefix = "test"
    fake.scaling = 1.0
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.qk_rope_head_dim = R
    fake.v_head_dim = V
    fake.param_sink_number = S
    fake.noncontiguous_kv = True
    fake.on_ascend950 = False
    fake.is_fa_metadata_producer = True
    fake.attn = SimpleNamespace(impl=SimpleNamespace(
        SHARE_MASK_TRIL_SPARSE=None,
        sink_compressed_kv=torch.zeros(S, L),   # [sink, L]
        sink_k_pe=torch.zeros(S, R),            # [sink, R]
    ))
    fake.sliding_window = 512

    q_nope = torch.zeros(T, N, L)
    q_pe = torch.zeros(T, N, R)

    # 非 PA 分支在 noncontiguous 时需要显式传入 sink 张量
    sink_k_nope = torch.zeros(S, N, L)   # 实际由调用者提供
    sink_k_pe = torch.zeros(S, N, R)
    sink_v = torch.zeros(S, N, V)

    # keys 使用单独的 k_nope / k_pe 形状
    k_nope = torch.zeros(T, N, L)
    k_pe = torch.zeros(T, N, R)

    q_cumlens = torch.tensor([T], dtype=torch.int32)
    kv_lens = torch.tensor([T], dtype=torch.int32)

    captured = {}
    def fake_sink(query, query_rope, key, value, key_rope, **kwargs):
        captured.update(kwargs)
        # 输出形状：与 query 的 T,N 相同，与 value 的最后一维相同
        out = query.new_zeros(query.size(0), query.size(1), value.size(-1))
        return (out,)

    with patch(
        "torch.ops.custom.npu_fused_infer_attention_sink",
        side_effect=fake_sink,
    ) as mock_sink, patch.object(
        layer_mla_mod,
        "npu_fused_infer_attention_sink_metadata",
        MagicMock(return_value=torch.zeros(1024, dtype=torch.int32)),
    ):
        out = NPUDeepseekMLAAttention._apply_sink_attention(
            fake,
            q_nope=q_nope,
            q_pe=q_pe,
            keys=(k_nope, k_pe),
            values=sink_v,        # 非 absorb 模式需要 values
            q_cumlens=q_cumlens,
            kv_lens=kv_lens,
            block_table=None,
            num_tokens=T,
            num_actual_tokens=T,
            layer_name="test",
            sink_k_nope=sink_k_nope,
            sink_k_pe=sink_k_pe,
            sink_v=sink_v,
        )

    mock_sink.assert_called_once()
    # 验证传入了 noncontiguous sink 参数
    assert "key_sink" in captured
    assert captured["key_sink"].shape == sink_k_nope.shape
    assert torch.equal(captured["key_sink"], sink_k_nope)

@pytest.mark.unit
def test_apply_sink_attention_non_pa_noncontiguous_and_fa_tiling():
    """覆盖 block_table is None 且 noncontiguous_kv=True 的 sink 注意力"""
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention
    import torch
    import omni_npu.attention.backends.mla as mla_mod
    from omni_npu.v1.layers.attention import npu_mla as layer_mla_mod

    R, L, N, V, T = 64, 512, 8, 128, 4
    S = 128   # sink 长度

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.prefix = "test"
    fake.scaling = 1.0
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.qk_rope_head_dim = R
    fake.v_head_dim = V
    fake.param_sink_number = S
    fake.noncontiguous_kv = True
    fake.on_ascend950 = False
    fake.is_fa_metadata_producer = True
    fake.attn = SimpleNamespace(impl=SimpleNamespace(
        SHARE_MASK_TRIL_SPARSE=None,
        sink_compressed_kv=torch.zeros(S, L),   # [sink, L]
        sink_k_pe=torch.zeros(S, R),            # [sink, R]
    ))
    fake.sliding_window = 512

    q_nope = torch.zeros(T, N, L)
    q_pe = torch.zeros(T, N, R)

    # 非 PA 分支在 noncontiguous 时需要显式传入 sink 张量
    sink_k_nope = torch.zeros(S, N, L)   # 实际由调用者提供
    sink_k_pe = torch.zeros(S, N, R)
    sink_v = torch.zeros(S, N, V)

    # keys 使用单独的 k_nope / k_pe 形状
    k_nope = torch.zeros(T, N, L)
    k_pe = torch.zeros(T, N, R)

    q_cumlens = torch.tensor([T], dtype=torch.int32)
    kv_lens = torch.tensor([T], dtype=torch.int32)

    captured = {}
    def fake_sink(query, query_rope, key, value, key_rope, **kwargs):
        captured.update(kwargs)
        # 输出形状：与 query 的 T,N 相同，与 value 的最后一维相同
        out = query.new_zeros(query.size(0), query.size(1), value.size(-1))
        return (out,)

    with patch(
        "torch.ops.custom.npu_fused_infer_attention_sink",
        side_effect=fake_sink
    ) as mock_sink, patch.object(
        layer_mla_mod,
        "npu_fused_infer_attention_sink_metadata",
        MagicMock(return_value=torch.zeros(1024, dtype=torch.int32)),
    ) as mock_sink_metadata, patch.object(
        mla_mod.model_extra_config.operator_opt_config,
        "use_aicpu_fa_tiling",
        True,
    ):
        out = NPUDeepseekMLAAttention._apply_sink_attention(
            fake,
            q_nope=q_nope,
            q_pe=q_pe,
            keys=(k_nope, k_pe),
            values=sink_v,        # 非 absorb 模式需要 values
            q_cumlens=q_cumlens,
            kv_lens=kv_lens,
            block_table=None,
            num_tokens=T,
            num_actual_tokens=T,
            layer_name="test",
            sink_k_nope=sink_k_nope,
            sink_k_pe=sink_k_pe,
            sink_v=sink_v,
        )

    mock_sink.assert_called_once()
    mock_sink_metadata.assert_called_once()
    # 验证传入了 noncontiguous sink 参数
    assert "key_sink" in captured
    assert captured["key_sink"].shape == sink_k_nope.shape
    assert torch.equal(captured["key_sink"], sink_k_nope)

@pytest.mark.unit
def test_apply_attention_routes_sink_to_a5_path():
    """Ascend 950 sink attention routes through the dedicated A5 implementation."""
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.param_sink_number = 128
    fake.on_ascend950 = True
    fake.is_fa_metadata_producer = True
    fake.num_local_heads = 2
    expected = torch.zeros(3, 2, 4)
    fake._apply_sink_attention_a5 = MagicMock(return_value=expected)
    fake._apply_sink_attention = MagicMock()
    fake._apply_standard_attention = MagicMock()

    q_nope = torch.zeros(3, 2, 4)
    q_pe = torch.zeros(3, 2, 2)
    attn_metadata=MagicMock(num_tokens=3)
    out = NPUDeepseekMLAAttention._apply_attention(
        fake,
        q_nope=q_nope,
        q_pe=q_pe,
        keys=(q_nope, q_pe),
        q_cumlens=torch.tensor([3], dtype=torch.int32),
        kv_lens=torch.tensor([3], dtype=torch.int32),
        values=torch.zeros(3, 2, 5),
        block_table=None,
        num_tokens=3,
        layer_name="test",
        attn_metadata=attn_metadata,

    )

    assert out is expected
    fake._apply_sink_attention_a5.assert_called_once()
    fake._apply_sink_attention.assert_not_called()
    fake._apply_standard_attention.assert_not_called()


@pytest.mark.unit
def test_forward_prefill_a5_sink_uses_standard_path():
    """A5 sink prefill bypasses absorb PA and uses standard prefill."""
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.ena_sp = False
    fake.mla_absorb = False
    fake.param_sink_number = 128
    fake.on_ascend950 = True
    fake.noncontiguous_kv = False
    fake.use_mome = False
    fake._maybe_quant = lambda x: x
    expected = torch.ones(2, 4)
    fake._forward_prefill_absorb_pa = MagicMock()
    fake._forward_prefill_standard = MagicMock(return_value=expected)

    out = NPUDeepseekMLAAttention._forward_prefill(
        fake,
        torch.zeros(2, 4),
        torch.zeros(2, 1, 1, 2),
        torch.zeros(2, 1, 1, 2),
        attn_metadata=SimpleNamespace(),
    )

    assert out is expected
    fake._forward_prefill_absorb_pa.assert_not_called()
    fake._forward_prefill_standard.assert_called_once()


@pytest.mark.unit
def test_apply_sink_attention_ascend950_prefill_path_calls_pioneer(monkeypatch):
    """A5 prefill path concatenates query/key rope tensors for pioneer attention."""
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock())
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", None, raising=False)

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.num_local_heads = 2
    fake.qk_head_dim = 6
    fake.v_head_dim = 5
    fake.qk_rope_head_dim = 2
    fake.kv_lora_rank = 4
    fake.param_sink_number = 3
    fake.scaling = 0.5
    fake.sliding_window = 8
    fake.noncontiguous_kv = False
    fake.on_ascend950 = True
    fake.is_fa_metadata_producer = True

    captured = {}

    def fake_metadata(op_args, recompute, caller):
        captured["metadata"] = op_args
        captured["metadata_call"] = (recompute, caller)
        return "meta"

    def fake_pioneer(query, key, value, meta_data, **kwargs):
        captured["query_shape"] = tuple(query.shape)
        captured["key_shape"] = tuple(key.shape)
        captured["kwargs"] = kwargs
        assert meta_data == "meta"
        return (query.new_zeros(query.size(0), query.size(1), value.size(-1)),)

    with patch.object(
        mla_mod,
        "npu_ai_infra_attention_pioneer_metadata",
        MagicMock(side_effect=fake_metadata),
    ), patch(
        "torch.ops.custom",
        npu_ai_infra_attention_pioneer=MagicMock(side_effect=fake_pioneer),
    ):
        out = NPUDeepseekMLAAttention._apply_sink_attention_a5(
            fake,
            q_nope=torch.zeros(4, 2, 4),
            q_pe=torch.zeros(4, 2, 2),
            keys=(torch.zeros(4, 2, 4), torch.zeros(4, 2, 2)),
            values=torch.zeros(4, 2, 5),
            q_cumlens=torch.tensor([4], dtype=torch.int32),
            kv_lens=torch.tensor([4], dtype=torch.int32),
            block_table=None,
            num_tokens=4,
            valid_tok=4,
            q_heads=2,
            sink_k_nope=torch.zeros(3, 2, 4),
            sink_k_pe=torch.zeros(3, 2, 2),
            sink_v=torch.zeros(3, 2, 5),
        )

    assert out.shape == (4, 2, 5)
    assert captured["query_shape"] == (4, 2, 6)
    assert captured["key_shape"] == (4, 2, 6)
    assert captured["metadata"]["soc_version"] == "ascend950"
    assert captured["metadata_call"] == (True, ("", 7))
    assert captured["kwargs"]["input_layout"] == "TND"
    assert captured["metadata"]["actual_seq_lengths"].dtype == torch.int64
    assert captured["metadata"]["actual_seq_lengths_kv"].dtype == torch.int64


@pytest.mark.unit
def test_apply_sink_attention_ascend950_decode_path_calls_pioneer(monkeypatch):
    """A5 decode path builds TND_NTD metadata for paged attention."""
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self, raising=False)
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock())
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", None, raising=False)

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.num_local_heads = 2
    fake.kv_lora_rank = 4
    fake.qk_rope_head_dim = 2
    fake.param_sink_number = 3
    fake.scaling = 0.5
    fake.sliding_window = None
    fake.noncontiguous_kv = True
    fake.on_ascend950 = True
    fake.is_fa_metadata_producer = True
    fake.attn = SimpleNamespace(impl=SimpleNamespace(
        sink_compressed_kv=torch.zeros(3, 4),
        sink_k_pe=torch.zeros(3, 2),
    ))

    captured = {}

    def fake_metadata(op_args, recompute, caller):
        captured["metadata"] = op_args
        captured["metadata_call"] = (recompute, caller)
        return "meta"

    def fake_pioneer(query, key, value, meta_data, **kwargs):
        captured["kwargs"] = kwargs
        assert meta_data == "meta"
        return (query.new_zeros(query.size(1), query.size(0), value.size(-1)),)

    with patch.object(mla_mod.NPUMLAImpl, "MAX_WINDOW_SIZE", 4096, create=True), patch.object(
        mla_mod,
        "npu_ai_infra_attention_pioneer_metadata",
        MagicMock(side_effect=fake_metadata),
    ), patch(
        "torch.ops.custom",
        npu_ai_infra_attention_pioneer=MagicMock(side_effect=fake_pioneer),
    ):
        out = NPUDeepseekMLAAttention._apply_sink_attention_a5(
            fake,
            q_nope=torch.zeros(4, 2, 4),
            q_pe=torch.zeros(4, 2, 2),
            keys=(torch.zeros(2, 8, 1, 4), torch.zeros(2, 8, 1, 2)),
            values=None,
            q_cumlens=torch.tensor([4], dtype=torch.int32),
            kv_lens=torch.tensor([4], dtype=torch.int32),
            block_table=torch.zeros(1, 2, dtype=torch.int32),
            num_tokens=4,
            valid_tok=4,
            q_heads=2,
        )

    assert out.shape == (2, 4, 4)
    assert captured["metadata"]["input_layout"] == "TND_NTD"
    assert captured["metadata"]["soc_version"] == "ascend950"
    assert captured["metadata_call"] == (True, ("", 4096))
    assert captured["kwargs"]["key_sink"] is fake.attn.impl.sink_compressed_kv
    assert captured["metadata"]["actual_seq_lengths"].dtype == torch.int64
    assert captured["metadata"]["actual_seq_lengths_kv"].dtype == torch.int64


@pytest.mark.unit
def test_chunked_prefill_cumlens_helpers_handle_tensor_and_list_refs():
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    tensor_ref = torch.tensor([2, 5], dtype=torch.int32)
    assert NPUDeepseekMLAAttention._as_cumlens_list(None) is None
    assert NPUDeepseekMLAAttention._as_cumlens_list(tensor_ref) == [2, 5]
    assert NPUDeepseekMLAAttention._as_cumlens_list((3, 7)) == [3, 7]

    tensor_like = NPUDeepseekMLAAttention._cumlens_like([4, 9], tensor_ref)
    assert torch.equal(tensor_like, torch.tensor([4, 9], dtype=torch.int32))
    assert NPUDeepseekMLAAttention._cumlens_like([4, 9], [2, 5]) == [4, 9]


@pytest.mark.unit
def test_lengths_to_i64_converts_int32_tensor_and_list():
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    device = torch.device("cpu")
    int32_tensor = torch.tensor([4, 8], dtype=torch.int32, device=device)
    int64_tensor = torch.tensor([4, 8], dtype=torch.int64, device=device)

    converted = NPUDeepseekMLAAttention._lengths_to_i64(int32_tensor, device)
    assert converted.dtype == torch.int64
    assert torch.equal(converted, int64_tensor)

    same = NPUDeepseekMLAAttention._lengths_to_i64(int64_tensor, device)
    assert same is int64_tensor

    from_list = NPUDeepseekMLAAttention._lengths_to_i64([1, 3], device)
    assert from_list.dtype == torch.int64
    assert from_list.device == device
    assert torch.equal(
        from_list, torch.tensor([1, 3], dtype=torch.int64, device=device)
    )


@pytest.mark.unit
def test_prepend_chunked_prefill_context_adds_swa_history_from_paged_cache(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    monkeypatch.setattr(mla_mod, "cache_fit_shape", lambda cache, mode: cache)

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.on_ascend950 = True
    fake.enable_chunked_prefill = True
    fake.noncontiguous_kv = True
    fake.sliding_window = 3

    L, R = 2, 1
    kv_a = torch.arange(5 * L, dtype=torch.float32).view(5, L)
    k_pe = torch.arange(100, 100 + 5 * R, dtype=torch.float32).view(5, R)
    nope_cache = torch.arange(4 * 2 * L, dtype=torch.float32).view(4, 2, L)
    rope_cache = torch.arange(200, 200 + 4 * 2 * R, dtype=torch.float32).view(4, 2, R)
    block_table = torch.tensor([[1, 3, 0], [2, 0, 1]], dtype=torch.int32)
    q_cumlens = torch.tensor([2, 5], dtype=torch.int32)
    seq_lens = torch.tensor([5, 6], dtype=torch.int32)

    out_kv, out_rope, out_cumlens = NPUDeepseekMLAAttention._prepend_chunked_prefill_context(
        fake, kv_a, k_pe, q_cumlens, seq_lens, block_table, (nope_cache, rope_cache),
    )

    expected_kv = torch.cat([
        nope_cache[torch.tensor([1, 3]), torch.tensor([1, 0])],
        kv_a[:2],
        nope_cache[torch.tensor([2, 0]), torch.tensor([1, 0])],
        kv_a[2:5],
    ], dim=0)
    expected_rope = torch.cat([
        rope_cache[torch.tensor([1, 3]), torch.tensor([1, 0])],
        k_pe[:2],
        rope_cache[torch.tensor([2, 0]), torch.tensor([1, 0])],
        k_pe[2:5],
    ], dim=0)

    assert torch.equal(out_kv, expected_kv)
    assert torch.equal(out_rope, expected_rope)
    assert torch.equal(out_cumlens, torch.tensor([4, 9], dtype=torch.int32))


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    [
        "missing_metadata",
        "disabled_flags",
        "empty_cumlens",
        "closed_window",
        "no_history",
    ],
)
def test_prepend_chunked_prefill_context_returns_original_when_not_applicable(
    case, monkeypatch,
):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    monkeypatch.setattr(mla_mod, "cache_fit_shape", lambda cache, mode: cache)

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.on_ascend950 = True
    fake.enable_chunked_prefill = True
    fake.noncontiguous_kv = True
    fake.sliding_window = 2

    kv_a = torch.zeros(2, 2)
    k_pe = torch.zeros(2, 1)
    q_cumlens = torch.tensor([2], dtype=torch.int32)
    seq_lens = torch.tensor([2], dtype=torch.int32)
    block_table = torch.zeros(1, 2, dtype=torch.int32)
    kv_cache = (torch.zeros(1, 2, 2), torch.zeros(1, 2, 1))

    if case == "missing_metadata":
        block_table = None
    elif case == "disabled_flags":
        fake.enable_chunked_prefill = False
    elif case == "empty_cumlens":
        q_cumlens = []
    elif case == "closed_window":
        fake.sliding_window = 1
        seq_lens = torch.tensor([4], dtype=torch.int32)
    elif case == "no_history":
        seq_lens = torch.tensor([2], dtype=torch.int32)

    out_kv, out_rope, out_cumlens = NPUDeepseekMLAAttention._prepend_chunked_prefill_context(
        fake, kv_a, k_pe, q_cumlens, seq_lens, block_table, kv_cache,
    )

    assert out_kv is kv_a
    assert out_rope is k_pe
    assert out_cumlens is q_cumlens


@pytest.mark.unit
def test_prepend_chunked_prefill_context_asserts_non_swa_a5_chunked_prefill():
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.on_ascend950 = True
    fake.enable_chunked_prefill = True
    fake.noncontiguous_kv = True
    fake.sliding_window = None

    with pytest.raises(AssertionError, match="only supported for SWA layers"):
        NPUDeepseekMLAAttention._prepend_chunked_prefill_context(
            fake,
            torch.zeros(2, 2),
            torch.zeros(2, 1),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            torch.zeros(1, 2, dtype=torch.int32),
            (torch.zeros(1, 2, 2), torch.zeros(1, 2, 1)),
        )


@pytest.mark.unit
def test_forward_prefill_standard_passes_metadata_to_chunked_context(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.num_local_heads = 2
    fake.qk_rope_head_dim = 1
    fake.v_head_dim = 3
    fake.qk_nope_head_dim = 4
    fake.kv_lora_rank = 5
    fake.ena_sp = False
    fake.use_mome = False
    fake.param_sink_number = 0
    fake.attn = SimpleNamespace(kv_cache=[("nope_cache", "rope_cache")])

    T, N, R, QK, L, V = 3, 2, 1, 4, 5, 3
    fake.q_a_proj = MagicMock(return_value=(torch.zeros(T, 6),))
    fake.q_a_layernorm = lambda x: x
    fake.q_b_proj = MagicMock(return_value=(torch.zeros(T, N * (QK + R)),))
    fake._apply_rope = lambda q_pe, cos, sin: q_pe
    fake.kv_a_proj_with_mqa = MagicMock(return_value=(torch.zeros(T, L + R),))
    fake._maybe_mome_q = lambda q, get_mome_args: q
    fake._maybe_mome_kv = lambda kv, get_mome_args: kv
    fake._maybe_mome_out = lambda out, get_mome_args: out
    fake._kv_norm_rope_cache = MagicMock(
        return_value=(torch.zeros(T, 1, L), torch.zeros(T, 1, R)),
    )
    fake._prepend_chunked_prefill_context = MagicMock(
        side_effect=lambda kv_a, k_pe, q_cumlens, seq_lens, block_table, kv_cache: (
            kv_a, k_pe, q_cumlens,
        ),
    )
    fake.kv_b_proj = MagicMock(
        side_effect=lambda kv: (torch.zeros(kv.size(0), N * (QK + V)),),
    )
    fake._apply_attention = MagicMock(return_value=torch.zeros(T, N, V))
    fake.o_proj = MagicMock(side_effect=lambda out: (out,))

    attn_metadata = SimpleNamespace(
        query_cumlens=torch.tensor([T], dtype=torch.int32),
        seq_lens=torch.tensor([T + 2], dtype=torch.int32),
        block_table=torch.zeros(1, 2, dtype=torch.int32),
    )
    monkeypatch.setattr(
        mla_mod,
        "get_forward_context",
        lambda: SimpleNamespace(virtual_engine=0),
    )

    out = NPUDeepseekMLAAttention._forward_prefill_standard(
        fake,
        torch.zeros(T, 8),
        torch.zeros(T, 1, 1, R),
        torch.zeros(T, 1, 1, R),
        get_mome_args=lambda: {},
        attn_metadata=attn_metadata,
    )

    fake._prepend_chunked_prefill_context.assert_called_once()
    _, _, passed_q_cumlens, passed_seq_lens, passed_block_table, passed_cache = (
        fake._prepend_chunked_prefill_context.call_args.args
    )
    assert passed_q_cumlens is attn_metadata.query_cumlens
    assert passed_seq_lens is attn_metadata.seq_lens
    assert passed_block_table is attn_metadata.block_table
    assert passed_cache is fake.attn.kv_cache[0]
    assert out.shape == (T, N * V)


@pytest.mark.unit
def test_kv_norm_rope_cache_noncontiguous_batch_invariant_uses_scatter_block_update(
    monkeypatch,
):
    """When enable_kv_rmsnorm_rope_cache is enabled, fused_op is disabled and the non-contiguous
    cache update falls back to npu_ai_infra_scatter_block_update_."""
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.qk_rope_head_dim = 2
    fake.kv_lora_rank = 4
    fake.on_ascend950 = False
    fake.noncontiguous_kv = True
    fake.kv_nz = False
    fake.kv_a_layernorm = lambda x: x
    fake._apply_rope = lambda x, cos, sin: x

    monkeypatch.setattr(
        mla_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_kv_rmsnorm_rope_cache=False)
        ),
    )

    block_calls = []
    nd_calls = []

    def fake_scatter_block_update(cache, indices, updates):
        block_calls.append((cache, indices, updates))
        return cache

    def fake_scatter_nd_update(cache, indices, updates):
        nd_calls.append((cache, indices, updates))
        return cache

    monkeypatch.setattr(
        torch.ops.custom,
        "npu_ai_infra_scatter_block_update_",
        MagicMock(side_effect=fake_scatter_block_update),
        raising=False,
    )
    monkeypatch.setattr(
        mla_mod.torch_npu,
        "npu_scatter_nd_update_",
        MagicMock(side_effect=fake_scatter_nd_update),
    )

    slots = SimpleNamespace(
        slot_mapping=torch.arange(3, dtype=torch.int64),
        slot_mapping_2d=torch.arange(3, dtype=torch.int64).view(3, 1),
    )
    nope_cache = torch.zeros(4, 8, 1, 4)
    rope_cache = torch.zeros(4, 8, 1, 2)

    k_nope, k_pe = NPUDeepseekMLAAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(3, 6),
        cos=torch.zeros(3, 1, 1, 2),
        sin=torch.zeros(3, 1, 1, 2),
        slots=slots,
        kv_cache=(nope_cache, rope_cache),
    )

    assert k_nope.shape == (3, 1, 4)
    assert k_pe.shape == (3, 1, 2)
    assert len(block_calls) == 2
    assert len(nd_calls) == 0

    nope_cache_arg, nope_indices_arg, nope_data = block_calls[0]
    assert nope_cache_arg is nope_cache
    assert torch.equal(nope_indices_arg, slots.slot_mapping_2d)
    assert nope_data.shape == (3, 4)

    rope_cache_arg, rope_indices_arg, rope_data = block_calls[1]
    assert rope_cache_arg is rope_cache
    assert torch.equal(rope_indices_arg, slots.slot_mapping_2d)
    assert rope_data.shape == (3, 2)


@pytest.mark.unit
def test_build_decode_sink_fia_kwargs_shape_and_keys(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    monkeypatch.setattr(mla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", "mask", raising=False)

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.scaling = 0.25
    query = torch.zeros(2, 2, 4)
    query_rope = torch.zeros(2, 2, 2)
    kv_cache = (torch.zeros(1, 8, 4), torch.zeros(1, 8, 2))
    q_cumlens = torch.tensor([2], dtype=torch.int64)
    kv_lens = torch.tensor([8], dtype=torch.int64)
    block_table = torch.zeros(1, 2, dtype=torch.int32)

    kwargs = NPUDeepseekMLAAttention._build_decode_sink_fia_kwargs(
        fake, query, query_rope, kv_cache, q_cumlens, kv_lens, block_table, 2, 7,
    )

    assert kwargs["query"] is query
    assert kwargs["key"] is kv_cache[0]
    assert kwargs["key_rope"] is kv_cache[1]
    assert kwargs["block_size"] == 8
    assert kwargs["pre_tokens"] == 7
    assert kwargs["sparse_mode"] == 4
    assert kwargs["atten_mask"] == "mask"
    assert kwargs["softmax_scale"] == 0.25


@pytest.mark.unit
def test_apply_sink_attention_consistency_core_merges_outputs(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    T, N, L, R, S = 2, 2, 4, 2, 3
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.param_sink_number = S
    fake.scaling = 0.5
    fake.sliding_window = None
    fake.attn = SimpleNamespace(
        sink_compressed_kv=torch.zeros(S, L),
        sink_k_pe=torch.zeros(S, R),
    )

    monkeypatch.setattr(mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock())
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", None, raising=False)
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "MAX_WINDOW_SIZE", 100, raising=False)
    monkeypatch.setattr(
        mla_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                use_noncontiguous_kv=True,
                enable_precision_strong_consistency=True,
            )
        ),
    )

    expected = torch.full((T, N, L), 9.0)
    meta = MagicMock(return_value="meta")
    sink_v2 = MagicMock(
        side_effect=[
            (torch.zeros(T, N, L), None, torch.zeros(T, N, 1), torch.ones(T, N, 1)),
            (torch.ones(T, N, L), None, torch.zeros(T, N, 1), torch.ones(T, N, 1)),
        ]
    )
    rescale = MagicMock(return_value=(expected[:T], None, None))
    monkeypatch.setattr(mla_mod.ops, "apply_FA_rescale_forward", rescale)
    monkeypatch.setattr(
        torch.ops.custom,
        "_npu_fused_infer_attention_sink_metadata",
        meta,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_fused_infer_attention_sink_v2",
        sink_v2,
        raising=False,
    )

    out = NPUDeepseekMLAAttention._apply_sink_attention_consistency_core(
        fake,
        query=torch.zeros(T, N, L),
        query_rope=torch.zeros(T, N, R),
        kv_cache=(torch.zeros(1, 8, L), torch.zeros(1, 8, R)),
        q_cumlens=torch.tensor([T], dtype=torch.int32),
        kv_lens=torch.tensor([8], dtype=torch.int32),
        block_table=torch.zeros(1, 2, dtype=torch.int32),
        num_actual_tokens=T,
    )

    assert out.shape == (T, N, L)
    torch.testing.assert_close(out[:T], expected[:T])
    assert meta.call_count == 2
    assert sink_v2.call_count == 2
    rescale.assert_called_once()


@pytest.mark.unit
def test_apply_sink_attention_consistency_core_uses_sliding_window(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    T, N, L, R, S = 2, 2, 4, 2, 3
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.param_sink_number = S
    fake.scaling = 0.5
    fake.sliding_window = 512
    fake.attn = SimpleNamespace(
        sink_compressed_kv=torch.zeros(S, L),
        sink_k_pe=torch.zeros(S, R),
    )
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock())
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", None, raising=False)
    monkeypatch.setattr(
        mla_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                use_noncontiguous_kv=True,
                enable_precision_strong_consistency=True,
            )
        ),
    )
    captured = {}

    def _build(*args, **kwargs):
        captured["window_size"] = args[-1]
        return {
            "query": args[0],
            "query_rope": args[1],
            "key": args[2][0],
            "value": args[2][0],
            "key_rope": args[2][1],
            "num_query_heads": args[-2],
            "num_key_value_heads": 1,
            "input_layout": "TND",
            "softmax_scale": fake.scaling,
            "block_table": args[5],
            "block_size": args[2][0].shape[1],
            "actual_seq_qlen": args[3],
            "actual_seq_kvlen": args[4],
            "atten_mask": None,
            "sparse_mode": 4,
            "pre_tokens": args[-1],
            "next_tokens": 0,
        }

    monkeypatch.setattr(
        NPUDeepseekMLAAttention,
        "_build_decode_sink_fia_kwargs",
        _build,
    )
    monkeypatch.setattr(
        torch.ops.custom,
        "_npu_fused_infer_attention_sink_metadata",
        MagicMock(return_value="meta"),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_fused_infer_attention_sink_v2",
        MagicMock(
            side_effect=[
                (torch.zeros(T, N, L), None, torch.zeros(T, N, 1), torch.ones(T, N, 1)),
                (torch.ones(T, N, L), None, torch.zeros(T, N, 1), torch.ones(T, N, 1)),
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        mla_mod.ops,
        "apply_FA_rescale_forward",
        MagicMock(return_value=(torch.zeros(T, N, L), None, None)),
    )

    NPUDeepseekMLAAttention._apply_sink_attention_consistency_core(
        fake,
        query=torch.zeros(T, N, L),
        query_rope=torch.zeros(T, N, R),
        kv_cache=(torch.zeros(1, 8, L), torch.zeros(1, 8, R)),
        q_cumlens=torch.tensor([T], dtype=torch.int32),
        kv_lens=torch.tensor([8], dtype=torch.int32),
        block_table=torch.zeros(1, 2, dtype=torch.int32),
        num_actual_tokens=T,
    )
    assert captured["window_size"] == 511


@pytest.mark.unit
def test_apply_sink_attention_precision_path_transposes_output(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    T, N, L = 2, 3, 4
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.num_local_heads = N
    fake.sliding_window = None
    fake.attn = SimpleNamespace(kv_cache=[(torch.zeros(1), torch.zeros(1))])
    core_out = torch.arange(T * N * L, dtype=torch.float32).view(T, N, L)
    fake._apply_sink_attention_consistency_core = MagicMock(return_value=core_out)

    monkeypatch.setattr(mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock())
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "MAX_WINDOW_SIZE", 100, raising=False)
    monkeypatch.setattr(
        mla_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                enable_precision_strong_consistency=True,
            )
        ),
    )
    monkeypatch.setattr(
        mla_mod, "get_forward_context", lambda: SimpleNamespace(virtual_engine=0)
    )

    out = NPUDeepseekMLAAttention._apply_sink_attention(
        fake,
        q_nope=torch.zeros(T, N, L),
        q_pe=torch.zeros(T, N, 2),
        keys=(torch.zeros(1), torch.zeros(1)),
        values=None,
        q_cumlens=torch.tensor([T], dtype=torch.int32),
        kv_lens=torch.tensor([8], dtype=torch.int32),
        block_table=torch.zeros(1, 2, dtype=torch.int32),
        num_tokens=T,
        num_actual_tokens=T,
        layer_name="test",
    )

    assert out.shape == (N, T, L)
    torch.testing.assert_close(out, core_out.transpose(0, 1))
    fake._apply_sink_attention_consistency_core.assert_called_once()


@pytest.mark.unit
def test_apply_sink_attention_consistency_core_requires_flags(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    monkeypatch.setattr(
        mla_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                use_noncontiguous_kv=False,
                enable_precision_strong_consistency=True,
            )
        ),
    )
    with pytest.raises(AssertionError, match="must be both True"):
        NPUDeepseekMLAAttention._apply_sink_attention_consistency_core(
            fake,
            query=torch.zeros(1, 1, 1),
            query_rope=torch.zeros(1, 1, 1),
            kv_cache=(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)),
            q_cumlens=torch.tensor([1]),
            kv_lens=torch.tensor([1]),
            block_table=torch.zeros(1, 1),
            num_actual_tokens=1,
        )


@pytest.mark.unit
def test_forward_prefill_standard_noncontiguous_sink_calls_kv_b_proj(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    T, N, R, QK, L, V, S = 3, 2, 1, 4, 5, 3, 2
    fake.num_local_heads = N
    fake.qk_rope_head_dim = R
    fake.v_head_dim = V
    fake.qk_nope_head_dim = QK
    fake.kv_lora_rank = L
    fake.ena_sp = False
    fake.use_mome = False
    fake.param_sink_number = S
    fake.noncontiguous_kv = True
    fake.attn = SimpleNamespace(
        kv_cache=[("nope_cache", "rope_cache")],
        sink_compressed_kv=torch.zeros(S, L),
        sink_k_pe=torch.zeros(S, R),
    )
    fake.q_a_proj = MagicMock(return_value=(torch.zeros(T, 6),))
    fake.q_a_layernorm = lambda x: x
    fake.q_b_proj = MagicMock(return_value=(torch.zeros(T, N * (QK + R)),))
    fake._apply_rope = lambda q_pe, cos, sin: q_pe
    fake.kv_a_proj_with_mqa = MagicMock(return_value=(torch.zeros(T, L + R),))
    fake._maybe_mome_q = lambda q, get_mome_args: q
    fake._maybe_mome_kv = lambda kv, get_mome_args: kv
    fake._maybe_mome_out = lambda out, get_mome_args: out
    fake._kv_norm_rope_cache = MagicMock(
        return_value=(torch.zeros(T, 1, L), torch.zeros(T, 1, R)),
    )
    fake._prepend_chunked_prefill_context = MagicMock(
        side_effect=lambda kv_a, k_pe, q_cumlens, seq_lens, block_table, kv_cache: (
            kv_a, k_pe, q_cumlens,
        ),
    )
    kv_b_calls = []

    def _kv_b_proj(x):
        kv_b_calls.append(x)
        return (torch.zeros(x.size(0), N * (QK + V)),)

    fake.kv_b_proj = MagicMock(side_effect=_kv_b_proj)
    fake._apply_attention = MagicMock(return_value=torch.zeros(T, N, V))
    fake.o_proj = MagicMock(side_effect=lambda out: (out,))

    attn_metadata = SimpleNamespace(
        query_cumlens=torch.tensor([T], dtype=torch.int32),
        seq_lens=torch.tensor([T], dtype=torch.int32),
        block_table=torch.zeros(1, 2, dtype=torch.int32),
        query_start_loc=torch.tensor([0, T], dtype=torch.int32),
    )
    monkeypatch.setattr(
        mla_mod, "get_forward_context", lambda: SimpleNamespace(virtual_engine=0)
    )

    NPUDeepseekMLAAttention._forward_prefill_standard(
        fake,
        torch.zeros(T, 8),
        torch.zeros(T, 1, 1, R),
        torch.zeros(T, 1, 1, R),
        get_mome_args=lambda: {},
        attn_metadata=attn_metadata,
    )

    assert len(kv_b_calls) == 2
    assert kv_b_calls[0] is fake.attn.sink_compressed_kv
    assert fake._apply_attention.call_args.kwargs["sink_k_nope"] is not None


@pytest.mark.unit
def test_import_guard_warns_on_omni_training_custom_ops_missing(monkeypatch):
    """G.ERR.05: bare except replaced with except ImportError.
    When omni_training_custom_ops raises ImportError, a warning is logged
    and the module continues to load."""
    # Block the import by setting sentinel to None so ImportError is raised.
    monkeypatch.delitem(
        sys.modules, "omni_training_custom_ops", raising=False,
    )
    monkeypatch.setitem(
        sys.modules, "omni_training_custom_ops", None,
    )

    mod = importlib.import_module(MLA_MODULE)
    # reload() preserves old module globals, so clear the stale attribute first.
    mod.__dict__.pop("omni_training_custom_ops", None)
    importlib.reload(mod)

    assert not hasattr(mod, "omni_training_custom_ops"), (
        "omni_training_custom_ops should not be accessible when import fails"
    )

    # Restore sentinel so later tests can import the real package again.
    sys.modules.pop("omni_training_custom_ops", None)


@pytest.mark.unit
def test_import_guard_warns_on_omni_custom_ops_missing(monkeypatch):
    """G.ERR.05: bare except replaced with except ImportError.
    When omni_custom_ops raises ImportError, a warning is logged
    and the module continues to load."""
    monkeypatch.delitem(
        sys.modules, "omni_custom_ops", raising=False,
    )
    monkeypatch.setitem(
        sys.modules, "omni_custom_ops", None,
    )

    mod = importlib.import_module(MLA_MODULE)
    mod.__dict__.pop("omni_custom_ops", None)
    importlib.reload(mod)

    assert not hasattr(mod, "omni_custom_ops"), (
        "omni_custom_ops should not be accessible when import fails"
    )

    sys.modules.pop("omni_custom_ops", None)


@pytest.mark.unit
def test_import_guard_both_custom_ops_importable():
    """G.ERR.05: both custom_ops are importable, so the module exposes them."""
    mod = importlib.import_module(MLA_MODULE)
    importlib.reload(mod)

    assert hasattr(mod, "omni_training_custom_ops"), (
        "omni_training_custom_ops should be available when importable"
    )
    assert hasattr(mod, "omni_custom_ops"), (
        "omni_custom_ops should be available when importable"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "forward_name", "deferred_name"),
    [
        (
            "omni_npu.v1.layers.attention.npu_mla",
            "npu_mla_forward",
            "npu_mla_forward_mhc_deferred",
        ),
        (
            "omni_npu.v1.layers.attention.npu_dsa",
            "npu_dsa_forward",
            "npu_dsa_forward_mhc_deferred",
        ),
    ],
)
def test_attention_mhc_deferred_launches_before_epilog(
    monkeypatch, module_name, forward_name, deferred_name
):
    module = importlib.import_module(module_name)
    attention = SimpleNamespace(pre_epilog_callback=None)
    h_post = torch.randn(2, 2)
    h_res = torch.randn(2, 2, 2)
    mhc = SimpleNamespace(
        launch_fused_split_sinkhorn=MagicMock(
            return_value=(h_post, h_res)
        )
    )
    # Deferred MHC looks up the forward context from mhc_deferred, not
    # from the attention module that registers the custom op.
    monkeypatch.setattr(
        "omni_npu.layers.mhc.mhc_deferred.get_forward_context",
        lambda: SimpleNamespace(
            no_compile_layers={"attention": attention, "mhc": mhc}
        ),
    )
    output = torch.randn(2, 3)

    def forward(*_args):
        callback = attention.pre_epilog_callback
        attention.pre_epilog_callback = None
        callback()
        return output

    monkeypatch.setattr(module, forward_name, forward)
    residual = torch.randn(2, 6)
    result = getattr(module, deferred_name)(
        torch.randn(2, 3),
        torch.zeros(2, 1, 1, 1),
        torch.zeros(2, 1, 1, 1),
        residual,
        "attention",
        "mhc",
        "task",
    )

    assert result[0] is output
    assert result[1] is h_post
    assert result[2] is h_res
    mhc.launch_fused_split_sinkhorn.assert_called_once_with(
        residual, "task"
    )
    assert attention.pre_epilog_callback is None


# Allow running directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
