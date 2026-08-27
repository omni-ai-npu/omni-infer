# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import pytest

DSA_MODULE = "omni_npu.v1.layers.attention.npu_dsa"

cfg_i32 = {"device": "cpu", "dtype": torch.int32}
cfg_i64 = {"device": "cpu", "dtype": torch.int64}
cfg_bf16 = {"device": "cpu", "dtype": torch.bfloat16}


@pytest.mark.unit
@pytest.mark.parametrize("requires_partition", [True, False])
def test_pangu_mla_epilog_partitions_only_when_o_proj_requires_it(
    monkeypatch, requires_partition
):
    from omni_npu.v1.layers.attention import npu_pangu as pangu_mod
    from omni_npu.v1.layers.attention.npu_pangu import NPUPanguSparseAttention

    attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
    torch.nn.Module.__init__(attention)
    mome_output = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    split_output = (mome_output[:, :2], mome_output[:, 2:])
    split = MagicMock(return_value=split_output)
    monkeypatch.setattr(pangu_mod, "split_tensor_along_last_dim", split)
    attention.use_mome = True
    attention.disable_o_conv_tp = True
    attention.o_conv = SimpleNamespace(dim=4)
    attention.o_proj = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        requires_input_partition=MagicMock(return_value=requires_partition),
    )
    attention._apply_MOME = MagicMock(return_value=mome_output)
    attention._apply_o_proj = MagicMock(side_effect=lambda output: output)

    output = attention._mla_epilog(torch.zeros_like(mome_output))

    attention.o_proj.requires_input_partition.assert_called_once_with()
    if requires_partition:
        split.assert_called_once_with(mome_output, num_partitions=2)
        torch.testing.assert_close(output, split_output[1])
        assert output.is_contiguous()
    else:
        split.assert_not_called()
        assert output is mome_output


@pytest.mark.unit
@pytest.mark.parametrize("requires_partition", [True, False])
def test_dsa_mome_out_partitions_only_when_o_proj_requires_it(
    monkeypatch, requires_partition
):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    attention = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    torch.nn.Module.__init__(attention)
    mome_output = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    split_output = (mome_output[:, :2], mome_output[:, 2:])
    split = MagicMock(return_value=split_output)
    monkeypatch.setattr(mla_mod, "split_tensor_along_last_dim", split)
    attention.use_mome = True
    attention.kv_b_proj = SimpleNamespace(tp_size=1)
    attention.o_proj = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        requires_input_partition=MagicMock(return_value=requires_partition),
    )
    attention._apply_mome = MagicMock(return_value=mome_output)

    output = attention._maybe_mome_out(
        torch.zeros_like(mome_output), MagicMock()
    )

    attention.o_proj.requires_input_partition.assert_called_once_with()
    if requires_partition:
        split.assert_called_once_with(mome_output, num_partitions=2)
        torch.testing.assert_close(output, split_output[1])
        assert output.is_contiguous()
    else:
        split.assert_not_called()
        assert output is mome_output


# =========================
# 1. 无效果或简单mock
# =========================

@contextmanager
def _mock_torch_npu_stream():
    mock_npu = MagicMock()
    mock_npu.current_stream.return_value = MagicMock()
    mock_npu.Stream.return_value = MagicMock()
    mock_npu.stream.side_effect = lambda x: nullcontext()
    with patch("torch.npu", mock_npu):
        yield

@contextmanager
def _mock_misc(yarn_get_mscale_ret: float = 1.0):
    with (
        patch("vllm.logger.init_logger", return_value=MagicMock()),
        patch(f"{DSA_MODULE}.current_platform", MagicMock(device_type="cpu")),
        patch(f"{DSA_MODULE}.current_stream", MagicMock(), create=True),
        patch(f"{DSA_MODULE}.extract_layer_index", return_value=0),
        patch(f"{DSA_MODULE}.get_rope", return_value=None),
        patch(f"{DSA_MODULE}.yarn_get_mscale", return_value=yarn_get_mscale_ret),
        patch("vllm.model_executor.layers.rotary_embedding.get_rope_wrapper", MagicMock(return_value=None), create=True),
    ):
        yield


# =========================
# 2. 功能模块 mock
# =========================

@contextmanager
def _mock_torch_npu():
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
        R = cos.size(-1)
        L = latent_kv.size(-1) - R
        k_pe = latent_kv.new_zeros(T, 1, 1, R)
        k_nope = latent_kv.new_zeros(T, 1, 1, L)
        return k_pe, k_nope

    def sparse_flash_attention_pioneer(
        query, key, value, **kw,
    ):
        return query.new_zeros(*query.shape[:2], value.size(-1)), None

    def fused_causal_conv1d(x, *_, **kw):
        return x

    def lightning_indexer(query, key, weights, sparse_count=2048, **kw):
        assert query.dim() == 3
        assert key.shape[-1] == query.shape[-1]
        assert weights.dim() >= 2
        T, N, D = query.shape
        assert sparse_count > 0
        K = sparse_count
        indices = torch.zeros(T, 1, K, **cfg_i32)
        attn_out = torch.zeros(T, N, K, **cfg_bf16)
        return (indices, attn_out)

    def mla_prolog_v3(token_x, weight_uk, kv_cache, rope_sin, rope_cos, rmsnorm_gamma_cq, **kw):
        assert token_x.dim() == 3
        assert kv_cache.dim() == 4
        assert rope_sin.shape[-1] == rope_cos.shape[-1]
        assert token_x.shape[0] == rope_sin.shape[0]
        bs, _, _ = token_x.shape
        kv_lora_rank = kv_cache.shape[-1]
        rope_dim = rope_sin.shape[-1]
        q_lora_rank = rmsnorm_gamma_cq.shape[0]
        n_head = weight_uk.shape[0]  # W_UK_T: (num_heads, qk_nope_head_dim, kv_lora_rank)
        assert n_head > 0
        q_nope = torch.zeros(bs, n_head, kv_lora_rank, **cfg_bf16)
        q_pe = torch.zeros(bs, n_head, rope_dim, **cfg_bf16)
        q_norm = torch.zeros(bs, q_lora_rank, **cfg_bf16)
        return (q_nope, q_pe, None, q_norm, None)

    def scatter_nd_update_(x, indices, updates):
        return x

    def rotary_mul(x, cos, sin):
        return x

    def interleave_rope(x, cos, sin):
        return x

    def transpose_batchmatmul(input, weight=None, perm_x1=None, perm_x2=None, perm_y=None):
        if weight is not None:
            assert input.dim() == 3
            assert weight.dim() == 3
            x_perm = input.permute(*perm_x1) if perm_x1 else input
            weight_perm = weight.permute(*perm_x2) if perm_x2 else weight
            out = torch.bmm(x_perm, weight_perm)
            if perm_y is not None:
                out = out.permute(*perm_y)
            return out
        else:
            return input

    def sparse_flash_attention(query, key, value, query_rope, key_rope, sparse_indices, **kw):
        # only for absorbed PA
        assert query.dim() == 3
        assert key.dim() == 4
        assert value.shape == key.shape
        assert query_rope.shape[0] == query.shape[0]
        assert query_rope.shape[1] == query.shape[1]
        assert sparse_indices.dim() == 3
        assert sparse_indices.shape[0] == query.shape[0]
        T, N, D = query.shape
        attn_out = query.new_zeros(T, N, D)
        return attn_out, None

    def kv_rmsnorm_rope_cache(
        latent_kv, weight, cos, sin, slot_mapping, rope_cache, nope_cache, is_output_kv, **kw
    ):
        assert latent_kv.dim() == 4
        assert weight.dim() == 1
        assert cos.shape == sin.shape
        assert rope_cache.dim() == 4
        assert nope_cache.dim() == 4
        assert latent_kv.shape[-1] == rope_cache.shape[-1] + nope_cache.shape[-1]
        assert slot_mapping.dim() == 1
        k_pe, k_nope = None, None
        if is_output_kv:
            T = latent_kv.size(0)
            D, R = latent_kv.size(-1), cos.size(-1)
            k_pe = latent_kv.new_zeros(T, 1, 1, R)
            k_nope = latent_kv.new_zeros(T, 1, 1, D - R)
        return rope_cache, nope_cache, k_pe, k_nope

    with (
        patch.multiple(
            "torch_npu",
            npu_lightning_indexer=MagicMock(side_effect=lightning_indexer),
            npu_mla_prolog_v3=MagicMock(side_effect=mla_prolog_v3),
            npu_scatter_nd_update_=MagicMock(side_effect=scatter_nd_update_),
            npu_rotary_mul=MagicMock(side_effect=rotary_mul),
            npu_interleave_rope=MagicMock(side_effect=interleave_rope),
            npu_transpose_batchmatmul=MagicMock(side_effect=transpose_batchmatmul),
            npu_kv_rmsnorm_rope_cache=MagicMock(side_effect=kv_rmsnorm_rope_cache),
            npu_sparse_flash_attention=MagicMock(side_effect=sparse_flash_attention),
        ),
        patch(
            "torch.ops.custom",
            npu_sparse_flash_attention_enhance=MagicMock(side_effect=sparse_flash_attention),
            npu_lightning_indexer_enhance=MagicMock(side_effect=lightning_indexer),
            npu_fused_infer_attention_sink=MagicMock(side_effect=fused_infer_attention_sink),
            npu_ai_infra_kv_rmsnorm_rope_cache_v2=MagicMock(side_effect=kv_rmsnorm_rope_cache_v2),
            npu_ai_infra_sparse_flash_attention_pioneer=MagicMock(side_effect=sparse_flash_attention_pioneer),
            npu_ai_infra_fused_causal_conv1d=MagicMock(side_effect=fused_causal_conv1d),
        ),
    ):
        yield


@contextmanager
def _mock_vllm_distributed(cards: int = 4, rank: int = 0, dcp: bool = False):
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

    def all_to_all_single(output, input, output_split_sizes, input_split_sizes, **kw):
        assert output.size(0) == sum(output_split_sizes)
        assert input.size(0) == sum(input_split_sizes)
    def all_reduce(x, *_, **kw):
        return x
    def all_gather_into_tensor(output_tensor, input_tensor, **kw):
        assert output_tensor.dim() == input_tensor.dim()

    coord_0 = MockGroupCoordinator(world_size=1, rank_in_group=0)
    coord = MockGroupCoordinator(world_size=cards, rank_in_group=rank)
    with (
        patch.multiple(
            "vllm.distributed.parallel_state",
            _WORLD=coord,
            _DP=coord_0,
            _TP=coord,
            _DCP=coord if dcp else coord_0,
        ),
        patch("torch.distributed.all_to_all_single", side_effect=all_to_all_single),
        patch("torch.distributed.all_reduce", side_effect=all_reduce),
        patch("torch.distributed.all_gather_into_tensor", side_effect=all_gather_into_tensor),
    ):
        yield


@contextmanager
def _mock_omni_cache(
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    index_head_dim: int = 128,
    num_slots: int = 16,
    pg: int = 128,
):
    fake_omni = MagicMock()
    kv0 = torch.zeros(num_slots, pg, 1, kv_lora_rank, **cfg_bf16)
    kv1 = torch.zeros(num_slots, pg, 1, qk_rope_head_dim, **cfg_bf16)
    kv2 = torch.zeros(num_slots, pg, 1, index_head_dim, **cfg_bf16)
    fake_omni_npu.device_cache = (kv0, kv1, kv2)
    fake_omni_npu.synchronize_d2h = MagicMock()
    fake_omni_npu.synchronize_h2d = MagicMock()

    cache_mod = MagicMock()
    cache_mod.omni_cache = fake_omni
    omni_parent = MagicMock()
    omni_parent.cache = cache_mod

    with patch.dict(
        sys.modules,
        {"omni_cache": omni_parent, "omni_cache.cache": cache_mod},
        clear=False,
    ):
        yield


@contextmanager
def _mock_mla_attention(
    num_slots: int = 64,
    pg: int = 128,
    use_omni_cache: bool = False,
    cont_kv: bool = False,
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
            use_sparse: bool = True,
            indexer=None,
            **kwargs,
        ):
            super().__init__()
            impl = SimpleNamespace()
            impl.W_UK_T = torch.zeros(
                num_heads,
                qk_nope_head_dim,
                kv_lora_rank,
                **cfg_bf16,
            )
            impl.W_UV = torch.zeros(
                num_heads,
                kv_lora_rank,
                v_head_dim,
                **cfg_bf16,
            )
            impl.sink_kv = None
            impl.sink_k_nope = None
            self.impl = impl
            self.sink_k_pe = torch.zeros((128, qk_rope_head_dim), dtype=torch.bfloat16)
            self.sink_compressed_kv = torch.zeros((128, kv_lora_rank), dtype=torch.bfloat16)
            self.sink_populated = False

            if use_omni_cache:
                self.kv_cache = [None]
            elif cont_kv:
                idx_dim = getattr(indexer, "head_dim", 128) if indexer else 128
                kv0 = torch.zeros(num_slots, pg, 1, kv_lora_rank + qk_rope_head_dim, **cfg_bf16)
                kv1 = torch.zeros(num_slots, pg, 1, idx_dim, **cfg_bf16)
                self.kv_cache = [(kv0, kv1)]
            else:
                idx_dim = getattr(indexer, "head_dim", 128) if indexer else 128
                kv0 = torch.zeros(num_slots, pg, 1, kv_lora_rank, **cfg_bf16)
                kv1 = torch.zeros(num_slots, pg, 1, qk_rope_head_dim, **cfg_bf16)
                kv2 = torch.zeros(num_slots, pg, 1, idx_dim, **cfg_bf16)
                self.kv_cache = [(kv0, kv1, kv2)]

        def populate_sink_kv(self, k_nope_cache: torch.Tensor, k_pe_cache: torch.Tensor):
            self.sink_populated = True

        def update_sink_kv(self, k_pe: torch.Tensor, compressed_kv: torch.Tensor):
            self.sink_k_pe = k_pe
            self.sink_compressed_kv = compressed_kv
            self.sink_populated = True

    with (
        patch(f"{DSA_MODULE}.MLAAttention", MockMLAAttention, create=True),
        patch(f"{DSA_MODULE}.StaticSinkMLAAttention", MockMLAAttention, create=True),
    ):
        yield


@contextmanager
def _mock_flash_comm_linear(init_comm=None): # callback

    def flash_comm(linear, x, op: str):
        if op in ["NoOp", "AllReduce"]:
            return x
        elif op == "ReduceScatter":
            T, tp = x.size(0), linear.tp_size
            assert T % tp == 0
            return x.new_zeros(T // tp, *x.shape[1:])
        else:
            raise ValueError(f"flash_comm err op {op}")

    def init_comm_0(linear):
        from vllm.distributed import get_tp_group
        linear.x_transform = "NoOp"
        linear.y_transform = "NoOp"
        linear.tp_rank = get_tp_group().rank_in_group
        linear.tp_size = get_tp_group().world_size
        if init_comm is not None:
            init_comm(linear)

    class MockReplicatedFlashCommLinear(torch.nn.Module):

        def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            quant_config=None,
            prefix: str = "",
            **kwargs,
        ):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.weight = torch.zeros(out_features, in_features, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = flash_comm(self, x, self.x_transform)
            assert x.size(-1) == self.in_features
            y = x.new_zeros(*x.shape[:-1], self.out_features)
            return flash_comm(self, y, self.y_transform), None

    class MockColumnParallelFlashCommLinear(torch.nn.Module):

        def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            quant_config=None,
            prefix: str = "",
            **kwargs,
        ):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.out_per_part = max(1, out_features // self.tp_size)
            self.weight = torch.zeros(self.out_per_part, in_features, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = flash_comm(self, x, self.x_transform)
            assert x.size(-1) == self.in_features
            y = x.new_zeros(*x.shape[:-1], self.out_per_part)
            return flash_comm(self, y, self.y_transform), None

    class MockRowParallelFlashCommLinear(torch.nn.Module):

        def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            quant_config=None,
            prefix: str = "",
        ):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.in_per_part = max(1, in_features // self.tp_size)
            self.weight = torch.zeros(out_features, self.in_per_part, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = flash_comm(self, x, self.x_transform)
            assert x.size(-1) == self.in_per_part
            y = x.new_zeros(*x.shape[:-1], self.out_features)
            return flash_comm(self, y, self.y_transform), None

        def requires_input_partition(self):
            return self.tp_size > 1 and self.x_transform != "DP2TPAll2All"

    with (
        patch(f"{DSA_MODULE}.ReplicatedFlashCommLinear", MockReplicatedFlashCommLinear),
        patch(f"{DSA_MODULE}.ColumnParallelFlashCommLinear", MockColumnParallelFlashCommLinear),
        patch(f"{DSA_MODULE}.RowParallelFlashCommLinear", MockRowParallelFlashCommLinear),
    ):
        yield


@contextmanager
def _mock_layernorm_rmsnorm():
    class MockLayerNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones(dim, **cfg_bf16),
            )
            self.eps = eps

        def forward(self, x: torch.Tensor):
            assert x.size(-1) == self.weight.size(0)
            return x

    class MockRMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6, dtype=None):
            super().__init__()
            weight_cfg = dict(cfg_bf16)
            if dtype is not None:
                weight_cfg["dtype"] = dtype
            self.weight = torch.nn.Parameter(
                torch.ones(dim, **weight_cfg),
            )
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor):
            assert x.size(-1) == self.weight.size(0)
            return x

    with (
        patch(f"{DSA_MODULE}.LayerNorm", MockLayerNorm),
        patch(f"{DSA_MODULE}.RMSNorm", MockRMSNorm),
    ):
        yield


@contextmanager
def _mock_mome():
    class MockAggregateConv:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, x, **kwargs):
            return x

    class MockMomeAttention:

        def __init__(self, *args, state_shapes, **kwargs):
            class MOME:
                def __init__(self, *args, **kwargs):
                    self.weight = None
                def __call__(self, x, *args, **kwargs):
                    return x
            self.qa_conv = MOME()
            self.compresskv_conv = MOME()
            self.o_conv = MOME()

            # caches = [torch.zeros(16, 48, *size) for size in state_shapes]
            # self.kv_cache = [(*caches,)]
            self.prefix = ".conv"
            self.kv_cache = MagicMock()

        def __call__(self, x, *args, **kwargs):
            return x

    import omni_npu.v1.layers.attention.npu_dsa as mla_mod
    mla_mod.AggregateConv = MockAggregateConv
    mla_mod.MomeAttention = MockMomeAttention
    yield
    mla_mod.AggregateConv = None
    mla_mod.MomeAttention = None


# =========================
# 3. 配置相关 mock
# =========================

def _make_dsa_config(
    index_topk=2048,
    index_n_heads=64,
    index_head_dim=128,
    qk_rope_head_dim=64,
    rms_norm_eps=1e-6,
    rope_type="default",
    factor=2.0,
    mscale_all_dim=False,
    apply_yarn_scaling=True,
    indexer_rope_interleave=False,
    kv_lora_rank=512,
    use_mome=False,
    param_sink_number=0,
    rope_scaling=None,
    num_hidden_layers=49,
    is_mtp_layer=False,
    num_nextn_predict_layers=0,
    index_topk_freq=1,
    indexer_types=None,
    index_topk_pattern=None,
    index_skip_topk_offset=2,
):
    return SimpleNamespace(
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_parameters={
            "rope_type": rope_type,
            "factor": factor,
            "mscale_all_dim": mscale_all_dim,
            "apply_yarn_scaling": apply_yarn_scaling,
            "rope_theta": 10000.0,
        },
        rope_interleave=True,
        indexer_rope_interleave=indexer_rope_interleave,
        kv_lora_rank=kv_lora_rank,
        use_mome=use_mome,
        router_sliding_window=3,
        param_sink_number=param_sink_number,
        param_sink_with_value=True,
        torch_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        rope_scaling=rope_scaling,
        num_hidden_layers=num_hidden_layers,
        is_mtp_layer=is_mtp_layer,
        num_nextn_predict_layers=num_nextn_predict_layers,
        index_topk_freq=index_topk_freq,
        indexer_types=indexer_types,
        index_topk_pattern=index_topk_pattern,
        index_skip_topk_offset=index_skip_topk_offset,
    )


def _make_vllm_config(
    speculative_config=None,
    cudagraph_capture_sizes=None,
):
    return SimpleNamespace(
        speculative_config=speculative_config,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            static_forward_context={},
        ),
        cache_config=SimpleNamespace(block_size=128),
    )


@contextmanager
def _mock_model_extra_config(
    seq_parallel=False,
    dsa_seq_parallel=False,
    omni_cache=False,
    mlaprolog_op=False,
    mtp_remove_redundant_kv=False,
    enable_dsa=False,
    sharded_o_proj=False,
    use_noncontiguous_kv=False,
    use_batch_invariant_op=False,
    enable_precision_strong_consistency=False,
    dtype=torch.bfloat16,
):
    with patch(
        f"{DSA_MODULE}.model_extra_config",
        MagicMock(
            dtype=dtype,
            parall_config=MagicMock(
                ena_seq_parallel=seq_parallel,
                ena_context_parallel=dsa_seq_parallel,
                sharded_o_proj=sharded_o_proj,
            ),
            operator_opt_config=MagicMock(
                use_omni_cache=omni_cache,
                enable_mlaprolog=mlaprolog_op,
                mtp_remove_redundant_kv=mtp_remove_redundant_kv,
                enable_dsa=enable_dsa,
                use_noncontiguous_kv=use_noncontiguous_kv,
                merge_q_kv_conv=False,
                use_batch_invariant_op=use_batch_invariant_op,
                enable_precision_strong_consistency=enable_precision_strong_consistency,
            ),
        ),
    ):
        yield


@contextmanager
def _mock_forward_context(seq_lens: list, mode: str="prefill", use_mome=False):
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        virtual_engine=0,
        capturing=False,
        no_compile_layers={},
    )

    class _StageMetadata:
        def __init__(self, q_lens: list, kv_lens: list, pg: int = 128):
            cu_q_lens = torch.cumsum(torch.tensor(q_lens, **cfg_i32), dim=0)
            self.query_cumlens = cu_q_lens
            self.query_start_loc = torch.tensor([0] + cu_q_lens.tolist(), **cfg_i32)
            self.seq_lens = torch.tensor(kv_lens, **cfg_i32)
            blocks = max(32, sum(kv_lens) // pg + 1)
            self.block_table = torch.zeros(len(q_lens), blocks, **cfg_i32)
            self.prefix_meta = None
            self.slot_mapping_2d = None

    if mode == "pd_mixed":
        ctx.attn_metadata.prefill = _StageMetadata(seq_lens, seq_lens)
        ctx.attn_metadata.decode = _StageMetadata([1] * len(seq_lens), seq_lens)
        ctx.attn_metadata.slot_mapping = torch.arange(len(seq_lens) + sum(seq_lens), **cfg_i64)
        ctx.attn_metadata.num_decodes = len(seq_lens)
        ctx.attn_metadata.num_prefills = len(seq_lens)
        ctx.attn_metadata.num_actual_tokens = len(seq_lens) + sum(seq_lens)
        ctx.attn_metadata.num_decode_tokens = len(seq_lens)
    elif mode == "prefill":
        ctx.attn_metadata.prefill = _StageMetadata(seq_lens, seq_lens)
        ctx.attn_metadata.decode = None
        ctx.attn_metadata.slot_mapping = torch.arange(sum(seq_lens), **cfg_i64)
        ctx.attn_metadata.num_decodes = 0
        ctx.attn_metadata.num_prefills = len(seq_lens)
        ctx.attn_metadata.num_actual_tokens = sum(seq_lens)
        ctx.attn_metadata.num_decode_tokens = 0

        from omni_npu.v1.layers.attention.npu_dsa import (
            get_tp_group,
            get_dcp_group,
            model_extra_config,
            SPManager,
            KVSPMaganer,
        )

        from omni_npu.attention.backends.utils import paged_cache
        if model_extra_config.parall_config.ena_seq_parallel:
            metadata = ctx.attn_metadata.prefill
            computed_lens = metadata.seq_lens - (metadata.query_start_loc[1:] - metadata.query_start_loc[:-1])
            metadata.sp_manager = SPManager.init_cp(
                sp_group=get_tp_group(),
                cumlens=metadata.query_start_loc,
                computed_lens=computed_lens,
                block_table_ref=metadata.block_table,
                table_size=metadata.block_table.size(1),
            )
            metadata.cache_fn = paged_cache(
                ctx.attn_metadata.slot_mapping,
                metadata.query_start_loc, # [B + 1]
            )
            if get_dcp_group().world_size > 1:
                metadata.kvsp_manager = KVSPMaganer(
                    q_cumlens=metadata.query_start_loc, # [B + 1]
                    kv_lens=metadata.seq_lens,
                    blk_table=metadata.block_table,
                )

    elif mode == "decode":
        ctx.attn_metadata.decode = _StageMetadata([1] * len(seq_lens), seq_lens)
        ctx.attn_metadata.prefill = None
        ctx.attn_metadata.slot_mapping = torch.arange(len(seq_lens), **cfg_i64)
        ctx.attn_metadata.decode.slot_mapping = ctx.attn_metadata.slot_mapping
        ctx.attn_metadata.num_decodes = 2
        ctx.attn_metadata.num_prefills = 0
        ctx.attn_metadata.num_actual_tokens = 2
        ctx.attn_metadata.num_decode_tokens = 2

        from omni_npu.v1.layers.attention.npu_dsa import (
            get_dcp_group,
            KVSPMaganer,
        )
        if get_dcp_group().world_size > 1:
            metadata = ctx.attn_metadata.decode
            metadata.kvsp_manager = KVSPMaganer(
                q_cumlens=metadata.query_start_loc,
                kv_lens=metadata.seq_lens,
                blk_table=metadata.block_table,
            )
    elif mode == "dummy_run":
        ctx.attn_metadata = None
    else:
        raise ValueError(f"error mode: {mode}")

    if ctx.attn_metadata is not None:
        slots = ctx.attn_metadata.slot_mapping
        ctx.attn_metadata.get_slot_mapping_2d = lambda _layer_idx=-1: torch.stack(
            [slots, slots], dim=-1
        )

    if use_mome:
        ctx.attn_metadata = {
            ".attn": ctx.attn_metadata,
            ".conv": MagicMock(),
        }

    with patch(f"{DSA_MODULE}.get_forward_context", return_value=ctx):
        yield


# =========================
# patch_and_gen_configs
# =========================

@contextmanager
def _patch_and_gen_configs(
    mode: str = "prefill",
    ena_seq_parallel: bool = False,
    ena_context_parallel: bool = False,
    use_omni_cache: bool = False,
    enable_mlaprolog: bool = False,
    use_noncontiguous_kv: bool = False,
    use_batch_invariant_op: bool = False,
    enable_precision_strong_consistency: bool = False,
    use_mome: bool = False,
    init_flash_comm = None,
    seq_lens: list = [32, 47],
    hidden_size: int = 2048,
    q_lora_rank: int = 1536,
    index_topk: int = 2048,
    index_n_heads: int = 64,
    index_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    rms_norm_eps: float = 1e-6,
    num_heads: int = 128,
    qk_nope_head_dim: int = 64,
    v_head_dim: int = 128,
    kv_lora_rank: int = 512,
    tp_size: int = 4,
    tp_rank: int = 0,
    pg: int = 128,
    rope_type: str = "default",
    indexer_rope_interleave: bool = False,
    factor: float = 2.0,
    mscale_all_dim: bool = False,
    apply_yarn_scaling: bool = True,
    param_sink_number: int = 0,
    ena_kvsp: bool = False,
    rope_scaling=None,
    num_hidden_layers: int = 49,
    is_mtp_layer: bool = False,
    num_nextn_predict_layers: int = 0,
    index_topk_freq: int = 1,
    indexer_types: list | None = None,
    index_topk_pattern: str | None = None,
    index_skip_topk_offset: int = 2,
    layer_idx: int | None = None,
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
        layer_idx=layer_idx,
    )
    cfg = _make_dsa_config(
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_type=rope_type,
        factor=factor,
        mscale_all_dim=mscale_all_dim,
        apply_yarn_scaling=apply_yarn_scaling,
        indexer_rope_interleave=indexer_rope_interleave,
        kv_lora_rank=kv_lora_rank,
        use_mome=use_mome,
        param_sink_number=param_sink_number,
        rope_scaling=rope_scaling,
        num_hidden_layers=num_hidden_layers,
        is_mtp_layer=is_mtp_layer,
        num_nextn_predict_layers=num_nextn_predict_layers,
        index_topk_freq=index_topk_freq,
        indexer_types=indexer_types,
        index_topk_pattern=index_topk_pattern,
        index_skip_topk_offset=index_skip_topk_offset,
    )
    vllm_cfg = _make_vllm_config(
        speculative_config=None,
    )
    mock_omni = _mock_omni_cache(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        index_head_dim=index_head_dim,
        num_slots=16, pg=pg,
    ) if use_omni_cache else nullcontext()

    # patch讲究顺序
    with (
        # 基础模块
        _mock_torch_npu_stream(),
        _mock_torch_npu(),
        _mock_misc(),
        # 配置项
        _mock_vllm_distributed(cards=tp_size, rank=tp_rank, dcp=ena_kvsp),
        _mock_model_extra_config(
            seq_parallel=ena_seq_parallel,
            dsa_seq_parallel=ena_context_parallel,
            omni_cache=use_omni_cache,
            mlaprolog_op=enable_mlaprolog,
            enable_dsa=True,
            use_noncontiguous_kv=use_noncontiguous_kv,
            use_batch_invariant_op=use_batch_invariant_op,
            enable_precision_strong_consistency=enable_precision_strong_consistency,
        ),
        patch(f"{DSA_MODULE}.get_current_vllm_config", return_value=vllm_cfg, create=True),
        # 功能模块
        _mock_flash_comm_linear(init_comm=init_flash_comm),
        _mock_layernorm_rmsnorm(),
        _mock_mome(),
        _mock_mla_attention(
            num_slots=64, pg=pg,
            use_omni_cache=use_omni_cache,
            cont_kv=use_noncontiguous_kv,
        ),
        mock_omni,
        # 运行时
        _mock_forward_context(seq_lens=seq_lens, mode=mode, use_mome=use_mome),
    ):
        yield cfg, vllm_cfg, env


# =========================
# Indexer 用例
# =========================

class _ConstProj(torch.nn.Module):
    def __init__(self, out: torch.Tensor):
        super().__init__()
        self._out = out

    def forward(self, _x: torch.Tensor):
        return self._out, None


class _IdentityNorm(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x


def _install_indexer_li_prolog_fakes(
    idx,
    *,
    wi_raw: torch.Tensor,
    qi_flat: torch.Tensor,
    ki_flat: torch.Tensor,
):
    idx.weights_proj = _ConstProj(wi_raw.clone())
    idx.wq_b = _ConstProj(qi_flat)
    idx.wk = _ConstProj(ki_flat)
    idx.k_norm = _IdentityNorm()
    idx._apply_rope = lambda x, *_args: x


class TestIndexer:

    def test_case_1(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            indexer_rope_interleave=False,
            rope_type="default",
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import Indexer, get_forward_context

            idx = Indexer(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                q_lora_rank=env.q_lora_rank,
                quant_config=None,
                cache_config=vllm_cfg.cache_config,
                sink_len=128,
            )
            D, R = env.hidden_size, env.qk_rope_head_dim
            QR, KI = env.q_lora_rank, cfg.index_head_dim

            attn_metadata = get_forward_context().attn_metadata
            meta = attn_metadata.prefill
            meta.slot_mapping = attn_metadata.slot_mapping
            meta.slot_mapping_2d = attn_metadata.get_slot_mapping_2d()
            T = meta.query_cumlens.flatten()[-1].item()
            K = cfg.index_topk
            tok_idx, ki = idx.forward(
                x=torch.zeros(T, D, **env.cfg_bf16),
                qr=torch.zeros(T, QR, **env.cfg_bf16),
                cos=torch.zeros(T, 1, 1, R, **env.cfg_bf16),
                sin=torch.zeros(T, 1, 1, R, **env.cfg_bf16),
                attn_metadata=meta,
                ki_cache=torch.zeros(16, env.pg, 1, KI, **env.cfg_bf16),
            )
            assert tok_idx.shape == (T, 1, K)
            assert ki.shape == (T, 1, KI)

    def test_case_2(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            indexer_rope_interleave=True,
            rope_type="deepseek_yarn",
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import Indexer, get_forward_context

            idx = Indexer(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                q_lora_rank=env.q_lora_rank,
                quant_config=None,
                cache_config=vllm_cfg.cache_config,
            )
            D, R = env.hidden_size, env.qk_rope_head_dim
            QR, KI = env.q_lora_rank, cfg.index_head_dim

            attn_metadata = get_forward_context().attn_metadata
            meta = attn_metadata.prefill
            meta.slot_mapping = attn_metadata.slot_mapping
            meta.slot_mapping_2d = attn_metadata.get_slot_mapping_2d()
            T = meta.query_cumlens.flatten()[-1].item()
            K = cfg.index_topk
            tok_idx, ki = idx.forward(
                x=torch.zeros(T, D, **env.cfg_bf16),
                qr=torch.zeros(T, QR, **env.cfg_bf16),
                cos=torch.zeros(T, 1, 1, R, **env.cfg_bf16),
                sin=torch.zeros(T, 1, 1, R, **env.cfg_bf16),
                attn_metadata=meta,
                ki_cache=torch.zeros(16, env.pg, 1, KI, **env.cfg_bf16),
            )
            assert tok_idx.shape == (T, 1, K)
            assert ki.shape == (T, 1, KI)

    def test_precision_strong_consistency_sets_scales(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            enable_precision_strong_consistency=True,
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import Indexer

            idx = Indexer(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                q_lora_rank=env.q_lora_rank,
                quant_config=None,
                cache_config=vllm_cfg.cache_config,
                sink_len=128,
            )
            assert idx.softmax_scale == pytest.approx(cfg.index_head_dim ** -0.5)
            assert idx.weights_scale == pytest.approx(cfg.index_n_heads ** -0.5)

    def test_precision_strong_consistency_disabled_skips_scales(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            enable_precision_strong_consistency=False,
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import Indexer

            idx = Indexer(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                q_lora_rank=env.q_lora_rank,
                quant_config=None,
                cache_config=vllm_cfg.cache_config,
                sink_len=128,
            )
            assert not hasattr(idx, "softmax_scale")
            assert not hasattr(idx, "weights_scale")

    def test_precision_strong_consistency_scales_wi(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            enable_precision_strong_consistency=True,
            index_n_heads=4,
            index_head_dim=8,
            hidden_size=16,
            q_lora_rank=16,
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import Indexer

            idx = Indexer(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                q_lora_rank=env.q_lora_rank,
                quant_config=None,
                cache_config=vllm_cfg.cache_config,
                sink_len=128,
            )
            T, N, D = 3, cfg.index_n_heads, cfg.index_head_dim
            wi_raw = torch.full((T, N), 2.0, dtype=torch.float32)
            _install_indexer_li_prolog_fakes(
                idx,
                wi_raw=wi_raw,
                qi_flat=torch.zeros(T, N * D, dtype=torch.float32),
                ki_flat=torch.zeros(T, D, dtype=torch.float32),
            )

            wi, _qi, _ki = idx._li_prolog_ext(
                wx=torch.zeros(T, env.hidden_size, dtype=torch.float32),
                qr=torch.zeros(T, env.q_lora_rank, dtype=torch.float32),
                kx=torch.zeros(T, env.hidden_size, dtype=torch.float32),
                q_cos_sin=(torch.zeros(1), torch.zeros(1)),
                k_cos_sin=(torch.zeros(1), torch.zeros(1)),
            )

            expected = wi_raw * idx.weights_scale * idx.softmax_scale
            torch.testing.assert_close(wi, expected)

    def test_precision_strong_consistency_disabled_keeps_raw_wi(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            enable_precision_strong_consistency=False,
            index_n_heads=4,
            index_head_dim=8,
            hidden_size=16,
            q_lora_rank=16,
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import Indexer

            idx = Indexer(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                q_lora_rank=env.q_lora_rank,
                quant_config=None,
                cache_config=vllm_cfg.cache_config,
                sink_len=128,
            )
            T, N, D = 3, cfg.index_n_heads, cfg.index_head_dim
            wi_raw = torch.full((T, N), 2.0, dtype=torch.float32)
            _install_indexer_li_prolog_fakes(
                idx,
                wi_raw=wi_raw,
                qi_flat=torch.zeros(T, N * D, dtype=torch.float32),
                ki_flat=torch.zeros(T, D, dtype=torch.float32),
            )

            wi, _qi, _ki = idx._li_prolog_ext(
                wx=torch.zeros(T, env.hidden_size, dtype=torch.float32),
                qr=torch.zeros(T, env.q_lora_rank, dtype=torch.float32),
                kx=torch.zeros(T, env.hidden_size, dtype=torch.float32),
                q_cos_sin=(torch.zeros(1), torch.zeros(1)),
                k_cos_sin=(torch.zeros(1), torch.zeros(1)),
            )

            torch.testing.assert_close(wi, wi_raw)

    def test_forward_with_buffer(self):
        """Indexer.forward with topk_indices_buffer → 覆盖 line 276"""
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            indexer_rope_interleave=False,
            rope_type="default",
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import Indexer, get_forward_context

            idx = Indexer(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                q_lora_rank=env.q_lora_rank,
                quant_config=None,
                cache_config=vllm_cfg.cache_config,
                sink_len=128,
            )
            D, R = env.hidden_size, env.qk_rope_head_dim
            QR, KI = env.q_lora_rank, cfg.index_head_dim

            attn_metadata = get_forward_context().attn_metadata
            meta = attn_metadata.prefill
            meta.slot_mapping = attn_metadata.slot_mapping
            meta.slot_mapping_2d = attn_metadata.get_slot_mapping_2d()
            T = meta.query_cumlens.flatten()[-1].item()
            K = cfg.index_topk
            buf = torch.empty(128, K, dtype=torch.int32, device="cpu")
            tok_idx, ki = idx.forward(
                x=torch.zeros(T, D, **env.cfg_bf16),
                qr=torch.zeros(T, QR, **env.cfg_bf16),
                cos=torch.zeros(T, 1, 1, R, **env.cfg_bf16),
                sin=torch.zeros(T, 1, 1, R, **env.cfg_bf16),
                attn_metadata=meta,
                ki_cache=torch.zeros(16, env.pg, 1, KI, **env.cfg_bf16),
                topk_indices_buffer=buf,
            )
            assert tok_idx.shape == (T, 1, K)
            assert ki.shape == (T, 1, KI)


# =========================
# NPUDeepseekSparseAttention 用例
# =========================

class TestNPUDeepseekSparseAttention:

    def _test_with_cfg(self,
        mode: str = "prefill",
        ena_seq_parallel: bool = False,
        ena_context_parallel: bool = False,
        use_noncontiguous_kv: bool = False,
        use_batch_invariant_op: bool = False,
        use_mome: bool = False,
        use_omni_cache: bool = False,
        enable_mlaprolog: bool = False,
        qk_nope_head_dim: int = 512,
        rope_type: str = "deepseek_yarn",
        param_sink_number: int = 0,
        ena_kvsp: bool = False,
        seq_lens: list | None = None,
        o_proj_unit_tp: bool = False,
        rope_scaling=None,
        num_hidden_layers: int = 49,
        is_mtp_layer: bool = False,
        num_nextn_predict_layers: int = 0,
        index_topk_freq: int = 1,
        indexer_types: list | None = None,
        index_topk_pattern: str | None = None,
        index_skip_topk_offset: int = 2,
        prefix: str | None = None,
        use_topk_indices_buffer: bool = False,
    ):
        def init_flash_comm(linear):
            if ena_seq_parallel:
                if ena_context_parallel:
                    if "q_b_proj" in linear.prefix:
                        linear.tp_size = 1
                    if "kv_b_proj" in linear.prefix:
                        linear.tp_size = 1
                if "o_proj" in linear.prefix:
                    if o_proj_unit_tp:
                        linear.tp_size = 1
                        linear.y_transform = "NoOp"
                    else:
                        linear.y_transform = "ReduceScatter"

        with _patch_and_gen_configs(
            mode=mode,
            ena_seq_parallel=ena_seq_parallel,
            ena_context_parallel=ena_context_parallel,
            use_noncontiguous_kv=use_noncontiguous_kv,
            use_batch_invariant_op=use_batch_invariant_op,
            use_mome=use_mome,
            use_omni_cache=use_omni_cache,
            enable_mlaprolog=enable_mlaprolog,
            qk_nope_head_dim=qk_nope_head_dim,
            rope_type=rope_type,
            init_flash_comm=init_flash_comm,
            param_sink_number=param_sink_number,
            ena_kvsp=ena_kvsp,
            seq_lens=seq_lens if seq_lens is not None else [32, 47],
            rope_scaling=rope_scaling,
            num_hidden_layers=num_hidden_layers,
            is_mtp_layer=is_mtp_layer,
            num_nextn_predict_layers=num_nextn_predict_layers,
            index_topk_freq=index_topk_freq,
            indexer_types=indexer_types,
            index_topk_pattern=index_topk_pattern,
            index_skip_topk_offset=index_skip_topk_offset,
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import (
                NPUDeepseekSparseAttention,
                get_forward_context,
            )
            import omni_npu.v1.layers.attention.npu_dsa as npu_dsa_mod
            if hasattr(npu_dsa_mod, "npu_dsa_forward"):
                torch.ops.vllm.npu_dsa_forward = npu_dsa_mod.npu_dsa_forward

            topk_indices_buffer = None
            if use_topk_indices_buffer:
                topk_indices_buffer = torch.empty(
                    128, cfg.index_topk, dtype=torch.int32, device="cpu"
                )

            attn_kwargs = dict(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                num_heads=env.num_heads,
                qk_nope_head_dim=env.qk_nope_head_dim,
                qk_rope_head_dim=env.qk_rope_head_dim,
                v_head_dim=env.v_head_dim,
                q_lora_rank=env.q_lora_rank,
                kv_lora_rank=env.kv_lora_rank,
                topk_indices_buffer=topk_indices_buffer,
                cache_config=vllm_cfg.cache_config,
                quant_config=None,
            )
            if prefix is not None:
                attn_kwargs["prefix"] = prefix

            m = NPUDeepseekSparseAttention(**attn_kwargs)
            if not hasattr(m, "conv"):
                m.conv = None
            D, R = env.hidden_size, env.qk_rope_head_dim
            forward_ctx = get_forward_context()
            forward_ctx.no_compile_layers = {m.prefix: m}
            attn_metadata = forward_ctx.attn_metadata
            if type(attn_metadata) is dict:
                attn_metadata = attn_metadata[".attn"]
            if attn_metadata:
                T = 0
                for meta in [attn_metadata.prefill, attn_metadata.decode]:
                    if meta is not None:
                        T += meta.query_cumlens.flatten()[-1].item()
            else:
                T = 65536

            T0 = T # for cos, sin
            if ena_seq_parallel:
                T = -(-T // env.tp_size) # ceil_div
            out = m.forward(
                torch.zeros(T, D, **env.cfg_bf16),
                cos=torch.zeros(T0, 1, 1, R, **env.cfg_bf16),
                sin=torch.zeros(T0, 1, 1, R, **env.cfg_bf16),
            )
            assert out.shape == (T, D)

    def test_dummy_run(self):
        self._test_with_cfg(mode="dummy_run")

    def test_prefill(self):
        self._test_with_cfg(mode="prefill")

    def test_decode(self):
        self._test_with_cfg(mode="decode")

    def test_pd_mixed(self):
        self._test_with_cfg(mode="pd_mixed")

    def test_decode_op_mlaprolog(self):
        self._test_with_cfg(
            mode="decode",
            enable_mlaprolog=True,
            rope_type="default"
        )

    def test_prefill_sp(self):
        self._test_with_cfg(
            mode="prefill",
            qk_nope_head_dim=128,
            ena_seq_parallel=True,
        )

    def test_prefill_cp(self):
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
        )

    def test_prefill_kvsp(self):
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
            ena_kvsp=True,
        )

    def test_decode_sp(self):
        self._test_with_cfg(
            mode="decode",
            ena_seq_parallel=True,
        )

    def test_pd_mixed_sp(self):
        self._test_with_cfg(
            mode="pd_mixed",
            ena_seq_parallel=True,
        )

    def test_prefill_sink(self):
        self._test_with_cfg(
            mode="prefill",
            param_sink_number=128,
        )

    def test_decode_sink(self):
        self._test_with_cfg(
            mode="decode",
            param_sink_number=128,
        )

    def test_prefill_mome(self):
        self._test_with_cfg(
            mode="prefill",
            use_mome=True,
            use_noncontiguous_kv=True,
            param_sink_number=128,
        )

    def test_prefill_mome_cp(self):
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
            use_mome=True,
            use_noncontiguous_kv=True,
            param_sink_number=128,
        )

    def test_prefill_mome_cp_fallback_o_proj_tp1(self):
        """
        _forward_prefill_cp uses the standard MoME path with context parallelism.
        With o_proj.tp_size==1, it returns after _maybe_mome_out and o_proj.
        """
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
            use_mome=True,
            use_noncontiguous_kv=True,
            param_sink_number=128,
            seq_lens=[8, 8],
            o_proj_unit_tp=True,
        )

    def test_decode_mome(self):
        self._test_with_cfg(
            mode="decode",
            use_mome=True,
            use_noncontiguous_kv=True,
            param_sink_number=128,
        )

    def test_prefill_noncontiguous_batch_invariant(self):
        self._test_with_cfg(
            mode="prefill",
            use_noncontiguous_kv=True,
            use_batch_invariant_op=True,
            param_sink_number=128,
        )

    def test_mrope_without_rope_scaling(self):
        self._test_with_cfg(rope_scaling=None)

    def test_mrope_rope_scaling_no_mrope_section(self):
        self._test_with_cfg(rope_scaling={"factor": 2.0})

    def test_mrope_with_mrope_section(self):
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [12, 10, 10]},
            num_hidden_layers=49,
        )

    def test_mrope_with_mrope_section_mtp_layer(self):
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [12, 10, 10]},
            num_hidden_layers=49,
            is_mtp_layer=True,
            num_nextn_predict_layers=3,
        )

    def test_mrope_with_mrope_section_not_mtp_layer(self):
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [12, 10, 10]},
            num_hidden_layers=49,
            is_mtp_layer=False,
            num_nextn_predict_layers=3,
        )

    # =========================
    # topk_indices_buffer is not None 分支覆盖
    # =========================

    def test_prefill_full_layer_with_buffer(self):
        """Full layer + prefill + buffer → 覆盖 _forward_prefill 写 buffer (line 1029)"""
        self._test_with_cfg(mode="prefill", use_topk_indices_buffer=True)

    def test_decode_full_layer_with_buffer(self):
        """Full layer + decode + buffer → 覆盖 _forward_decode 写 buffer (line 1273)"""
        self._test_with_cfg(mode="decode", use_topk_indices_buffer=True)

    def test_prefill_cp_full_layer_with_buffer(self):
        """Full layer + CP + buffer → 覆盖 _forward_prefill_cp 写 buffer (line 1173)"""
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
            use_topk_indices_buffer=True,
        )

    def test_prefill_shared_layer_with_buffer(self):
        """Shared layer + prefill + buffer → 覆盖 index_topk_pattern (line 450),
        self.indexer=None (line 471), _forward_prefill 读 buffer (lines 1031-1032)"""
        self._test_with_cfg(
            mode="prefill",
            index_topk_pattern="S",
            use_topk_indices_buffer=True,
        )

    def test_decode_shared_layer_with_buffer(self):
        """Shared layer + decode + buffer → 覆盖 _forward_decode 读 buffer (line 1275)"""
        self._test_with_cfg(
            mode="decode",
            index_topk_pattern="S",
            use_topk_indices_buffer=True,
        )

    def test_dummy_run_shared_layer(self):
        """Shared layer + dummy_run → 覆盖 _forward_prefill else 分支 dummy path (line 1034)"""
        self._test_with_cfg(mode="dummy_run", index_topk_pattern="S")

    def test_decode_kvsp_with_buffer(self):
        """Decode + KVSP + buffer → 覆盖 KVSP 路径 (lines 1255-1260, 1262) 及写 buffer (line 1273)"""
        self._test_with_cfg(
            mode="decode",
            ena_kvsp=True,
            use_topk_indices_buffer=True,
        )

    def test_prefill_indexer_types_shared(self):
        """Shared layer via indexer_types → 覆盖 indexer_types 分支 (line 452)"""
        self._test_with_cfg(
            mode="prefill",
            indexer_types=["shared"],
            use_topk_indices_buffer=True,
        )

@pytest.mark.unit
def test_indexer_update_cache_uses_a5_scatter(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.on_ascend950 = True
    scatter_mock = MagicMock()
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_scatter_nd_update_", scatter_mock)

    ki = torch.zeros(3, 1, 8)
    slots_2d = torch.arange(3, dtype=torch.int64).view(3, 1)
    ki_cache = torch.zeros(4, 8, 1, 8)

    Indexer._update_cache(idx, ki, slots_2d, ki_cache)

    scatter_mock.assert_called_once()
    assert scatter_mock.call_args.args[0] is ki_cache
    assert torch.equal(scatter_mock.call_args.args[1], slots_2d)


@pytest.mark.unit
def test_indexer_noncontiguous_lightning_uses_a5_op(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.on_ascend950 = True
    idx.topk_tokens = 7
    expected = torch.ones(2, 1, 7, dtype=torch.int32)
    lightning_mock = MagicMock(return_value=(expected,))
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_lightning_indexer", lightning_mock)
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(use_noncontiguous_kv=True)),
    )

    out = Indexer._apply_lightning_indexer(
        idx,
        wi=torch.zeros(2, 4),
        qi=torch.zeros(2, 3, 8),
        ki_cache=torch.zeros(5, 8),
        q_cumlens=torch.tensor([2], dtype=torch.int32),
        kv_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )

    assert out is expected
    lightning_mock.assert_called_once()
    assert lightning_mock.call_args.kwargs["sparse_count"] == 7
    assert "sparse_block_size" not in lightning_mock.call_args.kwargs


@pytest.mark.unit
def test_sparse_kv_norm_rope_cache_a5_updates_combined_cache(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.qk_rope_head_dim = 2
    fake.kv_lora_rank = 4
    fake.on_ascend950 = True
    fake.noncontiguous_kv = False
    fake.rope_interleaved = False
    fake.kv_a_layernorm = lambda x: x
    fake._apply_rope = lambda x, cos, sin: x

    scatter_mock = MagicMock()
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_scatter_nd_update_", scatter_mock)

    slots = SimpleNamespace(
        slot_mapping=torch.arange(3, dtype=torch.int64),
        slot_mapping_2d=torch.arange(3, dtype=torch.int64).view(3, 1),
    )
    kv_cache = (torch.zeros(4, 8, 1, 6), torch.zeros(4, 8, 1, 2))

    k_nope, k_pe = NPUDeepseekSparseAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(3, 6),
        cos=torch.zeros(3, 1, 1, 2),
        sin=torch.zeros(3, 1, 1, 2),
        slots=slots,
        kv_cache=kv_cache,
    )

    assert k_nope.shape == (3, 1, 4)
    assert k_pe.shape == (3, 1, 2)
    scatter_mock.assert_called_once()
    assert scatter_mock.call_args.args[0] is kv_cache[0]
    assert scatter_mock.call_args.args[2].shape == (3, 6)


@pytest.mark.unit
def test_sparse_kv_norm_rope_cache_a5_uses_v2_for_noncontiguous_kv(monkeypatch):
    """A5 + noncontiguous_kv: npu_ai_infra_kv_rmsnorm_rope_cache_v2 is used instead of scatter."""
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    R, L, T = 2, 4, 3

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
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
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_kv_rmsnorm_rope_cache=True)
        ),
    )
    combined_cache = torch.zeros(4, 8, 1, L + R)
    kv_cache = (combined_cache, torch.zeros(4, 8, 1, R))

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
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_scatter_nd_update_", scatter_mock)

    k_nope, k_pe = NPUDeepseekSparseAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(T, L + R),
        cos=torch.zeros(T, 1, 1, R),
        sin=torch.zeros(T, 1, 1, R),
        slots=torch.arange(T, dtype=torch.int64),
        kv_cache=kv_cache,
    )

    assert k_nope.shape == (T, 1, L)
    assert k_pe.shape == (T, 1, R)
    v2_mock.assert_called_once()
    scatter_mock.assert_not_called()


@pytest.mark.unit
def test_post_attn_absorb_a5_uses_transposed_input(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.on_ascend950 = True
    fake.attn = SimpleNamespace(
        impl=SimpleNamespace(W_UV=torch.zeros(2, 4, 5))
    )
    captured = {}

    def fake_bmm(**kwargs):
        captured.update(kwargs)
        return torch.zeros(2, 3, 5)

    monkeypatch.setattr(
        dsa_mod.torch_npu,
        "npu_transpose_batchmatmul",
        MagicMock(side_effect=fake_bmm),
    )

    out = NPUDeepseekSparseAttention._post_attn_absorb(
        fake,
        torch.zeros(3, 2, 4),
    )

    assert out.shape == (3, 10)
    assert captured["input"].shape == (3, 2, 4)
    assert captured["perm_x1"] == (1, 0, 2)


@pytest.mark.unit
def test_kv_norm_rope_cache_noncontiguous_batch_invariant_uses_scatter_block_update(
    monkeypatch,
):
    """When enable_kv_rmsnorm_rope_cache is enabled, fused_op is disabled and the non-contiguous
    cache update falls back to npu_ai_infra_scatter_block_update_."""
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.qk_rope_head_dim = 2
    fake.kv_lora_rank = 4
    fake.on_ascend950 = False
    fake.noncontiguous_kv = True
    fake.kv_nz = False
    fake.kv_a_layernorm = lambda x: x
    fake._apply_rope = lambda x, cos, sin: x

    monkeypatch.setattr(
        dsa_mod,
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
        dsa_mod.torch_npu,
        "npu_scatter_nd_update_",
        MagicMock(side_effect=fake_scatter_nd_update),
    )

    slots = SimpleNamespace(
        slot_mapping=torch.arange(3, dtype=torch.int64),
        slot_mapping_2d=torch.arange(3, dtype=torch.int64).view(3, 1),
    )
    kv_cache = (torch.zeros(4, 8, 1, 6), torch.zeros(4, 8, 1, 2))

    k_nope, k_pe = NPUDeepseekSparseAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(3, 6),
        cos=torch.zeros(3, 1, 1, 2),
        sin=torch.zeros(3, 1, 1, 2),
        slots=slots,
        kv_cache=kv_cache,
    )

    assert k_nope.shape == (3, 1, 4)
    assert k_pe.shape == (3, 1, 2)
    assert len(block_calls) == 1
    assert len(nd_calls) == 0
    cache_arg, indices_arg, data_arg = block_calls[0]
    assert cache_arg is kv_cache[0]
    assert torch.equal(indices_arg, slots.slot_mapping_2d)
    assert data_arg.shape == (3, 6)


@pytest.mark.unit
def test_apply_attention_rescale_pioneer_calls_custom_ops(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    T, N, L, R, S, B = 2, 2, 4, 2, 3, 1
    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.scaling = 0.5
    fake.param_sink_number = S
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.dummy_value_cache = torch.zeros(1)
    fake.attn = SimpleNamespace(
        impl=SimpleNamespace(
            sink_k_nope=torch.zeros(S, 1, L),
            sink_kv=torch.zeros(S, 1, L + R),
        )
    )

    expected = torch.full((T, N, L), 7.0)
    pioneer = MagicMock(
        return_value=(
            torch.zeros(T, N, L),
            torch.zeros(1, T, N),
            torch.ones(1, T, N),
        )
    )
    meta = MagicMock(return_value="meta")
    sink_v2 = MagicMock(
        return_value=(
            torch.ones(T, N, L),
            None,
            torch.zeros(T, N, 1),
            torch.ones(T, N, 1),
        )
    )
    rescale = MagicMock(return_value=(expected, None, None))
    monkeypatch.setattr(dsa_mod.ops, "apply_FA_rescale_forward", rescale)
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_ai_infra_sparse_flash_attention_pioneer",
        pioneer,
        raising=False,
    )
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

    out = NPUDeepseekSparseAttention._apply_attention_rescale_pioneer(
        fake,
        q_nope=torch.zeros(T, N, L),
        q_pe=torch.zeros(T, N, R),
        q_cumlens=torch.tensor([T], dtype=torch.int32),
        kv_lens=torch.tensor([T], dtype=torch.int32),
        topk_idx=torch.zeros(T, 1, 2, dtype=torch.int32),
        block_table=torch.zeros(B, 4, dtype=torch.int32),
        kv_cache=(torch.zeros(4, 8, L),),
    )

    assert out is expected
    pioneer.assert_called_once()
    meta.assert_called_once()
    sink_v2.assert_called_once()
    rescale.assert_called_once()


@pytest.mark.unit
def test_apply_attn_absorb_precision_path_delegates_to_rescale_pioneer(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.noncontiguous_kv = True
    fake.param_sink_number = 128
    expected = torch.full((2, 2, 4), 3.0)
    pioneer = MagicMock(return_value=expected)
    fake._apply_attention_rescale_pioneer = pioneer
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                enable_precision_strong_consistency=True,
            )
        ),
    )

    out = NPUDeepseekSparseAttention._apply_attn_absorb(
        fake,
        q_nope=torch.zeros(2, 2, 4),
        q_pe=torch.zeros(2, 2, 2),
        q_cumlens=torch.tensor([2], dtype=torch.int32),
        kv_lens=torch.tensor([2], dtype=torch.int32),
        topk_idx=torch.zeros(2, 1, 2, dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        kv_cache=(torch.zeros(4, 8, 4),),
    )

    assert out is expected
    pioneer.assert_called_once()


# Allow running directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
