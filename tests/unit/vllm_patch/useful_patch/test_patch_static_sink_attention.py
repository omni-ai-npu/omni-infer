# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from omni_npu.vllm_patches.usefull_patch.models.high_throughout.patch_static_sink_attention import (
    StaticSinkAttentionPatch,
    create_static_sink_attention_backendPatch,
)


class _FakeBuilder:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return common_attn_metadata


class _FakeBackend:
    @classmethod
    def get_builder_cls(cls):
        return _FakeBuilder


def _vllm_config(block_size=16, max_model_len=64, max_num_seqs=4):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def _patch_model_extra_config(monkeypatch, *, use_noncontiguous_kv):
    # Import the module object so pytest does not walk omni_npu.model_config.*.
    import omni_npu.model_config.config_loader.loader as loader_mod

    monkeypatch.setattr(
        loader_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                use_noncontiguous_kv=use_noncontiguous_kv
            )
        ),
        raising=False,
    )


def _static_sink_mla_spec(
    monkeypatch,
    *,
    use_noncontiguous_kv,
    use_sparse,
    indexer=None,
    sink_len=None,
    vllm_config=None,
):
    from vllm.v1 import kv_cache_interface
    from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_base import (
        patch_kv_cache_interface as kv_mod,
    )
    from omni_npu.vllm_patches.usefull_patch.models.high_throughout import (
        patch_sink_attention_spec as sink_spec_mod,
    )
    import omni_npu.vllm_patches.usefull_patch.models.high_throughout.patch_static_sink_attention as sink_mod

    # These types are omni patches, not upstream vLLM 0.25.1 attributes.
    monkeypatch.setattr(
        kv_cache_interface,
        "DSAAttentionSpec",
        kv_mod.DSAAttentionSpec,
        raising=False,
    )
    monkeypatch.setattr(
        kv_cache_interface,
        "SinkMLAAttentionSpec",
        sink_spec_mod.SinkMLAAttentionSpec,
        raising=False,
    )

    def _bf16(*_a, **_k):
        return torch.bfloat16

    monkeypatch.setattr(sink_mod, "kv_cache_dtype_str_to_dtype", _bf16)
    _patch_model_extra_config(monkeypatch, use_noncontiguous_kv=use_noncontiguous_kv)
    attn = StaticSinkAttentionPatch.StaticSinkMLAAttention.__new__(
        StaticSinkAttentionPatch.StaticSinkMLAAttention
    )
    attn.kv_cache_dtype = "auto"
    attn.use_sparse = use_sparse
    attn.sliding_window = None
    attn.indexer = indexer
    attn.head_size = 576
    if sink_len is not None:
        attn.sink_len = sink_len
    if vllm_config is None:
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(index_head_dim=128)),
            cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        )
    spec = StaticSinkAttentionPatch.StaticSinkMLAAttention.get_kv_cache_spec(
        attn, vllm_config
    )
    return spec, kv_mod, sink_spec_mod


def _make_builder(monkeypatch, *, use_noncontiguous_kv, sink_len=32):
    _patch_model_extra_config(monkeypatch, use_noncontiguous_kv=use_noncontiguous_kv)
    backend = create_static_sink_attention_backendPatch.create_static_sink_attention_backend(
        _FakeBackend, sink_len=sink_len
    )
    return backend.get_builder_cls()(
        kv_cache_spec=SimpleNamespace(),
        layer_names=["layer0"],
        vllm_config=_vllm_config(),
        device=torch.device("cpu"),
    )


@pytest.mark.unit
def test_static_sink_builder_skips_prefix_for_noncontiguous_kv(monkeypatch):
    builder = _make_builder(monkeypatch, use_noncontiguous_kv=True)
    assert builder.sinks_in_band is False
    assert not hasattr(builder, "block_table_sink_buf")
    builder.reinit_block_table_with_sink()

    meta = SimpleNamespace(
        block_table_tensor=torch.tensor([[5, 6, -1]], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_seq_len=8,
        num_reqs=1,
    )
    out = builder.build(0, meta)
    assert out is meta
    assert meta.seq_lens.cpu().tolist() == [8]
    assert meta.max_seq_len == 8
    assert not hasattr(meta, "_omni_sink_prefixed")


@pytest.mark.unit
def test_static_sink_builder_prefixes_block_table_and_seq_lens(monkeypatch):
    builder = _make_builder(monkeypatch, use_noncontiguous_kv=False, sink_len=32)
    assert builder.sinks_in_band is True
    assert builder.num_sink_slots == 2
    # DeviceMode in this suite redirects torch.tensor(...) to npu; compare as lists.
    assert builder.block_table_sink_buf[0, :2].cpu().tolist() == [1, 2]

    orig_seq = torch.tensor([8, 0], dtype=torch.int32)
    meta = SimpleNamespace(
        block_table_tensor=torch.tensor([[10, 11, -1], [12, -1, -1]], dtype=torch.int32),
        seq_lens=orig_seq.clone(),
        max_seq_len=8,
        num_reqs=2,
    )
    out = builder.build(0, meta)
    assert out is meta
    assert getattr(meta, "_omni_sink_prefixed") is True
    assert meta.max_seq_len == 40
    assert meta.seq_lens.cpu().tolist() == [40, 0]
    assert orig_seq.cpu().tolist() == [8, 0]
    assert meta.block_table_tensor[0, :4].cpu().tolist() == [1, 2, 10, 0]
    assert int(meta.block_table_tensor[1, 3].item()) == 0

    prefixed_seq = meta.seq_lens.clone()
    builder.build(0, meta)
    assert meta.seq_lens.cpu().tolist() == prefixed_seq.cpu().tolist()
    assert meta.max_seq_len == 40


@pytest.mark.unit
def test_static_sink_mla_get_kv_cache_spec_uses_dsa_for_noncontiguous_sparse(
    monkeypatch,
):
    spec, kv_mod, _sink_spec_mod = _static_sink_mla_spec(
        monkeypatch,
        use_noncontiguous_kv=True,
        use_sparse=True,
        indexer=SimpleNamespace(head_dim=128),
    )
    assert isinstance(spec, kv_mod.DSAAttentionSpec)
    assert spec.head_size == 576 + 128


@pytest.mark.unit
def test_static_sink_mla_get_kv_cache_spec_uses_sink_mla_otherwise(monkeypatch):
    spec, _kv_mod, sink_spec_mod = _static_sink_mla_spec(
        monkeypatch,
        use_noncontiguous_kv=False,
        use_sparse=False,
        sink_len=128,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(),
            cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        ),
    )
    assert isinstance(spec, sink_spec_mod.SinkMLAAttentionSpec)
    assert spec.sink_len == 128
