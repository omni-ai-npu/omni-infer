# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for NPU OffloadingConnector registration helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import AttentionSpec

from omni_npu.connector.npu_offloading_connector import (
    NPUOffloadingConnector,
    _full_page_int8_view,
    _layer_kv_cache_spec,
    prepare_kv_caches_for_offloading_registration,
)


def _attn_spec(**overrides):
    kwargs = dict(
        block_size=1,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
    )
    kwargs.update(overrides)
    return AttentionSpec(**kwargs)


def test_layer_kv_cache_spec_direct_and_uniform():
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    attn = _attn_spec()
    group = SimpleNamespace(layer_names=["a"], kv_cache_spec=attn)
    cfg = SimpleNamespace(kv_cache_groups=[group])
    assert _layer_kv_cache_spec("a", cfg) is attn

    uniform = MagicMock(spec=UniformTypeKVCacheSpecs)
    uniform.kv_cache_specs = {"b": attn}
    cfg = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["b"], kv_cache_spec=uniform)]
    )
    assert _layer_kv_cache_spec("b", cfg) is attn


def test_layer_kv_cache_spec_missing():
    cfg = SimpleNamespace(kv_cache_groups=[])
    with pytest.raises(KeyError, match="missing"):
        _layer_kv_cache_spec("missing", cfg)


def test_full_page_int8_view_and_prepare_attention():
    spec = _attn_spec()
    page = spec.page_size_bytes
    num_blocks = 2
    raw = torch.zeros(num_blocks * page, dtype=torch.int8)
    first = raw.view(num_blocks, page)
    view = _full_page_int8_view(first, num_blocks, page)
    assert view.shape == (num_blocks, page)
    assert view.dtype == torch.int8

    cfg = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["L0"], kv_cache_spec=spec)
        ],
    )
    prepared = prepare_kv_caches_for_offloading_registration(
        {"L0": (first, first)}, cfg
    )
    assert prepared["L0"].shape == (num_blocks, page)

    tensor_passthrough = prepare_kv_caches_for_offloading_registration(
        {"L0": raw}, cfg
    )
    assert tensor_passthrough["L0"] is raw


def test_full_page_int8_view_rejects_short_storage():
    spec = _attn_spec()
    first = torch.zeros(4, dtype=torch.int8)
    with pytest.raises(RuntimeError, match="KV storage"):
        _full_page_int8_view(first, num_blocks=8, page_size_bytes=spec.page_size_bytes)


def test_prepare_rejects_bad_sequence_and_unknown_spec():
    spec = _attn_spec()
    cfg = SimpleNamespace(
        num_blocks=1,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["L0"], kv_cache_spec=spec)
        ],
    )
    with pytest.raises(TypeError, match="Unsupported KV cache type"):
        prepare_kv_caches_for_offloading_registration({"L0": 1}, cfg)
    with pytest.raises(ValueError, match="Empty KV cache"):
        prepare_kv_caches_for_offloading_registration({"L0": []}, cfg)
    with pytest.raises(TypeError, match="Expected tensor"):
        prepare_kv_caches_for_offloading_registration({"L0": ("x",)}, cfg)

    other = SimpleNamespace(page_size_bytes=8)
    cfg = SimpleNamespace(
        num_blocks=1,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["L0"], kv_cache_spec=other)
        ],
    )
    with pytest.raises(NotImplementedError, match="Unsupported KV cache spec"):
        prepare_kv_caches_for_offloading_registration(
            {"L0": (torch.zeros(8, dtype=torch.int8),)}, cfg
        )


def test_prepare_mamba_keeps_list():
    from vllm.v1.kv_cache_interface import MambaSpec

    spec = MagicMock(spec=MambaSpec)
    spec.page_size_bytes = 4
    t = torch.zeros(4, dtype=torch.int8)
    cfg = SimpleNamespace(
        num_blocks=1,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["m"], kv_cache_spec=spec)
        ],
    )
    prepared = prepare_kv_caches_for_offloading_registration({"m": (t, t)}, cfg)
    assert prepared["m"] == [t, t]


def test_prepare_warns_on_nonzero_storage_offset(caplog):
    from vllm.v1.kv_cache_interface import MambaSpec

    # Attention path: storage_offset != 0 triggers warning then full-page view.
    attn = _attn_spec()
    page = attn.page_size_bytes
    raw = torch.zeros(2 * page + 8, dtype=torch.int8)
    offset_view = raw.narrow(0, 8, page).view(1, page)
    assert offset_view.storage_offset() != 0
    cfg = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["L0"], kv_cache_spec=attn)
        ],
    )
    # Use base storage with enough bytes via a zero-offset sibling for view helper;
    # warning path only needs first.storage_offset() != 0 before _full_page_int8_view.
    first = torch.as_strided(raw, size=(1, page), stride=(page, 1), storage_offset=8)
    with caplog.at_level("WARNING"):
        prepared = prepare_kv_caches_for_offloading_registration(
            {"L0": (first, first)}, cfg
        )
    assert prepared["L0"].shape == (2, page)
    assert any("storage_offset" in r.message for r in caplog.records)

    mamba = MagicMock(spec=MambaSpec)
    mamba.page_size_bytes = 4
    m_raw = torch.zeros(16, dtype=torch.int8)
    m_first = torch.as_strided(m_raw, size=(4,), stride=(1,), storage_offset=4)
    cfg_m = SimpleNamespace(
        num_blocks=1,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["m"], kv_cache_spec=mamba)
        ],
    )
    with caplog.at_level("WARNING"):
        prepared_m = prepare_kv_caches_for_offloading_registration(
            {"m": (m_first, m_first)}, cfg_m
        )
    assert prepared_m["m"] == [m_first, m_first]
    assert any("Mamba" in r.message or "storage_offset" in r.message for r in caplog.records)


def test_register_kv_caches_requires_config_and_forwards():
    connector = NPUOffloadingConnector.__new__(NPUOffloadingConnector)
    connector._kv_cache_config = None
    with pytest.raises(RuntimeError, match="_kv_cache_config"):
        connector.register_kv_caches({})

    connector._kv_cache_config = SimpleNamespace(num_blocks=1, kv_cache_groups=[])
    with patch(
        "omni_npu.connector.npu_offloading_connector.OffloadingConnector.register_kv_caches"
    ) as super_reg, patch(
        "omni_npu.connector.npu_offloading_connector.prepare_kv_caches_for_offloading_registration",
        return_value={"ok": torch.zeros(1)},
    ):
        connector.register_kv_caches({"L": torch.zeros(1)})
        super_reg.assert_called_once()
