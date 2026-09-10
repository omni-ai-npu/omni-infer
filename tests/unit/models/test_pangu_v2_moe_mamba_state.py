# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from omni_npu.v1.models.pangu.pangu_v2_moe import OpenPanguV2ForCausalLM


pytestmark = pytest.mark.unit


def _vllm_config(
    *,
    dtype=torch.float16,
    mamba_cache_dtype="auto",
    q_lora_rank=8,
    kv_lora_rank=16,
    num_heads=4,
    v_head_dim=32,
    router_sliding_window=5,
    num_spec=None,
    include_router_sliding_window=True,
):
    hf_config = SimpleNamespace(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        num_attention_heads=num_heads,
        v_head_dim=v_head_dim,
    )
    if include_router_sliding_window:
        hf_config.router_sliding_window = router_sliding_window
    spec = (
        SimpleNamespace(num_speculative_tokens=num_spec)
        if num_spec is not None
        else None
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=dtype, hf_config=hf_config),
        cache_config=SimpleNamespace(mamba_cache_dtype=mamba_cache_dtype),
        speculative_config=spec,
    )


def test_mamba_state_dtype_follows_model_dtype_when_cache_is_auto():
    cfg = _vllm_config(dtype=torch.float16, mamba_cache_dtype="auto")

    result = OpenPanguV2ForCausalLM.get_mamba_state_dtype_from_config(cfg)

    assert result == (torch.float16, torch.float16, torch.float16)


def test_mamba_state_dtype_uses_explicit_cache_dtype():
    cfg = _vllm_config(dtype=torch.float16, mamba_cache_dtype="bfloat16")

    result = OpenPanguV2ForCausalLM.get_mamba_state_dtype_from_config(cfg)

    assert result == (torch.bfloat16, torch.bfloat16, torch.bfloat16)


def test_mamba_state_shape_uses_kernel_history_and_spec_tokens():
    cfg = _vllm_config(router_sliding_window=5, num_spec=3)

    result = OpenPanguV2ForCausalLM.get_mamba_state_shape_from_config(cfg)

    assert result == ((7, 8), (7, 16), (7, 128))


def test_mamba_state_shape_supports_zero_history():
    cfg = _vllm_config(router_sliding_window=1, num_spec=None)

    result = OpenPanguV2ForCausalLM.get_mamba_state_shape_from_config(cfg)

    assert result == ((0, 8), (0, 16), (0, 128))


def test_mamba_state_shape_defaults_kernel_size_when_missing():
    cfg = _vllm_config(include_router_sliding_window=False, num_spec=None)

    result = OpenPanguV2ForCausalLM.get_mamba_state_shape_from_config(cfg)

    assert result == ((0, 8), (0, 16), (0, 128))
