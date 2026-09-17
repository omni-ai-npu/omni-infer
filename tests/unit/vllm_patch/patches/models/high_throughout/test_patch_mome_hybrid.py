# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MomeAttention must hand the caller's inplace choice to the conv kernel."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from omni_npu.vllm_patches.patches.models.high_throughout import (
    patch_mome_hybrid as patch_mod,
)

MomeAttention = patch_mod.NPUMoMEPatch.MomeAttention


def _attention():
    attention = MomeAttention.__new__(MomeAttention)
    torch.nn.Module.__init__(attention)
    attention.prefix = "model.layers.0.self_attn.mome"
    attention.kv_cache = tuple(torch.zeros(4, 8) for _ in range(3))
    for name in ("qa_conv", "compresskv_conv", "o_conv"):
        conv = MagicMock()
        conv.forward.return_value = torch.ones(2, 8)
        object.__setattr__(attention, name, conv)
    return attention


def _context():
    metadata = SimpleNamespace(prefill=object(), decode=object())
    return SimpleNamespace(attn_metadata={"model.layers.0.self_attn.mome": metadata}), metadata


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("is_prefill", [True, False])
def test_forward_passes_inplace_to_conv(inplace, is_prefill):
    # MomeAttentionMixin._maybe_mome_kv drops the return value when it asks for
    # inplace, so swallowing the flag silently skips the KV MoME.
    attention = _attention()
    context, metadata = _context()
    x = torch.zeros(2, 8)
    with patch.object(patch_mod, "get_forward_context", return_value=context):
        result = attention.forward(x, 1, is_prefill=is_prefill, inplace=inplace)
    conv = attention.compresskv_conv
    assert result is conv.forward.return_value
    conv.forward.assert_called_once_with(
        x=x,
        conv_states=attention.kv_cache[1],
        mome_metadata=metadata.prefill if is_prefill else metadata.decode,
        inplace=inplace,
    )
    attention.qa_conv.forward.assert_not_called()
    attention.o_conv.forward.assert_not_called()


def test_forward_defaults_to_out_of_place():
    attention = _attention()
    context, metadata = _context()
    x = torch.zeros(2, 8)
    with patch.object(patch_mod, "get_forward_context", return_value=context):
        attention.forward(x, 0)
    attention.qa_conv.forward.assert_called_once_with(
        x=x,
        conv_states=attention.kv_cache[0],
        mome_metadata=metadata.decode,
        inplace=False,
    )
