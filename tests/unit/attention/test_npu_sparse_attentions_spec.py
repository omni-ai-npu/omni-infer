# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV cache spec preconditions of the MoME attention layer.

MoME reads mamba_block_size straight from the cache config instead of deriving
it a second time, so an unresolved value must fail here rather than travel into
the returned spec. The check has to survive `python -O`, so it is a raise.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omni_npu.layers.attention import npu_sparse_attentions as sparse_mod


pytestmark = pytest.mark.unit


def test_get_kv_cache_spec_rejects_unresolved_mamba_block_size():
    """vLLM resolves mamba_block_size; an unset one is a config regression."""
    layer = sparse_mod.MomeAttention.__new__(sparse_mod.MomeAttention)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_block_size=None)
    )

    with patch.dict(
        "vllm.v1.kv_cache_interface.__dict__", {"MomeSpec": object}
    ):
        with pytest.raises(ValueError, match="mamba_block_size"):
            layer.get_kv_cache_spec(vllm_config)
