# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# Disable all omni-* plugins (e.g. omni-cache) in UT: these out-of-tree plugins
# register hooks via entry points under "omni_npu.*" groups. When they are present
# in the test environment they can mutate omni-npu behavior or mock return
# values in unexpected ways, so UT for omni-npu runs without them enabled.
import importlib.metadata as _metadata
import os
os.environ.setdefault("VLLM_DISABLE_PYNCCL", "1")

_orig_entry_points = _metadata.entry_points


def _filtered_entry_points(*args, **kwargs):
    group = kwargs.get("group") or (args[0] if args else None)
    if isinstance(group, str) and group.startswith("omni_npu."):
        return []
    return _orig_entry_points(*args, **kwargs)


_metadata.entry_points = _filtered_entry_points

import pytest


@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture(scope="session", autouse=True)
def _register_npu_mla_prefill_backend():
    """Re-point the default FlashAttention MLA prefill backend at omni's NPU impl.

    vLLM 0.25.1's own ``mla/prefill/flash_attn.py`` imports
    ``compile_flash_attn_varlen_func_from_specs`` from ``fa_utils``, a symbol that
    no longer exists there, so it cannot be imported on NPU.  At runtime
    ``omni/platform.py`` (pre_register) already re-registers FLASH_ATTN to
    ``NPUMLAPrefillBackend`` to avoid this, but that hook only runs through
    EngineArgs init, which UT bypasses.  Register it here for the whole session so
    MLA layer/attention tests select omni's backend instead of vLLM's broken one.
    """
    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
        register_mla_prefill_backend,
    )
    register_mla_prefill_backend(
        MLAPrefillBackendEnum.FLASH_ATTN,
        "omni_npu.attention.backends.mla.NPUMLAPrefillBackend",
    )
