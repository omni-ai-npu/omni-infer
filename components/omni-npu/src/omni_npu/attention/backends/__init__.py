# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

import sys
from omni_npu.attention.backends.attention import (
    NPUAttentionBackendImpl,
    NPUMetadata,
    NPUAttentionBackend,
    NPUAttentionMetadataBuilder,
)
from omni_npu.attention.backends.mome import NPUPanguMomeBackend
from omni_npu.attention.backends.dsa import NPUDSABackend
from omni_npu.attention.backends.mla import NPUMLABackend
from omni_npu.attention.backends.attention import NPUAttentionBackend

# NOTE: After all base backends are imported, apply plugin overrides.
# This loads entry points from omni.attention_backends and replaces
# base classes with their plugin counterparts in NPU_ATTENTION_BACKEND.
# It also returns the override map so we can rebind module-level names.
from omni_npu.attention.backends.utils import apply_plugin_overrides
_plugin_overrides, _base_paths = apply_plugin_overrides()

# NOTE: Rebind module-level names to plugin classes when overridden.
# This ensures that code doing `from omni_npu.attention.backends import NPUDSABackend`
# gets the extended class when a plugin is active.
# We also write back to the original submodule (e.g. mome.py) so that
# `from omni_npu.attention.backends.mome import NPUPanguMomeBackend`
# also resolves to the plugin class.
for _bk, _cls in _plugin_overrides.items():
    _base_path = _base_paths[_bk]  # e.g. "omni_npu.attention.backends.mome.NPUPanguMomeBackend"
    _mod_path, _cls_name = _base_path.rsplit(".", 1)
    # Rebind in __init__.py (this module)
    globals()[_cls_name] = _cls
    # Rebind in the original submodule so direct imports also get the plugin class
    if _mod_path in sys.modules:
        setattr(sys.modules[_mod_path], _cls_name, _cls)


__all__ = [
    "NPUAttentionBackendImpl",
    "NPUMetadata",
    "NPUAttentionBackend",
    "NPUAttentionMetadataBuilder",
    "NPUPanguMomeBackend",
    "NPUDSABackend",
]
