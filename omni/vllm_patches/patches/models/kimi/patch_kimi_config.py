# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Patch: Inject KimiK25Config into vLLM's config system.

Lazily imports config classes from the model modules and injects them into:
  - vllm.transformers_utils.configs.kimi_k25   (new submodule)
  - vllm.transformers_utils.configs             (package attributes)
  - vllm.transformers_utils.config._CONFIG_REGISTRY  (kimi_k25 key)

Applied automatically when kimi_k25 model_type is detected.
"""

from __future__ import annotations

import logging
import sys
import types

logger = logging.getLogger(__name__)

from omni_npu.vllm_patches.patches.models.kimi.kimi_k25 import KimiK25Config
from omni_npu.vllm_patches.patches.models.kimi.kimi_k25_vit import KimiK25VisionConfig

# 1. Inject submodule vllm.transformers_utils.configs.kimi_k25
submod_name = "vllm.transformers_utils.configs.kimi_k25"
if submod_name not in sys.modules:
    submod = types.ModuleType(submod_name)
    sys.modules[submod_name] = submod
else:
    submod = sys.modules[submod_name]

submod.KimiK25Config = KimiK25Config
submod.KimiK25VisionConfig = KimiK25VisionConfig

# 2. Inject into vllm.transformers_utils.configs package attributes
import vllm.transformers_utils.configs as _configs_pkg

_configs_pkg.KimiK25Config = KimiK25Config
_configs_pkg.KimiK25VisionConfig = KimiK25VisionConfig
_configs_pkg.kimi_k25 = sys.modules[submod_name]

if hasattr(_configs_pkg, "__all__"):
    for name in ("KimiK25Config", "KimiK25VisionConfig"):
        if name not in _configs_pkg.__all__:
            _configs_pkg.__all__.append(name)

# 3. Register in vllm.transformers_utils.config._CONFIG_REGISTRY
try:
    import vllm.transformers_utils.config as _config_mod
    registry = getattr(_config_mod, "_CONFIG_REGISTRY", None)
    if registry is not None and "kimi_k25" not in registry:
        registry["kimi_k25"] = KimiK25Config
except Exception as e:
    logger.warning(f"Failed to register kimi_k25 in _CONFIG_REGISTRY: {e}")

logger.info(
    "patch applied: KimiConfigPatch => "
    "vllm.transformers_utils.configs.[KimiK25Config, KimiK25VisionConfig]"
)
