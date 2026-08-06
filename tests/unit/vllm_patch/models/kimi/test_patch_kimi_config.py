# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for kimi patch_kimi_config: config injection into vLLM."""

import importlib
import sys


def test_injects_config_into_vllm_configs():
    """After import, KimiK25Config should be on vllm.transformers_utils.configs."""
    import omni.vllm_patches.patches.models.kimi.patch_kimi_config  # noqa: F401

    import vllm.transformers_utils.configs as configs_pkg
    from omni.vllm_patches.patches.models.kimi.kimi_k25 import KimiK25Config
    from omni.vllm_patches.patches.models.kimi.kimi_k25_vit import KimiK25VisionConfig

    assert configs_pkg.KimiK25Config is KimiK25Config
    assert configs_pkg.KimiK25VisionConfig is KimiK25VisionConfig


def test_creates_kimi_k25_submodule():
    """After import, vllm.transformers_utils.configs.kimi_k25 should exist in sys.modules."""
    import omni.vllm_patches.patches.models.kimi.patch_kimi_config  # noqa: F401

    submod_name = "vllm.transformers_utils.configs.kimi_k25"
    assert submod_name in sys.modules

    from omni.vllm_patches.patches.models.kimi.kimi_k25 import KimiK25Config

    submod = sys.modules[submod_name]
    assert submod.KimiK25Config is KimiK25Config


def test_registers_in_config_registry(monkeypatch):
    """After import, kimi_k25 should be in _CONFIG_REGISTRY."""
    import vllm.transformers_utils.config as config_mod
    import omni.vllm_patches.patches.models.kimi.patch_kimi_config as patch_mod

    from omni.vllm_patches.patches.models.kimi.kimi_k25 import KimiK25Config

    # Isolate this test from any suite-level monkeypatching of vLLM modules.
    monkeypatch.setattr(config_mod, "_CONFIG_REGISTRY", {}, raising=False)
    importlib.reload(patch_mod)

    registry = getattr(config_mod, "_CONFIG_REGISTRY")
    assert "kimi_k25" in registry
    assert registry["kimi_k25"] is KimiK25Config


def test_reimport_does_not_raise():
    """Reloading the module should not raise (all guards use 'not in' checks)."""
    import omni.vllm_patches.patches.models.kimi.patch_kimi_config as mod

    importlib.reload(mod)
