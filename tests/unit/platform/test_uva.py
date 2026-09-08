# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
import importlib
import sys
import types
from pathlib import Path

from unit.vllm_patch.patches.patch_test_utils import make_fake_torch

PACKAGE_ROOT = next(
    p / "omni"
    for p in Path(__file__).resolve().parents
    if (p / "omni" / "platform.py").is_file()
)


def _ensure_omni_npu_package():
    """Other patch UTs may have stubbed omni_npu as a non-package module."""
    existing = sys.modules.get("omni_npu")
    if getattr(existing, "__path__", None):
        return
    omni_npu = types.ModuleType("omni_npu")
    omni_npu.__file__ = str(PACKAGE_ROOT / "__init__.py")
    omni_npu.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["omni_npu"] = omni_npu


def _silent_logger(_name):
    def _noop(*_args, **_kwargs):
        return None

    return types.SimpleNamespace(info=_noop, warning=_noop)


def _install_fake_platform_deps(monkeypatch, torch_module):
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.DEFAULT_LOGGING_CONFIG = {}
    logger_mod._DATE_FORMAT = "%m-%d %H:%M:%S"
    logger_mod.init_logger = _silent_logger

    envs_mod = types.ModuleType("vllm.envs")
    envs_mod.VLLM_CONFIGURE_LOGGING = False
    envs_mod.VLLM_LOGGING_CONFIG_PATH = None
    envs_mod.VLLM_LOGGING_LEVEL = "INFO"
    envs_mod.VLLM_LOGGING_PREFIX = ""
    envs_mod.VLLM_LOGGING_STREAM = "ext://sys.stdout"

    interface_mod = types.ModuleType("vllm.platforms.interface")

    class Platform:
        pass

    class PlatformEnum:
        HUAWEI_NPU = "huawei_npu"
        CUDA = "cuda"
        ROCM = "rocm"
        OOT = "oot"

    interface_mod.Platform = Platform
    interface_mod.PlatformEnum = PlatformEnum

    registry_mod = types.ModuleType("vllm.v1.attention.backends.registry")
    registry_mod.AttentionBackendEnum = type("AttentionBackendEnum", (), {})

    vllm.envs = envs_mod
    vllm.logger = logger_mod
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_mod)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)

    for name in (
        "vllm.platforms",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    monkeypatch.setitem(sys.modules, "vllm.platforms.interface", interface_mod)
    monkeypatch.setitem(
        sys.modules, "vllm.v1.attention.backends.registry", registry_mod
    )


def _import_platform(monkeypatch, npu_available=True):
    _ensure_omni_npu_package()
    _install_fake_platform_deps(
        monkeypatch, make_fake_torch(npu_available=npu_available)
    )
    sys.modules.pop("omni_npu.platform", None)
    sys.modules.pop("omni_npu.logger", None)
    return importlib.import_module("omni_npu.platform")


def test_npu_platform_uva_available_requires_alloc_conf(monkeypatch):
    platform_mod = _import_platform(monkeypatch)
    monkeypatch.delenv("PYTORCH_NPU_ALLOC_CONF", raising=False)

    assert platform_mod.NPUPlatform.is_uva_available() is False


def test_npu_platform_uva_available_rejects_expandable_segments(monkeypatch):
    platform_mod = _import_platform(monkeypatch)
    monkeypatch.setenv(
        "PYTORCH_NPU_ALLOC_CONF",
        "pinned_mem_register:True,pin_memory_expandable_segments:True",
    )

    assert platform_mod.NPUPlatform.is_uva_available() is False


def test_npu_platform_uva_available_requires_npu_runtime(monkeypatch):
    platform_mod = _import_platform(monkeypatch, npu_available=False)
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "pinned_mem_register:True")

    assert platform_mod.NPUPlatform.is_uva_available() is False


def test_npu_platform_uva_available_requires_npu_uva_extension(monkeypatch):
    platform_mod = _import_platform(monkeypatch)
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "pinned_mem_register:True")
    real_import = builtins.__import__

    def fail_import_npu_uva(
        name, global_ns=None, local_ns=None, fromlist=(), level=0
    ):
        if name == "omni_npu.allocator" and "npu_uva" in fromlist:
            raise ImportError("npu_uva is unavailable")
        return real_import(name, global_ns, local_ns, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_import_npu_uva)

    assert platform_mod.NPUPlatform.is_uva_available() is False


def test_npu_platform_uva_available_accepts_valid_runtime(monkeypatch):
    platform_mod = _import_platform(monkeypatch)
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "pinned_mem_register:True")
    import omni_npu.allocator as allocator

    monkeypatch.setattr(allocator, "npu_uva", object(), raising=False)

    assert platform_mod.NPUPlatform.is_uva_available() is True
