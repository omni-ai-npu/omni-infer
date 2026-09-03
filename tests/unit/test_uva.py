# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
import importlib
import sys
import types
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "omni"
if "omni_npu" not in sys.modules:
    omni_npu = types.ModuleType("omni_npu")
    omni_npu.__file__ = str(PACKAGE_ROOT / "__init__.py")
    omni_npu.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["omni_npu"] = omni_npu


def _fake_torch(npu_available=True, current_device=0):
    torch = types.ModuleType("torch")

    class _Npu:
        @staticmethod
        def is_available():
            return npu_available

        @staticmethod
        def current_device():
            return current_device

        @staticmethod
        def set_device(device):
            return None

        @staticmethod
        def manual_seed_all(seed):
            return None

        @staticmethod
        def get_device_name(device_id=0):
            return f"npu:{device_id}"

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def reset_peak_memory_stats(device=None):
            return None

        @staticmethod
        def max_memory_allocated(device=None):
            return 0

        @staticmethod
        def mem_get_info():
            return (0, 0)

        @staticmethod
        def get_device_properties(device_id=0):
            return types.SimpleNamespace(multi_processor_count=1)

    torch.npu = _Npu()
    torch.device = lambda value=None: value
    torch.dtype = type("dtype", (), {})
    torch.Tensor = type("Tensor", (), {})
    torch.types = types.SimpleNamespace(Device=object)
    torch.no_grad = lambda: object()
    torch.zeros = lambda *args, **kwargs: object()
    return torch


def _install_fake_platform_deps(monkeypatch, torch_module):
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.DEFAULT_LOGGING_CONFIG = {}
    logger_mod._DATE_FORMAT = "%m-%d %H:%M:%S"
    logger_mod.init_logger = lambda name: types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )

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
    _install_fake_platform_deps(monkeypatch, _fake_torch(npu_available=npu_available))
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

    def fail_import_npu_uva(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "omni_npu.allocator" and "npu_uva" in fromlist:
            raise ImportError("npu_uva is unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_import_npu_uva)

    assert platform_mod.NPUPlatform.is_uva_available() is False


def test_npu_platform_uva_available_accepts_valid_runtime(monkeypatch):
    platform_mod = _import_platform(monkeypatch)
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "pinned_mem_register:True")
    import omni_npu.allocator as allocator

    monkeypatch.setattr(allocator, "npu_uva", object(), raising=False)

    assert platform_mod.NPUPlatform.is_uva_available() is True


def _install_fake_patch_deps(monkeypatch, *, device_type="npu", uva_available=True):
    torch = _fake_torch(npu_available=True, current_device=3)
    monkeypatch.setitem(sys.modules, "torch", torch)

    current_platform = types.SimpleNamespace(
        device_type=device_type,
        is_uva_available=lambda: uva_available,
    )

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = current_platform
    vllm.platforms = platforms

    buffer_utils = types.ModuleType("vllm.v1.worker.gpu.buffer_utils")
    buffer_utils.is_uva_available = lambda: "orig-uva"
    buffer_utils.get_accelerator_view_from_cpu_tensor = lambda tensor: "orig-view"

    for name in (
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.gpu.buffer_utils", buffer_utils
    )
    return torch, current_platform, buffer_utils


@pytest.fixture
def patch_uva_module(monkeypatch):
    module_name = "omni_npu.vllm_patches.usefull_patch.common.patch_uva"
    sys.modules.pop(module_name, None)
    yield lambda: importlib.import_module(module_name)
    sys.modules.pop(module_name, None)


def test_patch_uva_delegates_availability_to_npu_platform(
    monkeypatch, patch_uva_module
):
    _install_fake_patch_deps(monkeypatch, uva_available=True)
    mod = patch_uva_module()

    assert mod.is_uva_available() is True


def test_patch_uva_preserves_original_availability_for_non_npu(
    monkeypatch, patch_uva_module
):
    _install_fake_patch_deps(monkeypatch, device_type="cuda")
    mod = patch_uva_module()

    assert mod.is_uva_available() is True


def test_patch_uva_creates_npu_view_from_pinned_cpu_tensor(
    monkeypatch, patch_uva_module
):
    _install_fake_patch_deps(monkeypatch)
    import omni_npu.allocator as allocator

    calls = []

    class FakeNpuUva:
        @staticmethod
        def get_npu_view_from_cpu_tensor(cpu_tensor, device_index):
            calls.append((cpu_tensor, device_index))
            return "npu-view"

    monkeypatch.setattr(allocator, "npu_uva", FakeNpuUva, raising=False)
    mod = patch_uva_module()

    tensor = types.SimpleNamespace(
        device=types.SimpleNamespace(type="cpu"),
        is_pinned=lambda: True,
    )

    assert mod.get_accelerator_view_from_cpu_tensor(tensor) == "npu-view"
    assert calls == [(tensor, 3)]


def test_patch_uva_rejects_unpinned_cpu_tensor(monkeypatch, patch_uva_module):
    _install_fake_patch_deps(monkeypatch)
    mod = patch_uva_module()
    tensor = types.SimpleNamespace(
        device=types.SimpleNamespace(type="cpu"),
        is_pinned=lambda: False,
    )

    with pytest.raises(RuntimeError, match="must be pinned"):
        mod.get_accelerator_view_from_cpu_tensor(tensor)


def test_patch_uva_patch_class_targets_v0251_buffer_utils(
    monkeypatch, patch_uva_module
):
    _, _, buffer_utils = _install_fake_patch_deps(monkeypatch)
    mod = patch_uva_module()

    mod.BufferUtilsPatch.apply()

    assert buffer_utils.is_uva_available is mod.is_uva_available
    assert (
        buffer_utils.get_accelerator_view_from_cpu_tensor
        is mod.get_accelerator_view_from_cpu_tensor
    )
    assert "get_cuda_view_from_cpu_tensor" not in mod.BufferUtilsPatch._attr_names_to_apply