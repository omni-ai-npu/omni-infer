# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import sys
import types

import pytest

from unit.vllm_patch.patches.patch_test_utils import make_fake_torch


def _orig_is_uva_available():
    return "orig-uva"


def _orig_get_accelerator_view(_tensor):
    return "orig-view"


def _install_fake_patch_deps(monkeypatch, *, device_type="npu", uva_available=True):
    torch = make_fake_torch(npu_available=True, current_device=3)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def _is_uva_available():
        return uva_available

    current_platform = types.SimpleNamespace(
        device_type=device_type,
        is_uva_available=_is_uva_available,
    )

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = current_platform
    vllm.platforms = platforms

    buffer_utils = types.ModuleType("vllm.v1.worker.gpu.buffer_utils")
    buffer_utils.is_uva_available = _orig_is_uva_available
    buffer_utils.get_accelerator_view_from_cpu_tensor = _orig_get_accelerator_view

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
    module_name = "omni_npu.vllm_patches.patches.common.patch_uva"
    sys.modules.pop(module_name, None)

    def _import_patch():
        return importlib.import_module(module_name)

    yield _import_patch
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

    def _is_pinned():
        return True

    tensor = types.SimpleNamespace(
        device=types.SimpleNamespace(type="cpu"),
        is_pinned=_is_pinned,
    )

    assert mod.get_accelerator_view_from_cpu_tensor(tensor) == "npu-view"
    assert calls == [(tensor, 3)]


def test_patch_uva_rejects_unpinned_cpu_tensor(monkeypatch, patch_uva_module):
    _install_fake_patch_deps(monkeypatch)
    mod = patch_uva_module()

    def _is_unpinned():
        return False

    tensor = types.SimpleNamespace(
        device=types.SimpleNamespace(type="cpu"),
        is_pinned=_is_unpinned,
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
    assert (
        "get_cuda_view_from_cpu_tensor"
        not in mod.BufferUtilsPatch._attr_names_to_apply
    )
