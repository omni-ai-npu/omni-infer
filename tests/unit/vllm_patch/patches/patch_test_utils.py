"""Shared helpers for VLLM patch tests."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Walk parents until omni/vllm_patches is found."""
    current = (start or Path(__file__)).resolve()
    search = current.parents if current.is_file() else (current, *current.parents)
    for parent in search:
        if (parent / "omni" / "vllm_patches").is_dir():
            return parent
    raise RuntimeError(f"cannot find omniinfer repo root from {current}")


def install_named_module(name, **attrs):
    """Create a stub module, register it in sys.modules, and return it."""
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def silent_logger(_name=None):
    """No-op logger used when tests stub vllm.logger.init_logger."""

    def _noop(*_args, **_kwargs):
        return None

    return types.SimpleNamespace(warning=_noop, info=_noop, debug=_noop)


class PatchCallRecorder:
    """Collects patch registrations and hook calls for dump/health UTs."""

    def __init__(self):
        self.calls = []
        self.registered = []
        self.stalled = []
        self.stalled_args = []

    def register_patch(self, name, target):
        def decorator(cls):
            cls._target = target
            self.registered.append((name, target, tuple(cls._attr_names_to_apply)))
            return cls

        return decorator


class StubVLLMPatch:
    """Stand-in for omni_npu.vllm_patches.core.VLLMPatch."""

    _attr_names_to_apply: list[str] = []

    @classmethod
    def apply(cls):
        target = cls._target
        if not hasattr(target, "_omni_npu_applied_patches"):
            target._omni_npu_applied_patches = {}
        for name in cls._attr_names_to_apply:
            if name in target._omni_npu_applied_patches:
                raise ValueError(
                    f"{target.__name__}.{name} already patched by "
                    f"{target._omni_npu_applied_patches[name]}"
                )
            target._omni_npu_applied_patches[name] = cls.__name__
            setattr(target, name, cls.__dict__[name])


def make_fake_torch(*, npu_available=True, current_device=0):
    """Minimal torch stub used by UVA unit tests."""
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

    def device(value=None):
        return value

    def no_grad():
        return object()

    def zeros(*args, **kwargs):
        return object()

    torch.npu = _Npu()
    torch.device = device
    torch.dtype = type("dtype", (), {})
    torch.Tensor = type("Tensor", (), {})
    torch.types = types.SimpleNamespace(Device=object)
    torch.no_grad = no_grad
    torch.zeros = zeros
    return torch


def run_standalone_tests(namespace):
    """Run test_* callables when a patch UT is executed as a script."""
    failed = 0
    for name, fn in sorted(namespace.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("=" * 60)
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    if failed:
        raise RuntimeError(f"{failed} tests failed")


@contextmanager
def applied_patches(classes):
    """Apply patch classes and restore their targets when the test finishes."""
    saved = []
    owners = {}
    for patch_cls in classes:
        target = patch_cls._target
        if target not in owners:
            owners[target] = dict(
                getattr(target, "_omni_npu_applied_patches", {})
            )
        for name in patch_cls._attr_names_to_apply:
            saved.append((target, name, getattr(target, name)))

    for target in owners:
        target._omni_npu_applied_patches = {}

    try:
        for patch_cls in classes:
            patch_cls.apply()
        yield classes
    finally:
        for target, name, value in saved:
            setattr(target, name, value)
        for target, snapshot in owners.items():
            target._omni_npu_applied_patches = snapshot
