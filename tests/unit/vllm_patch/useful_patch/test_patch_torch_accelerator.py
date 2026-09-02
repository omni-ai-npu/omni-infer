# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Tests for patch_torch_accelerator.

The patch points torch.accelerator's memory APIs at torch.npu. Every test that
calls apply() restores torch.accelerator afterwards, because the patch mutates
a real module that the rest of the suite shares.
"""

import importlib
import sys

import pytest
import torch


@pytest.fixture
def PATCH():
    """
    Import the patch module against the real omni_npu packages.

    Sibling tests in this directory push stubs into sys.modules: a
    VLLMPatch with no apply(), and plain ModuleType objects for omni_npu
    and omni_npu.vllm_patches. Those stubs have no __path__, so they are
    not packages and nothing beneath them can be imported. They restore
    sys.modules on teardown, but this file sorts last, so a module-level
    import here would already have bound the stubs.

    Import inside a temporarily cleared omni_npu namespace, then put the
    original entries back: tests that run later hold references to the
    module objects they imported, and leaving fresh copies in sys.modules
    would silently detach their mocks.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "omni_npu" or name.startswith("omni_npu.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        module = importlib.import_module(
            "omni_npu.vllm_patches.usefull_patch.common.patch_torch_accelerator"
        )
        yield module.TorchAcceleratorMemoryPatch
    finally:
        for name in [
            n for n in sys.modules
            if n == "omni_npu" or n.startswith("omni_npu.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


# What each patched name must end up pointing at.
EXPECTED_TARGETS = {
    "empty_cache": torch.npu.empty_cache,
    "memory_stats": torch.npu.memory_stats,
    "memory_reserved": torch.npu.memory_reserved,
    "memory_allocated": torch.npu.memory_allocated,
    "reset_peak_memory_stats": torch.npu.reset_peak_memory_stats,
    "get_memory_info": torch.npu.mem_get_info,
}


@pytest.fixture
def restore_accelerator():
    """Undo everything apply() does to the torch.accelerator module."""
    saved = {
        name: getattr(torch.accelerator, name, None)
        for name in EXPECTED_TARGETS
    }
    saved_bookkeeping = getattr(
        torch.accelerator, "_omni_npu_applied_patches", None
    )
    # apply() refuses to patch the same attribute twice, so start from empty
    # bookkeeping rather than whatever an earlier apply() left behind.
    torch.accelerator._omni_npu_applied_patches = {}

    yield

    for name, value in saved.items():
        if value is None:
            if hasattr(torch.accelerator, name):
                delattr(torch.accelerator, name)
        else:
            setattr(torch.accelerator, name, value)
    if saved_bookkeeping is None:
        del torch.accelerator._omni_npu_applied_patches
    else:
        torch.accelerator._omni_npu_applied_patches = saved_bookkeeping


def test_patch_targets_the_accelerator_module(PATCH):
    assert PATCH._target is torch.accelerator


def test_patch_is_registered_with_the_patch_manager(PATCH):
    """
    Importing the module must register the patch, since apply_patches() drives
    everything off PatchManager.registered_patches rather than the class.
    """
    from omni_npu.vllm_patches.patch_manager import PatchManager

    assert PatchManager.registered_patches["TorchAcceleratorMemory"] is PATCH


def test_every_declared_attr_exists_on_the_class(PATCH):
    """apply() raises if a name in _attr_names_to_apply is missing."""
    for name in PATCH._attr_names_to_apply:
        assert name in PATCH.__dict__, f"missing {name!r} on {PATCH.__name__}"


def test_declares_all_six_memory_apis(PATCH):
    assert set(PATCH._attr_names_to_apply) == {
        "empty_cache",
        "memory_stats",
        "memory_reserved",
        "memory_allocated",
        "reset_peak_memory_stats",
        "get_memory_info",
    }


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_TARGETS.items()))
def test_apply_redirects_each_api_to_torch_npu(name, expected, PATCH, restore_accelerator):
    PATCH.apply()
    assert getattr(torch.accelerator, name) is expected


def test_patched_attrs_are_plain_functions_not_staticmethods(PATCH):
    """
    The target is a module, so attributes are read back and called directly.
    A staticmethod object is only callable itself from Python 3.10 on.
    """
    for name in PATCH._attr_names_to_apply:
        assert not isinstance(PATCH.__dict__[name], staticmethod), (
            f"{name} must be a plain function, not staticmethod"
        )


def test_apply_does_not_rebind_as_methods(PATCH):
    """
    apply() rebinds MethodType attributes onto the target. These are plain
    functions, so they must pass through untouched -- no wrapper needed.
    """
    from types import MethodType

    for name in PATCH._attr_names_to_apply:
        assert not isinstance(PATCH.__dict__[name], MethodType), (
            f"{name} would be rebound by apply()"
        )


def test_empty_cache_is_callable_with_no_arguments(PATCH, restore_accelerator):
    """torch.accelerator.empty_cache() is called without arguments upstream."""
    PATCH.apply()

    torch.accelerator.empty_cache()  # must not raise


def test_get_memory_info_returns_free_and_total(PATCH, restore_accelerator):
    """The API that used to raise 'Allocator for npu is not a DeviceAllocator'."""
    PATCH.apply()

    free, total = torch.accelerator.get_memory_info(torch.device("npu:0"))

    assert 0 < free <= total


def test_apply_is_idempotent_guarded(PATCH, restore_accelerator):
    """
    apply() records what it patched and refuses a second pass, so a double
    plugin load surfaces as an error instead of silently re-wrapping.
    """
    PATCH.apply()

    with pytest.raises(ValueError, match="already patched"):
        PATCH.apply()


def test_memory_snapshot_measures_through_the_patch(PATCH, restore_accelerator, monkeypatch):
    """
    vLLM's MemorySnapshot reads torch.accelerator.*; after the patch it must
    reach torch.npu and compute non_torch_memory from those numbers.
    """
    from vllm.utils.mem_utils import MemorySnapshot

    PATCH.apply()
    monkeypatch.setattr(
        torch.accelerator,
        "memory_stats",
        lambda device: {"allocated_bytes.all.peak": 100},
    )
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda device: (900, 1000)
    )
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda device: 50)

    snapshot = MemorySnapshot(device=torch.device("npu:0"))

    assert snapshot.torch_peak == 100
    assert (snapshot.free_memory, snapshot.total_memory) == (900, 1000)
    assert snapshot.torch_memory == 50
    assert snapshot.non_torch_memory == 50  # (1000 - 900) - 50
