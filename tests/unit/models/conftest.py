# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared fixtures for the models contract tests.

The registration-chain tests in test_2bv2_component_contracts.py must apply the
real patches before importing the component openpangu (injected symbols are a
prerequisite of its top-level import). This fixture loads patch files from the
patch source dirs, mirroring vllm_patch/patches/common/conftest.py.

Patches live in different groups since the patch-tree reorg:
patch_static_sink_attention in patches/models/high_throughout. The loader
takes "group" to pick the dir: "common" -> patches/common, anything else ->
patches/models/<group>.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# patch_test_utils.py lives under tests/unit/vllm_patch/patches/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm_patch" / "patches"))

from patch_test_utils import repo_root  # noqa: E402


@pytest.fixture
def load_patch():
    """Load a patch by filename and group, returning the module object.

    Patch files have module-level side effects (injecting symbols into vllm
    modules; see vllm_patch/patches/common/conftest.py). Snapshot affected
    modules' __dict__ before loading and restore on teardown to prevent
    cross-test leaks.
    """
    loaded = []
    module_snapshots = []  # [(module, dict(module.__dict__)), ...]

    # vllm modules touched by each patch file's module-level side effects
    sideeffect_modules = {}

    def _load(filename, group="common"):
        root = repo_root()
        subdir = "common" if group == "common" else Path("models") / group
        path = root / "omni" / "vllm_patches" / "patches" / subdir / f"{filename}.py"
        name = f"contract_test_{group}_{filename}"
        # Snapshot modules touched by module-level side effects before loading.
        for mod_name in sideeffect_modules.get(filename, []):
            module = importlib.import_module(mod_name)
            module_snapshots.append((module, dict(module.__dict__)))
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        loaded.append(name)
        return mod

    yield _load

    # Restore module __dict__ (undo module-level injections), then drop loaded patch modules.
    for module, snapshot in module_snapshots:
        for key in list(module.__dict__):
            if key not in snapshot:
                del module.__dict__[key]
        module.__dict__.update(snapshot)
    for name in loaded:
        sys.modules.pop(name, None)
