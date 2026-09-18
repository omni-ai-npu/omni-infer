# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared fixtures for the common patch tests that load real patch files.

Locates the repo via ``patch_test_utils.repo_root`` and loads each patch under
a standalone module name, snapshotting/restoring any module-level side effects.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patch_test_utils import repo_root  # noqa: E402


@pytest.fixture
def load_patch():
    """Load a common patch by filename, returning the module object.

    Uses a fresh module name each time; pops it on teardown and restores
    snapshot side effects to avoid registry/sys.modules collisions.
    """

    loaded = []
    module_snapshots = []  # [(module, dict(module.__dict__)), ...]

    # vllm modules touched by each patch file's module-level side effects
    sideeffect_modules = {}

    def _load(filename):
        root = repo_root()
        path = root / "omni" / "vllm_patches" / "patches" / "common" / f"{filename}.py"
        name = f"common_test_{filename}"
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
