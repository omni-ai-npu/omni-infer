# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Isolated interface doubles for CPU patch tests, not a fake vLLM runtime."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def patch_env(monkeypatch):
    root = Path(__file__).resolve().parents[5]

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            if parent not in sys.modules:
                stub(parent)
            monkeypatch.setattr(sys.modules[parent], attr, module, raising=False)
        return module

    def load_file(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    registered = {}
    stub("vllm")
    stub("vllm.logger", init_logger=logging.getLogger)
    stub("omni_npu.vllm_patches",
         PatchManager=SimpleNamespace(register=registered.__setitem__))
    load_file("omni_npu.vllm_patches.core", root / "omni/vllm_patches/core.py")

    def load(filename, group="openpangu_v1_vl"):
        return load_file(
            f"vl_test_{filename}",
            root / "omni/vllm_patches/patches/models" / group
            / f"{filename}.py",
        )

    return SimpleNamespace(stub=stub, load=load, registered=registered)
