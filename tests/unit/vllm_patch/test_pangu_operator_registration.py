# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Isolated registration/order tests; no torch, vLLM or NPU is required.

Run this file directly to avoid the repository's hardware-dependent conftest.
The module under test is loaded from this tree, without importing omni.layers.
Operator imports are mocked; runtime dependencies need integration validation.
"""

import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
MHC_MODULE = "omni_npu.layers.mhc.mhc"
TRACE_MODULE = "omni_npu.vllm_patches.patches.common.patch_trace"


class OperatorImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Use normal import caching, with only the operator bodies substituted."""

    def __init__(self, events):
        self.events = events
        self.failures = {}
        self.registry = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in (MHC_MODULE, TRACE_MODULE):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == TRACE_MODULE:
            module.ProfilerDynamicPatch = lambda: self.events.append("trace")
            return
        self.events.append("mhc")
        if "mhc" in self.failures:
            raise self.failures["mhc"]
        registered_class = type("NPUmHCModule", (), {})
        self.registry["mhc"] = registered_class
        module.NPUmHCModule = registered_class


class PanguOperatorRegistrationTests(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.events = []
        self.finder = None
        self.module = None

    def setUp(self):
        self.events = []
        self.finder = OperatorImports(self.events)
        packages = {}
        for leaf in (MHC_MODULE, TRACE_MODULE):
            parts = leaf.split(".")
            for index in range(1, len(parts)):
                name = ".".join(parts[:index])
                package = ModuleType(name)
                package.__path__ = []
                packages[name] = package
        name = "_operator_registration_under_test"
        manager_module = ModuleType(name + ".patch_manager")
        events = self.events

        class PatchManager:
            def apply_patches(self):
                events.append("apply_patches")

        manager_module.PatchManager = PatchManager
        packages[manager_module.__name__] = manager_module
        modules_patch = patch.dict(sys.modules, packages)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        for leaf in (MHC_MODULE, TRACE_MODULE):
            sys.modules.pop(leaf, None)
        imports_patch = patch.object(sys, "meta_path", [self.finder, *sys.meta_path])
        imports_patch.start()
        self.addCleanup(imports_patch.stop)
        env_patch = patch.dict(os.environ, {}, clear=False)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop("OMNI_VLLM_PATCHES_DIR", None)
        os.environ.pop("OMNI_NPU_PATCHES_DIR", None)
        source = ROOT / "omni/vllm_patches/__init__.py"
        spec = importlib.util.spec_from_file_location(name, source)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.module
        spec.loader.exec_module(self.module)
        self.module.auto_import_patches = lambda: self.events.append("import_patches")

    def run_with_dirs(self, value):
        os.environ["OMNI_VLLM_PATCHES_DIR"] = value
        self.module.apply_patches()

    def test_vl_registers_mhc_after_patches_before_trace(self):
        for value in ("openpangu_v1_vl", "openpangu_v1_vl,pangu_v2_base",
                      "openpangu_v1_vl,high_throughout", "openpangu_v1_vl,low_latency"):
            with self.subTest(value=value):
                self.events.clear()
                sys.modules.pop(MHC_MODULE, None)
                self.run_with_dirs(value)
                self.assertEqual(self.events, ["import_patches", "apply_patches", "mhc", "trace"])

    def test_no_registration_without_vl(self):
        for value in ("", "pangu_v2_base", "high_throughout", "low_latency",
                      "pangu_sink_swa_mla", "openpangu_v1_vl_extra,high_throughout"):
            with self.subTest(value=value):
                self.events.clear()
                self.run_with_dirs(value)
                self.assertEqual(self.events, ["import_patches", "apply_patches", "trace"])

    def test_existing_high_alias_case_whitespace_and_duplicates(self):
        self.run_with_dirs(" OpenPangu_V1_VL , PANGU_V2_HYBRID, openpangu_v1_vl ")
        self.assertEqual(self.events, ["import_patches", "apply_patches", "mhc", "trace"])

    def test_legacy_environment_is_used_when_current_is_unset(self):
        os.environ["OMNI_NPU_PATCHES_DIR"] = "openpangu_v1_vl"
        self.module.apply_patches()
        self.assertEqual(self.events, ["import_patches", "apply_patches", "mhc", "trace"])

    def test_current_environment_has_precedence(self):
        os.environ["OMNI_NPU_PATCHES_DIR"] = "openpangu_v1_vl"
        self.run_with_dirs("pangu_v2_base")
        self.assertEqual(self.events, ["import_patches", "apply_patches", "trace"])

    def test_repeated_apply_keeps_one_operator_class_and_module_execution(self):
        self.run_with_dirs("openpangu_v1_vl")
        registered_class = self.finder.registry["mhc"]
        self.module.apply_patches()
        self.assertIs(self.finder.registry["mhc"], registered_class)
        self.assertIs(sys.modules[MHC_MODULE].NPUmHCModule, registered_class)
        self.assertEqual(self.events.count("mhc"), 1)
        self.assertEqual(self.events.count("trace"), 2)

    def test_required_mhc_import_error_cannot_fall_back(self):
        self.finder.failures["mhc"] = ImportError("missing NPU extension")
        with self.assertRaisesRegex(ImportError, "missing NPU extension"):
            self.run_with_dirs("openpangu_v1_vl")
        self.assertEqual(self.events, ["import_patches", "apply_patches", "mhc"])

    def test_registration_conflicts_are_not_suppressed(self):
        self.finder.failures["mhc"] = AssertionError("Duplicate op name")
        with self.assertRaisesRegex(AssertionError, "Duplicate op name"):
            self.run_with_dirs("openpangu_v1_vl,high_throughout")


if __name__ == "__main__":
    unittest.main()
