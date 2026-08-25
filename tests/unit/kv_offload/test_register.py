# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for NPU OffloadingSpec factory override."""

from unittest.mock import MagicMock, patch

from omni_npu.v1.kv_offload.register import _override_spec, register_kv_offload_specs


def test_override_spec_installs_loader():
    factory = MagicMock()
    factory._registry = {}
    with patch(
        "vllm.v1.kv_offload.factory.OffloadingSpecFactory", factory, create=True
    ), patch("importlib.import_module") as import_mod:
        sentinel = MagicMock()
        import_mod.return_value = MagicMock(NPUCPUOffloadingSpec=sentinel)
        _override_spec(
            "CPUOffloadingSpec",
            "omni_npu.v1.kv_offload.cpu.spec",
            "NPUCPUOffloadingSpec",
        )
        loaded = factory._registry["CPUOffloadingSpec"]()
        assert loaded is sentinel
        import_mod.assert_called_once_with("omni_npu.v1.kv_offload.cpu.spec")


def test_register_kv_offload_specs_overrides_cpu_spec():
    with patch("omni_npu.v1.kv_offload.register._override_spec") as override:
        register_kv_offload_specs()
        override.assert_called_once_with(
            "CPUOffloadingSpec",
            "omni_npu.v1.kv_offload.cpu.spec",
            "NPUCPUOffloadingSpec",
        )
