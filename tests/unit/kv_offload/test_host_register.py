# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Coverage for HostRegister wrappers with mocked extensions."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from omni_npu.v1.kv_offload.cpu import host_register as hr


def test_pin_and_register_reject_non_tensor():
    with pytest.raises(TypeError, match="torch.Tensor"):
        hr.pin_host_mmap("x")
    with pytest.raises(TypeError, match="torch.Tensor"):
        hr.register_host_tensor(object())


def test_pin_and_register_reject_non_cpu_tensor():
    class FakeDev:
        type = "npu"

    t = MagicMock(spec=torch.Tensor)
    t.device = FakeDev()
    with pytest.raises(ValueError, match="CPU tensor"):
        hr.pin_host_mmap(t)
    with pytest.raises(ValueError, match="CPU tensor"):
        hr.register_host_tensor(t)


def test_ext_loaders_and_pin_register_unregister():
    host_ext = MagicMock()
    zero_ext = MagicMock()
    npu_out = torch.zeros(1, dtype=torch.int8, device="cpu")
    zero_ext.register_hugepage_as_npu_tensor.return_value = (None, npu_out)
    host_ext.unregister_tensor.return_value = 0

    cpu = torch.zeros(8, dtype=torch.int8, device="cpu")
    with patch.object(hr, "_host_register_ext", return_value=host_ext), patch.object(
        hr, "_zero_copy_ext", return_value=zero_ext
    ), patch.object(torch, "zeros", return_value=cpu):
        hr.pin_host_mmap(cpu, device_id=1)
        host_ext.pin_host.assert_called_once_with(cpu.data_ptr(), cpu.nbytes, 1)

        out_host, out_npu = hr.register_host_tensor(cpu, device_id=2)
        assert out_host is cpu
        assert out_npu is npu_out
        host_ext.register_tensor.assert_called_once_with(
            cpu.data_ptr(), cpu.nbytes, 2
        )
        zero_ext.register_hugepage_as_npu_tensor.assert_called_once_with(cpu, 2)

        assert hr.unregister_host_tensor(0xABC) == 0
        host_ext.unregister_tensor.assert_called_once_with(0xABC)


def test_ext_import_helpers():
    host_mod = MagicMock(name="host")
    zero_mod = MagicMock(name="zero")
    with patch.dict(
        "sys.modules",
        {
            "omni_npu.v1.kv_offload.cpu._host_register": host_mod,
            "omni_npu.v1.kv_offload.cpu._zero_copy_npu": zero_mod,
        },
    ):
        assert hr._host_register_ext() is host_mod
        assert hr._zero_copy_ext() is zero_mod
