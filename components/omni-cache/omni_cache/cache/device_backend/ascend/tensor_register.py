# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU Tensor registration for zero-copy host-device transfers.

This module provides NPUTensorRegister class for registering CPU tensors
with the NPU device for zero-copy memory access.
"""

import os
import ctypes
import ctypes.util
import torch
import sys

# Import zero_copy_npu from tensor_register_lib
sys.path.append(os.path.dirname(__file__))
try:
    from .tensor_register_lib import zero_copy_npu
    _ZERO_COPY_AVAILABLE = True
except ImportError:
    zero_copy_npu = None
    _ZERO_COPY_AVAILABLE = False

from vllm.logger import init_logger

logger = init_logger(__name__)

_lib_path = os.path.join(os.path.dirname(__file__), "tensor_register_lib", "tensor_register.so")


class _MissingTensorRegisterLib:
    def register_tensor(self, *args, **kwargs):
        return -1

    def unregister_tensor(self, *args, **kwargs):
        return -1

    def get_dev_ptr_from_cpu(self, *args, **kwargs):
        return None


try:
    _lib = ctypes.CDLL(_lib_path)

    _lib.register_tensor.argtypes = [
        ctypes.c_void_p,                   # cpu_ptr
        ctypes.c_size_t,                   # size
        ctypes.POINTER(ctypes.c_void_p),   # dev_ptr (out)
        ctypes.c_int                       # device_id
    ]
    _lib.register_tensor.restype = ctypes.c_int

    _lib.unregister_tensor.argtypes = [ctypes.c_void_p]
    _lib.unregister_tensor.restype = ctypes.c_int

    _lib.get_dev_ptr_from_cpu.argtypes = [ctypes.c_void_p]
    _lib.get_dev_ptr_from_cpu.restype = ctypes.c_void_p
except OSError as exc:
    logger.warning("tensor_register.so is unavailable, native registration is disabled: %s", exc)
    _lib = _MissingTensorRegisterLib()


class NPUTensorRegister:
    """Register CPU tensors for zero-copy NPU access.

    This class provides functionality to register host (CPU) tensors with
    the Ascend NPU device, enabling zero-copy data transfers between
    host and device memory.
    """

    def __init__(self) -> None:
        pass

    def host_tensor_register(self, host_tensor: torch.Tensor, device_id: int = 0) -> torch.Tensor:
        """Register a host tensor for zero-copy NPU access.

        Args:
            host_tensor: CPU tensor to register
            device_id: NPU device ID to register with

        Returns:
            Tuple of (host_tensor, npu_tensor) where npu_tensor is the
            NPU-accessible view of the host memory.

        Raises:
            TypeError: If host_tensor is not a torch.Tensor
            ValueError: If host_tensor is not on CPU
            RuntimeError: If registration fails or extension not available
        """
        if not isinstance(host_tensor, torch.Tensor):
            raise TypeError("host_tensor must be a torch.Tensor")
        if host_tensor.device.type != "cpu":
            raise ValueError("host_tensor must be a CPU tensor")

        # Check if native extension is available
        if not _ZERO_COPY_AVAILABLE or zero_copy_npu is None:
            raise RuntimeError(
                "Tensor registration requires zero_copy_npu extension. "
                "Please build the extension with: "
                "cd tensor_register_lib && bash build.sh"
            )

        # Trigger ACL initialization only when an NPU backend is present.
        # Keep this after validation so unit tests can assert input errors.
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                _ = torch.zeros((1, 1), device="npu")
        except Exception:
            # Registration path below will surface actionable errors.
            pass

        host_ptr = ctypes.c_void_p(host_tensor.data_ptr())
        size = host_tensor.nbytes

        dev_ptr = ctypes.c_void_p()
        ret = _lib.register_tensor(host_ptr, size, ctypes.byref(dev_ptr), device_id)
        if ret != 0:
            raise RuntimeError(f"Register tensor failed: ACL error code {ret}")

        if dev_ptr.value is None:
            raise RuntimeError("Register tensor failed: native returned null device pointer")

        _, npu_tensor = zero_copy_npu.register_hugepage_as_npu_tensor(host_tensor, device_id)
        return host_tensor, npu_tensor


__all__ = ['NPUTensorRegister']
