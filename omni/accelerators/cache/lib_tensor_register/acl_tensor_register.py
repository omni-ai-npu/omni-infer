import os
import ctypes
import ctypes.util
import torch
from . import zero_copy_npu
from vllm.logger import init_logger

logger = init_logger(__name__)

_lib_path = os.path.join(os.path.dirname(__file__), "tensor_register.so")
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


class NPUTensorRegister:
    def __init__(self) -> None:
        pass

    def host_tensor_register(self, host_tensor: torch.Tensor, device_id: int = 0) -> torch.Tensor:
        x = torch.zeros((10,2), device="npu") # to trigger acl initialization
        if not isinstance(host_tensor, torch.Tensor):
            raise TypeError("host_tensor must be a torch.Tensor")
        if host_tensor.device.type != "cpu":
            raise ValueError("host_tensor must be a CPU tensor")

        host_ptr = ctypes.c_void_p(host_tensor.data_ptr())
        size = host_tensor.numel() * host_tensor.element_size()

        dev_ptr = ctypes.c_void_p()
        ret = _lib.register_tensor(host_ptr, size, ctypes.byref(dev_ptr), device_id)
        if ret != 0:
            raise RuntimeError(f"Register tensor failed: ACL error code {ret}")

        if dev_ptr.value is None:
            raise RuntimeError("Register tensor failed: native returned null device pointer")
        
        npu_tensor = zero_copy_npu.wrap_registered_host_tensor(
            host_tensor, dev_ptr.value, device_id
        )
        return host_tensor, npu_tensor
