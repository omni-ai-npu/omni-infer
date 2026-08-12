// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
//
// NPU UVA view support for vLLM MRv2.
//
// vLLM's UvaBuffer needs a device Tensor view of a pinned CPU tensor. The
// pinned allocation is registered by torch_npu when
// PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True is set before process start.
// This extension only queries the existing CANN device pointer and wraps it as
// a device=npu Tensor. It must not register or unregister host memory.

#include <torch/extension.h>
#include <torch_npu/csrc/aten/common/from_blob.h>

#include <acl/acl.h>

#include <cstdint>
#include <limits>

namespace py = pybind11;

namespace {

#define ACL_CHECK(expr)                                                       \
  do {                                                                        \
    const aclError ret = (expr);                                              \
    TORCH_CHECK(ret == ACL_SUCCESS, #expr, " failed with aclError=", ret);    \
  } while (0)

constexpr c10::DeviceType kNPUDeviceType = c10::DeviceType::PrivateUse1;

void check_cpu_tensor(const at::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "input tensor must be on CPU");
  TORCH_CHECK(cpu_tensor.is_pinned(), "CPU tensor must be pinned");
  TORCH_CHECK(cpu_tensor.numel() > 0,
              "cannot create an NPU UVA view for an empty CPU tensor");
}

void* get_registered_device_pointer(const at::Tensor& cpu_tensor) {
  void* dev_ptr = nullptr;
  // CANN signature:
  //   aclrtHostGetDevicePointer(void *pHost, void **pDevice, uint32_t flag)
  ACL_CHECK(aclrtHostGetDevicePointer(cpu_tensor.data_ptr(), &dev_ptr, 0));
  TORCH_CHECK(dev_ptr != nullptr,
              "aclrtHostGetDevicePointer returned NULL. Ensure "
              "PYTORCH_NPU_ALLOC_CONF includes pinned_mem_register:True before "
              "the process creates pinned memory, and the CANN/driver stack "
              "supports mapped pinned host memory.");
  return dev_ptr;
}

c10::Device make_npu_device(int64_t device_index) {
  TORCH_CHECK(device_index >= 0, "NPU device index must be non-negative");
  TORCH_CHECK(device_index <= std::numeric_limits<c10::DeviceIndex>::max(),
              "NPU device index is out of range: ", device_index);
  return c10::Device(kNPUDeviceType,
                     static_cast<c10::DeviceIndex>(device_index));
}

}  // namespace

uintptr_t get_device_pointer(const at::Tensor& cpu_tensor) {
  check_cpu_tensor(cpu_tensor);
  return reinterpret_cast<uintptr_t>(get_registered_device_pointer(cpu_tensor));
}

at::Tensor get_npu_view_from_cpu_tensor(const at::Tensor& cpu_tensor,
                                        int64_t device_index) {
  check_cpu_tensor(cpu_tensor);

  void* dev_ptr = get_registered_device_pointer(cpu_tensor);
  const c10::Device npu_device = make_npu_device(device_index);
  auto options = cpu_tensor.options().device(npu_device).pinned_memory(false);

  // Keep the source CPU tensor alive. Registration/unregistration remains owned
  // by torch_npu's pinned host allocator.
  auto deleter = [base = cpu_tensor](void*) mutable {};

  return at_npu::native::from_blob(dev_ptr, cpu_tensor.sizes(),
                                   cpu_tensor.strides(), deleter, options,
                                   npu_device);
}

PYBIND11_MODULE(npu_uva, m) {
  m.def("get_device_pointer", &get_device_pointer, py::arg("cpu_tensor"),
        "Return the CANN device pointer for a registered pinned CPU tensor");
  m.def("get_npu_view_from_cpu_tensor", &get_npu_view_from_cpu_tensor,
        py::arg("cpu_tensor"), py::arg("device_index") = 0,
        "Return a device=npu Tensor view aliasing a pinned CPU tensor");
}