// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Vendored from omni-cache zero_copy_npu.cpp.
#include <torch/extension.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <tuple>

static void* g_dev_ptr = nullptr;

bool mlock_host(uintptr_t ptr, size_t size) {
    if (ptr == 0 || size == 0) {
        return false;
    }
    // mlock is a performance hint — hugepages are already non-swappable.
    // Ignore failure (e.g. RLIMIT_MEMLOCK exhausted) rather than aborting.
    if (mlock(reinterpret_cast<void*>(ptr), size) != 0) {
        std::cerr << "[ZeroCopy] mlock warning: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

std::tuple<torch::Tensor, torch::Tensor> register_hugepage_as_npu_tensor(
    torch::Tensor host_tensor, int device_id) {
    TORCH_CHECK(host_tensor.device().is_cpu());

    void* host_ptr = reinterpret_cast<void*>(host_tensor.data_ptr());
    size_t size = static_cast<size_t>(host_tensor.nbytes());
    mlock_host(reinterpret_cast<uintptr_t>(host_ptr), size);

    aclrtHostUnregister(host_ptr);
    aclrtSetDevice(device_id);
    void* dev_ptr = nullptr;
    aclError ret = aclrtHostRegister(
        host_ptr,
        size,
        ACL_HOST_REGISTER_MAPPED,
        &dev_ptr
    );

    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtHostRegister failed: " + std::to_string(ret));
    }

    g_dev_ptr = dev_ptr;
    c10::DeviceType device_type = c10::DeviceType::PrivateUse1;

    auto options = torch::TensorOptions()
        .dtype(host_tensor.dtype())
        .device(torch::Device(device_type, device_id));

    // 0-element stub: empty(host_tensor.sizes()) would allocate the full mmap
    // on NPU HBM via NPUCachingAllocator (e.g. 800 GiB) and OOM. Real storage
    // is the host-mapped pointer installed by set_ below.
    auto npu_tensor = torch::empty({0}, options);

    size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
        host_tensor.sizes(), host_tensor.dtype().itemsize());

    c10::DataPtr data_ptr(
        dev_ptr,
        dev_ptr,
        [](void*) {},
        c10::Device(c10::DeviceType::PrivateUse1, device_id)
    );

    at::Storage storage;
    auto fptr = c10::GetStorageImplCreate(device_type);
    auto allocator = c10::GetAllocator(device_type);

    storage = fptr(
        c10::StorageImpl::use_byte_size_t(),
        0,
        allocator->allocate(0),
        allocator,
        true);
    storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
    storage.set_data_ptr(std::move(data_ptr));

    npu_tensor.set_(storage, 0, host_tensor.sizes());

    return std::make_tuple(host_tensor, npu_tensor);
}

void unregister() {
    if (g_dev_ptr) {
        aclrtHostUnregister(g_dev_ptr);
        g_dev_ptr = nullptr;
    }
}

PYBIND11_MODULE(_zero_copy_npu, m) {
    m.def("mlock_host", &mlock_host);
    m.def("register_hugepage_as_npu_tensor", &register_hugepage_as_npu_tensor);
    m.def("unregister", &unregister);
}
