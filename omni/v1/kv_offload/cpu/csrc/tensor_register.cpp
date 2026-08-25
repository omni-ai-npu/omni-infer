// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Vendored from omni-cache tensor_register.cpp (aclrtHostRegisterV2 PINNED|MAPPED).
// pin_host() is PINNED-only for shared mmap + MemcpyBatchAsync H2D/D2H.
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <pybind11/pybind11.h>

struct RegisteredTensor {
    void* cpu_ptr;      // CPU original ptr
    void* dev_ptr;      // NPU device addr (MAPPED only; else nullptr)
    size_t size;        // size (byte)
    int device_id;      // device where it was registered

    RegisteredTensor() : cpu_ptr(nullptr), dev_ptr(nullptr), size(0), device_id(-1) {}
};

static std::unordered_map<void*, RegisteredTensor> g_registry;
static std::mutex g_registry_mutex;

static void ensure_acl_initialized(int device_id) {
    if (aclrtSetDevice(device_id) != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice(" << device_id << ") failed\n";
        std::abort();
    }
}

static int restore_device(int prev_device, int device_id) {
    if (prev_device >= 0 && prev_device != device_id) {
        return aclrtSetDevice(prev_device) == ACL_ERROR_NONE ? 0 : -1;
    }
    return 0;
}

extern "C" int pin_host(void* cpu_ptr, size_t size, int device_id) {
    if (!cpu_ptr || size == 0) {
        return -1;
    }
    ensure_acl_initialized(device_id);
    int prev_device = -1;
    if (aclrtGetDevice(&prev_device) != ACL_ERROR_NONE) {
        prev_device = -1;
    }
    if (aclrtSetDevice(device_id) != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice(" << device_id << ") failed\n";
        return -1;
    }
    if (reinterpret_cast<uintptr_t>(cpu_ptr) % 4096 != 0) {
        std::cerr << "pin_host: CPU pointer not 4K aligned: " << cpu_ptr
                  << std::endl;
        restore_device(prev_device, device_id);
        return -1;
    }

    (void)aclrtHostUnregister(cpu_ptr);

    std::cout << "HostRegister PINNED mmap: cpu_ptr=" << cpu_ptr
              << " size=" << size << " device_id=" << device_id << std::endl;
    aclError ret = aclrtHostRegisterV2(cpu_ptr, size, ACL_HOST_REG_PINNED);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtHostRegisterV2(PINNED) failed: " << ret << std::endl;
        restore_device(prev_device, device_id);
        return static_cast<int>(ret);
    }

    {
        std::lock_guard<std::mutex> lk(g_registry_mutex);
        RegisteredTensor info;
        info.cpu_ptr = cpu_ptr;
        info.dev_ptr = nullptr;
        info.size = size;
        info.device_id = device_id;
        g_registry[cpu_ptr] = info;
    }
    restore_device(prev_device, device_id);
    return 0;
}

extern "C" int register_tensor(
    void* cpu_ptr,      // input: CPU ptr (must be valid)
    size_t size,        // input: memory size
    void** dev_ptr,     // output: NPU device addr
    int device_id       // input: which NPU device to register with
) {
    if (!cpu_ptr || size == 0 || dev_ptr == nullptr) return -1;

    ensure_acl_initialized(device_id);

    int prev_device = -1;
    if (aclrtGetDevice(&prev_device) != ACL_ERROR_NONE) {
        prev_device = -1;
    }
    if (aclrtSetDevice(device_id) != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice(" << device_id << ") failed\n";
        return -1;
    }

    if (reinterpret_cast<uintptr_t>(cpu_ptr) % 4096 != 0) {
        std::cerr << "Warning: CPU pointer not 4K aligned: " << cpu_ptr << std::endl;
    }

    void* out_dev_ptr = nullptr;
    std::cout << "Register hugepage host tensor: cpu_ptr=" << cpu_ptr
              << " size=" << size << " device_id=" << device_id << std::endl;
    aclError ret = aclrtHostRegisterV2(
        cpu_ptr, size, ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtHostRegisterV2 failed: " << ret << std::endl;
        restore_device(prev_device, device_id);
        return static_cast<int>(ret);
    }

    aclError ret_ptr = aclrtHostGetDevicePointer(cpu_ptr, &out_dev_ptr, 0);
    if (ret_ptr != ACL_SUCCESS) {
        std::cerr << "aclrtHostGetDevicePointer failed: " << ret_ptr << std::endl;
        aclrtHostUnregister(cpu_ptr);
        restore_device(prev_device, device_id);
        return static_cast<int>(ret_ptr);
    }
    std::cout << "Mapped hugepage host tensor: cpu_ptr=" << cpu_ptr
              << " dev_ptr=" << out_dev_ptr
              << " size=" << size << " device_id=" << device_id << std::endl;

    {
        std::lock_guard<std::mutex> lk(g_registry_mutex);
        RegisteredTensor info;
        info.cpu_ptr = cpu_ptr;
        info.dev_ptr = out_dev_ptr;
        info.size = size;
        info.device_id = device_id;
        g_registry[cpu_ptr] = info;
    }

    *dev_ptr = out_dev_ptr;
    restore_device(prev_device, device_id);
    return 0;
}

extern "C" int unregister_tensor(void* cpu_ptr) {
    if (cpu_ptr == nullptr) return -1;

    RegisteredTensor info;
    {
        std::lock_guard<std::mutex> lk(g_registry_mutex);
        auto it = g_registry.find(cpu_ptr);
        if (it == g_registry.end()) {
            std::cerr << "unregister_tensor: cpu_ptr not found\n";
            return -1;
        }
        info = it->second;
        g_registry.erase(it);
    }

    if (info.device_id >= 0) {
        if (aclrtSetDevice(info.device_id) != ACL_ERROR_NONE) {
            std::cerr << "aclrtSetDevice(" << info.device_id
                      << ") failed for unregister\n";
        }
    }

    aclError ret = aclrtHostUnregister(info.cpu_ptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtHostUnregister failed: " << ret << std::endl;
    }
    return static_cast<int>(ret);
}

extern "C" void* get_dev_ptr_from_cpu(void* cpu_ptr) {
    std::lock_guard<std::mutex> lk(g_registry_mutex);
    auto it = g_registry.find(cpu_ptr);
    if (it != g_registry.end()) return it->second.dev_ptr;
    return nullptr;
}

namespace py = pybind11;

PYBIND11_MODULE(_host_register, m) {
    m.doc() = "ACL host register: PINNED mmap (pin_host) or PINNED|MAPPED (register_tensor)";
    m.attr("register_mode") = "PINNED";
    m.def(
        "pin_host",
        [](uintptr_t cpu_ptr, size_t size, int device_id) {
            int ret = pin_host(reinterpret_cast<void*>(cpu_ptr), size, device_id);
            if (ret != 0) {
                throw std::runtime_error(
                    "aclrtHostRegisterV2(PINNED) failed: " +
                    std::to_string(ret));
            }
            return ret;
        },
        py::arg("cpu_ptr"),
        py::arg("size"),
        py::arg("device_id") = 0);
    m.def(
        "register_tensor",
        [](uintptr_t cpu_ptr, size_t size, int device_id) {
            void* dev = nullptr;
            int ret = register_tensor(
                reinterpret_cast<void*>(cpu_ptr), size, &dev, device_id);
            if (ret != 0) {
                throw std::runtime_error(
                    "aclrtHostRegisterV2 failed: " + std::to_string(ret));
            }
            return ret;
        },
        py::arg("cpu_ptr"),
        py::arg("size"),
        py::arg("device_id") = 0);
    m.def(
        "unregister_tensor",
        [](uintptr_t cpu_ptr) {
            return unregister_tensor(reinterpret_cast<void*>(cpu_ptr));
        },
        py::arg("cpu_ptr"));
    m.def(
        "get_dev_ptr_from_cpu",
        [](uintptr_t cpu_ptr) {
            return reinterpret_cast<uintptr_t>(
                get_dev_ptr_from_cpu(reinterpret_cast<void*>(cpu_ptr)));
        },
        py::arg("cpu_ptr"));
}
