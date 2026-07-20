// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <mutex>

struct RegisteredTensor {
    void* cpu_ptr;      // CPU original ptr
    void* dev_ptr;      // NPU device addr
    size_t size;        // size (byte)
    int device_id;      // device where it was registered

    RegisteredTensor() : cpu_ptr(nullptr), dev_ptr(nullptr), size(0), device_id(-1) {}
};

static std::unordered_map<void*, RegisteredTensor> g_registry;
static std::mutex g_registry_mutex;
static bool g_acl_inited = false;

static void ensure_acl_initialized(int device_id) {
    // if (!g_acl_inited) {
    //     if (aclInit(nullptr) != ACL_ERROR_NONE) {
    //         std::cerr << "aclInit failed\n";
    //         std::abort();
    //     }
    //     g_acl_inited = true;
    // }
    // set device for current thread/context
    if (aclrtSetDevice(device_id) != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice(" << device_id << ") failed\n";
        std::abort();
    }
}

extern "C" int register_tensor(
    void* cpu_ptr,      // input: CPU ptr (must be valid)
    size_t size,        // input: memory size
    void** dev_ptr,     // output: NPU device addr
    int device_id       // input: which NPU device to register with
) {
    if (!cpu_ptr || size == 0 || dev_ptr == nullptr) return -1;

    // initialize ACL once and set requested device
    ensure_acl_initialized(device_id);

    // record previous device and restore semantics (optional)
    // Note: some ACL runtimes don't provide aclrtGetDevice for all versions; check return.
    int prev_device = -1;
    if (aclrtGetDevice(&prev_device) != ACL_ERROR_NONE) {
        prev_device = -1; // ignore if not available
    }
    // set desired device (ensure_acl_initialized already set it, but keep pattern)
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
        if (prev_device >= 0) {
            aclrtSetDevice(prev_device);
        }
        return static_cast<int>(ret);
    }

    aclError ret_ptr = aclrtHostGetDevicePointer(cpu_ptr, &out_dev_ptr, 0);
    if (ret_ptr != ACL_SUCCESS) {
        std::cerr << "aclrtHostGetDevicePointer failed: " << ret_ptr << std::endl;
        aclrtHostUnregister(cpu_ptr);
        if (prev_device >= 0) {
            aclrtSetDevice(prev_device);
        }
        return static_cast<int>(ret_ptr);
    }
    std::cout << "Mapped hugepage host tensor: cpu_ptr=" << cpu_ptr
              << " dev_ptr=" << out_dev_ptr
              << " size=" << size << " device_id=" << device_id << std::endl;

    // save registration info
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

    // restore previous device if we changed it
    if (prev_device >= 0 && prev_device != device_id) {
        aclrtSetDevice(prev_device);
    }

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

    // Set device corresponding to this registration before unregistering
    if (info.device_id >= 0) {
        if (aclrtSetDevice(info.device_id) != ACL_ERROR_NONE) {
            std::cerr << "aclrtSetDevice(" << info.device_id << ") failed for unregister\n";
            // continue attempt to unregister anyway
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
