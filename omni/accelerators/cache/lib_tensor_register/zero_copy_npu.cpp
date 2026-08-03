#include <torch/extension.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <stdexcept>
#include <sys/mman.h>

static void* g_dev_ptr = nullptr;
static bool g_acl_inited = false;

namespace {

torch::Tensor MakeMappedNpuView(
    const torch::Tensor& host_tensor,
    uintptr_t mapped_device_ptr,
    int device_id) {
    TORCH_CHECK(host_tensor.device().is_cpu(), "host_tensor must be on CPU");
    TORCH_CHECK(host_tensor.is_contiguous(), "host_tensor must be contiguous");
    TORCH_CHECK(mapped_device_ptr != 0, "mapped device pointer must be non-zero");

    const c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
    const auto device = c10::Device(device_type, device_id);
    const auto options = torch::TensorOptions().dtype(host_tensor.dtype()).device(device);

    // Allocate tensor metadata only. Allocating host_tensor.sizes() here would
    // temporarily reserve the entire mapped host cache in GM.
    auto npu_tensor = torch::empty({0}, options);
    const size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
        host_tensor.sizes(), host_tensor.dtype().itemsize());

    c10::DataPtr data_ptr(
        reinterpret_cast<void*>(mapped_device_ptr),
        reinterpret_cast<void*>(mapped_device_ptr),
        [](void*) {},
        device);

    auto create_storage = c10::GetStorageImplCreate(device_type);
    // This pointer belongs to aclrtHostRegister, not the NPU caching allocator.
    // A null allocator plus a no-op DataPtr deleter keeps ownership external.
    at::Storage storage = create_storage(
        c10::StorageImpl::use_byte_size_t(), tensor_nbytes, std::move(data_ptr), nullptr, false);
    npu_tensor.set_(storage, 0, host_tensor.sizes(), host_tensor.strides());
    return npu_tensor;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> register_hugepage_as_npu_tensor(torch::Tensor host_tensor, int device_id) {

    TORCH_CHECK(host_tensor.device().is_cpu());

    void* host_ptr = reinterpret_cast<void*>(host_tensor.data_ptr());
    size_t size = static_cast<size_t>(host_tensor.nbytes());

    if (mlock(host_ptr, size) != 0) {
        throw std::runtime_error("mlock failed: " + std::string(strerror(errno)));
    }

    std::cout << "<<< register_hugepage_as_npu_tensor [DEBUG] Host tensor ptr: " << std::hex << host_ptr
            << ", size: " << std::dec << size
            << ", 2M aligned: " << (((uintptr_t)host_ptr % (2 * 1024 * 1024)) == 0 ? "YES" : "NO")
            << ", 4K aligned: " << (((uintptr_t)host_ptr % (4096)) == 0 ? "YES" : "NO")
            << ", page aligned: " << ((uintptr_t)host_ptr % getpagesize() == 0 ? "YES" : "NO")
            << std::endl;

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
    auto npu_tensor = MakeMappedNpuView(
        host_tensor, reinterpret_cast<uintptr_t>(dev_ptr), device_id);
    return std::make_tuple(host_tensor, npu_tensor);
}

torch::Tensor wrap_registered_host_tensor(
    torch::Tensor host_tensor,
    uintptr_t mapped_device_ptr,
    int device_id) {
    return MakeMappedNpuView(host_tensor, mapped_device_ptr, device_id);
}

void unregister() {
    if (g_dev_ptr) {
        aclrtHostUnregister(g_dev_ptr);
        g_dev_ptr = nullptr;
        std::cout << "[ZeroCopy] Memory unregistered" << std::endl;
    }
}

void finalize_acl() {
    if (g_acl_inited) {
        aclrtDestroyStream(nullptr);
        aclrtDestroyContext(nullptr);
        aclFinalize();
        g_acl_inited = false;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("register_hugepage_as_npu_tensor", &register_hugepage_as_npu_tensor);
    m.def("wrap_registered_host_tensor", &wrap_registered_host_tensor);
    m.def("unregister", &unregister);
    m.def("finalize", &finalize_acl);
}
