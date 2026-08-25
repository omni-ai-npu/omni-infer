// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Vendored from vllm-ascend csrc/torch_binding.cpp::swap_blocks_batch.
// Host attrs must be HOST: rtsMemcpyBatchAsync treats mmap/malloc as HOST(0)
// even when PointerGetAttributes reports UNREGISTERED. See tests/memcpybatch.
#include <torch/extension.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

// Host attrs must be HOST. rtsMemcpyBatchAsync rejects numBatches > 4096
// with 107000 SIZE_MAX (see tests/memcpybatch --batch 4231).
static constexpr const char kBuildTag[] = "host-attr-chunk4096-v4";
static constexpr size_t kMaxMemcpyBatch = 4096;

namespace {

const char* loc_name(aclrtMemLocationType type) {
    switch (type) {
        case ACL_MEM_LOCATION_TYPE_HOST:
            return "HOST";
        case ACL_MEM_LOCATION_TYPE_DEVICE:
            return "DEVICE";
        case ACL_MEM_LOCATION_TYPE_UNREGISTERED:
            return "UNREGISTERED";
        default:
            return "UNKNOWN";
    }
}

std::string ptr_attr_str(const void* ptr) {
    std::ostringstream oss;
    oss << "ptr=" << ptr;
    if (ptr == nullptr) {
        return oss.str();
    }
    aclrtPtrAttributes attr = {};
    aclError ret = aclrtPointerGetAttributes(ptr, &attr);
    oss << " getAttr=" << ret;
    if (ret == ACL_SUCCESS) {
        oss << " loc=" << loc_name(attr.location.type)
            << " loc_id=" << attr.location.id
            << " pageSize=" << attr.pageSize;
    }
    return oss.str();
}

}  // namespace

void swap_blocks_batch(
    const torch::Tensor& src_ptrs,
    const torch::Tensor& dst_ptrs,
    const torch::Tensor& sizes,
    int64_t direction) {
    TORCH_CHECK(src_ptrs.device().is_cpu(), "src_ptrs must be on CPU");
    TORCH_CHECK(dst_ptrs.device().is_cpu(), "dst_ptrs must be on CPU");
    TORCH_CHECK(sizes.device().is_cpu(), "sizes must be on CPU");
    TORCH_CHECK(src_ptrs.dtype() == torch::kInt64, "src_ptrs must be int64");
    TORCH_CHECK(dst_ptrs.dtype() == torch::kInt64, "dst_ptrs must be int64");
    TORCH_CHECK(sizes.dtype() == torch::kInt64, "sizes must be int64");

    const int64_t n = src_ptrs.size(0);
    TORCH_CHECK(dst_ptrs.size(0) == n, "dst_ptrs length must match src_ptrs");
    TORCH_CHECK(sizes.size(0) == n, "sizes length must match src_ptrs");
    if (n == 0) {
        return;
    }

    auto src_cpu = src_ptrs.contiguous();
    auto dst_cpu = dst_ptrs.contiguous();
    auto size_cpu = sizes.contiguous();
    const int64_t* src_data = src_cpu.data_ptr<int64_t>();
    const int64_t* dst_data = dst_cpu.data_ptr<int64_t>();
    const int64_t* size_data = size_cpu.data_ptr<int64_t>();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    aclrtMemcpyKind memcpy_kind;
    switch (direction) {
        case 0:
            memcpy_kind = ACL_MEMCPY_HOST_TO_DEVICE;
            break;
        case 1:
            memcpy_kind = ACL_MEMCPY_DEVICE_TO_HOST;
            break;
        case 2:
            memcpy_kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            break;
        default:
            TORCH_CHECK(false,
                        "swap_blocks_batch: invalid direction ", direction,
                        " (expected 0=H2D, 1=D2H, 2=D2D)");
    }

#if defined(CANN_MEMCPY_BATCH_ASYNC)
    if (memcpy_kind != ACL_MEMCPY_DEVICE_TO_DEVICE) {
        std::vector<void*> src_vec(static_cast<size_t>(n));
        std::vector<void*> dst_vec(static_cast<size_t>(n));
        std::vector<size_t> size_vec(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            src_vec[static_cast<size_t>(i)] = reinterpret_cast<void*>(src_data[i]);
            dst_vec[static_cast<size_t>(i)] = reinterpret_cast<void*>(dst_data[i]);
            size_vec[static_cast<size_t>(i)] = static_cast<size_t>(size_data[i]);
        }

        aclrtPtrAttributes src_pa = {};
        aclrtPtrAttributes dst_pa = {};
        TORCH_CHECK(
            aclrtPointerGetAttributes(src_vec[0], &src_pa) == ACL_SUCCESS,
            "aclrtPointerGetAttributes failed on src[0]");
        TORCH_CHECK(
            aclrtPointerGetAttributes(dst_vec[0], &dst_pa) == ACL_SUCCESS,
            "aclrtPointerGetAttributes failed on dst[0]");

        int32_t get_dev = -1;
        aclrtGetDevice(&get_dev);

        // Runtime memcpy validator classifies host mmap/malloc/mallochost as
        // HOST. Device loc_id comes from the NPU pointer, not GetDevice().
        aclrtMemLocation host_loc = {};
        host_loc.type = ACL_MEM_LOCATION_TYPE_HOST;
        host_loc.id = 0;

        aclrtMemLocation device_loc = {};
        device_loc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        if (src_pa.location.type == ACL_MEM_LOCATION_TYPE_DEVICE) {
            device_loc.id = src_pa.location.id;
        } else if (dst_pa.location.type == ACL_MEM_LOCATION_TYPE_DEVICE) {
            device_loc.id = dst_pa.location.id;
        } else {
            device_loc.id = static_cast<uint32_t>(get_dev);
        }

        aclrtMemcpyBatchAttr attr = {};
        if (memcpy_kind == ACL_MEMCPY_HOST_TO_DEVICE) {
            attr.srcLoc = host_loc;
            attr.dstLoc = device_loc;
        } else {
            attr.srcLoc = device_loc;
            attr.dstLoc = host_loc;
        }

        size_t fail_index = 0;
        size_t chunk_offset = 0;
        aclError result = ACL_SUCCESS;
        const size_t total = static_cast<size_t>(n);
        for (size_t offset = 0; offset < total; offset += kMaxMemcpyBatch) {
            const size_t chunk = std::min(kMaxMemcpyBatch, total - offset);
            size_t attrs_index = 0;
            fail_index = 0;
            result = aclrtMemcpyBatchAsync(
                dst_vec.data() + offset, size_vec.data() + offset,
                src_vec.data() + offset, size_vec.data() + offset, chunk,
                &attr, &attrs_index, 1, &fail_index, stream);
            if (result != ACL_SUCCESS) {
                chunk_offset = offset;
                break;
            }
        }
        if (result != ACL_SUCCESS) {
            std::ostringstream oss;
            oss << "KV_OFFLOAD_BATCH tag=" << kBuildTag
                << " aclrtMemcpyBatchAsync failed error=" << result
                << " fail_index=" << (chunk_offset + fail_index)
                << " chunk_offset=" << chunk_offset
                << " chunk_fail_index=" << fail_index
                << " n=" << n
                << " direction=" << direction
                << " getDevice=" << get_dev
                << " srcLoc=" << loc_name(attr.srcLoc.type)
                << " srcLoc_id=" << attr.srcLoc.id
                << " dstLoc=" << loc_name(attr.dstLoc.type)
                << " dstLoc_id=" << attr.dstLoc.id
                << " size0=" << size_vec[0]
                << " copy[0] src{" << ptr_attr_str(src_vec[0]) << "}"
                << " dst{" << ptr_attr_str(dst_vec[0]) << "}";
            TORCH_CHECK(false, oss.str());
        }
        return;
    }
#endif

    for (int64_t i = 0; i < n; i++) {
        void* dst = reinterpret_cast<void*>(dst_data[i]);
        const void* src = reinterpret_cast<const void*>(src_data[i]);
        size_t copy_size = static_cast<size_t>(size_data[i]);
        aclError ret = aclrtMemcpyAsync(
            dst, copy_size, src, copy_size, memcpy_kind, stream);
        TORCH_CHECK(ret == ACL_SUCCESS,
                    "aclrtMemcpyAsync failed at index ", i,
                    " with error code ", ret);
    }
}

PYBIND11_MODULE(_swap_blocks_batch, m) {
    m.attr("build_tag") = kBuildTag;
#if defined(CANN_MEMCPY_BATCH_ASYNC)
    m.attr("cann_memcpy_batch") = true;
    m.attr("host_location") = "HOST";
#else
    m.attr("cann_memcpy_batch") = false;
    m.attr("host_location") = "unused";
#endif
    m.def("swap_blocks_batch", &swap_blocks_batch);
}
