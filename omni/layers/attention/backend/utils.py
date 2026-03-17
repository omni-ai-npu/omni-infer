import os
import torch
import torch_npu

def create_aligend_tensor(shape, dtype, device, pin_memory=False):
    enable_2mb_alignment = os.getenv("VLLM_ENABLE_KV_CACHE_TENSOR_2MB_ALIGNMENT", "0") == "1"

    if not enable_2mb_alignment:
        return torch.zeros(shape, dtype=dtype, pin_memory=pin_memory, device=device)

    total_elements = 1
    for dim in shape:
        total_elements *= dim

    alignment = 2 * 1024 * 1024
    aligned_elements = total_elements + (alignment // dtype.itemsize)
    temp = torch.zeros(aligned_elements, dtype=dtype, pin_memory=pin_memory, device=device)

    ptr = temp.data_ptr()
    aligned_ptr = (ptr + alignment - 1) & ~(alignment - 1)
    offset = (aligned_ptr - ptr) // dtype.itemsize
    return temp[offset:offset + total_elements].view(shape)