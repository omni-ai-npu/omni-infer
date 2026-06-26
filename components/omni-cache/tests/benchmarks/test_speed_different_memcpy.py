# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import ctypes, torch, logging, time
logging.basicConfig(level=logging.WARNING, format='%(message)s')

libascendcl = ctypes.CDLL('libascendcl.so')
ACL_MEMCPY_HOST_TO_DEVICE = 1
aclrtStream = ctypes.c_void_p

# --- AscendCL API Setup ---
aclrtMemcpyAsync = libascendcl.aclrtMemcpyAsync
aclrtMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, aclrtStream]
aclrtMemcpyAsync.restype = ctypes.c_int

aclrtCreateStream = libascendcl.aclrtCreateStream
aclrtCreateStream.argtypes = [ctypes.POINTER(aclrtStream)]
aclrtCreateStream.restype = ctypes.c_int

aclrtSynchronizeStream = libascendcl.aclrtSynchronizeStream
aclrtSynchronizeStream.argtypes = [aclrtStream]
aclrtSynchronizeStream.restype = ctypes.c_int

aclrtDestroyStream = libascendcl.aclrtDestroyStream
aclrtDestroyStream.argtypes = [aclrtStream]
aclrtDestroyStream.restype = ctypes.c_int

aclrtMemcpyBatch = libascendcl.aclrtMemcpyBatch

try:
    aclrtMemcpyBatchAsync = libascendcl.aclrtMemcpyBatchAsync
    FLAG_WITH_ACLRT_MEMCPYBATCHASYNC = True
except AttributeError:
    logging.warning("aclrtMemcpyBatchAsync not found in libascendcl.so, please check your AscendCL version.")
    FLAG_WITH_ACLRT_MEMCPYBATCHASYNC = False

aclrtMemcpyBatch.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # dst
    ctypes.POINTER(ctypes.c_size_t),  # dst_max
    ctypes.POINTER(ctypes.c_void_p),  # src
    ctypes.POINTER(ctypes.c_size_t),  # count
    ctypes.c_size_t,                  # batch_count
    ctypes.c_void_p,                  # attrs (can be None)
    ctypes.POINTER(ctypes.c_size_t),  # attrsIndex (can be None)
    ctypes.c_int,                     # kind
    ctypes.POINTER(ctypes.c_size_t)   # fails
]
aclrtMemcpyBatch.restype = ctypes.c_int

class AscendCLStream:
    def __init__(self):
        self._stream = aclrtStream()
        aclrtCreateStream(ctypes.byref(self._stream))
    def memcpy_async(self, dst, dst_max, src, count, kind):
        aclrtMemcpyAsync(dst, dst_max, src, count, kind, self._stream)
    def memcpy_batch(self, dst, dst_max, src, count, batch_count, kind):
        attrs = ctypes.c_void_p(0)
        attrsIndex = (ctypes.c_size_t * 1)()
        fails = (ctypes.c_size_t * 1)()
        return aclrtMemcpyBatch(
            dst, dst_max, src, count, batch_count,
            attrs, attrsIndex, kind, fails
        )
    def memcpy_batch_async(self, dst, dst_max, src, count, batch_count, kind):
        if not FLAG_WITH_ACLRT_MEMCPYBATCHASYNC:
            return
        attrs = ctypes.c_void_p(0)
        attrsIndex = (ctypes.c_size_t * 1)()
        fails = (ctypes.c_size_t * 1)()
        return aclrtMemcpyBatchAsync(
            dst, dst_max, src, count, batch_count,
            attrs, attrsIndex, kind, fails, self._stream
        )
    def sync(self):
        aclrtSynchronizeStream(self._stream)
    def __del__(self):
        aclrtDestroyStream(self._stream)

def batch_layer_copy_to_npu_async(num_copies=10):
    ascend_cl_stream = AscendCLStream()
    cpu_ts = []
    npu_ts = []
    for i in range(num_copies):
        cpu_t = torch.ones([4, 1, 100], dtype=torch.int32)
        npu_t = torch.empty_like(cpu_t, device='npu:0')
        cpu_ts.append(cpu_t)
        npu_ts.append(npu_t)
    start = time.time()
    for i in range(num_copies):
        ascend_cl_stream.memcpy_async(
            ctypes.c_void_p(npu_ts[i].data_ptr()),
            npu_ts[i].nbytes,
            ctypes.c_void_p(cpu_ts[i].data_ptr()),
            cpu_ts[i].nbytes,
            ACL_MEMCPY_HOST_TO_DEVICE
        )
    ascend_cl_stream.sync()
    async_time = (time.time()-start)*1000
    logging.warning(f"AsyncCopy ({num_copies} calls) took {async_time:.2f} ms")
    return async_time

def batch_layer_copy_to_npu_async_consecutive(num_copies=10):
    ascend_cl_stream = AscendCLStream()
    # allocate a big CPU tensor and NPU tensor
    cpu_big = torch.ones([num_copies, 4, 1, 100], dtype=torch.int32)
    npu_big = torch.empty([num_copies, 4, 1, 100], dtype=torch.int32, device='npu:0')
    cpu_ts = [cpu_big[i] for i in range(num_copies)]
    npu_ts = [npu_big[i] for i in range(num_copies)]
    tensor_size = cpu_ts[0].nbytes
    # merge consecutive tensors to reduce memcpy calls
    start = time.time()
    batch_start = 0
    num_called = 0
    while batch_start < num_copies:
        batch_end = batch_start + 1
        prev_cpu_addr = cpu_ts[batch_start].data_ptr()
        prev_npu_addr = npu_ts[batch_start].data_ptr()
        while batch_end < num_copies:
            curr_cpu_addr = cpu_ts[batch_end].data_ptr()
            curr_npu_addr = npu_ts[batch_end].data_ptr()
            if (curr_cpu_addr == prev_cpu_addr + tensor_size and
                curr_npu_addr == prev_npu_addr + tensor_size):
                prev_cpu_addr = curr_cpu_addr
                prev_npu_addr = curr_npu_addr
                batch_end += 1
            else:
                break
        # merge group [batch_start, batch_end)
        device_mem = ctypes.c_void_p(npu_ts[batch_start].data_ptr())
        device_max = tensor_size * (batch_end - batch_start)
        host_mem = ctypes.c_void_p(cpu_ts[batch_start].data_ptr())
        host_sizes = tensor_size * (batch_end - batch_start)
        ascend_cl_stream.memcpy_async(
            device_mem,
            device_max,
            host_mem,
            host_sizes,
            ACL_MEMCPY_HOST_TO_DEVICE
        )
        num_called += 1
        batch_start = batch_end
    ascend_cl_stream.sync()
    async_time = (time.time()-start)*1000
    logging.warning(f"AsyncConsecutiveCopy ({num_copies} tensors, merged to {num_called} times) took {async_time:.2f} ms")
    return async_time

def batch_layer_copy_to_npu_batch(num_copies=10):
    ascend_cl_stream = AscendCLStream()
    cpu_ts = []
    npu_ts = []
    for i in range(num_copies):
        cpu_t = torch.ones([4, 1, 100], dtype=torch.int32)
        npu_t = torch.empty_like(cpu_t, device='npu:0')
        cpu_ts.append(cpu_t)
        npu_ts.append(npu_t)
    # construct parameter arrays
    device_mem = (ctypes.c_void_p * num_copies)()
    device_max = (ctypes.c_size_t * num_copies)()
    host_mem = (ctypes.c_void_p * num_copies)()
    host_sizes = (ctypes.c_size_t * num_copies)()
    for i in range(num_copies):
        device_mem[i] = ctypes.c_void_p(npu_ts[i].data_ptr())
        device_max[i] = npu_ts[i].nbytes
        host_mem[i] = ctypes.c_void_p(cpu_ts[i].data_ptr())
        host_sizes[i] = cpu_ts[i].nbytes
    # call batch memcpy
    start = time.time()
    ascend_cl_stream.memcpy_batch(
        device_mem,
        device_max,
        host_mem,
        host_sizes,
        num_copies,
        ACL_MEMCPY_HOST_TO_DEVICE
    )
    ascend_cl_stream.sync()
    batch_time = (time.time()-start)*1000
    logging.warning(f"MemcpyBatch ({num_copies} ops) took {batch_time:.2f} ms")
    return batch_time

def batch_layer_copy_to_npu_batch_async(num_copies=10):
    ascend_cl_stream = AscendCLStream()
    cpu_ts = []
    npu_ts = []
    for i in range(num_copies):
        cpu_t = torch.ones([4, 1, 100], dtype=torch.int32)
        npu_t = torch.empty_like(cpu_t, device='npu:0')
        cpu_ts.append(cpu_t)
        npu_ts.append(npu_t)
    # construct parameter arrays
    device_mem = (ctypes.c_void_p * num_copies)()
    device_max = (ctypes.c_size_t * num_copies)()
    host_mem = (ctypes.c_void_p * num_copies)()
    host_sizes = (ctypes.c_size_t * num_copies)()
    for i in range(num_copies):
        device_mem[i] = ctypes.c_void_p(npu_ts[i].data_ptr())
        device_max[i] = npu_ts[i].nbytes
        host_mem[i] = ctypes.c_void_p(cpu_ts[i].data_ptr())
        host_sizes[i] = cpu_ts[i].nbytes
    # async call batch memcpy
    start = time.time()
    ascend_cl_stream.memcpy_batch_async(
        device_mem,
        device_max,
        host_mem,
        host_sizes,
        num_copies,
        ACL_MEMCPY_HOST_TO_DEVICE
    )
    ascend_cl_stream.sync()
    batch_time = (time.time()-start)*1000
    logging.warning(f"MemcpyBatchAsync ({num_copies} ops) took {batch_time:.2f} ms")
    return batch_time

if __name__ == "__main__":
    num_copies = 1000
    t_async = batch_layer_copy_to_npu_async(num_copies)
    t_consecutive = batch_layer_copy_to_npu_async_consecutive(num_copies)
    t_batch = batch_layer_copy_to_npu_batch(num_copies)
    if FLAG_WITH_ACLRT_MEMCPYBATCHASYNC:
        t_batch_async = batch_layer_copy_to_npu_batch_async(num_copies)
    print(f"\nAsync total: {t_async:.2f} ms")
    print(f"Async consecutive total: {t_consecutive:.2f} ms")
    print(f"Batch total: {t_batch:.2f} ms")
    if FLAG_WITH_ACLRT_MEMCPYBATCHASYNC:
        print(f"Batch async total: {t_batch_async:.2f} ms")