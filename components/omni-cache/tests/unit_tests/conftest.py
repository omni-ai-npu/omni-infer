# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import sys
import types
import logging
import hashlib
import importlib.util
import os
from dataclasses import dataclass, field
from enum import Enum
import inspect

import pytest

# ========== Critical: Disable torch extension auto-loading ONLY in test envs without NPU ==========
os.environ.setdefault("ENABLE_HOST_MAPPING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROCESS_METHOD", "spawn")
os.environ.setdefault("VLLM_PLUGINS", "__none__")

try:
    import transformers.utils.import_utils as _transformers_import_utils

    _transformers_import_utils._torchvision_available = False
except Exception:
    pass


# ========== COMPLETE TRITON MOCK ==========
# This must be done BEFORE any torch/vllm import
def _create_complete_triton_mock():
    """Create a complete mock of the triton package structure."""
    import inspect

    # Core module
    triton_module = types.ModuleType("triton")
    triton_module.__path__ = []
    triton_module.__version__ = "3.0.0"

    # Create a proper ModuleSpec for triton (required by importlib.util.find_spec)
    from importlib.machinery import ModuleSpec
    triton_spec = ModuleSpec("triton", loader=None, is_package=True)
    triton_module.__spec__ = triton_spec

    triton_module.cdiv = lambda x, y: (x + y - 1) // y

    # Config must be a class that can be subclassed
    class _TritonConfig:
        def __init__(self, *args, **kwargs):
            pass
    triton_module.Config = _TritonConfig
    triton_module.jit = lambda *args, **kwargs: (lambda func: func)

    # triton.language
    triton_lang = types.ModuleType("triton.language")
    triton_lang.__path__ = []
    triton_module.language = triton_lang

    # triton.language.core (needed by torch._inductor.runtime.triton_compat)
    triton_lang_core = types.ModuleType("triton.language.core")
    triton_lang_core.view = lambda *args, _semantic=None, **kwargs: None
    triton_lang.core = triton_lang_core

    # triton.language.math
    triton_lang_math = types.ModuleType("triton.language.math")
    for func_name in ["rsqrt", "sqrt", "sin", "cos", "exp", "log", "abs", "pow", "tanh", "sigmoid"]:
        setattr(triton_lang_math, func_name, lambda *args, **kwargs: None)
    triton_lang.math = triton_lang_math

    # triton.language.extra
    triton_lang_extra = types.ModuleType("triton.language.extra")
    triton_lang_extra.__path__ = []
    triton_lang.extra = triton_lang_extra

    # triton.language.extra.cuda
    triton_lang_extra_cuda = types.ModuleType("triton.language.extra.cuda")
    triton_lang_extra_cuda.libdevice = types.SimpleNamespace()
    for func_name in ["rsqrt", "sqrt", "sin", "cos", "exp", "log", "abs", "div_rn", "fma"]:
        setattr(triton_lang_extra_cuda.libdevice, func_name, lambda *args, **kwargs: None)
    triton_lang_extra.cuda = triton_lang_extra_cuda

    # triton.language.extra.libdevice
    triton_lang_extra_libdevice = types.ModuleType("triton.language.extra.libdevice")
    for func_name in ["rsqrt", "sqrt", "sin", "cos", "exp", "log", "abs", "div_rn", "fma"]:
        setattr(triton_lang_extra_libdevice, func_name, lambda *args, **kwargs: None)
    triton_lang_extra.libdevice = triton_lang_extra_libdevice

    # triton.language attributes
    triton_dtype = type("dtype", (), {"__init__": lambda self, *args, **kwargs: None})
    triton_lang.dtype = triton_dtype
    triton_lang.tensor = type("tensor", (), {})  # Required by torch._inductor.runtime.triton_helpers
    triton_lang.constexpr = type("constexpr", (), {"__init__": lambda self, x: None})
    triton_lang.dtype = type("dtype", (), {})  # Required by torch._dynamo.utils
    for attr in ["program_id", "arange", "load", "store", "max", "min", "where", "cumsum", "sum", "exp", "log", "sqrt", "rsqrt", "abs", "sin", "cos"]:
        setattr(triton_lang, attr, lambda *args, **kwargs: None)
    for attr in ["float16", "float32", "int32", "int64", "pointer_type", "block_type"]:
        setattr(triton_lang, attr, attr)

    # triton.compiler
    triton_compiler = types.ModuleType("triton.compiler")
    triton_compiler.__path__ = []
    triton_compiler.CompiledKernel = type("CompiledKernel", (), {})  # Must be on triton.compiler directly
    triton_module.compiler = triton_compiler

    # triton.compiler.compiler
    triton_compiler_compiler = types.ModuleType("triton.compiler.compiler")
    triton_compiler_compiler.CompiledKernel = triton_compiler.CompiledKernel
    triton_compiler_compiler.triton_key = lambda *args, **kwargs: "fake_key"
    for name in ["compile", "ASTFunction", "ASTModule", "ASTMutability", "ASTTensor", "ASTType"]:
        setattr(triton_compiler_compiler, name, type(name, (), {}))
    triton_compiler.compiler = triton_compiler_compiler

    # triton.backends
    triton_backends = types.ModuleType("triton.backends")
    triton_backends.__path__ = []
    triton_module.backends = triton_backends

    # triton.backends.compiler
    triton_backends_compiler = types.ModuleType("triton.backends.compiler")
    triton_backends.compiler = triton_backends_compiler

    # triton.runtime
    triton_runtime = types.ModuleType("triton.runtime")
    triton_runtime.__path__ = []
    triton_module.runtime = triton_runtime

    # triton.runtime.cache
    triton_runtime_cache = types.ModuleType("triton.runtime.cache")
    triton_runtime_cache.triton_key = lambda *args, **kwargs: "fake_key"
    triton_runtime.cache = triton_runtime_cache

    # triton.runtime.autotuner
    triton_runtime_autotuner = types.ModuleType("triton.runtime.autotuner")
    triton_runtime_autotuner.OutOfResources = type("OutOfResources", (Exception,), {})
    triton_runtime.autotuner = triton_runtime_autotuner

    # triton.runtime.jit
    triton_runtime_jit = types.ModuleType("triton.runtime.jit")
    triton_runtime_jit.KernelInterface = type("KernelInterface", (), {})
    for name in ["LaunchedKernel", "JITFunction", "KernelMetadata"]:
        setattr(triton_runtime_jit, name, type(name, (), {}))
    triton_runtime.jit = triton_runtime_jit

    # Register all modules
    sys.modules["triton"] = triton_module
    sys.modules["triton.language"] = triton_lang
    sys.modules["triton.language.core"] = triton_lang_core
    sys.modules["triton.language.math"] = triton_lang_math
    sys.modules["triton.language.extra"] = triton_lang_extra
    sys.modules["triton.language.extra.cuda"] = triton_lang_extra_cuda
    sys.modules["triton.language.extra.libdevice"] = triton_lang_extra_libdevice
    sys.modules["triton.compiler"] = triton_compiler
    sys.modules["triton.compiler.compiler"] = triton_compiler_compiler
    sys.modules["triton.backends"] = triton_backends
    sys.modules["triton.backends.compiler"] = triton_backends_compiler
    sys.modules["triton.runtime"] = triton_runtime
    sys.modules["triton.runtime.cache"] = triton_runtime_cache
    sys.modules["triton.runtime.autotuner"] = triton_runtime_autotuner
    sys.modules["triton.runtime.jit"] = triton_runtime_jit

_create_complete_triton_mock()


# ========== TorchAir Mock ==========
# Required by omni_npu.platform
def _create_torchair_mock():
    """Create mock for torchair package."""
    torchair_module = types.ModuleType("torchair")
    torchair_module.__path__ = []
    torchair_module.lib = types.ModuleType("torchair.lib")
    torchair_module.lib.__path__ = []
    sys.modules["torchair"] = torchair_module
    sys.modules["torchair.lib"] = torchair_module.lib

_create_torchair_mock()

def _check_npu_available():
    try:
        import torch
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
    except Exception:
        pass
    return False

_NPU_AVAILABLE = _check_npu_available()


# ========== AscendCL Mock ==========
# Mock the AscendCL library availability before any ascend modules are imported
def _create_ascend_acl_mock():
    """Create mock for ascend acl modules to avoid libascendcl.so dependency."""
    # Pre-create the memcopy module with _ACL_AVAILABLE = True before the real module is loaded
    import ctypes

    # Create the memcopy module mock
    memcopy_module = types.ModuleType("omni_cache.cache.device_backend.ascend.memcopy")
    memcopy_module.__file__ = "<mock>"
    memcopy_module._ACL_AVAILABLE = True
    memcopy_module._ensure_acl_available = lambda: None
    memcopy_module.libascendcl = None

    # Define constants
    memcopy_module.ACL_MEM_LOCATION_TYPE_HOST = 0
    memcopy_module.ACL_MEM_LOCATION_TYPE_DEVICE = 1
    memcopy_module.ACL_MEMCPY_HOST_TO_HOST = 0
    memcopy_module.ACL_MEMCPY_HOST_TO_DEVICE = 1
    memcopy_module.ACL_MEMCPY_DEVICE_TO_HOST = 2
    memcopy_module.ACL_MEMCPY_DEVICE_TO_DEVICE = 3
    memcopy_module.ACL_MEMCPY_DEFAULT = 4
    memcopy_module.ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE = 5
    memcopy_module.ACL_MEMCPY_INNER_DEVICE_TO_DEVICE = 6
    memcopy_module.ACL_MEMCPY_INTER_DEVICE_TO_DEVICE = 7

    # Define ctypes structures
    class aclrtMemLocation(ctypes.Structure):
        _fields_ = [
            ('id', ctypes.c_uint32),
            ('type', ctypes.c_uint32)
        ]
    memcopy_module.aclrtMemLocation = aclrtMemLocation

    class aclrtMemcpyBatchAttr(ctypes.Structure):
        _fields_ = [
            ('dstLoc', aclrtMemLocation),
            ('srcLoc', aclrtMemLocation),
            ('rsv', ctypes.c_uint8 * 16)
        ]
    memcopy_module.aclrtMemcpyBatchAttr = aclrtMemcpyBatchAttr

    memcopy_module.aclrtStream = ctypes.c_void_p
    memcopy_module.aclrtMemcpyBatch = lambda *args, **kwargs: 0
    memcopy_module.aclrtMemcpyBatchAsync = lambda *args, **kwargs: 0
    memcopy_module.aclrtCreateStream = lambda *args, **kwargs: 0
    memcopy_module.aclrtSynchronizeStream = lambda *args, **kwargs: 0
    memcopy_module.aclrtDestroyStream = lambda *args, **kwargs: 0
    memcopy_module.aclrtMemcpyAsync = lambda *args, **kwargs: 0

    # Register the memcopy module
    sys.modules["omni_cache.cache.device_backend.ascend.memcopy"] = memcopy_module

    # Create the ascend module with proper structure
    ascend_module = types.ModuleType("omni_cache.cache.device_backend.ascend")
    # Set __path__ to the real package directory so Python can find submodules
    # (e.g., tensor_register) that are NOT mocked and should use real source code.
    ascend_module.__path__ = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "omni_cache",
                "cache",
                "device_backend",
                "ascend",
            )
        )
    ]
    ascend_module.aclrtMemLocation = aclrtMemLocation
    ascend_module.aclrtMemcpyBatchAttr = aclrtMemcpyBatchAttr
    ascend_module.aclrtStream = ctypes.c_void_p
    ascend_module._ACL_AVAILABLE = True
    ascend_module.ACL_MEM_LOCATION_TYPE_HOST = 0
    ascend_module.ACL_MEM_LOCATION_TYPE_DEVICE = 1
    ascend_module.ACL_MEMCPY_HOST_TO_HOST = 0
    ascend_module.ACL_MEMCPY_HOST_TO_DEVICE = 1
    ascend_module.ACL_MEMCPY_DEVICE_TO_HOST = 2
    ascend_module.ACL_MEMCPY_DEVICE_TO_DEVICE = 3
    ascend_module.ACL_MEMCPY_DEFAULT = 4
    ascend_module.ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE = 5
    ascend_module.ACL_MEMCPY_INNER_DEVICE_TO_DEVICE = 6
    ascend_module.ACL_MEMCPY_INTER_DEVICE_TO_DEVICE = 7
    ascend_module.aclrtMemcpyBatch = lambda *args, **kwargs: 0
    ascend_module.aclrtMemcpyBatchAsync = lambda *args, **kwargs: 0
    ascend_module.aclrtCreateStream = lambda *args, **kwargs: 0
    ascend_module.aclrtSynchronizeStream = lambda *args, **kwargs: 0
    ascend_module.aclrtDestroyStream = lambda *args, **kwargs: 0
    ascend_module.aclrtMemcpyAsync = lambda *args, **kwargs: 0
    sys.modules["omni_cache.cache.device_backend.ascend"] = ascend_module

    # Now create the streams module that depends on memcopy
    streams_module = types.ModuleType("omni_cache.cache.device_backend.ascend.streams")
    streams_module.__file__ = "<mock>"

    # Add the low-level functions that tests try to patch
    streams_module.aclrtCreateStream = lambda *args, **kwargs: 0
    streams_module.aclrtSynchronizeStream = lambda *args, **kwargs: 0
    streams_module.aclrtDestroyStream = lambda *args, **kwargs: 0
    streams_module.aclrtMemcpyAsync = lambda *args, **kwargs: 0
    streams_module.aclrtMemcpyBatch = lambda *args, **kwargs: 0
    streams_module.aclrtMemcpyBatchAsync = lambda *args, **kwargs: 0
    streams_module._ensure_acl_available = lambda: None

    class AscendCLStream:
        def __init__(self):
            self._stream = ctypes.c_void_p()
        def create(self):
            streams_module._ensure_acl_available()
            rc = streams_module.aclrtCreateStream(ctypes.byref(self._stream))
            if rc != 0:
                raise RuntimeError(f"aclrtCreateStream failed with error code: {rc}")
            return rc
        def sync(self):
            streams_module._ensure_acl_available()
            rc = streams_module.aclrtSynchronizeStream(self._stream)
            if rc != 0:
                raise RuntimeError(f"aclrtSynchronizeStream failed with error code: {rc}")
            return rc
        def memcpy_async(self, dst, dst_max, src, count, kind):
            streams_module._ensure_acl_available()
            rc = streams_module.aclrtMemcpyAsync(dst, dst_max, src, count, kind, self._stream)
            if rc != 0:
                raise RuntimeError(f"aclrtMemcpyAsync failed with error code: {rc}")
            return rc
        def memcpy_batch(self, dst, dst_max, src, count, batch_count, attrs, attrsIndex, kind, fails):
            streams_module._ensure_acl_available()
            rc = streams_module.aclrtMemcpyBatch(dst, dst_max, src, count, batch_count, attrs, attrsIndex, kind, fails)
            if rc != 0:
                raise RuntimeError(f"aclrtMemcpyBatch failed with error code: {rc}")
            return rc
        def memcpy_batch_async(self, dst, dst_max, src, count, batch_count, attrs, attrsIndex, kind, fails):
            streams_module._ensure_acl_available()
            rc = streams_module.aclrtMemcpyBatchAsync(dst, dst_max, src, count, batch_count, attrs, attrsIndex, kind, fails, self._stream)
            if rc != 0:
                raise RuntimeError(f"aclrtMemcpyBatch failed with error code: {rc}")
            return rc
        @property
        def stream_ptr(self):
            return self._stream
        def __del__(self):
            try:
                streams_module.aclrtDestroyStream(self._stream)
            except Exception:
                pass

    streams_module.AscendCLStream = AscendCLStream
    sys.modules["omni_cache.cache.device_backend.ascend.streams"] = streams_module
    ascend_module.AscendCLStream = AscendCLStream

    # Add ops submodule mock
    ops_module = types.ModuleType("omni_cache.cache.device_backend.ascend.ops")
    ops_module.__path__ = []
    sys.modules["omni_cache.cache.device_backend.ascend.ops"] = ops_module
    ascend_module.ops = ops_module

    # Mock tensor_register_lib (it requires libascendcl.so which is not available)
    trl_module = types.ModuleType("omni_cache.cache.device_backend.ascend.tensor_register_lib")
    trl_module.__path__ = []
    trl_module.zero_copy_npu = types.SimpleNamespace()
    sys.modules["omni_cache.cache.device_backend.ascend.tensor_register_lib"] = trl_module
    ascend_module.tensor_register_lib = trl_module

    # tensor_register is NOT mocked — it should use real source code.
    # The real module handles the no-NPU case gracefully:
    # _ZERO_COPY_AVAILABLE=False, _lib=_MissingTensorRegisterLib().
    # Import the real NPUTensorRegister and expose it on the mock ascend module
    # so that code doing `from omni_cache.cache.device_backend.ascend import NPUTensorRegister`
    # gets the real class.
    import importlib
    tr_real = importlib.import_module("omni_cache.cache.device_backend.ascend.tensor_register")
    ascend_module.NPUTensorRegister = tr_real.NPUTensorRegister

    # Set up parent module references only if parents exist
    if "omni_cache.cache.device_backend" in sys.modules:
        sys.modules["omni_cache.cache.device_backend"].ascend = ascend_module
    if "omni_cache.cache" in sys.modules and "omni_cache.cache.device_backend" in sys.modules:
        sys.modules["omni_cache.cache"].device_backend = sys.modules["omni_cache.cache.device_backend"]

# Mock AscendCL library only if NPU is not available
if not _NPU_AVAILABLE:
    _create_ascend_acl_mock()


def _module_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except Exception:
        return True


class _AutoStubModule(types.ModuleType):
    def __getattr__(self, name: str):
        placeholder = type(name, (), {})
        setattr(self, name, placeholder)
        return placeholder


def _ensure_module(name: str, modules: dict[str, types.ModuleType], module: types.ModuleType | None = None):
    existing = modules.get(name) or sys.modules.get(name)
    if existing is not None:
        return existing

    module = module or _AutoStubModule(name)
    if "." not in name:
        module.__path__ = []
    modules[name] = module

    parts = name.split(".")
    for idx in range(1, len(parts)):
        pkg_name = ".".join(parts[:idx])
        pkg = modules.get(pkg_name) or sys.modules.get(pkg_name)
        if pkg is None:
            if pkg_name.startswith("omni_cache"):
                try:
                    __import__(pkg_name)
                    pkg = sys.modules.get(pkg_name)
                except Exception:
                    pkg = None
            if pkg is not None:
                pass
            else:
                pkg = _AutoStubModule(pkg_name)
                repo_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..")
                )
                real_pkg_paths = {
                    "omni_cache": os.path.join(repo_root, "omni_cache"),
                    "omni_cache.cache": os.path.join(
                        repo_root, "omni_cache", "cache"
                    ),
                }
                pkg.__path__ = (
                    [real_pkg_paths[pkg_name]] if pkg_name in real_pkg_paths else []
                )
                modules[pkg_name] = pkg
        child_name = parts[idx]
        setattr(pkg, child_name, modules.get(".".join(parts[:idx + 1]), module if idx == len(parts) - 1 else pkg))

    return module


def _install_missing_runtime_stubs():
    modules: dict[str, types.ModuleType] = {}

    # vLLM is a real dependency in this workspace. Do not install fake vllm.*
    # modules here: tests that need Pangu-specific vLLM classes should apply the
    # real omni-npu patch classes, and tests that need runtime-only shims should
    # install those narrow shims locally.

    if _module_missing("llm_datadist"):
        llm_types = _ensure_module("llm_datadist.v2.llm_types", modules, types.ModuleType("llm_datadist.v2.llm_types"))
        llm_types.Cache = type("Cache", (), {})

        @dataclass
        class CacheDesc:
            num_tensors: int = 0
            shape: tuple = ()
            data_type: object = None

        @dataclass
        class BlocksCacheKey:
            prompt_cluster_id: int = 0
            model_id: int = 0

        llm_types.CacheDesc = CacheDesc
        llm_types.BlocksCacheKey = BlocksCacheKey

    # Check if real torch_npu and NPU are available
    # This must be done BEFORE creating any stubs
    real_npu_available = False
    try:
        import torch
        if hasattr(torch, 'npu') and callable(getattr(torch.npu, 'is_available', None)) and torch.npu.is_available():
            real_npu_available = True
    except Exception:
        pass

    # Only install torch_npu stub if real NPU is not available
    if not real_npu_available:
        # Dummy classes for NPU stub
        class _DummyEvent:
            def __init__(self, *args, **kwargs):
                pass

            def record(self, *args, **kwargs):
                return None

            def synchronize(self):
                return None

            def wait(self):
                return None

        class _DummyStream:
            def __init__(self, *args, **kwargs):
                pass

            def synchronize(self):
                return None

            def wait_event(self, *args, **kwargs):
                return None

        class _DummyStreamContext:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self.stream

            def __exit__(self, exc_type, exc, tb):
                return False

        class _DummyNPUPluggableAllocator:
            def __init__(self, *args, **kwargs):
                self._allocator = object()

        class _DummyMemPool:
            def __init__(self, *args, **kwargs):
                pass

        class _DummyMemPoolContext:
            def __init__(self, mem_pool):
                self.mem_pool = mem_pool

            def __enter__(self):
                return self.mem_pool

            def __exit__(self, exc_type, exc, tb):
                return False

        class _DummyNPUGraph:
            def __init__(self, *args, **kwargs):
                pass

            def capture_begin(self, *args, **kwargs):
                return None

            def capture_end(self, *args, **kwargs):
                return None

            def replay(self, *args, **kwargs):
                return None

        npu_memory_ns = types.SimpleNamespace(
            NPUPluggableAllocator=_DummyNPUPluggableAllocator,
            MemPool=_DummyMemPool,
            use_mem_pool=lambda mem_pool: _DummyMemPoolContext(mem_pool),
        )
        npu_config_ns = types.SimpleNamespace(allow_internal_format=False)

        npu_ns = types.SimpleNamespace(
            set_device=lambda *args, **kwargs: None,
            is_available=lambda: False,
            Event=_DummyEvent,
            Stream=_DummyStream,
            NPUGraph=_DummyNPUGraph,
            stream=lambda stream: _DummyStreamContext(stream),
            current_stream=lambda: _DummyStream(),
            mem_get_info=lambda: (0, 0),
            memory=npu_memory_ns,
            config=npu_config_ns,
        )

        torch_npu_module = types.ModuleType("torch_npu")
        torch_npu_module.npu = npu_ns
        torch_npu_module.npu_gather_selection_kv_cache = lambda **kwargs: None

        # Add proper ModuleSpec (required by importlib.util.find_spec)
        from importlib.machinery import ModuleSpec
        torch_npu_spec = ModuleSpec("torch_npu", loader=None, is_package=True)
        torch_npu_module.__spec__ = torch_npu_spec
        torch_npu_module.__path__ = []

        sys.modules["torch_npu"] = torch_npu_module
        modules["torch_npu"] = torch_npu_module

        # Also set torch.npu
        try:
            import torch
            torch.npu = npu_ns
        except Exception:
            pass

    if _module_missing("zero_copy_npu"):
        zero_copy_module = _ensure_module("zero_copy_npu", modules, types.ModuleType("zero_copy_npu"))
        zero_copy_module.TensorRegistrar = type("TensorRegistrar", (), {})
        zero_copy_module.register_tensor = lambda host_tensor, *args, **kwargs: (host_tensor, host_tensor)

    if _module_missing("numba"):
        numba_module = _ensure_module("numba", modules, types.ModuleType("numba"))
        numba_module.njit = lambda *args, **kwargs: (lambda func: func)
        numba_module.prange = range

    ascend_acl_module = _ensure_module("omni_cache.cache.ascend_acl", modules, types.ModuleType("omni_cache.cache.ascend_acl"))
    ascend_acl_module.ACL_MEMCPY_HOST_TO_HOST = 0
    ascend_acl_module.ACL_MEMCPY_HOST_TO_DEVICE = 1
    ascend_acl_module.ACL_MEMCPY_DEVICE_TO_HOST = 2
    ascend_acl_module.ACL_MEMCPY_DEVICE_TO_DEVICE = 3
    ascend_acl_module.ACL_MEMCPY_DEFAULT = 4
    ascend_acl_module.ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE = 5
    ascend_acl_module.ACL_MEMCPY_INNER_DEVICE_TO_DEVICE = 6
    ascend_acl_module.ACL_MEMCPY_INTER_DEVICE_TO_DEVICE = 7

    class AscendCLStream:
        def synchronize(self):
            return None

        def memcpy_async(self, *args, **kwargs):
            return 0

        def memcpy_batch(self, *args, **kwargs):
            return 0

        def memcpy_batch_async(self, *args, **kwargs):
            return 0

        def __del__(self):
            return None

    ascend_acl_module.AscendCLStream = AscendCLStream

    for name, module in modules.items():
        sys.modules.setdefault(name, module)


_install_missing_runtime_stubs()


# ========== pytest hook to add module-level attributes for patching ==========
def pytest_configure(config):
    """Add module-level attributes that tests try to patch.

    Some modules import dependencies inside functions (lazy imports).
    Tests try to patch these at module level, which fails because
    the attributes don't exist until the function runs.
    We add stub attributes here, after all modules are loaded.
    """
    import types
    from vllm.logger import init_logger

    # 1. omni_cache.gather_selection.core.gather_selection stubs
    try:
        import omni_cache.gather_selection.core.gather_selection as gs_mod
        if not hasattr(gs_mod, 'init_logger'):
            gs_mod.init_logger = init_logger
        if not hasattr(gs_mod, 'torch_npu'):
            gs_mod.torch_npu = sys.modules.get("torch_npu", types.SimpleNamespace())
        if not hasattr(gs_mod, 'torch'):
            gs_mod.torch = sys.modules.get("torch", types.SimpleNamespace())
    except ImportError:
        pass

    # 2. omni_cache.gather_selection.core.buffers stubs
    try:
        import omni_cache.gather_selection.core.buffers as buf_mod
        if not hasattr(buf_mod, 'torch'):
            buf_mod.torch = sys.modules.get("torch", types.SimpleNamespace())
    except ImportError:
        pass

    # 3. omni_cache.attention.metadata.block_table stubs
    try:
        import omni_cache.attention.metadata.block_table as bt_mod
        if not hasattr(bt_mod, 'torch_to_numpy_zero_copy'):
            bt_mod.torch_to_numpy_zero_copy = lambda *args, **kwargs: None
    except ImportError:
        pass

    # 4. vllm.forward_context stubs
    try:
        import vllm.forward_context as fc_mod
        if not hasattr(fc_mod, 'get_forward_context'):
            fc_mod.get_forward_context = lambda: None
    except ImportError:
        fc_mod = types.ModuleType("vllm.forward_context")
        fc_mod.get_forward_context = lambda: None
        sys.modules["vllm.forward_context"] = fc_mod

    # 5. omni_cache.cache.kv_cache_interface stubs
    try:
        import omni_cache.cache.kv_cache_interface as kvi_mod
        from dataclasses import dataclass
        from enum import Enum

        if not hasattr(kvi_mod, 'KVCacheTensor'):
            @dataclass
            class KVCacheTensor:
                size: int = 0
            kvi_mod.KVCacheTensor = KVCacheTensor
        if not hasattr(kvi_mod, 'AttentionType'):
            class AttentionType(Enum):
                DECODER = "decoder"
            kvi_mod.AttentionType = AttentionType
        if not hasattr(kvi_mod, 'BlockTable'):
            class BlockTable:
                pass
            kvi_mod.BlockTable = BlockTable
    except ImportError:
        # Create alias module if it doesn't exist
        try:
            import omni_cache.attention.kv_cache_interface as real_kvi
            sys.modules["omni_cache.cache.kv_cache_interface"] = real_kvi
        except ImportError:
            pass

    # 6. omni_cache.cache submodule attributes for patching
    try:
        import omni_cache.cache as cache_mod
        # Add device_backend submodule reference
        if not hasattr(cache_mod, 'device_backend'):
            try:
                import omni_cache.cache.device_backend as db_mod
                cache_mod.device_backend = db_mod
            except ImportError:
                pass
        # Add functions that tests try to patch
        if not hasattr(cache_mod, 'check_omni_attn_cmd_arg'):
            cache_mod.check_omni_attn_cmd_arg = lambda x: False
        if not hasattr(cache_mod, '_apply_sink_config'):
            cache_mod._apply_sink_config = lambda *args, **kwargs: None
        if not hasattr(cache_mod, '_apply_runtime_config'):
            cache_mod._apply_runtime_config = lambda *args, **kwargs: None
        if not hasattr(cache_mod, '_apply_recent_config'):
            cache_mod._apply_recent_config = lambda *args, **kwargs: None
        if not hasattr(cache_mod, '_apply_pattern_config'):
            cache_mod._apply_pattern_config = lambda *args, **kwargs: None
        if not hasattr(cache_mod, '_apply_beta_config'):
            cache_mod._apply_beta_config = lambda *args, **kwargs: None
        if not hasattr(cache_mod, '_apply_cache_manager_runtime_patch'):
            cache_mod._apply_cache_manager_runtime_patch = lambda *args, **kwargs: None
        if not hasattr(cache_mod, '_apply_attention_runtime_patch'):
            cache_mod._apply_attention_runtime_patch = lambda *args, **kwargs: None
        if not hasattr(cache_mod, '_load_patch_dependencies'):
            cache_mod._load_patch_dependencies = lambda: {}
    except ImportError:
        pass

    # 7. torch_npu attributes
    try:
        import torch_npu
        if not hasattr(torch_npu, 'npu_gather_selection_kv_cache'):
            torch_npu.npu_gather_selection_kv_cache = lambda **kwargs: None
    except ImportError:
        raise


def _build_module(name: str, spec):
    if isinstance(spec, types.ModuleType):
        module = spec
    else:
        module = types.ModuleType(name)
        for attr_name, attr_value in spec.items():
            setattr(module, attr_name, attr_value)
    return module


@pytest.fixture
def install_stub_modules(monkeypatch):
    def install(module_specs: dict[str, object]) -> dict[str, types.ModuleType]:
        modules: dict[str, types.ModuleType] = {
            name: _build_module(name, spec)
            for name, spec in module_specs.items()
        }

        for name in list(modules):
            parts = name.split(".")
            for idx in range(1, len(parts)):
                pkg_name = ".".join(parts[:idx])
                if pkg_name in modules or pkg_name in sys.modules:
                    continue
                pkg = types.ModuleType(pkg_name)
                pkg.__path__ = []
                modules[pkg_name] = pkg

        for name, module in modules.items():
            if "." not in name:
                continue
            parent_name, attr_name = name.rsplit(".", 1)
            parent = modules.get(parent_name) or sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, attr_name, module)

        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)

        return modules

    return install
