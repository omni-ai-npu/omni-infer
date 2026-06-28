# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

import os
import traceback
from typing import Optional
import torch
import os
from importlib.metadata import entry_points
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum
from omni_npu.logger import update_configure_vllm_root_logger


update_configure_vllm_root_logger()
logger = init_logger(__name__)


class ConfigUpdater:
    """Handles configuration validation and updates for the NPU platform."""

    @classmethod
    def update_vllm_config(cls, vllm_config: 'VllmConfig') -> None:
        """Update the vLLM configuration for NPU compatibility.

        Args:
            vllm_config: The vLLM configuration to update.
        """
        from omni_npu.compilation.ge_compile_config import NPUCompilationConfig
        vllm_config.npu_compilation_config = NPUCompilationConfig()
        import os
        vllm_config.npu_compilation_config.use_gegraph = os.getenv("TORCH_COMPILE_GE", "False").lower() == "true"
        cls._handle_graph_mode(vllm_config)

        additional_config = vllm_config.additional_config
        if additional_config:
            graph_model_compile_config = additional_config.get("graph_model_compile_config", None)
            if graph_model_compile_config is not None:
                vllm_config.npu_compilation_config.build_from_cli(graph_model_compile_config, vllm_config)
                logger.debug(f"Graph model compile config: {graph_model_compile_config}")

    @staticmethod
    def _handle_graph_mode(vllm_config: 'VllmConfig') -> None:
        """Handle graph mode configuration for NPU."""
        from vllm.utils.torch_utils import supports_dynamo
        if not supports_dynamo():
            logger.warning("Graph mode unsupported due to low torch version. Disabling.")
            vllm_config.npu_compilation_config.use_gegraph = False


class NPUPlatform(Platform):
    try:
        # In case vllm already defined HUAWEI_NPU platform
        _enum = PlatformEnum.HUAWEI_NPU
    except AttributeError:
        # fallback to OOT
        _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "NPU"
    dist_backend: str = "hccl"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    def __init__(self):
        """Initialize the NPU platform and configure environment."""
        super().__init__()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.npu.set_device(device)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def import_kernels(cls):
        from omni_npu.compilation.decorators import patch_compile_decorators
        patch_compile_decorators()
        
        from omni_npu.connector import register_connectors
        register_connectors()
        
        for ep in entry_points().select(group="omni.kv_connectors"):
            try:
                register_fn = ep.load()
                register_fn()
            except Exception as e:
                logger.warning(f"Failed to load connector {ep.name}: {e}")

    @classmethod
    def get_current_memory_usage(cls, device: Optional[torch.types.Device] = None) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def device_count(cls) -> int:
        return torch.npu.device_count()

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        return torch.npu.mem_get_info()

    def is_cuda_alike(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        scope = traceback.format_stack()[-2]
        caller = traceback.extract_stack(limit=2)[0]
        caller_file = caller.filename.replace("\\", "/")
        is_npu = getattr(self, "device_type", None) == "npu"
        if is_npu:
            if "parallel.py" in scope and "is_cuda_alike" in scope:
                return True
            if (
                caller.name == "bind_kv_cache"
                and caller_file.endswith("vllm/v1/worker/utils.py")
            ):
                return True
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    @classmethod
    def pre_register_and_update(
        cls, parser: Optional["FlexibleArgumentParser"] = None
    ) -> None:
        """
        Do some pre-registration or update action for the current platform.

        This function is called before global VllmConfig is initialized or cli
        arguments are parsed. It's used for out-of-tree platforms to register or
        update the configuration.

        For example, the out-of-tree quantization config can be imported and
        registered here dynamically.
        """
        # Import NPU custom layers to ensure they're registered before use
        from omni_npu import layers

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:  # type: ignore[name-defined]
        ConfigUpdater.update_vllm_config(vllm_config)
        # Minimal defaults to match vLLM expectations.
        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = "omni_npu.worker.npu_worker.NPUWorker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128

        vllm_config.compilation_config.pass_config.fuse_norm_quant = False
        vllm_config.compilation_config.pass_config.fuse_act_quant = False
        vllm_config.compilation_config.pass_config.fuse_attn_quant = False

        from vllm.config.compilation import CUDAGraphMode

        if (vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY or
           vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL):
            logger.info("using only ACL Graph Mode")
            vllm_config.compilation_config.splitting_ops = []

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # Use CPU punica wrapper by default
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # Point vLLM to our HCCL-based communicator implementation
        return "omni_npu.distributed.communicator.NPUCommunicator"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
    ) -> str:
        # Import here to avoid circular import and ensure plugins are loaded
        from omni_npu.attention.backends.utils import get_attention_backend
        if attn_selector_config.use_mla:
            if attn_selector_config.use_sparse:
                backend_name = "NPUDSA"
            else:
                backend_name = "NPUMLA"
        else:
            backend_name = "VLLM_NPU_ATTN"

        # Query registry first (allows plugins to override)
        registered_path = get_attention_backend(backend_name)
        return registered_path

    @property
    def simple_compile_backend(self):
        return "eager"

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """
        Returns if the graph mode is supported by the current platform.
        """
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        """
        Get piecewise backend class for piecewise graph.
        """
        return "omni_npu.compilation.acl_graph.ACLGraphWrapper"

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        """
        Get the pass manager class for the current platform.
        """
        return "omni_npu.compilation.pass_manager.GraphPassManager"
        
    @classmethod
    def get_compile_backend(cls) -> str:
        """
        Get the custom compile backend for current platform.
        """
        return "omni_npu.compilation.npugraph_ex.NpuGraphExAdaptor"

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    def is_sleep_mode_available(self) -> bool:
        """
        Returns if the sleep mode is available for the current platform.
        """
        return True
    
    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True
