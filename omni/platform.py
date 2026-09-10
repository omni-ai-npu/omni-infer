# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import traceback
from typing import TYPE_CHECKING, Optional
import torch
from importlib.metadata import entry_points

from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from omni_npu.logger import update_configure_vllm_root_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    FlexibleArgumentParser = object


update_configure_vllm_root_logger()
logger = init_logger(__name__)


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
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    ]

    def __init__(self):
        """Initialize the NPU platform and configure environment."""
        super().__init__()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.npu.set_device(device)
        # With this trick we can force the device to be set eagerly
        # see https://github.com/pytorch/pytorch/issues/155668
        # for why and when it is needed
        _ = torch.zeros(1, device=device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.npu.manual_seed_all(seed)

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
        from omni_npu.connector.ec_connector import register_ec_connectors
        register_ec_connectors()

        from omni_npu.connector.mm_feature_transfer import register_mm_feature_transfer
        register_mm_feature_transfer()
        
        from omni_npu.connector import register_connectors
        register_connectors()

        from omni_npu.v1.kv_offload.register import register_kv_offload_specs
        register_kv_offload_specs()

        for ep in entry_points().select(group="omni_npu.kv_connectors"):
            try:
                register_fn = ep.load()
                register_fn()
            except Exception as e:
                logger.warning(f"Failed to load connector {ep.name}: {e}")

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def device_count(cls) -> int:
        return torch.npu.device_count()

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        return torch.npu.mem_get_info()

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.npu.get_device_properties(device_id).multi_processor_count

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        return

    @classmethod
    def is_uva_available(cls) -> bool:
        alloc_conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        if "pinned_mem_register:True" not in alloc_conf:
            return False
        if "pin_memory_expandable_segments:True" in alloc_conf:
            return False

        try:
            if not torch.npu.is_available():
                return False
            from omni_npu.allocator import npu_uva  # noqa: F401
        except Exception:
            return False
        return True

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
        from omni_npu import layers
        from vllm.v1.attention.backends.mla.prefill.registry import (
            MLAPrefillBackendEnum,
            register_mla_prefill_backend,
        )

        # NPU has no device capability; selector always picks FLASH_ATTN.
        register_mla_prefill_backend(
            MLAPrefillBackendEnum.FLASH_ATTN,
            "omni_npu.attention.backends.mla.NPUMLAPrefillBackend",
        )

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:  # type: ignore[name-defined]
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
    def _align_hybrid_block_size(
        cls, vllm_config: "VllmConfig", backend_cls
    ) -> None:
        # Run the base alignment (pads the mamba page to the MLA attention page).
        super()._align_hybrid_block_size(vllm_config, backend_cls)
        # The base per-token attention page uses MLAAttentionSpec, which omits
        # the top-k indexer (index_head_dim) stored in the DSA KV block. Re-pad
        # the mamba page to the DSA page so the MOME spec and the DSA/MLA
        # attention pages unify. DSA uses different packed layouts for the
        # quantized KV-cache dtypes, so the page size cannot be derived from
        # model_config.dtype alone.

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        index_topk = getattr(hf_config, "index_topk", 0)
        if not (model_config.use_mla and index_topk > 0):
            return
        # The base returns without setting a padding when the model has no
        # mamba layers, and such a model has nothing to align - correct only a
        # padding it actually established, so a sparse model without mamba
        # (DeepSeek V3.2) is left alone.
        if cache_config.mamba_page_size_padded is None:
            return
        index_head_dim = getattr(hf_config, "index_head_dim", 0)
        block_size = cache_config.block_size
        cache_dtype = cache_config.cache_dtype
        if cache_dtype in ("fp8_ds_mla", "hif8_ds_mla"):
            # 2 * block_size * (656 + 128 + 4): FP8/HIF8 DSA layout.
            dsa_page = 2 * block_size * (656 + 128 + 4)
        elif cache_dtype == "int8_ds_mla":
            # 2 * block_size * (656 + 128 + 2): INT8 DSA layout.
            dsa_page = 2 * block_size * (656 + 128 + 2)
        elif cache_dtype == "li_int8_ds_mla":
            # block_size * (576 * 2 + 128 + 2): Li-INT8 DSA layout.
            dsa_page = block_size * (576 * 2 + 128 + 2)
        else:
            # auto/non-quantized DSA layout (BF16 KV cache).
            dsa_page = block_size * (576 + index_head_dim) * 2
        if dsa_page > cache_config.mamba_page_size_padded:
            cache_config.mamba_page_size_padded = dsa_page
            logger.info(
                "DSA-aware mamba page size padding: mamba_page_size_padded=%d",
                dsa_page,
            )

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
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
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
