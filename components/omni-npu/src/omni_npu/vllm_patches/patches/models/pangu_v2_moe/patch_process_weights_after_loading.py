# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch
from torch import nn

from vllm.attention.layer import Attention, MLAAttention
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

import vllm.model_executor.model_loader.utils as model_loader_utils
import vllm.model_executor.model_loader.base_loader as base_loader_module

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.v1.layers.attention.npu_pangu import NPUPanguSparseAttention
from omni_npu.layers.mhc.npu_mhc import NPUmHC
from omni_npu.layers.npu_rms_norm import NPURMSNorm

logger = init_logger(__name__)


def _patched_process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    if getattr(model, "process_weights_after_loading_already_called", False):
        # In case `process_weights_after_loading` is called multiple times
        # we'll skip it at later times
        logger.debug_once(
            "process_weights_after_loading already called for model %s", model
        )
        return

    # to avoid circular dependency
    from vllm.model_executor.model_loader.online_quantization import (
        maybe_save_metadata_and_attributes_for_weight_reloading,
    )

    maybe_save_metadata_and_attributes_for_weight_reloading(model, model_config)

    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with model_loader_utils.device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    # NOTE: Happens after other modules so we can easily decompress weights.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading(model_config.dtype)

        if isinstance(module, (NPUPanguSparseAttention, NPUmHC, NPURMSNorm)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading()


@register_patch("PanguV2MoeProcessWeightsUtilsPatch", model_loader_utils)
class PanguV2MoeProcessWeightsUtilsPatch(VLLMPatch):
    _attr_names_to_apply = ["process_weights_after_loading"]
    process_weights_after_loading = _patched_process_weights_after_loading


@register_patch("PanguV2MoeProcessWeightsBaseLoaderPatch", base_loader_module)
class PanguV2MoeProcessWeightsBaseLoaderPatch(VLLMPatch):
    _attr_names_to_apply = ["process_weights_after_loading"]
    process_weights_after_loading = _patched_process_weights_after_loading
