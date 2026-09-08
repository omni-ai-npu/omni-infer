# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch
from torch import nn

from vllm.config import ModelConfig

import vllm.model_executor.model_loader.utils as model_loader_utils
import vllm.model_executor.model_loader.base_loader as base_loader_module

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.v1.layers.attention.npu_pangu import NPUPanguSparseAttention
from omni_npu.layers.mhc.npu_mhc import NPUmHC
from omni_npu.layers.mhc.mhc_rl import NPUmHCRL
from omni_npu.layers.npu_rms_norm import NPURMSNorm


_ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING = (
    model_loader_utils.process_weights_after_loading
)


def _patched_process_weights_after_loading(
    model: nn.Module,
    model_config: ModelConfig,
    target_device: torch.device,
) -> None:
    _ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING(
        model,
        model_config,
        target_device,
    )

    for _, module in model.named_modules():
        if isinstance(
            module,
            (NPUPanguSparseAttention, NPUmHC, NPUmHCRL, NPURMSNorm),
        ) and hasattr(module, "process_weights_after_loading"):
            with model_loader_utils.device_loading_context(
                module,
                target_device,
            ):
                module.process_weights_after_loading()

    model.process_weights_after_loading_already_called = True


@register_patch("PanguV2MoeProcessWeightsUtilsPatch", model_loader_utils)
class PanguV2MoeProcessWeightsUtilsPatch(VLLMPatch):
    _attr_names_to_apply = ["process_weights_after_loading"]
    process_weights_after_loading = _patched_process_weights_after_loading


@register_patch("PanguV2MoeProcessWeightsBaseLoaderPatch", base_loader_module)
class PanguV2MoeProcessWeightsBaseLoaderPatch(VLLMPatch):
    _attr_names_to_apply = ["process_weights_after_loading"]
    process_weights_after_loading = _patched_process_weights_after_loading
