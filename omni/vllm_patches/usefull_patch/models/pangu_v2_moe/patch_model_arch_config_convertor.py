# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.transformers_utils.model_arch_config_convertor import ModelArchConfigConvertorBase


_ORIGINAL_IS_DEEPSEEK_MLA = ModelArchConfigConvertorBase.is_deepseek_mla


@register_patch("PanguV2MoeModelArchConfigConvertorPatch", ModelArchConfigConvertorBase)
class PanguV2MoeModelArchConfigConvertorPatch(VLLMPatch):
    """Add Pangu v2 MoE model types to vLLM's MLA detection."""

    _attr_names_to_apply = ["is_deepseek_mla"]

    def is_deepseek_mla(self) -> bool:
        if getattr(self.hf_text_config, "model_type", None) in (
            "openpangu_v2",
            "openpangu_v2_vl_moe",
            "openpangu_v2_omni_moe",
        ):
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        return _ORIGINAL_IS_DEEPSEEK_MLA(self)
