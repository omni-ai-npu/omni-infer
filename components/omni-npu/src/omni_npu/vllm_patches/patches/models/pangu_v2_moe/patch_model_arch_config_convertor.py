# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.transformers_utils.model_arch_config_convertor import ModelArchConfigConvertorBase


@register_patch("PanguV2MoeModelArchConfigConvertorPatch", ModelArchConfigConvertorBase)
class PanguV2MoeModelArchConfigConvertorPatch(VLLMPatch):
    """
    Patch for ModelArchConfigConvertor to support pangu_v2_moe MLA architecture.
    """

    def _is_deepseek_mla(self) -> bool:
        """
        Check if the model uses DeepSeek-style MLA (Multi-Head Latent Attention).
        """
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "pangu_ultra_moe_mtp",
            "pangu_v2_moe",
        ):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == "eagle":
            return (
                self.hf_text_config.model.model_type
                in ("deepseek_v2", "deepseek_v3", "deepseek_v32")
                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    _attr_names_to_apply = ["is_deepseek_mla"]

    ##### patch start: for pangu_v2_moe
    is_deepseek_mla = _is_deepseek_mla
    ##### patch end