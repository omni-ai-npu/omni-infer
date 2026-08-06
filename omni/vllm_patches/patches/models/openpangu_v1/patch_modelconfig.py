# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from vllm.config import ModelConfig

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("ModelConfigPatch", ModelConfig)
class ModelConfigPatch(VLLMPatch):
    _attr_names_to_apply = ['get_total_num_hidden_layers']

    
    #####patch start: for pangu72B-VL
    def get_total_num_hidden_layers(self) -> int:
        if (
            self.hf_text_config.model_type == "deepseek_mtp"
            or self.hf_config.model_type == "mimo_mtp"
            or self.hf_config.model_type == "glm4_moe_mtp"
            or self.hf_config.model_type == "ernie_mtp"
            or self.hf_config.model_type == "qwen3_next_mtp"
            or self.hf_config.model_type == "pangu_ultra_moe_mtp"
            or self.hf_config.model_type == "openpangu_vl_mtp"
        ):
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_nextn_predict_layers", 0
            )
        elif self.hf_config.model_type == "longcat_flash_mtp":
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_nextn_predict_layers", 1
            )
        else:
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_hidden_layers", 0
            )
        return total_num_hidden_layers
    #####patch end