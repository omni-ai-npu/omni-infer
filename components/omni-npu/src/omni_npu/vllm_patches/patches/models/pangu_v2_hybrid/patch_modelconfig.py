# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from vllm.logger import init_logger
from vllm.transformers_utils import model_arch_config_convertor
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
    MODEL_ARCH_CONFIG_CONVERTORS,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


class OpenpanguMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 0)


_original_is_deepseek_mla = ModelArchConfigConvertorBase.is_deepseek_mla


def _patched_is_deepseek_mla(self) -> bool:
    if hasattr(self.hf_text_config, "model_type"):
        if self.hf_text_config.model_type in ("openpangu_v2", "openpangu_mtp", "openpangu_v2_vl_moe","openpangu_v2_omni_moe"):
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None
    return _original_is_deepseek_mla(self)


ModelArchConfigConvertorBase.is_deepseek_mla = _patched_is_deepseek_mla

MODEL_ARCH_CONFIG_CONVERTORS["openpangu_mtp"] = OpenpanguMTPModelArchConfigConvertor


@register_patch("ModelArchConfigConvertorPatch", model_arch_config_convertor)
class ModelArchConfigConvertorPatch(VLLMPatch):
    _attr_names_to_apply = ['MODEL_ARCH_CONFIG_CONVERTORS']

    MODEL_ARCH_CONFIG_CONVERTORS = MODEL_ARCH_CONFIG_CONVERTORS
