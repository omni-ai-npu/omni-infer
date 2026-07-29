from vllm.logger import init_logger
from vllm.transformers_utils import model_arch_config_convertor
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
    MODEL_ARCH_CONFIG_CONVERTORS,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


class Qwen3_5MTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "mtp_num_hidden_layers", 0)


MODEL_ARCH_CONFIG_CONVERTORS["qwen3_5_mtp"] = Qwen3_5MTPModelArchConfigConvertor


@register_patch("ModelArchConfigConvertorPatch", model_arch_config_convertor)
class ModelArchConfigConvertorPatch(VLLMPatch):
    _attr_names_to_apply = ['MODEL_ARCH_CONFIG_CONVERTORS']

    MODEL_ARCH_CONFIG_CONVERTORS = MODEL_ARCH_CONFIG_CONVERTORS
