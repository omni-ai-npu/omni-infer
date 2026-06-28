# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from typing import TYPE_CHECKING, Any, Literal

from pydantic.dataclasses import dataclass

from vllm.logger import init_logger
from vllm.config.utils import config
from vllm.config.speculative import SpeculativeConfig
from vllm.config import speculative
from vllm.utils.import_utils import LazyLoader
if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


@register_patch("SpeculativePatch", speculative)
class SpeculativePatch(VLLMPatch):
    _attr_names_to_apply = ['MTPModelTypes']

    MTPModelTypes = Literal[
        "deepseek_mtp",
        "mimo_mtp",
        "glm4_moe_mtp",
        "ernie_mtp",
        "qwen3_next_mtp",
        "longcat_flash_mtp",
        "mtp",
        "pangu_ultra_moe_mtp",
        "openpangu_mtp",
    ]


_origin_hf_config_override = SpeculativeConfig.hf_config_override


@register_patch("OpenPanguV2SpeculativeConfigPatch", SpeculativeConfig)
@config
@dataclass
class PanguV2MoeSpeculativeConfigPatch(VLLMPatch):
    """Patch to add openpangu_v2 -> openpangu_mtp MTP mapping."""

    _attr_names_to_apply = ['hf_config_override']

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:

        if hf_config.model_type in ("openpangu_v2_vl_moe", "openpangu_v2_omni_moe"):
            hf_config.model_type = "openpangu_mtp"
        if hf_config.model_type == "openpangu_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )
            return hf_config

        #####patch start: for openpangu_v2_mtp
        if hf_config.model_type == "openpangu_v2":
            hf_config.model_type = "mtp"
        if hf_config.model_type == "mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguV2MTPModel"]}
            )
            return hf_config

        #####patch end

        return _origin_hf_config_override(hf_config)
