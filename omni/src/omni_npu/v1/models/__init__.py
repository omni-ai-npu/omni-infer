# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from importlib import import_module
import os

from vllm import ModelRegistry


def register_configuration():
    from transformers import AutoConfig
    from transformers.configuration_utils import PretrainedConfig

    class PanguV2MoEConfig(PretrainedConfig):
        model_type = "pangu_v2_moe"
        keys_to_ignore_at_inference = ["past_key_values"]
    AutoConfig.register("pangu_v2_moe", PanguV2MoEConfig)

def register_models():
    register_configuration()

    ModelRegistry.register_model(
        "PanguV2MoEForCausalLM",
        "omni_npu.v1.models.pangu.pangu_v2_moe:PanguV2MoEForCausalLM")

    ModelRegistry.register_model(
        "PanguV2MTPModel",
        "omni_npu.v1.models.pangu.pangu_v2_moe_mtp:PanguV2MTP")
