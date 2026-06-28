# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm import ModelRegistry


def register_models():
    from transformers import AutoConfig
    from transformers.configuration_utils import PretrainedConfig

    class OpenPanguV2Config(PretrainedConfig):
        model_type = "openpangu_v2"
        keys_to_ignore_at_inference = ["past_key_values"]

    AutoConfig.register("openpangu_v2", OpenPanguV2Config)
    
    ModelRegistry.register_model(
        "OpenPanguV2ForCausalLM",
        "omni_npu.v1.models.pangu.pangu_v2_moe:OpenPanguV2ForCausalLM")
    ModelRegistry.register_model(
        "OpenPanguV2MTPModel",
        "omni_npu.v1.models.pangu.pangu_v2_moe_mtp:OpenPanguV2MTP")
