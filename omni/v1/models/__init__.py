# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from vllm import ModelRegistry


def register_models():
    ModelRegistry.register_model(
        "OpenPanguV2ForCausalLM",
        "omni_npu.v1.models.pangu.pangu_v2_moe:OpenPanguV2ForCausalLM")
    ModelRegistry.register_model(
        "OpenPanguV2MTPModel",
        "omni_npu.v1.models.pangu.pangu_v2_moe_mtp:OpenPanguV2MTP")
