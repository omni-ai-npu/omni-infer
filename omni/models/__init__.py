# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import torch_npu
from vllm import ModelRegistry

if os.getenv("PROFILING_NAMELIST", None):
    print("<<<Profiler patch environmental variable is enabled, applying profiler patches.")
    from omni.tools.profiler import apply_profiler_patches

def register_model():
    is_A2 = torch_npu.npu.get_device_name(0).startswith("Ascend910B")
    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "omni.models.deepseek.deepseek_v2:CustomDeepseekV2ForCausalLM")

    ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            "omni.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
    from transformers import AutoConfig, DeepseekV3Config
    class DeepseekV32Config(DeepseekV3Config):
        model_type = "deepseek_v32"
        keys_to_ignore_at_inference = ["past_key_values"]
    AutoConfig.register("deepseek_v32", DeepseekV32Config)
    ModelRegistry.register_model(
            "DeepseekV32ForCausalLM",
            "omni.models.deepseek.deepseek_v32:DeepseekV32ForCausalLM")
    ModelRegistry.register_model(
            "PanguUltraMoEForCausalLM",
            "omni.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM")

    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTP")
    
    ModelRegistry.register_model(
        "DeepSeekMTPModelDuo",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTPDuo")

    ModelRegistry.register_model(
        "DeepSeekMTPModelTres",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTPTres")

    ModelRegistry.register_model(
        "DeepseekForCausalLM",
        "omni.models.deepseek.deepseek:DeepseekForCausalLM")
    
    ModelRegistry.register_model(
        "DeepseekOCRForCausalLM",
        "omni.models.deepseek.deepseek_ocr:DeepseekOCRForCausalLM")

    ModelRegistry.register_model(
            "LongcatFlashForCausalLM",
            "omni.models.longcat.longcat_flash:LongcatFlashForCausalLM")

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "omni.models.qwen.qwen2:Qwen2ForCausalLM")
    
    ModelRegistry.register_model(
        "EagleQwen2ForCausalLM",
        "omni.models.qwen.qwen2_eagle:EagleQwen2ForCausalLM")

    ModelRegistry.register_model(
        "Eagle3Qwen2ForCausalLM",
        "omni.models.qwen.qwen2_eagle3:Eagle3Qwen2ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "omni.models.qwen.qwen3:Qwen3ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "omni.models.qwen.qwen3_moe:Qwen3MoeForCausalLM"
    )
    ModelRegistry.register_model(
        "Qwen3MTPModel",
        "omni.models.qwen.qwen3_mtp:Qwen3MTP"
    )

    ModelRegistry.register_model(
        "Eagle3LlamaForCausalLMEagle3",
        "omni.models.qwen.qwen2_eagle3:Eagle3Qwen2ForCausalLM")

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "omni.models.llama.llama:LlamaForCausalLM")

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "omni.models.qwen.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration")

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "omni.models.qwen.qwen2_vl:Qwen2VLForConditionalGeneration")

    ModelRegistry.register_model(
        "PanguUltraMoEMTPModel",
        "omni.models.pangu.pangu_ultra_moe_mtp:PanguUltraMoEMTP")

    ModelRegistry.register_model(
        "PanguProMoEForCausalLM",
        "omni.models.pangu.pangu_pro_moe.pangu_moe:PanguProMoEForCausalLM")
    
    ModelRegistry.register_model(
        "PanguProMoEV2ForCausalLM",
        "omni.models.pangu.pangu_pro_moe_v2.pangu_moe_v2:PanguProMoEV2ForCausalLM")

    ModelRegistry.register_model(
        "PanguProMoEMTPModel",
        "omni.models.pangu.pangu_pro_moe_v2.pangu_moe_v2_mtp:PanguProMoEMTP")

    ModelRegistry.register_model(
        "PanguEmbeddedForCausalLM",
        "omni.models.pangu.pangu_dense:PanguEmbeddedForCausalLM")

    ModelRegistry.register_model(
        "InternLM2ForCausalLM",
        "omni.models.internvl.internlm2:InternLM2ForCausalLM")
    
    ModelRegistry.register_model(
        "InternVLChatModel",
        "omni.models.internvl.internvl:InternVLChatModel")
    
    ModelRegistry.register_model(
        "Gemma3ForCausalLM",
        "omni.models.gemma.gemma3:Gemma3ForCausalLM")

    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",
        "omni.models.gemma.gemma3_mm:Gemma3ForConditionalGeneration")

    ModelRegistry.register_model(
        "BailingMoeV2ForCausalLM",
        "omni.models.bailing.bailing:BailingMoeV2ForCausalLM")
		
    ModelRegistry.register_model(
        "GlmForCausalLM",
        "omni.models.glm.glm:GlmForCausalLM")
    
    ModelRegistry.register_model(
        "OpenPanguVLForConditionalGeneration",
        "omni.models.pangu.modeling_openpangu_vl:OpenPanguVLForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "GptOssForCausalLM",
        "omni.models.openai.gpt_oss:GptOssForCausalLM")

    ModelRegistry.register_model(
        "HSTUInferenceForCausalLM",
        "omni.models.hstu.hstu:HSTUInferenceForCausalLM")

    ModelRegistry.register_model(
        "Glm4MoeForCausalLM",
        "omni.models.glm.glm4_moe:Glm4MoeForCausalLM",
    )
    
    ModelRegistry.register_model(
        "Glm4MoeMTPModel",
        "omni.models.glm.glm4_moe_mtp:Glm4MoeMTP",
    )

    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from omni.models.hstu.hstu_config import HSTUInferenceRankingConfig
    _CONFIG_REGISTRY["hstu_inference_ranking"] = HSTUInferenceRankingConfig

    if (
        int(os.getenv("RANDOM_MODE", default='0')) or
        int(os.getenv("CAPTURE_MODE", default='0')) or
        int(os.getenv("REPLAY_MODE", default='0'))
    ):
        from omni.models.mock.mock import mock_model_class_factory

        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        ModelRegistry.register_model(
            "Qwen2ForCausalLM",
            mock_model_class_factory(Qwen2ForCausalLM))
        from omni.models.deepseek.deepseek_v3 import DeepseekV3ForCausalLM
        ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            mock_model_class_factory(DeepseekV3ForCausalLM))
