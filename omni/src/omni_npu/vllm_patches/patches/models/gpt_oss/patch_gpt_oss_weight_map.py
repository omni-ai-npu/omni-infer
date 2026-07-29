# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from vllm.model_executor.models.gpt_oss import GptOssForCausalLM, WeightsMapper

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


_GPT_OSS_MOE_SUFFIX_COMPAT_MAPPER = WeightsMapper(
    orig_to_new_suffix={
        # Restrict to MoE expert weights to avoid matching dense MLP projections.
        ".experts.gate_up_proj.weight": ".experts.w13_weight",
        ".experts.down_proj.weight": ".experts.w2_weight",
        ".experts.gate_up_proj.scale": ".experts.w13_weight_scale",
        ".experts.down_proj.scale": ".experts.w2_weight_scale",
        ".experts.gate_up_proj.bias": ".experts.w13_bias",
        ".experts.down_proj.bias": ".experts.w2_bias",
        ".experts.gate_up_proj": ".experts.w13_weight",
        ".experts.down_proj": ".experts.w2_weight",
        ".experts.gate_up_proj_bias": ".experts.w13_bias",
        ".experts.down_proj_bias": ".experts.w2_bias",
    },
)


@register_patch("GptOssMoEWeightNameCompatPatch", GptOssForCausalLM)
class GptOssMoEWeightNameCompatPatch(VLLMPatch):
    _attr_names_to_apply = ["hf_to_vllm_mapper"]

    hf_to_vllm_mapper = (
        GptOssForCausalLM.hf_to_vllm_mapper | _GPT_OSS_MOE_SUFFIX_COMPAT_MAPPER
    )
