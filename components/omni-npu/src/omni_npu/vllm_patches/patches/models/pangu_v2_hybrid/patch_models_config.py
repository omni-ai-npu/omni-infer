# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    HybridAttentionMambaModelConfig,
    MambaModelConfig,
    VerifyAndUpdateConfig,
)
import vllm.model_executor.models.config as models_config_module
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

from vllm.config import VllmConfig

logger = init_logger(__name__)

_SKIP_HYBRID_ATTENTION_MAMBA_ARCHITECTURES = {"PanguV2MoEForCausalLM"}

_original_hybrid_verify_and_update_config = (
    HybridAttentionMambaModelConfig.__dict__["verify_and_update_config"]
)

class PanguV2HybridForCausalLMConfig(MambaModelConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.utils.torch_utils import get_dtype_size
        from math import prod

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        is_pd_disagg = vllm_config.kv_transfer_config is not None
        num_extra_token = 1 if is_pd_disagg else 0
        num_speculative_tokens = 0 if not vllm_config.speculative_config else vllm_config.speculative_config.num_speculative_tokens
        fake_num_spec_tokens = max(num_speculative_tokens, num_extra_token)

        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 128
        block_size = cache_config.block_size
        assert block_size % 16 == 0, f"block size {block_size} must be divisible by 16."

        MambaModelConfig.verify_and_update_config(vllm_config)
        
        bf16_size = 2

        # Calculate MLA/SWA page size
        mla_head_size = model_config.hf_config.kv_lora_rank + model_config.hf_config.qk_rope_head_dim
        mla_page_dim = mla_head_size * bf16_size

        # Calculate DSA page size if DSA layer exists
        dsa_page_dim = None
        if hasattr(model_config.hf_config, "index_topk") and model_config.hf_config.index_topk > 0:
            from vllm.v1.kv_cache_interface import DSAAttentionSpec
            index_head_dim = getattr(model_config.hf_config, "index_head_dim", 0)
            # Non-quant case: standard attention format
            dsa_page_dim = DSAAttentionSpec(
                block_size=1,
                num_kv_heads=1,
                head_size=mla_head_size + index_head_dim,
                dtype=kv_cache_dtype,
                cache_dtype_str=cache_config.cache_dtype,
            ).page_size_bytes

        mome_state_shapes = (
            (model_config.hf_config.q_lora_rank,),
            (model_config.hf_config.kv_lora_rank,),
            (model_config.hf_config.num_attention_heads * model_config.hf_config.v_head_dim,),
        )
        mome_state_dtypes = (
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
        )

        # Calculate MOME page size if MOME is enabled
        mome_page_size = None
        if getattr(model_config.hf_config, "use_mome", False):
            kernel_size = getattr(model_config.hf_config, 'router_sliding_window', 0)
            num_total_tokens = kernel_size - 1 + fake_num_spec_tokens
            mome_page_size = sum(
                prod(shape) * get_dtype_size(dtype)
                for (shape, dtype) in zip(mome_state_shapes, mome_state_dtypes)
            ) * num_total_tokens

        # Determine alignment priority
        if dsa_page_dim is not None:
            target_page_size = dsa_page_dim * block_size
            if mome_page_size is not None:
                assert target_page_size >= mome_page_size, f"block size {block_size} * dsa page dim {dsa_page_dim} \
                    should be greater than mome page size {mome_page_size}."
        elif mome_page_size is not None:
            target_page_size = max(mome_page_size, mla_page_dim * block_size)
        else:
            target_page_size = mla_page_dim * block_size

        cache_config.mamba_page_size_padded = target_page_size

class PanguV2MoEForCausalLMConfig(MambaModelConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:

        # set mamba_block_size
        # TODO: align pagesizes, particularly for dsa c8
        super().verify_and_update_config(vllm_config)

@register_patch("PanguV2HybridModelsConfigMapPatch", models_config_module)
class PanguV2HybridModelsConfigMapPatch(VLLMPatch):
    """Add PanguUltraMoEForCausalLM to MODELS_CONFIG_MAP."""

    _attr_names_to_apply = ["MODELS_CONFIG_MAP"]
    MODELS_CONFIG_MAP = {
        **MODELS_CONFIG_MAP,
        "PanguUltraMoEForCausalLM": PanguV2HybridForCausalLMConfig,
        "OpenPanguV2VLForConditionalGeneration": PanguV2HybridForCausalLMConfig,
        "OpenPanguUltraOmniForConditionalGeneration": PanguV2HybridForCausalLMConfig,
        "PanguV2MoEForCausalLM": PanguV2MoEForCausalLMConfig,
    }

@register_patch("PanguV2HybridVllmConfigPatch", VllmConfig)
class PanguV2HybridVllmConfigPatch(VLLMPatch):

    _attr_names_to_apply = ["try_verify_and_update_config"]

    def try_verify_and_update_config(self):
        if self.model_config is None:
            return

        # Avoid running try_verify_and_update_config multiple times
        if getattr(self.model_config, "config_updated", False):
            return
        self.model_config.config_updated = True

        architecture = self.model_config.architecture
        if architecture is None:
            return

        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
        )

        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)

        # patch remove mamba hybrid

        if self.model_config.convert_type == "classify":
            # Maybe convert ForCausalLM into ForSequenceClassification model.
            from vllm.model_executor.models.adapters import SequenceClassificationConfig

            SequenceClassificationConfig.verify_and_update_config(self)

        if hasattr(self.model_config, "model_weights") and is_runai_obj_uri(
            self.model_config.model_weights
        ):
            if self.load_config.load_format == "auto":
                logger.info(
                    "Detected Run:ai model config. "
                    "Overriding `load_format` to 'runai_streamer'"
                )
                self.load_config.load_format = "runai_streamer"
            elif self.load_config.load_format not in (
                "runai_streamer",
                "runai_streamer_sharded",
            ):
                raise ValueError(
                    f"To load a model from S3, 'load_format' "
                    f"must be 'runai_streamer' or 'runai_streamer_sharded', "
                    f"but got '{self.load_config.load_format}'. "
                    f"Model: {self.model_config.model}"
                )

@register_patch(
    "PanguV2MoEHybridAttentionMambaConfigPatch",
    HybridAttentionMambaModelConfig,
)
class PanguV2MoEHybridAttentionMambaConfigPatch(VLLMPatch):
    """Override HybridAttentionMambaModelConfig.verify_and_update_config
    to skip the hybrid page size alignment for PanguV2MoEForCausalLM,
    while still running MambaModelConfig.verify_and_update_config."""

    _attr_names_to_apply = ["verify_and_update_config"]

    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        architecture = vllm_config.model_config.architecture
        if architecture in _SKIP_HYBRID_ATTENTION_MAMBA_ARCHITECTURES:
            logger.info(
                "Skipping HybridAttentionMambaModelConfig."
                "verify_and_update_config for %s, "
                "only running MambaModelConfig.verify_and_update_config",
                architecture,
            )
            MambaModelConfig.verify_and_update_config(vllm_config)
            return
        _original_hybrid_verify_and_update_config.__func__(cls, vllm_config)
