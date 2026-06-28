# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

import vllm.model_executor.models.utils as utils
from vllm.multimodal import NestedTensors
from vllm.model_executor.models.utils import (
            _flatten_embeddings, 
            _embedding_count_expression
        )
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
import torch


@register_patch("NPU_merge_multimodal_embeddings", utils)  
class NPU_MergeMultimodalEmbeddingsPatch(VLLMPatch):
    """
    optimized patch for _merge_multimodal_embeddings.
    Replaces masked_scatter_ with index_put_.
    """

    _attr_names_to_apply = ['_merge_multimodal_embeddings'] 
    
    @staticmethod
    def _merge_multimodal_embeddings( 
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: NestedTensors,
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        """
        NPU-optimized version using index_put_ instead of masked_scatter_.
        """
        if len(multimodal_embeddings) == 0:
            return inputs_embeds
        
        mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
        input_dtype = inputs_embeds.dtype
        
        try:
            indices = is_multimodal.nonzero(as_tuple=True)
            inputs_embeds.index_put_(
                indices,
                mm_embeds_flat.to(dtype=input_dtype)
            )
        except RuntimeError as e:
            num_actual_tokens = len(mm_embeds_flat)
            num_expected_tokens = is_multimodal.sum().item()
            
            if num_actual_tokens != num_expected_tokens:
                expr = _embedding_count_expression(multimodal_embeddings)
                raise ValueError(
                    f"Attempted to assign {expr} = {num_actual_tokens} "
                    f"multimodal tokens to {num_expected_tokens} placeholders"
                ) from e
            raise ValueError("Error during index_put operation") from e
        
        return inputs_embeds
