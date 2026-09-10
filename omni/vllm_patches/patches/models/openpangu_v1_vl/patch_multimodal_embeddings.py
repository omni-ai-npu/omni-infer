# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch
import vllm.model_executor.models.utils as utils
from vllm.model_executor.models.utils import (
    _embedding_count_expression,
    _flatten_embeddings,
)
from vllm.multimodal import NestedTensors

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("NPU_merge_multimodal_embeddings", utils)
class NPU_MergeMultimodalEmbeddingsPatch(VLLMPatch):
    """Use index_put_ on NPU while enforcing one embedding per placeholder."""

    _attr_names_to_apply = ["_merge_multimodal_embeddings"]

    @staticmethod
    def _merge_multimodal_embeddings(
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: NestedTensors,
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        if len(multimodal_embeddings) == 0:
            return inputs_embeds

        mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
        indices = is_multimodal.nonzero(as_tuple=True)
        num_actual_tokens = mm_embeds_flat.shape[0]
        num_expected_tokens = indices[0].numel()
        # index_put_ can broadcast one embedding into several placeholders.
        # Reject mismatches before the write instead of relying on an exception.
        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            )

        try:
            inputs_embeds.index_put_(
                indices, mm_embeds_flat.to(dtype=inputs_embeds.dtype)
            )
        except RuntimeError as exc:
            raise ValueError("Error during index_put operation") from exc
        return inputs_embeds
