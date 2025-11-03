# SPDX-License-Identifier: Apache-2.0

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Any, List, Literal, Optional, TypedDict, TypeVar, Union
import torch
from vllm.attention import AttentionMetadata
from vllm.model_executor.models.internvl import InternVLChatModel
from vllm.sequence import IntermediateTensors
from vllm.compilation.decorators import support_torch_compile
from omni.layers.attention.backend.attention import AscendAttentionState

@support_torch_compile()
class InternVLChatModel(InternVLChatModel):

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor] = None,
        attn_metadata: AttentionMetadata = None,
        selected_indices: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> IntermediateTensors:

        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.language_model.model.layers[self.language_model.model.start_layer].layer_name]
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly