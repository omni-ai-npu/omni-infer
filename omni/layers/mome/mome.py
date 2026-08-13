# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mome import AggregateConv

import omni_training_custom_ops


@AggregateConv.register_oot
class NPUAggregateConv(AggregateConv):
    def __init__(
        self,
        hidden_size: int,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        output_parallel: bool,
        attn_prefix: str,
        # True for padding 0 and calculating imcomplete seqs
        padding: bool = False
    ):
        super().__init__(hidden_size, config, vllm_config, output_parallel, attn_prefix, padding)
        self.conv_weight = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        only_prefill=False,
        force_decode=False,
        short_prefill=False,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # V1 profile run
            return hidden_states
        attn_metadata = attn_metadata[self.attn_prefix]
        batch_descriptor = forward_context.batch_descriptor
        if only_prefill:
            query_start_loc = attn_metadata.prefill.query_start_loc
            cache_slot_id = forward_context.cache_slot_id[attn_metadata.num_decodes:attn_metadata.num_reqs]
            cache_idx_offset = attn_metadata.num_decodes
        elif force_decode:
            cache_slot_id = forward_context.cache_slot_id[:attn_metadata.num_decodes]
        elif short_prefill:
            # query_start_loc is a prefix-sum style boundary array and needs
            # num_decodes + 1 entries to cover all decode requests.
            query_start_loc = attn_metadata.query_start_loc[:attn_metadata.num_decodes + 1]
            cache_slot_id = forward_context.cache_slot_id[:attn_metadata.num_decodes]
            cache_idx_offset = 0
        else:
            query_start_loc = attn_metadata.query_start_loc
            cache_slot_id = forward_context.cache_slot_id
            cache_idx_offset = 0

        decode_token_num = self.spec_token_num + 1

        requires_per_request_conv = (
            attn_metadata.num_prefills > 0 or not batch_descriptor.uniform or short_prefill
        )
        if requires_per_request_conv and not force_decode:
            batch_size = len(query_start_loc) - 1
            conv_output_list = []
            for i in range(batch_size):
                s = query_start_loc[i]
                e = query_start_loc[i + 1]
                local_input = hidden_states[s: e]
                conv_input = torch.cat(
                    [
                        self.cache_states[
                            cache_slot_id[i], : self.cache_length
                        ].contiguous(),
                        local_input,
                    ],
                    dim=0,
                )
                conv_input_transpose = conv_input.unsqueeze(dim=1)
                weight = self.merge_conv.weight.squeeze(1).transpose(0, 1)
                conv_output = torch.ops.custom.npu_aggregate_hidden(
                                conv_input_transpose, weight).reshape(conv_input.shape)
                conv_output = conv_output[self.cache_length:]
                if not self.padding and cache_slot_id[i] == 0:
                    conv_output[:self.cache_length] = 0
                conv_output_list.append(conv_output)
                if e - s == decode_token_num:
                    self.cache_states[
                        i + 1 + cache_idx_offset, : self.cache_capacity, :
                    ] = conv_input[-self.cache_capacity:, :]
                else:
                    self.cache_states[
                        i + 1 + cache_idx_offset, : self.cache_length, :
                    ] = conv_input[-self.cache_length:, :]

            if e < hidden_states.shape[0]:
                conv_output_list.append(hidden_states[e:])
            conv_output = torch.cat(conv_output_list, dim=0)
        else:
            batch_size = hidden_states.shape[0] // (self.spec_token_num + 1)
            current_state = hidden_states.view(batch_size, decode_token_num, -1)
            conv_input = torch.cat(
                [
                    self.cache_states[cache_slot_id[:batch_size], : self.cache_length],
                    current_state,
                ],
                dim=1,
            )

            conv_input_transpose = conv_input.permute(0, 2, 1)
            conv_output = self.merge_conv(conv_input_transpose).permute(0, 2, 1).reshape(-1, self.hidden_size)

            self.cache_states[1: batch_size + 1, :self.cache_capacity, :] = conv_input[:, -self.cache_capacity:, :]
            
        return conv_output

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        super().weight_loader(param, loaded_weight)
        self.conv_weight = param.data.squeeze(1).transpose(0, 1).contiguous()
