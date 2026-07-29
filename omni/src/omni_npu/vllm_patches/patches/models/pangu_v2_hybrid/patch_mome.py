# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
import types

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor import layers
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)
dynamic_module = types.ModuleType("mome")
sys.modules[layers.__name__ + ".mome"] = dynamic_module
layers.mome = dynamic_module


@register_patch("MoMEPatch", layers)
class MoMEPatch(VLLMPatch):
    _attr_names_to_apply = ['AggregateConv']

    # patch start
    @CustomOp.register("PanguV2HybridAggregateConv")
    class AggregateConv(CustomOp):
        def __init__(
            self,
            hidden_size: int,
            config: PretrainedConfig,
            vllm_config: VllmConfig,
            output_parallel: bool,
            attn_prefix: str,
            # True for padding 0 and calculating imcomplete seqs
            padding: bool = False,
        ):
            super().__init__()
            # Current working aclop is Conv2D.

            torch_npu.npu.config.allow_internal_format = False
            self.hidden_size = hidden_size
            self.output_parallel = output_parallel
            self.router_sliding_window = getattr(config, "router_sliding_window", 0)
            self.merge_conv = torch.nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=self.router_sliding_window,
                groups=hidden_size,
                bias=False,
            )
            self.attn_prefix = attn_prefix
            set_weight_attrs(
                self.merge_conv.weight,
                {"weight_loader": self.weight_loader},
            )
            self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            self.cache_length = self.router_sliding_window - 1
            self.cache_states = torch.zeros(
                (self.max_num_seqs + 1, self.cache_length, hidden_size),
                device=current_platform.device_type,
            )
            self.padding = padding

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
                cache_slot_id = forward_context.cache_slot_id[
                    attn_metadata.num_decodes:attn_metadata.num_reqs
                ]
                cache_idx_offset = attn_metadata.num_decodes
            elif force_decode:
                cache_slot_id = forward_context.cache_slot_id[
                    : attn_metadata.num_decodes
                ]
            elif short_prefill:
                query_start_loc = attn_metadata.query_start_loc[
                    : attn_metadata.num_decodes + 1
                ]
                cache_slot_id = forward_context.cache_slot_id[
                    : attn_metadata.num_decodes
                ]
                cache_idx_offset = 0
            else:
                query_start_loc = attn_metadata.query_start_loc
                cache_slot_id = forward_context.cache_slot_id
                cache_idx_offset = 0
            if (
                attn_metadata.num_prefills > 0
                or batch_descriptor.num_tokens > attn_metadata.num_decodes
                or short_prefill
            ) and not force_decode:
                batch_size = len(query_start_loc) - 1
                conv_output_list = []
                for i in range(batch_size):
                    s = query_start_loc[i]
                    e = query_start_loc[i + 1]
                    local_input = hidden_states[s: e]
                    conv_input = torch.cat(
                        [self.cache_states[cache_slot_id[i]], local_input],
                        dim=0,
                    )
                    conv_input_transpose = conv_input.transpose(1, 0)
                    conv_output = self.merge_conv(conv_input_transpose).transpose(1, 0)
                    if not self.padding and cache_slot_id[i] == 0:
                        conv_output[:self.cache_length] = 0
                    conv_output_list.append(conv_output)
                    self.cache_states[i + 1 + cache_idx_offset] = conv_input[
                        -self.cache_length:,
                        :,
                    ]
                if e < hidden_states.shape[0]:
                    conv_output_list.append(hidden_states[e:])
                conv_output = torch.cat(conv_output_list, dim=0)
            else:
                num_tokens = hidden_states.shape[0]
                conv_input = torch.cat(
                    [
                        self.cache_states[cache_slot_id[:num_tokens], ...],
                        hidden_states.unsqueeze(1),
                    ],
                    dim=1,
                )
                conv_input_transpose = conv_input.permute(0, 2, 1)
                conv_output = (
                    self.merge_conv(conv_input_transpose)
                    .permute(0, 2, 1)
                    .view(-1, self.hidden_size)
                )
                # idx 0 for new requests padding 0
                self.cache_states[1:num_tokens + 1, :, :] = conv_input[
                    :,
                    -self.cache_length:,
                    :,
                ]
            return conv_output

        def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
            tp_rank = get_tensor_model_parallel_rank()
            param_data = param.data
            if self.output_parallel:
                loaded_weight = loaded_weight.narrow(
                    0,
                    tp_rank * self.hidden_size,
                    self.hidden_size,
                )
            assert param_data.shape == loaded_weight.shape, (
                f"weight shape mismatch: expected {param_data.shape}, got {loaded_weight.shape}."
            )
            param_data.copy_(loaded_weight)
    layers.mome.AggregateConv = AggregateConv
