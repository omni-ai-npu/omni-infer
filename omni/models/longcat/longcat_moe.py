# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Longcat-flash model."""
from typing import Optional
import torch, torch_npu
from torch import nn
from transformers import PretrainedConfig
import torchair as tng
torch._logging.set_logs(recompiles=True)
# vllm adaptor
from vllm.platforms import current_platform
from vllm.config import QuantizationConfig
from vllm.attention import AttentionMetadata
from vllm.distributed import (
    get_ep_group,
    get_dp_group,
    get_world_group,
    get_tensor_model_parallel_rank
)
from vllm.model_executor.layers.linear import ReplicatedLinear

from omni.models.longcat.longcat_fused_moe import FusedMoE
from omni.models.config_loader.loader import model_extra_config
from omni.adaptors.vllm.utils import get_attr_by_names


"""NPU Stream Switch Names"""
STREAM_SHARED_EXPERT = 'stream_shared_expert'
SEQ_SPLIT_LENGTH = 4096
def rank0_log(msg):
    if get_tensor_model_parallel_rank() == 0:
        print(msg)

class LongcatFlashTopkRouter(nn.Module):
    def __init__(
        self, 
        config, 
        zero_expert_num: int = 0,
        router_expert_num: int = 256,
        prefix: str = "", 
        rounter_params_dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.config = config
        self.top_k = config.moe_topk
        self.n_routed_experts = router_expert_num + zero_expert_num
        self.routed_scaling_factor = config.routed_scaling_factor

        self.classifier = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=config.router_bias,
            params_dtype=rounter_params_dtype,
            quant_config=None,
            prefix=f"{prefix}.classifier",
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros((self.n_routed_experts), dtype=rounter_params_dtype)
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states)

    def get_topk_indices(self, classify_score):
        n_routed_experts = classify_score.shape[-1]
        scores = classify_score.softmax(dim=-1)
        scores_for_choice = scores.view(-1, n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        # TODO：npu_moe_gating_top_k算子有问题，只支持256和384专家。
        # return torch_npu.npu_moe_gating_top_k(
        #     router_logits.float(),
        #     k=self.top_k,  # topk is currently 8
        #     bias=self.e_score_correction_bias,  # float32
        #     k_group=1,  # fix: 4
        #     group_count=self.n_routed_experts,  # fix 8
        #     group_select_mode=1,  # 0: maximum in group; 1: topk2.sum(fix)
        #     renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
        #     norm_type=1,  # 0: softmax; 1: sigmoid(fix)
        #     routed_scaling_factor=self.routed_scaling_factor,
        #     eps=float(1e-20)
        # )
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights.to(torch.float32) * self.routed_scaling_factor
        return topk_weights, topk_indices.to(torch.int32)

class LongcatFlashMoE(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.ep_size = get_ep_group().world_size
        self.routed_scaling_factor = config.routed_scaling_factor

        n_routed_experts_names = ['num_routed_experts', 'n_routed_experts']
        self.n_routed_experts = get_attr_by_names(config, n_routed_experts_names, 256)
        self.zero_expert_num = config.zero_expert_num
        self.redundancy_shared_expert_num = model_extra_config.parall_config.redundancy_shared_expert_num
        self.quant_symbol = quant_config is not None
        self.is_init_gate = False

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        router_params_dtype = torch.float32
        self.router = LongcatFlashTopkRouter(config=config,
                                             zero_expert_num=self.zero_expert_num,
                                             router_expert_num=self.n_routed_experts,
                                             rounter_params_dtype=router_params_dtype,
                                             prefix=f"{prefix}.router")

        self.top_k = config.moe_topk
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.global_rank = get_world_group().rank_in_group

        moe_prefix = f"{prefix}.experts"
        # omni placement for redundancy route experts
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_ffn_hidden_size,
            reduce_results=False,
            quant_config=quant_config,
            prefix=moe_prefix,
        )
        self.local_expert_num = self.experts.w13_weight.shape[0]
        if self.quant_symbol:
            self.in_scale_2 = torch.ones((self.local_expert_num, config.expert_ffn_hidden_size), dtype=torch.float32, device=current_platform.device_type)
            # call the mark_static to reduce memory usage
            torch._dynamo.mark_static(self.in_scale_2)

        self.tuning_config = None
        if model_extra_config.operator_opt_config.gmm_nz:
            self.tuning_config = model_extra_config.task_config.decode_gear_list[:1]

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        if not self.is_init_gate:
            self.router.classifier.weight.data = torch_npu.npu_format_cast(self.router.classifier.weight.data, 2)
            self.is_init_gate = True
        if attn_metadata is None or attn_metadata.prefill is not None:
            return self._forward_prefill_norm(hidden_states, attn_metadata)
        else:
            return self._forward_decode_norm(hidden_states, attn_metadata)

    def tranfer_zero_expert_ids(self, topk_ids: torch.Tensor, zero_expert_mask: torch.Tensor, normal_expert_mask: torch.Tensor) -> torch.Tensor:
        device = topk_ids.device
        out_dtype = topk_ids.dtype

        row_idx, slot_idx = torch.nonzero(normal_expert_mask, as_tuple=True)
        used_ids_flat = topk_ids[row_idx, slot_idx].to(torch.long)

        used_mask = torch.zeros((topk_ids.shape[0], self.n_routed_experts), dtype=torch.bool, device=device)
        used_mask.index_put_((row_idx, used_ids_flat), torch.ones_like(used_ids_flat, dtype=torch.bool), accumulate=True)
        available_mask = ~used_mask

        num_zero_each_row = zero_expert_mask.sum(dim=1)
        max_zero_each_row = int(num_zero_each_row.max().item())
        if max_zero_each_row == 0:
            return topk_ids

        idxN = torch.arange(self.n_routed_experts, device=device).float().unsqueeze(0).expand(topk_ids.shape[0], -1)
        score_cand = torch.where(available_mask, -idxN, torch.full_like(idxN, float('-inf')))
        _, cand_idx = torch.topk(score_cand, k=max_zero_each_row, largest=True, sorted=True)

        col_idx = torch.arange(topk_ids.shape[1], device=device).float().unsqueeze(0).expand(topk_ids.shape[0], -1)
        score_zero = torch.where(zero_expert_mask, -col_idx, torch.full_like(col_idx, float('-inf')))
        _, z_indices_pad = torch.topk(score_zero, k=max_zero_each_row, largest=True, sorted=True)

        j = torch.arange(max_zero_each_row, device=device).unsqueeze(0).expand(topk_ids.shape[0], max_zero_each_row)
        valid_assign = j < num_zero_each_row.view(-1, 1)

        rows_expanded = torch.arange(topk_ids.shape[0], device=device).unsqueeze(1).expand_as(j)
        rows_sel = rows_expanded[valid_assign]
        cols_sel = z_indices_pad[valid_assign]
        vals_sel = cand_idx[valid_assign].to(out_dtype)

        topk_ids.index_put_((rows_sel, cols_sel), vals_sel, accumulate=False)
        return topk_ids

    def compute_zero_experts(self, hidden_states, topk_weights, topk_ids):
        zero_expert_mask = topk_ids >= self.n_routed_experts
        normal_expert_mask = topk_ids < self.n_routed_experts

        zero_expert_weights = topk_weights.clone()
        zero_expert_weights[normal_expert_mask] = 0.0
        total_weights = zero_expert_weights.sum(dim=-1, keepdim=True)   # [T, 1]
        zero_expert_output = hidden_states * total_weights              # [T, H]
        zero_expert_output = zero_expert_output.to(hidden_states.dtype)
        
        topk_ids = self.tranfer_zero_expert_ids(topk_ids, zero_expert_mask, normal_expert_mask)
        topk_weights[zero_expert_mask] = 0.0
        return zero_expert_output, topk_ids, topk_weights

    def _forward_prefill_norm(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        router_logits, _ = self.router.forward(hidden_states.float())
        topk_weights, topk_ids = self.router.get_topk_indices(router_logits)
        zero_expert_output, topk_ids, topk_weights = self.compute_zero_experts(hidden_states, topk_weights, topk_ids)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=None,
            attn_metadata=attn_metadata
        )
        if isinstance(final_hidden_states, tuple):
            gathered_tokens = final_hidden_states[1]
            expanded_row_idx = final_hidden_states[3]
            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=zero_expert_output,
                skip2=None,
                bias=None,
                scales=topk_weights.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None,
                drop_pad_mode=2
            )
        else:
            final_hidden_states = final_hidden_states + zero_expert_output

        return final_hidden_states

    def _forward_decode_norm(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        if model_extra_config.operator_opt_config.use_super_kernel:
            with tng.scope.super_kernel(self.prefix, 'stream-fusion=1'):
                return self._forward_decode_dispatch_combine(hidden_states, attn_metadata)
        else:
            return self._forward_decode_dispatch_combine(hidden_states, attn_metadata)

    def _forward_decode_dispatch_combine(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        router_logits, _ = self.router.forward(hidden_states.float())
        topk_weights, topk_ids = self.router.get_topk_indices(router_logits)
        zero_expert_output, topk_ids, topk_weights = self.compute_zero_experts(hidden_states, topk_weights, topk_ids)

        mc2_mask = attn_metadata.decode.mc2_mask if attn_metadata is not None and attn_metadata.decode is not None else None
        layer = self.experts

        max_num_deployed_expert = self.local_expert_num * get_dp_group().world_size
        act_dtype = hidden_states.dtype
        shared_expert_rank_num = 0
        kwargs = {
            "x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,  # Set it to 0 for now
            "shared_expert_rank_num": shared_expert_rank_num,  # 32
            "moe_expert_num": max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }

        experts_tp_size = layer.tp_size
        world_size = get_world_group().world_size
        # In fact, what we get is the die number, and the ep group is not adapted by default.
        # The default ep group is experts_num/die_num.
        global_rank = get_world_group().rank_in_group
        all_to_all_group_size = world_size // experts_tp_size

        kwargs.update({
            "scales": None,  # Quantization coefficient
            "quant_mode": layer.quant_mode,  # 0: Non-quantization; 1: Static quantization; 2: Dynamic quantization
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        })

        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

        group_list = expert_token_nums.to(torch.int64)

        # cal experts
        weight1_3 = self.experts.w13_weight
        weight2 = self.experts.w2_weight

        # bf16
        gate_up_proj = torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=None, group_list=group_list,
                                                split_item=3, group_type=0, group_list_type=1)[0]
    
        gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

        hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2],bias=None,
                                        group_list=group_list, split_item=3, output_dtype=act_dtype,
                                        group_type=0, group_list_type=1)[0]

        # moeCombine
        kwargs = {
            "expand_x": hidden_states_experts,
            "expert_ids": topk_ids,  # [n*topk]
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,
            "moe_expert_num":  max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }
        tp_recv_counts = output[5]
        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,  # dispatch's send_counts
            "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": all_to_all_group_size,
            "ep_rank_id": global_rank // experts_tp_size,
            "tp_send_counts": tp_recv_counts,
            "group_tp": layer.moe_rs_group_name,
            "tp_world_size": experts_tp_size,
            "tp_rank_id": global_rank % experts_tp_size,
            "x_active_mask": mc2_mask,
        }
        kwargs.update(stage3_kwargs)


        hidden_states_route = torch_npu.npu_moe_distribute_combine_v2(**kwargs)

        final_hidden_states = (hidden_states_route, zero_expert_output)
        return final_hidden_states
