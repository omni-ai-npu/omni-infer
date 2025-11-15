# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Iterable
from typing import Any, Optional, Union, List

import torch
import torch_npu
import itertools
from torch import nn
from transformers import Qwen2Config
from transformers import PretrainedConfig

from vllm.forward_context import get_forward_context, set_forward_context
from vllm.attention import Attention, AttentionType, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig, ModelConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group, get_ep_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from omni.layers.layernorm import RMSNormFlashComm
from omni.layers.linear import (RowParallelFlashCommLinear, QKVParallelFlashCommLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from omni.layers.rotary_embedding import get_rope
from omni.layers.fused_mlp import FusedMLP
from omni.layers.attention.backend.attention import AscendAttentionState
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix, extract_layer_index)
from vllm.model_executor.layers.linear import ReplicatedLinear

import math
from vllm.platforms import current_platform
import torch.distributed as dist
from torch.nn.parameter import Parameter
from vllm.model_executor.utils import set_weight_attrs


logger = init_logger(__name__)

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)

class GptOssExperts(torch.nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        prefix: str = "",
    ):
        super().__init__()
        self.num_total_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.ep_size = get_ep_group().world_size
        self.ep_rank = get_ep_group().rank_in_group
        assert self.num_total_experts % self.ep_size == 0
        
        self.num_experts = self.num_total_experts // self.ep_size
        self.experts_start_idx = self.ep_rank * self.num_experts
        self.experts_end_idx = self.experts_start_idx + self.num_experts
        
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size, config.intermediate_size * 2), dtype=torch.int8),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.gate_up_proj_scale = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size * 2), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj_scale, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.gate_up_proj_bias = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size * 2), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj_bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size, config.hidden_size), dtype=torch.int8),
            requires_grad=False)
        set_weight_attrs(self.down_proj, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj_scale = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.down_proj_scale, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj_bias = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.down_proj_bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.ep_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
                
    def process_weights_after_loading(self) -> None:
        gate_up_proj = torch_npu.npu_format_cast(self.gate_up_proj.data, 29)
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj, requires_grad=False)
        down_proj = torch_npu.npu_format_cast(self.down_proj.data, 29)
        self.down_proj = torch.nn.Parameter(down_proj, requires_grad=False)


    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:

        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(router_logits, self.experts_per_token,
                                                                    k_group=1, group_count=1, group_select_mode=1)
        topk_weights_sum = torch.sum(topk_weights, dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum

        hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        
        expert_range = [self.experts_start_idx, self.experts_end_idx]
        sorted_tokens, expanded_x_idx, expert_tokens, dynamic_quant_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states_int8, topk_ids, scale=pertoken_scale, offset=None, active_num=topk_ids.numel(), expert_capacity=-1, expert_num=self.num_total_experts, 
            drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=-1,active_expert_range=expert_range, row_idx_type=0)

        mask = (topk_ids >= self.experts_start_idx)   
        mask2 = (topk_ids < self.experts_start_idx)
        topk_ids_clamped =  mask2 * self.num_total_experts + mask * topk_ids
        topk_ids_sorted, _ = torch.sort(topk_ids_clamped.view(-1))
        topk_ids_sorted = torch.clamp(topk_ids_sorted, min=self.experts_start_idx, max=self.experts_end_idx-1).view(-1)
        topk_ids_sorted = topk_ids_sorted - self.experts_start_idx

        gate_up_proj_output = torch_npu.npu_grouped_matmul([sorted_tokens], [self.gate_up_proj], bias=None, group_list=expert_tokens,
                                                    scale=[self.gate_up_proj_scale],
                                                    per_token_scale=[dynamic_quant_scale],
                                                    split_item=3, output_dtype=torch.bfloat16, group_type=0,
                                                    group_list_type=1)[0]
        
        gate_up_proj_output += torch.index_select(self.gate_up_proj_bias, dim=0, index=topk_ids_sorted)
        
        gate_up_proj_output = swiglu(gate_up_proj_output, limit=self.swiglu_limit)
        
        gate_up_proj_output_int8, pertoken_scale = torch_npu.npu_dynamic_quant(gate_up_proj_output)
        down_proj_output = torch_npu.npu_grouped_matmul([gate_up_proj_output_int8], [self.down_proj], scale=[self.down_proj_scale], 
                                            per_token_scale=[pertoken_scale], bias=None,
                                            group_list=expert_tokens, split_item=3, output_dtype=torch.bfloat16,
                                            group_type=0, group_list_type=1)[0]

        down_proj_output += torch.index_select(self.down_proj_bias, dim=0, index=topk_ids_sorted)

        output = torch_npu.npu_moe_finalize_routing(down_proj_output.unsqueeze(1), None, None, None,
                                                    topk_weights.to(torch.float32),
                                                    expanded_x_idx, topk_ids, drop_pad_mode=3)

        return output
    
class GptOssExpertsBF16(torch.nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        prefix: str = "",
    ):
        super().__init__()
        self.num_total_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.ep_size = get_ep_group().world_size
        self.ep_rank = get_ep_group().rank_in_group
        assert self.num_total_experts % self.ep_size == 0
        
        self.num_experts = self.num_total_experts // self.ep_size
        self.experts_start_idx = self.ep_rank * self.num_experts
        self.experts_end_idx = self.experts_start_idx + self.num_experts
        
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size, config.intermediate_size * 2), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.gate_up_proj_bias = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size * 2), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj_bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size, config.hidden_size), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.down_proj, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj_bias = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.down_proj_bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        
        
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.ep_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(router_logits, self.experts_per_token,
                                                                    k_group=1, group_count=1, group_select_mode=1)
        topk_weights_sum = torch.sum(topk_weights, dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
      
        expert_range = [self.experts_start_idx, self.experts_end_idx]
        sorted_tokens, expanded_x_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states, topk_ids, offset=None, active_num=topk_ids.numel(), expert_capacity=-1, expert_num=self.num_total_experts, 
            drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=-1,active_expert_range=expert_range, row_idx_type=0)
        
        row_index = expanded_x_idx // topk_ids.shape[-1]
        row_index = row_index.to(torch.int64)
        
        gate_up_proj_output = torch_npu.npu_grouped_matmul([sorted_tokens], [self.gate_up_proj], 
                                                            bias=[self.gate_up_proj_bias.to(torch.float)], group_list=expert_tokens,
                                                            split_item=3, output_dtype=torch.bfloat16, group_type=0,
                                                            group_list_type=1)[0]
        
        gate_up_proj_output = swiglu(gate_up_proj_output, limit=self.swiglu_limit)
        
        down_proj_output = torch_npu.npu_grouped_matmul([gate_up_proj_output], [self.down_proj],
                                            bias=[self.down_proj_bias.to(torch.float)],
                                            group_list=expert_tokens, split_item=3, output_dtype=torch.bfloat16,
                                            group_type=0, group_list_type=1)[0]
        
        output = torch_npu.npu_moe_finalize_routing(down_proj_output.unsqueeze(1), None, None, None,
                                                    topk_weights.to(torch.float32),
                                                    expanded_x_idx, topk_ids, drop_pad_mode=3)
        
        return output

class GptOssMoE(torch.nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        prefix: str = "",
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.ep_size = get_ep_group().world_size
        
        self.gating_dtype = torch.float32
        self.router = ReplicatedLinear(config.hidden_size,
                                     config.num_local_experts,
                                     bias=True,
                                     quant_config=None,
                                     params_dtype=self.gating_dtype,
                                     prefix=f"{prefix}.router")
        
        if hasattr(config, "quantization_config"):
            self.experts = GptOssExperts(config, prefix=f"{prefix}.experts")
        else :
            self.experts = GptOssExpertsBF16(config, prefix=f"{prefix}.experts")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        router_logits, _ = self.router(hidden_states.to(self.gating_dtype))
        output = self.experts(hidden_states, router_logits)
        if self.ep_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output

class GptOssAttention(nn.Module):

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config

        # attention form
        self.layer_idx = extract_layer_index(prefix)
        self.attn_form = config.layer_types[self.layer_idx] # sliding window or full attention
        cache_config.sliding_window = config.sliding_window if "sliding" in self.attn_form else None
        self.max_model_len = model_config.max_model_len

        # attention sink parameter
        self.sinks = torch.nn.Parameter(torch.empty((self.num_heads), dtype=torch.bfloat16), requires_grad=False)
        set_weight_attrs(self.sinks, {"output_dim": 0, "weight_loader": self.weight_loader})

        self.qkv_proj = QKVParallelFlashCommLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
            params_dtype=torch.bfloat16
        )
        self.o_proj = RowParallelFlashCommLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
            params_dtype=torch.bfloat16
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=torch.float32,
            rope_scaling={
                "rope_type":
                "yarn",
                "factor":
                config.rope_scaling["factor"],
                "original_max_position_embeddings":
                config.rope_scaling["original_max_position_embeddings"],
                "beta_fast":
                config.rope_scaling["beta_fast"],
                "beta_slow":
                config.rope_scaling["beta_slow"],
            },
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            attn_sinks=self.sinks,
            prefix=f"{prefix}.attn")

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

class GptOssDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.layer_name = f"{prefix}.self_attn.attn"
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = GptOssAttention(
            config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        
        self.mlp = GptOssMoE(
            config,
            prefix=f"{prefix}.mlp",
        )
        
        self.input_layernorm = RMSNormFlashComm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFlashComm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor],
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            cos=cos,
            sin=sin
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GptOssModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 model_config: Optional[ModelConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "model",
                 decoder_layer_type: type[nn.Module] = GptOssDecoderLayer):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                ))

        self.config = config
        self.model_config = model_config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        base = getattr(config, "rope_theta", 1000000)
        rotary_dim = getattr(config, "head_dim", 64)
        max_len = config.max_position_embeddings
        full_cos, full_sin = None, None
        self.register_buffer("full_cos", full_cos, persistent=False)
        self.register_buffer("full_sin", full_sin, persistent=False)

        # Use the provided decoder layer type or default to GptOssDecoderLayer
        decoder_layer_type = decoder_layer_type or GptOssDecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config,
                                              model_config,
                                              cache_config,
                                              quant_config,
                                              prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNormFlashComm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        cos, sin = None, None

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None and isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata['model.layers.0.self_attn.attn']

        if attn_metadata is not None and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            hidden_states, residual = self.forward_layers(positions, hidden_states, residual, kv_caches, cos, sin)
        else:
            hidden_states, residual = self.forward_layers(positions, hidden_states, residual, kv_caches, cos, sin)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual, y_transform=None)
        return hidden_states

    def forward_layers(self, positions, hidden_states, residual, kv_caches, cos, sin):
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_caches[layer_idx] if kv_caches is not None else None,
                cos, sin
            )
        return hidden_states, residual

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.endswith(".dequant_scale") and name not in params_dict:
                name = name.replace("dequant_scale", "weight_scale")
            if name.endswith(".down_proj.scale") and name not in params_dict:
                name = name.replace("down_proj.scale", "down_proj_scale")
            if name.endswith(".gate_up_proj.scale") and name not in params_dict:
                name = name.replace("gate_up_proj.scale", "gate_up_proj_scale")
            if name.endswith(".down_proj.bias") and name not in params_dict:
                name = name.replace("down_proj.bias", "down_proj_bias")
            if name.endswith(".gate_up_proj.bias") and name not in params_dict:
                name = name.replace("gate_up_proj.bias", "gate_up_proj_bias")
            if name.endswith(".down_proj.weight") and name not in params_dict:
                name = name.replace("down_proj.weight", "down_proj")
            if name.endswith(".gate_up_proj.weight") and name not in params_dict:
                name = name.replace("gate_up_proj.weight", "gate_up_proj")
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:     
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        print(loaded_params)
        
        return loaded_params

@support_torch_compile
class GptOssForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        lora_config = None

        self.config = vllm_config.model_config.hf_config
        self.lora_config = lora_config

        self.quant_config = vllm_config.quant_config
        self.model = GptOssModel(self.config,
                                 vllm_config.model_config,
                                 vllm_config.cache_config,
                                 vllm_config.quant_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.sampler = Sampler()

        if get_pp_group().is_last_rank:
            if self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(self.config.vocab_size,
                                              self.config.hidden_size,
                                              quant_config=None,
                                              params_dtype=torch.bfloat16,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"),
                                              parallel_lmhead=False)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor] = None,
        attn_metadata: AttentionMetadata = None,
        selected_indices: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, None)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
    
    def process_weights_after_loading(self):
        for _, module in self.model.named_modules():
            if isinstance(module, GptOssExperts):
                module.process_weights_after_loading()

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly