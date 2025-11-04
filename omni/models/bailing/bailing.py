# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
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
"""Inference-only BaiLing model."""
from typing import Iterable, List, Optional, Set, Tuple, Union
import torch
from torch import nn
from transformers import PretrainedConfig
import torch.distributed as dist
import torchair as tng
torch._logging.set_logs(recompiles=True)
# vllm adaptor
from vllm.platforms import current_platform
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.attention import AttentionMetadata, Attention
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_rank,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import (
    PPMissingLayer, 
    is_pp_missing_parameter, 
    make_layers, 
    make_empty_intermediate_tensors_factory,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from omni.layers.linear import (
    AscendMergedColumnParallelLinear,
    AscendRowParallelLinear,
)
from omni.layers.linear import (
    QKVParallelFlashCommLinear,
    RowParallelFlashCommLinear
)
from omni.layers.vocab_parallel_embedding import (
    ParallelLMHead, 
    VocabParallelEmbedding
)
from omni.layers.activation import SiluAndMul
from omni.layers.layernorm import RMSNorm
from omni.adaptors.vllm.distributed.parallel_state import (
    get_mlp_tp_group,
)
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.rotary_embedding import get_rope
from omni.layers.moe.fused_moe.layer import FusedMoE
from omni.layers.moe.deepseek_moe import DeepseekMoE
from omni.models.config_loader.loader import model_extra_config

if model_extra_config.operator_opt_config.unquant_bmm_nz:
    # if use weight nz, this config must be True
    torch.npu.config.allow_internal_format = True

"""MLP module activation split length, split by 64G VRAM, need to confirm the optimal split length based on sequence length and performance"""
SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER = 64


class ParallelDeepseekMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.gate_up_proj = AscendMergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            tp_size=get_mlp_tp_group().world_size,
            tp_rank=get_mlp_tp_group().rank_in_group,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = AscendRowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           tp_size=get_mlp_tp_group().world_size,
                                           tp_rank=get_mlp_tp_group().rank_in_group,
                                           quant_config=quant_config,
                                           reduce_results=False,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False

    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)


    def forward(self, x, residual, attn_metadata, layerid=None):
        x = get_mlp_tp_group().all_gather(x, dim=0)

        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        # P and D are both cut, and are concave at the node (16)
        x = get_mlp_tp_group().reduce_scatter(x)
        return x, residual


class BaiLingAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = QKVParallelFlashCommLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=config.use_qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.query_key_value"
        )
        self.dense = RowParallelFlashCommLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=config.use_qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense"
        )

        if hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        elif hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        else:
            self.rotary_dim = self.head_dim

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn")
        
        self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        is_prefill = attn_metadata is None or not attn_metadata.is_pd_seperate_d
        qkv, _ = self.query_key_value(hidden_states, x_transform='AG', is_prefill = is_prefill)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.query_layernorm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                            self.head_dim)
        k_by_head = self.key_layernorm(k_by_head)
        k = k_by_head.view(k.shape)

        if attn_metadata is None:
            cos, sin = self.rotary_emb.get_cos_sin(positions)
        else:
            cos = attn_metadata.cos
            sin = attn_metadata.sin

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.dense(attn_output, reduce_type="RS")
        return output

class BaiLingDecoderLayer(nn.Module):

    is_split_hidden_states = False

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        layer_idx = int(prefix.split(sep='.')[-1])
        self.prefix = prefix
        self.hidden_size = config.hidden_size

        self.layer_name = f"{prefix}.self_attn.attn"
        self.quant_symbol = quant_config is not None

        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        
        self.self_attn = BaiLingAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)

        if config.num_experts is not None and layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            self.mlp = ParallelDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False
        
    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            layer_id: Optional[int] = None
    ) -> torch.Tensor:
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
        is_prefill = attn_metadata is None or not attn_metadata.is_pd_seperate_d

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, quant_symbol=self.quant_symbol)
            # Adapt end.
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        if self.is_moe == True and not is_prefill and model_extra_config.operator_opt_config.use_super_kernel:
            with tng.scope.super_kernel(self.mlp.prefix, 'stream-fusion=1'):
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Perform full hidden splitting to avoid OOM
        if get_dp_group().world_size > 1 and is_prefill \
            and not model_extra_config.parall_config.enable_attn_ffn_disaggregation:
            # During prefill, chunk is only triggered when an extremely large number of identical tokens is detected — to prevent GMM from OOM. 
            # Prefill performance may degrade slightly as a trade-off. 
            # For longer sequences (e.g., >256K or 512K tokens), consider adjusting SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER to optimize memory usage or avoid OOM.
            reduce_length = torch.tensor(hidden_states.shape[0], dtype=torch.int64, device=current_platform.device_type)
            local_length = hidden_states.shape[0]
            # global_max_length = torch.tensor(0, dtype=torch.int64)
            dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
            global_max_length = reduce_length.item()
            pad_size = global_max_length - hidden_states.shape[0]
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, pad_size)
            )
            residual = torch.nn.functional.pad(
                residual, (0, 0, 0, pad_size)
            )
            hidden_states_list = hidden_states.split(SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER)
            residual_list = residual.split(SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER)
            hidden_state_out = []
            residual_out = []
            for i in range(len(hidden_states_list)):
                hidden_states, residual = self.mlp(hidden_states_list[i], residual_list[i], attn_metadata, layer_id)
                hidden_state_out.append(hidden_states)
                residual_out.append(residual)
            hidden_states = torch.cat(hidden_state_out)[:local_length]
            residual = torch.cat(residual_out)[:local_length]
        else:
            if self.is_moe == True:
                # omni placement do not support super kernel
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id)
                if isinstance(hidden_states, (tuple, list)):
                    assert len(hidden_states) == 2
                    hidden_states = hidden_states[0] + hidden_states[1]
            else:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata)

        return hidden_states, residual

class BaiLingModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):

        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.prefix = f"{prefix}.layers"
        self.postfix = ".self_attn.attn"
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        if get_pp_group().is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: BaiLingDecoderLayer(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids, reduce=1)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors]
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            layer_id = i - self.first_k_dense_replace

            hidden_states, residual = layer(positions,
                                            hidden_states,
                                            kv_caches[i - self.start_layer] if kv_caches is not None else None,
                                            attn_metadata,
                                            residual,
                                            layer_id)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        return hidden_states


@support_torch_compile
class BailingMoeV2ForCausalLM(nn.Module):

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = BaiLingModel(vllm_config=vllm_config, prefix="model")

        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      quant_config=self.quant_config,
									  parallel_lmhead=(get_dp_group().world_size > 1))
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.return_hidden_states = True

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor] = None,
            attn_metadata: Union[AttentionMetadata, dict] = None,
            selected_indices: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds = None,
            **kwargs
    ) -> Optional[torch.Tensor]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        
        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            logits = self.compute_lmhead(hidden_states, selected_indices)

        if self.return_hidden_states:
            return hidden_states, logits
        else:
            return logits

    def compute_lmhead(
            self,
            hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if selected_indices is not None:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if hidden_states.shape[0] != selected_indices.shape[0]:
                hidden_states = hidden_states.index_select(0, selected_indices)

        # Get the logits for the next tokens.
        logits = self.lm_head(hidden_states, embedding_bias)
        return logits

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        return logits

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
            "residual":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    if "attention" in name and "post_attention" not in name:
                        name = name.replace("attention", "self_attn")
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]

        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly