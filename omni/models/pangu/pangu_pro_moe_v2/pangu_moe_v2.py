#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
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

from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torchair as tng
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch import nn
from torch.nn import Parameter
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, CompilationLevel
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import get_dp_group, get_tp_group, get_world_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, sharded_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.utils import supports_dynamo
from omni.layers.activation import SiluAndMul
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.layernorm import RMSNorm
from omni.layers.linear import (
    AscendMergedColumnParallelLinear,
    AscendRowParallelLinear,
    ColumnParallelFlashCommLinear,
    RowParallelFlashCommLinear,
)
from omni.layers.moe.deepseek_moe import DeepseekMoE
from omni.layers.moe.fused_moe.layer import FusedMoE
from omni.layers.rotary_embedding import get_rope
from omni.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

from omni.models.config_loader.loader import model_extra_config
from omni.layers.utils import ConditionalTNGScope

logger = init_logger(__name__)

_ROUTER_SCALE = None
MAX_PREFETCH_SIZE = 56

class CustomQKVRearrangeColumnParallelLinear(ColumnParallelFlashCommLinear):
    def __init__(
        self,
        config,
        hidden_size: int,
        head_size: int,
        v_channels: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""):
        self.config = config
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_channels = v_channels
        self.total_num_heads = total_num_heads

        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        # Divide the weight matrix along the last dimension.
        self.prefix = prefix
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * tp_size * self.v_channels
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_channels * tp_size,  # v_proj
        ]

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         tp_size=tp_size,
                         tp_rank=tp_rank,
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_rearrange(self, loaded_weight):
        tp_size = get_tp_group().world_size
        # if tp_size is 1, we do not need to rearrange the weight
        if tp_size == 1:
            return loaded_weight
        else:
            qk_nope_dim = getattr(self.config, "qk_nope_dim", None)
            qk_rope_dim = getattr(self.config, "qk_rope_dim", None)
            v_channels = getattr(self.config, "v_channels", None)
            num_kv_heads = getattr(self.config, "num_key_value_heads", None)
            num_heads = getattr(self.config, "num_attention_heads", None)
            head_dim = qk_nope_dim + qk_rope_dim

            q_size = num_heads * head_dim
            k_size = num_kv_heads * head_dim
            v_size = num_kv_heads * v_channels
            q_weight, k_weight, v_weight = loaded_weight.split([q_size, k_size, v_size], dim=0)

            q_origin_dim = q_weight.size(0)
            k_origin_dim = k_weight.size(0)
            v_origin_dim = v_weight.size(0)
            assert q_origin_dim % tp_size == 0, f"tp_size is not correct. tp_size {tp_size} must be divisible by q_origin_dim {q_origin_dim}"
            assert k_origin_dim % tp_size == 0, f"tp_size is not correct. tp_size {tp_size} must be divisible by k_origin_dim {k_origin_dim}"
            assert v_origin_dim % tp_size == 0, f"tp_size is not correct. tp_size {tp_size} must be divisible by v_origin_dim {v_origin_dim}"

            q_weight = q_weight.reshape(tp_size, q_origin_dim//tp_size, -1)
            k_weight = k_weight.reshape(tp_size, k_origin_dim//tp_size, -1)
            v_weight = v_weight.reshape(tp_size, v_origin_dim//tp_size, -1)

            loaded_weight_rearrange = []
            for i in range(tp_size):
                loaded_weight_rearrange.append(q_weight[i])
                loaded_weight_rearrange.append(k_weight[i])
                loaded_weight_rearrange.append(v_weight[i])

            loaded_weight_rearrange = torch.cat(loaded_weight_rearrange, dim=0)

            return loaded_weight_rearrange

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        # Based on tp_size, rearrange the pattern from q1q2k1k2v1v2 to q1k1v1q2k2v2.
        loaded_weight_rearrange = self.weight_rearrange(loaded_weight)
        super().weight_loader(param, loaded_weight_rearrange)

    def weight_loader_v2(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        # Based on tp_size, rearrange the pattern from q1q2k1k2v1v2 to q1k1v1q2k2v2.
        loaded_weight_rearrange = self.weight_rearrange(loaded_weight)
        super().weight_loader_v2(param, loaded_weight_rearrange)


class ParallelPanguProMoEMLP(nn.Module):

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
            tp_size=get_tp_group().world_size,
            tp_rank=get_tp_group().rank_in_group,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = AscendRowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           tp_size=get_tp_group().world_size,
                                           tp_rank=get_tp_group().rank_in_group,
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
        x = get_tp_group().all_gather(x, dim=0)

        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        # P and D are both cut, and are concave at the node (16)
        x = get_tp_group().reduce_scatter(x)
        return x, residual

def Attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    # For some alternate attention backends like MLA the attention output
    # shape does not match the query shape, so we optionally let the model
    # definition specify the output tensor shape.
    output_shape: Optional[torch.Size] = None,
    # patch for pangu 72Bv2 with attention sink
    sink_pad_params: Optional[dict] = None,
    sink_query: Optional[torch.Tensor] = None,
    sink_key: Optional[torch.Tensor] = None,
    sink_value: Optional[torch.Tensor] = None,
    v_head_size: Optional[int] = None,
) -> torch.Tensor:
    """
    The KV cache is stored inside this class and is accessed via
    `self.kv_cache`.

    Attention metadata (`attn_metadata`) is set using a context manager in
    the model runner's `execute_model` method. It is accessed via forward
    context using
    `vllm.forward_context.get_forward_context().attn_metadata`.
    """
    if self.calculate_kv_scales:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata.enable_kv_scales_calculation:
            self.calc_kv_scales(query, key, value)
    # print("self.use_output_test", self.use_output)   #true
    if self.use_output:
        output_shape = (output_shape
                        if output_shape is not None else query.shape)
        output = torch.empty(output_shape,
                                dtype=query.dtype,
                                device=query.device)
        hidden_size = output_shape[-1]
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        # We skip reshaping query, key and value tensors for the MLA
        # backend since these tensors have different semantics and are
        # processed differently.
        if not self.use_mla:
            # Reshape the query, key, and value tensors.
            # NOTE(woosuk): We do this outside the custom op to minimize the
            # CPU overheads from the non-CUDA-graph regions.
            query = query.view(-1, self.num_heads, self.head_size)
            # patch for pangu 72Bv2: v head size is different from q and k
            if v_head_size is None:
                v_head_size = hidden_size // self.num_heads
            output = output.view(-1, self.num_heads, v_head_size)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                # patch for pangu 72Bv2: v head size is different from q and k
                value = value.view(-1, self.num_kv_heads, value.shape[-1] // self.num_kv_heads)
        if self.use_direct_call:
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            self.impl.forward(self,
                                query,
                                key,
                                value,
                                self_kv_cache,
                                attn_metadata,
                                output=output,
                                # patch for pangu 72Bv2 with attention sink
                                **(dict(sink_query=sink_query,
                                sink_key=sink_key,
                                sink_value=sink_value,
                                sink_pad_params=sink_pad_params,
                                v_head_size=v_head_size) if sink_query is not None else {}))
        else:
            torch.ops.vllm.unified_attention_with_output(
                query, key, value, output, self.layer_name)
        return output.view(-1, hidden_size)
    else:
        if self.use_direct_call:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            return self.impl.forward(self, query, key, value,
                                        self_kv_cache, attn_metadata)
        else:
            return torch.ops.vllm.unified_attention(
                query, key, value, self.layer_name)

class PanguProMoEV2Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(f"self.total_num_heads % tp_size must be 0, but it is {self.total_num_heads % tp_size}")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError("self.total_num_kv_heads % tp_size must be 0, "
                                 f"but it is {self.total_num_kv_heads % tp_size}")
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            if tp_size % self.total_num_kv_heads != 0:
                raise ValueError("self.total_num_kv_heads % tp_size must be 0, "
                                 f"but it is {tp_size % self.total_num_kv_heads}")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.qk_nope_dim = getattr(config, "qk_nope_dim", None)
        self.qk_rope_dim = getattr(config, "qk_rope_dim", None)
        self.v_channels = getattr(config, "v_channels", None)
        self.head_dim = self.qk_nope_dim + self.qk_rope_dim
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_channels
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.param_sink_with_value = getattr(config, "param_sink_with_value", False)
        self.param_sink_scalar = getattr(config, "param_sink_scalar", None)
        self.param_sink_of_head_num = getattr(config, "param_sink_of_head_num", False)

        self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.qkv_proj = CustomQKVRearrangeColumnParallelLinear(
            config,
            self.hidden_size,
            self.head_dim,
            self.v_channels,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj"
        )

        self.o_proj = RowParallelFlashCommLinear(
            self.total_num_heads * self.v_channels,
            self.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj"
        )

        if rope_scaling is None:
            rope_scaling = {'factor': '0'}
        rope_scaling["rope_type"] = 'pangu_pro_moe'
        
        # native support for partial rope: qk[:64]
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.qk_rope_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        Attention.forward = Attention_forward

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        if self.param_sink_number > 0:
            self.param_sink_query = torch.zeros((
                    self.param_sink_number, 
                    self.num_heads, 
                    self.head_dim),
                device=torch.npu.current_device(),
                dtype=config.torch_dtype,
            )
            if self.param_sink_of_head_num:
                self.param_sink_num_heads_per_partition = self.num_heads
                self.q_mult = divide(self.num_heads, self.num_kv_heads)
            else:
                self.param_sink_num_heads_per_partition = self.num_kv_heads
            if self.param_sink_scalar:
                self.param_sink_key_zero_pad = torch.zeros((
                        self.param_sink_number, 
                        self.param_sink_num_heads_per_partition, 
                        self.param_sink_scalar - 1),
                    device=torch.npu.current_device(),
                    dtype=config.torch_dtype,
                )
                self.param_sink_key = torch.nn.Parameter(
                    torch.empty(
                        (self.param_sink_number, self.param_sink_num_heads_per_partition),
                        device=torch.npu.current_device(),
                        dtype=config.torch_dtype,
                    )
                )
                setattr(self.param_sink_key, 'allreduce', True)
            else:
                self.param_sink_key = torch.nn.Parameter(
                    torch.empty((
                            self.param_sink_number, 
                            self.param_sink_num_heads_per_partition, 
                            self.head_dim),
                        device=torch.npu.current_device(),
                        dtype=config.torch_dtype,
                    )
                )
                setattr(self.param_sink_key, 'allreduce', True)
            if self.param_sink_with_value:
                self.param_sink_value = torch.nn.Parameter(
                    torch.empty((
                            self.param_sink_number, 
                            self.param_sink_num_heads_per_partition, 
                            self.v_channels),
                        device=torch.npu.current_device(),
                        dtype=config.torch_dtype,
                    )
                )
                setattr(self.param_sink_value, 'allreduce', True)
            else:
                self.param_sink_value = torch.zeros((
                        self.param_sink_number, 
                        self.param_sink_num_heads_per_partition, 
                        self.v_channels),
                    device=torch.npu.current_device(),
                    dtype=config.torch_dtype,
                )
        self.enable_sink = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sink_pad_params: Optional[dict] = None,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        is_prefill = attn_metadata is None or not attn_metadata.is_pd_seperate_d
        qkv, _ = self.qkv_proj(hidden_states, x_transform='AG', is_prefill=is_prefill)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        k = self.k_layernorm(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary_emb(positions, q.contiguous(), k, cos, sin)

        q = q.view(-1, self.q_size)
        k = k.view(-1, self.k_size)

        # pad v and attention sink after kv cache update in pangu_infer/patches/vllm_ascend/attention/attention_v1.py
        param_sink_key = self.param_sink_key
        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata

        if self.param_sink_number > 0 and attn_metadata is not None:
            if hasattr(self, 'k_layernorm') and self.k_layernorm is not None:
                param_sink_key = self.k_layernorm(self.param_sink_key)
            self.enable_sink = True

        attn_output = self.attn(
            q,
            k,
            v,
            output_shape=(q.shape[0], self.num_heads * self.v_channels),
            v_head_size=self.v_channels,
            **(dict(
                sink_query=self.param_sink_query,
                sink_key=param_sink_key,
                sink_value=self.param_sink_value,
                sink_pad_params=sink_pad_params,
            ) if self.enable_sink and attn_metadata is not None else {}),
        )

        attn_output = attn_output.reshape(-1, self.num_heads * self.v_channels)
        output, _ = self.o_proj(attn_output, reduce_type="RS")
        return output


def DeepseekMoE_forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata, layer_id: int, next_attention_weights: Optional[dict]=None, is_hybrid_chunked_prefill_graph_mode=False) -> torch.Tensor:
        # when is_hybrid_chunked_prefill_graph_mode is True, enable chunkprefill
        if is_hybrid_chunked_prefill_graph_mode:
            if attn_metadata is None or attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                is_prefill = True
            else:
                is_prefill = False
        else:
            if attn_metadata is None or attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                is_prefill = True
            else:
                is_prefill = False
         
        if not self.is_init_gate:
            self.gate.weight.data = torch_npu.npu_format_cast(self.gate.weight.data, 2)
            self.is_init_gate = True
        if is_prefill:
            return self._forward_prefill_norm(hidden_states, residual, attn_metadata)
        else:
            return self._forward_decode_norm(hidden_states, residual, attn_metadata, layer_id, next_attention_weights)

class PanguProMoEDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = PanguProMoEV2Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # `mlp_only_layers` in the config.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_name = f"{prefix}.self_attn.attn"
        self.quant_symbol = quant_config is not None

        mlp_only_layers = [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        if (layer_idx not in mlp_only_layers) and (config.num_experts > 0):
            DeepseekMoE.forward = DeepseekMoE_forward
            self.mlp = DeepseekMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            self.mlp = ParallelPanguProMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, 'sandwich_norm', False):
            self.sandwich_norm = True
            self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.sandwich_norm = False

        self.enable_torchair_graph_mode = (
                    vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())
        self.is_pd_seperate_d = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        self.is_hybrid_chunked_prefill_graph_mode = self.enable_torchair_graph_mode and not self.is_pd_seperate_d and \
            not vllm_config.additional_config.get("enable_hybrid_graph_mode", False) and vllm_config.scheduler_config.enable_chunked_prefill

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        sink_pad_params: Optional[dict] = None,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        layer_id: Optional[int] = None,
        next_attn_weights: Optional[dict] = None,
        next_input_layernorm: Optional[nn.Module] = None
    ) -> torch.Tensor:

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
            
        is_prefill = attn_metadata is None or not attn_metadata.is_pd_seperate_d
        enable_superkernel = not is_prefill and model_extra_config.operator_opt_config.use_super_kernel

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif not enable_superkernel:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, quant_symbol=self.quant_symbol)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            sink_pad_params=sink_pad_params,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        
        with ConditionalTNGScope(super_kernel=enable_superkernel, scope='superkernel_decode_layer'):
            if model_extra_config.operator_opt_config.use_prefetch:
                if self.is_moe:
                    torch_npu.npu_prefetch(self.mlp.gate.weight, hidden_states, model_extra_config.operator_opt_config.dense_mlp_prefetch * 1024 * 1024)
                else:
                    torch_npu.npu_prefetch(self.mlp.gate_up_proj.weight, hidden_states, model_extra_config.operator_opt_config.dense_mlp_prefetch * 1024 * 1024)

            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, residual = self.pre_mlp_layernorm(hidden_states, residual)
            
            if self.is_moe == True:
                # omni placement do not support super kernel
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id, next_attn_weights, is_hybrid_chunked_prefill_graph_mode=self.is_hybrid_chunked_prefill_graph_mode)
                if isinstance(hidden_states, (tuple, list)):
                    assert len(hidden_states) == 2
                    hidden_states = hidden_states[0] + hidden_states[1]
            else:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata)

            hidden_states = self.post_mlp_layernorm(hidden_states)

            if enable_superkernel and next_input_layernorm is not None:
                hidden_states, residual = next_input_layernorm(
                    hidden_states, residual, quant_symbol=self.quant_symbol
                )

        return hidden_states, residual


class PanguProMoEModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.prefix = f"{prefix}.layers"
        self.postfix = ".self_attn.attn"

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: PanguProMoEDecoderLayer(
                config=config,
                vllm_config=vllm_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        lm_head=None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # print(f"get_world_group().world_size {get_world_group().world_size}" , flush = True)
        if get_pp_group().is_first_rank:
            hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        sink_pad_params = None

        if attn_metadata is None :
            cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(positions)
        else:
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[next(iter(attn_metadata))]
            cos = attn_metadata.cos
            sin = attn_metadata.sin

            sink_pad_params = {}
            block_tables = F.pad(attn_metadata.block_tables, (1, 0, 0, 0), value=0)
            actual_seq_lengths_kv = attn_metadata.seq_lens + 128
            torch._dynamo.mark_static(block_tables)
            torch._dynamo.mark_static(actual_seq_lengths_kv)
            sink_pad_params['sink_block_tables'] = block_tables
            sink_pad_params['sink_actual_seq_lengths_kv'] = actual_seq_lengths_kv

        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            layer_id = layer_idx - 4
            if layer_idx < self.end_layer - 1:
                next_attn_weights = {
                    'qkv_proj_weight': self.layers[layer_idx + 1].self_attn.qkv_proj.weight,
                    'o_proj_weight': self.layers[layer_idx + 1].self_attn.o_proj.weight,
                }
            else:
                next_attn_weights = None

            if layer_idx < self.end_layer - 1:
                next_input_layernorm = self.layers[layer_idx + 1].input_layernorm
            else:
                next_input_layernorm = None

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                cos,
                sin,
                sink_pad_params,
                kv_caches[layer_idx - self.start_layer] if kv_caches is not None else None,
                attn_metadata,
                layer_id,
                next_attn_weights,
                next_input_layernorm
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        if model_extra_config.operator_opt_config.use_prefetch and lm_head is not None:
            torch_npu.npu_prefetch(lm_head.weight, hidden_states, model_extra_config.operator_opt_config.lm_head_prefetch * 1024 * 1024)

        return hidden_states


@support_torch_compile
class PanguProMoEV2ForCausalLM(nn.Module, SupportsPP):

    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # patch_fused_moe_ops()

        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = PanguProMoEModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      quant_config=self.quant_config,
                                      parallel_lmhead=False)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                logits_as_input=True)

        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self.return_hidden_states = True

        self.enable_torchair_graph_mode = (
                    vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())
        self.is_pd_seperate_d = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        self.is_hybrid_chunked_prefill_graph_mode = self.enable_torchair_graph_mode and not self.is_pd_seperate_d and \
            not vllm_config.additional_config.get("enable_hybrid_graph_mode", False) and vllm_config.scheduler_config.enable_chunked_prefill

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        selected_indices: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            self.lm_head
        )

        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            # when use ChunkedPrefill, selected_indices can cause GE graph recompilation, temporarily set to None
            if self.is_hybrid_chunked_prefill_graph_mode:
                logits = self.compute_lmhead(hidden_states, None)
            else:
                logits = self.compute_lmhead(hidden_states, selected_indices)

        return hidden_states, logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

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

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
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

        # expert_params_mapping = []

        params_dict = dict(self.named_parameters())  # from model
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if 'layers' in name:
                layer_idx = int(name.split('layers.')[-1].split('.')[0])
                if layer_idx >= self.model.end_layer:
                    continue

            if "rotary_emb.inv_freq" in name:
                continue

            if "module" in name:
                continue

            if name.endswith('kv_cache_offset'):
                continue

            if name.endswith("k_proj.kv_cache_scale"):
                remapped_kv_scale_name = name.replace("k_proj.kv_cache_scale", "attn.key_antiquant_scale")
                if remapped_kv_scale_name not in params_dict:
                    logger.warning_once(
                        "Found kv scale in the checkpoint "
                        f"(e.g. {name}), but not found the expected "
                        f"name in the model "
                        f"(e.g. {remapped_kv_scale_name}). "
                        "kv-scale is not loaded.")
                    continue
                else:
                    name = remapped_kv_scale_name
                    param = params_dict[name]
                    set_weight_attrs(param, {"is_2_dims": True})
                    loaded_weight = torch.tensor_split(loaded_weight, tp_size, dim=0)[tp_rank]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

            if name.endswith("v_proj.kv_cache_scale"):
                remapped_kv_scale_name = name.replace("v_proj.kv_cache_scale", "attn.value_antiquant_scale")
                if remapped_kv_scale_name not in params_dict:
                    logger.warning_once(
                        "Found kv scale in the checkpoint "
                        f"(e.g. {name}), but not found the expected "
                        f"name in the model "
                        f"(e.g. {remapped_kv_scale_name}). "
                        "kv-scale is not loaded.")
                    continue
                else:
                    name = remapped_kv_scale_name
                    param = params_dict[name]
                    set_weight_attrs(param, {"is_2_dims": True})
                    loaded_weight = torch.tensor_split(loaded_weight,
                                                       tp_size,
                                                       dim=0)[tp_rank]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

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
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
                    continue

                # Skip layers on other devices.
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
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
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
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict):
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint "
                                f"(e.g. {name}), but not found the expected "
                                f"name in the model "
                                f"(e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded.")
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    # Parameters that need 2-dims attribute
                    is_2_dims_suffixes = (
                        "kv_scale",
                        "key_antiquant_scale",
                        "value_antiquant_scale",
                        "param_sink_key",
                        "param_sink_value",
                    )
                    if name.endswith(is_2_dims_suffixes):
                        if not hasattr(param, "is_2_dims"):
                            set_weight_attrs(param, {"is_2_dims": True})

                    if name.endswith("param_sink_key") or name.endswith("param_sink_value"):
                        weight_loader = getattr(param, "weight_loader", sharded_weight_loader(-2))
                    else:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def should_use_eager_mode(self, *args, **kwargs):
        """Return if a layer should use eager mode. This function is
        to fit the attention backend of Omni infer.

        Returns:
            bool: True for eager mode, False for graph mode
        """
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]
        if self.is_hybrid_chunked_prefill_graph_mode and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            return False
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly