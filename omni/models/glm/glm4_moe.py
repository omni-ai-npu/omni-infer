import typing
from typing import Iterable, List, Optional, Set, Tuple, Union
from collections.abc import Callable,Iterable 
from itertools import islice
from typing import Any
import time
import torch
import torch_npu
from torch import nn
import torch.distributed as dist
from vllm.transformers_utils.configs.glm4_moe import Glm4MoeConfig
from vllm.platforms import current_platform
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.attention import AttentionMetadata, Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_rank,
)
from vllm.logger import init_logger
from omni.layers.activation import SiluAndMul
from omni.layers.linear import (
    AscendMergedColumnParallelLinear,
    AscendRowParallelLinear,
)
from omni.layers.moe.fused_moe.layer import FusedMoE
from omni.layers.moe.deepseek_moe import DeepseekMoE
from omni.layers.layernorm import RMSNorm
from omni.adaptors.vllm.distributed.parallel_state import (
    get_mlp_tp_group,
    GroupCoordinator
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from omni.layers.rotary_embedding import get_rope
from omni.layers.vocab_parallel_embedding import (
    ParallelLMHead, 
    VocabParallelEmbedding
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from omni.layers.moe.deepseek_moe import ParallelDeepseekMLP

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix
)

from omni.layers.linear import (
    QKVParallelFlashCommLinear,
    RowParallelFlashCommLinear
)
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.models.config_loader.loader import model_extra_config
logger = init_logger(__name__)

SEQ_SPLIT_LENGTH = 4096

class RMSNormGLM(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)

        self.bias = None
        if model_extra_config.operator_opt_config.load_rms_bias:
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    def forward(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            quant_symbol: bool = False,
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if self.bias is not None:
                x.add_(self.bias)
            if quant_symbol:
                x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
                x = {"x_int8": x_int8, "pertoken_scale": pertoken_scale}
            return x, residual

        return torch_npu.npu_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
        )[0]


class ParallelGlmMoeMLP(ParallelDeepseekMLP):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
            comm_group: Optional[GroupCoordinator] = None
    ) -> None:
        super().__init__(hidden_size, intermediate_size, hidden_act, quant_config, reduce_results, prefix, comm_group)
        self.prefix = prefix
        if comm_group is None:
            comm_group = get_mlp_tp_group()
        
        self.quant_symbol = False if quant_config is None or (quant_config.ignore is not None and f"{prefix}.down_proj" in quant_config.ignore) else True
        self.gate_up_proj = AscendMergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            tp_size=comm_group.world_size,
            tp_rank=comm_group.rank_in_group,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            throw_dequant=self.quant_symbol)
        
        self.down_proj = AscendRowParallelLinear(intermediate_size,
                                                hidden_size,
                                                bias=False,
                                                tp_size=comm_group.world_size,
                                                tp_rank=comm_group.rank_in_group,
                                                quant_config=quant_config,
                                                reduce_results=False,
                                                prefix=f"{prefix}.down_proj")
        
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()

class Glm4MoeAttention(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[tuple] = None,
        max_position_embeddings: int = 131072,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        use_qk_norm: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = use_qk_norm

        self.qkv_proj = QKVParallelFlashCommLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelFlashCommLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        if self.use_qk_norm:
            self.q_norm = RMSNormGLM(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNormGLM(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states, x_transform='AG')
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(
                q.shape
            )
            k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(
                k.shape
            )

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output, reduce_type='RS')
        return output
    
class Glm4MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm4MoeConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_name = f"{prefix}.self_attn.attn"
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = Glm4MoeAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_qk_norm=config.use_qk_norm,
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = DeepseekMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            self.mlp = ParallelGlmMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False

        self.input_layernorm = RMSNormGLM(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormGLM(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            layer_id: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        is_prefill = attn_metadata is None or not attn_metadata.is_pd_seperate_d
        if is_prefill:
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
            hidden_states_list = hidden_states.split(SEQ_SPLIT_LENGTH)
            residual_list = residual.split(SEQ_SPLIT_LENGTH)
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
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id)
                if isinstance(hidden_states, (tuple, list)):
                    assert len(hidden_states) == 2
                    hidden_states = hidden_states[0] + hidden_states[1]
            else:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id)

        return hidden_states, residual
    
class Glm4MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.first_k_dense_replace = config.first_k_dense_replace
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm4MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNormGLM(config.hidden_size, eps=config.rms_norm_eps)
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
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_caches: List[torch.Tensor] = None,
        attn_metadata: Union[AttentionMetadata, dict] = None,
    ) -> torch.Tensor | IntermediateTensors:
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
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        return hidden_states

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)
        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
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
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)
                    name = name_mapped
                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name.endswith(".deq_scale") and name not in params_dict:
                        continue
                    if name.endswith(".input_offset") and name not in params_dict:
                        continue
                    if name.endswith(".input_scale") and name not in params_dict:
                        continue
                    if name.endswith(".quant_bias") and name not in params_dict:
                        continue 
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params

@support_torch_compile
class Glm4MoeForCausalLM(nn.Module):
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

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Glm4MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size, logits_as_input=True)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.expert_weights = []

        # Set MoE hyperparameters
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_expert_groups = config.n_group

        self.return_hidden_states = True

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        selected_indices: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_caches: List[torch.Tensor] = None,
        attn_metadata: Union[AttentionMetadata, dict] = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, kv_caches, attn_metadata
        )
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
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

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
    
def get_spec_layer_idx_from_weight_name(
    config: Glm4MoeConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if f"layers.{layer_idx + i}." in weight_name:
                return layer_idx + i
    return None