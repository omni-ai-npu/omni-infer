# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next MTP model."""
from collections.abc import Iterable
from typing import Iterable, List, Optional, Tuple, Set

import torch
from torch import nn
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import QuantizationConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.distributed.communication_op import tensor_model_parallel_all_gather

from transformers import PretrainedConfig
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration

from omni.layers.layernorm import RMSNorm
from omni.models.qwen.qwen3_next import Qwen3NextDecoderLayer, Qwen3NextModel, Qwen3NextForCausalLM, Qwen3NextRMSNorm 
from omni.models.qwen.fused_moe import FusedMoE

from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.linear import (
    RowParallelFlashCommLinear,
    QKVParallelFlashCommLinear,
    ColumnParallelFlashCommLinear,
    MergedColumnParallelFlashCommLinear
)
logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]

class Qwen3NextMultiTokenPredictor(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        config: PretrainedConfig = model_config.hf_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.ignore_share_weight = True
        self.embed_tokens = (
            None
            if self.ignore_share_weight
            else VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=prefix,
            )
        )

        self.fc = ColumnParallelLinear(self.config.hidden_size * 2,
                                       self.config.hidden_size,
                                       gather_output=True,
                                       bias=False,
                                       return_bias=False)

        self.layers = torch.nn.ModuleList(
            Qwen3NextDecoderLayer(
                config,
                layer_type="full_attention",
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f'{prefix}.layers.{self.mtp_start_layer_idx + idx}',
            ) for idx in range(self.num_mtp_layers))

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.norm = Qwen3NextRMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(config.hidden_size,
                                                   eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)

        self.head = (
            None
            if self.ignore_share_weight
            else ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            previous_hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            mtp_layer_idx = 0,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        hidden_states = previous_hidden_states
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None

        current_step_idx = mtp_layer_idx
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            logits = self.compute_lmhead(hidden_states, selected_indices)
        return logits, hidden_states

    def compute_lmhead(
            self,
            hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if get_dp_group().world_size <= 1 and selected_indices is not None:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if hidden_states.shape[0] != selected_indices.shape[0]:
                hidden_states = hidden_states.index_select(0, selected_indices)
        # Get the logits for the next tokens.
        logits = self.head(hidden_states, embedding_bias)
        return logits

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(
            self.head, hidden_states, sampling_metadata
        )
        return logits

    def set_share_weight(self, target_model):
        if self.ignore_share_weight:
            self.embed_tokens = target_model.model.embed_tokens
            self.head = target_model.lm_head

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
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
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
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
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


@support_torch_compile
class Qwen3NextMTP(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        assert not cache_config.enable_prefix_caching, \
            "Qwen3NextMTP currently does not support prefix caching"

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3NextMultiTokenPredictor(vllm_config=vllm_config,
                                                  prefix=maybe_prefix(
                                                      prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = None
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.n_predictor = getattr(self.config, "num_nextn_predict_layers", 1)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def set_share_weight(self, target_model):
        self.lm_head = target_model.lm_head
        self.model.set_share_weight(target_model)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            previous_hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            mtp_layer_idx = 0,
            **kwargs,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            previous_hidden_states=previous_hidden_states,
            selected_indices=selected_indices,
            mtp_layer_idx=min(self.n_predictor - 1, mtp_layer_idx),
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                else:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(remap_weight_names(weights))

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            # attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]
            # written by cursor
            attn_metadata = attn_metadata[self.model.layers[0].layer_name]
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly

    def mark_static_for_graph(self, input_ids, positions, attn_metadata, kv_caches, **kwargs):
        pass


