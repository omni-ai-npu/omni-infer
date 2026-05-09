# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Iterable, List, Optional, Tuple, Set

import os
import torch
import torch.nn as nn

from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import QuantizationConfig, VllmConfig
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
 
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.deepseek_v2 import get_spec_layer_idx_from_weight_name
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.distributed import get_ep_group

from omni.adaptors.vllm.distributed import get_eh_proj_tp_group
from omni.layers.moe.fused_moe.fused_moe import set_num_speculative_tokens
from omni.models.config_loader.loader import model_extra_config
from .deepseek_v3 import (
    build_mixed_attn_metadata,
    generate_sp_inputs,
    is_mixed_batch_metadata,
    reverse_sp_outputs,
    _pad_1d_tensor,
    _pad_2d_tensor,
)

should_use_a2_layer = not model_extra_config.operator_opt_config.prefill_moe_all_to_all and not model_extra_config.operator_opt_config.enable_dsa
if os.getenv("ASCEND_PLATFORM", "A3")=="A2" and should_use_a2_layer:
    from .deepseek_v3_a2 import DeepseekDecoderLayer
else:
    from .deepseek_v3 import DeepseekDecoderLayer

from omni.layers.layernorm import RMSNorm #zxp: not use
from omni.layers.linear import ColumnParallelFlashCommLinear
from omni.layers.moe.fused_moe.layer import FusedMoE
from omni.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding
)

class SharedHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        ignore_share_weight: bool = True,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = None if ignore_share_weight else \
            ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

@support_torch_compile
class DeepseekMultiTokenPredictorLayer(DeepseekDecoderLayer):
    def __init__(self, *,
                 vllm_config,
                 prefix: str,
                 kv_ind: int,
                 is_ffn_die: Optional[bool] = False,
    ):
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.kv_ind = kv_ind

        super().__init__(self.config, prefix,
                         cache_config=self.cache_config,
                         quant_config=self.quant_config,
                         **({"is_ffn_die": True} if is_ffn_die else {})
                        )
        # adapt: add max init times to determine how many time the MTP layer will be initialized
        self.self_attn.max_init_count = vllm_config.speculative_config.num_speculative_tokens
        # adapt end

        self.ignore_share_weight = True # TODO get from config
        self.embed_tokens = None if self.ignore_share_weight else \
            VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                prefix=prefix,
            )
        self.shared_head = SharedHead(self.config, self.quant_config, self.ignore_share_weight)
        self.enorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.hnorm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.eh_tp_size = get_eh_proj_tp_group().world_size
        self.eh_tp_rank = get_eh_proj_tp_group().rank_in_group
        self.eh_proj = ColumnParallelFlashCommLinear(
            input_size=2 * self.config.hidden_size,
            output_size=self.config.hidden_size,
            bias=False,
            tp_size=self.eh_tp_size,
            tp_rank=self.eh_tp_rank,
            quant_config=None,
            prefix=f"{prefix}.eh_proj",
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size, logits_as_input=True)
        self.layer_idx = int(prefix.split('.')[-1])
        self.prefix = prefix
        self.postfix = ".self_attn.attn"

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            previous_hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        attn_metadata_first = self.get_layer_attn_metadata(attn_metadata)
        if is_mixed_batch_metadata(attn_metadata_first):
            return self.forward_mixed_batch(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                previous_hidden_states=previous_hidden_states,
                selected_indices=selected_indices,
            )

        tok_embeds = self.enorm(self.get_input_embeddings(input_ids))
        if len(tok_embeds.shape) > 2:
            tok_embeds = tok_embeds.view(-1, self.config.hidden_size)

        tp_size = get_tensor_model_parallel_world_size()  # cloud: get_tp_group().world_size
        rank_in_group = get_tensor_model_parallel_rank()

        is_prefill = attn_metadata is None or (isinstance(attn_metadata, dict) and self.get_layer_attn_metadata(attn_metadata).prefill is not None)
        if is_prefill and model_extra_config.parall_config.attn_sp_size > 1:
            # split input for sp attention
            if not model_extra_config.operator_opt_config.use_mlaprolog:
               tok_embeds = tensor_model_parallel_all_gather(tok_embeds, dim=0)
            tok_embeds = generate_sp_inputs(tok_embeds, self.get_layer_attn_metadata(attn_metadata))
            previous_hidden_states = generate_sp_inputs(previous_hidden_states, self.get_layer_attn_metadata(attn_metadata))


        if tp_size > 1 and model_extra_config.parall_config.attn_sp_size == 1:
            token_num = previous_hidden_states.shape[0]
            start_range = rank_in_group * (token_num // tp_size)
            end_range = (1 + rank_in_group) * (token_num // tp_size)
            previous_hidden_states = previous_hidden_states[start_range: end_range, :]

        previous = self.hnorm(previous_hidden_states)
        cat_hidden_states = torch.cat([tok_embeds, previous], dim=-1)
        if self.eh_tp_size > 1:
            cat_hidden_states = get_eh_proj_tp_group().all_gather(cat_hidden_states, dim=0)

        hidden_states, _ = self.eh_proj.forward(cat_hidden_states)

        if self.eh_tp_size > 1:
            hidden_states = get_eh_proj_tp_group().all_to_all(hidden_states)

        encoded_states, residual = DeepseekDecoderLayer.forward(
            self,
            positions=positions,
            kv_cache=kv_caches[self.kv_ind] if kv_caches is not None else None,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=None,
        )
        if residual is not None:
            hidden_states, _ = self.shared_head.norm(encoded_states, residual)
        else:
            hidden_states = self.shared_head.norm(encoded_states)
        
        if is_prefill:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        if model_extra_config.parall_config.attn_sp_size > 1 and is_prefill:
            # reverse sp split
            if attn_metadata is not None:
                prefill_meta = self.get_layer_attn_metadata(attn_metadata).prefill
                hidden_states = reverse_sp_outputs(hidden_states, prefill_meta)

        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            logits = self.compute_lmhead(hidden_states, selected_indices)

        return logits, hidden_states

    def forward_mixed_batch(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            previous_hidden_states: torch.Tensor,
            selected_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_metadata_first = self.get_layer_attn_metadata(attn_metadata)
        decode_tokens = attn_metadata_first.decode.num_decode_tokens
        decode_padded_tokens = attn_metadata_first.decode.num_padded_tokens
        actual_tokens = attn_metadata_first.num_actual_tokens

        decode_input_ids = _pad_1d_tensor(input_ids[:decode_tokens], decode_padded_tokens)
        prefill_input_ids = input_ids[decode_tokens:actual_tokens]
        decode_positions = _pad_1d_tensor(positions[:decode_tokens], decode_padded_tokens)
        prefill_positions = positions[decode_tokens:actual_tokens]

        decode_previous_hidden_states = _pad_2d_tensor(previous_hidden_states[:decode_tokens], decode_padded_tokens)
        prefill_previous_hidden_states = previous_hidden_states[decode_tokens:]

        decode_tok_embeds = self.enorm(self.get_input_embeddings(decode_input_ids))
        prefill_tok_embeds = self.enorm(self.get_input_embeddings(prefill_input_ids))
        if len(decode_tok_embeds.shape) > 2:
            decode_tok_embeds = decode_tok_embeds.view(-1, self.config.hidden_size)
        if len(prefill_tok_embeds.shape) > 2:
            prefill_tok_embeds = prefill_tok_embeds.view(-1, self.config.hidden_size)

        if model_extra_config.parall_config.attn_sp_size > 1:
            if not model_extra_config.operator_opt_config.use_mlaprolog:
                prefill_tok_embeds = tensor_model_parallel_all_gather(prefill_tok_embeds, dim=0)
            prefill_tok_embeds = generate_sp_inputs(prefill_tok_embeds, attn_metadata_first)
            prefill_previous_hidden_states = generate_sp_inputs(prefill_previous_hidden_states, attn_metadata_first)

        tp_size = get_tensor_model_parallel_world_size()
        rank_in_group = get_tensor_model_parallel_rank()
        if tp_size > 1 and model_extra_config.parall_config.attn_sp_size == 1:
            decode_token_num = decode_previous_hidden_states.shape[0]
            decode_start = rank_in_group * (decode_token_num // tp_size)
            decode_end = (1 + rank_in_group) * (decode_token_num // tp_size)
            prefill_token_num = prefill_previous_hidden_states.shape[0]
            prefill_start = rank_in_group * (prefill_token_num // tp_size)
            prefill_end = (1 + rank_in_group) * (prefill_token_num // tp_size)
            decode_previous_hidden_states = decode_previous_hidden_states[decode_start:decode_end, :]
            prefill_previous_hidden_states = prefill_previous_hidden_states[prefill_start:prefill_end, :]

        previous_hidden_states = torch.cat(
            [decode_previous_hidden_states, prefill_previous_hidden_states],
            dim=0)
        tok_embeds = torch.cat([decode_tok_embeds, prefill_tok_embeds], dim=0)
        previous = self.hnorm(previous_hidden_states)
        cat_hidden_states = torch.cat([tok_embeds, previous], dim=-1)
        if self.eh_tp_size > 1:
            cat_hidden_states = get_eh_proj_tp_group().all_gather(cat_hidden_states, dim=0)

        hidden_states, _ = self.eh_proj.forward(cat_hidden_states)

        if self.eh_tp_size > 1:
            hidden_states = get_eh_proj_tp_group().all_to_all(hidden_states)

        decode_attn_metadata = build_mixed_attn_metadata(attn_metadata, is_decode=True)
        prefill_attn_metadata = build_mixed_attn_metadata(attn_metadata, is_decode=False)
        decode_hidden_states = hidden_states[:decode_padded_tokens]
        prefill_hidden_states = hidden_states[decode_padded_tokens:]

        decode_hidden_states, decode_residual = DeepseekDecoderLayer.forward(
            self,
            positions=decode_positions,
            kv_cache=kv_caches[self.kv_ind] if kv_caches is not None else None,
            hidden_states=decode_hidden_states,
            attn_metadata=decode_attn_metadata,
            residual=None,
        )
        prefill_hidden_states, prefill_residual = DeepseekDecoderLayer.forward(
            self,
            positions=prefill_positions,
            kv_cache=kv_caches[self.kv_ind] if kv_caches is not None else None,
            hidden_states=prefill_hidden_states,
            attn_metadata=prefill_attn_metadata,
            residual=None,
        )

        if decode_residual is not None:
            decode_hidden_states, _ = self.shared_head.norm(decode_hidden_states, decode_residual)
        else:
            decode_hidden_states = self.shared_head.norm(decode_hidden_states)
        if prefill_residual is not None:
            prefill_hidden_states, _ = self.shared_head.norm(prefill_hidden_states, prefill_residual)
        else:
            prefill_hidden_states = self.shared_head.norm(prefill_hidden_states)

        decode_hidden_states = decode_hidden_states[:decode_tokens]
        prefill_hidden_states = tensor_model_parallel_all_gather(prefill_hidden_states, dim=0)

        if model_extra_config.parall_config.attn_sp_size > 1:
            prefill_meta = attn_metadata_first.prefill
            prefill_hidden_states = reverse_sp_outputs(prefill_hidden_states, prefill_meta)

        hidden_states = torch.cat([decode_hidden_states, prefill_hidden_states], dim=0)
        logits = self.compute_lmhead(hidden_states, selected_indices)
        return logits, hidden_states

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        reduce = 1
        if model_extra_config.operator_opt_config.use_mlaprolog and model_extra_config.parall_config.attn_sp_size > 1:
            reduce = 0
        return self.embed_tokens(input_ids, reduce=reduce)

    def get_layer_attn_metadata(self, attn_metadata):
        if attn_metadata is None:
            return None
        if isinstance(attn_metadata, dict):
            key_idx = self.prefix + self.postfix
            return attn_metadata[key_idx]

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
        logits = self.shared_head.head(hidden_states, embedding_bias)
        return logits

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(self.shared_head["head"], hidden_states, sampling_metadata)
        return logits

    def should_use_eager_mode(self, *args, **kwargs):
        if len(kwargs) == 0:
           return True

        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]

        if attn_metadata.prefill:
            return True

        return False

class DeepseekMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.mtp_start_layer_idx = self.config.num_hidden_layers
        self.num_mtp_layers = self.config.num_nextn_predict_layers
        self.ignore_share_weight = True # TODO get from config
        real_num_mtp = min(self.num_mtp_layers, vllm_config.speculative_config.num_speculative_tokens)
        kwargs = {}
        if model_extra_config.task_config.enable_attn_ffn_disaggregation:
            if os.getenv("ASCEND_PLATFORM", "A3") == "A2":
                raise NotImplementedError("Attention FFN disaggregation on A2 is not supported")
            else:
                ffn_dies = get_ep_group().world_size - model_extra_config.parall_config.attn_dies
                if get_ep_group().rank_in_group < ffn_dies:
                    kwargs["is_ffn_die"] = True
            if vllm_config.speculative_config:
                set_num_speculative_tokens(real_num_mtp)
        self.layers = nn.ModuleDict({
            str(i + self.mtp_start_layer_idx):
            DeepseekMultiTokenPredictorLayer(
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{i + self.mtp_start_layer_idx}",
                kv_ind=i - real_num_mtp,
                **kwargs
            )
            for i in range(real_num_mtp)
        })
        self.logits_processor = LogitsProcessor(self.config.vocab_size, logits_as_input=True)
        self.greedy_sampler = Sampler()
    
    def set_share_weight(self, target_model):
        if self.ignore_share_weight:
            for _, layer in self.layers.items():
                layer.embed_tokens = target_model.model.embed_tokens
                layer.shared_head.head = target_model.lm_head

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
        return self.layers[str(self.mtp_start_layer_idx + mtp_layer_idx)](
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            previous_hidden_states=previous_hidden_states,
            selected_indices=selected_indices,
        )


class DeepseekV3MTP(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepseekMultiTokenPredictor(vllm_config=vllm_config, prefix=f"model")
        self.n_predictor = self.config.num_nextn_predict_layers
    
    def set_share_weight(self, target_model):
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # 字段说明: (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # 字段说明: (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.model.ignore_share_weight and any(
                    substring in name for substring in ["embed_tokens.weight", "shared_head.head"]):
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
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
