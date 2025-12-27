#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/deepseek_mtp.py
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, CompilationLevel, ModelConfig, VllmConfig
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_tp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.deepseek_mtp import DeepSeekMultiTokenPredictor
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.utils import supports_dynamo

from omni.adaptors.vllm.distributed import get_eh_proj_tp_group
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.layernorm import RMSNorm
from omni.layers.linear import ColumnParallelFlashCommLinear
from omni.layers.moe.fused_moe.layer import FusedMoE
from omni.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

from .pangu_moe_v2 import PanguProMoEDecoderLayer


class PanguProMoEShareHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "") -> None:

        nn.Module.__init__(self)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class PanguProMoEMultiTokenPredictorLayer(PanguProMoEDecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=prefix
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_tp_size = get_eh_proj_tp_group().world_size
        self.eh_tp_rank = get_eh_proj_tp_group().rank_in_group
        self.eh_proj = ColumnParallelFlashCommLinear(
            input_size=2 * config.hidden_size,
            output_size=config.hidden_size,
            bias=False,
            tp_size=self.eh_tp_size,
            tp_rank=self.eh_tp_rank,
            quant_config=None,
            prefix=f"{prefix}.eh_proj",
        )
        self.shared_head = PanguProMoEShareHead(
            config=config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "shared_head"))
        self.mtp_block = PanguProMoEDecoderLayer(
            config,
            vllm_config,
            cache_config,
            quant_config,
            prefix)

        self.enable_torchair_graph_mode = (
            vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())
        self.is_pd_seperate_d = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        self.is_hybrid_chunked_prefill_graph_mode = (
            self.enable_torchair_graph_mode and not self.is_pd_seperate_d and
            not vllm_config.additional_config.get("enable_hybrid_graph_mode", False) and
            vllm_config.scheduler_config.enable_chunked_prefill)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: torch.Tensor,
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        selected_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids, reduce=1)
        inputs_embeds = self.enorm(inputs_embeds)

        tp_size = get_tensor_model_parallel_world_size()
        rank_in_group = get_tensor_model_parallel_rank()
        if tp_size > 1:
            token_num = previous_hidden_states.shape[0]
            start_range = rank_in_group * (token_num // tp_size)
            end_range = (1 + rank_in_group) * (token_num // tp_size)
            previous_hidden_states = previous_hidden_states[start_range:end_range, :]

        previous_hidden_states = self.hnorm(previous_hidden_states)

        cat_hidden_states = torch.cat([inputs_embeds, previous_hidden_states], dim=-1)

        if self.eh_tp_size > 1:
            cat_hidden_states = get_eh_proj_tp_group().all_gather(cat_hidden_states, dim=0)
        hidden_states, _ = self.eh_proj.forward(cat_hidden_states)
        if self.eh_tp_size > 1:
            hidden_states = get_eh_proj_tp_group().all_to_all(hidden_states)

        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_caches,
            attn_metadata=attn_metadata,
            residual=None,
            cos=None,
            sin=None)

        if residual is not None:
            hidden_states = residual + hidden_states

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        if attn_metadata is None:
            logits = self.compute_lmhead(hidden_states[-1:, ...], None)
        else:
            # when use ChunkedPrefill, selected_indices can cause GE graph recompilation, temporarily set to None
            if self.is_hybrid_chunked_prefill_graph_mode:
                logits = self.compute_lmhead(hidden_states, None)
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
        logits = self.shared_head.head(hidden_states, embedding_bias)
        return logits

class PanguProMoeMultiTokenPredictor(DeepSeekMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_mtp_layers
        self.ignore_share_weight = True  # TODO get from config
        # to map the exact layer index from weights
        real_num_mtp = min(self.num_mtp_layers, vllm_config.speculative_config.num_speculative_tokens)
        self.layers = torch.nn.ModuleDict({
            str(i + self.mtp_start_layer_idx):
            PanguProMoEMultiTokenPredictorLayer(
                config,
                vllm_config,
                f"{prefix}.layers.{i + self.mtp_start_layer_idx}",
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
            )
            for i in range(real_num_mtp)
        })

        self.logits_processor = LogitsProcessor(config.vocab_size, logits_as_input=True)

    def set_share_weight(self, target_model):
        if self.ignore_share_weight:
            for _, layer in self.layers.items():
                layer.embed_tokens = target_model.model.embed_tokens
                layer.shared_head.head = target_model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: torch.Tensor,
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor,
        selected_indices: Optional[torch.Tensor],
        mtp_layer_idx: int = 0
    ) -> torch.Tensor:

        return self.layers[str(self.mtp_start_layer_idx + mtp_layer_idx)](
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            previous_hidden_states=previous_hidden_states,
            selected_indices=selected_indices
        )

@support_torch_compile
class PanguProMoEMTP(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.model = PanguProMoeMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"))
        self.n_predictor = self.config.num_mtp_layers

        self.enable_torchair_graph_mode = (
            vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())
        self.is_pd_seperate_d = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        self.is_hybrid_chunked_prefill_graph_mode = (
            self.enable_torchair_graph_mode and not self.is_pd_seperate_d and
            not vllm_config.additional_config.get("enable_hybrid_graph_mode", False) and
            vllm_config.scheduler_config.enable_chunked_prefill)

    def set_share_weight(self, target_model):
        self.model.set_share_weight(target_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        previous_hidden_states: Optional[torch.Tensor] = None,
        selected_indices: Optional[torch.Tensor] = None,
        mtp_layer_idx = 0,
    ) -> torch.Tensor:

        logits, hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            previous_hidden_states=previous_hidden_states,
            selected_indices=selected_indices,
            mtp_layer_idx=min(self.n_predictor - 1, mtp_layer_idx))

        return logits, hidden_states


    def load_weights(self, weights: Iterable[tuple[str,
                                                torch.Tensor]]) -> set[str]:
        logger = init_logger(__name__)
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

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

            spec_layer = None
            if 'layers' in name and hasattr(self.config, "num_mtp_layers") \
                and (self.config.num_mtp_layers > 0):
                layer_idx = int(name.split('layers.')[-1].split('.')[0])
                mtp_idx = layer_idx - self.config.num_hidden_layers
                if mtp_idx >= 0 and mtp_idx < self.config.num_mtp_layers:
                    spec_layer = layer_idx

            if spec_layer is None:
                continue

            if name.endswith("k_proj.kv_cache_scale"):
                remapped_kv_scale_name = name.replace(
                    "k_proj.kv_cache_scale", "attn.key_antiquant_scale")
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
                    loaded_weight = torch.tensor_split(loaded_weight, tp_size, dim=0)[tp_rank]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

            if name.endswith("v_proj.kv_cache_scale"):
                remapped_kv_scale_name = name.replace(
                    "v_proj.kv_cache_scale", "attn.value_antiquant_scale")
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
                    loaded_weight = torch.tensor_split(loaded_weight, tp_size, dim=0)[tp_rank]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

            name = self._rewrite_spec_layer_name(spec_layer, name)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
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

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
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

                    if name not in params_dict:
                        logger.warning_once(f"Missing parameter '{name}' in checkpoint. Skipping.")
                        continue

                    param = params_dict[name]
                    if name.endswith("param_sink_key") or name.endswith("param_sink_value"):
                        weight_loader = getattr(param, "weight_loader", sharded_weight_loader(-2))
                    else:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        """
        spec_layer_weight_names = [
            "embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head"
        ]
        spec_layer_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(f"model.layers.{spec_layer}.",
                                f"model.layers.{spec_layer}.mtp_block.")
        return name

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
            attn_metadata = attn_metadata[next(iter(attn_metadata))]
        if self.is_hybrid_chunked_prefill_graph_mode and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            return False
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly