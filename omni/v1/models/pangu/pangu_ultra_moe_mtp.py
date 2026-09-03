# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is based on vLLM implementation:
# Copyright 2023 The vLLM team.
# https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/model_executor/models/openpangu_mtp.py
#
# The upstream vLLM implementation retains attribution to vllm-ascend and is
# adapted from:
# https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_mtp.py

from collections.abc import Iterable
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from .pangu_ultra_moe import OpenPanguDecoderLayer
from omni_npu.attention.backends.utils import SPManager
from omni_npu.layers.mhc.mhc_rl import NPUmHCRL
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.layers.attention.weight_utils import (
    mark_split_q_up_params_loaded,
)


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


@support_torch_compile
class OpenPanguMultiTokenPredictorLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(
            config=config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "shared_head"),
        )

        self.mtp_block = OpenPanguDecoderLayer(config, prefix, vllm_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None

        if model_extra_config.parall_config.ena_seq_parallel:
            sp_manager = SPManager.init_sp(previous_hidden_states.size(0), get_tp_group())
            previous_hidden_states = sp_manager.slice_tokens(previous_hidden_states)

        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        cos, sin = self.mtp_block.self_attn.rotary_emb.get_cos_sin(positions)
        topk_indices_buffer = None
        index_topk = getattr(self.config, "index_topk", 0) or 0
        if index_topk > 0:
            topk_indices_buffer = torch.zeros(
                (hidden_states.shape[0], 1, index_topk),
                dtype=torch.int32,
                device=hidden_states.device,
            )
        hidden_states, residual, _ = self.mtp_block(
            hidden_states,
            cos,
            sin,
            residual=None,
            topk_indices_buffer=topk_indices_buffer,
        )
        hidden_states = residual + hidden_states

        if model_extra_config.parall_config.ena_seq_parallel:
            hidden_states = sp_manager.ag_tokens(hidden_states)
        return hidden_states


class OpenPanguMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        mtp_layer_indices = range(
            self.mtp_start_layer_idx,
            self.mtp_start_layer_idx + self.num_mtp_layers,
        )
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): OpenPanguMultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in mtp_layer_indices
            }
        )
        self.wrapped_layers = None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(
                input_ids, 
                enable_scatter=model_extra_config.parall_config.ena_seq_parallel
            )
        else:
            # inputs_embeds bypasses embedding scatter; split tokens manually in SP mode.
            if model_extra_config.parall_config.ena_seq_parallel:
                sp_manager = SPManager.init_sp(inputs_embeds.size(0), get_tp_group())
                inputs_embeds = sp_manager.slice_tokens(inputs_embeds)

        current_step_idx = spec_step_idx % self.num_mtp_layers

        if self.wrapped_layers is not None:
            layer = self.wrapped_layers[str(self.mtp_start_layer_idx + current_step_idx)]
        else:
            layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]

        return layer(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        dtype = mtp_layer.shared_head.head.weight.dtype
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states).to(dtype)
        )
        return logits


class OpenPanguMTP(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = OpenPanguMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_spec_layer(self, name):
        if (
            "layers" in name
            and hasattr(self.config, "num_nextn_predict_layers")
            and self.config.num_nextn_predict_layers > 0
        ):
            layer_idx = int(name.split("layers.")[-1].split(".")[0])
            mtp_idx = layer_idx - self.config.num_hidden_layers
            if mtp_idx >= 0 and mtp_idx < self.config.num_nextn_predict_layers:
                return layer_idx
        return None

    def set_shared_weight(self, target_model: nn.Module) -> None:
        if hasattr(target_model, "embed_tokens"):
            del self.model.embed_tokens
            self.model.embed_tokens = target_model.embed_tokens
        if hasattr(target_model, "lm_head"):
            for layer in self.model.layers.values():
                del layer.shared_head.head
                layer.shared_head.head = target_model.lm_head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        record_conv_name = []
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = self.get_spec_layer(name)
            if spec_layer is None:
                continue

            name = self._rewrite_spec_layer_name(spec_layer, name)
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
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped

                # Skip loading extra bias for GPTQ models.
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
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if (
                        spec_layer != self.model.mtp_start_layer_idx
                        and ".layers" not in name
                    ):
                        continue

                    if name.endswith("e_score_correction_bias"):
                        name = name.replace(
                            "e_score_correction_bias", "gate.e_score_correction_bias"
                        )
                    if "_conv" in name:
                        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                            name = insert_conv_before(name)
                        else:
                            name = name.replace("_conv", "_conv.merge_conv")
                        if model_extra_config.operator_opt_config.merge_q_kv_conv and "qa_conv" in name:
                            merge_conv_name = name.replace("qa_conv", "merge_conv")
                            if merge_conv_name not in record_conv_name:
                                record_conv_name.append(merge_conv_name)
                                loaded_params.add(merge_conv_name)

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        mark_split_q_up_params_loaded(self, loaded_params)
        self.post_weight_load()
        
        return loaded_params

    def post_weight_load(self) -> None:
        for _, module in self.named_modules():
            if module is self:
                continue
            if hasattr(module, "post_weight_load"):
                module.post_weight_load()

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name


def insert_conv_before(name: str) -> str:
    parts = name.split('.')
    for i in range(len(parts) - 1, -1, -1):
        if '_conv' in parts[i]:
            parts.insert(i, 'conv')
            break
    return '.'.join(parts)
