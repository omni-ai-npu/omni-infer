# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
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
from vllm.distributed import get_dp_group
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.distributed.parallel_state_ext import get_local_world_group
from omni_npu.v1.layers.logits_processor import NPULogitsProcessor
from omni_npu.v1.layers.vocab_parallel_embedding import NPUParallelLMHead, NPUVocabParallelEmbedding

from .pangu_v2_moe import PanguV2DecoderLayer, _maybe_gather_and_unpadding, _maybe_padding_and_slice

class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        parall_cfg = model_extra_config.parall_config
        if parall_cfg.ena_local_lmhead_parallel:
            local_lmhead = get_local_world_group().world_size > 1
            dp_lmhead = False
        elif parall_cfg.ena_dp_lmhead_parallel:
            local_lmhead = False
            dp_lmhead = get_dp_group().world_size > 1
        else:
            local_lmhead = False
            dp_lmhead = False
        self.head = NPUParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
            local_lmhead_parallel=local_lmhead,
            dp_parallel=dp_lmhead,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


@support_torch_compile
class PanguV2MultiTokenPredictorLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "eh_proj"),
        )
        self.shared_head = SharedHead(
            config=config,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "shared_head"),
        )
        self.mtp_block = PanguV2DecoderLayer(config, prefix, vllm_config)
        self.mtp_block._tail_refs = (
            None,            # tail_mhc_pre — unused when tail_use_mhc is False
            nn.Identity(),   # tail_layernorm — no-op
            True,            # is_model_tail
        )

        self.need_tp_padding = not model_extra_config.operator_opt_config.moe_comm_strategy == "allreduce"

    def set_side_stream(self, side_stream: torch.npu.Stream, fetch_stream: torch.npu.Stream = None) -> None:
        """Set the shared side/fetch streams for MTP block."""
        self.mtp_block.set_side_stream(side_stream, fetch_stream)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states, _ = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        
        ### Add padding for sequence parallel (TP > 1 with non-naive backend)
        if self.need_tp_padding:
            hidden_states, original_num_tokens = _maybe_padding_and_slice(hidden_states)

        cos, sin = self.mtp_block.self_attn.rotary_emb.get_cos_sin(positions)
        
        hidden_states, residual, _, _, sk_event = self.mtp_block.mhc_head(hidden_states)
        hidden_states, _, _, _, _ = self.mtp_block(
            hidden_states, residual, None, None, cos, sin, sk_event,
        )

        # Unpad hidden_states: after all layers, gather and remove any padding that was added
        # for sequence parallelism to restore the original number of tokens
        if self.need_tp_padding:
            hidden_states = _maybe_gather_and_unpadding(hidden_states, original_num_tokens)
        return hidden_states


class PanguV2MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): PanguV2MultiTokenPredictorLayer(
                    vllm_config=vllm_config, 
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
            }
        )
        self.wrapped_layers = None
        self.embed_tokens = NPUVocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = NPULogitsProcessor(config.vocab_size)

        use_multi_stream = model_extra_config.operator_opt_config.enable_multi_stream
        if use_multi_stream:
            self.side_stream = torch.npu.Stream()
            self.fetch_stream = torch.npu.Stream()
            for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers):
                self.layers[str(idx)].set_side_stream(self.side_stream, self.fetch_stream)

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
            inputs_embeds = self.embed_tokens(input_ids)
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
        hidden_states = mtp_layer.shared_head(hidden_states)
        logits = self.logits_processor(
            mtp_layer.shared_head.head, hidden_states
        )
        return logits


class PanguV2MTP(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = PanguV2MultiTokenPredictor(
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            routed_experts_prefix="",
        )

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
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

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            # MLA multi-stream split path: q_b_proj's weight_loader (installed
            # by NPUPanguSparseAttention._install_q_b_split_loaders) also
            # populates the q_b_nope_proj / q_b_pe_proj children's matching
            # params in-place. vLLM's "unloaded weights" check looks at
            # state_dict keys, so mark those synthetic names as loaded too
            # whenever the parent q_b_proj key was just consumed.
            if ".q_b_proj." in name:
                for synthetic in (
                    name.replace(".q_b_proj.", ".q_b_nope_proj."),
                    name.replace(".q_b_proj.", ".q_b_pe_proj."),
                ):
                    if synthetic in params_dict:
                        loaded_params.add(synthetic)

            loaded_params.add(name)
        return loaded_params

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
    
    def set_shared_weight(self, target_model: nn.Module) -> None:
        if hasattr(target_model, "embed_tokens"):
            del self.model.embed_tokens
            self.model.embed_tokens = target_model.embed_tokens
        if hasattr(target_model, "lm_head"):
            for layer in self.model.layers.values():
                del layer.shared_head.head
                layer.shared_head.head = target_model.lm_head
