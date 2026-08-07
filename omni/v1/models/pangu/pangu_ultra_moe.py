# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Callable, Iterable
from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ParallelConfig, VllmConfig, CacheConfig
from vllm.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from omni_npu.layers.fused_moe.layer import NPUSharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.layers.mhc.mhc_rl import NPUmHCRL
from omni_npu.plugin_decorators import post_model_forward_decorator
from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention
from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention
from omni_npu.v1.layers.fused_mlp.layer import FusedMLP
from omni_npu.v1.layers.vocab_parallel_embedding import (
    NPUParallelLMHead,
    NPUVocabParallelEmbedding,
)


def check_ffn_act_fn(act_fn: str) -> None:
    """Validate FFN activation function.

    Note: current NPU fused kernels only support SiLU in this implementation.
    """
    if act_fn != "silu":
        raise ValueError(
            f"Unsupported activation: {act_fn}. Only silu is supported for now."
        )


def _normalize_rope_parameters(
    config: PretrainedConfig, *, max_position_embeddings: int
) -> None:
    """Normalize rope parameters in-place for compatibility.

    Some upstream configs may use `rope_type="default"`. For DeepSeek-style MLA,
    vLLM expects a concrete rope type; we map it to `deepseek_yarn` and fill
    commonly-required defaults.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if not isinstance(rope_params, dict):
        return

    if rope_params.get("rope_type") != "default":
        return

    # Mutate in-place on purpose: vLLM/hf_config is treated as a shared config.
    rope_params["rope_type"] = "deepseek_yarn"
    rope_params.setdefault("factor", 1.0)
    rope_params.setdefault("original_max_position_embeddings", max_position_embeddings)
    rope_params.setdefault("apply_yarn_scaling", False)


def _has_mla_config(config: PretrainedConfig) -> bool:
    """Whether the config contains required MLA fields used by this model."""
    return (
        hasattr(config, "qk_nope_head_dim")
        and hasattr(config, "qk_rope_head_dim")
        and hasattr(config, "v_head_dim")
        and hasattr(config, "kv_lora_rank")
    )


class OpenPanguMLP(FusedMLP):
    """Dense FFN block used by Pangu layers (and shared experts)."""
    pass


class OpenPanguMoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group

        self.routed_scaling_factor = config.routed_scaling_factor
        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts
        self.model_type: str = config.model_type

        self.is_sequence_parallel = False
        check_ffn_act_fn(config.hidden_act)
        if model_extra_config.operator_opt_config.router_gating_in_fp32:
            self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
            params_dtype=torch.float32
        )
        else:
            self.gate = ReplicatedLinear(
                config.hidden_size,
                config.n_routed_experts,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.gate",
            )
        if (
            hasattr(config, "router_enable_expert_bias")
            and config.router_enable_expert_bias
        ):
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = OpenPanguMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                disable_tp=True,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        if config.model_type in ("openpangu_v2", "openpangu_mtp","openpangu_v2_vl_moe","openpangu_v2_omni_moe"):
            self.experts = NPUSharedFusedMoE(
                shared_experts=self.shared_experts,
                gate=self.gate,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                reduce_results=False,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=1,
                topk_group=1,
                prefix=f"{prefix}.experts",
                scoring_func="sigmoid",
                # we do scaling outside, set factor to 1.0 to avoid double mul
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
            )
        else:
            self.experts = NPUSharedFusedMoE(
                shared_experts=self.shared_experts,
                gate=self.gate,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                reduce_results=False,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=1,
                topk_group=1,
                prefix=f"{prefix}.experts",
                scoring_func="sigmoid",
                # we do scaling outside, set factor to 1.0 to avoid double mul
                routed_scaling_factor=1.0,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        fused_moe_out = self.experts(
            hidden_states=hidden_states, router_logits=None
        )

        shared_output, final_hidden_states = fused_moe_out

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class OpenPanguDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig | None,
        prefix: str,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.use_mla = _has_mla_config(config)
        if not self.use_mla:
            raise ValueError(
                "PanguUltraMoE expects MLA-related config fields "
                "(qk_nope_head_dim/qk_rope_head_dim/v_head_dim/kv_lora_rank)."
            )

        _normalize_rope_parameters(config, max_position_embeddings=max_position_embeddings)

        is_dsa = hasattr(config, "index_topk") and (not hasattr(config, "dsa_layers") or layer_idx in config.dsa_layers)
        if is_dsa:
            attn_cls = NPUDeepseekSparseAttention
        else:
            attn_cls = NPUDeepseekMLAAttention

        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        if (
            getattr(config, "n_routed_experts", None) is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            self.mlp = OpenPanguMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = OpenPanguMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.num_hidden_layers = config.num_hidden_layers
        self.first_k_dense_replace = getattr(
            config, "first_k_dense_replace", self.num_hidden_layers
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.tp_group = get_tp_group().device_group
        self.sandwich_norm = getattr(config, "sandwich_norm", False)
        if self.sandwich_norm:
            self.pre_mlp_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_mlp_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        block_post_layernorm_hidden_size = config.hidden_size

        self.is_mtp_layer = layer_idx >= getattr(config, "num_hidden_layers", float('inf'))
        self.use_mhc = getattr(config, "use_mhc", False) and not self.is_mtp_layer
        if self.use_mhc:
            self.attn_mhc_module = NPUmHCRL(
                config=config,
                pre_only=False,
                prefix=f"{prefix}.attn_mhc_module",
            )
            self.mlp_mhc_module = NPUmHCRL(
                config=config,
                pre_only=False,
                prefix=f"{prefix}.mlp_mhc_module",
            )
            block_post_layernorm_hidden_size *= getattr(config, "mhc_num_stream", 4)
        self.has_block_post_layernorm = layer_idx in getattr(config, "block_post_layernorm_idx", [])
        if self.has_block_post_layernorm:
            self.block_post_layernorm = RMSNorm(block_post_layernorm_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_mhc:
            return self.forward_mhc(hidden_states, cos, sin, residual)
        else:
            return self.forward_normal(hidden_states, cos, sin, residual)
    
    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, cos, sin)

        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, residual = self.pre_mlp_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        return hidden_states, residual

    def forward_mhc(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, h_post, h_res = self.attn_mhc_module.mhc_pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        h_res = self.attn_mhc_module.mhc_sinkhorn(h_res)
        hidden_states = self.attn_mhc_module.mhc_post(hidden_states, h_post, residual, h_res)
        
        residual = hidden_states
        hidden_states, h_post, h_res = self.mlp_mhc_module.mhc_pre(hidden_states)

        hidden_states = self.pre_mlp_layernorm(hidden_states)
        # Fully Connected
        hidden_states = self.mlp(hidden_states)
        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)
        h_res = self.mlp_mhc_module.mhc_sinkhorn(h_res)
        hidden_states = self.mlp_mhc_module.mhc_post(hidden_states, h_post, residual, h_res)
        if self.has_block_post_layernorm:
            hidden_states = self.block_post_layernorm(hidden_states)
        
        return hidden_states, None


@support_torch_compile
class OpenPanguModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config
        self.config = config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = NPUVocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OpenPanguDecoderLayer(config, prefix, vllm_config),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.use_mhc = getattr(config, "use_mhc", False)
        if self.use_mhc:
            self.num_stream = getattr(config, "mhc_num_stream", 4)
            self.merge_mhc_module = NPUmHCRL(
                config=config,
                pre_only=True,
                prefix=f"{prefix}.merge_mhc_module",
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, enable_scatter=model_extra_config.parall_config.ena_seq_parallel)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
                if self.use_mhc:
                    hidden_states = hidden_states.repeat(1, self.num_stream)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        cos, sin = self.layers[self.start_layer].self_attn.rotary_emb.get_cos_sin(
            positions
        )

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(hidden_states, cos, sin, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if self.use_mhc:
            hidden_states, _, _ = self.merge_mhc_module.mhc_pre(hidden_states)
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if model_extra_config.parall_config.ena_seq_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        return hidden_states


class OpenPanguModelBase(nn.Module, SupportsPP, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = OpenPanguModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = NPUParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @post_model_forward_decorator
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        dtype = self.lm_head.weight.dtype
        logits = self.logits_processor(self.lm_head, hidden_states.to(dtype))
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


class OpenPanguMoEModel(OpenPanguModelBase, MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        # Set MoE hyperparameters
        self.expert_weights = []
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_expert_groups = 1

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, OpenPanguDecoderLayer)
            if isinstance(layer.mlp, OpenPanguMoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No MOE layer found in model.layers.")

        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.n_routed_experts = example_moe.n_routed_experts
        self.n_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, OpenPanguMoE):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping: list[tuple[str, str, int]] = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # MLA packed weights: present on some checkpoints/configs.
        mla_params_mapping: list[tuple[str, str, int]] = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        stacked_params_mapping.extend(mla_params_mapping)

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict: dict[str, torch.nn.Parameter] = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def _skip_weight(name: str) -> bool:
            if "rotary_emb.inv_freq" in name:
                return True
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                return True
            return get_spec_layer_idx_from_weight_name(self.config, name) is not None

        def _try_load_stacked(name: str, loaded_weight: torch.Tensor) -> str | None:
            """Try loading weights that are stacked/packed into a single param."""
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # Some checkpoints have expert weights; ignore if param doesn't exist.
                if "mlp.experts." in name and name not in params_dict:
                    continue

                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue

                if is_pp_missing_parameter(name_mapped, self):
                    continue

                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                return name_mapped
            return None

        def _try_load_expert(name: str, loaded_weight: torch.Tensor) -> str | None:
            """Try loading per-expert weights via fused MoE mapping."""
            saw_expert_key = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue

                saw_expert_key = True
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue

                param = params_dict[name_mapped]
                weight_loader = cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    return name_mapped

            # If it looks like an expert weight but couldn't be loaded, skip it.
            if saw_expert_key:
                return ""
            return None

        record_conv_name = []
        for name, loaded_weight in weights:
            if _skip_weight(name):
                continue

            loaded_name = _try_load_stacked(name, loaded_weight)
            if loaded_name is not None:
                loaded_params.add(loaded_name)
                continue

            loaded_name = _try_load_expert(name, loaded_weight)
            if loaded_name == "":
                continue
            if loaded_name is not None:
                loaded_params.add(loaded_name)
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            remapped_name = maybe_remap_kv_scale_name(name, params_dict)

            if remapped_name.endswith("e_score_correction_bias"):
                remapped_name = remapped_name.replace(
                    "e_score_correction_bias", "gate.e_score_correction_bias"
                )
            
            if remapped_name is None:
                continue

            if is_pp_missing_parameter(remapped_name, self):
                continue

            if "_conv" in remapped_name:
                if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                    remapped_name = insert_conv_before(remapped_name)
                else:
                    remapped_name = remapped_name.replace("_conv", "_conv.merge_conv")
                if model_extra_config.operator_opt_config.merge_q_kv_conv and "qa_conv" in remapped_name:
                    merge_conv_name = remapped_name.replace("qa_conv", "merge_conv")
                    if merge_conv_name not in record_conv_name:
                        record_conv_name.append(merge_conv_name)
                        loaded_params.add(merge_conv_name)

            param = params_dict[remapped_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(remapped_name)

        self.post_weight_load()
        return loaded_params

    def post_weight_load(self) -> None:
        for name, module in self.named_modules():
            if module is self:
                continue
            if hasattr(module, "post_weight_load"):
                module.post_weight_load()


def insert_conv_before(name: str) -> str:
    parts = name.split('.')
    for i in range(len(parts) - 1, -1, -1):
        if '_conv' in parts[i]:
            parts.insert(i, 'conv')
            break
    return '.'.join(parts)


class PanguUltraMoEForCausalLM(OpenPanguMoEModel, IsHybrid):
    """
    Hybrid architecture support (Attention + MOME conv states) for vLLM speculative 
    decoding and Multi-Token Prediction (MTP) state management.

    Key Functionalities:
    1. State Synchronization: By setting `is_hybrid` and implementing `get_mamba_state_*` 
       interfaces, the `GPUModelRunner` can automatically refresh `num_accepted_tokens` 
       based on rejection-sampler outputs after model execution.
    2. Memory Alignment: Block-size alignment for hybrid models utilizes 
       `get_mamba_state_shape_from_config` as a footprint estimate for MambaSpec 
       (refer to `MomeSpec` on attention layers).
    3. Hybrid Detection Logic:
       - If `hf_config.layer_types` exists and contains only "attention", 
         `ModelConfig.is_hybrid` defaults to False.
       - To enable hybrid features, use HF overrides to add non-attention markers 
         or adjust `layer_types` accordingly.
    """

    # Allow vLLM to keep APC enabled for this MOME-based model.
    supports_mamba_prefix_caching = True

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:

        hf = vllm_config.model_config.hf_text_config
        tp = vllm_config.parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )

        if not getattr(hf, "use_mome", False):
            conv_shape = MambaStateShapeCalculator.short_conv_state_shape(
                tp, divide(hf.hidden_size, tp), 2
            )[0]
            temporal_shape = (1, 1, 1)
            return (conv_shape, temporal_shape)

        kernel = int(getattr(hf, "router_sliding_window", 0) or 2)
        state_len = kernel - 1 + num_spec
        q_dim = getattr(hf, "q_lora_rank", hf.hidden_size)
        kv_dim = hf.kv_lora_rank
        o_dim = hf.num_attention_heads * hf.v_head_dim
        combined = q_dim + kv_dim + o_dim
        conv_shape = (state_len, divide(combined, tp))
        temporal_shape = (
            divide(hf.num_attention_heads, tp),
            hf.v_head_dim,
            1,
        )
        return (conv_shape, temporal_shape)


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> int | None:
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
