# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.v1.attention.backend import AttentionType
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    is_pp_missing_parameter,
)
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.model_executor.models import openpangu
from vllm.model_executor.models.openpangu import (
    OpenPanguMoE,
    OpenPanguMLAAttention,
    OpenPanguEmbeddedAttention,
    OpenPanguMoEModel,
    OpenPanguDecoderLayer,
    OpenPanguModel,
    OpenPanguMLP,
    check_ffn_act_fn,
    OpenPanguSinkAttention
) 


@register_patch("OpenPanguSinkAttentionPatch", OpenPanguSinkAttention)
class OpenPanguSinkAttentionPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__', '_init_rotary_emb']

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        # super().__init__()

        nn.Module.__init__(self)

        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError(
                f"total_num_heads {self.total_num_heads} "
                f"is not divisible by tp_size {self.tp_size}."
            )
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if (
            self.total_num_kv_heads > self.tp_size
            and self.total_num_kv_heads % self.tp_size != 0
        ):
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel ranks.
            raise ValueError(
                "Number of KV heads is greater than TP size, "
                f"but total_num_kv_heads {self.total_num_kv_heads} "
                f"is not divisible by tp_size {self.tp_size}."
            )
        elif self.total_num_kv_heads < self.tp_size:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel ranks.
            raise ValueError(
                f"Number of KV heads {self.total_num_kv_heads} is less than "
                f"TP size {self.tp_size}, KV heads replication is not support yet."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.qk_nope_dim = getattr(config, "qk_nope_dim", None)
        self.qk_rope_dim = getattr(config, "qk_rope_dim", None)
        self.v_channels = getattr(config, "v_channels", None)
        self.head_dim = self.qk_rope_dim + self.qk_nope_dim
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_channels
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.param_sink_with_value = getattr(config, "param_sink_with_value", False)
        self.param_sink_scalar = getattr(config, "param_sink_scalar", None)
        self.param_sink_of_head_num = getattr(config, "param_sink_of_head_dim", False)

        self.qkv_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[
                self.q_size * self.tp_size,
                self.k_size * self.tp_size,
                self.v_size * self.tp_size,
            ],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.v_channels,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        cache_layer = config.num_hidden_layers
        is_mtp_layer = getattr(config, "is_mtp_layer", False)
        if is_mtp_layer:
            cache_layer = config.num_nextn_predict_layers
        self._init_rotary_emb(
            config, cache_layer, rope_parameters=rope_parameters, quant_config=quant_config)

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} "
                    "for interleaved_sliding_window is not supported."
                )
        else:
            sliding_window = None

        # Resolve StaticSinkAttention at runtime so we pick patched symbol.
        from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkAttention

        self.attn = StaticSinkAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            sink_len=self.param_sink_number,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            attn_backend=None,
            head_size_v=self.v_channels,
        )

        if self.param_sink_number > 0:
            self.param_sink_key = torch.nn.Parameter(
                torch.empty(
                    (
                        self.param_sink_number,
                        self.num_kv_heads,
                        self.head_dim,
                    ),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                )
            )
            set_weight_attrs(
                self.param_sink_key,
                {
                    "output_dim": 1,
                    "weight_loader": self.weight_loader,
                },
            )

            if self.param_sink_with_value:
                self.param_sink_value = torch.nn.Parameter(
                    torch.empty(
                        (
                            self.param_sink_number,
                            self.num_kv_heads,
                            self.v_channels,
                        ),
                        device=current_platform.current_device(),
                        dtype=config.torch_dtype,
                    )
                )
                set_weight_attrs(
                    self.param_sink_value,
                    {
                        "output_dim": 1,
                        "weight_loader": self.weight_loader,
                    },
                )
            else:
                self.param_sink_value = torch.zeros(
                    (
                        self.param_sink_number,
                        self.num_kv_heads,
                        self.v_channels,
                    ),
                    device=current_platform.current_device(),
                    dtype=config.torch_dtype,
                )
        self.post_weight_load()

    def _init_rotary_emb(
        self,
        config: PretrainedConfig,
        cache_layer: int,
        rope_parameters: dict[str, Any] | None,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = False
        rope_parameters = {"partial_rotary_factor": self.qk_rope_dim / self.head_dim}

        from vllm.model_executor.layers.rotary_embedding import get_rope_wrapper
        self.rotary_emb = get_rope_wrapper(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rotary_dim=self.qk_rope_dim,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
            is_neox_style=is_neox_style,
            num_hidden_layers_cache=cache_layer
        )
    

@register_patch("OpenPanguModelPatch", OpenPanguModel)
class OpenPanguModelPatch(VLLMPatch):
    _attr_names_to_apply = ['load_weights'] #, 'post_weight_load']

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        attn_mlp_replace_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        has_experts = hasattr(self.config, "n_routed_experts")
        if has_experts:
            expert_merge_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
                num_redundant_experts=self.num_redundant_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            if (
                "layers" in name
                and hasattr(self.config, "num_nextn_predict_layers")
                and (self.config.num_nextn_predict_layers > 0)
            ):
                layer_idx = int(name.split("layers.")[-1].split(".")[0])
                mtp_idx = layer_idx - self.config.num_hidden_layers
                if mtp_idx >= 0 and mtp_idx < self.config.num_nextn_predict_layers:
                    continue  # skip spec decode layers for main model

            flag_dict = {"is_expert_weight": False}
            if (
                self.load_attn_mlp_weight(
                    attn_mlp_replace_mapping,
                    params_dict,
                    name,
                    loaded_weight,
                    loaded_params,
                )
                or has_experts
                and self.load_expert_weight(
                    expert_merge_mapping,
                    params_dict,
                    name,
                    loaded_weight,
                    loaded_params,
                    flag_dict,
                )
            ):
                continue
            else:
                if flag_dict["is_expert_weight"]:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)

                #####patch start: for pangu72B-VL
                if name.endswith("e_score_correction_bias"):
                    name = name.replace(
                        "e_score_correction_bias", "gate.e_score_correction_bias"
                    )
                #####patch end

                if name is None:
                    continue

                #####patch start: for pangu72B-VL
                if name not in params_dict:
                    continue
                #####patch end

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        #####patch start: for pangu72B-VL
        self.post_weight_load()
        #####patch end

        return loaded_params
