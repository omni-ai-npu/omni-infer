# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.


import contextlib
import torch
import torch_npu
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_dcp_group,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

try:
    from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkMLAAttention
except ImportError:
    logger.warning("StaticSinkMLAAttention has not being defined, skipping...")

try:
    from vllm.model_executor.layers.mome import AggregateConv
except ImportError:
    logger.warning("AggregateConv has not being defined, skipping...")

try:
    from vllm.model_executor.layers.npumome import MomeAttention
except ImportError:
    logger.warning("MomeAttention has not being defined, skipping...")

from omni_npu.attention import ops
from omni_npu.attention.backends.dsa import NPUDSAMetadata
from omni_npu.attention.backends.utils import (
    DummyKVSPMaganer,
    DummySPManager,
    KVSPMaganer,
    SPManager,
    cache_fit_shape,
    get_batch_desc,
    lazy_zero_like,
    sp_disabled,
)
from omni_npu.layers.mhc.mhc_deferred import (
    attention_mhc_deferred_fake,
    call_attention_mhc_deferred,
)
from omni_npu.layers.utils import SIDE_STREAM_NAME, named_stream
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.plugin_decorators import attn_decorator
from omni_npu.v1.layers.attention.npu_mla import MomeAttentionMixin
from omni_npu.v1.layers.attention.weight_utils import (
    install_q_b_split_loaders,
    release_q_b_proj_storage,
)
from omni_npu.v1.layers.linear import (
    ColumnParallelFlashCommLinear,
    ReplicatedFlashCommLinear,
    RowParallelFlashCommLinear,
    ShardedLinear,
)
from omni_npu.v1.layers.utils import yarn_get_mscale
from omni_npu.v1.utils import on_ascend950


class Indexer(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        sink_len: int = 0,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.on_ascend950 = on_ascend950()
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
            self.softmax_scale = self.head_dim ** -0.5
            self.weights_scale = self.n_head ** -0.5
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedFlashCommLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedFlashCommLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        if sink_len > 0:
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.weights_proj = ReplicatedFlashCommLinear(
                hidden_size, self.n_head, bias=False, quant_config=None, prefix=f"{prefix}.weights_proj"
            )
        else:
            self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
            self.weights_proj = ReplicatedFlashCommLinear(
                hidden_size, self.n_head, quant_config=None, prefix=f"{prefix}.weights_proj"
            )
        self.sink_len = sink_len
        self.wi_stream = None
        self.ki_stream = None
        self.use_rope_fusion_op = (
            model_extra_config.operator_opt_config.use_rope_fusion_op
            and not model_extra_config.operator_opt_config.enable_precision_strong_consistency
        )

    def _apply_rope(
        self,
        x: torch.Tensor,  # TND
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
    ) -> torch.Tensor:  # TND
        assert x.dim() == 3  # TND
        R, D, N = self.rope_dim, self.head_dim, x.size(1)
        if self.use_rope_fusion_op:
            shape = x.shape
            x, _ = torch_npu.npu_apply_rotary_pos_emb(
                x.view(-1, 1, N, D),
                x.view(-1, 1, N, D),
                cos.view(-1, 1, 1, R),
                sin.view(-1, 1, 1, R),
                layout="BSND",
                rotary_mode="half",
            )
            return x.view(shape)

        pe, nope = torch.split(x, [R, D - R], dim=-1)
        pe = pe.view(-1, N, 1, R)  # BNSD
        if getattr(self.config, "indexer_rope_interleave", False):
            pe = torch_npu.npu_interleave_rope(pe, cos, sin)
        else:
            pe = torch_npu.npu_rotary_mul(pe, cos, sin)
        pe = pe.view(-1, N, R)  # TND
        return torch.cat([pe, nope], dim=-1)  # TND

    def _li_prolog_ext(
        self,
        wx: torch.Tensor, # TD
        qr: torch.Tensor | dict[str, torch.Tensor], # TD
        kx: torch.Tensor, # TD
        q_cos_sin: tuple[torch.Tensor, torch.Tensor], # BNSD, BNSD
        k_cos_sin: tuple[torch.Tensor, torch.Tensor], # BNSD, BNSD
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, D = self.n_head, self.head_dim

        if self.ki_stream is not None:
            wi_ctx = torch.npu.stream(self.wi_stream)
            ki_ctx = torch.npu.stream(self.ki_stream)
            cur_stream = torch.npu.current_stream()
        else:
            wi_ctx = contextlib.nullcontext()
            ki_ctx = contextlib.nullcontext()

        qi = self.wq_b(qr)[0].view(-1, N, D)  # TND
        qi = self._apply_rope(qi, *q_cos_sin) # TND
        with wi_ctx:
            wi = self.weights_proj(wx)[0]         # TN
            if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
                wi = wi * self.weights_scale * self.softmax_scale
        with ki_ctx:
            ki = self.wk(kx)[0]                   # TD
            ki = self.k_norm(ki).view(-1, 1, D)   # T1D
            ki = self._apply_rope(ki, *k_cos_sin) # T1D
        if self.ki_stream is not None:
            cur_stream.wait_stream(self.wi_stream)
            cur_stream.wait_stream(self.ki_stream)

        return wi, qi, ki # TN, TND, T1D

    def _li_prolog(
        self,
        x: torch.Tensor,
        qr: torch.Tensor | dict[str, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # -> wi, qi, ki
        return self._li_prolog_ext(x, qr, x, (cos, sin), (cos, sin))

    def _update_cache(
        self,
        ki: torch.Tensor,  # TD or T1D
        slots_2d: torch.Tensor | NPUDSAMetadata | None,
        ki_cache: torch.Tensor | None,  # [*, pg, D] or [*, pg, 1, D]
    ):
        slots_2d = getattr(slots_2d, "slot_mapping_2d", slots_2d)
        if None in [slots_2d, ki_cache] or ki.size(0) == 0:
            assert (slots_2d is None and ki_cache is None) or ki.size(0) == 0
            return  # dummy_run or no local_kv
        assert slots_2d.dim() == 2 and ki.dim() in [2, 3] and ki_cache.dim() in [3, 4]
        assert slots_2d.size(0) == ki.size(0)
        if self.on_ascend950:
            torch_npu.npu_scatter_nd_update_(
                ki_cache,
                slots_2d,
                ki.view(-1, ki.size(-1)),
            )
        else:
            torch.ops.custom.npu_ai_infra_scatter_block_update_(
                cache_fit_shape(ki_cache, "3D"),
                slots_2d,
                ki.view(-1, ki.size(-1)),
            )

    def _apply_lightning_indexer(
        self,
        wi: torch.Tensor,  # [T, N]
        qi: torch.Tensor,  # [T, N, D]
        ki_cache: torch.Tensor,  # [*, pg, 1, D]
        q_cumlens: torch.Tensor = None,  # int32 [B]
        kv_lens: torch.Tensor = None,  # int32 [B]
        block_table: torch.Tensor = None,  # int32 [B, *]
    ) -> torch.Tensor:  # int32 [T, 1, K]
        if None in [q_cumlens, kv_lens, block_table]:
            return None  # dummy_run

        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            kwargs = dict(
                query=qi,
                key=ki_cache.unsqueeze(2),
                weights=wi,
                actual_seq_lengths_query=q_cumlens,
                actual_seq_lengths_key=kv_lens,
                block_table=block_table,
                layout_key="PA_BSND",
                layout_query="TND",
                sparse_count=self.topk_tokens,
                sparse_mode=3,
            )
            if self.on_ascend950:
                return torch_npu.npu_lightning_indexer(**kwargs)[0]

            return torch.ops.custom.npu_lightning_indexer_enhance(
                **kwargs,
                sparse_block_size=1,
                sparse_block_mode=False,
            )[0]

        if self.sink_len > 0:
            num_sink_blocks = self.sink_len // self.vllm_config.cache_config.block_size
            block_table = block_table[:, num_sink_blocks:]

        return torch_npu.npu_lightning_indexer(
            query=qi,
            key=ki_cache,
            weights=wi,
            actual_seq_lengths_query=q_cumlens,
            actual_seq_lengths_key=kv_lens - self.sink_len,
            block_table=block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.topk_tokens,
            sparse_mode=3,
        )[0]

    def forward(
        self,
        x: torch.Tensor,  # TD
        qr: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        attn_metadata: NPUDSAMetadata,
        ki_cache: torch.Tensor,  # [*, pg, 1, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wi, qi, ki = self._li_prolog(x, qr, cos, sin)
        if None in [attn_metadata, ki_cache]:
            return None, ki  # dummy_run

        self._update_cache(ki, attn_metadata, ki_cache)
        tok_idx = self._apply_lightning_indexer(
            wi,
            qi,
            ki_cache,
            q_cumlens=attn_metadata.query_cumlens.to(torch.int32),
            kv_lens=attn_metadata.seq_lens.to(torch.int32),
            block_table=attn_metadata.block_table,
        )
        return tok_idx, ki


class NPUDeepseekSparseAttention(MomeAttentionMixin, torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.dtype = model_extra_config.dtype
        self.default_cfg = {"device": current_platform.device_type, "dtype": self.dtype}

        self.num_heads = num_heads
        self.tp_size = get_tp_group().world_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix)
        self.quant_symbol = quant_config is not None
        self._init_wuk_t_uv = False
        self.is_pd_disagg = vllm_config.kv_transfer_config is not None

        self.kv_nz = model_extra_config.operator_opt_config.kv_nz
        self.noncontiguous_kv = model_extra_config.operator_opt_config.use_noncontiguous_kv
        self.enable_decode_multi_stream = model_extra_config.operator_opt_config.enable_multi_stream
        self.split_q_up_in_multistream = (
            self.enable_decode_multi_stream
            and model_extra_config.operator_opt_config.split_q_up_in_multistream
            and self.q_lora_rank is not None
        )
        self.on_ascend950 = on_ascend950()
        self.use_mlaprolog = model_extra_config.operator_opt_config.enable_mlaprolog
        self.use_omni_cache = model_extra_config.operator_opt_config.use_omni_cache
        self.ena_sp = model_extra_config.parall_config.ena_seq_parallel
        self.ena_cp = model_extra_config.parall_config.ena_context_parallel
        self.ena_kvsp = bool(get_dcp_group().world_size > 1)
        self.sharded_o_proj = model_extra_config.parall_config.sharded_o_proj
        self.num_speculative_tokens = (
            0 if not vllm_config.speculative_config else vllm_config.speculative_config.num_speculative_tokens
        )

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedFlashCommLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelFlashCommLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
                disable_tp=self.ena_cp,
            )
            if self.split_q_up_in_multistream:
                self.q_b_nope_proj = ColumnParallelFlashCommLinear(
                    self.q_lora_rank,
                    self.num_heads * self.qk_nope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.q_b_nope_proj",
                    disable_tp=self.ena_cp,
                )
                self.q_b_pe_proj = ColumnParallelFlashCommLinear(
                    self.q_lora_rank,
                    self.num_heads * self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.q_b_pe_proj",
                    disable_tp=self.ena_cp,
                )
                install_q_b_split_loaders(
                    self.q_b_proj,
                    self.q_b_nope_proj,
                    self.q_b_pe_proj,
                    self.qk_head_dim,
                    self.qk_nope_head_dim,
                )
        else:
            self.q_proj = ColumnParallelFlashCommLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
                disable_tp=self.ena_cp,
            )

        self.kv_a_proj_with_mqa = ReplicatedFlashCommLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=self.dtype)
        self.kv_b_proj = ColumnParallelFlashCommLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
            disable_tp=self.ena_cp,
        )

        if self.sharded_o_proj:
            self.o_proj = ShardedLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                shard_group=get_tp_group(),
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
        else:
            self.o_proj = RowParallelFlashCommLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )

        assert self.q_b_proj.tp_size == self.kv_b_proj.tp_size
        self.num_local_heads = self.num_heads // self.q_b_proj.tp_size

        self.rope_interleaved = getattr(
            config,
            "rope_interleaved",
            getattr(config, "rope_interleave", True),
        )
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn" if config.rope_parameters.get("apply_yarn_scaling", True) else "deepseek_llama_scaling"
            )
            is_neox_style = False  # Deepseek V3.2
        else:
            is_neox_style = True  # GLM 5 (for generating neox style sin and cos caches.
            # gptj style will be applied by the npu_interleave_rope operator)

        rope_scaling = getattr(config, "rope_scaling", None)
        is_mrope = rope_scaling is not None and rope_scaling.get("mrope_section") is not None
        if is_mrope:
            cache_layer = config.num_hidden_layers
            is_mtp_layer = getattr(config, "is_mtp_layer", False)
            if is_mtp_layer:
                cache_layer = config.num_nextn_predict_layers
            from vllm.model_executor.layers.rotary_embedding import get_rope_wrapper

            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rotary_dim=qk_rope_head_dim,
                base=config.rope_parameters["rope_theta"],
                rope_scaling=rope_scaling,
                num_hidden_layers_cache=cache_layer,
            )
        else:
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=is_neox_style,
            )

        if config.rope_parameters["rope_type"] != "default" and config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.use_mome = getattr(config, "use_mome", False)
        self.merge_q_kv_conv = model_extra_config.operator_opt_config.merge_q_kv_conv
        self.param_sink_number = getattr(config, "param_sink_number", 0)
        self.index_topk = config.index_topk

        # IndexCache config
        index_topk_freq = getattr(config, "index_topk_freq", 1)
        index_topk_pattern = getattr(config, "index_topk_pattern", None)
        index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
        indexer_types = getattr(config, "indexer_types", None)
        layer_id = extract_layer_index(prefix)

        if index_topk_pattern is not None:
            _skip_topk = index_topk_pattern[layer_id] == "S"
        elif indexer_types is not None:
            _skip_topk = indexer_types[layer_id] == "shared"
        else:
            _skip_topk = (
                max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq
                != 0
            )
        self.skip_topk = _skip_topk
        if not self.skip_topk:
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                self.param_sink_number,
                f"{prefix}.indexer",
            )
        else:
            self.indexer = None

        if self.use_mome:
            if self.noncontiguous_kv:
                num_extra_token = 1 if self.is_pd_disagg else 0
                fake_num_spec_tokens = max(self.num_speculative_tokens, num_extra_token)
                self._build_mome_conv(
                    fake_num_spec_tokens, config, vllm_config, prefix
                )
            else:
                self.qa_conv = AggregateConv(
                    self.q_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn"
                )
                self.compresskv_conv = AggregateConv(
                    self.kv_lora_rank, config, vllm_config, output_parallel=False, attn_prefix=f"{prefix}.attn"
                )
                self.o_conv = AggregateConv(
                    self.num_local_heads * self.v_head_dim,
                    config,
                    vllm_config,
                    output_parallel=True,
                    attn_prefix=f"{prefix}.attn",
                )

        if self.param_sink_number == 0:
            self.attn = MLAAttention(
                **self._inner_mla_attn_kwargs(cache_config, quant_config, prefix),
                use_sparse=True,
                indexer=self.indexer,
            )
        else:
            self.attn = StaticSinkMLAAttention(
                **self._static_sink_attn_base_kwargs(
                    cache_config, quant_config, prefix
                ),
                use_sparse=True,
                indexer=self.indexer,
            )
            self._register_sink_params(config)

        if self.noncontiguous_kv:
            self.dummy_value_cache = torch.zeros((1, cache_config.block_size, 1, self.kv_lora_rank), **self.default_cfg)

        self.post_weight_load()  # To enable dummy run with out weight

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.side_stream = (
            named_stream(SIDE_STREAM_NAME)
            if self.enable_decode_multi_stream
            else None
        )
        self.pre_epilog_callback = None

        if not self.skip_topk and model_extra_config.operator_opt_config.li_prolog_multi_stream:
            self.wi_stream = named_stream("wi_stream")
            self.ki_stream = named_stream("ki_stream")
            self.indexer.wi_stream = self.wi_stream
            self.indexer.ki_stream = self.ki_stream

    def post_weight_load(self):
        if self._init_wuk_t_uv and getattr(self.attn.impl, "W_UK_T", None) is not None:
            is_weight_nz = getattr(self.kv_b_proj.weight, "is_weight_nz", False)
            if is_weight_nz:
                self.kv_b_proj.weight.data = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, torch_npu.Format.ND)
            self.attn.impl.process_weights_after_loading(self.kv_b_proj.weight.dtype)
            if is_weight_nz:
                self.kv_b_proj.weight.data = torch_npu.npu_format_cast(
                    self.kv_b_proj.weight.data, torch_npu.Format.FRACTAL_NZ
                )
        else:
            self._init_wuk_t_uv = True

        if self.param_sink_number > 0:
            param_sink_compressed_kv = self.kv_a_layernorm(self.param_sink_compressed_kv)
            self.attn.update_sink_kv(self.param_sink_k_pe, param_sink_compressed_kv)

        # With the split q-up projections populated, decode takes
        # _forward_decode_multistream, so the full q_b_proj storage is
        # redundant; release it (reload-safe, see release_q_b_proj_storage).
        # Skip when the fused mla_prolog path can run: it consumes
        # q_b_proj.weight directly.
        use_fused_mla_prolog = (
            self.use_mlaprolog and not self.use_mome and self.param_sink_number == 0
        )
        if self.split_q_up_in_multistream and not use_fused_mla_prolog:
            release_q_b_proj_storage(self.q_b_proj)

    def _apply_rope(
        self,
        x: torch.Tensor,  # BNSD or TND or TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
    ) -> torch.Tensor:
        assert x.dim() in [2, 3, 4]
        assert cos.dim() == 4 and sin.dim() == 4
        T, D, shape = x.size(0), x.size(-1), x.shape
        x = x.view(T, -1, 1, D)  # BNSD
        if self.rope_interleaved:
            x = torch_npu.npu_interleave_rope(x, cos, sin)
        else:
            x = torch_npu.npu_rotary_mul(x, cos, sin)
        return x.view(shape)

    def _q_absorb(
        self,
        q_lora: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
    ) -> tuple[torch.Tensor, torch.Tensor]:  # TND
        Q = self.qk_nope_head_dim
        R = self.qk_rope_head_dim
        N = self.num_heads // self.q_b_proj.tp_size

        if self.split_q_up_in_multistream:
            # The full q_b_proj storage is released after loading in this
            # mode; the split projections are numerically identical per row.
            q_nope = self.q_b_nope_proj(q_lora)[0].view(-1, N, Q)  # TND
            q_pe = self.q_b_pe_proj(q_lora)[0].view(-1, N, R)  # TND
        else:
            q = self.q_b_proj(q_lora)[0].view(-1, N, Q + R)  # TND
            q_nope, q_pe = torch.split(q, [Q, R], dim=-1)  # TND
        q_pe = self._apply_rope(q_pe, cos, sin)  # TND
        q_nope = self._q_nope_absorb(q_nope)

        return q_nope, q_pe  # TND

    def _q_nope_absorb(self, q_nope: torch.Tensor) -> torch.Tensor:
        absorb = self.attn.impl.W_UK_T
        if absorb.size(-1) % 128 != 0 or absorb.size(-2) % 128 != 0:
            return (q_nope.transpose(0, 1) @ absorb).transpose(1, 0)

        if q_nope.size(1) * q_nope.size(2) < 65536:
            args = {"input": q_nope, "perm_x1": (1, 0, 2)}
        else:  # perm_x1 only support batch*k < 65536
            args = {"input": q_nope.transpose(0, 1)}
        return torch_npu.npu_transpose_batchmatmul(
            **args,  # TND -> NTD
            weight=absorb,  # [N, Q, L]
            perm_y=(1, 0, 2),  # NTD -> TND
        )

    def _kv_norm_rope_cache(
        self,
        latent_kv: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        slots: torch.Tensor | NPUDSAMetadata | None,  # None for dummy_run
        kv_cache: tuple[torch.Tensor] | None,  # None for dummy_run
        fused_op: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # k_nope: T1D, k_pe: T1D
        R, L = self.qk_rope_head_dim, self.kv_lora_rank
        slots_2d = getattr(slots, "slot_mapping_2d", None)
        slots = getattr(slots, "slot_mapping", slots)
        assert latent_kv.dim() == 2 and latent_kv.size(1) == R + L
        # TODO: CP/SP does not support DP > 1
        assert slots is None or slots.size(0) == latent_kv.size(0)
        valid_cache = None not in [kv_cache, slots]
        if latent_kv.size(0) == 0:  # no local_kv
            return cos.new_zeros(0, 1, L), cos.new_zeros(0, 1, R)
        fused_op = fused_op and model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache

        if valid_cache and fused_op and self.noncontiguous_kv:
            assert kv_cache[0].size(-1) == L + R
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
                latent_kv.view(-1, 1, 1, L + R),  # BNSD
                self.kv_a_layernorm.weight,
                cos,
                sin,  # BNSD
                slots,
                k_cache=None,
                ckv_cache=cache_fit_shape(kv_cache[0], "4D"),
                k_rope_scale=None,
                k_rope_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if (self.kv_nz and self.dtype != torch.float16) else "PA",
                rotary_mode="half" if not self.rope_interleaved else "interleave",
                quant_mode="none",
                is_output_kv=True,
            )
        elif valid_cache and fused_op and self.rope_interleaved:
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_kv.view(-1, 1, 1, L + R),  # BNSD
                self.kv_a_layernorm.weight,
                cos,
                sin,  # BNSD
                slots,
                cache_fit_shape(kv_cache[1], "4D"),
                cache_fit_shape(kv_cache[0], "4D"),
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if (self.kv_nz and self.dtype != torch.float16) else "PA",
                is_output_kv=True,
            )  # -> [*, pg, 1, L], [*, pg, 1, R], BNSD, BNSD
        else:
            k_nope, k_pe = torch.split(latent_kv, [L, R], dim=-1)  # TD
            k_nope = self.kv_a_layernorm(k_nope)  # TD
            k_pe = self._apply_rope(k_pe, cos, sin)  # TD
            if valid_cache:
                if self.on_ascend950:
                    assert slots_2d is not None
                    kv = torch.cat([k_nope, k_pe], dim=-1)
                    torch_npu.npu_scatter_nd_update_(kv_cache[0], slots_2d, kv)
                elif self.noncontiguous_kv:
                    assert slots_2d is not None
                    slot_indices = slots_2d
                    kv = torch.cat([k_nope, k_pe], dim=-1)
                    torch.ops.custom.npu_ai_infra_scatter_block_update_(kv_cache[0], slot_indices, kv)
                else:
                    assert not self.noncontiguous_kv

                    def cache_kv(x: torch.Tensor, cache: torch.Tensor):
                        torch_npu.npu_scatter_nd_update_(
                            cache.view(-1, 1, cache.size(-1)),  # T1D
                            slots.view(-1, 1),  # T1
                            x.view(-1, 1, x.size(-1)),  # T1D
                        )

                    cache_kv(k_nope, kv_cache[0])
                    cache_kv(k_pe, kv_cache[1])
        return k_nope.view(-1, 1, L), k_pe.view(-1, 1, R)

    def _post_attn_absorb(self, out: torch.Tensor) -> torch.Tensor:
        assert out.dim() == 3
        if out.size(1) * out.size(2) < 65536:
            args = {"input": out, "perm_x1": (1, 0, 2)}
        else:  # perm_x1 only support batch*k < 65536
            args = {"input": out.transpose(0, 1)}

        return torch_npu.npu_transpose_batchmatmul(
            **args,  # TND -> NTD
            weight=self.attn.impl.W_UV,  # [N, L, V]
            perm_y=(1, 0, 2),  # NTD -> TND
        ).reshape(out.size(0), -1)  # [T, NV]

    def _mla_prolog(
        self,
        x: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        kv_cache: tuple[torch.Tensor],
        attn_metadata: NPUDSAMetadata,
    ):
        bs, _ = x.view(-1, x.shape[-1]).shape
        q_nope, q_pe, dequant_scale_q_nope, q_norm, dequant_scale_q_norm = torch_npu.npu_mla_prolog_v3(
            token_x=x.view(bs, 1, -1),
            weight_dq=self.q_a_proj.weight,  # BF16, NZ
            weight_uq_qr=self.q_b_proj.weight,  # BF16, NZ
            weight_uk=self.attn.impl.W_UK_T,  # BF16, ND
            weight_dkv_kr=self.kv_a_proj_with_mqa.weight,  # BF16, NZ
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rope_sin=sin.squeeze(1),
            rope_cos=cos.squeeze(1),
            kv_cache=kv_cache[0],
            kr_cache=kv_cache[1],
            cache_index=attn_metadata.slot_mapping.view(bs, -1),
            dequant_scale_x=None,
            dequant_scale_w_dq=None,
            dequant_scale_w_uq_qr=self.q_b_proj.weight_scale.view(1, -1) if self.quant_symbol else None,
            dequant_scale_w_dkv_kr=None,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_BSND",
            query_norm_flag=True,
            weight_quant_mode=1 if self.quant_symbol else 0,
        )
        k_nope = kv_cache[0]
        k_pe = kv_cache[1]
        q_nope = q_nope.view(bs, self.num_local_heads, self.kv_lora_rank)
        q_pe = q_pe.view(bs, self.num_local_heads, -1)
        if self.quant_symbol:
            q_norm = q_norm.view(-1, q_norm.shape[-1])
            dequant_scale_q_norm = dequant_scale_q_norm.view(-1)
            q_norm = {"x_int8": q_norm, "pertoken_scale": dequant_scale_q_norm}
        return q_nope, q_pe, q_norm, k_nope, k_pe, dequant_scale_q_nope, dequant_scale_q_norm

    # ======================= attention =======================

    def _apply_sink_offset(self, topk_idx: torch.Tensor | None):
        if self.param_sink_number == 0 or topk_idx is None:
            return topk_idx  # dummy_run or sink_disabled

        if not self.noncontiguous_kv:  # only contiguous kv
            sink_indices = torch.arange(
                self.param_sink_number,
                device=topk_idx.device,
                dtype=topk_idx.dtype,
            ).expand(topk_idx.size(0), 1, self.param_sink_number)
            mask = (topk_idx != -1).to(topk_idx.dtype)
            with_offset = topk_idx + mask * self.param_sink_number
            topk_idx = torch.concat([sink_indices, with_offset], dim=2)
        return topk_idx

    def _apply_attention_rescale_pioneer(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_cumlens: torch.Tensor,
        kv_lens: torch.Tensor,
        topk_idx: torch.Tensor,
        block_table: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:

        output_normal, smax_normal, ssum_normal = (
            torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(
                query=q_nope,
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                query_rope=q_pe,
                sparse_indices=topk_idx,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=q_cumlens.to(torch.int32),
                actual_seq_lengths_kv=kv_lens.to(torch.int32),
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                batch_invariant=True,
                return_softmax_lse=True,
            )
        )


        sink_k_nope = self.attn.impl.sink_k_nope.repeat(q_cumlens.numel(), 1, 1)
        sink_value = sink_k_nope
        sink_k_pe = self.attn.impl.sink_kv[..., self.kv_lora_rank:].repeat(q_cumlens.numel(), 1, 1)

        act_seq_qlen = q_cumlens.to(torch.int64)
        actual_seq_sink_len = torch.arange(
            self.param_sink_number,
            (q_cumlens.numel() + 1) * self.param_sink_number,
            self.param_sink_number,
            dtype=q_cumlens.dtype,
            device=q_cumlens.device,
        ).to(torch.int64)

        meta_data = torch.ops.custom._npu_fused_infer_attention_sink_metadata(
            num_heads_q=self.num_local_heads,
            num_heads_kv=1,
            head_dim_qk=q_nope.shape[-1],
            head_dim_v=sink_k_nope.shape[-1],
            actual_seq_lengths=act_seq_qlen,
            actual_seq_lengths_kv=actual_seq_sink_len,
            sparse_mode=0,
            batch_invariant=True,
            input_layout="TND",
            input_layout_kv="TND",
            rope_head_dim=q_pe.shape[-1],
        )

        output_sink, _, smax_sink, ssum_sink = (
            torch.ops.custom.npu_fused_infer_attention_sink_v2(
                q_nope,
                sink_k_nope,
                sink_value,
                query_rope=q_pe,
                key_rope=sink_k_pe,
                actual_seq_qlen=act_seq_qlen,
                actual_seq_kvlen=actual_seq_sink_len,
                num_query_heads=self.num_local_heads,
                num_key_value_heads=1,
                softmax_scale=self.scaling,
                input_layout="TND",
                sparse_mode=0,
                batch_invariant=True,
                return_softmax_lse=False,
                return_softmax_max_sum=True,
                meta_data=meta_data,
            )
        )

        attn_output = torch.empty_like(output_normal)
        smax_normal = smax_normal.squeeze(0).unsqueeze(2)  # 1TN -> TN1
        ssum_normal = ssum_normal.squeeze(0).unsqueeze(2)  # 1TN -> TN1

        attn_output, _, _ = ops.apply_FA_rescale_forward(
            output_normal, smax_normal, ssum_normal, output_sink, smax_sink, ssum_sink)

        return attn_output

    @attn_decorator(type="dsa")
    def _apply_attn_absorb(
        self,
        q_nope: torch.Tensor,  # [T, N, D]
        q_pe: torch.Tensor,  # [T, N, R]
        q_cumlens: torch.Tensor = None,  # int32 [B]
        kv_lens: torch.Tensor = None,  # int32 [B]
        topk_idx: torch.Tensor = None,  # int32 [T, 1, K]
        block_table: torch.Tensor = None,  # int32 [B, *]
        kv_cache: torch.Tensor = None,
        attn_metadata: NPUDSAMetadata = None,  # required by omni-cache
    ) -> torch.Tensor:  # [T, N, L]
        if None in [q_cumlens, kv_lens, block_table, topk_idx]:
            return torch.zeros_like(q_nope)  # dummy
        if self.noncontiguous_kv:
            assert self.param_sink_number > 1
            if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
                return self._apply_attention_rescale_pioneer(
                    q_nope, q_pe, q_cumlens, kv_lens,
                    topk_idx, block_table, kv_cache,
                )
            kwargs = dict(
                query=q_nope,
                query_rope=q_pe,
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                sparse_indices=topk_idx,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=q_cumlens,
                actual_seq_lengths_kv=kv_lens,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                key_sink=self.attn.impl.sink_kv,
                value_sink=self.attn.impl.sink_k_nope,
            )
            if model_extra_config.operator_opt_config.use_batch_invariant_op:
                kwargs["batch_invariant"] = True
            return torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(**kwargs)[0]

        return torch.ops.custom.npu_sparse_flash_attention_enhance(
            query=q_nope,  # [T, N, L]
            key=kv_cache[0],  # [*, pg, 1, L]
            value=kv_cache[0],  # [*, pg, 1, L]
            query_rope=q_pe,  # [T, N, R]
            key_rope=kv_cache[1],  # [*, pg, 1, R]
            sparse_indices=topk_idx,  # [T, 1, 2048]
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            block_table=block_table,
            actual_seq_lengths_query=q_cumlens,
            actual_seq_lengths_kv=kv_lens,
            scale_value=self.scaling,
            attention_mode=2,
            sparse_mode=3,
        )[0]  # -> [T, N, L]

    # ======================= forward =======================

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        topk_indices_buffer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.npu_dsa_forward(
            x, cos, sin, self.prefix, topk_indices_buffer
        )

    def forward_mhc_deferred(
        self, x, cos, sin, residual, mhc_layer_name, task_key, topk_indices_buffer=None
    ):
        hidden_states, h_post, h_res = torch.ops.vllm.npu_dsa_forward_mhc_deferred(
            x, cos, sin, residual, self.prefix, mhc_layer_name, task_key
        )
        return (hidden_states, h_post, h_res), topk_indices_buffer

    def _forward_prefill(
        self,
        x: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        attn_metadata: NPUDSAMetadata = None,
        kv_cache: tuple[torch.Tensor] = None,
        topk_indices_buffer: torch.Tensor | None = None,
        pd_mixed_flag: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not self.sharded_o_proj  # TODO: sharded_o_proj only support cp
        assert not attn_metadata or not self.ena_kvsp  # TODO: kvsp only support cp

        def get_mome_args():
            if self.noncontiguous_kv:
                return {"is_prefill": True}
            else:
                return {"only_prefill": bool(pd_mixed_flag)}

        if attn_metadata:
            kv_lens = attn_metadata.seq_lens.to(torch.int32)
            q_cumlens = attn_metadata.query_cumlens.to(torch.int32)
            block_table = attn_metadata.block_table
            ki_cache = kv_cache[1] if self.noncontiguous_kv else kv_cache[2]
        else:
            q_cumlens, kv_lens, block_table, ki_cache = None, None, None, None
        is_dummy = None in [q_cumlens, kv_lens, block_table]

        sp_manager = getattr(attn_metadata, "sp_manager", DummySPManager())

        if self.ena_sp:
            sp_cos = sp_manager.slice_tokens(cos, cached="cos")
            sp_sin = sp_manager.slice_tokens(sin, cached="sin")

        q = self.q_a_proj(x)[0]  # TD
        q = sp_manager.ag_tokens(q) if self.ena_sp and self.use_mome else q
        q = self._maybe_mome_q(q, get_mome_args)  # TD
        q = self.q_a_layernorm(q)  # TD
        q_lora = sp_manager.ag_tokens(q) if self.ena_sp and not self.use_mome else q

        q_nope, q_pe = self._q_absorb(q_lora, cos, sin)  # TND

        kv = self.kv_a_proj_with_mqa(x)[0]  # TD
        kv = self._maybe_mome_kv(kv, get_mome_args)  # TD
        if self.ena_sp:
            kv = sp_manager.ag_tokens(kv)
        self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)

        cur_stream = torch.npu.current_stream()

        if not self.skip_topk:
            if model_extra_config.operator_opt_config.li_prolog_multi_stream:
                self.ki_stream.wait_stream(cur_stream)
                self.wi_stream.wait_stream(cur_stream)

            if self.ena_sp:
                wi, qi, ki = self.indexer._li_prolog(x, q, sp_cos, sp_sin)
                ki = sp_manager.ag_tokens(ki)
                wi = sp_manager.ag_tokens(wi)
                qi = sp_manager.ag_tokens(qi)
            else:
                wi, qi, ki = self.indexer._li_prolog(x, q, cos, sin)
            self.indexer._update_cache(ki, attn_metadata, ki_cache)
            topk_idx = self.indexer._apply_lightning_indexer(wi, qi, ki_cache, q_cumlens, kv_lens, block_table)
        elif not is_dummy:
            topk_idx = topk_indices_buffer
        else:
            topk_idx = None

        next_topk_indices_buffer = (
            topk_idx if topk_idx is not None else topk_indices_buffer
        )
        topk_idx = self._apply_sink_offset(topk_idx)

        out = self._apply_attn_absorb(
            q_nope,
            q_pe,  # [T, N, D]
            q_cumlens,
            kv_lens,
            topk_idx,  # int32 [T, 1, K]
            block_table,  # int32 [B, *]
            kv_cache,
            attn_metadata=attn_metadata,
        )  # -> TND

        out = self._post_attn_absorb(out)  # T,ND
        out = self._maybe_mome_out(out, get_mome_args)  # T,ND
        if self.ena_sp:
            out = sp_manager.align_tokens(out)  # T,ND
        out = self.o_proj(out)[0]
        if self.ena_sp and out.size(0) != x.size(0):
            out = sp_manager.slice_tokens(out)
        return out, next_topk_indices_buffer

    def _forward_prefill_cp(
        self,
        sp_x: torch.Tensor,  # TD
        cos: torch.Tensor,  # BNSD
        sin: torch.Tensor,  # BNSD
        attn_metadata: NPUDSAMetadata = None,
        kv_cache: tuple[torch.Tensor] = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.q_b_proj.tp_size == 1  # full head required
        assert self.kv_b_proj.tp_size == 1  # full head required
        assert self.ena_sp  # dependency
        assert not self.use_omni_cache
        """
        SP here refers to standard SP splitting, i.e., splitting tokens with ceil division
        based on sp_size.

        CP refers to zigzag-form SP splitting, aimed at adjusting query distribution to
        balance attention computation across ranks.
        """

        if attn_metadata:
            ki_cache = kv_cache[1] if self.noncontiguous_kv else kv_cache[2]
        else:
            ki_cache = None

        def get_mome_args():
            if self.noncontiguous_kv:
                return {"is_prefill": True}
            else:
                return {"only_prefill": False}

        sp_manager = getattr(attn_metadata, "sp_manager", DummySPManager())
        q_cumlens, kv_lens, _, blk_table = sp_manager.cp_attn_meta()

        if self.ena_kvsp:
            assert not self.use_mome and not self.noncontiguous_kv
            kvsp_manager = getattr(attn_metadata, "kvsp_manager", DummyKVSPMaganer())
            local_cos = kvsp_manager.select_local(cos, cached="cos")
            local_sin = kvsp_manager.select_local(sin, cached="sin")
            local_slots, local_slots_2d = kvsp_manager.local_slots()

        cur_stream = torch.npu.current_stream()
        sub_stream = named_stream("npu_dsa_sub_stream")

        sp_cos_sin = (sp_manager.slice_tokens(cos, cached="cos"), sp_manager.slice_tokens(sin, cached="sin"))
        cp_cos_sin = (sp_manager.cp_slice(cos, cached="cos"), sp_manager.cp_slice(sin, cached="sin"))

        q_lora = self.q_a_proj(sp_x)[0]  # TD, sp

        if self.use_mome:
            q_lora = sp_manager.ag_tokens(q_lora)
            q_lora = self._maybe_mome_q(q_lora, get_mome_args)  # TD
            q_lora = sp_manager.cp_slice(q_lora)  # TD, cp
        else:
            q_lora = sp_manager.sp_to_cp(q_lora)  # TD, cp

        q_lora = self.q_a_layernorm(q_lora)  # TD, cp
        q_nope, q_pe = self._q_absorb(q_lora, *cp_cos_sin)  # TND, full head, cp

        kv = self.kv_a_proj_with_mqa(sp_x)[0]  # TD, sp
        if self.ena_kvsp:
            kv = kvsp_manager.sp_to_local(kv)  # TD, sp -> local
            self._kv_norm_rope_cache(kv, local_cos, local_sin, local_slots, kv_cache)
            nope_cache, rope_cache, _ = kv_cache or (None, None, None)
            ag_nope_cache, *_ = kvsp_manager.ag_pages(nope_cache, seperate=True)
            ag_rope_cache, *_ = kvsp_manager.ag_pages(rope_cache, seperate=True)
            sub_stream.wait_stream(cur_stream)
            with torch.npu.stream(sub_stream):  # all_gather in another stream
                nope_cache = ag_nope_cache()
                rope_cache = ag_rope_cache()
        else:
            kv = sp_manager.ag_tokens(kv)  # TD
            kv = self._maybe_mome_kv(kv, get_mome_args)  # TD
            self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)

        if not self.skip_topk:
            if model_extra_config.operator_opt_config.li_prolog_multi_stream:
                self.ki_stream.wait_stream(cur_stream)
                self.wi_stream.wait_stream(cur_stream)

            wi, qi, ki = self.indexer._li_prolog_ext(sp_x, q_lora, sp_x, cp_cos_sin, sp_cos_sin)  # TND
            wi = sp_manager.sp_to_cp(wi)  # T, cp

            if self.ena_kvsp:
                ki = kvsp_manager.sp_to_local(ki)
                self.indexer._update_cache(ki, local_slots_2d, ki_cache)
                ki_cache, _, blk_table = kvsp_manager.ag_pages(ki_cache)
                kv_cache = (nope_cache, rope_cache, ki_cache)
                cur_stream.wait_stream(sub_stream)  # wait for ag_pages
            else:
                ki = sp_manager.ag_tokens(ki)
                if hasattr(attn_metadata, "cache_fn") and not self.noncontiguous_kv:
                    attn_metadata.cache_fn(ki.view(-1, ki.size(-1)), ki_cache)
                else:
                    self.indexer._update_cache(ki, attn_metadata, ki_cache)

        if self.sharded_o_proj:
            sub_stream.wait_stream(cur_stream)
            with torch.npu.stream(sub_stream):
                self.o_proj.prefetch(sub_stream)

        if not self.skip_topk:
            topk_idx = self.indexer._apply_lightning_indexer(
                wi,
                qi,  # TND, cp
                ki_cache,
                q_cumlens,
                kv_lens,  # int32 [2B]
                blk_table,  # int32 [2B, *]
            )  # int32 [T_cp, 1, K] or None for dummy_run
        else:
            # A full DSA layer returns CP-layout top-k; shared consumes it directly.
            topk_idx = topk_indices_buffer

        next_topk_indices_buffer = (
            topk_idx if topk_idx is not None else topk_indices_buffer
        )
        topk_idx = self._apply_sink_offset(topk_idx)

        cp_out = self._apply_attn_absorb(
            q_nope,
            q_pe,  # TND, full head, cp
            q_cumlens,
            kv_lens,  # int32 [2B]
            topk_idx,  # int32 [T, 1, K]
            blk_table,  # int32 [2B, *]
            kv_cache,
            attn_metadata=attn_metadata,
        )
        cp_out = self._post_attn_absorb(cp_out)

        if self.use_mome:
            cp_out = sp_manager.cp_to_sp(cp_out)  # sp
            cp_out = sp_manager.ag_tokens(cp_out)  # TD
            cp_out = self._maybe_mome_out(cp_out, get_mome_args)  # TD
            # Pad to a multiple of sp_size before o_proj, exactly as the
            # non-CP prefill does: this branch hands o_proj the full token
            # set, and a ReduceScatter y_transform (the correct SP setting)
            # requires the row count to divide by the group size. Without
            # the pad, any request whose token count is not a multiple of
            # sp_size aborts in the communicator.
            cp_out = sp_manager.align_tokens(cp_out)  # TD, padded
            cp_out = self._apply_o_proj(cp_out)
            if cp_out.size(0) != sp_x.size(0):
                cp_out = sp_manager.slice_tokens(cp_out)
            return cp_out, next_topk_indices_buffer

        if self.o_proj.requires_input_partition():
            cp_out = sp_manager.sp_to_tp(cp_out)
        cp_out = self._apply_o_proj(cp_out)
        return sp_manager.cp_to_sp(cp_out), next_topk_indices_buffer

    def _forward_decode(
        self,
        x: torch.Tensor,   # TD
        cos: torch.Tensor, # BNSD
        sin: torch.Tensor, # BNSD
        attn_metadata: NPUDSAMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        topk_indices_buffer: torch.Tensor | None = None,
        pd_mixed_flag: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not self.ena_sp  # TODO: support decode sp in the future
        assert not self.sharded_o_proj  # sharded_o_proj is for prefill only
        use_fused_mla_prolog = self.use_mlaprolog and not self.use_mome and self.param_sink_number == 0
        can_decode_multistream = self.side_stream is not None and not self.skip_topk
        if can_decode_multistream and not use_fused_mla_prolog and not self.ena_kvsp:
            return self._forward_decode_multistream(
                x, cos, sin, attn_metadata, kv_cache,
                topk_indices_buffer, pd_mixed_flag,
            )

        ki_cache = kv_cache[1] if self.noncontiguous_kv else kv_cache[2]
        q_cumlens = attn_metadata.query_cumlens.to(torch.int32)
        kv_lens = attn_metadata.seq_lens.to(torch.int32)
        blk_table = attn_metadata.block_table

        cur_stream = torch.npu.current_stream()

        def get_mome_args():
            if self.noncontiguous_kv:
                return {}
            else:
                return {
                    "force_decode": True if pd_mixed_flag == 1 else False,
                    "short_prefill": True if pd_mixed_flag == 2 else False,
                }

        ki_cache = kv_cache[1] if self.noncontiguous_kv else kv_cache[2]

        if use_fused_mla_prolog:
            q_nope, q_pe, q_lora, *_ = self._mla_prolog(x, cos, sin, kv_cache, attn_metadata)
        else:
            q_lora = self.q_a_proj(x)[0]  # TD
            q_lora = self._maybe_mome_q(q_lora, get_mome_args)  # TD
            q_lora = self.q_a_layernorm(q_lora)  # TD
            q_nope, q_pe = self._q_absorb(q_lora, cos, sin)  # TND

            kv = self.kv_a_proj_with_mqa(x)[0]  # TD
            kv = self._maybe_mome_kv(kv, get_mome_args)  # TD
            self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)

        if not self.skip_topk:
            if model_extra_config.operator_opt_config.li_prolog_multi_stream:
                self.ki_stream.wait_stream(cur_stream)
                self.wi_stream.wait_stream(cur_stream)

            wi, qi, ki = self.indexer._li_prolog(x, q_lora, cos, sin)
            self.indexer._update_cache(ki, attn_metadata, ki_cache)

            if self.ena_kvsp:
                assert not self.use_mome and not self.noncontiguous_kv
                kvsp_manager: KVSPMaganer = attn_metadata.kvsp_manager
                nope_cache, rope_cache, ki_cache = kv_cache
                nope_cache, *_ = kvsp_manager.ag_pages(nope_cache)
                rope_cache, *_ = kvsp_manager.ag_pages(rope_cache)
                ki_cache, blk_table, _ = kvsp_manager.ag_pages(ki_cache)
                # replace local-cache with full-cache
                kv_cache = (nope_cache, rope_cache, ki_cache)

            topk_idx = self.indexer._apply_lightning_indexer(
                wi,
                qi,  # TND
                ki_cache,  # [*, pg, 1, D]
                q_cumlens,
                kv_lens,  # [B]
                blk_table,  # [B, *]
            )  # -> int32 [T, 1, K]
        else:
            topk_idx = topk_indices_buffer[:x.shape[0]]

        next_topk_indices_buffer = (
            topk_idx if topk_idx is not None else topk_indices_buffer
        )
        topk_idx = self._apply_sink_offset(topk_idx)

        out = self._apply_attn_absorb(
            q_nope=q_nope,
            q_pe=q_pe,  # TND
            q_cumlens=q_cumlens,
            kv_lens=kv_lens,  # [B]
            topk_idx=topk_idx,  # [T, 1, K]
            block_table=blk_table,  # [B, *]
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )  # -> TND

        return (
            self._decode_attn_epilog(out, get_mome_args),
            next_topk_indices_buffer,
        )

    def _forward_decode_multistream(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: NPUDSAMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        topk_indices_buffer: torch.Tensor | None = None,
        pd_mixed_flag: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Overlap attention Q/KV-cache work with the complete indexer path."""
        q_cumlens = attn_metadata.query_cumlens.to(torch.int32)
        kv_lens = attn_metadata.seq_lens.to(torch.int32)
        block_table = attn_metadata.block_table
        ki_cache = kv_cache[1] if self.noncontiguous_kv else kv_cache[2]

        def get_mome_args():
            if self.noncontiguous_kv:
                return {}
            return {
                "force_decode": pd_mixed_flag == 1,
                "short_prefill": pd_mixed_flag == 2,
            }

        main_stream = torch.npu.current_stream()
        side_stream = self.side_stream
        input_ready = torch.npu.Event()
        input_ready.record()
        x.record_stream(side_stream)
        cos.record_stream(side_stream)
        sin.record_stream(side_stream)

        with torch.npu.npugraph_ex.scope.limit_core_num(16, 24):
            q_lora = self.q_a_proj(x)[0]
            q_lora = self._maybe_mome_q(q_lora, get_mome_args)
            q_lora = self.q_a_layernorm(q_lora)
            q_lora_ready = torch.npu.Event()
            q_lora_ready.record()
            q_lora.record_stream(side_stream)

            indexer_k = self.indexer.wk(x)[0]
            indexer_k = self.indexer.k_norm(indexer_k).view(-1, 1, self.indexer.head_dim)
            indexer_weights = self.indexer.weights_proj(x)[0]
            if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
                indexer_weights = (
                    indexer_weights * self.indexer.weights_scale * self.indexer.softmax_scale
                )
            indexer_kw_ready = torch.npu.Event()
            indexer_kw_ready.record()
            indexer_k.record_stream(side_stream)
            indexer_weights.record_stream(side_stream)

        kv_ready = torch.npu.Event()

        with torch.npu.stream(side_stream):
            input_ready.wait(side_stream)
            with torch.npu.npugraph_ex.scope.limit_core_num(8, 24):
                kv = self.kv_a_proj_with_mqa(x)[0]
                kv = self._maybe_mome_kv(kv, get_mome_args)
                kv.record_stream(main_stream)
                kv_ready.record()

                q_lora_ready.wait(side_stream)
                indexer_q = self.indexer.wq_b(q_lora)[0].view(
                    -1, self.indexer.n_head, self.indexer.head_dim
                )
                indexer_q = self.indexer._apply_rope(indexer_q, cos, sin)

                indexer_kw_ready.wait(side_stream)
                indexer_k = self.indexer._apply_rope(indexer_k, cos, sin)
                self.indexer._update_cache(indexer_k, attn_metadata, ki_cache)
            with torch.npu.npugraph_ex.scope.limit_core_num(20, 40):
                topk_idx = self.indexer._apply_lightning_indexer(
                    indexer_weights,
                    indexer_q,
                    ki_cache,
                    q_cumlens,
                    kv_lens,
                    block_table,
                )
            side_done = torch.npu.Event()
            side_done.record()

        with torch.npu.npugraph_ex.scope.limit_core_num(12, 24):
            if self.split_q_up_in_multistream:
                q_nope = self.q_b_nope_proj(q_lora)[0].view(
                    -1, self.num_local_heads, self.qk_nope_head_dim
                )
                q_pe = self.q_b_pe_proj(q_lora)[0].view(
                    -1, self.num_local_heads, self.qk_rope_head_dim
                )
                q_nope = self._q_nope_absorb(q_nope)
                q_pe = self._apply_rope(q_pe, cos, sin)
            else:
                q_nope, q_pe = self._q_absorb(q_lora, cos, sin)
            kv_ready.wait(main_stream)
            self._kv_norm_rope_cache(kv, cos, sin, attn_metadata, kv_cache)
        side_done.wait(main_stream)
        topk_idx.record_stream(main_stream)
        next_topk_indices_buffer = topk_idx
        topk_idx = self._apply_sink_offset(topk_idx)

        out = self._apply_attn_absorb(
            q_nope=q_nope,
            q_pe=q_pe,
            q_cumlens=q_cumlens,
            kv_lens=kv_lens,
            topk_idx=topk_idx,
            block_table=block_table,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        return (
            self._decode_attn_epilog(out, get_mome_args),
            next_topk_indices_buffer,
        )


def npu_dsa_forward(
    hs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
    topk_indices_buffer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context = get_forward_context()
    self: NPUDeepseekSparseAttention = forward_context.no_compile_layers[layer_name]
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[f"{self.prefix}.attn"]
    kv_cache = self.attn.kv_cache if attn_metadata else None
    p_slice, d_slice, has_prefill, has_decode = get_batch_desc(
        attn_metadata, self.layer_idx
    )
    prefill = getattr(attn_metadata, "prefill", None)  # None for dummy_run
    decode = getattr(attn_metadata, "decode", None)  # None for dummy_run

    if self.param_sink_number > 0 and not self.noncontiguous_kv:
        assert self.attn.sink_k_pe is not None and self.attn.sink_compressed_kv is not None, (
            "sink_k_pe and sink_compressed_kv have not been prepared"
        )
        if not self.attn.sink_populated and kv_cache is not None:
            self.attn.populate_sink_kv(kv_cache[0], kv_cache[1])

    if has_prefill and has_decode:
        sp_enabled = self.ena_sp
        if sp_enabled:
            topk_indices_buffer = get_tp_group().all_gather(topk_indices_buffer, dim=0)
        with sp_disabled(self, hs) as (x, y, out):
            y[p_slice], topk_indices_buffer[p_slice] = self._forward_prefill(
                x[p_slice],
                cos[p_slice],
                sin[p_slice],
                prefill,
                kv_cache,
                topk_indices_buffer[p_slice],
                pd_mixed_flag=True,
            )
            # short prefill in decode or pure decode
            pd_mixed_flag = (
                2 if attn_metadata.num_decode_tokens > attn_metadata.num_decodes else 1
            )
            y[d_slice], topk_indices_buffer[d_slice] = self._forward_decode(
                x[d_slice],
                cos[d_slice],
                sin[d_slice],
                decode,
                kv_cache,
                topk_indices_buffer[d_slice],
                pd_mixed_flag,
            )
        if sp_enabled:
            topk_indices_buffer = topk_indices_buffer.split(hs.size(0))[get_tp_group().rank_in_group]
    elif has_prefill:
        out = lazy_zero_like(hs)
        if self.ena_cp:
            out[:], topk_indices_buffer = self._forward_prefill_cp(
                hs,
                cos,
                sin,
                prefill,
                kv_cache,
                topk_indices_buffer
            )
        elif self.ena_sp:
            out[:], topk_indices_buffer = self._forward_prefill(
                hs,
                cos,
                sin,
                prefill,
                kv_cache,
                topk_indices_buffer
            )
        else:
            out[p_slice], topk_indices_buffer = self._forward_prefill(
                hs[p_slice],
                cos[p_slice],
                sin[p_slice],
                prefill,
                kv_cache,
                topk_indices_buffer[p_slice],
            )
    else:
        with sp_disabled(self, hs) as (x, y, out):
            y[d_slice], topk_indices_buffer[d_slice] = self._forward_decode(
                x[d_slice],
                cos[d_slice],
                sin[d_slice],
                decode,
                kv_cache,
                topk_indices_buffer[d_slice],
            )
    return out.tensor(), topk_indices_buffer


def npu_dsa_forward_fake(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
    topk_indices_buffer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(topk_indices_buffer)


def npu_dsa_forward_mhc_deferred(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    residual: torch.Tensor,
    layer_name: str,
    mhc_layer_name: str,
    task_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch next-layer MHC work between attention and the DSA epilog."""
    layer = get_forward_context().no_compile_layers[layer_name]
    topk_indices_buffer = torch.zeros(
        (hidden_states.shape[0], 1, layer.index_topk),
        dtype=torch.int32, device=hidden_states.device,
    )
    return call_attention_mhc_deferred(
        lambda x, cos, sin, name: npu_dsa_forward(
            x, cos, sin, name, topk_indices_buffer
        )[0],
        hidden_states,
        cos,
        sin,
        residual,
        layer_name,
        mhc_layer_name,
        task_key,
    )


direct_register_custom_op(
    op_name="npu_dsa_forward_mhc_deferred",
    op_func=npu_dsa_forward_mhc_deferred,
    mutates_args=[],
    fake_impl=attention_mhc_deferred_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_dsa_forward",
    op_func=npu_dsa_forward,
    mutates_args=[],
    fake_impl=npu_dsa_forward_fake,
    dispatch_key="PrivateUse1",
)
