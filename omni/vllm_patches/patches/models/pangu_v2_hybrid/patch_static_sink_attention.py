# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import cast

import torch
import torch_npu

from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import static_sink_attention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    subclass_attention_backend,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
)

from omni_npu.attention.backends.dsa import NPUDSABackend
from omni_npu.attention.backends.mla import NPUMLABackend
from omni_npu.model_config.config_loader.loader import model_extra_config

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


@register_patch("create_static_sink_attention_backendPatch", static_sink_attention)
class create_static_sink_attention_backendPatch(VLLMPatch):
    _attr_names_to_apply = ['create_static_sink_attention_backend']

    # patch start
    @functools.lru_cache
    def create_static_sink_attention_backend(
        underlying_attn_backend: type[AttentionBackend],
        sink_len: int = 0,
    ) -> type[AttentionBackend]:
        prefix = "StaticSink_"
        underlying_builder = underlying_attn_backend.get_builder_cls()

        class StaticSinkAttentionBuilder(underlying_builder):  # type: ignore
            def __init__(
                self,
                kv_cache_spec: AttentionSpec,
                layer_names: list[str],
                vllm_config: VllmConfig,
                device: torch.device,
            ):
                super().__init__(kv_cache_spec, layer_names, vllm_config, device)
                model_config = vllm_config.model_config
                scheduler_config = vllm_config.scheduler_config
                self.sink_len = sink_len
                self.block_size = vllm_config.cache_config.block_size
                self.num_sink_blocks = (
                    self.sink_len // vllm_config.cache_config.block_size
                )
                self.max_num_blocks = cdiv(
                    model_config.max_model_len, vllm_config.cache_config.block_size
                )
                self.scheduler_config = scheduler_config
                self.device = device
                self.block_table_with_sink = torch.zeros(
                    (
                        scheduler_config.max_num_seqs,
                        self.max_num_blocks + self.num_sink_blocks,
                    ),
                    device=device,
                    dtype=torch.int32,
                )
                self.block_table_with_sink[:, : self.num_sink_blocks] = torch.arange(
                    1,
                    self.num_sink_blocks + 1,
                    device=device,
                    dtype=torch.int32,
                )

            def reinit_block_table_with_sink(self):
                self.block_table_with_sink[:, :] = torch.zeros(
                    (
                        self.scheduler_config.max_num_seqs,
                        self.max_num_blocks + self.num_sink_blocks,
                    ),
                    device=self.device,
                    dtype=torch.int32,
                )
                self.block_table_with_sink[:, : self.num_sink_blocks] = torch.arange(
                    1,
                    self.num_sink_blocks + 1,
                    device=self.device,
                    dtype=torch.int32,
                )

            def build(
                self,
                common_prefix_len: int,
                common_attn_metadata: CommonAttentionMetadata,
                fast_build: bool = False,
            ) -> AttentionMetadata:
                if common_attn_metadata.block_table_tensor[0, 0] != 1:
                    max_num_blocks = cdiv(
                        common_attn_metadata.max_seq_len,
                        self.block_size,
                    )
                    num_reqs = common_attn_metadata.num_reqs
                    self.block_table_with_sink[
                        :num_reqs,
                        self.num_sink_blocks:self.num_sink_blocks + max_num_blocks,
                    ] = common_attn_metadata.block_table_tensor[:, :max_num_blocks]
                    common_attn_metadata.block_table_tensor = (
                        self.block_table_with_sink[:num_reqs]
                    )

                    zero_mask = common_attn_metadata.seq_lens.eq(0)
                    common_attn_metadata.seq_lens.add_(self.sink_len)
                    common_attn_metadata.seq_lens.masked_fill_(zero_mask, 0)
                    common_attn_metadata.max_seq_len += self.sink_len

                return super().build(
                    common_prefix_len,
                    common_attn_metadata,
                    fast_build,
                )

        attn_backend = subclass_attention_backend(
            name_prefix=prefix,
            attention_backend_cls=underlying_attn_backend,
            builder_cls=StaticSinkAttentionBuilder,
        )
        return attn_backend
    # patch end

    static_sink_attention.create_static_sink_attention_backend = (
        create_static_sink_attention_backend
    )


@register_patch("StaticSinkAttentionPatch", static_sink_attention)
class StaticSinkAttentionPatch(VLLMPatch):
    _attr_names_to_apply = ['StaticSinkMLAAttention']

    class StaticSinkMLAAttention(MLAAttention):
        """MLAAttention with static sink tokens for NPU."""

        def __init__(
            self,
            num_heads: int,
            scale: float,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int | None,
            kv_lora_rank: int,
            kv_b_proj: ColumnParallelLinear,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            use_sparse: bool = False,
            indexer: object | None = None,
            sink_len: int | None = None,
            sliding_window: int | None = None,
            **extra_impl_args,
        ):
            super().__init__(
                num_heads=num_heads,
                scale=scale,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                kv_b_proj=kv_b_proj,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                use_sparse=use_sparse,
                indexer=indexer,
                **extra_impl_args,
            )
            self.sink_len = sink_len
            self.sliding_window = sliding_window

            if not model_extra_config.operator_opt_config.use_noncontiguous_kv:
                self.attn_backend = (
                    static_sink_attention.create_static_sink_attention_backend(
                        self.attn_backend,
                        sink_len=self.sink_len,
                    )
                )
            else:
                if use_sparse:
                    self.attn_backend = NPUDSABackend()
                else:
                    self.attn_backend = NPUMLABackend()

            impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
            self.impl = impl_cls(
                num_heads=self.num_heads,
                head_size=self.head_size,
                scale=self.scale,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=self.sliding_window,
                kv_cache_dtype=self.kv_cache_dtype,
                logits_soft_cap=None,
                attn_type=AttentionType.DECODER,
                kv_sharing_target_layer_name=None,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                kv_b_proj=kv_b_proj,
                indexer=indexer,
                **extra_impl_args,
            )
            self.block_size = (
                cache_config.block_size if cache_config is not None else 16
            )
            self.sink_populated = False
            self.sink_k_pe = None
            self.sink_compressed_kv = None
            self.use_sparse = use_sparse
            self.indexer = indexer

        def update_sink_kv(self, sink_k_pe, sink_compressed_kv) -> None:
            self.sink_k_pe = sink_k_pe
            self.sink_compressed_kv = sink_compressed_kv
            self.impl.update_sink_kv(sink_k_pe, sink_compressed_kv)

        def forward(self, *args, **kwargs) -> torch.Tensor:
            assert self.sink_k_pe is not None and self.sink_compressed_kv is not None, (
                "sink_k_pe and sink_compressed_kv have not been prepared"
            )
            if not model_extra_config.operator_opt_config.use_noncontiguous_kv:
                if not self.sink_populated:
                    forward_context: ForwardContext = get_forward_context()
                    self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                    if self_kv_cache is not None and len(self_kv_cache) > 0:
                        self.populate_sink_kv(self_kv_cache[0], self_kv_cache[1])
            return super().forward(*args, **kwargs)

        def maybe_populate_sink_kv_after_wakeup(self, self_k_cache, self_v_cache):
            if not model_extra_config.operator_opt_config.use_noncontiguous_kv:
                self.populate_sink_kv(self_k_cache, self_v_cache)

        def populate_sink_kv(
            self,
            k_nope_cache: torch.Tensor,
            k_pe_cache: torch.Tensor,
        ):
            sink_kv_slot_mapping = torch.arange(
                self.block_size,
                self.sink_len + self.block_size,
                device=current_platform.current_device(),
                dtype=torch.long,
            ).view(-1, 1)

            torch_npu.npu_scatter_nd_update_(
                k_nope_cache.reshape(-1, k_nope_cache.shape[-1]),
                sink_kv_slot_mapping,
                self.sink_compressed_kv,
            )
            torch_npu.npu_scatter_nd_update_(
                k_pe_cache.reshape(-1, k_pe_cache.shape[-1]),
                sink_kv_slot_mapping,
                self.sink_k_pe,
            )
            self.sink_populated = True

        def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
            from vllm.v1.kv_cache_interface import (
                DSAAttentionSpec,
                ShareKVSlidingWindowSpec,
                SinkMLAAttentionSpec,
            )

            kv_cache_dtype = kv_cache_dtype_str_to_dtype(
                self.kv_cache_dtype, vllm_config.model_config
            )
            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                if self.sliding_window:
                    block_size = vllm_config.cache_config.block_size
                    if self.use_sparse and vllm_config.quant_config is not None:
                        block_size = block_size // 2
                    return ShareKVSlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=self.head_size,
                        dtype=model_extra_config.dtype,
                        sliding_window=2048,
                        page_size_padded=(
                            vllm_config.cache_config.mamba_page_size_padded
                        ),
                    )

                if self.use_sparse:
                    return DSAAttentionSpec(
                        block_size=vllm_config.cache_config.block_size,
                        num_kv_heads=1,
                        head_size=self.head_size + self.indexer.head_dim,
                        dtype=kv_cache_dtype,
                        cache_dtype_str=vllm_config.cache_config.cache_dtype,
                    )
            return SinkMLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=kv_cache_dtype,
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
                sink_len=self.sink_len,
                page_size_padded=vllm_config.cache_config.mamba_page_size_padded,
            )

    static_sink_attention.StaticSinkMLAAttention = StaticSinkMLAAttention
