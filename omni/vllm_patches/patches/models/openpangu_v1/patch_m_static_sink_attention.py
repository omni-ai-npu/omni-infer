# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import functools
from typing import cast

import torch
from torch import nn
import torch_npu

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
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
from vllm.model_executor.layers.attention import static_sink_attention
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.selector import get_attn_backend
from vllm.model_executor.layers.attention.static_sink_attention import (
    StaticSinkAttention,
    create_static_sink_attention_backend,
)
from vllm.model_executor.custom_op import CustomOp
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
                self.num_sink_blocks = self.sink_len // vllm_config.cache_config.block_size
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
            def build_for_drafting(self,
                common_attn_metadata: CommonAttentionMetadata,
                draft_index: int,):

                return super().build(0, common_attn_metadata, True)
            def build(
                self,
                common_prefix_len: int,
                common_attn_metadata: CommonAttentionMetadata,
                fast_build: bool = False,
            ) -> AttentionMetadata:
                if common_attn_metadata.block_table_tensor[0, 0] != 1:
                    max_num_blocks = cdiv(common_attn_metadata.max_seq_len, self.block_size)
                    num_reqs = common_attn_metadata.num_reqs
                    self.block_table_with_sink[
                        :num_reqs, self.num_sink_blocks : self.num_sink_blocks + max_num_blocks
                    ] = common_attn_metadata.block_table_tensor[:, :max_num_blocks]
                    common_attn_metadata.block_table_tensor = self.block_table_with_sink[:num_reqs]
                    common_attn_metadata.block_table_tensor.masked_fill_(
                        common_attn_metadata.block_table_tensor == -1 , 0
                    )

                    zero_mask = common_attn_metadata.seq_lens.eq(0)
                    common_attn_metadata.seq_lens.add_(self.sink_len)
                    common_attn_metadata.seq_lens.masked_fill_(zero_mask, 0)
                    common_attn_metadata.max_seq_len += self.sink_len

                return super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_backend = subclass_attention_backend(
            name_prefix=prefix,
            attention_backend_cls=underlying_attn_backend,
            builder_cls=StaticSinkAttentionBuilder,
        )
        return attn_backend
    # patch end
    
    static_sink_attention.create_static_sink_attention_backend = create_static_sink_attention_backend


@register_patch("StaticSinkAttentionPatch", StaticSinkAttention)
class StaticSinkAttentionPatch(VLLMPatch):
    """
    Attention with static sink  tokens
    """
    _attr_names_to_apply = ['__init__', 'forward_native', 'populate_sink_kv']

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        sink_len: int,
        attn_backend: type[AttentionBackend] | None = None,
        cache_config: CacheConfig | None = None,
        **kwargs,
    ):
        dtype = torch.get_default_dtype()
        CustomOp.__init__(self)

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        if attn_backend is not None:
            underlying_attn_backend = attn_backend
        else:
            underlying_attn_backend = get_attn_backend(
                head_size, dtype, kv_cache_dtype, block_size
            )
        attn_backend = static_sink_attention.create_static_sink_attention_backend(
            underlying_attn_backend,  # type: ignore[arg-type]
            sink_len=sink_len,
        )
        Attention.__init__(
            self=self,
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            **kwargs,
        )

        self.sink_len = sink_len
        self.block_size = block_size
        self.sink_populated = False
        self.sink_key = None
        self.sink_value = None

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        assert self.sink_key is not None and self.sink_value is not None, (
            "sink_key and sink_value have not been prepared"
        )
        if not self.sink_populated:
            forward_context: ForwardContext = get_forward_context()
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            if self_kv_cache is not None and len(self_kv_cache) > 0:
                self.populate_sink_kv(self_kv_cache[0], self_kv_cache[1])

        return Attention.forward(self, query, key, value, output_shape)

    def populate_sink_kv(
        self,
        self_k_cache: torch.Tensor,
        self_v_cache: torch.Tensor
    ) -> None:
        sink_kv_slot_mapping = torch.arange(
            self.block_size,
            self.sink_len + self.block_size,
            device=current_platform.current_device(),
            dtype=torch.long,
        ).view(-1, 1)

        torch_npu.npu_scatter_nd_update_(
            self_k_cache.view(-1, self_k_cache.shape[-1]),
            sink_kv_slot_mapping,
            self.sink_key
        )
        torch_npu.npu_scatter_nd_update_(
            self_v_cache.view(-1, self_v_cache.shape[-1]),
            sink_kv_slot_mapping,
            self.sink_value
        )
        self.sink_populated = True

    StaticSinkAttention.__init__ = __init__
    StaticSinkAttention.forward_native = forward_native
    StaticSinkAttention.populate_sink_kv = populate_sink_kv