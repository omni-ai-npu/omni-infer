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
from vllm.model_executor.layers.attention.static_sink_attention import (
    StaticSinkAttention,
    get_attn_backend,
    create_static_sink_attention_backend,
)
from vllm.model_executor.custom_op import CustomOp
from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


@register_patch("create_static_sink_attention_backendPatch", static_sink_attention)
class create_static_sink_attention_backendPatch(VLLMPatch):
    _attr_names_to_apply = ['create_static_sink_attention_backend']

    # patch start
    @staticmethod
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
                from omni_npu.model_config.config_loader.loader import (
                    model_extra_config,
                )

                model_config = vllm_config.model_config
                scheduler_config = vllm_config.scheduler_config
                self.sink_len = sink_len
                self.block_size = vllm_config.cache_config.block_size
                self.num_sink_slots = self.sink_len // vllm_config.cache_config.block_size
                self.max_blocks_sink = cdiv(
                    model_config.max_model_len, vllm_config.cache_config.block_size
                )
                self.scheduler_config = scheduler_config
                self.device = device
                # Sinks are weights, not activations. Under noncontiguous KV the
                # layers hand key_sink/value_sink/key_rope_sink straight to the
                # kernels - what npu_pangu has always done - so no sink KV is
                # written to the cache (populate_sink_kv is skipped) and the
                # top-k indices are not shifted (_apply_sink_offset is skipped).
                # Prefixing the block table and inflating seq_lens there would
                # bill every request for a sink block nobody wrote.
                self.sinks_in_band = (
                    not model_extra_config.operator_opt_config.use_noncontiguous_kv
                )
                if not self.sinks_in_band:
                    return
                self.block_table_sink_buf = torch.zeros(
                    (
                        scheduler_config.max_num_seqs,
                        self.max_blocks_sink + self.num_sink_slots,
                    ),
                    device=device,
                    dtype=torch.int32,
                )
                self.block_table_sink_buf[:, : self.num_sink_slots] = torch.arange(
                    1,
                    self.num_sink_slots + 1,
                    device=device,
                    dtype=torch.int32,
                )

            def reinit_block_table_with_sink(self):
                if not self.sinks_in_band:
                    return  # no sink prefix to rebuild
                self.block_table_sink_buf[:, :] = torch.zeros(
                    (
                        self.scheduler_config.max_num_seqs,
                        self.max_blocks_sink + self.num_sink_slots,
                    ),
                    device=self.device,
                    dtype=torch.int32,
                )
                self.block_table_sink_buf[:, : self.num_sink_slots] = torch.arange(
                    1,
                    self.num_sink_slots + 1,
                    device=self.device,
                    dtype=torch.int32,
                )

            def build(
                self,
                common_prefix_len: int,
                common_attn_metadata: CommonAttentionMetadata,
                fast_build: bool = False,
            ) -> AttentionMetadata:
                # Track prefixing explicitly. The old test - "first block
                # id != 1" - conflates "not yet prefixed" with "this request
                # was allocated block 1", and block 1 IS the first sink block,
                # so whichever request got it was silently left un-prefixed.
                if self.sinks_in_band and not getattr(
                    common_attn_metadata, "_omni_sink_prefixed", False
                ):
                    max_blocks_with_sink = cdiv(common_attn_metadata.max_seq_len, self.block_size)
                    num_reqs_with_sink = common_attn_metadata.num_reqs
                    self.block_table_sink_buf[
                        :num_reqs_with_sink, self.num_sink_slots:self.num_sink_slots + max_blocks_with_sink
                    ] = common_attn_metadata.block_table_tensor[:, :max_blocks_with_sink]
                    common_attn_metadata.block_table_tensor = self.block_table_sink_buf[:num_reqs_with_sink]
                    common_attn_metadata.block_table_tensor.masked_fill_(
                        common_attn_metadata.block_table_tensor == -1, 0
                    )

                    # Inflate into this builder's own buffer, not the
                    # shared tensor. Each KV-cache group gets a shallow copy of
                    # the metadata but they share one seq_lens, so an in-place
                    # add ran once per group (19 -> 147 -> 275 -> 403) and
                    # attention covered blocks the request never wrote.
                    src_seq_lens = common_attn_metadata.seq_lens
                    zero_seq_mask = src_seq_lens.eq(0)
                    buf = getattr(self, "_seq_lens_with_sink", None)
                    if (
                        buf is None
                        or buf.size(0) < src_seq_lens.size(0)
                        or buf.dtype != src_seq_lens.dtype
                    ):
                        buf = torch.zeros(
                            max(src_seq_lens.size(0), self.scheduler_config.max_num_seqs),
                            dtype=src_seq_lens.dtype,
                            device=src_seq_lens.device,
                        )
                        self._seq_lens_with_sink = buf
                    seq_lens_sink_buf = buf[: src_seq_lens.size(0)]
                    torch.add(src_seq_lens, self.sink_len, out=seq_lens_sink_buf)
                    seq_lens_sink_buf.masked_fill_(zero_seq_mask, 0)
                    common_attn_metadata.seq_lens = seq_lens_sink_buf
                    common_attn_metadata.max_seq_len += self.sink_len
                    common_attn_metadata._omni_sink_prefixed = True

                return super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_backend = subclass_attention_backend(
            name_prefix=prefix,
            attention_backend_cls=underlying_attn_backend,
            builder_cls=StaticSinkAttentionBuilder,
        )
        return attn_backend
    # patch end

    static_sink_attention.create_static_sink_attention_backend = (
        create_static_sink_attention_backend.__func__
    )


@register_patch("StaticSinkAttentionPatch", static_sink_attention)
class StaticSinkAttentionPatch(VLLMPatch):
    _attr_names_to_apply = [
        'PanguSinkAttentionBase',
        'StaticSinkMLAAttention',
    ]

    class PanguSinkAttentionBase:

        def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
            from omni_npu.v1.layers.attention.weight_utils import load_sharded_param_weight

            load_sharded_param_weight(self, param, loaded_weight)

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
            self.attn_backend = static_sink_attention.create_static_sink_attention_backend(
                self.attn_backend,
                sink_len=self.sink_len,
            )
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
            self.block_size = cache_config.block_size if cache_config is not None else 16
            self.sink_populated = False
            self.sink_k_pe = None
            self.sink_compressed_kv = None

        def update_sink_kv(self, sink_k_pe, sink_compressed_kv) -> None:
            self.sink_k_pe = sink_k_pe
            self.sink_compressed_kv = sink_compressed_kv
            self.impl.update_sink_kv(sink_k_pe, sink_compressed_kv)

        def forward(self, *args, **kwargs) -> torch.Tensor:
            if self.sink_k_pe is None or self.sink_compressed_kv is None:
                raise RuntimeError(
                    "sink_k_pe and sink_compressed_kv have not been prepared"
                )
            if not self.sink_populated:
                forward_context: ForwardContext = get_forward_context()
                self_kv_cache = self.kv_cache
                if self_kv_cache is not None and len(self_kv_cache) > 0:
                    self.populate_sink_kv(self_kv_cache[0], self_kv_cache[1])
            return super().forward(*args, **kwargs)

        def populate_sink_kv(self, k_nope_cache: torch.Tensor, k_pe_cache: torch.Tensor):
            from omni_npu.model_config.config_loader.loader import model_extra_config

            if model_extra_config.operator_opt_config.use_noncontiguous_kv:
                # Sinks go to the kernels as tensors there, and blocks 1..N are
                # ordinary blocks a request may already own - writing into them
                # would overwrite live KV. The layers skip this call; the
                # sleep-mode wake-up path in NPUModelRunner does not.
                self.sink_populated = True
                return
            sink_kv_slot_mapping = torch.arange(
                self.block_size,
                self.sink_len + self.block_size,
                device=current_platform.current_device(),
                dtype=torch.long,
            ).view(-1, 1)

            torch_npu.npu_scatter_nd_update_(
                k_nope_cache.view(-1, k_nope_cache.shape[-1]),
                sink_kv_slot_mapping,
                self.sink_compressed_kv,
            )
            torch_npu.npu_scatter_nd_update_(
                k_pe_cache.view(-1, k_pe_cache.shape[-1]),
                sink_kv_slot_mapping,
                self.sink_k_pe,
            )
            self.sink_populated = True

        def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
            from vllm.v1.kv_cache_interface import (
                DSAAttentionSpec,
                SinkMLAAttentionSpec,
            )

            from omni_npu.model_config.config_loader.loader import model_extra_config

            kv_dtype = kv_cache_dtype_str_to_dtype(
                self.kv_cache_dtype, vllm_config.model_config
            )
            if (
                model_extra_config.operator_opt_config.use_noncontiguous_kv
                and self.use_sparse
                and not self.sliding_window
            ):
                # The sparse (DSA) cache also stores the top-k indexer ki
                # (index_head_dim) per token; see
                # NPUDSABackend._reshape_kv_cache_noncontiguous (shapes
                # ((576,), (128,))). Without it the raw buffer is small.
                # skip_topk / indexer_types=="shared" layers still use this
                # DSA layout but never construct Indexer, so indexer is None.
                indexer_head_dim = getattr(self.indexer, "head_dim", None)
                if indexer_head_dim is None:
                    indexer_head_dim = getattr(
                        vllm_config.model_config.hf_config,
                        "index_head_dim",
                        0,
                    )
                return DSAAttentionSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=self.head_size + indexer_head_dim,
                    dtype=kv_dtype,
                    cache_dtype_str=vllm_config.cache_config.cache_dtype,
                )
            return SinkMLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=kv_dtype,
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
                sink_len=self.sink_len,
            )

    static_sink_attention.PanguSinkAttentionBase = PanguSinkAttentionBase
    static_sink_attention.StaticSinkMLAAttention = StaticSinkMLAAttention


@register_patch("StaticSinkAttentionClassPatch", StaticSinkAttention)
class StaticSinkAttentionClassPatch(VLLMPatch):
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
            sink_len=sink_len,
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
        if self.sink_key is None or self.sink_value is None:
            raise RuntimeError(
                "sink_key and sink_value have not been prepared"
            )
        if not self.sink_populated:
            forward_context: ForwardContext = get_forward_context()
            self_kv_cache = self.kv_cache
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
