# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Extended MoME backend for omni_cache.

Decode: rewrite ``cache_indices`` so each batch row addresses the HBM lane
that omni_cache's decode H2D filled for that request. For MoME/Mamba
``req_offset = 1``, so row i uses slot ``idx_reqs[i] + 1`` (slot 0 of each
lane is reserved — see ``cache/decode/hbm_buffer_utils.py``).

Prefill + APC: compute ``prefix_meta`` from the MoME block table
(``cache_indices``) and ``num_computed_tokens``, then trigger H2D to
restore MoME conv state from the dedicated APC prefix buffers.

Prefill D2H: on each step the number of already-computed tokens is
recorded so the D2H transfer engine only offloads KV cache blocks for
tokens that were newly processed in the current step, skipping blocks
already persisted during previous steps.
"""

import os

import torch

from vllm.logger import init_logger

from omni_npu.attention.backends.mome import (
    NPUPanguMomeBackend,
    NPUMomeAttentionMetadataBuilder,
    NPUPanguMome,
)
from omni_npu.attention.backends.utils import register_attention_backend
from omni_cache.cache.utils.support import (
    get_active_prefill_cache,
    get_active_decode_cache,
    should_compute_prefix_meta,
    attach_prefix_meta_to_metadata,
)

logger = init_logger(__name__)

ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))
ENABLE_HOST_MAPPING = int(os.getenv("ENABLE_HOST_MAPPING", "1"))
USE_OMNI_INPUT_BATCH = int(os.getenv("USE_OMNI_INPUT_BATCH", "0"))


if ENABLE_OMNI_CACHE and not USE_OMNI_INPUT_BATCH:
    from omni_cache.attention.metadata import resolve_batch_idx_reqs

    class NPUMomeAttentionMetadataBuilderExt(NPUMomeAttentionMetadataBuilder):
        """MoME metadata builder with omni_cache extensions.

        Decode: rewrites cache_indices to address HBM lanes.
        Prefill+APC: computes prefix_meta and triggers MoME H2D.
        Prefill D2H: records num_computed_tokens so the transfer engine
        only offloads newly-added blocks per step.
        """

        def __init__(
            self,
            kv_cache_spec,
            layer_names: list[str],
            vllm_config,
            device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self.layer_names = layer_names

        def build(
            self,
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=None,
            num_decode_draft_tokens_cpu=None,
            fast_build: bool = False,
            num_prompt_tokens: torch.Tensor | None = None,
        ):
            # Decode with host-mapping: rewrite block table for HBM lane access
            if not hasattr(self, "_oc"):
                import omni_cache.cache as _ocache
                from omni_cache.cache.decode import DecodeOmniCache

                self._oc = _ocache.omni_cache
                self.is_decode = isinstance(self._oc, DecodeOmniCache)
            if self.is_decode and ENABLE_HOST_MAPPING:
                self._rewrite_decode_block_table(common_attn_metadata.block_table_tensor)

            # Prefill-side: Force prefill only construct prefill metadata
            if get_active_prefill_cache() is not None and getattr(self, "reorder_batch_threshold", 1) != 0:
                self.reorder_batch_threshold = 0

            metadata = super().build(
                common_prefix_len=common_prefix_len,
                common_attn_metadata=common_attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                fast_build=fast_build,
                num_prompt_tokens=num_prompt_tokens,
            )

            # Prefill + APC: compute prefix_meta and trigger H2D
            if self._should_add_prefix_meta(metadata):
                self._add_prefix_meta(metadata, common_attn_metadata)

            # Record num_computed_tokens so the D2H transfer engine can
            # skip blocks that were already offloaded in previous steps.
            self._oc.num_computed_tokens = metadata.num_computed_tokens

            return metadata

        def update_block_table(self, metadata, blk_table, slot_mapping):
            self._rewrite_decode_block_table(blk_table)
            return super().update_block_table(metadata, blk_table, slot_mapping)

        def _rewrite_decode_block_table(self, block_table_tensor):
            # APC path keeps real block_table-based 2D cache_indices.
            if self.vllm_config.cache_config.enable_prefix_caching:
                return
            cache = get_active_decode_cache()
            if cache is None:
                return

            num_decodes = block_table_tensor.shape[0]
            if num_decodes <= 0:
                return

            ci = block_table_tensor[:num_decodes, 0]
            host_buf = self._get_lane_plus1_host_buffer(cache, ci)
            idx_reqs = resolve_batch_idx_reqs(cache, num_decodes)
            self._fill_lane_plus1_host_buffer(host_buf, cache, num_decodes, idx_reqs)
            ci.copy_(host_buf[:num_decodes], non_blocking=True)

        def _get_lane_plus1_host_buffer(self, cache, ci):
            host_buf = getattr(self, "_lane_plus1_pin", None)
            if host_buf is None or host_buf.numel() < ci.shape[0]:
                host_buf = torch.empty(
                    max(ci.shape[0], cache.num_max_batch_pool + 1),
                    dtype=ci.dtype,
                    pin_memory=True,
                )
                self._lane_plus1_pin = host_buf
            return host_buf

        def _fill_lane_plus1_host_buffer(self, host_buf, cache, num_decodes, idx_reqs):
            pad_lane = max(0, cache.num_max_batch_pool - 1)
            host_buf.fill_(pad_lane)
            if idx_reqs is not None:
                for i, idx in enumerate(idx_reqs[:num_decodes]):
                    if idx >= 0:
                        host_buf[i] = idx + 1
            else:
                # dummy_run: use sequential lanes so capture sees distinct slots.
                for i in range(num_decodes):
                    host_buf[i] = 0

        def _should_add_prefix_meta(self, metadata) -> bool:
            """Check whether we should compute prefix_meta for MoME APC."""
            return should_compute_prefix_meta(self.vllm_config, metadata)

        def _add_prefix_meta(self, metadata, common_attn_metadata):
            """Compute prefix_meta for MoME and attach to metadata.

            Also triggers H2D for Layer 0 - since post_attn prefetches the NEXT layer,
            Layer 0's data must be loaded during build before any forward pass.
            """
            omni_cache = get_active_prefill_cache()
            from vllm.model_executor.models.utils import extract_layer_index

            attach_prefix_meta_to_metadata(self.vllm_config, metadata, common_attn_metadata)

            prefix_meta = getattr(metadata, "prefix_meta", None)

            # H2D for Layer 0: Load current layer's KV cache from host to device
            # This is needed because post_attn prefetches NEXT layer, so Layer 0
            # would never get loaded otherwise
            if prefix_meta is not None and omni_cache is not None:
                if self.layer_names:
                    first_layer_name = self.layer_names[0]
                    from omni_cache.cache.transfer_engine.synchronize import (
                        synchronize_h2d_prefill,
                    )

                    synchronize_h2d_prefill(omni_cache, prefix_meta, first_layer_name, load_next_layer=False)


    @register_attention_backend(NPUPanguMome)
    class NPUPanguMomeBackendExt(NPUPanguMomeBackend):
        @staticmethod
        def get_builder_cls():
            return NPUMomeAttentionMetadataBuilderExt

    logger.info("omni_cache extension enabled, NPUPanguMomeBackendExt registered")

else:
    NPUPanguMomeBackendExt = NPUPanguMomeBackend
    NPUMomeAttentionMetadataBuilderExt = NPUMomeAttentionMetadataBuilder
    logger.info("omni_cache extension disabled, using base NPUPanguMomeBackend")
