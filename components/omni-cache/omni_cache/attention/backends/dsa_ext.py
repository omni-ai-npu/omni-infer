# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# omni_cache/attention/backends/dsa_ext.py

"""
Extended DSA backend for omni_cache.

This backend extends the base NPUDSABackend to add omni_cache-specific
functionality for APC (Attention Prefix Copy) and other optimizations.

IMPORTANT: This extension is only active when VLLM_PLUGINS env var contains "omni_cache".
Otherwise, the base NPUDSABackend from omni-npu is used.
"""

import os

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec

# Import base implementation
from omni_npu.attention.backends.dsa import (
    NPUDSABackend,
    NPUDSAMetadataBuilder,
    NPUDSA,
)

from omni_npu.attention.backends.utils import register_attention_backend

from omni_cache.cache.prefill import PrefillOmniCache
from omni_cache.cache.utils.support import (
    get_active_prefill_cache,
    attach_prefix_meta_to_metadata,
    should_compute_prefix_meta
)

logger = init_logger(__name__)

# Switch control: Check if omni_cache extension is enabled
ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

if ENABLE_OMNI_CACHE:
    class NPUDSABackendExt(NPUDSABackend):
        """Extended DSA backend that adds omni_cache functionality."""
        pass

    class NPUDSAMetadataBuilderExt(NPUDSAMetadataBuilder):
        """
        Extended metadata builder that adds omni_cache APC functionality.

        Overrides the build() method to add prefix_meta handling.
        """

        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            # Store layer_names since parent class doesn't
            self.layer_names = layer_names

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata,
            fast_build: bool = False,
        ):
            # Prefill-side: Force prefill only construct prefill metadata
            if get_active_prefill_cache() is not None and getattr(self, 'reorder_batch_threshold', 1) != 0:
                self.reorder_batch_threshold = 0
            metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

            # Add omni_cache specific logic: set prefix_meta
            # IMPORTANT: prefix_meta must be set on metadata.prefill (not parent metadata)
            # because npu_dsa_forward passes attn_metadata.prefill to _forward_prefill
            # which then passes it to _apply_attention decorated by @dsa_attn_decorator
            if metadata is not None and metadata.prefill is not None and self._should_add_prefix_meta(metadata):
                prefix_meta = self._add_prefix_meta(metadata, common_attn_metadata)
                setattr(metadata.prefill, "prefix_meta", prefix_meta)

            return metadata

        def _should_add_prefix_meta(self, metadata) -> bool:
            """Check whether we should compute prefix_meta for APC."""
            return should_compute_prefix_meta(self.vllm_config, metadata)

        def _add_prefix_meta(self, metadata, common_attn_metadata):
            """Compute prefix_meta for DSA and attach to metadata.

            Also triggers H2D for Layer 0 - since post_attn prefetches the NEXT layer,
            Layer 0's data must be loaded during build before any forward pass.
            """
            from vllm.model_executor.models.utils import extract_layer_index
            omni_cache = get_active_prefill_cache()

            attach_prefix_meta_to_metadata(self.vllm_config, metadata, common_attn_metadata)

            prefix_meta = metadata.prefix_meta

            # H2D for Layer 0: Load current layer's KV cache from host to device
            # This is needed because post_attn prefetches NEXT layer, so Layer 0
            # would never get loaded otherwise
            if prefix_meta is not None and omni_cache is not None:
                if self.layer_names:
                    first_layer_name = self.layer_names[0]
                    from omni_cache.cache.transfer_engine.synchronize import (
                        synchronize_h2d_prefill,
                    )
                    synchronize_h2d_prefill(
                        omni_cache, prefix_meta, first_layer_name, load_next_layer=False
                    )

            return metadata.prefix_meta

    @register_attention_backend(NPUDSA)
    class NPUDSABackendExt(NPUDSABackend):
        @staticmethod
        def get_builder_cls():
            return NPUDSAMetadataBuilderExt

    logger.info("omni_cache extension enabled, NPUDSABackendExt registered")

else:
    # When extension is not enabled, export base implementation (ensure module is importable)
    NPUDSABackendExt = NPUDSABackend
    NPUDSAMetadataBuilderExt = NPUDSAMetadataBuilder
    logger.info("omni_cache extension disabled, using base NPUDSABackend")