# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# omni_cache/attention/backends/stride_compress_ext.py
"""
Extended StridedCompress backend for omni_cache.

This backend extends the base StridedCompressAttentionBackend to add omni_cache-specific
functionality for APC (Attention Prefix Copy) and other optimizations.

IMPORTANT: This extension is only active when ENABLE_OMNI_CACHE env var is "1".
Otherwise, the base StridedCompressAttentionBackend from omni-npu is used.

Classes that need to be overridden:
- StridedCompressBuilder: Override build method to add omni_cache-specific logic

Classes that don't need modification (use base implementation directly):
- StridedCompressAttentionBackend: Core backend class, no modification needed
- StridedCompressAttentionMetadata: Metadata class, no modification needed
- StridedCompressAttentionImpl: Implementation class, no modification needed
"""

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

# Import base implementation
from omni_npu.attention.backends.geneva.stride_compress import (
    StridedCompressAttentionBackend,
    StridedCompressBuilder,
    StridedCompressAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import StridedCompressKVCacheSpec

logger = init_logger(__name__)

# Toggle control: check if omni_cache extension is enabled
ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))


if ENABLE_OMNI_CACHE:
    # ========== Override Builder class ==========
    class StridedCompressBuilderExt(StridedCompressBuilder):
        """
        Extended StridedCompress builder that adds omni_cache functionality.

        Overrides the build() method to add:
        1. Fake attention metadata construction for DecodeOmniCache
        2. Batch token indices initialization for PrefillOmniCache
        """

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> StridedCompressAttentionMetadata:
            """
            Build attention metadata with omni_cache-specific logic.

            This method first checks if omni_cache is enabled and handles
            special cases for DecodeOmniCache before delegating to the base
            implementation. After building the metadata, it initializes
            batch token indices for PrefillOmniCache.
            """
            # Check if omni_cache is enabled and handle DecodeOmniCache case
            from omni_cache.cache import omni_cache
            from omni_cache.cache.decode import DecodeOmniCache

            if isinstance(omni_cache, DecodeOmniCache) and int(
                os.getenv("ENABLE_HOST_MAPPING", "1")
            ):
                return omni_cache._construct_fake_attn_metatata(
                    self, common_attn_metadata
                )

            # Call base implementation to build standard metadata
            metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

            # Initialize batch token indices for PrefillOmniCache
            if os.environ.get("ENABLE_OMNI_CACHE", "0") == "1":
                from omni_cache.cache import omni_cache
                from omni_cache.cache.prefill import PrefillOmniCache

                if isinstance(omni_cache, PrefillOmniCache):
                    omni_cache.init_batch_token_indices_hybrid(metadata.slot_mapping)

            return metadata

    # Export extended class (for entry points registration)
    StridedCompressAttentionBackendExt = StridedCompressAttentionBackend

    logger.info(
        "omni_cache extension enabled, StridedCompressBuilderExt registered"
    )

else:
    # When extension is disabled, export base implementation (ensure module is importable)
    StridedCompressBuilderExt = StridedCompressBuilder
    StridedCompressAttentionBackendExt = StridedCompressAttentionBackend
    logger.info(
        "omni_cache extension disabled, using base StridedCompressBuilder"
    )


__all__ = [
    "StridedCompressAttentionBackendExt",
    "StridedCompressBuilderExt",
]