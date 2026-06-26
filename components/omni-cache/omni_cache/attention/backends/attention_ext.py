# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# omni_cache/attention/backends/attention_ext.py

"""
Extended NPUAttentionBackend for omni_cache.

This backend extends the base NPUAttentionBackend to add omni_cache-specific
functionality for metadata construction in decode and drafting modes.

IMPORTANT: This extension is only active when VLLM_PLUGINS env var contains "omni_cache".
Otherwise, the base NPUAttentionBackend from omni-npu is used.

Classes that need to be rewritten:
- NPUAttentionMetadataBuilder: Rewrite build and build_for_drafting methods
- NPUMetadata: May need to add new fields (if omni_cache requires)

Classes that don't need to be rewritten (use base implementation directly):
- NPUAttentionBackend: The Backend class itself doesn't need modification
- NPUAttentionBackendImpl: The implementation class doesn't need modification
"""

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata

# Import base implementation
from omni_npu.attention.backends.attention import (
    NPUAttentionMetadataBuilder,
    NPUMetadata,
)
from omni_npu.attention.backends.utils import register_attention_backend

logger = init_logger(__name__)

# Switch control: Check if omni_cache extension is enabled
ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

VLLM_NPU_ATTN = "VLLM_NPU_ATTN"

if ENABLE_OMNI_CACHE:
    # ========== Override MetadataBuilder class ==========
    class NPUAttentionMetadataBuilderExt(NPUAttentionMetadataBuilder):
        """
        Extended metadata builder that adds omni_cache functionality.

        Overrides:
        - build(): Adds omni_cache metadata construction for decode mode
        - build_for_drafting(): Adds omni_cache metadata construction for drafting
        """

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> NPUMetadata:
            """
            Build attention metadata with omni_cache support.

            When omni_cache is enabled and in decode mode with appropriate
            settings, returns the fake attention metadata from omni_cache.
            Otherwise, falls back to base implementation.
            """
            # omni_cache special handling: fake metadata in decode mode
            from omni_cache.cache import omni_cache
            from omni_cache.cache.decode import DecodeOmniCache

            if (isinstance(omni_cache, DecodeOmniCache) and
                (not int(os.getenv("DISABLE_SWA_MAPPING", "0"))) and
                int(os.getenv("ENABLE_HOST_MAPPING", "1"))):
                return omni_cache._construct_fake_attn_metatata(
                    self, common_attn_metadata
                )

            # Call base implementation
            metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

            # omni_cache initialization in prefill mode
            from omni_cache.cache import omni_cache
            from omni_cache.cache.prefill import PrefillOmniCache

            if isinstance(omni_cache, PrefillOmniCache):
                omni_cache.init_batch_token_indices_hybrid(metadata.slot_mapping)

            return metadata

        def build_for_drafting(
            self,
            common_attn_metadata: CommonAttentionMetadata,
            draft_index: int,
        ):
            """
            Build attention metadata for draft model with omni_cache support.

            When omni_cache is enabled in decode mode, returns the fake
            metadata from omni_cache for drafting.
            Otherwise, falls back to base implementation.
            """
            # omni_cache special handling
            from omni_cache.cache import omni_cache
            from omni_cache.cache.decode import DecodeOmniCache

            if (isinstance(omni_cache, DecodeOmniCache) and
                (not int(os.getenv("DISABLE_SWA_MAPPING", "0"))) and
                int(os.getenv("ENABLE_HOST_MAPPING", "1"))):

                import time
                st = time.time()
                fake_meta = omni_cache._construct_fake_attn_metatata(
                    self, common_attn_metadata, draft_index
                )
                duration = time.time() - st
                logger.debug(f"<<< DEBUG metadata in attention_ext.py cost: {duration}")
                return fake_meta

            # Call base implementation
            return super().build_for_drafting(common_attn_metadata, draft_index)

    logger.info("omni_cache extension enabled, NPUAttentionMetadataBuilderExt registered")

else:
    # When extension is disabled, export base implementation (ensure module is importable)
    NPUAttentionMetadataBuilderExt = NPUAttentionMetadataBuilder
    logger.info("omni_cache extension disabled, using base NPUAttentionMetadataBuilder")