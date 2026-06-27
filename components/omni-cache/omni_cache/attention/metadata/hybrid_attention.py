# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Hybrid attention metadata construction for decode operations.

This module provides the main entry point for building fake attention metadata
in decode operations with hybrid attention mode.
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.attention.backends.utils import (
        AttentionMetadataBuilder,
        CommonAttentionMetadata,
    )


def construct_fake_attn_metadata(
    cache,
    builder: "AttentionMetadataBuilder",
    common_attn_metadata: "CommonAttentionMetadata",
    draft_index: int = -1
) -> "AttentionMetadata":
    """Construct fake attention metadata for decode operations.

    Main entry point for building fake attention metadata in decode operations.
    Routes to appropriate builder based on metadata group ID.

    Args:
        cache: The DecodeOmniCache instance.
        builder: Attention metadata builder with buffers.
        common_attn_metadata: Common attention metadata.
        draft_index: Draft index for MTP (-1 for normal decode, 0 for first draft).

    Returns:
        Attention metadata (NPUMetadata or StridedCompressAttentionMetadata).
    """
    import time
    from vllm.logger import init_logger

    logger = init_logger("vllm.v1.omni")
    t0 = time.time()

    # Handle draft tokens for MTP
    if draft_index >= 0:
        if draft_index == 0:
            return cache.record_attn_metadata_mtp
        from .attention import fake_build_attention
        return fake_build_attention(cache, builder, common_attn_metadata, draft_index)

    # Handle SWA mapping disable
    if int(os.getenv("DISABLE_SWA_MAPPING", "0")):
        while cache.metadata_grp_id < 2:
            cache.metadata_grp_id = (cache.metadata_grp_id + 1) % cache.num_attn_group

    kv_cache_config_tmp = cache.kv_cache_config.kv_cache_groups[
        cache.metadata_grp_id
    ].kv_cache_spec

    if cache.metadata_grp_id < 2:
        # Standard attention for groups 0 and 1
        from .attention import fake_build_attention
        attn_metadata = fake_build_attention(cache, builder, common_attn_metadata)
        if cache.metadata_grp_id == 1:
            cache.record_attn_metadata_mtp = attn_metadata
        t1 = time.time()
        logger.debug(f"<<< DEBUG build attn: {t1-t0=}")
    else:
        # Compressed attention for groups >= 2
        cache.num_extra_blocks = kv_cache_config_tmp.num_extra_blocks
        cache.num_state_blocks = cache.num_extra_blocks // 2

        from .compress import fake_build_compress
        attn_metadata = fake_build_compress(cache, builder, common_attn_metadata)
        t2 = time.time()
        logger.debug(f"<<< DEBUG build attn: {t2-t0=}")

    cache.metadata_grp_id = (cache.metadata_grp_id + 1) % cache.num_attn_group

    return attn_metadata
