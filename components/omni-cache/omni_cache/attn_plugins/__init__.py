# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Attention Plugins for Omni-Cache

This package provides a plugin system for hooking into attention layers
with pre/post_attn callbacks for KV cache offload operations.

Usage:
    from omni_cache.attn_plugins import compressed_mqa_attn_decorator

    @compressed_mqa_attn_decorator
    def _apply_attention(self, ...):
        ...
"""

import logging
import os

from .base import (
    AttentionPlugin,
    AttentionPluginRegistry,
    create_attn_decorator,
    mqa_attn_decorator,
    compressed_mqa_attn_decorator,
    mla_attn_decorator,
    dsa_attn_decorator,
    mome_attn_decorator,
)
from omni_cache.cache.utils.support import sync_h2d_event

from .implementations import (
    CompressMQAAttnPlugin,
    MLAAttnPlugin,
    GeneralMQAAttnPlugin,
    DSAAttnPlugin,
    MOMEAttnPlugin,
)

logger = logging.getLogger(__name__)

# Available plugins mapping
_AVAILABLE_PLUGINS = {
    "compressed_mqa": CompressMQAAttnPlugin,
    "mqa": GeneralMQAAttnPlugin,
    "mla": MLAAttnPlugin,
    "dsa": DSAAttnPlugin,
    "mome": MOMEAttnPlugin,
}


def register_omni_cache_plugins() -> None:
    """Register Omni-Cache attention plugins based on environment variable.

    Registration is controlled by the environment variable `OMNI_CACHE_ATTN_PLUGINS`:
    - If not set or empty: register ALL available plugins
    - If set: register only the specified plugins (comma-separated)

    Examples:
        # Register all plugins (default)
        export OMNI_CACHE_ATTN_PLUGINS=""  # or not set at all

        # Register only compressed_mqa plugin
        export OMNI_CACHE_ATTN_PLUGINS="compressed_mqa"

        # Register multiple specific plugins
        export OMNI_CACHE_ATTN_PLUGINS="compressed_mqa,mla"

    Available plugin types:
        - compressed_mqa: Compressed multi-query attention (DeepSeek-V3 style)
        - mqa: Standard multi-query attention
        - mla: Multi-head latent attention (DeepSeek-MLA style)
        - dsa: Dynamic Sparse Attention (DeepSeek with sparse indexing)
    """
    # Get plugin filter from environment
    plugin_filter = os.environ.get("OMNI_CACHE_ATTN_PLUGINS", "").strip()

    if plugin_filter:
        # Parse comma-separated list of plugin types
        requested_plugins = [
            p.strip() for p in plugin_filter.split(",")
            if p.strip()  # Skip empty strings
        ]

        # Validate and filter plugins
        valid_plugins = []
        invalid_plugins = []
        for plugin_type in requested_plugins:
            if plugin_type in _AVAILABLE_PLUGINS:
                valid_plugins.append(plugin_type)
            else:
                invalid_plugins.append(plugin_type)

        if invalid_plugins:
            logger.warning(
                f"attn_plugin: unknown plugin types: {invalid_plugins}. "
                f"Available types: {list(_AVAILABLE_PLUGINS.keys())}"
            )

        # Register only the requested valid plugins
        if valid_plugins:
            for plugin_type in valid_plugins:
                plugin_class = _AVAILABLE_PLUGINS[plugin_type]
                AttentionPluginRegistry.register(plugin_type, plugin_class())
            logger.info(
                f"attn_plugin: registered selected plugins: "
                f"{AttentionPluginRegistry.list_plugins()}"
            )
        else:
            logger.warning("attn_plugin: no valid plugins to register")

    else:
        # No filter specified, register all available plugins
        for plugin_type, plugin_class in _AVAILABLE_PLUGINS.items():
            AttentionPluginRegistry.register(plugin_type, plugin_class())
        logger.info(
            f"attn_plugin: registered all available plugins: "
            f"{AttentionPluginRegistry.list_plugins()}"
        )


__all__ = [
    # From base
    "AttentionPlugin",
    "AttentionPluginRegistry",
    "create_attn_decorator",
    "mqa_attn_decorator",
    "compressed_mqa_attn_decorator",
    "mla_attn_decorator",
    "dsa_attn_decorator",
    "mome_attn_decorator",
    "sync_h2d_event",
    # From implementations
    "CompressMQAAttnPlugin",
    "MLAAttnPlugin",
    "GeneralMQAAttnPlugin",
    "DSAAttnPlugin",
    "MOMEAttnPlugin",
    # Registration function
    "register_omni_cache_plugins",
]