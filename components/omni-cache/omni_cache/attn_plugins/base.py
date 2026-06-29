# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Attention Plugin System for Omni-Cache (Simplified Version)

This module provides a lightweight plugin system for hooking into attention layers.
The design focuses on simplicity and extensibility for specific attention types.

Key Components:
- AttentionPlugin: Base interface for plugins
- AttentionPluginRegistry: Simple registry for plugins
- create_attn_decorator(): Factory to create decorators
"""

import functools
import logging
from typing import Dict, Optional, Any, Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Base Plugin Interface
# ============================================================================


class AttentionPlugin:
    """Base class for attention plugins.

    Subclasses should implement pre_attn and post_attn methods.

    The interface uses flexible *args, **kwargs to be compatible with
    different external calling conventions. Subclasses should extract
    necessary parameters from args and kwargs as needed.
    """

    def pre_attn(self, *args, **kwargs) -> None:
        """Called before attention computation.

        This method receives all arguments from the wrapped function.
        Subclasses should extract needed parameters:

        - instance: Typically args[0] or kwargs.get('instance')
        - layer_name: From kwargs.get('layer_name') or instance.prefix
        - Other args: Available in kwargs dict

        Args:
            *args: Positional arguments from the wrapped function
            **kwargs: Keyword arguments from the wrapped function
        """
        pass

    def post_attn(self, *args, **kwargs) -> None:
        """Called after attention computation.

        This method receives all arguments from the wrapped function,
        including the result which is typically passed as 'result' in kwargs.

        Args:
            *args: Positional arguments from the wrapped function
            **kwargs: Keyword arguments, may include 'result' key
        """
        pass


# ============================================================================
# Plugin Registry
# ============================================================================


class AttentionPluginRegistry:
    """Simple registry for attention plugins.

    Maps attention types (str) to plugin instances.
    """

    _plugins: Dict[str, AttentionPlugin] = {}

    @classmethod
    def register(cls, attn_type: str, plugin: AttentionPlugin) -> None:
        """Register a plugin for a specific attention type.

        Args:
            attn_type: Identifier for this attention type (e.g., "mqa")
            plugin: Plugin instance to register
        """
        cls._plugins[attn_type] = plugin
        logger.info(f"attn_plugin: registered plugin '{attn_type}' -> {plugin.__class__.__name__}")

    @classmethod
    def get(cls, attn_type: str) -> Optional[AttentionPlugin]:
        """Get a registered plugin."""
        return cls._plugins.get(attn_type)

    @classmethod
    def has(cls, attn_type: str) -> bool:
        """Check if a plugin is registered."""
        return attn_type in cls._plugins

    @classmethod
    def list_plugins(cls) -> Dict[str, str]:
        """Return a dict mapping attn_type to plugin class name."""
        return {k: v.__class__.__name__ for k, v in cls._plugins.items()}


# ============================================================================
# Decorator Factory
# ============================================================================


def create_attn_decorator(attn_type: str) -> Callable:
    """Create a decorator that wraps attention with pre/post hooks.

    This is a simplified version focused on specific use cases.

    Args:
        attn_type: The attention type identifier (e.g., "mqa")

    Returns:
        A decorator function

    Example:
        @create_attn_decorator("mqa")
        def _apply_attention(self, ...):
            ...
    """

    def decorator(orig_func: Callable) -> Callable:
        @functools.wraps(orig_func)
        def wrapper(*args, **kwargs):
            # Get instance and ensure layer_name is in kwargs for plugins
            instance = args[0] if args else None
            if "layer_name" not in kwargs and instance is not None:
                kwargs["layer_name"] = getattr(instance, "prefix", "default_layer_name")

            # Get plugin
            plugin = AttentionPluginRegistry.get(attn_type)
            if plugin is not None:
                try:
                    plugin.pre_attn(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"pre_attn hook failed for {attn_type}: {e}")

            # Call original function
            result = orig_func(*args, **kwargs)

            # Post hook - pass result in kwargs for plugins
            if plugin is not None:
                try:
                    plugin.post_attn(*args, **kwargs, result=result)
                except Exception as e:
                    logger.warning(f"post_attn hook failed for {attn_type}: {e}")

            return result

        return wrapper

    return decorator


# ============================================================================
# Pre-defined Decorators for Common Attention Types
# ============================================================================

# Use these decorators directly for specific attention types

# MQA (Multi-Query Attention) decorator
mqa_attn_decorator = create_attn_decorator("mqa")

# Compressed MQA decorator
compressed_mqa_attn_decorator = create_attn_decorator("compressed_mqa")

# MLA (Multi-Head Latent Attention) decorator
mla_attn_decorator = create_attn_decorator("mla")

# DSA (Dynamic Sparse Attention) decorator
dsa_attn_decorator = create_attn_decorator("dsa")

# MOME decorator
mome_attn_decorator = create_attn_decorator("mome")


def _drain_d2h_stage(oc, s):
    """Wait for all D2H futures + d2h_stream events on stage `s`, then clear."""
    fut_stages = getattr(oc, "copy_futures_stages", None)
    if fut_stages and 0 <= s < len(fut_stages):
        bucket = fut_stages[s]
        items = list(bucket.values()) if isinstance(bucket, dict) else list(bucket)
        for fut in items:
            if fut is None:
                continue
            try:
                fut.result()
            except Exception as e:
                logger.warning("d2h stage=%d future: %s", s, e)
        if isinstance(bucket, dict):
            bucket.clear()
        elif isinstance(bucket, list):
            bucket.clear()
    evt_stages = getattr(oc, "d2h_event_stages", None)
    if evt_stages and 0 <= s < len(evt_stages):
        _sync_and_clear(oc, evt_stages, s, "d2h")


def _drain_h2d_stage(oc, s):
    """Wait for all h2d_stream events on stage `s`, then clear."""
    h2d_stages = getattr(oc, "h2d_event_stages", None)
    if h2d_stages and 0 <= s < len(h2d_stages):
        _sync_and_clear(oc, h2d_stages, s, "h2d")


def _sync_and_clear(oc, stages, s, kind):
    bucket = stages[s]
    if isinstance(bucket, dict):
        evts = list(bucket.values())
    elif isinstance(bucket, list):
        evts = list(bucket)
    else:
        evts = [bucket] if bucket is not None else []
    for evt in evts:
        if evt is None:
            continue
        try:
            evt.synchronize()
        except Exception as e:
            logger.warning("drain stage=%d %s event: %s", s, kind, e)
    if isinstance(bucket, dict):
        bucket.clear()
    elif isinstance(bucket, list):
        bucket.clear()


def _moe_post_sync():
    """Perform the layer-boundary KV-pipeline sync.

    Called from inside moe_attn_decorator after every MoE forward (i.e.
    once per decoder block). MoE is not an attention layer, so this hook
    intentionally does NOT live in AttentionPluginRegistry; it is a
    plain module-level function so the decorator can call it without
    any registry lookup.

    What it waits for:
      - copy_futures[stage_record] (single wait that, transitively,
        protects HBM[stage_record] AND the host pinned buffer for that
        stage: the worker thread runs d2h_event.synchronize() before
        reading pinned and finishes its scatter into the host pool
        before resolving the future). When num_stages_layer_copy == 1
        this resolves immediately because the legacy synchronize_d2h
        already waited at submit time.
      - h2d_event.synchronize() for APC prefix prefetch. When APC is
        disabled the event has no pending work so this is effectively
        free.
    """
    import os

    if not int(os.environ.get("ENABLE_OMNI_CACHE", "0")):
        return
    try:
        import omni_cache.cache as _cache_mod
    except Exception:
        return
    oc = getattr(_cache_mod, "omni_cache", None)
    if oc is None:
        return
    if type(oc).__name__ != "PrefillOmniCache":
        return
    # Wait for the previous D2H+scatter to finish before the next layer
    # overwrites the same HBM/host stage. This must run regardless of
    # num_stages_layer_copy: even with N=1 there is still a race window
    # between layer N's d2h_stream copy and layer N+1's main_stream KV
    # write to the same HBM[0]. Wait on copy_futures[stage_record] when
    # available (multi-stage), else fall back to cache.copy_future
    # (single-stage legacy handle).
    s = int(getattr(oc, "stage_record", 0) or 0)

    h2d_event = getattr(oc, "h2d_event", None)
    if h2d_event is not None:
        try:
            h2d_event.synchronize()
        except Exception:
            pass
    n_stages = max(1, int(getattr(oc, "num_stages_layer_copy", 1) or 1))
    cur = int(getattr(oc, "stage_record", 0) or 0)
    oc.stage_record = (cur + 1) % n_stages
    # Next block's D2H reads device_cache[oc.stage_record] → drain D2H only
    _drain_d2h_stage(oc, oc.stage_record)
    # Next block's H2D writes device_cache[(oc.stage_record + 1) % n_stages] → drain H2D only
    _drain_h2d_stage(oc, (oc.stage_record + 1) % n_stages)


def _create_moe_decorator():
    """MoE-boundary decorator. Wraps FusedMoE.forward and calls the sync
    after the wrapped call returns. Does not use AttentionPluginRegistry
    (MoE is not an attention layer) and does not inject `layer_name=`
    into the wrapped function (FusedMoE.forward has a strict signature).
    """
    import functools as _ft

    def decorator(orig_func):
        @_ft.wraps(orig_func)
        def wrapper(*args, **kwargs):
            result = orig_func(*args, **kwargs)
            try:
                _moe_post_sync()
            except Exception as e:
                logger.warning(f"moe post-sync failed: {e}")
            return result

        return wrapper

    return decorator


# MoE FFN decorator. Applied to FusedMoE.forward; fires the
# layer-boundary KV pipeline sync (_moe_post_sync) after each forward.
moe_attn_decorator = _create_moe_decorator()


__all__ = [
    # Base classes
    "AttentionPlugin",
    "AttentionPluginRegistry",
    # Factory function
    "create_attn_decorator",
    # Pre-defined decorators
    "mqa_attn_decorator",
    "compressed_mqa_attn_decorator",
    "mla_attn_decorator",
    "dsa_attn_decorator",
    "mome_attn_decorator",
    "moe_attn_decorator",
]
