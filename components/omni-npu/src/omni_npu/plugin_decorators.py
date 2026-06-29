# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Plugin-based decorators for extending omni-npu functionality."""

from importlib.metadata import entry_points

import logging
logger = logging.getLogger(__name__)

_cached_eps = {}


def create_plugin_decorator(entry_point_group: str, pre_method: str, post_method: str):
    """
    Factory function to create a plugin-based decorator.

    Args:
        entry_point_group: The entry point group name to load plugins from
        pre_method: The method name to call before the decorated function
        post_method: The method name to call after the decorated function

    Returns:
        A decorator function that applies plugin pre/post hooks
    """
    def decorator(func):
        """Decorator that applies plugin pre/post hooks."""
        if entry_point_group not in _cached_eps:
            _cached_eps[entry_point_group] = entry_points(group=entry_point_group)

        eps = _cached_eps[entry_point_group]

        _plugin_classes = []
        for ep in eps:
            try:
                cls = ep.load()
                _plugin_classes.append(cls)
                logger.info(f"Pre-loaded plugin class: {cls.__name__}")
            except Exception as e:
                logger.warning(f"Failed to pre-load plugin {ep.name}: {e}")

        def wrapper(*args, **kwargs):
            plugin_list = [cls() for cls in _plugin_classes]

            # Pre-hooks
            for plugin in plugin_list:
                if hasattr(plugin, pre_method):
                    result = getattr(plugin, pre_method)(*args, **kwargs)
                    if isinstance(result, dict):
                        kwargs.update(result)

            ret = func(*args, **kwargs)

            # Post-hooks
            for plugin in plugin_list:
                if hasattr(plugin, post_method):
                    getattr(plugin, post_method)(*args, result=ret, **kwargs)

            return ret
        return wrapper
    return decorator


def create_conditional_plugin_decorator(entry_point_group: str, pre_method: str, post_method: str):
    """
    Factory function to create a conditional plugin-based decorator.

    This decorator allows plugins to conditionally skip the original function
    by returning True from the pre_method. This is useful when a plugin wants
    to completely replace the original function's behavior under certain conditions.

    Args:
        entry_point_group: The entry point group name to load plugins from
        pre_method: The method name to call before the decorated function.
                    If this method returns True, the original function will be skipped.
        post_method: The method name to call after the decorated function (or after skip)

    Returns:
        A decorator function that applies plugin pre/post hooks with conditional skip

    Example:
        class MyPlugin:
            def pre_init_config(self, *args, **kwargs):
                if should_skip_original():
                    # Do replacement logic here
                    return True  # Skip original function
                return False  # Execute original function

            def post_init_config(self, *args, result=None, **kwargs):
                pass
    """
    def decorator(func):
        """Decorator that applies plugin pre/post hooks with conditional skip."""

        if entry_point_group not in _cached_eps:
            _cached_eps[entry_point_group] = entry_points(group=entry_point_group)
        eps = _cached_eps[entry_point_group]

        def wrapper(*args, **kwargs):
            # Load plugin classes from entry points

            plugin_list = []
            for ep in eps:
                try:
                    plugin_class = ep.load()
                    plugin_list.append(plugin_class())
                except Exception as e:
                    logger.warning(f'Failed to load plugin {ep.name}: {e}')

            # Pre-hooks: loop through all plugins
            skip_original = False
            for plugin in plugin_list:
                if hasattr(plugin, pre_method):
                    try:
                        if getattr(plugin, pre_method)(*args, **kwargs) is True:
                            skip_original = True
                    except Exception as e:
                        logger.exception(f"Plugin {plugin.__class__.__name__}.{pre_method} failed: {e}")
                        raise
                else:
                    logger.warning(f'Plugin {plugin.__class__.__name__} missing method: {pre_method}')

            # Execute original function unless skipped
            ret = None
            if not skip_original:
                ret = func(*args, **kwargs)

            # Post-hooks: loop through all plugins
            for plugin in plugin_list:
                if hasattr(plugin, post_method):
                    try:
                        getattr(plugin, post_method)(*args, **kwargs)
                    except Exception as e:
                        logger.exception(f"Plugin {plugin.__class__.__name__}.{post_method} failed: {e}")
                        raise
                else:
                    logger.warning(f'Plugin {plugin.__class__.__name__} missing method: {post_method}')

            return ret
        return wrapper
    return decorator


# Pre-defined decorators for common use cases
load_model_decorator = create_plugin_decorator(
    entry_point_group="omni.load_model_decorators",
    pre_method="pre_load",
    post_method="post_load"
)

# Use conditional decorator for init_config to allow plugins to skip original function
init_config_decorator = create_conditional_plugin_decorator(
    entry_point_group="omni.init_config_decorators",
    pre_method="pre_init_config",
    post_method="post_init_config"
)

prepare_inputs_decorator = create_plugin_decorator(
    entry_point_group="omni.prepare_inputs_decorators",
    pre_method="pre_prepare_inputs",
    post_method="post_prepare_inputs"
)

reinitialize_input_batch_decorator = create_plugin_decorator(
    entry_point_group="omni.reinitialize_input_batch_decorators",
    pre_method="pre_reinitialize_input_batch",
    post_method="post_reinitialize_input_batch"
)


def attn_decorator(type: str):
    """
    Unified attention decorator factory.

    This decorator provides a unified interface for all attention types,
    allowing plugins to hook into pre_attn and post_attn phases.

    Args:
        type: Attention type, e.g., 'dsa', 'mla', 'mome', 'compress_mqa', 'mqa'

    Returns:
        A decorator that applies plugin pre/post hooks for the specified attention type

    Example:
        @attn_decorator(type='dsa')
        def _apply_attention(...): ...

    The decorator will look for entry points in the group "omni.{type}_attn_decorators"
    and call pre_attn/post_attn methods on the loaded plugin classes.
    """
    entry_point_group = f"omni.{type}_attn_decorators"

    return create_plugin_decorator(
        entry_point_group=entry_point_group,
        pre_method="pre_attn",
        post_method="post_attn"
    )



post_model_forward_decorator = create_plugin_decorator(
    entry_point_group="omni.model_forward_decorators",
    pre_method="pre_model_forward",
    post_method="post_model_forward"
)

update_from_output_decorator = create_plugin_decorator(
    entry_point_group="omni.update_from_output_decorators",
    pre_method="pre_update_from_output",
    post_method="post_update_from_output"
)

model_output_decorator = create_plugin_decorator(
    entry_point_group="omni.model_output_decorators",
    pre_method="pre_model_output",
    post_method="post_model_output"
)