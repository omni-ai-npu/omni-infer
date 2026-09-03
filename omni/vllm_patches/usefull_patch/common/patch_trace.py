# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import inspect
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

import yaml
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request, RequestStatus

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

trace_enabled = bool((envs.OMNI_TRACE_OUTPUT_DIRECTORY or "").strip())
namelist_path = (
    os.path.join(
        os.environ["OMNIINFER_ROOT"],
        "tools/omni_trace/omnilogger_namelist.yml",
    )
    if trace_enabled
    else ""
)


# This class is registered during auto_import_patches(), but is instantiated
# later by manager.apply_patches().  Keep the dynamic namelist wrapping here so
# it runs after all patch classes have been imported and registered.
@register_patch("ProfilerDynamicPatch", EngineCore)
class ProfilerDynamicPatch(VLLMPatch):
    _attr_names_to_apply = []

    def __init__(self):
        super().__init__()
        if not trace_enabled:
            logger.info(
                "<<< ProfilerDynamicPatch: Trace disabled, "
                "OMNI_TRACE_OUTPUT_DIRECTORY is not set."
            )
            return

        namelist_file = Path(namelist_path)
        if not namelist_file.exists():
            error_msg = (
                "<<< ProfilerDynamicPatch: Enable failed! "
                f"Trace configuration does not exist: {namelist_path}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(
            "<<< ProfilerDynamicPatch: Enabled successfully. "
            f"Loading configuration file: {namelist_path}"
        )
        self.apply_patches(namelist_path)

    def apply_patches(self, config_path: str):
        from omni_trace.prof_wrapper import (
            marker_prof_wrapper,
            timer_prof_wrapper,
            torchnpu_prof_wrapper,
            viztracer_prof_wrapper,
        )
        wrapper_dict = {
            "torchnpu": torchnpu_prof_wrapper,
            "timer": timer_prof_wrapper,
            "viztracer": viztracer_prof_wrapper,
            "marker": marker_prof_wrapper
        }
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            profiler_type = config.get('type')
            if not (profiler_type == 'torchnpu' or
                    profiler_type == 'timer' or
                    profiler_type == 'viztracer' or
                    profiler_type == 'marker'):
                logger.error(f"<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
                raise RuntimeError("<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
            logger.info(f"<<<Applying {profiler_type} profiler patches from {config_path}")
            wrapper_method = wrapper_dict.get(profiler_type)
            if wrapper_method is None:
                raise KeyError(
                    f"Unknown profiler_type: {profiler_type}. "
                    f"Available types: {list(wrapper_dict.keys())}"
                )

            base_params = config.get("base_params", {})

            # Extract target modules and methods
            targets: List[Tuple[str, Optional[str], Optional[str], tuple, tuple]] = []
            for target in config.get('targets', []):
                module_name = target.get('module')
                class_name = None
                if ":" in module_name:
                    module_name, class_name = module_name.split(":")
                function_name = target.get('function_name')
                entry_operation = target.get('entry_operation', None)
                exit_operation = target.get('exit_operation', None)
                entry_message = target.get('entry_message', None)
                exit_message = target.get('exit_message', None)
                if module_name:
                    targets.append(
                        (
                            module_name,
                            class_name,
                            function_name,
                            (entry_operation, exit_operation),
                            (entry_message, exit_message)
                        )
                    )
                else:
                    logger.warning(f"<<<Skipping target with missing 'module': {target}")

            if not targets:
                logger.warning(f"<<<No valid targets found in {config_path}")
                return

            for module_name, class_name, function_name, \
                    (entry_operation, exit_operation), \
                    (entry_message, exit_message) in targets:
                logger.info(f"<<<Patching {module_name}.{function_name or 'all methods'}")
                try:
                    original_module = importlib.import_module(module_name)

                    base_params['entry_operation'] = entry_operation
                    base_params['exit_operation'] = exit_operation
                    base_params['entry_message'] = entry_message
                    base_params['exit_message'] = exit_message
                    if class_name:
                        try:
                            target_class = getattr(original_module, class_name)
                            try:
                                original_attr = inspect.getattr_static(target_class, function_name)
                                if isinstance(original_attr, staticmethod):
                                    original_function = original_attr.__func__
                                    wrapped_function = wrapper_method(original_function, base_params)
                                    setattr(target_class, function_name, staticmethod(wrapped_function))
                                elif isinstance(original_attr, classmethod):
                                    original_function = original_attr.__func__
                                    wrapped_function = wrapper_method(original_function, base_params)
                                    setattr(target_class, function_name, classmethod(wrapped_function))
                                else:
                                    original_function = getattr(target_class, function_name)
                                    wrapped_function = wrapper_method(original_function, base_params)
                                    setattr(target_class, function_name, wrapped_function)
                                logger.info(f"<<<<{module_name}.{class_name}.{function_name} is wrapped")
                            except AttributeError:
                                logger.warning(
                                    f"<<<Function '{function_name}' not found in class '{class_name}' "
                                    f"of module '{module_name}'"
                                )
                                continue
                        except AttributeError:
                            logger.warning(f"<<<Class '{class_name}' not found in module '{module_name}'")
                            continue
                    else:
                        try:
                            original_function = getattr(original_module, function_name)
                            wrapped_function = wrapper_method(original_function, base_params)
                            setattr(original_module, function_name, wrapped_function)
                            logger.info(f"<<<<{module_name}.{function_name} is wrapped")
                        except AttributeError:
                            logger.warning(f"<<<Function '{function_name}' not found in module '{module_name}'")
                            continue
                except ImportError as e:
                    logger.warning(f"<<<Failed to import module '{module_name}': {str(e)}")
                    continue
                except Exception as e:
                    logger.warning(
                        f"<<<Unexpected error while wrapping {module_name}.{class_name or ''}."
                        f"{function_name}: {str(e)}"
                    )
                    continue

        except (FileNotFoundError, ImportError, AttributeError, RuntimeError, yaml.YAMLError) as e:
            logger.error(f"<<<Failed to apply model patches: {e}")
            raise


@register_patch("RequestStatusPatch", Request)
class RequestStatusPatch(VLLMPatch):
    if not trace_enabled:
        logger.info("<<< RequestStatusPatch: Trace disabled, OMNI_TRACE_OUTPUT_DIRECTORY is not set.")
        _attr_names_to_apply = []
    else:
        _attr_names_to_apply = ['status']

    def status(self):
        return self._status

    def status_set(self, value):
        from omni_trace.utils import safe_print, ip_str, trace_output_directory
        if not hasattr(self, "waiting_pull_len"):
            self.waiting_pull_len = 0
        if not hasattr(self, "_status"):
            self._status = None
        self._status = value
        self.waiting_pull_len += 1
        if value == RequestStatus.WAITING_FOR_REMOTE_KVS:
            safe_print(
                trace_output_directory,
                f"<<<Action: Add need pulling sequence; "
                f"Timestamp:{time.time()}; "
                f"RequestID:{self.request_id}; "
                f"Role:{envs.OMNI_PD_ROLE or 'unknown_role'}_{ip_str}"
            )

    status = property(status, status_set)


_DECODE_YIELD_ACTIONS = {
    2: "First decode output token",
    3: "Second decode output token",
    4: "Third decode output token",
}


def _trace_decode_stream_line(action: str, request_id: str, ip_str: str) -> str:
    role = envs.OMNI_PD_ROLE or "unknown_role"
    return (
        f"<<<Action: {action}; Timestamp:{time.time()}; "
        f"RequestID:{request_id}; Role:{role}_{ip_str}"
    )


async def _iter_traced_stream(original, serving, args, kwargs, request_id):
    from omni_trace.utils import safe_print, ip_str, trace_output_directory
    yield_count = 0
    async for item in original(serving, *args, **kwargs):
        yield_count += 1
        action = _DECODE_YIELD_ACTIONS.get(yield_count)
        if action is not None:
            safe_print(
                trace_output_directory,
                _trace_decode_stream_line(action, request_id, ip_str),
            )
        if item == "data: [DONE]\n\n":
            safe_print(
                trace_output_directory,
                _trace_decode_stream_line(
                    "Finish decode pickle and start response",
                    request_id,
                    ip_str,
                ),
            )
        yield item


if trace_enabled:
    from omni_npu.vllm_patches.usefull_patch.common.patch_serving_apc import (
        OpenAIServingChatStreamAPCPatch,
        OpenAIServingCompletionStreamAPCPatch,
    )
    _ORIGINAL_CHAT_COMPLETION_STREAM_GENERATOR = (
        OpenAIServingChatStreamAPCPatch.chat_completion_stream_generator
    )
    _ORIGINAL_COMPLETION_STREAM_GENERATOR = (
        OpenAIServingCompletionStreamAPCPatch.completion_stream_generator
    )

    @register_patch("ExpertIdServingChatStream", OpenAIServingChat)
    class OpenAIServingChatTokenLoggerPatch(VLLMPatch):
        # Relay patch: wrap APC stream patch instead of the original vLLM stream.
        _attr_names_to_apply = ['chat_completion_stream_generator']

        async def chat_completion_stream_generator(
            self, *args, **kwargs
        ) -> AsyncGenerator:
            async for item in _iter_traced_stream(
                _ORIGINAL_CHAT_COMPLETION_STREAM_GENERATOR,
                self, args, kwargs, args[2],
            ):
                yield item


    @register_patch("ExpertIdServingCompletionStream", OpenAIServingCompletion)
    class OpenAIServingCompletionTokenLoggerPatch(VLLMPatch):
        # Relay patch: wrap APC stream patch instead of the original vLLM stream.
        _attr_names_to_apply = ['completion_stream_generator']

        async def completion_stream_generator(
            self, *args, **kwargs
        ) -> AsyncGenerator:
            async for item in _iter_traced_stream(
                _ORIGINAL_COMPLETION_STREAM_GENERATOR,
                self, args, kwargs, args[3],
            ):
                yield item
