# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# export OMNI_NPU_VLLM_PATCHES="ProfilerDynamicPatch,RequestStatusPatch,OpenAIServingChatTokenLoggerPatch"

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
import time
from typing import Optional, List, Tuple
import os
import logging
import importlib
import inspect
from pathlib import Path
import yaml

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from functools import wraps
from typing import AsyncGenerator
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.v1.request import Request, RequestStatus
from vllm.v1.engine.core import EngineCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

namelist_path = os.getenv("PROFILING_NAMELIST")

@register_patch("ProfilerDynamicPatch", EngineCore)
class ProfilerDynamicPatch(VLLMPatch):
    _attr_names_to_apply = []

    def __init__(self):
        super().__init__()
        if not namelist_path:
            logger.info("<<< ProfilerDynamicPatch: Trace disabled, PROFILING_NAMELIST environment variable is not set.")
            return
        
        patches_all = os.getenv("OMNI_NPU_VLLM_PATCHES", "").strip()
        enabled_patches = os.getenv("OMNI_NPU_VLLM_PATCHES", "")
        enabled_patch_list = [p.strip() for p in enabled_patches.split(",") if p.strip()]
        
        if patches_all == "ALL":
            logger.info("<<< ProfilerDynamicPatch: Trace enabled, OMNI_NPU_VLLM_PATCHES is set to ALL.")
        elif "ProfilerDynamicPatch" in enabled_patch_list:
            logger.info("<<< ProfilerDynamicPatch: Trace enabled, found in OMNI_NPU_VLLM_PATCHES.")
        else:
            logger.info("<<< ProfilerDynamicPatch: Trace disabled, not found in OMNI_NPU_VLLM_PATCHES and OMNI_NPU_VLLM_PATCHES is not ALL.")
            return
        
        namelist_file = Path(namelist_path)
        if not namelist_file.exists():
            error_msg = f"<<< ProfilerDynamicPatch: Enable failed! Configuration file specified by PROFILING_NAMELIST does not exist: {namelist_path}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"<<< ProfilerDynamicPatch: Enabled successfully. Loading configuration file: {namelist_path}")
        self.apply_patches(namelist_path)

    def apply_patches(self, namelist_path: str):
        from omni_trace.prof_wrapper import (torchnpu_prof_wrapper, timer_prof_wrapper, viztracer_prof_wrapper, marker_prof_wrapper)
        wrapper_dict = {
            "torchnpu": torchnpu_prof_wrapper,
            "timer": timer_prof_wrapper,
            "viztracer": viztracer_prof_wrapper,
            "marker": marker_prof_wrapper
        }
        try:
            with open(namelist_path, 'r') as f:
                config = yaml.safe_load(f)

            profiler_type = config.get('type')
            if not (profiler_type == 'torchnpu' or
                    profiler_type == 'timer' or
                    profiler_type == 'viztracer' or
                    profiler_type == 'marker'):
                logger.error(f"<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
                raise RuntimeError("<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
            logger.info(f"<<<Applying {profiler_type} profiler patches from {namelist_path}")
            wrapper_method = wrapper_dict[profiler_type]

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
                logger.warning(f"<<<No valid targets found in {namelist_path}")
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

profiler_patch_instance = ProfilerDynamicPatch()


@register_patch("RequestStatusPatch", Request)
class RequestStatusPatch(VLLMPatch):
    if not namelist_path:
        logger.info(f"<<< RequestStatusPatch: Trace disabled, PROFILING_NAMELIST environment variable is not set.")
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
                f"Role:{os.getenv('ROLE', 'unknown_role')}_{ip_str}"
            )

    status = property(status, status_set)


_ORIGINAL_CHAT_COMPLETION_STREAM_GENERATOR = OpenAIServingChat.chat_completion_stream_generator
@register_patch("OpenAIServingChatTokenLoggerPatch", OpenAIServingChat)
class OpenAIServingChatTokenLoggerPatch(VLLMPatch):
    if not namelist_path:
        logger.info(f"<<< OpenAIServingChatTokenLoggerPatch: Trace disabled, PROFILING_NAMELIST environment variable is not set.")
        _attr_names_to_apply = []
    else:
        _attr_names_to_apply = ['chat_completion_stream_generator']

    async def chat_completion_stream_generator(self, *args, **kwargs) -> AsyncGenerator:
        from omni_trace.utils import safe_print, ip_str, trace_output_directory
        yield_count = 0
        request_id = args[2]
        async for item in _ORIGINAL_CHAT_COMPLETION_STREAM_GENERATOR(self, *args, **kwargs):
            yield_count += 1
            if yield_count == 1:
                # First chat_completion_stream_generator yield.
                pass
            elif yield_count == 2:
                # Second chat_completion_stream_generator yield.
                safe_print(trace_output_directory, f"<<<Action: First decode output token; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            elif yield_count == 3:
                # Third chat_completion_stream_generator yield.
                safe_print(trace_output_directory, f"<<<Action: Second decode output token; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            elif yield_count == 4:
                # Fourth chat_completion_stream_generator yield.
                safe_print(trace_output_directory, f"<<<Action: Third decode output token; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            if item == "data: [DONE]\n\n":
                safe_print(trace_output_directory, f"<<<Action: Finish decode pickle and start response; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            yield item