# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import json
import os
from pathlib import Path
import logging
import yaml
from .prof_wrapper import marker_prof_wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wrapper_dict = {
    "marker": marker_prof_wrapper
}

_PATCHES_ATTR_NAME = "__dynamic_patches__"


# Parse config from namelist, apply profiler monkey patch
def apply_patches(namelist_path: str):
    try:
        namelist_file = Path(__file__).parent / namelist_path

        # Load namelist
        with namelist_file.open('r') as f:
            config = yaml.safe_load(f)

        profiler_type = config.get('type')
        if not profiler_type:
            logger.info(f"<<<type is empty from {namelist_path}")
        if not (profiler_type == 'torchnpu' or
                profiler_type == 'timer' or
                profiler_type == 'viztracer' or
                profiler_type == 'marker'):
            logger.error(f"<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
            raise RuntimeError("<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
        logger.info(f"<<<Applying {profiler_type} profiler patches from {namelist_path}")
        wrapper_method = wrapper_dict[profiler_type]

        # { (module_name, class_name, function_name): [patch_info1, patch_info2, ...] }
        patch_groups = {}

        # Extract target modules and methods
        for target in config.get('targets', []):
            module_name = target.get('module')
            class_name = None
            if ":" in module_name:
                module_name, class_name = module_name.split(":")
            scope_name = target.get('scope')
            scope_class_name = None
            if scope_name and ":" in scope_name:
                scope_name, scope_class_name = scope_name.split(":")
            scope_function = target.get('scope_function', None)
            function_name = target.get('function_name')
            entry_operation = target.get('entry_operation', None)
            exit_operation = target.get('exit_operation', None)
            entry_message = target.get('entry_message', None)
            exit_message = target.get('exit_message', None)
            if module_name:
                patch_info = {
                    'scope_name': scope_name,
                    'scope_class_name': scope_class_name,
                    'scope_function': scope_function,
                    'entry_operation': entry_operation,
                    'exit_operation': exit_operation,
                    'entry_message': entry_message,
                    'exit_message': exit_message
                }
                key = (module_name, class_name, function_name)
                if key not in patch_groups:
                    patch_groups[key] = []
                patch_groups[key].append(patch_info)
            else:
                logger.warning(f"<<<Skipping target with missing 'module': {target}")

        if len(patch_groups.items()) <= 0:
            logger.warning(f"<<<No valid targets found in {namelist_path}")
            return

        for (module_name, class_name, function_name), patch_list in patch_groups.items():
            logger.info(f"<<<Patching {module_name}.{function_name or 'all methods'}")
            try:
                original_module = importlib.import_module(module_name)

                if class_name:
                    try:
                        target_class = getattr(original_module, class_name)
                        try:
                            original_function = getattr(target_class, function_name)
                            wrapped_function = wrapper_method(original_function, {_PATCHES_ATTR_NAME: patch_list})
                            setattr(target_class, function_name, wrapped_function)
                            logger.info(
                                f"<<<<{module_name}.{class_name}.{function_name} is wrapped with {len(patch_list)} patch(es)")
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
                        wrapped_function = wrapper_method(original_function, {_PATCHES_ATTR_NAME: patch_list})
                        setattr(original_module, function_name, wrapped_function)
                        logger.info(
                            f"<<<<{module_name}.{class_name}.{function_name} is wrapped with {len(patch_list)} patch(es)")
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


def monkey_patch_prefill_pop_bootstrapped_logger():
    from functools import wraps
    from typing import AsyncGenerator
    from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
    from omni.adaptors.sglang.patches.profiler_patches.omni_logger import omni_logger_is_prefilling, \
        omni_logger_print_timestamp, omni_logger_enable, omni_logger_print

    original_method = PrefillBootstrapQueue.pop_bootstrapped

    @wraps(original_method)
    def new_method(self, *args, **kwargs) -> AsyncGenerator:
        result = original_method(self, *args, **kwargs)

        if isinstance(result, tuple):
            bootstrapped_reqs, failed_reqs = result
        else:
            bootstrapped_reqs = result

        if omni_logger_enable() and omni_logger_is_prefilling() and len(bootstrapped_reqs) > 0:
            for request in bootstrapped_reqs:
                omni_logger_print_timestamp(request.bootstrap_room, 'p_6 握手完成加到waiting队列')

        return result

    PrefillBootstrapQueue.pop_bootstrapped = new_method
    print("<<< Monkey patch monkey_patch_prefill_pop_bootstrapped_logger is applied")


def monkey_patch_async_wait_one_response_logger():
    from functools import wraps
    from typing import AsyncGenerator
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from omni.adaptors.sglang.patches.profiler_patches.omni_logger import omni_logger_is_prefilling, \
        omni_logger_print_timestamp, omni_logger_enable, omni_logger_print

    original_method = TokenizerManager._wait_one_response

    @wraps(original_method)
    async def new_method(self, *args, **kwargs) -> AsyncGenerator:
        yield_count = 0
        raw_request_id = args[0].bootstrap_room
        async for item in original_method(self, *args, **kwargs):
            if omni_logger_enable():
                if omni_logger_is_prefilling():
                    omni_logger_print_timestamp(raw_request_id, "p_16 client收到输出并入队")
                    omni_logger_print_timestamp(raw_request_id, "p_17 client出队")
                    omni_logger_print_timestamp(raw_request_id, "p_18 api server收到请求准备返回")

            yield_count += 1
            if yield_count == 1 and not omni_logger_is_prefilling():
                omni_logger_print_timestamp(raw_request_id, 'd_19 decoder返回第一个token')
            elif yield_count == 2 and not omni_logger_is_prefilling():
                omni_logger_print_timestamp(raw_request_id, 'd_20 decoder返回第二个token')
            yield item

    TokenizerManager._wait_one_response = new_method
    print("<<< Monkey patch monkey_patch_async_wait_one_response_logger is applied")


def monkey_patch_async_wait_handle_request_logger():
    from functools import wraps
    from typing import AsyncGenerator
    from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
    from omni.adaptors.sglang.patches.profiler_patches.omni_logger import omni_logger_is_prefilling, \
        omni_logger_print_timestamp, omni_logger_enable, omni_logger_print

    original_method = OpenAIServingBase.handle_request

    @wraps(original_method)
    async def new_method(self, *args, **kwargs) -> AsyncGenerator:
        if omni_logger_enable() and len(args) >= 1:
            raw_request_id = args[0].bootstrap_room
            if omni_logger_is_prefilling():
                omni_logger_print_timestamp(raw_request_id, "p_1 prefill api server收到请求")
            else:
                omni_logger_print_timestamp(raw_request_id, "d_1 decode api server收到请求")

        return await original_method(self, *args, **kwargs)

    OpenAIServingBase.handle_request = new_method
    print("<<< Monkey patch monkey_patch_async_wait_handle_request_logger is applied")


def monkey_patch_async_wait_tokenize_one_request_logger():
    from functools import wraps
    from typing import AsyncGenerator
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from omni.adaptors.sglang.patches.profiler_patches.omni_logger import omni_logger_is_prefilling, \
        omni_logger_print_timestamp, omni_logger_enable, omni_logger_print

    original_method = TokenizerManager._tokenize_one_request

    @wraps(original_method)
    async def new_method(self, *args, **kwargs) -> AsyncGenerator:
        if omni_logger_enable() and len(args) >= 1:
            raw_request_id = args[0].bootstrap_room
            if omni_logger_is_prefilling():
                omni_logger_print_timestamp(raw_request_id, f'p_3 engine开始tokennizer')
            else:
                omni_logger_print_timestamp(raw_request_id, f'd_3 engine开始tokennizer')

        result = await original_method(self, *args, **kwargs)

        if omni_logger_enable() and len(args) >= 1:
            raw_request_id = args[0].bootstrap_room
            if omni_logger_is_prefilling():
                omni_logger_print_timestamp(raw_request_id, f'p_3 engine结束tokennizer')
            else:
                omni_logger_print_timestamp(raw_request_id, f'd_3 engine结束tokennizer')

        return result

    TokenizerManager._tokenize_one_request = new_method
    print("<<< Monkey patch monkey_patch_async_wait_tokenize_one_request_logger is applied")


def get_num_completion_tokens(prev_data_str):
    from omni.adaptors.sglang.patches.profiler_patches.omni_logger import omni_logger_enable

    num_completion_tokens = 0
    if prev_data_str:
        try:
            data = json.loads(prev_data_str)
            if isinstance(data, dict) and "usage" in data:
                num_completion_tokens = data["usage"].get("completion_tokens", 0)
        except json.JSONDecodeError:
            if omni_logger_enable():
                print(f"<<< num_completion_tokens parse failed: {prev_data_str}")
    return num_completion_tokens


def monkey_patch_async_generate_chat_stream_logger():
    from functools import wraps
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from omni.adaptors.sglang.patches.profiler_patches.omni_logger import (
        omni_logger_is_prefilling,
        omni_logger_print_timestamp,
        omni_logger_enable
    )

    original_method = OpenAIServingChat._generate_chat_stream

    @wraps(original_method)
    async def new_method(self, *args, **kwargs):
        prev_data_str = None
        async for item in original_method(self, *args, **kwargs):
            item_str = str(item).strip()
            if item_str == 'data: [DONE]':
                break
            if item_str.startswith('data: '):
                data_str = item_str[len('data: '):].strip()
                if data_str:
                    prev_data_str = data_str

            yield item

        num_completion_tokens = get_num_completion_tokens(prev_data_str)

        if omni_logger_enable() and not omni_logger_is_prefilling():
            if len(args) >= 2 and hasattr(args[1], 'bootstrap_room'):
                request = args[1]
                raw_request_id = request.bootstrap_room
                omni_logger_print_timestamp(raw_request_id, 'd_21 api server收到推理结果')
                omni_logger_print_timestamp(raw_request_id, f'CompletionMetric: {num_completion_tokens=}')

    OpenAIServingChat._generate_chat_stream = new_method
    print("<<< Monkey patch monkey_patch_async_generate_chat_stream_logger is applied")


def monkey_patch_async_generate_completions_stream_logger():
    from functools import wraps
    from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
    from omni.adaptors.sglang.patches.profiler_patches.omni_logger import (
        omni_logger_is_prefilling,
        omni_logger_print_timestamp,
        omni_logger_enable
    )

    original_method = OpenAIServingCompletion._generate_completion_stream

    @wraps(original_method)
    async def new_method(self, *args, **kwargs):
        prev_data_str = None
        async for item in original_method(self, *args, **kwargs):
            item_str = str(item).strip()
            if item_str == 'data: [DONE]':
                break
            if item_str.startswith('data: '):
                data_str = item_str[len('data: '):].strip()
                if data_str:
                    prev_data_str = data_str

            yield item

        num_completion_tokens = get_num_completion_tokens(prev_data_str)

        if omni_logger_enable() and not omni_logger_is_prefilling():
            if len(args) >= 2 and hasattr(args[1], 'bootstrap_room'):
                request = args[1]
                raw_request_id = request.bootstrap_room
                omni_logger_print_timestamp(raw_request_id, 'd_21 api server收到推理结果')
                omni_logger_print_timestamp(raw_request_id, f'CompletionMetric: {num_completion_tokens=}')

    OpenAIServingCompletion._generate_completion_stream = new_method
    print("<<< Monkey patch monkey_patch_async_generate_completions_stream_logger is applied")


profiling_namelist = os.getenv("PROFILING_NAMELIST", None)
if profiling_namelist is not None:
    if os.path.isfile(profiling_namelist):
        apply_patches(profiling_namelist)
        monkey_patch_async_wait_one_response_logger()
        monkey_patch_prefill_pop_bootstrapped_logger()
        monkey_patch_async_wait_handle_request_logger()
        monkey_patch_async_wait_tokenize_one_request_logger()
        monkey_patch_async_generate_chat_stream_logger()
        monkey_patch_async_generate_completions_stream_logger()
    else:
        logger.error(f"'{profiling_namelist}' does not exist.")
        raise FileNotFoundError(f"'{profiling_namelist}' does not exist.")
