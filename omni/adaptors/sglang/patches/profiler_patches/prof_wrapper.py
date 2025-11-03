# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def _execute_operation(operation_str: Optional[str], param_dict: Dict[str, Any]) -> None:
    """execute_operation"""
    if operation_str:
        try:
            exec(operation_str, param_dict)
        except Exception as e:
            logging.error(f"Error executing operation: {e}")


def _get_caller_info(stack: List[inspect.FrameInfo]) -> Tuple[str, str, str]:
    """Obtain the caller's module name, class name, and function name"""
    caller_module: str = ""
    caller_class: str = ""
    caller_function: str = ""
    try:
        if len(stack) > 1:
            caller_frame: inspect.FrameType = stack[1].frame
            # module name
            caller_module = caller_frame.f_globals.get('__name__', '')
            # class name
            caller_self: Optional[Union[object, Type]] = caller_frame.f_locals.get('self') or caller_frame.f_locals.get(
                'cls')

            if caller_self:
                if inspect.isclass(caller_self):
                    caller_class = caller_self.__name__
                else:
                    caller_class = caller_self.__class__.__name__

            # function name
            caller_function = stack[1].function
    except Exception as e:
        logging.error(f"Error getting caller info: {e}")

    return caller_module, caller_class, caller_function


def _should_wrap(
        scope_name: Optional[str],
        scope_class_name: Optional[str],
        scope_function: Optional[str],
        caller_module: str,
        caller_class: str,
        caller_function: str
) -> bool:
    """Determine whether packaging logic needs to be executed """
    if not all([scope_name, scope_class_name, scope_function]):
        return True

    return (caller_module == scope_name and
            caller_class == scope_class_name and
            caller_function == scope_function)


def _set_param_dict(args, first_arg, is_cls, kwargs):
    """Fill in cls or self """
    param_dict: Dict[str, Any] = {
        "args": args,
        "kwargs": kwargs
    }
    if is_cls:
        param_dict["cls"] = first_arg
    else:
        param_dict["self"] = first_arg
    return param_dict


def _is_cls_method(original_method):
    """Determine whether it is a class function. """
    is_cls = False
    if inspect.ismethod(original_method):
        is_cls = isinstance(original_method.__self__, type)
    return is_cls


# Define method wrapper type
SyncWrapper = Callable[..., Any]
AsyncWrapper = Callable[..., Any]


def _sync_func_multi(original_method: Callable, patch_list: List[Dict[str, Any]]) -> SyncWrapper:
    """Create a sync wrapper that executes multiple patches based on caller context"""

    @functools.wraps(original_method)
    def wrapper(first_arg: Any, *args: Any, **kwargs: Any) -> Any:
        stack: List[inspect.FrameInfo] = inspect.stack()
        caller_module, caller_class, caller_function = _get_caller_info(stack)

        matched_patches = []
        for patch_info in patch_list:
            scope_name = patch_info.get('scope_name')
            scope_class_name = patch_info.get('scope_class_name')
            scope_function = patch_info.get('scope_function')

            if _should_wrap(scope_name, scope_class_name, scope_function,
                            caller_module, caller_class, caller_function):
                matched_patches.append(patch_info)
                logger.debug(f"<<< Matched patch for caller {caller_module}.{caller_class}.{caller_function}")

        if not matched_patches:
            # d_12 in while true frantically prints
            # logger.info(
            #     f"<<<INFO: No matching patches for {original_method.__qualname__} "
            #     f"from caller {caller_module}.{caller_class}.{caller_function}")
            return original_method(first_arg, *args, **kwargs)

        is_cls = _is_cls_method(original_method)
        param_dict = _set_param_dict(args, first_arg, is_cls, kwargs)

        for patch_info in matched_patches:
            entry_op = patch_info.get('entry_operation')
            entry_msg = patch_info.get('entry_message')
            if entry_msg:
                logger.info(f"<<<{entry_msg}")
            _execute_operation(entry_op, param_dict)

        result = {}
        try:
            result = original_method(first_arg, *args, **kwargs)
        finally:
            param_dict["result"] = result

        for patch_info in reversed(matched_patches):
            exit_op = patch_info.get('exit_operation')
            exit_msg = patch_info.get('exit_message')
            if exit_msg:
                logger.info(f"<<<{exit_msg}")
            _execute_operation(exit_op, param_dict)

        return result

    return cast(SyncWrapper, wrapper)


def _async_func_multi(original_method: Callable, patch_list: List[Dict[str, Any]]) -> AsyncWrapper:
    """Create an async wrapper that executes multiple patches based on caller context"""

    @functools.wraps(original_method)
    async def async_wrapper(first_arg: Any, *args: Any, **kwargs: Any) -> Any:
        stack: List[inspect.FrameInfo] = inspect.stack()
        caller_module, caller_class, caller_function = _get_caller_info(stack)

        matched_patches = []
        for patch_info in patch_list:
            scope_name = patch_info.get('scope_name')
            scope_class_name = patch_info.get('scope_class_name')
            scope_function = patch_info.get('scope_function')
            if _should_wrap(scope_name, scope_class_name, scope_function,
                            caller_module, caller_class, caller_function):
                matched_patches.append(patch_info)
                logger.debug(f"<<< Matched patch for caller {caller_module}.{caller_class}.{caller_function}")

        if not matched_patches:
            # d_12 in while true frantically prints
            # logger.info(
            #     f"<<<INFO: No matching patches for {original_method.__qualname__} "
            #     f"from caller {caller_module}.{caller_class}.{caller_function}")
            return await original_method(first_arg, *args, **kwargs)

        is_cls = _is_cls_method(original_method)
        param_dict = _set_param_dict(args, first_arg, is_cls, kwargs)

        for patch_info in matched_patches:
            entry_op = patch_info.get('entry_operation')
            entry_msg = patch_info.get('entry_message')
            if entry_msg:
                logger.info(f"<<<{entry_msg}")
            _execute_operation(entry_op, param_dict)

        result = {}
        try:
            result = await original_method(first_arg, *args, **kwargs)
        finally:
            param_dict["result"] = result

        for patch_info in reversed(matched_patches):
            exit_op = patch_info.get('exit_operation')
            exit_msg = patch_info.get('exit_message')
            if exit_msg:
                logger.info(f"<<<{exit_msg}")
            _execute_operation(exit_op, param_dict)

        return result

    return cast(AsyncWrapper, async_wrapper)


_PATCHES_ATTR_NAME = "__dynamic_patches__"


def marker_prof_wrapper(
        original_method: Callable,
        params: Dict[str, Any]
) -> Union[SyncWrapper, AsyncWrapper]:
    """marker_prof_wrapper"""
    patch_list: List[Dict[str, Any]] = params.get(_PATCHES_ATTR_NAME, [])

    if not patch_list:
        logger.info(f"<<<INFO: No patches provided for {original_method.__qualname__}, returning original.")
        return original_method

    is_async: bool = inspect.iscoroutinefunction(original_method)

    if is_async:
        logger.info(f"<<<INFO: {original_method.__qualname__} is async function, use async wrapper")
        return _async_func_multi(original_method, patch_list)
    else:
        logger.info(f"<<<INFO: {original_method.__qualname__} is sync function, use sync wrapper")
        return _sync_func_multi(original_method, patch_list)
