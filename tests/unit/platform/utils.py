# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import inspect
import types

from itertools import count
from unittest.mock import MagicMock
from typing import Any

import torch
from vllm.config import (VllmConfig, SchedulerConfig, ModelConfig, CacheConfig,
                         KVTransferConfig, DeviceConfig, ParallelConfig)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput, KVConnectorOutput
from vllm.v1.request import Request
from vllm import SamplingParams


def mock_with_default_value(cls):
    mock_obj = MagicMock()
    for k, v in cls.__dict__.items():
        if k.startswith("__"):
            continue
        if not callable(v):
            setattr(mock_obj, k, v)
        elif isinstance(v, staticmethod):
            setattr(mock_obj, k, v.__func__)
        elif isinstance(v, classmethod):
            setattr(mock_obj, k, types.MethodType(v.__func__, cls))
        elif inspect.isfunction(v):
            setattr(mock_obj, k, types.MethodType(v, mock_obj))
    return mock_obj

def create_vllm_config(
    kv_role: str="kv_producer",
    kv_connector: str = "LLMDataDistConnector",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
    max_model_len: int = 10000,
    enable_chunked_prefill: bool = True,
    enable_permute_local_kv: bool = False,
) -> VllmConfig:
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=False,
    )
    model_config = mock_with_default_value(ModelConfig)
    model_config.max_model_len = max_model_len
    model_config.is_encoder_decoder = False
    model_config.is_multimodal_model = False
    model_config.inputs_embeds_size = 16
    model_config.num_query_heads = 16
    model_config.attention_chunk_size = 16
    model_config.dtype = torch.float16
    model_config.uses_xdrope_dim = 0
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector=kv_connector,
        kv_role=kv_role,
        enable_permute_local_kv=enable_permute_local_kv,
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
        parallel_config=parallel_config,
    )
    vllm_config.model_config = model_config
    return vllm_config


# ============================================================================
# vLLM Interface Compatibility Check Utilities
# ============================================================================


def get_class_from_path(class_path: str):
    """Get class object from path string"""
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def check_interface_exists(class_path: str, method_name: str) -> tuple[bool, str]:
    """
    Check if the interface exists
    
    Returns: (exists, error_message)
    """
    try:
        cls = get_class_from_path(class_path)
    except (ImportError, AttributeError) as e:
        return False, f"Failed to import class {class_path}: {e}"

    if not hasattr(cls, method_name):
        return False, f"Class {class_path} does not have method {method_name}"

    method = getattr(cls, method_name)
    if not callable(method):
        return False, f"{method_name} of class {class_path} is not callable"

    return True, ""


def check_method_signature(
    class_path: str, method_name: str, expected_params: list[str]
) -> tuple[bool, str]:
    """
    Check if method signature matches the snapshot
    
    Returns: (is_match, error_message)
    """
    exists, error_msg = check_interface_exists(class_path, method_name)
    if not exists:
        return False, error_msg

    try:
        cls = get_class_from_path(class_path)
        method = getattr(cls, method_name)
        sig = inspect.signature(method)
    except Exception as e:
        return False, f"Failed to get method signature: {e}"

    # Get parameter list (excluding self)
    actual_params = [
        name
        for name, param in sig.parameters.items()
        if name != "self"
    ]

    # Check parameter count
    if len(actual_params) < len(expected_params):
        missing = set(expected_params) - set(actual_params)
        return (
            False,
            f"Insufficient parameters: expected {len(expected_params)} parameters {expected_params}, "
            f"but only {len(actual_params)} parameters {actual_params}, missing: {missing}",
        )

    # Check parameter names and order (only check first N, as new parameters may be added)
    for i, expected_param in enumerate(expected_params):
        if i >= len(actual_params):
            return (
                False,
                f"Insufficient parameters: expected parameter {i+1} to be '{expected_param}', "
                f"but only {len(actual_params)} parameters exist",
            )
        if actual_params[i] != expected_param:
            return (
                False,
                f"Parameter {i+1} mismatch: expected '{expected_param}', got '{actual_params[i]}'",
            )

    return True, ""


def check_return_type(
    class_path: str, method_name: str, expected_return_type: str | None
) -> tuple[bool, str]:
    """
    Check if return type matches the snapshot
    
    Returns: (is_match, error_message)
    """
    exists, error_msg = check_interface_exists(class_path, method_name)
    if not exists:
        return False, error_msg

    try:
        cls = get_class_from_path(class_path)
        method = getattr(cls, method_name)
        sig = inspect.signature(method)
        actual_return = sig.return_annotation
    except Exception as e:
        return False, f"Failed to get return type: {e}"

    # Skip check if no expected return type
    if expected_return_type is None:
        return True, ""

    # Skip check if actual method has no return type annotation
    if actual_return == inspect.Signature.empty:
        return True, ""  # No type annotation, cannot check

    # Process actual return type: if it's a type object, get its name
    if isinstance(actual_return, type):
        # For built-in types (e.g., int, str), use __name__
        actual_return_str = actual_return.__name__
    else:
        # For other types (e.g., Union, string annotations), convert to string
        actual_return_str = str(actual_return)
        # If string representation is <class 'int'> format, extract type name
        if actual_return_str.startswith("<class '") and actual_return_str.endswith("'>"):
            actual_return_str = actual_return_str[8:-2]

    expected_return_str = str(expected_return_type)

    # Simple string matching
    if actual_return_str != expected_return_str:
        # Try more lenient matching (handle different Union type representations)
        if "Union" in expected_return_str or "|" in expected_return_str:
            # For Union types, only record difference, don't fail
            return (
                True,
                f"Return type difference (may be different Union type representation): "
                f"expected {expected_return_str}, got {actual_return_str}",
            )
        return (
            False,
            f"Return type mismatch: expected {expected_return_str}, got {actual_return_str}",
        )

    return True, ""


def check_interface_snapshot(snapshot: dict[str, Any]) -> tuple[bool, str]:
    """
    Check if interface snapshot matches (comprehensive check of existence, signature, return type)
    
    Returns: (is_match, error_message)
    """
    class_path = snapshot["class_path"]
    method_name = snapshot["method_name"]

    # Check if interface exists
    exists, error_msg = check_interface_exists(class_path, method_name)
    if not exists:
        return False, f"Interface does not exist: {error_msg}"

    # Check method signature
    is_match, error_msg = check_method_signature(
        class_path, method_name, snapshot["params"]
    )
    if not is_match:
        return False, f"Method signature changed: {error_msg}"

    # Check return type
    is_match, error_msg = check_return_type(
        class_path, method_name, snapshot.get("return_type")
    )
    if not is_match:
        return False, f"Return type changed: {error_msg}"

    return True, ""
