# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
vLLM Interface Compatibility Tests

Detect changes in vLLM interfaces used by omni-npu after vLLM upgrades.
Main checks:
1. Whether the interface exists (whether it was deleted)
2. Whether method signatures changed (parameter names, count, order)
3. Whether return types changed
"""

from typing import Any

import pytest

from tests.unit.platform.utils import check_interface_snapshot


# ============================================================================
# Interface Snapshot Definitions
# ============================================================================

WORKER_BASE_INIT_SNAPSHOT = {
    "class_path": "vllm.v1.worker.worker_base.WorkerBase",
    "method_name": "__init__",
    "params": [
        "vllm_config",
        "local_rank",
        "rank",
        "distributed_init_method",
        "is_driver_worker",  # Optional parameter with default value
    ],
    "return_type": None,  # __init__ returns None
}

GPU_MODEL_RUNNER_SNAPSHOTS = [
    {
        "class_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "method_name": "__init__",
        "params": ["vllm_config", "device"],
        "return_type": None,
    },
    {
        "class_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "method_name": "get_kv_cache_spec",
        "params": [],  # Only self
        "return_type": "dict[str, vllm.v1.kv_cache_interface.KVCacheSpec]",
    },
    {
        "class_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "method_name": "load_model",
        "params": ["eep_scale_up"],  # Has default value False
        "return_type": None,
    },
    {
        "class_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "method_name": "capture_model",
        "params": [],  # Only self
        "return_type": "int",
    },
    {
        "class_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "method_name": "execute_model",
        "params": ["scheduler_output", "intermediate_tensors"],  # intermediate_tensors has default value None
        "return_type": "ModelRunnerOutput | IntermediateTensors | None",
    },
    {
        "class_path": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
        "method_name": "sample_tokens",
        "params": ["grammar_output"],
        "return_type": "ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors",
    },
]

PLATFORM_INIT_SNAPSHOT = {
    "class_path": "vllm.platforms.interface.Platform",
    "method_name": "__init__",
    "params": [],  # Only self
    "return_type": None,
}


# ============================================================================
# WorkerBase Interface Tests
# ============================================================================


def test_worker_base_init():
    """Test if WorkerBase.__init__ interface exists and whether signature and return type changed"""
    is_match, error_msg = check_interface_snapshot(WORKER_BASE_INIT_SNAPSHOT)
    assert is_match, f"WorkerBase.__init__ interface changed: {error_msg}"


# ============================================================================
# GPUModelRunner Interface Tests
# ============================================================================


@pytest.mark.parametrize("snapshot", GPU_MODEL_RUNNER_SNAPSHOTS)
def test_gpu_model_runner_interface(snapshot: dict[str, Any]):
    """Test if GPUModelRunner interface exists and whether signature and return type changed"""
    is_match, error_msg = check_interface_snapshot(snapshot)
    assert is_match, f"GPUModelRunner.{snapshot['method_name']} interface changed: {error_msg}"


# ============================================================================
# Platform Interface Tests
# ============================================================================


def test_platform_init():
    """Test if Platform.__init__ interface exists and whether signature and return type changed"""
    is_match, error_msg = check_interface_snapshot(PLATFORM_INIT_SNAPSHOT)
    assert is_match, f"Platform.__init__ interface changed: {error_msg}"
