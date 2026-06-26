# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Project-level pytest configuration to handle custom operator registration conflicts"""

import pytest
import torch


def _is_custom_op_registered(op_name: str, namespace: str = "vllm") -> bool:
    try:
        ops_namespace = getattr(torch.ops, namespace, None)
        if ops_namespace is None:
            return False
        return hasattr(ops_namespace, op_name)
    except AttributeError:
        return False

def pytest_configure(config):
    try:
        from vllm.utils.torch_utils import direct_register_custom_op as originial_direct_register_custom_op

        def safe_direct_register_custom_op(op_name, op_func, **kwargs):
            if _is_custom_op_registered(op_name, kwargs.get('namespace', 'vllm')):
                return None
            return originial_direct_register_custom_op(op_name=op_name, op_func=op_func, **kwargs)

        import vllm.utils.torch_utils
        vllm.utils.torch_utils.direct_register_custom_op = safe_direct_register_custom_op
    except ImportError:
        pass