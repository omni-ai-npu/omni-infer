# omni_cache/attention/__init__.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
omni_cache attention module.

This module provides attention-related functionality including:
- metadata: Attention metadata operations
- kv_managers: KV cache managers for different attention patterns
- backends: Attention backend extensions
"""

from . import backends
from . import metadata

__all__ = ["backends", "metadata"]
