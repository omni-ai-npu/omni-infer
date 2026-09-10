# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Omni-NPU configuration access and cross-source validation.

This package contains runtime accessors and validators, and is separate from
``omni_npu.model_config.configs``, which stores JSON model configurations.
"""
from omni_npu.configs.additional_config import OmniAdditionalConfig

__all__ = ["OmniAdditionalConfig"]
