# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MM Feature Connector - Connectors for persisting multimodal features.

This package provides connectors that persist multimodal features
(processed inputs like images, audio) across EPD nodes.
"""

from .base import BaseMMFeatureConnector
from .disk_connector import DiskMMFeatureConnector
from .factory import MMFeatureConnectorFactory

__all__ = [
    "BaseMMFeatureConnector",
    "DiskMMFeatureConnector",
    "MMFeatureConnectorFactory",
]