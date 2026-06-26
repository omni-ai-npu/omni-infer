# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""OmniCache KV diagnostics: dump and consistency checking for PD disaggregation.

Activate by setting KV_DUMP_DIR (or DUMP_KV_TENSOR=1) before launching vLLM.
"""

from tools.diagnostics.config import get_config, is_active

__all__ = ["get_config", "is_active"]