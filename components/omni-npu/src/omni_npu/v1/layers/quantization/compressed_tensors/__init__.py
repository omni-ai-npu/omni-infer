# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from .npu_compressed_tensors_linear import W8A8Int8FCLinearMethod, W8A8Int8MlpMethod

__all__ = [
    "W8A8Int8FCLinearMethod",
    "W8A8Int8MlpMethod",
]

