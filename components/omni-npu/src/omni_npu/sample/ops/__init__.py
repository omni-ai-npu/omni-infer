# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from omni_npu.sample.ops.topk_topp_sampler import (
    NPUTopKTopPSampler,
    apply_top_k_top_p,
    apply_top_k_top_p_npu,
    apply_top_k_only,
    generate_coins,
    random_sample,
)

__all__ = [
    "NPUTopKTopPSampler",
    "apply_top_k_top_p",
    "apply_top_k_top_p_npu",
    "apply_top_k_only",
    "generate_coins",
    "random_sample",
]
