# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from typing import Callable

from vllm.v1.worker.worker_base import WorkerWrapperBase

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("WorkerWrapperBasePatch", WorkerWrapperBase)
class WorkerWrapperBasePatch(VLLMPatch):
    """Patch to fix pp bugs using ray."""

    _attr_names_to_apply = [
        "adjust_rank",
    ]

    def adjust_rank(self, rank_mapping: dict[int, int]) -> None:
        if self.rpc_rank in rank_mapping:
            self.rpc_rank = rank_mapping[self.rpc_rank]
            self.global_rank = self.rpc_rank
