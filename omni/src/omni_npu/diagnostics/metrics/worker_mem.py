# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""worker 进程视角显存采样。结果由 kv_transfer 汇入 KVConnectorStats 通道，不用 multiprocess。"""

import os
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# 采样节流：显存是慢变量，每多少次采集调用才真读一次。
SAMPLE_EVERY_N_STEPS = int(os.getenv("OMNI_METRICS_WORKER_MEM_EVERY", "50"))

_step = 0


def maybe_sample(rank: int) -> Optional[dict]:
    # 到点则读 torch allocator（不触发 device sync），返回 {rank: {alloc, reserved}}，否则 None。
    # 用 rank 作 key，经 stats 通道 aggregate 并集汇总以保 per-rank。异常隔离，绝不影响推理。
    global _step
    _step += 1
    if _step % SAMPLE_EVERY_N_STEPS != 0:
        return None
    try:
        import torch

        return {
            str(rank): {
                "alloc": float(torch.npu.memory_allocated()),
                "reserved": float(torch.npu.memory_reserved()),
            }
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("worker_mem sample failed on rank %s: %s", rank, e)
        return None
