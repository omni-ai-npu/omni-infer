# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Register NPU OffloadingSpec into vLLM OffloadingSpecFactory."""

import importlib

from vllm.logger import init_logger

logger = init_logger(__name__)


def _override_spec(name: str, module: str, class_name: str) -> None:
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    def loader():
        mod = importlib.import_module(module)
        return getattr(mod, class_name)

    OffloadingSpecFactory._registry[name] = loader
    logger.info(
        "kv_offload: overridden OffloadingSpec: %s -> %s.%s",
        name,
        module,
        class_name,
    )


def register_kv_offload_specs() -> None:
    """Replace CPUOffloadingSpec with the NPU implementation on Ascend."""
    logger.info("kv_offload: starting NPU OffloadingSpec registration")
    _override_spec(
        "CPUOffloadingSpec",
        "omni_npu.v1.kv_offload.cpu.spec",
        "NPUCPUOffloadingSpec",
    )
    logger.info("kv_offload: NPU OffloadingSpec registration finished")
