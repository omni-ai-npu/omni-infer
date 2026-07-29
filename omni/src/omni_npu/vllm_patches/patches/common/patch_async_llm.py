# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.logger import init_logger
from vllm.v1.engine.async_llm import AsyncLLM

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


@register_patch("AsyncLLMResumePatch", AsyncLLM)
class AsyncLLMResumePatch(VLLMPatch):
    """Re-capture ACLGraph on /resume."""

    _attr_names_to_apply = ["resume_generation"]

    async def resume_generation(self) -> None:
        try:
            await self.collective_rpc("recapture_model")
        except Exception:
            logger.exception(
                "recapture_model failed during resume_generation; "
                "engine stays paused"
            )
            raise

        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()
