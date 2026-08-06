# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCore

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


@register_patch("EngineCorePatch", EngineCore)
class EngineCorePatch(VLLMPatch):
    _attr_names_to_apply = ["post_step"]

    def post_step(self, model_executed: bool) -> None:
        # To support prefill PP + MTP, we skip updating draft token ids
        # This may cause draft tokens to be lost (which is ok for prefill nodes
        # in PD disaggregation scenario)
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        should_update = (
            pp_size == 1
            and not self.async_scheduling
            and self.use_spec_decode
            and model_executed
        )
        if should_update:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)