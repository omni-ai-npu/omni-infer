# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from .pangu_v2_moe import OpenPanguV2ForCausalLM


class OpenPanguV3ForCausalLM(OpenPanguV2ForCausalLM):
    """OpenPangu V3 model using the shared Pangu MoE inference implementation."""

    # V3 has no MoME state cache, so override V2's attention/Mamba hybrid flag.
    is_hybrid = False

    @classmethod
    def get_model_state_cls(cls):
        # MRv2 checks this hook before is_hybrid, so override V2's MoME state selection as well.
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState

        return DefaultModelState
