# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ModelState for MoME models on NPU."""

from __future__ import annotations

import torch
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class MomeModelState(MambaHybridModelState):
    """Publish each step's prompt lengths to the MoME metadata builders.

    MoME's two decode-node corrections in NPUMomeAttentionMetadataBuilder
    are gated on num_prompt_tokens; without it the kernel reads one token too
    many and the first decoded token of every request is wrong. build() can
    only reach the value through the builder attribute, because the upstream
    call site passes a fixed kwarg set. MRv1 binds it from _prepare_inputs;
    under MRv2 the owner is this state, which also records the lengths.
    """

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        # req slot -> prompt length, the mirror of req_states.prompt_len.
        self.prompt_len = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # batch row -> prompt length, rebuilt for the builders every step.
        self.num_prompt_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Set by preprocess_state, consumed by prepare_attn. See both.
        self._real_batch = False

    def preprocess_state(self, *args, **kwargs) -> None:
        """Mark this step as a real batch.

        Upstream guarantees this hook runs on real batches only and before
        prepare_attn (see the ModelState interface and MambaHybridModelState),
        which is the only per-step signal that tells a dummy batch apart -- a dummy one
        reaches prepare_attn with the same shapes and a fabricated idx_mapping.
        """
        self._real_batch = True
        return super().preprocess_state(*args, **kwargs)

    def add_request(self, req_index, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        # prompt_len, not prefill_len: upstream RequestState documents that
        # the latter grows once a preempted request resumes, and the two disagree
        # silently on exactly the requests that were preempted.
        self.prompt_len[req_index] = len(new_req_data.prompt_token_ids)

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture: bool = False,
    ):
        from omni_npu.attention.backends.mome import bind_num_prompt_tokens

        real_batch = False
        if not for_capture:
            # Consume the token so a later dummy batch cannot inherit it.
            # Capture never sets it, and must not consume a pending one.
            real_batch, self._real_batch = self._real_batch, False

        if real_batch:
            num_reqs = input_batch.num_reqs
            buf = self.num_prompt_tokens
            buf[:num_reqs].copy_(
                self.prompt_len[input_batch.idx_mapping], non_blocking=True
            )
            # Padding rows must not keep the previous step's lengths.
            buf[num_reqs:].fill_(0)
            bind_num_prompt_tokens(attn_groups, buf)
        else:
            # Capture fabricates seq_lens and a dummy batch fabricates
            # idx_mapping; neither carries usable prompt lengths. MRv1
            # unbinds the same way at the top of _dummy_run.
            bind_num_prompt_tokens(attn_groups, None)

        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )
