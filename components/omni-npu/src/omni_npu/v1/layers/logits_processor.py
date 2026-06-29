# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A layer that compute logits from hidden_states."""

import torch
import os
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    get_dp_group,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from omni_npu.v1.distributed.parallel_state_ext import get_local_world_group


# @LogitsProcessor.register_oot
class NPULogitsProcessor(LogitsProcessor):

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        use_local_comm = getattr(lm_head, 'local_lmhead_parallel', False)
        use_dp_comm = getattr(lm_head, 'dp_parallel', False)

        local_n = None
        if use_local_comm or use_dp_comm:
            comm_group = (
                get_local_world_group() if use_local_comm else get_dp_group()
            )
            local_n = hidden_states.shape[0]
            pad_n = lm_head._dp_pad_n
            if local_n < pad_n:
                pad = torch.zeros(pad_n - local_n, *hidden_states.shape[1:],
                                  dtype=hidden_states.dtype,
                                  device=hidden_states.device)
                hidden_states = torch.cat([hidden_states, pad], dim=0)
            hidden_states = comm_group.all_gather(hidden_states, dim=0)

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

        if use_local_comm or use_dp_comm:
            comm_group = (
                get_local_world_group() if use_local_comm else get_dp_group()
            )
            if os.environ.get("OMNI_NPU_USE_DEVICE_COMM_A2A", "0") == "1":
                logits = comm_group.device_communicator.all_to_all(
                    logits, scatter_dim=0, gather_dim=-1,
                )[:local_n]
            else:
                group_world = comm_group.world_size
                x = logits.view(group_world, -1, logits.shape[-1])
                output = torch.empty_like(x)
                torch.distributed.all_to_all_single(
                    output.flatten(), x.flatten(),
                    group=comm_group.device_group,
                )
                logits = output.transpose(0, 1).reshape(
                    logits.shape[0] // group_world, -1
                )[:local_n]
        else:
            logits = tensor_model_parallel_all_gather(logits)
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits
