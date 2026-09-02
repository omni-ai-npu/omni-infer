# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from __future__ import annotations

from typing import Callable

import torch


def activate_shared_expert_on_side_stream(
    side_stream,
    layer: torch.nn.Module,
    prepare_permute_result,
    quant_fn: Callable[[torch.Tensor], object],
):
    """Activate + quantize the shared-expert gate_up projection on a side stream.

    Used by the ``with_routed_experts_cv`` schedule, where the shared expert
    runs concurrently with the routed-expert cube ops. Waits on the
    prepare/permute finished event, then applies ``act_fn`` followed by the
    caller-supplied ``quant_fn`` on ``side_stream``.

    ``quant_fn`` keeps this helper agnostic of the quantization scheme:
    HiFloat8 casts the activation, MXFP8 dynamically quantizes it (returning a
    scale alongside the activation). Whatever ``quant_fn`` returns is returned
    verbatim.
    """
    prepare_permute_result.shared_expert_finished_event.wait(torch.npu.current_stream())
    side_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side_stream):
        shared_expert_gate_up = prepare_permute_result.shared_expert_gate_up
        shared_expert_act = layer._shared_experts._layer.act_fn(shared_expert_gate_up)
        return quant_fn(shared_expert_act)
