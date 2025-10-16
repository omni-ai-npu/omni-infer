# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import math
import os
from typing import Optional

import torch
import torch_npu
from torch.distributed import ProcessGroup
from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)
from sglang.srt.utils import set_weight_attrs
from omni.adaptors.sglang.layers.moe.ep_moe.layer import moe_infer_fusion

SEQ_SPLIT_LENGTH = 4096
torch.npu.config.allow_internal_format = True


class AscendCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(self):
        self.warm_up = True
        self.n_routed_experts = None
        self.smooth_scale = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update({"quant_method": "channel"})

        w13_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1,
                dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16
            ),
            requires_grad=False,
        )
        w13_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts, 2 * intermediate_size_per_partition, 1,
                dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        layer.register_parameter("w13_weight_offset", w13_offset)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w13_offset, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, hidden_size, 1,
                dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16
            ),
            requires_grad=False,
        )
        w2_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts, hidden_size, 1,
                dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        layer.register_parameter("w2_weight_offset", w2_offset)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        #enable weight_nz default, need control by model config later
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight, 29)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight, 29)
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.to(torch.float32), requires_grad=False)
        self.n_routed_experts = len(layer.w13_weight)

        self.local_expert_indices_offset = (
            get_moe_ep_group().rank_in_group * self.n_routed_experts
        )
        self.local_expert_indices = [
            self.local_expert_indices_offset + i for i in range(self.n_routed_experts)
        ]

        self.smooth_scale = torch.ones(
            (self.n_routed_experts, layer.w13_weight_scale.shape[-1] // 2),
            dtype=torch.float32,
            device="npu",
        )
        torch._dynamo.mark_static(self.smooth_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: torch.Tensor,
        forward_batch,
        comm_group: Optional[ProcessGroup] = None
    ) -> torch.Tensor:
        if forward_batch.is_extend_in_batch:
            out = moe_infer_fusion(
                layer,
                hidden_states,
                topk_ids,
                scale,
                forward_batch,
                comm_group=comm_group
            )
        else:
            raise NotImplementedError("moe quant apply not support decode")

        return out