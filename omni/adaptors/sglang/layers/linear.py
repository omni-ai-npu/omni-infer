"""Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/linear.py"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    split_tensor_along_last_dim,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear as MergedColumnParallelLinearGPU,
)
from sglang.srt.layers.linear import RowParallelLinear as RowParallelLinearGPU
from torch.nn.parameter import Parameter, UninitializedParameter

from omni.adaptors.sglang.distributed import get_mlp_tp_group, get_o_proj_tp_group

logger = logging.getLogger(__name__)


class MergedColumnParallelLinear(MergedColumnParallelLinearGPU):

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if "mlp.gate_up_proj" in self.prefix:
            tp_group = get_mlp_tp_group()
        else:
            tp_group = parallel_state.get_tp_group()
        if tp_group is None:
            tp_group = parallel_state.get_tp_group()

        if self.gather_output:
            # All-gather across the partitions.
            output = tp_group.all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(RowParallelLinearGPU):

    def __init__(self, *args, prefix: str = "", **kwargs):
        super().__init__(*args, prefix=prefix, **kwargs)
        self.prefix = prefix

    def forward(self, input_, skip_all_reduce=False):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        if "mlp.down_proj" in self.prefix:
            tp_group = get_mlp_tp_group()
        elif "o_proj" in self.prefix:
            tp_group = get_o_proj_tp_group()
        else:
            tp_group = parallel_state.get_tp_group()

        if tp_group is None:
            tp_group = parallel_state.get_tp_group()

        with use_symmetric_memory(tp_group) as sm:
            output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)
            sm.tag(output_parallel)

        if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
            output = tp_group.all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias
