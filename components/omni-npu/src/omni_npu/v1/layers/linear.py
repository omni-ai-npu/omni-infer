# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from abc import abstractmethod
from typing import Any, Literal, Optional, Union

import torch
from torch.nn.parameter import Parameter
import torch_npu

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
from vllm.distributed import (
    divide,
    split_tensor_along_last_dim,
    GroupCoordinator,
)

from omni_npu.v1.distributed.communication_op_ext import (
    layer_parallel_all_reduce,
    layer_parallel_all_gather,
    layer_parallel_reduce_scatter,
    layer_parallel_all2all_single
)
from omni_npu.v1.distributed.parallel_state_ext import (
    get_layer_transform_type,
    get_layer_dim,
    get_layer_parallel_world_size,
    get_layer_parallel_rank,
    get_layer_parallel_group,
)
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.utils import get_last_two_parts
from omni_npu.compilation.acl_graph import set_aclgraph_recapture

logger = init_logger(__name__)
MB_TO_BYTES = 1024 * 1024


def layer_parallel_communication_op(data: torch.Tensor,
                                    transform_type: str,
                                    layer_name_inside_block: str,
                                    tensor_tag: str,
                                    dim: str):
    if transform_type == "ALL2ALL":
        return layer_parallel_all2all_single(data, layer_name_inside_block, tensor_tag, dim)
    elif transform_type == "AllReduce":
        return layer_parallel_all_reduce(data, layer_name_inside_block, tensor_tag)
    elif transform_type == "ReduceScatter":
        return layer_parallel_reduce_scatter(data, layer_name_inside_block, tensor_tag, dim)
    elif transform_type == "AllGather":
        return layer_parallel_all_gather(data, layer_name_inside_block, tensor_tag, dim)
    else:
        return data


class FlashCommLinearMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        x_transform: str = "NoOp",
        x_dim: Optional[int] = 0,
        throw_dequant: Optional[bool] = False
    ) -> torch.Tensor: 
        raise NotImplementedError


class UnquantizedFlashCommLinearMethod(FlashCommLinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        weight_nz = (
            model_extra_config.operator_opt_config.unquant_bmm_nz 
            and not (model_extra_config.operator_opt_config.enable_mlaprolog and "kv_b_proj" in layer.prefix)
        )
        weight_t = layer.weight.data.t().contiguous()
        if weight_nz:
            layer.weight.data = torch_npu.npu_format_cast(weight_t, torch_npu.Format.FRACTAL_NZ)
            opt_raw = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") # bytes
            allow_internal_format = opt_raw.strip().lower() == b"enable"
            if allow_internal_format:
                set_weight_attrs(layer.weight, {"is_weight_nz": True})
        else:
            layer.weight.data = weight_t
        if not hasattr(layer.weight, "is_weight_transposed"):
            set_weight_attrs(layer.weight, {"is_weight_transposed": True})

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        x_transform: Optional[str] = None,
        x_dim: Optional[int] = 0,
        throw_dequant: Optional[bool] = False,
    ) -> torch.Tensor:
        input_parallel = layer_parallel_communication_op(x, x_transform, layer.layer_name_inside_block, "x", x_dim)
        if bias is not None:
            if input_parallel.ndim == 3:
                return torch.matmul(input_parallel, layer.weight) + bias
            return torch.addmm(bias, input_parallel, layer.weight)
        else:
            return torch.matmul(input_parallel, layer.weight)


class FlashCommLinearBase(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        self.layer_name_inside_block = get_last_two_parts(prefix)
        self.x_transform = get_layer_transform_type(self.layer_name_inside_block, "x")
        self.y_transform = get_layer_transform_type(self.layer_name_inside_block, "y")
        self.x_dim = get_layer_dim(self.layer_name_inside_block, "x")
        self.y_dim = get_layer_dim(self.layer_name_inside_block, "y")

        if quant_config is None or (quant_config.ignore is not None and prefix in quant_config.ignore):
            self.quant_method = UnquantizedFlashCommLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        self.tp_rank = (get_layer_parallel_rank(self.layer_name_inside_block)
                        if not disable_tp else 0)
        self.tp_size = (get_layer_parallel_world_size(self.layer_name_inside_block)
                        if not disable_tp else 1)

    def update_param_tp_status(self):
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size


class ReplicatedFlashCommLinear(FlashCommLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True
    ):
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=True,
        )

        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # veRL special case: transpose the weight back to original shape
        is_weight_nz = getattr(param, "is_weight_nz", False)
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.ND)
        if is_weight_transposed:
            param.data = param.data.t_()

        param_data = param.data
        is_2_dims = getattr(param, "is_2_dims", False)
        if not is_2_dims:
            loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t_()
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.FRACTAL_NZ)
            set_aclgraph_recapture(True)

    def forward(
        self,
        input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None

        output = self.quant_method.apply(self, input_, bias, self.x_transform, self.x_dim)
        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


class ColumnParallelFlashCommLinear(FlashCommLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[list[int]] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.layer_name_inside_block = get_last_two_parts(prefix)
        self.tp_rank = (get_layer_parallel_rank(self.layer_name_inside_block)
                        if not disable_tp else 0)
        self.tp_size = (get_layer_parallel_world_size(self.layer_name_inside_block)
                        if not disable_tp else 1)

        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        super().__init__(input_size,
                    output_size,
                    skip_bias_add,
                    params_dtype,
                    quant_config,
                    prefix,
                    return_bias=return_bias,
                    disable_tp=disable_tp)

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # veRL special case: transpose the weight back to original shape
        is_weight_nz = getattr(param, "is_weight_nz", False)
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.ND)
        if is_weight_transposed:
            param.data = param.data.t_()
        output_dim = getattr(param, "output_dim", None)

        param_data = param.data
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        is_2_dims = getattr(param, "is_2_dims", False)
        if not is_2_dims:
            loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t_()
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.FRACTAL_NZ)
            set_aclgraph_recapture(True)

    def forward(
        self,
        input_: torch.Tensor,
        throw_dequant: Optional[bool] = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None

        output_parallel = self.quant_method.apply(
            self, input_, bias, self.x_transform, self.x_dim,
            throw_dequant=throw_dequant,
        )
        output = layer_parallel_communication_op(
            output_parallel, self.y_transform, self.layer_name_inside_block, "y",
            self.y_dim,
        )
        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        return s


class QKVParallelFlashCommLinear(ColumnParallelFlashCommLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.layer_name_inside_block = get_last_two_parts(prefix)
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = (get_layer_parallel_world_size(self.layer_name_inside_block)
                if not disable_tp else 1)
        self.y_world_size = get_layer_parallel_world_size(self.layer_name_inside_block, "y")
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj 
        ]

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix,
                         return_bias=return_bias,
                         disable_tp=disable_tp)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[str] = None):
        if self.y_transform == "ALL2ALL":
            assert self.tp_size == 1
            self.load_qkv_weights_interleaved(param, loaded_weight)
            return
        # veRL special case: transpose the weight back to original shape
        is_weight_nz = getattr(param, "is_weight_nz", False)
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.ND)
        if is_weight_transposed:
            param.data = param.data.t_()
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        
        if loaded_shard_id is None:
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                ("k", self.total_num_heads * self.head_size,
                 self.total_num_kv_heads * self.head_size),
                ("v", (self.total_num_heads + self.total_num_kv_heads) *
                 self.head_size, self.total_num_kv_heads * self.head_size),
            ]
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        assert output_dim is not None
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads +
                            self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size

        param_data = param_data.narrow(output_dim, shard_offset,
                                        shard_size)
        if loaded_shard_id == "q":
            shard_id = self.tp_rank
        else:
            shard_id = self.tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size

        loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                shard_size)

        is_2_dims = getattr(param, "is_2_dims", False)
        if not is_2_dims:
            loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t_()
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.FRACTAL_NZ)
            set_aclgraph_recapture(True)

    # When y_transform = ALL2ALL for the weight_loader, weights must be rearranged.
    # before rearranged:
    # | q_head1..q_head8 | k_head1..k_head4 | v_head1..v_head4 |
    # after rearranged:
    # y_world_size=2:
    # | q_head1..q_head4 | k_head1..k_head2 | v_head1..v_head2 |
    # | q_head5..q_head8 | k_head3..k_head4 | v_head3..v_head4 |
    # y_world_size=4:
    # | q_head1..q_head2 | k_head1 | v_head1 |
    # | q_head3..q_head4 | k_head2 | v_head2 |
    # | q_head5..q_head6 | k_head3 | v_head3 |
    # | q_head7..q_head8 | k_head4 | v_head4 |
    # y_world_size=8:
    # | q_head1 | k_head1 | v_head1 | q_head2 | k_head1 | v_head1 |
    # | q_head3 | k_head2 | v_head2 | q_head4 | k_head2 | v_head2 |
    # | q_head5 | k_head3 | v_head3 | q_head6 | k_head3 | v_head3 |
    # | q_head7 | k_head4 | v_head4 | q_head8 | k_head4 | v_head4 |
    def load_qkv_weights_interleaved(self,
                                     param: Parameter,
                                     loaded_weight: torch.Tensor):
        is_weight_nz = getattr(param, "is_weight_nz", False)
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.ND)
        if is_weight_transposed:
            param.data = param.data.t_()
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None, "output_dim must be defined for QKV weight loading"

        q_total_size = self.total_num_heads * self.head_size
        k_total_size = self.total_num_kv_heads * self.head_size
        v_total_size = self.total_num_kv_heads * self.head_size

        q_weight_global = loaded_weight.narrow(output_dim, 0, q_total_size)
        k_weight_global = loaded_weight.narrow(output_dim, q_total_size, k_total_size)
        v_weight_global = loaded_weight.narrow(output_dim, q_total_size + k_total_size, v_total_size)
        
        y_num_heads = divide(self.total_num_heads, self.y_world_size)
        if self.y_world_size >= self.total_num_kv_heads:
            y_num_kv_heads = 1
            y_num_kv_head_replicas = divide(self.y_world_size, self.total_num_kv_heads)
        else:
            y_num_kv_heads = divide(self.total_num_kv_heads, self.y_world_size)
            y_num_kv_head_replicas = 1

        current_offset = 0
        q_shard_size = y_num_heads * self.head_size
        kv_shard_size = y_num_kv_heads * self.head_size
        is_2_dims = getattr(param, "is_2_dims", False)
        
        for rank in range(self.y_world_size):
            q_shard_id = rank
            q_start_idx = q_shard_id * q_shard_size
            
            kv_shard_id = rank // y_num_kv_head_replicas
            kv_start_idx = kv_shard_id * kv_shard_size
            
            q_shard = q_weight_global.narrow(output_dim, q_start_idx, q_shard_size)
            param_target_q = param_data.narrow(output_dim, current_offset, q_shard_size)
            if not is_2_dims:
                q_shard = torch.squeeze(q_shard)
            assert q_shard.shape == param_target_q.shape, (
                f"Q shard shape {q_shard.shape} != "
                f"target {param_target_q.shape} (rank {rank})"
            )
            param_target_q.copy_(q_shard)
            current_offset += q_shard_size
            
            k_shard = k_weight_global.narrow(output_dim, kv_start_idx, kv_shard_size)
            param_target_k = param_data.narrow(output_dim, current_offset, kv_shard_size)
            if not is_2_dims:
                k_shard = torch.squeeze(k_shard)
            assert k_shard.shape == param_target_k.shape, (
                f"K shard shape {k_shard.shape} != "
                f"target {param_target_k.shape} (rank {rank})"
            )
            param_target_k.copy_(k_shard)
            current_offset += kv_shard_size
            
            v_shard = v_weight_global.narrow(output_dim, kv_start_idx, kv_shard_size)
            param_target_v = param_data.narrow(output_dim, current_offset, kv_shard_size)
            if not is_2_dims:
                v_shard = torch.squeeze(v_shard)
            assert v_shard.shape == param_target_v.shape, (
                f"V shard shape {v_shard.shape} != "
                f"target {param_target_v.shape} (rank {rank})"
            )
            param_target_v.copy_(v_shard)
            current_offset += kv_shard_size

        total_expected_size = self.y_world_size * (q_shard_size + 2 * kv_shard_size)
        assert current_offset == total_expected_size, (
            f"Total copied size {current_offset} != expected {total_expected_size}"
        )
        assert param_data.shape[output_dim] == total_expected_size, (
            f"Param dimension {param_data.shape[output_dim]} "
            f"!= expected {total_expected_size}"
        )
        total_size = q_total_size + k_total_size + v_total_size
        assert loaded_weight.shape[output_dim] == total_size, (
            f"Loaded weight dimension {loaded_weight.shape[output_dim]} "
            f"!= Q+K+V size"
        )

        if is_weight_transposed:
            param.data = param.data.t_()
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.FRACTAL_NZ)
            set_aclgraph_recapture(True)


class MergedColumnParallelFlashCommLinear(ColumnParallelFlashCommLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.layer_name_inside_block = get_last_two_parts(prefix)
        self.output_sizes = output_sizes
        self.tp_size = (get_layer_parallel_world_size(self.layer_name_inside_block)
                        if not disable_tp else 1)
        self.tp_rank = (get_layer_parallel_rank(self.layer_name_inside_block)
                        if not disable_tp else 0)

        assert all(output_size % self.tp_size == 0
                   for output_size in output_sizes)
        super().__init__(input_size=input_size,
                         output_size=sum(output_sizes),
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix,
                         return_bias=return_bias,
                         disable_tp=disable_tp)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):
        # veRL special case: transpose the weight back to original shape
        is_weight_nz = getattr(param, "is_weight_nz", False)
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.ND)
        if is_weight_transposed:
            param.data = param.data.t_()
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        assert loaded_shard_id is not None
        assert loaded_shard_id < len(self.output_sizes)
        assert output_dim is not None
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        is_2_dims = getattr(param, "is_2_dims", False)
        if not is_2_dims:
            loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t_()
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.FRACTAL_NZ)
            set_aclgraph_recapture(True)


class RowParallelFlashCommLinear(FlashCommLinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.layer_name_inside_block = get_last_two_parts(prefix)
        # Divide the weight matrix along the first dimension.
        self.tp_rank = (get_layer_parallel_rank(self.layer_name_inside_block)
                        if not disable_tp else 0)
        self.tp_size = (get_layer_parallel_world_size(self.layer_name_inside_block)
                        if not disable_tp else 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias,
                         disable_tp=disable_tp)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # veRL special case: transpose the weight back to original shape
        is_weight_nz = getattr(param, "is_weight_nz", False)
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.ND)
        if is_weight_transposed:
            param.data = param.data.t_()
        input_dim = getattr(param, "input_dim", None)
        param_data = param.data

        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)

        is_2_dims = getattr(param, "is_2_dims", False)
        if not is_2_dims:
            loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t_()
        if is_weight_nz:
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.FRACTAL_NZ)
            set_aclgraph_recapture(True)

    def forward(
        self,
        input_,
        next_layer=None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_, self.x_transform, self.x_dim)
        if next_layer:
            # Config is expressed in MiB, while npu_prefetch expects bytes.
            prefetch_size_bytes = model_extra_config.operator_opt_config.attn_prefetch * MB_TO_BYTES
            if prefetch_size_bytes > 0:
                for layer in next_layer:
                    torch_npu.npu_prefetch(layer.weight, output_parallel, prefetch_size_bytes)

        output = layer_parallel_communication_op(
            output_parallel, self.y_transform, self.layer_name_inside_block, "y",
            self.y_dim,
        )

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s



class ShardedLinearMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        in_size: int,
        out_size: int,
        params_dtype: torch.dtype,
        shard_group: GroupCoordinator,
        bias: bool = True,
        weight_nz: bool = True,
    ):
        raise NotImplementedError

    @abstractmethod
    def prefetch(self, layer: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | dict,
        return_bias: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def register(layer, name, shape, dtype, loader):
        x = torch.empty(*shape, dtype=dtype)
        x = Parameter(x, requires_grad=False)
        set_weight_attrs(x, {"weight_loader": loader})
        layer.register_parameter(name, x)

    @staticmethod
    def shard_weight(w: torch.Tensor, group: tuple, weight_nz: bool):
        rank, size, dev_group = group
        like = {"size": list(w.shape), "device": w.device, "dtype": w.dtype}
        if weight_nz:
            _w = torch_npu.npu_format_cast(w, torch_npu.Format.FRACTAL_NZ)
            w = torch_npu.empty_with_format(**like, acl_format=torch_npu.Format.FRACTAL_NZ)
            w.copy_(_w)
        buf = w.untyped_storage()
        assert buf.size() % size == 0
        num = buf.size() // size
        buf = buf[rank * num : (rank + 1) * num]
        frag = torch.empty(num, dtype=torch.uint8, device=w.device)
        frag.untyped_storage().copy_(buf)

        # ref: frag, like, dev_group, weight_nz
        def gather_weight() -> torch.Tensor:
            dst = frag.new_empty(0)
            w = (torch_npu.empty_with_format(**like, acl_format=torch_npu.Format.FRACTAL_NZ)
                if weight_nz else torch.empty(**like))
            dst.set_(w.untyped_storage())
            torch.distributed.all_gather_into_tensor(dst, frag, group=dev_group)
            return w
        return gather_weight


class UnquantizedShardedLinearMethod(ShardedLinearMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        in_size: int,
        out_size: int,
        params_dtype: torch.dtype,
        shard_group: GroupCoordinator,
        bias: bool = True,
        weight_nz: bool = True,
    ):
        self.shape = torch.Size([in_size, out_size])
        self.shard_group = (shard_group.rank_in_group,
                            shard_group.world_size,
                            shard_group.device_group)
        self.weight_nz = weight_nz
        self.full_weight: torch.Tensor = None

        base = ShardedLinearMethodBase
        base.register(layer, "weight", [0], params_dtype, self._mat_loader)
        if bias:
            base.register(layer, "bias", [out_size], params_dtype, self._vec_loader)
        else:
            layer.bias = None

    def _vec_loader(self, weight: Parameter, loaded: torch.Tensor):
        weight.data.copy_(loaded.flatten())

    def _mat_loader(self, weight: Parameter, loaded: torch.Tensor):
        base = ShardedLinearMethodBase
        loaded = loaded.to(device=weight.data.device)
        loaded = loaded.transpose(0, 1).contiguous()
        assert loaded.shape == self.shape
        weight.gather_fn = base.shard_weight(
            loaded, self.shard_group, self.weight_nz)

    def prefetch(self, layer: torch.nn.Module):
        self.full_weight = layer.weight.gather_fn()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        return_bias: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.full_weight is not None, f"error: apply before prefetch"
        bias = None if return_bias else layer.bias
        if bias is None:
            y = torch.matmul(x, self.full_weight)
        elif x.dim() == 3:
            y = torch.matmul(x, self.full_weight) + bias
        else:
            y = torch.addmm(bias, x, self.full_weight)
        del self.full_weight
        self.full_weight = None
        return (y, layer.bias) if return_bias else y


class ShardedLinear(torch.nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        prefix: str,
        shard_group: GroupCoordinator,
        bias: bool = True,
        return_bias: bool = True,
        quant_config: QuantizationConfig = None,
        params_dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.prefix = prefix
        self.shape = [in_size, out_size]
        self.tp_size = 1
        self.shard_size = shard_group.world_size
        self.prefetch_stream = None
        self.return_bias = return_bias

        if quant_config is None:
            self.quant_method = UnquantizedShardedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

        assert isinstance(self.quant_method, ShardedLinearMethodBase)
        self.quant_method.create_weights(
            layer=self,
            in_size=in_size,
            out_size=out_size,
            params_dtype=params_dtype,
            shard_group=shard_group,
            bias=bias,
        )

    def prefetch(self, stream=None):
        self.prefetch_stream = stream
        self.quant_method.prefetch(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple:
        if self.prefetch_stream is not None:
            torch.npu.current_stream().wait_stream(self.prefetch_stream)
        y = self.quant_method.apply(self, x, self.return_bias)
        self.prefetch_stream = None
        return y

    def extra_repr(self):
        s = f"shape={self.shape}, "
        s += f"tp_size={self.tp_size}, "
        s += f"sharded_size={self.shard_size}, "
        s += f"bias={self.bias is not None}"
        return s


class ShardedFlashCommLinear(ShardedLinear):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        prefix: str,
        bias: bool = True,
        return_bias: bool = True,
        quant_config: QuantizationConfig = None,
        params_dtype: torch.dtype = torch.float,
    ):
        self.layer_name_inside_block = get_last_two_parts(prefix)
        shard_group = get_layer_parallel_group(self.layer_name_inside_block)
        super().__init__(
            in_size, out_size,
            prefix,
            shard_group,
            bias, return_bias,
            quant_config,
            params_dtype,
        )
        self.x_transform = get_layer_transform_type(self.layer_name_inside_block, "x")
        self.y_transform = get_layer_transform_type(self.layer_name_inside_block, "y")
        self.x_dim = get_layer_dim(self.layer_name_inside_block, "x")
        self.y_dim = get_layer_dim(self.layer_name_inside_block, "y")

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple:
        if self.prefetch_stream is not None:
            torch.npu.current_stream().wait_stream(self.prefetch_stream)

        if self.x_transform != "NoOp":
            assert isinstance(x, torch.Tensor)
            x = layer_parallel_communication_op(x, self.x_transform, self.layer_name_inside_block, "x", self.x_dim)

        y = self.quant_method.apply(self, x, self.return_bias)
        self.prefetch_stream = None

        if self.y_transform != "NoOp":
            assert not self.return_bias or self.bias is None
            if self.return_bias:
                y, bias = y
            y = layer_parallel_communication_op(y, self.y_transform, self.layer_name_inside_block, "y", self.y_dim)
            if self.return_bias:
                y = (y, bias)
        return y
