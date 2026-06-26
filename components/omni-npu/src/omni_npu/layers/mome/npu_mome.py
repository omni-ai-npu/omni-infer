# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
from omni_npu.v1.utils import on_ascend950

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.linear import (
    BasevLLMParameter, 
    ModelWeightParameter
)
from vllm.model_executor.models.utils import extract_layer_index

from omni_npu.attention.backends.mome import NPUMomeAttentionMetadata

logger = init_logger(__name__)
try:
    import omni_custom_ops
except ImportError as e:
    logger.warning(f"Failed to import omni_custom_ops: {e}")
except Exception as e:
    logger.warning(f"Error occurred while importing omni_custom_ops: {e}")


class ColumnParallelMOME(torch.nn.Module):
    """MOME layer with tensor (column) parallelism.

    An MOME layer is to perform causal convolution on the token dimension 
    for each feature channel independently. 
    The tensor parallelism will split the channel dimension. 

    Args:
        dim: Number of feature channels. 
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    def __init__(
        self,
        dim: int, 
        kernel_width: int = 3, 
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ):
        
        super().__init__()

        self.on_ascend950 = on_ascend950()

        # Keep input parameters
        self.dim = dim
        self.kernel_width = kernel_width
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix     
        self.disable_tp = disable_tp
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.layer_idx = extract_layer_index(self.prefix)

        assert kernel_width == 3, \
            "MOME supports kernel_width = 3 only. "
        assert quant_config is None, \
            "MOME does not support quantization currently. "

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(dim, self.tp_size)
        self.output_size_per_partition = self.input_size_per_partition
        self.output_partition_sizes = [self.output_size_per_partition]
        
        # create the MOME weights
        # following UnquantizedLienarMethod.create_weights
        conv_weight = ModelWeightParameter(
            data=torch.empty(
                kernel_width,
                self.input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1, # ModelWeightParameter.load_column_parallel_weight will split the output_dim. 
            weight_loader=self.weight_loader_v2,
        )

        self.register_parameter("weight", conv_weight)

        self.update_param_tp_status()

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):        
        param.load_column_parallel_weight(loaded_weight=loaded_weight.squeeze(1).transpose(0, 1))

    def forward(
        self, 
        x: torch.Tensor, 
        conv_states: torch.Tensor, 
        mome_metadata: NPUMomeAttentionMetadata, 
        inplace: bool = True, 
    ) -> torch.Tensor:
        assert not self.on_ascend950
        assert mome_metadata.num_computed_tokens is not None
        return torch.ops.custom.npu_ai_infra_fused_causal_conv1d(
            x, 
            self.weight, 
            conv_states, 
            query_start_loc=mome_metadata.query_start_loc, 
            cache_indices=mome_metadata.cache_indices,
            num_accepted_tokens=mome_metadata.num_accepted_tokens, 
            num_computed_tokens=mome_metadata.num_computed_tokens,
            block_idx_first_scheduled_token=mome_metadata.block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=mome_metadata.block_idx_last_scheduled_token,
            initial_state_idx=mome_metadata.block_idx_last_computed_token,
            pad_slot_id=mome_metadata.pad_slot_id,
            max_query_len=mome_metadata.max_query_len,
            residual_connection=1,
            block_size=mome_metadata.B_size,
            conv_mode=1, 
            inplace=inplace, 
        )

    def forward_prefill(
        self,
        x: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.on_ascend950
        return torch_npu.npu_fused_causal_conv1d(
            x, self.weight, conv_states,
            query_start_loc=query_start_loc.to(torch.int32),
            cache_indices=cache_indices,
            initial_state_mode=torch.Tensor([2]).npu().to(torch.int32),
            bias=None,
            num_accepted_tokens=None,
            activation_mode="None",
            run_mode=0,
            residual_connection=1,
        )

    def forward_decode(
        self,
        x: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        pad_slot_id: int = -1,
    ) -> torch.Tensor:
        assert self.on_ascend950
        return torch_npu.npu_fused_causal_conv1d(
            x, self.weight, conv_states,
            query_start_loc=query_start_loc.to(torch.int32),
            cache_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens.to(dtype=torch.int32) \
                                if num_accepted_tokens is not None else None,
            activation_mode="None",
            pad_slot_id=pad_slot_id,
            run_mode=1,
            residual_connection=1,
        )
    
    def update_param_tp_status(self):
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size

    def extra_repr(self) -> str:
        s = f"dim={self.dim}"
        s += f", tp_size={self.tp_size}"
        return s