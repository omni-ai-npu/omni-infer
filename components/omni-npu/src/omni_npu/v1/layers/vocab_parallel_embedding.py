# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is based on vLLM implementation:
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py

from typing import Tuple

import torch
import torch_npu
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod, 
    pad_vocab_size, 
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import get_dp_group
from omni_npu.v1.distributed.parallel_state_ext import (
    get_local_world_group,
    get_world_group,
)
from omni_npu.v1.distributed.communication_op_ext import reduce_scatter_local


DEFAULT_VOCAB_PADDING_SIZE = 64


def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.jit.script will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ <
                                                          org_vocab_end_index)
    # Adapt: avoid create added_vocab_mask when added_vocab_start_index == added_vocab_end_index.
    if added_vocab_start_index == added_vocab_end_index:
        valid_offset = (org_vocab_start_index *
                        org_vocab_mask)
        vocab_mask = org_vocab_mask
    else:
        added_vocab_mask = (input_ >= added_vocab_start_index) & (
            input_ < added_vocab_end_index)
        added_offset = added_vocab_start_index - (
            org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
        valid_offset = (org_vocab_start_index *
                        org_vocab_mask) + (added_offset * added_vocab_mask)
        vocab_mask = org_vocab_mask | added_vocab_mask
    # Adapt end.
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


# @VocabParallelEmbedding.register_oot
class NPUVocabParallelEmbedding(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        local_parallel: bool = False, # TODO: config like FlashCommLinear
    ):
        torch.nn.Module.__init__(self)

        self.local_parallel = local_parallel
        tp_size = get_tensor_model_parallel_world_size()
        local_size = get_local_world_group().world_size
        if local_parallel and local_size <= tp_size:
            tp_rank = get_world_group().local_rank
            self.tp_size = local_size
        else:
            tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = tp_size

        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size,
            self.padding_size,
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings,
            self.padding_size,
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = isinstance(self, VocabParallelEmbedding)
        quant_method_implements_embedding = method_has_implemented_embedding(type(quant_method))
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        # Divide the weight matrix along the vocabulary dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded,
            self.tp_size,
        )
        assert self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        si = self.shard_indices
        self.num_org_embeddings_per_partition = si.org_vocab_end_index - si.org_vocab_start_index
        self.num_added_embeddings_per_partition = si.added_vocab_end_index - si.added_vocab_start_index

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    def forward(self, input_, enable_scatter: bool = False):
        if enable_scatter: # global sequence parallel
            # pad because RS only support same size across ranks
            ceil = -(-input_.size(0) // self.tp_size) * self.tp_size
            if ceil > input_.size(0):
                padded = input_.new_zeros(ceil, *input_.shape[1:])
                padded[:input_.size(0)] = input_
                input_ = padded

        if self.tp_size > 1: # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index)
        else:
            masked_input = input_

        if masked_input.dtype != torch.long:
            masked_input = masked_input.long()

        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input)

        # Mask the output embedding.
        if self.tp_size > 1: # adapter for faster
            output_parallel *= ~input_mask.unsqueeze(-1)

        if enable_scatter: # RS, global sequence parallel
            if self.local_parallel:
                return reduce_scatter_local(output_parallel)
            else:
                return tensor_model_parallel_reduce_scatter(output_parallel, dim=0)
        else: # default
            assert not self.local_parallel
            return tensor_model_parallel_all_reduce(output_parallel)


# @ParallelLMHead.register_oot
class NPUParallelLMHead(NPUVocabParallelEmbedding):

    # Pad target for the DP all_gather inside NPULogitsProcessor. Set by
    # the runner per step from forward_context.dp_metadata; identical
    # across every DP-sharded lm_head, so we keep it as a class attribute
    # and avoid touching each instance every step.
    _dp_pad_n: int = 0

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        local_parallel: bool = False,
        dp_parallel: bool = False,
        local_lmhead_parallel: bool = False,
    ):
        # dp_parallel: use DP group instead of TP group for lm_head sharding
        self.dp_parallel = dp_parallel
        self.local_lmhead_parallel = local_lmhead_parallel
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
            local_parallel,
        )
        sharding_group = None
        if dp_parallel:
            sharding_group = get_dp_group()
        elif local_lmhead_parallel:
            sharding_group = get_local_world_group()
        if sharding_group is not None:
            tp_rank = sharding_group.rank_in_group
            self.tp_size = sharding_group.world_size
            self.shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            self.num_embeddings_per_partition = divide(
                self.num_embeddings_padded,
                self.tp_size,
            )
            assert self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
            self.num_org_embeddings_per_partition = (
                self.shard_indices.org_vocab_end_index
                - self.shard_indices.org_vocab_start_index
            )
            self.num_added_embeddings_per_partition = (
                self.shard_indices.added_vocab_end_index
                - self.shard_indices.added_vocab_start_index
            )
            if params_dtype is None:
                params_dtype = torch.get_default_dtype()
            self.quant_method.create_weights(
                self,
                self.embedding_dim,
                [self.num_embeddings_per_partition],
                self.embedding_dim,
                self.num_embeddings_padded,
                params_dtype=params_dtype,
                weight_loader=self.weight_loader,
            )
        self.quant_config = quant_config

        if bias:
            self.bias = Parameter(torch.empty(
                self.num_embeddings_per_partition,
                dtype=params_dtype,
            ))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: NPUVocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # GGUF quantized embed_tokens.
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            self.weight = embed_tokens.weight
            return self

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        super().weight_loader(param, loaded_weight)
        param.data = torch_npu.npu_format_cast(param.data, 29)
