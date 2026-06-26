# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC, abstractmethod

import torch

from omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize import (
    CommunicationStrategySelector,
    FusedMoEPreparePermuteAndUnpermuteFinalize,
    PreparePermuteResult,
)


class NPUFusedMoEMethodBase(ABC):

    def __init__(self):
        super().__init__()
        self.communication_strategy_selector = None

    def make_communication_strategy_selector(
        self,
        moe: torch.nn.Module,
    ):
        self.communication_strategy_selector = CommunicationStrategySelector(moe)

    def select_communication_strategy(self, num_tokens: int):
        return self.communication_strategy_selector.select_communication_strategy(num_tokens)

    def apply_prepare_permute(
        self,
        prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        return prepare_permute_and_unpermute_finalize.prepare_permute(layer, x, topk_ids)

    def prepare_finalize_metadata(
        self,
        prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        layer: torch.nn.Module,
        topk_weights: torch.Tensor,
        prepare_permute_result: PreparePermuteResult,
    ):
        return prepare_permute_and_unpermute_finalize.prepare_finalize_metadata(
            layer, topk_weights, prepare_permute_result
        )

    def apply_unpermute_finalize(
        self,
        prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        prepare_permute_result: PreparePermuteResult,
        finalize_params=None,
        finalize_metadata=None,
    ):
        return prepare_permute_and_unpermute_finalize.unpermute_finalize(
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            prepare_permute_result,
            finalize_params=finalize_params,
            finalize_metadata=finalize_metadata,
        )

    @abstractmethod
    def apply_experts(
        self,
        layer: torch.nn.Module,
        prepare_permute_result: PreparePermuteResult,
        activation: str = "silu",
    ):
        raise NotImplementedError
