# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
from torch import nn
from typing import Any
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP

from omni_npu.model_config.config_loader.loader import model_extra_config


@MiniMaxText01RMSNormTP.register_oot
class NPUMiniMaxText01RMSNormTP(MiniMaxText01RMSNormTP):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        CustomOp.__init__(self)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.weight.weight_loader = NPUMiniMaxText01RMSNormTP.weight_loader
        self.variance_epsilon = eps
        self.tp_world = get_tensor_model_parallel_world_size()
        assert hidden_size % self.tp_world == 0

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        param.data.copy_(loaded_weight)

    @staticmethod
    def local_rms_sq_from_rstd(rstd: torch.Tensor) -> torch.Tensor:
        if rstd.shape[-1] != 1:
            rstd = rstd.mean(-1, keepdim=True)
        return rstd.reciprocal_().square_()

    @staticmethod
    def npu_tp_rms_norm_qk(
        q: torch.Tensor,
        q_weight: torch.Tensor,
        k: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        tp_world: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_normed, q_rstd = torch_npu.npu_rms_norm(q, q_weight, eps)
        k_normed, k_rstd = torch_npu.npu_rms_norm(k, k_weight, eps)

        if tp_world > 1:
            q_local_rms_sq = NPUMiniMaxText01RMSNormTP.local_rms_sq_from_rstd(
                q_rstd
            )
            k_local_rms_sq = NPUMiniMaxText01RMSNormTP.local_rms_sq_from_rstd(
                k_rstd
            )

            global_rms_sq = tensor_model_parallel_all_reduce(
                torch.cat([q_local_rms_sq, k_local_rms_sq], dim=-1)
            ).div_(tp_world)
            q_global_rms_sq, k_global_rms_sq = global_rms_sq.chunk(2, dim=-1)

            q_scale = (q_local_rms_sq / q_global_rms_sq).sqrt_().to(q_normed.dtype)
            k_scale = (k_local_rms_sq / k_global_rms_sq).sqrt_().to(k_normed.dtype)
            q_normed = q_normed.mul_(q_scale)
            k_normed = k_normed.mul_(k_scale)

        return q_normed, k_normed
    
    @staticmethod
    def get_shard_index(weight):
        shard_world_size = get_tensor_model_parallel_world_size()
        shard_rank = get_tensor_model_parallel_rank()
        shard_size = weight.shape[0] // shard_world_size
        shard_index = slice(shard_rank * shard_size, (shard_rank + 1) * shard_size)
        return shard_index

    @staticmethod
    def forward_qk_prefill(
        q_norm: "MiniMaxText01RMSNormTP",
        k_norm: "MiniMaxText01RMSNormTP",
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return NPUMiniMaxText01RMSNormTP.npu_tp_rms_norm_qk(
            q,
            q_norm.weight[NPUMiniMaxText01RMSNormTP.get_shard_index(q_norm.weight)],
            k,
            k_norm.weight[NPUMiniMaxText01RMSNormTP.get_shard_index(k_norm.weight)],
            q_norm.variance_epsilon,
            q_norm.tp_world,
        )
    
    @staticmethod
    def all_gather_qk_heads(
        q: torch.Tensor,
        k: torch.Tensor,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp_group = get_tp_group()
        tp_world = get_tensor_model_parallel_world_size()
        tokens_len = q.shape[0]
        q_local_heads = q.shape[-1] // head_dim
        k_local_heads = k.shape[-1] // head_dim
        total_local_heads = q_local_heads + k_local_heads
        local_qk = torch.cat(
            (
                q.view(tokens_len, q_local_heads, head_dim),
                k.view(tokens_len, k_local_heads, head_dim),
            ),
            dim=1,
        )
        if tp_world > 1:
            local_qk = tp_group.all_gather(local_qk, dim=0).view(
                tp_world,
                tokens_len,
                total_local_heads,
                head_dim,
            ).permute(1, 0, 2, 3)
        else:
            local_qk = local_qk.unsqueeze(1)
        return local_qk.split([q_local_heads, k_local_heads], dim=2)

    @staticmethod
    def forward_qk_decoder(
        q_norm: "MiniMaxText01RMSNormTP",
        k_norm: "MiniMaxText01RMSNormTP",
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp_rank = get_tp_group().rank_in_group
        tp_world = get_tensor_model_parallel_world_size()
        head_dim = q_norm.head_dim
        q_ag, k_ag = NPUMiniMaxText01RMSNormTP.all_gather_qk_heads(
            q,
            k,
            head_dim=head_dim,
        )

        q_normed = torch_npu.npu_rms_norm(
            q_ag,
            q_norm.weight.view(tp_world, -1, head_dim),
            q_norm.variance_epsilon,
        )[0]
        k_normed = torch_npu.npu_rms_norm(
            k_ag,
            k_norm.weight.view(tp_world, -1, head_dim),
            k_norm.variance_epsilon,
        )[0]
        return (
            q_normed[:, tp_rank].reshape(q_normed.shape[0], -1),
            k_normed[:, tp_rank].reshape(k_normed.shape[0], -1),
        )

    @staticmethod
    def forward_qk(
        q_norm: "MiniMaxText01RMSNormTP",
        k_norm: "MiniMaxText01RMSNormTP",
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_metadata = getattr(get_forward_context(), "attn_metadata", None)
        is_prefill = attn_metadata is None or attn_metadata[next(iter(attn_metadata))].num_prefills > 0

        if is_prefill:
            return NPUMiniMaxText01RMSNormTP.forward_qk_prefill(
                q_norm, k_norm, q, k
            )

        return NPUMiniMaxText01RMSNormTP.forward_qk_decoder(
            q_norm, k_norm, q, k
        )


@RMSNorm.register_oot
class NPURMSNorm(RMSNorm):
    def process_weights_after_loading(self) -> None:
        self.weight_fp32 = self.weight.to(torch.float32)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        quant_symbol: bool = False,
        y_transform: str = "",
    ) -> torch.Tensor | tuple[torch.Tensor | dict[str, Any], torch.Tensor]:
        if getattr(model_extra_config.operator_opt_config, "omni_disable_npu_add_rms_norm", False):
            if residual is not None:
                x = x + residual
                residual = x
            res = torch_npu.npu_rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )[0]
            if residual is not None:
                return res, residual
            return res
        
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if y_transform == "AG":
                x = get_tp_group().all_gather(x, dim=0)
            if quant_symbol:
                x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
                x = {"x_int8": x_int8, "pertoken_scale": pertoken_scale}
            return x, residual

        return torch_npu.npu_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
        )[0]
