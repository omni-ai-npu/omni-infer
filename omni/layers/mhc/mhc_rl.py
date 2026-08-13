# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.logger import init_logger

from omni_npu.v1.utils import on_ascend950
from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)
try:
    import omni_training_custom_ops
except ImportError as e:
    logger.warning(f"Failed to import omni_training_custom_ops: {e}")
except Exception as e:
    logger.warning(f"Error occurred while importing omni_training_custom_ops: {e}")


class NPUmHCRL(torch.nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        pre_only: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        self.hc_eps = 1e-6
        assert config.mhc_use_gamma
        self.pre_only = pre_only
        self.on_ascend950 = on_ascend950()

        if not self.pre_only:
            self.branch_alpha = torch.nn.Parameter(
                torch.empty(3, dtype=torch.float32)
            )
            self.branch_beta = torch.nn.Parameter(
                torch.empty(self.num_stream * (self.num_stream + 2), dtype=torch.float32)
            ) 
        else:
            self.branch_alpha_pre = torch.nn.Parameter(
                torch.empty(1, dtype=torch.float32)
            )
            self.branch_beta_pre = torch.nn.Parameter(
                torch.empty(self.num_stream, dtype=torch.float32)
            )

        self.phi = ReplicatedLinear(
            self.hidden_size * self.num_stream,
            output_size=self.num_stream if self.pre_only \
                        else (self.num_stream + 2) * self.num_stream,
            bias=False,
            prefix=f"{prefix}.phi",
            params_dtype=torch.float32,
        )

        self.norm_gamma = torch.nn.Parameter(
            torch.empty(self.hidden_size * self.num_stream, dtype=torch.float32)
        )

    def post_weight_load(self) -> None:
        # Match the training path by rounding loaded weights to the model dtype once,
        # while keeping FP32 parameter storage for the MHC computation. For a BF16
        # checkpoint and FP16 model dtype: BF16 load -> FP32 -> FP16 round -> FP32.
        parameters = [self.phi.weight, self.norm_gamma]
        if self.pre_only:
            parameters.extend([self.branch_alpha_pre, self.branch_beta_pre])
        else:
            parameters.extend([self.branch_alpha, self.branch_beta])
        with torch.no_grad():
            for parameter in parameters:
                parameter.copy_(parameter.to(model_extra_config.dtype))

    def _mhc_pre_naive(self, hidden_states: torch.Tensor):
        shape, dtype = hidden_states.size(), hidden_states.dtype
        hidden_states = hidden_states.flatten(-2).float()
        rsqrt = torch.rsqrt(
            hidden_states.square().mean(-1, keepdim=True) + self.norm_eps,
        )
        mixes = torch.nn.functional.linear(
            hidden_states * rsqrt * self.norm_gamma,
            self.phi.weight,
        )
        h_pre, h_post, h_res = mixes.split(
            [self.num_stream, self.num_stream, self.num_stream**2], dim=-1,
        )

        h_pre = torch.nn.functional.sigmoid(
            h_pre * self.branch_alpha_pre + self.branch_beta_pre
        ) + self.hc_eps
        h_post = 2 * torch.nn.functional.sigmoid(
            h_post * self.branch_alpha_post + self.branch_beta_post
        )
        h_res = h_res.unflatten(-1, (self.num_stream, self.num_stream)) * self.branch_alpha_res \
                    + self.branch_beta_res.view(self.num_stream, self.num_stream)
        hidden_states = torch.sum(
            h_pre.unsqueeze(-1) * hidden_states.view(shape), dim=-2,
        ).to(dtype)
        return hidden_states, h_post, h_res

    def _mhc_sinkhorn_naive(self, h_res: torch.Tensor):
        h_res = h_res.softmax(-1) + self.hc_eps

        # === Step 1: Initial Col Norm ===
        col_sum = h_res.sum(-2, keepdim=True) + self.hc_eps
        h_res = h_res / col_sum

        # === Step 2: Loop ===
        for _ in range(self.mhc_recur_norm - 1):
            # Row Norm
            row_sum = h_res.sum(-1, keepdim=True) + self.hc_eps
            h_res = h_res / row_sum

            # Col Norm
            col_sum = h_res.sum(-2, keepdim=True) + self.hc_eps
            h_res = h_res / col_sum

        return h_res

    def _mhc_post_naive(
            self,
            hidden_states: torch.Tensor,
            h_post: torch.Tensor,
            residual: torch.Tensor,
            h_res: torch.Tensor,
    ):
        hidden_states = (
            h_post.unsqueeze(-1) * hidden_states.unsqueeze(-2) \
            + torch.sum(h_res.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
        ).to(hidden_states.dtype)
        return hidden_states

    def mhc_pre(
        self,
        hidden_states: torch.Tensor,
    ):
        if self.pre_only:
            dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, self.hidden_size * self.num_stream)
            hidden_states = hidden_states.float()
            if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
                normalized_hidden_states, _ = torch_npu.npu_rms_norm(
                    hidden_states,
                    self.norm_gamma.view(self.hidden_size * self.num_stream),
                    self.hc_eps,
                )
                hpre_weight = self.phi(normalized_hidden_states)[0]
            else:
                rsqrt = torch.rsqrt(
                    hidden_states.square().mean(-1, keepdim=True) + self.hc_eps
                )
                hpre_weight = self.phi(
                    hidden_states
                    * rsqrt
                    * self.norm_gamma.view(1, self.hidden_size * self.num_stream),
                )[0]

            hpre_weight = torch.nn.functional.sigmoid(
                hpre_weight * self.branch_alpha_pre
                + self.branch_beta_pre.view(1, self.num_stream)
            ) + self.hc_eps
            hpre_weight = hpre_weight.view(-1, self.num_stream, 1)
            hidden_states = hidden_states.view(-1, self.num_stream, self.hidden_size)
            hidden_states = torch.sum(
                hpre_weight * hidden_states,
                dim=1
            )
            hidden_states = hidden_states.view(-1, self.hidden_size)
            hidden_states = hidden_states.to(dtype)
            h_post = None
            h_res = None
        else:
            hidden_states = hidden_states.view(-1, self.num_stream, self.hidden_size)

            if not self.on_ascend950:
                if model_extra_config.operator_opt_config.use_batch_invariant_op:
                    phi_weight = self.phi.weight
                    gamma = self.norm_gamma.view(self.num_stream, self.hidden_size)
                    out_flag = 1
                else:
                    phi_weight = self.phi.weight * self.norm_gamma
                    gamma = None
                    out_flag = 0
                hidden_states, h_post, h_res, _, _, _ = \
                    torch.ops.custom.npu_manifold_constrained_hyper_connection_pre(
                        hidden_states,
                        phi_weight,
                        self.branch_alpha,
                        self.branch_beta,
                        gamma=gamma,
                        norm_eps=self.hc_eps,
                        hc_eps=self.hc_eps,
                        out_flag=out_flag,
                )
            else:
                hidden_states, h_post, h_res, _, _, _ = torch_npu.npu_mhc_pre(
                    hidden_states,
                    self.phi.weight,
                    self.branch_alpha,
                    self.branch_beta,
                    gamma=self.norm_gamma.view(self.num_stream, -1),
                    out_flag=0,
                )
            
        return hidden_states, h_post, h_res

    def mhc_sinkhorn(
        self,
        h_res: torch.Tensor,
    ):
        if self.pre_only:
            return h_res

        if not self.on_ascend950:
            h_res, _, _ = torch.ops.custom.npu_sinkhorn(
                h_res,
                eps=self.hc_eps,
                num_iters=self.mhc_recur_norm,
                out_flag=1 if model_extra_config.operator_opt_config.use_batch_invariant_op else 0,
            )
        else:
            h_res, _, _ = torch_npu.npu_mhc_sinkhorn(
                h_res,
                eps=self.hc_eps,
                num_iters=self.mhc_recur_norm,
                out_flag=0,
            )

        return h_res

    def mhc_post(
            self,
            hidden_states: torch.Tensor,
            h_post: torch.Tensor,
            residual: torch.Tensor,
            h_res: torch.Tensor,
    ):
        if self.pre_only:
            return residual

        residual = residual.view(-1, self.num_stream, self.hidden_size)
        if not self.on_ascend950:
            hidden_states = torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
                residual,
                h_res,
                hidden_states,
                h_post,
            )
        else:
            hidden_states = torch_npu.npu_mhc_post(
                residual,
                h_res,
                hidden_states,
                h_post,
            )
        
        hidden_states = hidden_states.view(-1, self.num_stream * self.hidden_size)

        return hidden_states
