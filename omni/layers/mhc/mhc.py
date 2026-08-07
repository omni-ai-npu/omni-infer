# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.layers.linear import ReplicatedLinear

from omni_models.models.pangu.openpangu import mHCModule
import omni_training_custom_ops


@mHCModule.register_oot
class NPUmHCModule(mHCModule):
    def __init__(
        self,
        config: PretrainedConfig,
        merge_layer_only_pre=False,
        prefix: str = "",
    ):
        super().__init__(config, merge_layer_only_pre, prefix)
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.merge_layer_only_pre = merge_layer_only_pre

        if not self.merge_layer_only_pre:
            phi_output_hidden_size = (self.num_stream + 2) * self.num_stream
            self.branch_alpha = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.branch_beta = nn.Parameter(
                torch.empty(self.num_stream * (self.num_stream + 2), dtype=torch.float32)
            )
        else:
            phi_output_hidden_size = self.num_stream
            self.branch_alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.branch_beta_pre = nn.Parameter(torch.empty(self.num_stream, dtype=torch.float32))

        self.phi = ReplicatedLinear(
            self.hidden_size * self.num_stream,
            phi_output_hidden_size,
            bias=False,
            prefix=f"{prefix}.phi",
            params_dtype=torch.float32,
        )
        self.mhc_use_gamma = config.mhc_use_gamma
        self.hc_eps = 1e-6
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        if self.mhc_use_gamma:
            self.norm_gamma = nn.Parameter(torch.empty(self.hidden_size * self.num_stream, dtype=torch.float32))
            self.register_buffer(
                "weight_absorb_gamma",
                torch.empty(phi_output_hidden_size, self.hidden_size * self.num_stream, dtype=torch.float32),
                persistent=False
            )

    def hc_pre(self, x: torch.Tensor):
        if self.merge_layer_only_pre:
            dtype = x.dtype
            x = x.float()
            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
            if self.mhc_use_gamma:
                weight = self.phi(x * rsqrt * self.norm_gamma.unsqueeze(0))[0]
            else:
                weight = self.phi(x)[0] * rsqrt
            h_pre, h_post, h_res = self.hc_split_sinkhorn_torch(weight)
            hidden_state = (
                torch.sum(
                    h_pre.unsqueeze(-1)
                    * x.unflatten(dim=-1, sizes=(self.num_stream, -1)),
                    dim=1,
                )
                .squeeze(1)
                .to(dtype)
            )
        else:
            hidden_state, h_post, h_res, _, _, _ = torch.ops.custom.npu_manifold_constrained_hyper_connection_pre(
                x.view(-1, self.num_stream, self.hidden_size),
                self.weight_absorb_gamma,
                self.branch_alpha,
                self.branch_beta,
                gamma=None,
                out_flag=0, # 表示算子走推理和decode分支（默认为0）；outflag=1走训练和prefill的模板，性能会比较差
                norm_eps=self.hc_eps,
                hc_eps=self.hc_eps
            )
            h_res, _, _ = torch.ops.custom.npu_sinkhorn(h_res, num_iters=self.mhc_recur_norm, eps=self.hc_eps)
        return hidden_state, h_post, h_res
    
    def hc_split_sinkhorn_torch(self, weight):
        if not self.merge_layer_only_pre:
            h_pre, h_post, h_res = weight.split(
                [self.num_stream, self.num_stream, self.num_stream * self.num_stream],
                dim=-1,
            )
            alpha_pre, alpha_post, alpha_res = self.branch_alpha.view(-1).split(
                [1, 1, 1]
            )
            beta_pre, beta_post, beta_res = self.branch_beta.view(-1).split(
                [self.num_stream, self.num_stream, self.num_stream * self.num_stream]
            )
            h_post = 2 * torch.sigmoid(h_post * alpha_post + beta_post)
            h_res = h_res.unflatten(-1, (self.num_stream, self.num_stream))
            h_res = h_res * alpha_res + beta_res.view(self.num_stream, self.num_stream)
            h_res, _, _ = torch.ops.custom.npu_sinkhorn(
                h_res, num_iters=self.mhc_recur_norm, eps=self.hc_eps
            )
        else:
            h_pre = weight
            h_post = None
            h_res = None
            alpha_pre = self.branch_alpha_pre
            beta_pre = self.branch_beta_pre
        h_pre = torch.sigmoid(h_pre * alpha_pre + beta_pre) + self.hc_eps
        return h_pre, h_post, h_res

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor):
        if self.merge_layer_only_pre:
            return x
        else:
            hidden_state = torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
                residual.unflatten(dim=-1, sizes=(self.num_stream, -1)), h_res, x, h_post
            ).view(-1, self.num_stream * self.hidden_size)
            return hidden_state

    def post_weight_load(self):
        if self.mhc_use_gamma and not self.merge_layer_only_pre:
            weight_absorb_gamma = (self.phi.weight * self.norm_gamma).contiguous()
            self.weight_absorb_gamma.copy_(weight_absorb_gamma)