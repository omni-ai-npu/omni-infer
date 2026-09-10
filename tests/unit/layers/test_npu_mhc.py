# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from omni_npu.layers.mhc.npu_mhc import NPUmHC


pytestmark = pytest.mark.unit


class _Linear(nn.Linear):
    def forward(self, value):
        return super().forward(value), None


def test_mhc_pre_pre_only_uses_hc_eps_for_norm_and_gate():
    mhc = NPUmHC.__new__(NPUmHC)
    nn.Module.__init__(mhc)
    mhc.num_stream = 2
    mhc.hidden_size = 2
    mhc.pre_only = True
    mhc.norm_eps = 1.0
    mhc.hc_eps = 0.25
    mhc.phi = _Linear(4, 2, bias=False)
    mhc.norm_gamma = nn.Parameter(torch.ones(4))
    mhc.branch_alpha_pre = nn.Parameter(torch.ones(1))
    mhc.branch_beta_pre = nn.Parameter(torch.zeros(2))

    with torch.no_grad():
        mhc.phi.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]))

    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    flattened = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + mhc.hc_eps)
    gates = F.sigmoid(F.linear(flattened * rsqrt * mhc.norm_gamma, mhc.phi.weight)) + mhc.hc_eps
    expected = torch.sum(gates.unsqueeze(-1) * hidden_states, dim=1)

    actual, h_post, h_res = mhc.mhc_pre(hidden_states)

    torch.testing.assert_close(actual, expected)
    assert h_post is None
    assert h_res is None
