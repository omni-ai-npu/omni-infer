# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from omni_npu.attention.ops import attention_update_torch

class TestAttentionUpdateTorch:

    def test_with_inf(self):

        cfg_fp32 = {"dtype": "npu", "dtype": torch.float32}
        cfg_bf16 = {"dtype": "npu", "dtype": torch.bfloat16}
        N, T, D = 3, 8, 128
        inf = float("inf")

        outs = torch.randn(N, T, D, **cfg_bf16)
        lses = torch.randn(N, T, **cfg_fp32)

        def set_lse(tok: int, var: list):
            for i, x in enumerate(var):
                if x is not None:
                    lses[i, tok] = x

        set_lse(0, [-inf, -inf, -inf])
        set_lse(1, [-inf, -inf, inf])
        set_lse(2, [-inf, inf, inf])
        set_lse(3, [inf, inf, inf])

        set_lse(4, [-inf, None, -inf])
        set_lse(5, [-inf, None, inf])
        set_lse(6, [inf, None, inf])

        lse_golden = torch.logsumexp(lses.nan_to_num(posinf=-inf, neginf=-inf), dim=0)
        out, lse = attention_update_torch(outs, lses)

        # ======================= check shape =======================

        assert type(out) is torch.Tensor
        assert type(lse) is torch.Tensor
        assert out.shape == (T, D)
        assert lse.shape == (T,)
        assert out.dtype == torch.float32
        assert lse.dtype == torch.float32

        print(lse.tolist(), flush=True)
        print(lse_golden.tolist(), flush=True)

        # ======================= check lse =======================

        for x, golden in zip(
            lse.tolist(),
            lse_golden.tolist(),
        ):
            if golden == -inf or golden == inf:
                assert x == -inf
            else:
                assert abs(x - golden) < 1e-5

        # ======================= check out =======================

        assert not any(out.isnan().flatten().tolist())

        def check_var(a, b):
            print(a, b)
            assert abs(a.item() - b.item()) < 1e-3

        check_var(out[4, 0], outs[1, 4, 0])
        check_var(out[5, 0], outs[1, 5, 0])
        check_var(out[6, 0], outs[1, 6, 0])

    def test_accuracy(self):

        cfg_fp32 = {"dtype": "npu", "dtype": torch.float32}
        cfg_bf16 = {"dtype": "npu", "dtype": torch.bfloat16}
        N, T, D = 16, 100, 128

        outs = [torch.randn(T, D, **cfg_bf16) for i in range(N)]
        lses = [torch.randn(T, **cfg_fp32) for i in range(N)]
        lse_golden = torch.logsumexp(torch.stack(lses), dim=0)

        out, lse = attention_update_torch(outs, lses)

        # ======================= check shape =======================

        assert type(out) is torch.Tensor
        assert type(lse) is torch.Tensor
        assert out.shape == (T, D)
        assert lse.shape == (T,)
        assert out.dtype == torch.float32
        assert lse.dtype == torch.float32

        # ======================= check lse =======================

        for x, golden in zip(
            lse.tolist(),
            lse_golden.tolist(),
        ):
            assert abs(x - golden) < 1e-5
