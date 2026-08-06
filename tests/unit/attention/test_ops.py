# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from omni.attention.ops import (
    apply_FA_rescale_forward,
    attention_update_torch,
    gather_and_maybe_dequant_cache,
)


def test_gather_and_maybe_dequant_cache_uses_cu_seq_lens():
    key_cache = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    value_cache = key_cache + 100
    block_table = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    cu_seq_lens = torch.tensor([0, 3, 5], dtype=torch.long)
    dst = torch.zeros(5, 4, dtype=torch.float32)

    gather_and_maybe_dequant_cache(
        src_cache=(key_cache, value_cache),
        dst=dst,
        block_table=block_table,
        cu_seq_lens=cu_seq_lens,
        batch_size=2,
        kv_cache_dtype="auto",
        scale=torch.empty(0),
    )

    expected = torch.cat(
        [
            torch.cat([key_cache[0, 0], value_cache[0, 0]]).unsqueeze(0),
            torch.cat([key_cache[0, 1], value_cache[0, 1]]).unsqueeze(0),
            torch.cat([key_cache[1, 0], value_cache[1, 0]]).unsqueeze(0),
            torch.cat([key_cache[2, 0], value_cache[2, 0]]).unsqueeze(0),
            torch.cat([key_cache[2, 1], value_cache[2, 1]]).unsqueeze(0),
        ],
        dim=0,
    )
    torch.testing.assert_close(dst, expected)


def test_apply_FA_rescale_forward_merges_with_softmax_max():
    T, N, D = 2, 3, 4
    output = torch.ones(T, N, D, dtype=torch.bfloat16)
    output_sink = torch.full((T, N, D), 3.0, dtype=torch.bfloat16)
    softmax_max = torch.zeros(T, N, 1, dtype=torch.float32)
    softmax_max_sink = torch.zeros(T, N, 1, dtype=torch.float32)
    softmax_sum = torch.ones(T, N, 1, dtype=torch.float32)
    softmax_sum_sink = torch.ones(T, N, 1, dtype=torch.float32)

    out, rescale_orig, rescale_sink = apply_FA_rescale_forward(
        output,
        softmax_max,
        softmax_sum,
        output_sink,
        softmax_max_sink,
        softmax_sum_sink,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == (T, N, D)
    torch.testing.assert_close(rescale_orig, torch.full_like(rescale_orig, 0.5))
    torch.testing.assert_close(rescale_sink, torch.full_like(rescale_sink, 0.5))
    torch.testing.assert_close(out.float(), torch.full((T, N, D), 2.0))


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
