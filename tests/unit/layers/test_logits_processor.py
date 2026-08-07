# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omni_npu.v1.layers.logits_processor import NPULogitsProcessor

MOD = "omni_npu.v1.layers.logits_processor"


def _make_lm_head(
    dp_parallel,
    vocab_shard,
    hidden,
    dp_pad_n=0,
    local_lmhead_parallel=False,
):
    weight = torch.randn(hidden, vocab_shard)
    return SimpleNamespace(
        dp_parallel=dp_parallel,
        local_lmhead_parallel=local_lmhead_parallel,
        _dp_pad_n=dp_pad_n,
        quant_method=SimpleNamespace(
            apply=lambda lm, hs, bias=None: hs @ weight,
        ),
    )


def _make_processor(org_vocab_size):
    proc = NPULogitsProcessor.__new__(NPULogitsProcessor)
    proc.org_vocab_size = org_vocab_size
    return proc


def _make_dp_group(dp_size, all_gather_fn, all_to_all_return):
    g = MagicMock()
    g.world_size = dp_size
    g.all_gather = MagicMock(side_effect=all_gather_fn)
    g.device_communicator = SimpleNamespace(
        all_to_all=MagicMock(return_value=all_to_all_return),
    )
    return g


class TestNPULogitsProcessorNonDP:
    def test_uses_tp_all_gather_and_truncates_to_org_vocab(self):
        proc = _make_processor(org_vocab_size=10)
        lm_head = _make_lm_head(dp_parallel=False, vocab_shard=16, hidden=8)
        hidden = torch.randn(2, 8)
        tp_gathered = torch.randn(2, 16)

        with patch(f"{MOD}.tensor_model_parallel_all_gather", return_value=tp_gathered) as tp_ag, \
             patch(f"{MOD}.get_dp_group") as get_dp:
            out = proc._get_logits(hidden, lm_head, None)

        tp_ag.assert_called_once()
        get_dp.assert_not_called()
        assert out.shape == (2, 10)
        torch.testing.assert_close(out, tp_gathered[..., :10])


class TestNPULogitsProcessorDP:
    def test_pads_and_trims_when_local_less_than_pad_target(self):
        # Default path: torch.distributed.all_to_all_single
        dp_size, local_n, pad_n = 2, 3, 5
        hidden_dim, vocab_shard = 8, 16
        vocab_full = vocab_shard * dp_size

        proc = _make_processor(org_vocab_size=vocab_full)
        lm_head = _make_lm_head(
            dp_parallel=True, vocab_shard=vocab_shard, hidden=hidden_dim,
            dp_pad_n=pad_n,
        )
        hidden = torch.randn(local_n, hidden_dim)

        dp_group = _make_dp_group(
            dp_size=dp_size,
            all_gather_fn=lambda t, dim=0: torch.cat([t, t], dim=dim),
            all_to_all_return=None,
        )
        with patch.dict("os.environ", {"OMNI_NPU_USE_DEVICE_COMM_A2A": "0"}), \
             patch(f"{MOD}.get_dp_group", return_value=dp_group), \
             patch("torch.distributed.all_to_all_single") as a2a_single:
            out = proc._get_logits(hidden, lm_head, None)

        # Padded to pad_n rows before all_gather
        gathered_input = dp_group.all_gather.call_args.args[0]
        assert gathered_input.shape == (pad_n, hidden_dim)
        torch.testing.assert_close(
            gathered_input[local_n:],
            torch.zeros(pad_n - local_n, hidden_dim),
        )

        # all_to_all_single called once, with the dp device_group
        a2a_single.assert_called_once()
        assert a2a_single.call_args.kwargs["group"] is dp_group.device_group
        # device_communicator.all_to_all NOT called on the default path
        dp_group.device_communicator.all_to_all.assert_not_called()

        # Output trimmed back to local_n rows
        assert out.shape == (local_n, vocab_full)

    def test_device_comm_a2a_path_when_env_enabled(self):
        # Fallback path: get_dp_group().device_communicator.all_to_all
        dp_size, local_n, pad_n = 2, 3, 5
        hidden_dim, vocab_shard = 8, 16
        vocab_full = vocab_shard * dp_size

        proc = _make_processor(org_vocab_size=vocab_full)
        lm_head = _make_lm_head(
            dp_parallel=True, vocab_shard=vocab_shard, hidden=hidden_dim,
            dp_pad_n=pad_n,
        )
        hidden = torch.randn(local_n, hidden_dim)
        a2a_out = torch.randn(pad_n, vocab_full)

        dp_group = _make_dp_group(
            dp_size=dp_size,
            all_gather_fn=lambda t, dim=0: torch.cat([t, t], dim=dim),
            all_to_all_return=a2a_out,
        )
        with patch.dict("os.environ", {"OMNI_NPU_USE_DEVICE_COMM_A2A": "1"}), \
             patch(f"{MOD}.get_dp_group", return_value=dp_group), \
             patch("torch.distributed.all_to_all_single") as a2a_single:
            out = proc._get_logits(hidden, lm_head, None)

        # all_to_all called with canonical dims
        a2a = dp_group.device_communicator.all_to_all
        a2a.assert_called_once()
        assert a2a.call_args.kwargs["scatter_dim"] == 0
        assert a2a.call_args.kwargs["gather_dim"] == -1
        # all_to_all_single NOT called on the fallback path
        a2a_single.assert_not_called()

        # Output trimmed back to local_n rows
        assert out.shape == (local_n, vocab_full)

    def test_no_pad_when_local_equals_pad_target(self):
        dp_size, pad_n = 2, 4
        hidden_dim, vocab_shard = 8, 16
        v_full = vocab_shard * dp_size

        proc = _make_processor(org_vocab_size=v_full)
        lm_head = _make_lm_head(
            dp_parallel=True, vocab_shard=vocab_shard, hidden=hidden_dim,
            dp_pad_n=pad_n,
        )
        hidden = torch.randn(pad_n, hidden_dim)

        dp_group = _make_dp_group(
            dp_size=dp_size,
            all_gather_fn=lambda t, dim=0: torch.cat([t, t], dim=dim),
            all_to_all_return=None,
        )
        with patch.dict("os.environ", {"OMNI_NPU_USE_DEVICE_COMM_A2A": "0"}), \
             patch(f"{MOD}.get_dp_group", return_value=dp_group), \
             patch("torch.distributed.all_to_all_single"):
            out = proc._get_logits(hidden, lm_head, None)

        # all_gather fed exactly the raw input (no concat-with-zeros)
        gathered_input = dp_group.all_gather.call_args.args[0]
        assert gathered_input.shape == (pad_n, hidden_dim)
        torch.testing.assert_close(gathered_input, hidden)
        # No trim (local_n == pad_n)
        assert out.shape == (pad_n, v_full)


class TestNPULogitsProcessorLocal:
    def test_uses_local_world_group_instead_of_dp_group(self):
        local_size, local_n, pad_n = 4, 3, 5
        hidden_dim, vocab_shard = 8, 16
        vocab_full = vocab_shard * local_size

        proc = _make_processor(org_vocab_size=vocab_full)
        lm_head = _make_lm_head(
            dp_parallel=False,
            local_lmhead_parallel=True,
            vocab_shard=vocab_shard,
            hidden=hidden_dim,
            dp_pad_n=pad_n,
        )
        hidden = torch.randn(local_n, hidden_dim)

        local_group = _make_dp_group(
            dp_size=local_size,
            all_gather_fn=lambda t, dim=0: torch.cat([t] * local_size, dim=dim),
            all_to_all_return=None,
        )
        with patch.dict("os.environ", {"OMNI_NPU_USE_DEVICE_COMM_A2A": "0"}), \
             patch(f"{MOD}.get_local_world_group", return_value=local_group), \
             patch(f"{MOD}.get_dp_group") as get_dp, \
             patch("torch.distributed.all_to_all_single") as a2a_single:
            out = proc._get_logits(hidden, lm_head, None)

        get_dp.assert_not_called()
        local_group.all_gather.assert_called_once()
        a2a_single.assert_called_once()
        assert a2a_single.call_args.kwargs["group"] is local_group.device_group
        assert out.shape == (local_n, vocab_full)

    def test_pads_and_trims_with_local_comm_group(self):
        local_size, local_n, pad_n = 4, 2, 6
        hidden_dim, vocab_shard = 8, 16
        vocab_full = vocab_shard * local_size

        proc = _make_processor(org_vocab_size=vocab_full)
        lm_head = _make_lm_head(
            dp_parallel=False,
            local_lmhead_parallel=True,
            vocab_shard=vocab_shard,
            hidden=hidden_dim,
            dp_pad_n=pad_n,
        )
        hidden = torch.randn(local_n, hidden_dim)

        local_group = _make_dp_group(
            dp_size=local_size,
            all_gather_fn=lambda t, dim=0: torch.cat([t] * local_size, dim=dim),
            all_to_all_return=None,
        )
        with patch.dict("os.environ", {"OMNI_NPU_USE_DEVICE_COMM_A2A": "0"}), \
             patch(f"{MOD}.get_local_world_group", return_value=local_group), \
             patch("torch.distributed.all_to_all_single"):
            out = proc._get_logits(hidden, lm_head, None)

        gathered_input = local_group.all_gather.call_args.args[0]
        assert gathered_input.shape == (pad_n, hidden_dim)
        torch.testing.assert_close(
            gathered_input[local_n:],
            torch.zeros(pad_n - local_n, hidden_dim),
        )
        assert out.shape == (local_n, vocab_full)
