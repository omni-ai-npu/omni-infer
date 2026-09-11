# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""MoME context boundaries: phase selection, state updates and dynamic FX."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import omni_npu.v1.layers.attention.npu_pangu as pangu_mod
import torch
from omni_npu.v1.layers.attention import npu_pangu_custom_ops as ops


def _metadata(num_reqs, *, prefill=False, apc=True, num_tokens=None):
    return SimpleNamespace(
        num_prefills=num_reqs if prefill else 0,
        num_decodes=0 if prefill else num_reqs,
        num_actual_tokens=num_reqs if num_tokens is None else num_tokens,
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        cache_indices=torch.zeros(
            (num_reqs, 2) if apc else (num_reqs,), dtype=torch.int32
        ),
        num_accepted_tokens=torch.ones(num_reqs, dtype=torch.int32)
        if not prefill
        else None,
        num_computed_tokens=torch.zeros(num_reqs, dtype=torch.int32),
        block_idx_first_scheduled_token=torch.zeros(num_reqs, dtype=torch.int32)
        if apc
        else None,
        block_idx_last_scheduled_token=torch.zeros(num_reqs, dtype=torch.int32)
        if apc
        else None,
        block_idx_last_computed_token=torch.zeros(num_reqs, dtype=torch.int32)
        if apc
        else None,
        pad_slot_id=-1,
        max_query_len=4,
        B_size=128,
    )


def _context(metadata):
    return SimpleNamespace(
        no_compile_layers={"layer": SimpleNamespace(prefix="model.layers.0.self_attn")},
        attn_metadata={"model.layers.0.self_attn.mome": metadata},
    )


class TestMomeContextOps(unittest.TestCase):
    def test_live_metadata_and_optional_fields_reach_kernel(self):
        context = _context(None)
        x, weight, states = torch.zeros(4, 8), torch.ones(3, 8), torch.zeros(16, 8)
        result = torch.ones_like(x)
        with (
            patch.object(ops, "get_forward_context", return_value=context),
            patch(
                "torch.ops.custom.npu_ai_infra_fused_causal_conv1d",
                create=True,
                return_value=result,
            ) as kernel,
        ):
            for num_reqs, prefill, apc in (
                (16, False, True),
                (4, False, False),
                (2, True, True),
            ):
                with self.subTest(num_reqs=num_reqs, prefill=prefill, apc=apc):
                    metadata = _metadata(num_reqs, prefill=prefill, apc=apc)
                    context.attn_metadata["model.layers.0.self_attn.mome"] = metadata
                    actual = ops.npu_pangu_mome_conv_from_context(
                        x,
                        weight,
                        states,
                        "layer",
                        "prefill" if prefill else "decode",
                    )
                    self.assertIs(actual, result)
                    args, kwargs = kernel.call_args
                    self.assertIs(args[0], x)
                    self.assertIs(args[1], weight)
                    self.assertIs(args[2], states)
                    for name in (
                        "query_start_loc",
                        "cache_indices",
                        "num_accepted_tokens",
                        "num_computed_tokens",
                        "block_idx_first_scheduled_token",
                        "block_idx_last_scheduled_token",
                    ):
                        self.assertIs(kwargs[name], getattr(metadata, name))
                    self.assertIs(
                        kwargs["initial_state_idx"],
                        metadata.block_idx_last_computed_token,
                    )
                    self.assertEqual(kwargs["block_size"], metadata.B_size)
                    self.assertEqual(kwargs["max_query_len"], metadata.max_query_len)
                    self.assertEqual(kwargs["pad_slot_id"], -1)
                    self.assertFalse(kwargs["inplace"])

    def test_mixed_batch_selects_explicit_phase(self):
        prefill, decode = _metadata(2, prefill=True), _metadata(4)
        metadata = SimpleNamespace(
            num_prefills=2, num_decodes=4, prefill=prefill, decode=decode
        )
        context = _context(metadata)
        # The MLA object may already have been restored by traced Python code.
        context.attn_metadata["model.layers.0.self_attn.attn"] = SimpleNamespace(
            prefill=object(), decode=object()
        )
        with patch.object(ops, "get_forward_context", return_value=context):
            self.assertIs(ops._lookup_mome_metadata("layer", "decode"), decode)
            self.assertIs(ops._lookup_mome_metadata("layer", "prefill"), prefill)
            with self.assertRaisesRegex(ValueError, "Unsupported MoME phase"):
                ops._lookup_mome_metadata("layer", "mixed")

    def test_kv_inplace_updates_only_valid_compressed_slice(self):
        kv = torch.arange(48, dtype=torch.float32).reshape(6, 8).clone()
        before = kv.clone()
        weight, states = torch.ones(3, 4), torch.zeros(16, 4)
        metadata = _metadata(2, num_tokens=2)
        mixed = SimpleNamespace(
            num_prefills=1,
            num_decodes=2,
            prefill=_metadata(1, prefill=True, num_tokens=4),
            decode=metadata,
        )

        def kernel(x, _weight, conv_states, **kwargs):
            self.assertEqual(x.shape, (2, 4))
            self.assertTrue(kwargs["inplace"])
            self.assertIs(kwargs["query_start_loc"], metadata.query_start_loc)
            x.add_(10)
            conv_states.add_(1)
            return x

        with (
            patch.object(ops, "get_forward_context", return_value=_context(mixed)),
            patch(
                "torch.ops.custom.npu_ai_infra_fused_causal_conv1d",
                create=True,
                side_effect=kernel,
            ),
        ):
            result = ops.npu_pangu_kv_down_mome_inplace_from_context(
                kv, weight, states, "layer", "decode", 4
            )
        self.assertIs(result, kv)
        torch.testing.assert_close(kv[:2, :4], before[:2, :4] + 10)
        torch.testing.assert_close(kv[2:], before[2:])
        torch.testing.assert_close(kv[:, 4:], before[:, 4:])
        torch.testing.assert_close(states, torch.ones_like(states))

    def test_fake_preserves_symbolic_tokens_without_context(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        mode = FakeTensorMode(shape_env=ShapeEnv())
        x = mode.from_tensor(torch.zeros(16, 8), static_shapes=False)
        weight = mode.from_tensor(torch.zeros(3, 8), static_shapes=True)
        states = mode.from_tensor(torch.zeros(16, 8), static_shapes=True)
        with (
            mode,
            patch.object(
                ops,
                "get_forward_context",
                side_effect=AssertionError("fake read context"),
            ),
        ):
            result = ops.npu_pangu_mome_conv_from_context_fake(
                x, weight, states, "layer", "decode"
            )
            self.assertIsInstance(result.shape[0], torch.SymInt)
            self.assertEqual(str(result.shape[0]), str(x.shape[0]))
            self.assertEqual(result.dtype, x.dtype)
            self.assertEqual(result.device, x.device)
            alias = ops.npu_pangu_kv_down_mome_inplace_from_context_fake(
                x, weight, states, "layer", "decode", 4
            )
            self.assertIs(alias, x)

    def test_one_fx_graph_accepts_changing_request_metadata(self):
        # Register the production body/fake on CPU; only the device kernel is
        # mocked. This verifies the opaque boundary, not ACLGraph replay.
        with torch.library._scoped_library("mome_context_ut", "FRAGMENT") as lib:
            lib.define(
                "conv(Tensor x, Tensor weight, Tensor conv_states, str layer_name, str phase) -> Tensor"
            )
            lib.impl("conv", ops.npu_pangu_mome_conv_from_context, "CPU")
            lib._register_fake("conv", ops.npu_pangu_mome_conv_from_context_fake)
            context, graphs = _context(None), []
            weight, states = torch.ones(3, 8), torch.zeros(16, 8)
            attention = pangu_mod.NPUPanguSparseAttention.__new__(
                pangu_mod.NPUPanguSparseAttention
            )
            torch.nn.Module.__init__(attention)
            attention.layer_name = "layer"
            attention.on_ascend950 = False
            attention.mome_attn = SimpleNamespace(kv_cache=[states])
            conv = SimpleNamespace(mome_cache_index=0, weight=weight)
            attn_metadata = SimpleNamespace(prefill=None)

            def backend(graph, _inputs):
                graphs.append(graph)
                return graph.forward

            def forward(x):
                metadata = context.attn_metadata["model.layers.0.self_attn.mome"]
                return attention._apply_MOME(x, conv, attn_metadata, metadata) * 2

            def kernel(x, _weight, conv_states, **kwargs):
                conv_states.add_(1)
                return x + kwargs["query_start_loc"].numel()

            torch._dynamo.reset()
            try:
                compiled = torch.compile(
                    forward, backend=backend, dynamic=True, fullgraph=True
                )
                with (
                    patch.object(ops, "get_forward_context", return_value=context),
                    patch(
                        "torch.ops.custom.npu_ai_infra_fused_causal_conv1d",
                        create=True,
                        side_effect=kernel,
                    ),
                    patch(
                        "torch.ops.vllm.npu_pangu_mome_conv_from_context",
                        torch.ops.mome_context_ut.conv,
                    ),
                ):
                    for tokens, reqs in ((64, 16), (4, 4), (16, 16), (8, 8), (64, 16)):
                        context.attn_metadata["model.layers.0.self_attn.mome"] = (
                            _metadata(reqs)
                        )
                        x = torch.ones(tokens, 8)
                        torch.testing.assert_close(compiled(x), (x + reqs + 1) * 2)
                self.assertEqual(len(graphs), 1)
                placeholders = [
                    n for n in graphs[0].graph.nodes if n.op == "placeholder"
                ]
                tensor_inputs = [
                    n
                    for n in placeholders
                    if isinstance(n.meta.get("example_value"), torch.Tensor)
                ]
                self.assertEqual(len(tensor_inputs), 3)  # x, weight, states only
                torch.testing.assert_close(states, torch.full_like(states, 5))
            finally:
                torch._dynamo.reset()


class TestMomeContextCallSites(unittest.TestCase):
    def setUp(self):
        self.attention = pangu_mod.NPUPanguSparseAttention.__new__(
            pangu_mod.NPUPanguSparseAttention
        )
        torch.nn.Module.__init__(self.attention)
        self.attention.layer_name = "layer"
        self.attention.on_ascend950 = False
        self.attention.mome_attn = SimpleNamespace(
            kv_cache=[torch.zeros(16, 8) for _ in range(3)]
        )

    def test_q_kv_o_calls_do_not_read_metadata_tensors(self):
        x = torch.zeros(4, 8)
        with patch(
            "torch.ops.vllm.npu_pangu_mome_conv_from_context", return_value=x + 1
        ) as op:
            for index in range(3):
                for phase in ("prefill", "decode"):
                    conv = SimpleNamespace(
                        mome_cache_index=index, weight=torch.ones(3, 8)
                    )
                    result = self.attention._apply_MOME(
                        x,
                        conv,
                        SimpleNamespace(
                            prefill=object() if phase == "prefill" else None
                        ),
                        object(),
                    )
                    self.assertIs(result, op.return_value)
                    op.assert_called_with(
                        x,
                        conv.weight,
                        self.attention.mome_attn.kv_cache[index],
                        "layer",
                        phase=phase,
                    )

    def test_warmup_and_ascend950_keep_original_paths(self):
        x = torch.zeros(4, 8)
        conv = SimpleNamespace(
            mome_cache_index=0, forward=MagicMock(return_value=x + 1)
        )
        self.assertIs(self.attention._apply_MOME(x, conv, None, None), x)
        self.attention.on_ascend950 = True
        metadata = object()
        result = self.attention._apply_MOME(x, conv, object(), metadata)
        self.assertIs(result, conv.forward.return_value)
        conv.forward.assert_called_once_with(
            x, self.attention.mome_attn.kv_cache[0], metadata, inplace=False
        )

    def test_prefill_inplace_keeps_legacy_view_contract(self):
        x = torch.zeros(4, 12)[:, :8]
        conv = SimpleNamespace(mome_cache_index=1, weight=torch.ones(3, 8))
        metadata = _metadata(2, prefill=True)
        with patch("torch.ops.vllm.npu_pangu_mome_conv", return_value=x) as op:
            result = self.attention._apply_MOME(x, conv, object(), metadata, inplace=True)
        self.assertIs(result, x)
        self.assertIs(op.call_args.args[0], x)
        self.assertIs(op.call_args.args[3], metadata.query_start_loc)
        self.assertTrue(op.call_args.kwargs["inplace"])

    def test_sp_keeps_explicit_metadata_path(self):
        x = torch.zeros(4, 8)
        conv = SimpleNamespace(mome_cache_index=0, weight=torch.ones(3, 8))
        metadata = SimpleNamespace(conv_sp_meta=(object(), object(), object()))
        with patch.object(pangu_mod, "conv_sp", return_value=x) as sp:
            result = self.attention._apply_MOME(x, conv, object(), metadata, True, True)
        self.assertIs(result, x)
        sp.assert_called_once_with(
            x,
            conv.weight,
            self.attention.mome_attn.kv_cache[0],
            *metadata.conv_sp_meta,
            True,
        )

    def test_kv_call_passes_whole_tensor_without_metadata_fields(self):
        kv = torch.zeros(4, 12)
        self.attention.kv_a_proj_with_mqa = MagicMock(return_value=kv)
        self.attention.use_mome = True
        self.attention.use_mome_inplace_update = True
        self.attention.kv_lora_rank = 8
        self.attention.compresskv_conv = SimpleNamespace(weight=torch.ones(3, 8))
        with patch(
            "torch.ops.vllm.npu_pangu_kv_down_mome_inplace_from_context",
            return_value=kv,
        ) as op:
            result = self.attention._kv_down_mome(
                torch.zeros(4, 8), SimpleNamespace(prefill=None), object()
            )
        self.assertIs(result, kv)
        op.assert_called_once_with(
            kv,
            self.attention.compresskv_conv.weight,
            self.attention.mome_attn.kv_cache[1],
            "layer",
            "decode",
            8,
        )


if __name__ == "__main__":
    unittest.main()
