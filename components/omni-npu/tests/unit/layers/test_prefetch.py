# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


class _TensorIs:
    """Matcher for unittest.mock that compares tensors by identity (is), not equality (==)."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __eq__(self, other):
        return other is self.tensor

    def __repr__(self):  # pragma: no cover
        return f"_TensorIs({type(self.tensor).__name__}@{id(self.tensor)})"


def _ensure_torch_npu_stubs():
    """
    Ensure `torch_npu` exists so we can import the module under test in a
    CPU-only CI environment.
    """
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # pragma: no cover
        # Keep unit suite green in minimal environments; in real CI this should
        # be installed so assertions run.
        raise unittest.SkipTest("PyTorch (`torch`) is not installed; skipping prefetch unit tests.")

    import torch  # type: ignore[import-not-found]  # noqa: E402

    # Provide a fake `torch_npu` module if it isn't installed.
    try:
        import torch_npu  # type: ignore  # noqa: F401
        return sys.modules["torch_npu"]
    except Exception:
        # Minimal stub to allow importing omni_npu modules in CPU-only CI.
        fake_mod = types.ModuleType("torch_npu")
        fake_mod.npu = types.SimpleNamespace()

        def _npu_prefetch(*_args, **_kwargs):
            return None

        fake_mod.npu_prefetch = _npu_prefetch  # type: ignore[attr-defined]
        sys.modules["torch_npu"] = fake_mod
        return fake_mod


def _import_prefetch_module():
    """
    Import the prefetch module in a way that works both:
    - when the package is installed (import path `omni_npu...`)
    - when running from source tree without installation (`src.omni_npu...`)
    """
    _ensure_torch_npu_stubs()
    try:
        return importlib.import_module("omni_npu.layers.prefetch")
    except Exception:
        return importlib.import_module("src.omni_npu.layers.prefetch")


def _reload_prefetch_module():
    """
    Reload the module under test so module-level state resets between test cases.
    """
    mod = _import_prefetch_module()
    return importlib.reload(mod)


class TestPrefetchWeight(unittest.TestCase):
    def setUp(self):
        _ensure_torch_npu_stubs()
        self.prefetch = _reload_prefetch_module()

        import torch  # type: ignore[import-not-found]  # noqa: E402

        self.weight = torch.randn(2, 3)
        self.trigger = torch.randn(1)
        self.pm = self.prefetch.PrefetchManager()

    def test_prefetch_weight_noop_on_none_inputs_or_non_positive_prefetch(self):
        with mock.patch("torch_npu.npu_prefetch", autospec=True) as npu_prefetch:
            self.pm.prefetch_weight(None, self.trigger, 16)
            self.pm.prefetch_weight(self.weight, None, 16)
            self.pm.prefetch_weight(self.weight, self.trigger, None)
            self.pm.prefetch_weight(self.weight, self.trigger, 0)
            self.pm.prefetch_weight(self.weight, self.trigger, -1)

            npu_prefetch.assert_not_called()

    def test_prefetch_weight_calls_torch_npu_prefetch(self):
        with mock.patch("torch_npu.npu_prefetch", autospec=True) as npu_prefetch:
            self.pm.prefetch_weight(self.weight, self.trigger, 128)
            npu_prefetch.assert_called_once_with(
                _TensorIs(self.weight),
                _TensorIs(self.trigger),
                128 * self.prefetch.PREFETCH_UNIT_SIZE,
            )


class TestPrefetchManagerMethods(unittest.TestCase):

    def setUp(self):
        _ensure_torch_npu_stubs()
        self.prefetch = _reload_prefetch_module()
        self.operator_opt = self.prefetch.model_extra_config.operator_opt_config

        import torch  # type: ignore[import-not-found]  # noqa: E402

        self.trigger = torch.randn(1)
        self.pm = self.prefetch.PrefetchManager()

    def test_prefetch_noop_when_layer_none_or_unknown_group(self):
        with mock.patch.object(self.pm, "prefetch_weight") as m:
            self.pm.prefetch("next_attn", self.trigger, layer=None)
            self.pm.prefetch("unknown_group", self.trigger, layer=object())
            m.assert_not_called()

    def test_prefetch_next_attn_calls_prefetch_weight_for_next_layer_linears(self):
        import torch  # type: ignore[import-not-found]  # noqa: E402

        wq = torch.randn(1, 1)
        next_attn = SimpleNamespace(q_a_proj=SimpleNamespace(weight=wq))
        next_layer = SimpleNamespace(self_attn=next_attn)
        moe = object()
        self.prefetch._prefetch_next_decoder_layer.clear()
        self.prefetch._attn_prefetch_linear.clear()
        self.prefetch._prefetch_next_decoder_layer[moe] = next_layer
        self.prefetch._attn_prefetch_linear.append("q_a_proj")

        with mock.patch.object(self.pm, "prefetch_weight") as pw:
            self.pm.prefetch("next_attn", self.trigger, layer=moe)
            pw.assert_called_once_with(
                _TensorIs(wq), _TensorIs(self.trigger), self.operator_opt.attn_prefetch
            )

    def test_prefetch_moe_calls_prefetch_weight_for_routed_and_shared(self):
        import torch  # type: ignore[import-not-found]  # noqa: E402

        layer = SimpleNamespace(
            w13_weight=torch.randn(2, 2),
            w2_weight=torch.randn(2, 2),
            shared_experts=SimpleNamespace(
                gate_up_proj=SimpleNamespace(weight=torch.randn(1)),
                down_proj=SimpleNamespace(weight=torch.randn(1)),
            ),
        )
        with mock.patch.object(self.pm, "prefetch_weight") as pw:
            self.pm.prefetch("moe", self.trigger, layer=layer)
            self.assertEqual(pw.call_count, 4)
            pw.assert_any_call(
                _TensorIs(layer.w13_weight),
                _TensorIs(self.trigger),
                self.operator_opt.expert_gate_up_prefetch,
            )
            pw.assert_any_call(
                _TensorIs(layer.w2_weight),
                _TensorIs(self.trigger),
                self.operator_opt.expert_down_prefetch,
            )
            pw.assert_any_call(
                _TensorIs(layer.shared_experts.gate_up_proj.weight),
                _TensorIs(self.trigger),
                self.operator_opt.shared_expert_gate_up_prefetch,
            )
            pw.assert_any_call(
                _TensorIs(layer.shared_experts.down_proj.weight),
                _TensorIs(self.trigger),
                self.operator_opt.shared_expert_down_prefetch,
            )


class TestSetupPrefetchForModel(unittest.TestCase):

    def setUp(self):
        _ensure_torch_npu_stubs()

    def test_setup_disabled_prefetch_returns_without_touching_globals(self):
        opt = SimpleNamespace(enable_prefetch=False, attn_prefetch=8)
        prefetch = _reload_prefetch_module()
        with mock.patch.object(prefetch, "model_extra_config", SimpleNamespace(operator_opt_config=opt)):
            prefetch._prefetch_next_decoder_layer[object()] = object()
            prefetch._attn_prefetch_linear.append("q_proj")
            prefetch.setup_prefetch_for_model(SimpleNamespace(layers=[]))
            self.assertEqual(len(prefetch._prefetch_next_decoder_layer), 1)
            self.assertEqual(prefetch._attn_prefetch_linear, ["q_proj"])

    def test_setup_clears_and_registers_moe_to_next_layer(self):
        opt = SimpleNamespace(enable_prefetch=True, attn_prefetch=8)
        prefetch = _reload_prefetch_module()
        with mock.patch.object(prefetch, "model_extra_config", SimpleNamespace(operator_opt_config=opt)):
            prefetch._prefetch_next_decoder_layer[object()] = object()

            moe = object()
            layer0 = SimpleNamespace(
                self_attn=SimpleNamespace(q_a_proj=object()),
                mlp=SimpleNamespace(experts=moe),
            )
            layer1 = SimpleNamespace()
            model = SimpleNamespace(layers=[layer0, layer1])

            prefetch.setup_prefetch_for_model(model)
            self.assertEqual(prefetch._prefetch_next_decoder_layer, {moe: layer1})
            self.assertIn("q_a_proj", prefetch._attn_prefetch_linear)

    def test_setup_attn_prefetch_non_positive_skips_after_clear(self):
        opt = SimpleNamespace(enable_prefetch=True, attn_prefetch=0)
        prefetch = _reload_prefetch_module()
        with mock.patch.object(prefetch, "model_extra_config", SimpleNamespace(operator_opt_config=opt)):
            moe = object()
            layer0 = SimpleNamespace(
                self_attn=SimpleNamespace(q_proj=object()),
                mlp=SimpleNamespace(experts=moe),
            )
            model = SimpleNamespace(layers=[layer0, SimpleNamespace()])
            prefetch.setup_prefetch_for_model(model)
            self.assertEqual(prefetch._attn_prefetch_linear, [])
            self.assertEqual(prefetch._prefetch_next_decoder_layer, {})


if __name__ == "__main__":
    unittest.main()
