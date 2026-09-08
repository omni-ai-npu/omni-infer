# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Runner orchestration test — mock backend + method, no real model.

Validates the loop wiring (build -> hook -> forward -> process -> save -> dequant
re-forward -> propagate -> checkpoint) end-to-end, independent of Pangu/smooth.
"""
import torch
import torch.nn as nn
import pytest

from jointfix.backends.base import LayerSpec, ModelBackend
from jointfix.core.primitives import rtn_quantize
from jointfix.core.runner import (
    RunConfig, _forward_collect, _split_chunks, calibrate_and_quantize,
)
from jointfix.core.stats import AccumActStats, StatsConfig
from jointfix.methods.base import QuantMethod

H = 4


class _FakeLayer(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(w)

    def forward(self, x):
        return self.lin(x)   # [*, H] -> [*, H], shape-stable for propagation


class _FakeBackend(ModelBackend):
    def __init__(self, n_layers=2):
        super().__init__("/fake")
        self.n_layers = n_layers
        self.saved = []          # (layer_idx, sorted tensor names)
        self.forward_calls = 0

    def layer_specs(self):
        return [LayerSpec(i, is_moe=False, hidden_size=H) for i in range(self.n_layers)]

    def skip_patterns(self):
        return []

    def config(self):
        return {}

    def weight_map(self):
        return {}

    def load_layer_weights(self, layer_idx):
        torch.manual_seed(layer_idx)
        return {f"model.layers.{layer_idx}.lin.weight": torch.randn(H, H)}

    def save_quantized(self, out_dir, layer_idx, tensors, quant_meta):
        self.saved.append((layer_idx, sorted(tensors)))

    def embed(self, input_ids, device):
        torch.manual_seed(99)
        return torch.randn(input_ids.shape[0], input_ids.shape[1], H, device=device)

    def build_layer(self, spec, weights, device):
        w = weights[f"model.layers.{spec.layer_idx}.lin.weight"].float()
        return _FakeLayer(w).to(device)

    def install_stat_hooks(self, layer, layer_idx, collectors, stats_config):
        pfx = f"model.layers.{layer_idx}"

        def hook(_mod, inputs):
            x = inputs[0]
            key = f"{pfx}.lin_in"
            if key not in collectors:
                collectors[key] = AccumActStats(x.shape[-1], stats_config)
            collectors[key].update(x)

        return [layer.lin.register_forward_pre_hook(hook)]


class _FakeMethod(QuantMethod):
    name = "fake"
    needs_activations = True

    def add_cli_args(self, parser):
        pass

    def stats_config(self):
        return StatsConfig(sample_limit=8, tokens_per_sample=2)

    def process_layer(self, layer_tensors, collectors, spec, backend, device, devices=None):
        # collectors must have been populated by the calibration forward
        assert any(k.endswith("lin_in") for k in collectors), "no stats collected"
        out = {}
        for name, w in layer_tensors.items():
            q, s = rtn_quantize(w.float())
            out[name] = q
            out[name.replace(".weight", ".weight_scale")] = s
        return out


def test_runner_full_loop(tmp_path):
    backend = _FakeBackend(n_layers=3)
    method = _FakeMethod()
    cfg = RunConfig(model_dir="/fake", output_dir=str(tmp_path), calib_data="/fake",
                    n_samples=2, seq_len=3, device="cpu", num_devices=1)
    input_ids = torch.randint(0, 5, (2, 3))

    calibrate_and_quantize(backend, method, cfg, input_ids=input_ids)

    # every layer was processed + saved, in order
    assert [li for li, _ in backend.saved] == [0, 1, 2]
    # each save carries the int8 weight + its scale
    _, names = backend.saved[0]
    assert "model.layers.0.lin.weight" in names
    assert "model.layers.0.lin.weight_scale" in names
    # checkpoint written
    assert (tmp_path / ".quantize_checkpoint.json").exists()


def test_text_path_reasserts_npu_compile_mode_after_tokenizer(tmp_path, monkeypatch):
    """Preserve legacy order: tokenizer load -> compile mode -> embedding."""
    import types
    import jointfix.core.runner as runner

    events = []
    backend = _FakeBackend(n_layers=0)

    def embed(input_ids, _device):
        events.append("embed")
        return torch.randn(input_ids.shape[0], input_ids.shape[1], H)

    backend.embed = embed
    monkeypatch.setattr(
        runner, "resolve_devices",
        lambda _name, _count: [types.SimpleNamespace(type="npu")],
    )
    monkeypatch.setattr(
        runner, "load_calibration_data",
        lambda *_args, **_kwargs: events.append("load") or torch.ones(2, 3, dtype=torch.long),
    )
    monkeypatch.setattr(
        runner, "set_npu_compile_mode", lambda: events.append("set_compile_mode"),
    )

    cfg = RunConfig(
        model_dir="/fake", output_dir=str(tmp_path), calib_data="calib.parquet",
        n_samples=2, seq_len=3, device="npu", num_devices=1,
    )
    calibrate_and_quantize(backend, _FakeMethod(), cfg)

    assert events == ["load", "set_compile_mode", "embed"]


def test_runner_rejects_cpu_multidevice(tmp_path):
    # multi-device requires cuda/npu; cpu is a single device
    backend = _FakeBackend()
    cfg = RunConfig(model_dir="/fake", output_dir=str(tmp_path), calib_data="/fake",
                    device="cpu", num_devices=2)
    with pytest.raises(ValueError):
        calibrate_and_quantize(backend, _FakeMethod(), cfg,
                               input_ids=torch.randint(0, 5, (1, 2)))


# ── multi-device forward + collector merge ────────────────────────────────────
def test_split_chunks():
    assert _split_chunks(8, 1) == [(0, 8)]
    assert _split_chunks(8, 2) == [(0, 4), (4, 8)]
    assert _split_chunks(5, 2) == [(0, 3), (3, 5)]      # earlier device gets the remainder
    assert _split_chunks(8, 3) == [(0, 3), (3, 6), (6, 8)]


def test_forward_collect_merge_matches_single():
    """Stats merged across 2 (mock) devices == single-device stats."""
    backend = _FakeBackend()
    spec = backend.layer_specs()[0]
    weights = backend.load_layer_weights(0)
    torch.manual_seed(5)
    hidden = torch.randn(4, 3, H)
    sc = StatsConfig(sample_limit=16, tokens_per_sample=3)
    cpu = torch.device("cpu")

    _, c1 = _forward_collect(backend, spec, weights, hidden, [cpu], sc, collect=True)
    _, c2 = _forward_collect(backend, spec, weights, hidden, [cpu, cpu], sc, collect=True)

    assert set(c1) == set(c2)
    k = next(iter(c1))
    assert c1[k].count == c2[k].count                       # same token count
    assert torch.allclose(c1[k].amax, c2[k].amax, atol=1e-5)
    assert torch.allclose(c1[k].sum_x2, c2[k].sum_x2, atol=1e-3)


def test_forward_collect_propagates_shape():
    backend = _FakeBackend()
    spec = backend.layer_specs()[0]
    weights = backend.load_layer_weights(0)
    hidden = torch.randn(4, 3, H)
    cpu = torch.device("cpu")
    out, _ = _forward_collect(backend, spec, weights, hidden, [cpu, cpu],
                              StatsConfig(), collect=False)
    assert out.shape == hidden.shape   # 2 chunks concatenated back to [4, 3, H]


def test_forward_collect_supports_ragged_multimodal_samples():
    """Real multimodal prompts keep their lengths; no EOS padding enters stats."""
    backend = _FakeBackend()
    spec = backend.layer_specs()[0]
    weights = backend.load_layer_weights(0)
    hidden = [torch.randn(1, 2, H), torch.randn(1, 5, H), torch.randn(1, 3, H)]
    cpu = torch.device("cpu")

    out, collectors = _forward_collect(
        backend, spec, weights, hidden, [cpu, cpu],
        StatsConfig(sample_limit=16, tokens_per_sample=3), collect=True,
    )

    assert [x.shape for x in out] == [torch.Size((1, 2, H)),
                                      torch.Size((1, 5, H)),
                                      torch.Size((1, 3, H))]
    key = next(iter(collectors))
    assert collectors[key].count == 2 + 5 + 3
