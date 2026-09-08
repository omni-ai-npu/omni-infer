# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
PanguBackend.layer_specs parsed against a real 92B config.

Fixture: tests/fixtures/pangu_92B/config.json (DSA, 46 layers, 256 experts,
first_k_dense_replace=2, MHC=4, MoME on). No weights needed.
"""
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from jointfix.backends.pangu import PanguBackend
from jointfix.core.stats import StatsConfig

_FIXTURE = Path(__file__).parent / "fixtures" / "pangu_92B"


def _backend():
    return PanguBackend(str(_FIXTURE))


def test_layer_count():
    specs = _backend().layer_specs()
    assert len(specs) == 46


def test_dense_then_moe_split():
    specs = _backend().layer_specs()
    # first_k_dense_replace = 2 -> layers 0,1 dense, 2..45 MoE
    assert specs[0].is_moe is False
    assert specs[1].is_moe is False
    assert specs[2].is_moe is True
    assert specs[45].is_moe is True


def test_dsa_flag_in_extra():
    specs = _backend().layer_specs()
    # dsa_layers starts [0, 3, 6, 9, ...]
    assert specs[0].extra["is_dsa"] is True
    assert specs[3].extra["is_dsa"] is True
    assert specs[1].extra["is_dsa"] is False


def test_hidden_size_and_mhc():
    specs = _backend().layer_specs()
    assert specs[0].hidden_size == 2560
    assert specs[0].extra["mhc_num_stream"] == 4
    # original PanguTorchLayerSpec preserved for build_layer
    assert specs[0].extra["_pangu_spec"].layer_idx == 0


def test_skip_patterns_compose_universal_plus_pangu():
    sp = _backend().skip_patterns()
    assert "embed" in sp                # universal
    assert "mlp.gate." in sp            # universal (router)
    assert "indexer.wk" in sp           # pangu-specific
    assert "mhc_module.phi" in sp       # pangu-specific
    assert "model.layers.46." not in sp  # legacy JointFix behavior is unchanged
    assert "model.layers.48." not in sp


def test_mtp_skip_patterns_are_separate_and_dynamic():
    assert _backend().mtp_skip_patterns() == [
        "model.layers.46.", "model.layers.47.", "model.layers.48.",
    ]


def test_load_layer_weights_reads_only_target_layer(tmp_path):
    """Shard-read logic, validated on a synthetic 2-layer safetensors (no real model)."""
    shard = "model-00001.safetensors"
    tensors = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4),
        "model.layers.0.mlp.gate.weight": torch.randn(4, 4),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(4, 4),
        "model.embed_tokens.weight": torch.randn(8, 4),
    }
    save_file(tensors, str(tmp_path / shard))
    index = {"weight_map": {k: shard for k in tensors}}
    import json
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    got = PanguBackend(str(tmp_path)).load_layer_weights(0)
    assert set(got) == {
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate.weight",
    }
    assert got["model.layers.0.self_attn.o_proj.weight"].shape == (4, 4)


# ── install_stat_hooks on a synthetic Pangu-like MoE layer ────────────────────
class _Expert(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.down_proj = nn.Linear(h, h)

    def forward(self, x):
        return self.down_proj(x)


class _MoE(nn.Module):
    def __init__(self, h, n_exp, routed):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(h) for _ in range(n_exp)])
        self._routed = routed   # which experts receive tokens (simulated top-k)

    def forward(self, x):
        return sum(self.experts[e](x) for e in self._routed)


class _Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q_b_proj = nn.Linear(h, h)
        self.o_proj = nn.Linear(h, h)

    def forward(self, x):
        return self.o_proj(self.q_b_proj(x))


class _Layer(nn.Module):
    def __init__(self, h, n_exp, routed):
        super().__init__()
        self.self_attn = _Attn(h)
        self.mlp = _MoE(h, n_exp, routed)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


def test_install_stat_hooks_collects_per_site_and_per_routed_expert():
    h, n_exp = 8, 4
    layer = _Layer(h, n_exp, routed=[0, 2])   # only experts 0 and 2 get tokens
    collectors = {}
    backend = PanguBackend(".")               # model_dir unused by this method
    handles = backend.install_stat_hooks(layer, 0, collectors, StatsConfig())

    with torch.no_grad():
        layer(torch.randn(1, 6, h))

    keys = set(collectors)
    assert "model.layers.0.q_b_in" in keys
    assert "model.layers.0.o_in" in keys
    assert "model.layers.0.mlp_in" in keys
    # per-expert: only routed experts created a collector
    assert "model.layers.0.exp0_down_in" in keys
    assert "model.layers.0.exp2_down_in" in keys
    assert "model.layers.0.exp1_down_in" not in keys   # expert 1 got no tokens
    assert "model.layers.0.exp3_down_in" not in keys

    for hk in handles:
        hk.remove()
