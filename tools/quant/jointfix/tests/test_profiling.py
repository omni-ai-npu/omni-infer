# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
CPU unit tests for the opt-in search profiler and its analyzer.

The analyzer arithmetic (FLOPs, utilisation, tile sizing) is pure and hand-checkable,
so the whole Q1/Q2/Q3 pipeline is validated here without an NPU. The profiler itself is
checked for the contract that matters: zero output when JOINTFIX_PROFILE is unset.
"""
import json
import pathlib
import sys

import pytest
import torch

import jointfix.core.profiling as prof

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))
import analyze_profile as ap  # noqa: E402


# ── analyzer pure math (no NPU) ──────────────────────────────────────────────

def test_parse_prof_lines_extracts_json_and_skips_garbage():
    log = ('startup noise\n'
           '[PROF] {"kind": "layer", "layer": 0}\n'
           'prefix text [PROF] {"kind": "routed_down", "layer": 1}\n'
           'bad [PROF] {not valid json}\n')
    recs = ap.parse_prof_lines(log)
    assert len(recs) == 2
    assert recs[0]["kind"] == "layer"
    assert recs[1]["layer"] == 1


def test_quantile_nearest_rank():
    assert ap._quantile([1, 2, 3, 4, 5], 0.5) == 3
    assert ap._quantile([10], 0.5) == 10


def test_layer_flop_counts_gemm_over_experts_and_candidates():
    # each expert: 2*tok*h_in*h_out * (n_candidates + 1 Y_ref build)
    rec = {"h_in": 2, "h_out": 3, "n_candidates": 4, "toks": [5, 10]}
    # 2*5*2*3*5 + 2*10*2*3*5 = 300 + 600
    assert ap.layer_flop(rec) == 900


def test_suggest_tile_hand_computed():
    # pair_bytes = 4*(8192*2048 + 256*2048 + 256*8192) = 77_594_624
    # budget = 0.5 * 10e9 = 5e9 ; max_pairs = 64 ; ab=min(25,34,64)=25 ; n=64//25=2
    t = ap.suggest_tile(h_in=2048, h_out=8192, tok_max=256, free_bytes=10e9,
                        budget_frac=0.5, n_experts=256, n_candidates=34)
    assert t["pair_bytes"] == 77_594_624
    assert t["ab_chunk"] == 25
    assert t["expert_chunk"] == 2
    assert t["tile_bytes"] <= 0.5 * 10e9


def test_suggest_tile_oversized_weight_falls_back():
    # one pair already exceeds budget -> (1,1) with a note
    t = ap.suggest_tile(h_in=8192, h_out=8192, tok_max=4096, free_bytes=1e9,
                        budget_frac=0.5, n_experts=256, n_candidates=34)
    assert t["expert_chunk"] == 1 and t["ab_chunk"] == 1
    assert t["note"]


def test_analyze_compute_bound_verdict_and_tile():
    # flop = 2*100000*1000*1000*(4+1) = 1e12 ; wall 0.01 -> 100 TFLOP/s ; peak 100 -> util 1.0
    rec = {"kind": "routed_down", "layer": 3, "n_dev": 1, "n_experts": 1,
           "h_in": 1000, "h_out": 1000, "n_candidates": 4, "toks": [100000],
           "wall_s": 0.01, "peak_gb": [5.0], "free_gb": [10.0]}
    res = ap.analyze([rec], peak_tflops=100.0, hbm_budget_frac=0.5)
    assert res["mean_util"] == pytest.approx(1.0)
    assert "COMPUTE-BOUND" in res["q1"]
    tile = res["per_layer"][0]["tile"]
    assert tile["expert_chunk"] == 1 and tile["ab_chunk"] == 4


def test_analyze_launch_bound_verdict():
    rec = {"kind": "routed_down", "layer": 0, "n_experts": 2, "h_in": 10, "h_out": 10,
           "n_candidates": 4, "toks": [50, 50], "wall_s": 1.0,
           "peak_gb": [1.0], "free_gb": [8.0]}
    res = ap.analyze([rec], peak_tflops=100.0)
    assert res["mean_util"] < 0.25
    assert "LAUNCH-BOUND" in res["q1"]


def test_analyze_reports_search_fraction_from_layer_records():
    recs = [{"kind": "layer", "layer": 0, "t_forward_s": 1.0,
             "t_search_quant_s": 2.0, "t_reforward_s": 1.0}]
    res = ap.analyze(recs)
    assert res["search_fraction"] == pytest.approx(0.5)


# ── profiler contract: silent unless explicitly enabled ──────────────────────

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("JOINTFIX_PROFILE", raising=False)
    assert prof.enabled() is False
    assert prof.clock() == 0.0


def test_enabled_via_env(monkeypatch):
    monkeypatch.setenv("JOINTFIX_PROFILE", "1")
    assert prof.enabled() is True
    assert prof.clock() > 0.0


def test_emit_layer_is_noop_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv("JOINTFIX_PROFILE", raising=False)
    prof.emit_layer(0, 128, 1.0, 2.0, 3.0)
    assert capsys.readouterr().out == ""


def test_emit_layer_payload_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("JOINTFIX_PROFILE", "1")
    prof.emit_layer(5, 128, 1.0, 2.0, 3.0)
    out = capsys.readouterr().out
    assert "[PROF]" in out
    payload = json.loads(out.split("[PROF] ", 1)[1])
    assert payload["kind"] == "layer" and payload["layer"] == 5


def test_emit_routed_down_payload_on_cpu(monkeypatch, capsys):
    monkeypatch.setenv("JOINTFIX_PROFILE", "1")
    tensors = {"e0.down": torch.zeros(3, 2)}            # h_out=3, h_in=2
    stats = {"X_sample": torch.zeros(7, 2)}             # tok=7
    routed = [(0, "e0.down", "e0.up", "ask0")]
    prof.emit_routed_down(layer_idx=2, routed=routed, tensors=tensors,
                          get_stats=lambda ask: stats, devices=[torch.device("cpu")],
                          t_start=0.0, config=object())  # object() -> default candidate count
    payload = json.loads(capsys.readouterr().out.split("[PROF] ", 1)[1])
    assert payload["h_out"] == 3 and payload["h_in"] == 2
    assert payload["toks"] == [7] and payload["n_experts"] == 1
    assert payload["n_candidates"] == 34            # 9 (stage1 default) + 25 (5x5 stage2)


# ── live progress: heartbeat / eta / timestamp (same silent-unless-enabled contract) ──

def test_heartbeat_and_eta_noop_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv("JOINTFIX_PROFILE", raising=False)
    prof.heartbeat("loading weights…")
    prof.eta(1, 8, 0.0)
    assert capsys.readouterr().out == ""


def test_heartbeat_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("JOINTFIX_PROFILE", "1")
    prof.heartbeat("layer 0: weights loaded")
    out = capsys.readouterr().out
    assert "[PROF " in out and "layer 0: weights loaded" in out


def test_eta_reports_progress_and_eta(monkeypatch, capsys):
    monkeypatch.setenv("JOINTFIX_PROFILE", "1")
    prof.eta(2, 8, prof.clock() - 100.0)            # 2/8 done, ~100s elapsed
    out = capsys.readouterr().out
    assert "progress 2/8 layers" in out and "25%" in out and "ETA" in out


def test_fmt_durations():
    assert prof.fmt(42) == "42s"
    assert prof.fmt(309) == "5m09s"
    assert prof.fmt(3725) == "1h02m"


def test_emit_layer_includes_ts_and_t_load(monkeypatch, capsys):
    monkeypatch.setenv("JOINTFIX_PROFILE", "1")
    prof.emit_layer(3, 32, t_forward=2.0, t_search_quant=4.0, t_reforward=1.0, t_load=9.0)
    payload = json.loads(capsys.readouterr().out.split("[PROF] ", 1)[1])
    assert payload["t_load_s"] == 9.0          # the formerly-untimed weight-load phase
    assert "ts" in payload                       # absolute wall-clock stamp on every data line


# ── phase breakdown: where the layer wall actually goes (the next-bottleneck finder) ──

def test_phase_breakdown_totals_dominant_and_stat_overhead():
    recs = [
        {"kind": "layer", "layer": 0, "t_load_s": 10, "t_forward_s": 100,
         "t_search_quant_s": 5, "t_reforward_s": 20},
        {"kind": "layer", "layer": 1, "t_load_s": 10, "t_forward_s": 40,
         "t_search_quant_s": 5, "t_reforward_s": 20},
    ]
    pb = ap.phase_breakdown(recs)
    assert pb["totals"]["forward"] == 140 and pb["totals"]["load"] == 20
    assert pb["dominant"] == "forward"
    assert pb["rows"][0]["stat_oh"] == 80          # forward 100 - reforward 20
    assert pb["rows"][0]["total"] == 135           # 10 + 100 + 5 + 20


def test_phase_breakdown_flags_compilation_warmup():
    recs = [{"kind": "layer", "layer": i, "t_forward_s": f, "t_reforward_s": 5,
             "t_load_s": 1, "t_search_quant_s": 1} for i, f in enumerate([4000, 40, 42, 41, 43])]
    pb = ap.phase_breakdown(recs)                  # median forward ≈ 42
    assert pb["rows"][0]["warmup"] is True         # 4000 >> 3*42
    assert pb["rows"][2]["warmup"] is False


def test_phase_breakdown_handles_missing_t_load():
    # older logs (pre t_load) -> load defaults to 0, no crash
    pb = ap.phase_breakdown([{"kind": "layer", "layer": 0, "t_forward_s": 9,
                              "t_search_quant_s": 1, "t_reforward_s": 2}])
    assert pb["totals"]["load"] == 0.0
    assert pb["rows"][0]["total"] == 12
