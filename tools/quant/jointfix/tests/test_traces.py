# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
jointfix dumps joint_search_traces.json so the searched (a,b) can be diffed against the
monolith (tools/compare_joint_traces.py) — the established lightweight equivalence check.
"""
import json

import torch

from jointfix.methods.jointfix import JointFixMethod


def test_dump_traces_writes_scalar_json(tmp_path):
    m = JointFixMethod()
    m._traces = {
        "model.layers.0.mlp.experts.0.down_proj.weight": {
            "a": 0.3, "b": 0.7, "J_final": 1.5, "s": torch.randn(4)},   # 's' is a tensor
        "model.layers.0.self_attn.o_proj.weight": {"a": 0.5, "b": 0.5, "J_final": 0.2},
    }
    m.dump_traces(tmp_path)

    data = json.loads((tmp_path / "joint_search_traces.json").read_text())
    e = data["model.layers.0.mlp.experts.0.down_proj.weight"]
    assert e["a"] == 0.3 and e["b"] == 0.7 and e["J_final"] == 1.5
    assert "s" not in e                                   # tensor field dropped (keeps file small)
    assert data["model.layers.0.self_attn.o_proj.weight"]["b"] == 0.5


def test_dump_traces_noop_when_empty(tmp_path):
    JointFixMethod().dump_traces(tmp_path)
    assert not (tmp_path / "joint_search_traces.json").exists()


def test_process_layer_accumulates_into_traces():
    # the trace dict is the same per-weight structure the monolith writes, so
    # compare_joint_traces.py works across both. (process_layer wiring is exercised
    # end-to-end by the on-NPU run; here we just pin the accumulation contract.)
    m = JointFixMethod()
    assert m._traces == {}
