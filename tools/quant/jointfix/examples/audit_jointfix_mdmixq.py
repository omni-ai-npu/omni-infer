#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Audit a completed JointFix-MDMixQ decoder artifact directory."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="original checkpoint config root")
    parser.add_argument("--artifacts", required=True, help="decoder layer artifact directory")
    args = parser.parse_args()

    model = Path(args.model)
    artifacts = Path(args.artifacts)
    config = json.loads((model / "config.json").read_text())
    depth = int(config["num_hidden_layers"])
    first_moe = int(config.get("first_k_dense_replace", depth))

    expected_layers = {
        f"layer_{layer:04d}.safetensors" for layer in range(depth)
    }
    actual_layers = {path.name for path in artifacts.glob("layer_*.safetensors")}
    if actual_layers != expected_layers:
        raise RuntimeError(
            "incomplete decoder artifacts: "
            f"missing={sorted(expected_layers - actual_layers)}, "
            f"unexpected={sorted(actual_layers - expected_layers)}"
        )

    trace_path = artifacts / "joint_search_traces.json"
    if not trace_path.is_file():
        raise RuntimeError(f"missing trace: {trace_path}")
    traces = json.loads(trace_path.read_text())

    before_sum = 0.0
    after_sum = 0.0
    improved = equal = regressed = 0
    for name, values in traces.items():
        if name.endswith(".mdmixq") or not isinstance(values, dict):
            continue
        before = values.get("weight_nmse_before_scale_refine")
        after = values.get("weight_nmse_after_scale_refine")
        if before is None or after is None:
            continue
        before, after = float(before), float(after)
        if not math.isfinite(before) or not math.isfinite(after):
            raise RuntimeError(f"non-finite weight NMSE: {name}")
        before_sum += before
        after_sum += after
        tolerance = max(1e-15, abs(before) * 1e-12)
        if after < before - tolerance:
            improved += 1
        elif after <= before + tolerance:
            equal += 1
        else:
            regressed += 1

    if not improved and not equal:
        raise RuntimeError("trace contains no scale-refinement weight NMSE")
    if regressed:
        raise RuntimeError(f"weight NMSE regressed for {regressed} matrices")

    expected_routes = {
        f"model.layers.{layer}.mdmixq" for layer in range(first_moe, depth)
    }
    actual_routes = {name for name in traces if name.endswith(".mdmixq")}
    if actual_routes != expected_routes:
        raise RuntimeError(
            "incomplete MDMixQ route summaries: "
            f"missing={sorted(expected_routes - actual_routes)}, "
            f"unexpected={sorted(actual_routes - expected_routes)}"
        )
    text_tokens = nontext_tokens = 0
    for name in sorted(actual_routes):
        values = traces[name]
        layer_text = int(values.get("text_tokens", 0))
        layer_nontext = int(values.get("nontext_tokens", 0))
        if layer_text <= 0 or layer_nontext <= 0:
            raise RuntimeError(
                f"{name} did not observe both modalities: "
                f"text={layer_text}, nontext={layer_nontext}"
            )
        if not values.get("text_experts") or not values.get("nontext_experts"):
            raise RuntimeError(f"{name} has empty protected expert sets")
        text_tokens += layer_text
        nontext_tokens += layer_nontext

    relative_gain = (
        (before_sum - after_sum) / before_sum if before_sum > 0 else 0.0
    )
    report = {
        "status": "PASS",
        "decoder_layers": depth,
        "moe_route_layers": len(actual_routes),
        "route_text_tokens_sum": text_tokens,
        "route_nontext_tokens_sum": nontext_tokens,
        "weight_matrices": improved + equal,
        "weight_nmse_improved": improved,
        "weight_nmse_equal": equal,
        "weight_nmse_regressed": regressed,
        "weight_nmse_sum_before_scale_refine": before_sum,
        "weight_nmse_sum_after_scale_refine": after_sum,
        "weight_nmse_relative_improvement": relative_gain,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
