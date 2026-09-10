# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "omni" / "model_config" / "configs"


def _load(relative_path):
    with (CONFIG_ROOT / relative_path).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _find_best_practice(relative_path, model, hardware, precision):
    entries = _load(relative_path)
    for entry in entries:
        if (
            entry["model"] == model
            and entry["hardware"] == hardware
            and entry["precision"] == precision
        ):
            return entry
    raise AssertionError(
        f"Missing best-practice config: {model}/{hardware}/{precision}"
    )


@pytest.mark.parametrize(
    ("model", "precision"),
    [
        ("openpangu_v2_35B", "bf16"),
        ("openpangu_v2_35B", "mxfp8"),
        ("openpangu_v2_92B", "bf16"),
        ("openpangu_v2_92B", "mxfp8"),
    ],
)
def test_a5_best_practice_registers_two_prefill_one_decode(model, precision):
    entry = _find_best_practice(
        "high_throughout/best_practice_configs.json", model, "A5", precision
    )

    assert entry["configs"]["2P1D"] == entry["configs"]["1P1D"]


def test_a5_best_practice_registers_fp16_and_w4a8_mxfp():
    fp16 = _find_best_practice(
        "high_throughout/best_practice_configs.json",
        "openpangu_v2_35B",
        "A5",
        "fp16",
    )
    w4a8 = _find_best_practice(
        "high_throughout/best_practice_configs.json",
        "openpangu_v2_35B",
        "A5",
        "w4a8_mxfp",
    )

    assert fp16["configs"]["1P1D"]["decode_config_file"].endswith(
        "openpangu_v2_35b_bf16_a5_1p1d_d.json"
    )
    assert w4a8["configs"]["hybrid"]["config_file"].endswith(
        "openpangu_v2_w4a8_mxfp_a5_hybrid.json"
    )


def test_a5_decode_enables_multistream_optimizations():
    config = _load(
        "high_throughout/openpangu_v2/"
        "openpangu_v2_35b_mxfp8_a5_1p1d_d.json"
    )
    operator_config = config["operator_optimization_config"]

    assert operator_config["enable_multi_stream"] is True
    assert operator_config["split_q_up_in_multistream"] is True
    assert operator_config["use_mome_inplace_update"] is True
    assert operator_config["enable_mhc_multistream"] is True


def test_vl_505b_best_practice_and_pd_configs_are_consistent():
    entry = _find_best_practice(
        "low_latency/best_practice_configs.json",
        "openpangu_v2_vl_505B",
        "A3",
        "bf16",
    )
    pd_config = entry["configs"]["1P1D"]
    prefill = _load("low_latency/" + pd_config["prefill_config_file"])
    decode = _load("low_latency/" + pd_config["decode_config_file"])

    prefill_operator = prefill["operator_optimization_config"]
    decode_operator = decode["operator_optimization_config"]
    assert prefill_operator["enable_mome_sp"] is True
    assert prefill_operator["moe_comm_strategy"] == "all2allv"
    assert decode_operator["moe_comm_strategy"] == "dispatch_combine"
    assert decode_operator["moe_dispatch_combine_max_batch_size"] == 256
    assert decode_operator["enable_multi_stream"] is True


@pytest.mark.parametrize(
    "relative_path",
    [
        "low_latency/openpangu_v2/"
        "pangu_v2_moe_bf16_a3_505B_xp1d_d_omnicache_claw.json",
        "low_latency/openpangu_v2/pangu_v2_moe_ull.json",
    ],
)
def test_mhc_configs_disable_sampler_multistream(relative_path):
    operator_config = _load(relative_path)["operator_optimization_config"]

    assert operator_config["use_mhc_fusion_op"] is True
    assert operator_config["sampler_multi_stream"] is False
