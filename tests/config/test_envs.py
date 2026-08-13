# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.envs (single env-var registry, lazy eval, 3-tier fallback).

Pure-stdlib: deliberately placed under tests/config/ (not tests/unit/) to avoid
the module-level `import torch` in tests/unit/conftest.py. Run with PYTHONPATH=src.
"""
import importlib
import logging
import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if (k.startswith("OMNI_") or k in {
                "ROLE", "PREFILL_POD_NUM", "DECODE_POD_NUM",
                "ENABLE_OMNI_CACHE", "NO_NPU_MOCK",
                "PROFILER_TOKEN_THRESHOLD", "REPETITION_DETECTION_CONFIG",
                "REASONING_CONFIG", "STRUCTURED_OUTPUT_CONFIG",
                "PANGU_TOOL_CALL_ENDS_THINKING",
                "MAX_DISPATCH_COMBINE_THRESHOLD",
                "PROFILER_STOP_STEP", "KV_DUMP_PATH", "CUSTOM_MODEL_CONFIG_PATH",
                "NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT"}):
            monkeypatch.delenv(k, raising=False)
    import omni_npu.envs as envs
    importlib.reload(envs)
    yield


def test_new_name_takes_priority(monkeypatch):
    monkeypatch.setenv("OMNI_PD_ROLE", "decode")
    monkeypatch.setenv("ROLE", "prefill")  # The legacy name is ignored.
    import omni_npu.envs as envs
    assert envs.OMNI_PD_ROLE == "decode"


def test_old_name_fallback_with_deprecation(monkeypatch, caplog):
    monkeypatch.delenv("OMNI_PD_ROLE", raising=False)
    monkeypatch.setenv("ROLE", "prefill")
    import omni_npu.envs as envs
    with caplog.at_level(logging.WARNING, logger="omni_npu.envs"):
        assert envs.OMNI_PD_ROLE == "prefill"
    assert any("deprecated" in r.message and "ROLE" in r.message for r in caplog.records)


def test_default_when_neither_set():
    import omni_npu.envs as envs
    assert envs.OMNI_PD_ROLE is None
    assert envs.OMNI_PD_PREFILL_POD_NUM == 1
    assert envs.OMNI_MAX_DISPATCH_COMBINE_THRESHOLD == 64


def test_recent_runtime_defaults_are_typed():
    import omni_npu.envs as envs

    assert envs.OMNI_DUMP_ENABLE is True
    assert envs.OMNI_DUMP_DIR == "/var/log/omni-npu/dump"
    assert envs.OMNI_HEALTH_HANG_SEC == 240
    assert isinstance(envs.OMNI_HEALTH_HANG_SEC, int)
    assert envs.OMNI_METRICS_KV_TRANSFER_SELFTEST is False
    assert envs.OMNI_METRICS_WORKER_MEM_EVERY == 50
    assert envs.OMNI_NPU_PENALTY_CACHE is False
    assert envs.OMNI_NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT is False


def test_recent_runtime_values_are_parsed(monkeypatch):
    monkeypatch.setenv("OMNI_DUMP_ENABLE", "0")
    monkeypatch.setenv("OMNI_DUMP_DIR", "/tmp/omni-dump")
    monkeypatch.setenv("OMNI_HEALTH_HANG_SEC", "12")
    monkeypatch.setenv("OMNI_METRICS_KV_TRANSFER_SELFTEST", "1")
    monkeypatch.setenv("OMNI_METRICS_WORKER_MEM_EVERY", "25")
    monkeypatch.setenv("OMNI_NPU_PENALTY_CACHE", "1")
    monkeypatch.setenv(
        "OMNI_NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT", "1"
    )
    import omni_npu.envs as envs

    assert envs.OMNI_DUMP_ENABLE is False
    assert envs.OMNI_DUMP_DIR == "/tmp/omni-dump"
    assert envs.OMNI_HEALTH_HANG_SEC == 12
    assert envs.OMNI_METRICS_KV_TRANSFER_SELFTEST is True
    assert envs.OMNI_METRICS_WORKER_MEM_EVERY == 25
    assert envs.OMNI_NPU_PENALTY_CACHE is True
    assert envs.OMNI_NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT is True


def test_npu_sample_float_compatibility_legacy_fallback(
    monkeypatch, caplog
):
    monkeypatch.setenv("NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT", "1")
    import omni_npu.envs as envs

    with caplog.at_level(logging.WARNING, logger="omni_npu.envs"):
        assert envs.OMNI_NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT is True
    assert any(
        "NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT" in record.message
        and "deprecated" in record.message
        for record in caplog.records
    )


def test_int_parser(monkeypatch):
    monkeypatch.setenv("OMNI_PD_DECODE_POD_NUM", "4")
    import omni_npu.envs as envs
    assert envs.OMNI_PD_DECODE_POD_NUM == 4
    assert isinstance(envs.OMNI_PD_DECODE_POD_NUM, int)


@pytest.mark.parametrize("name", [
    "OMNI_PROFILE_TOKEN_THRESHOLD",
    "PROFILER_TOKEN_THRESHOLD",
])
def test_profiler_token_threshold_is_int(monkeypatch, name):
    monkeypatch.setenv(name, "10")
    import omni_npu.envs as envs
    assert envs.OMNI_PROFILE_TOKEN_THRESHOLD == 10
    assert isinstance(envs.OMNI_PROFILE_TOKEN_THRESHOLD, int)


def test_bool_parser(monkeypatch):
    monkeypatch.setenv("OMNI_ENABLE_OMNI_CACHE", "true")
    import omni_npu.envs as envs
    assert envs.OMNI_ENABLE_OMNI_CACHE is True


def test_lazy_eval_picks_up_runtime_changes(monkeypatch):
    import omni_npu.envs as envs
    monkeypatch.delenv("OMNI_PD_ROLE", raising=False)
    assert envs.OMNI_PD_ROLE is None
    monkeypatch.setenv("OMNI_PD_ROLE", "decode")
    assert envs.OMNI_PD_ROLE == "decode"  # Lazy access observes updates.


@pytest.mark.parametrize(
    ("new_name", "old_name", "old_value", "expected", "new_value"),
    [
        (
            "OMNI_VLLM_PATCHES",
            "OMNI_NPU_VLLM_PATCHES",
            "foo,bar",
            "foo,bar",
            "new-patch",
        ),
        (
            "OMNI_VLLM_PATCHES_DIR",
            "OMNI_NPU_PATCHES_DIR",
            "pangu_v2_hybrid",
            "pangu_v2_hybrid",
            "deepseek",
        ),
        (
            "OMNI_LMHEAD_USE_DEVICE_COMM_A2A",
            "OMNI_NPU_USE_DEVICE_COMM_A2A",
            "1",
            True,
            "0",
        ),
        (
            "OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD",
            "OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD",
            "40",
            40,
            "24",
        ),
    ],
)
def test_omni_npu_names_are_compatibility_aliases(
    monkeypatch,
    caplog,
    new_name,
    old_name,
    old_value,
    expected,
    new_value,
):
    import omni_npu.envs as envs

    monkeypatch.setenv(old_name, old_value)
    with caplog.at_level(logging.WARNING, logger="omni_npu.envs"):
        assert getattr(envs, new_name) == expected
    assert any(
        old_name in record.message
        and new_name in record.message
        and "deprecated" in record.message
        for record in caplog.records
    )

    caplog.clear()
    monkeypatch.setenv(new_name, new_value)
    assert getattr(envs, new_name) != expected
    assert not caplog.records


def test_dir_lists_all_registered():
    import omni_npu.envs as envs
    names = set(dir(envs))
    for must in {"OMNI_PD_ROLE", "OMNI_PD_PREFILL_POD_NUM", "OMNI_ENABLE_OMNI_CACHE",
                 "OMNI_VLLM_PATCHES", "OMNI_CONFIG_SUMMARY",
                 "OMNI_DUMP_ENABLE", "OMNI_HEALTH_HANG_SEC",
                 "OMNI_METRICS_WORKER_MEM_EVERY",
                 "OMNI_TRACE_OUTPUT_DIRECTORY"}:
        assert must in names


def test_source_only_vars_are_not_registered():
    import omni_npu.envs as envs

    removed = {
        "OMNI_MOCK_RANDOM_MODE",
        "OMNI_MOCK_CAPTURE_MODE",
        "OMNI_MOCK_REPLAY_MODE",
        "OMNI_MOCK_KV_CACHE_MODE",
        "OMNI_MOCK_PREFILL_PROCESS",
        "OMNI_MOCK_CAPTURE_DIR",
        "OMNI_MOCK_CAPTURE_FILE",
        "OMNI_MOCK_CAPTURE_FILE_LOCK",
        "OMNI_MOCK_SIMULATE_ELAPSED_TIME",
        "OMNI_MOCK_FORWARD_TIME",
        "OMNI_MOCK_COMPUTE_LOGITS",
    }
    assert removed.isdisjoint(dir(envs))


def test_unknown_attr_raises():
    import omni_npu.envs as envs
    with pytest.raises(AttributeError):
        _ = envs.NOT_A_REAL_VAR


def test_default_none_for_unset_path_vars():
    import omni_npu.envs as envs
    # Path and profiler defaults preserve their original unset semantics.
    assert envs.OMNI_KV_DUMP_PATH == ""
    assert envs.OMNI_CUSTOM_MODEL_CONFIG_PATH is None
    assert envs.OMNI_TRACE_OUTPUT_DIRECTORY is None
    assert envs.OMNI_PROFILE_TOKEN_THRESHOLD is None


def test_removed_profile_namelist_is_not_registered():
    import omni_npu.envs as envs

    assert "OMNI_PROFILE_NAMELIST" not in dir(envs)
    with pytest.raises(AttributeError):
        _ = envs.OMNI_PROFILE_NAMELIST


def test_trace_output_directory_is_lazy(monkeypatch):
    import omni_npu.envs as envs

    monkeypatch.setenv("OMNI_TRACE_OUTPUT_DIRECTORY", "/tmp/omni-trace")
    assert envs.OMNI_TRACE_OUTPUT_DIRECTORY == "/tmp/omni-trace"

    monkeypatch.setenv("OMNI_TRACE_OUTPUT_DIRECTORY", "/tmp/omni-trace-next")
    assert envs.OMNI_TRACE_OUTPUT_DIRECTORY == "/tmp/omni-trace-next"

    monkeypatch.setenv("OMNI_TRACE_OUTPUT_DIRECTORY", "")
    assert envs.OMNI_TRACE_OUTPUT_DIRECTORY == ""

    monkeypatch.setenv("OMNI_TRACE_OUTPUT_DIRECTORY", "   ")
    assert envs.OMNI_TRACE_OUTPUT_DIRECTORY == "   "


def test_custom_model_config_path_legacy_fallback(monkeypatch, caplog):
    monkeypatch.setenv("CUSTOM_MODEL_CONFIG_PATH", "/tmp/model-extra.json")
    import omni_npu.envs as envs

    with caplog.at_level(logging.WARNING, logger="omni_npu.envs"):
        assert (
            envs.OMNI_CUSTOM_MODEL_CONFIG_PATH
            == "/tmp/model-extra.json"
        )
    assert any(
        "CUSTOM_MODEL_CONFIG_PATH" in record.message
        and "deprecated" in record.message
        for record in caplog.records
    )


def test_model_extra_cfg_path_is_not_a_compatibility_alias(monkeypatch):
    monkeypatch.setenv("MODEL_EXTRA_CFG_PATH", "/tmp/model-extra.json")
    import omni_npu.envs as envs

    assert envs.OMNI_CUSTOM_MODEL_CONFIG_PATH is None


@pytest.mark.parametrize(
    ("new_name", "old_name", "value"),
    [
        (
            "OMNI_REPETITION_DETECTION_CONFIG",
            "REPETITION_DETECTION_CONFIG",
            '{"max_pattern_size": 10}',
        ),
        ("OMNI_REASONING_CONFIG", "REASONING_CONFIG", '{"enabled": true}'),
        (
            "OMNI_STRUCTURED_OUTPUT_CONFIG",
            "STRUCTURED_OUTPUT_CONFIG",
            '{"backend": "xgrammar"}',
        ),
    ],
)
def test_new_string_vars_take_priority_and_fall_back(
    monkeypatch, caplog, new_name, old_name, value
):
    import omni_npu.envs as envs

    monkeypatch.setenv(old_name, value)
    with caplog.at_level(logging.WARNING, logger="omni_npu.envs"):
        assert getattr(envs, new_name) == value
    assert any(old_name in record.message for record in caplog.records)

    caplog.clear()
    monkeypatch.setenv(new_name, "new-value")
    assert getattr(envs, new_name) == "new-value"
    assert not caplog.records


def test_pangu_tool_call_ends_thinking_rename(monkeypatch, caplog):
    import omni_npu.envs as envs

    monkeypatch.setenv("PANGU_TOOL_CALL_ENDS_THINKING", "1")
    with caplog.at_level(logging.WARNING, logger="omni_npu.envs"):
        assert envs.OMNI_PANGU_TOOL_CALL_ENDS_THINKING is True
    assert any(
        "PANGU_TOOL_CALL_ENDS_THINKING" in record.message
        for record in caplog.records
    )

    monkeypatch.setenv("OMNI_PANGU_TOOL_CALL_ENDS_THINKING", "0")
    assert envs.OMNI_PANGU_TOOL_CALL_ENDS_THINKING is False


def test_benchmark_vars_are_registered_and_typed(monkeypatch):
    import omni_npu.envs as envs

    monkeypatch.setenv("OMNI_HYBRID_ALIGNED_DECODE", "ALL")
    monkeypatch.setenv("OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD", "32")
    monkeypatch.setenv("OMNI_DP_ROUND_ROBIN", "true")
    monkeypatch.setenv("OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD", "40")

    assert envs.OMNI_HYBRID_ALIGNED_DECODE is True
    assert envs.OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD == 32
    assert envs.OMNI_DP_ROUND_ROBIN is True
    assert envs.OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD == 40


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD", 16),
        ("OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD", 20),
    ],
)
def test_invalid_benchmark_threshold_preserves_default(
    monkeypatch, caplog, name, default
):
    import omni_npu.envs as envs

    monkeypatch.setenv(name, "invalid")
    with caplog.at_level(logging.WARNING, logger="omni_npu.envs"):
        assert getattr(envs, name) == default
    assert any(name in record.message for record in caplog.records)
