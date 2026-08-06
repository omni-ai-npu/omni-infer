# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for the declarative validators framework (coexists with loader._validate_config)."""
from types import SimpleNamespace

import pytest

from omni.configs.validators import (
    Severity,
    ValidationContext,
    ValidationRule,
    Violation,
    validate_all,
)


def _ctx(**kw):
    base = dict(
        model_extra_config=SimpleNamespace(),
        vllm_config=SimpleNamespace(additional_config={}),
    )
    base.update(kw)
    return ValidationContext(**base)


def test_no_rules_no_error():
    validate_all(_ctx(), rules=[])


def test_collects_all_rejects_at_once():
    rules = [
        ValidationRule("r1", "d1", Severity.REJECT,
                       lambda c: Violation(field_path="f1", message="m1")),
        ValidationRule("r2", "d2", Severity.REJECT,
                       lambda c: Violation(field_path="f2", message="m2")),
    ]
    with pytest.raises(ValueError) as ei:
        validate_all(_ctx(), rules=rules)
    msg = str(ei.value)
    assert "r1" in msg and "f1" in msg
    assert "r2" in msg and "f2" in msg     # All violations are reported.
    assert "2" in msg                       # The total is included.


def test_passing_rule_returns_nothing():
    rules = [ValidationRule("ok", "d", Severity.REJECT, lambda c: None)]
    validate_all(_ctx(), rules=rules)  # Does not raise.


def test_rule_exception_isolated(caplog):
    def boom(c):
        raise RuntimeError("bug in rule")

    rules = [
        ValidationRule("boom", "d", Severity.REJECT, boom),
        ValidationRule("good", "d", Severity.REJECT,
                       lambda c: Violation(field_path="f", message="m")),
    ]
    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError):          # Other violations are kept.
            validate_all(_ctx(), rules=rules)
    assert any("boom" in r.message for r in caplog.records)


def test_context_carries_vllm_config():
    vc = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"))
    ctx = _ctx(vllm_config=vc)
    assert ctx.vllm_config is vc


def test_default_rules_used_when_not_passed():
    # Omitting rules uses the module registry.
    validate_all(_ctx())


def test_warn_severity_logs_and_does_not_raise(caplog):
    rules = [ValidationRule("w", "d", Severity.WARN,
                            lambda c: Violation(field_path="f", message="m"))]
    with caplog.at_level("WARNING"):
        validate_all(_ctx(), rules=rules)
    messages = [record.message for record in caplog.records]
    assert sum("OMNI CONFIGURATION WARNING" in msg for msg in messages) == 1
    assert all("\n" not in msg for msg in messages)
    joined = "\n".join(messages)
    assert "#begin count=1 reject_count=0 outcome=continue" in joined
    assert "Startup will continue" in joined
    assert "[w]" in joined
    assert "Field: f" in joined
    assert "Detail: m" in joined
    assert "#end" in joined


def test_warn_severity_aggregates_violations(caplog):
    rules = [
        ValidationRule(
            "w1", "d1", Severity.WARN,
            lambda c: Violation(field_path="f1", message="m1"),
        ),
        ValidationRule(
            "w2", "d2", Severity.WARN,
            lambda c: Violation(field_path="f2", message="m2"),
        ),
    ]
    with caplog.at_level("WARNING"):
        validate_all(_ctx(), rules=rules)
    messages = [record.message for record in caplog.records]
    assert sum("OMNI CONFIGURATION WARNING" in msg for msg in messages) == 1
    joined = "\n".join(messages)
    assert "#begin count=2 reject_count=0 outcome=continue" in joined
    assert joined.index("[w1]") < joined.index("[w2]")
    assert "[w1]" in joined and "Field: f1" in joined
    assert "[w2]" in joined and "Field: f2" in joined


def test_warn_summary_reports_rejected_startup(caplog):
    rules = [
        ValidationRule(
            "w", "warning", Severity.WARN,
            lambda c: Violation(field_path="wf", message="wm"),
        ),
        ValidationRule(
            "r", "reject", Severity.REJECT,
            lambda c: Violation(field_path="rf", message="rm"),
        ),
    ]
    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError, match=r"\[r\]"):
            validate_all(_ctx(), rules=rules)
    joined = "\n".join(record.message for record in caplog.records)
    assert "#begin count=1 reject_count=1 outcome=reject" in joined
    assert "Startup will be rejected" in joined
    assert "[w]" in joined


def test_passing_warn_rule_does_not_log_summary(caplog):
    rules = [ValidationRule("w", "d", Severity.WARN, lambda c: None)]
    with caplog.at_level("WARNING"):
        validate_all(_ctx(), rules=rules)
    assert not any(
        "OMNI CONFIGURATION WARNING" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Cross-source validation rules
# ---------------------------------------------------------------------------
import omni.envs  # noqa: F401,E402 - used by rules through lazy access


def _ctx_with_role(monkeypatch, role, kv_role=None, add_cfg=None, omni_cache_env=None):
    monkeypatch.delenv("OMNI_PD_ROLE", raising=False)
    monkeypatch.delenv("ROLE", raising=False)
    monkeypatch.delenv("OMNI_ENABLE_OMNI_CACHE", raising=False)
    monkeypatch.delenv("ENABLE_OMNI_CACHE", raising=False)
    if role:
        monkeypatch.setenv("OMNI_PD_ROLE", role)
    if omni_cache_env is not None:
        monkeypatch.setenv("OMNI_ENABLE_OMNI_CACHE", omni_cache_env)
    vllm = SimpleNamespace(
        additional_config=add_cfg or {},
        kv_transfer_config=(
            SimpleNamespace(kv_role=kv_role) if kv_role else None),
    )
    return ValidationContext(
        model_extra_config=SimpleNamespace(),
        vllm_config=vllm,
    )


def test_role_kv_role_consistent_pass(monkeypatch):
    from omni.configs.validators import _ALL_RULES
    rule = next(r for r in _ALL_RULES if r.name == "role_kv_role_consistent")
    ctx = _ctx_with_role(monkeypatch, "prefill", kv_role="kv_producer")
    assert rule.check(ctx) is None


def test_role_kv_role_consistent_fail(monkeypatch):
    from omni.configs.validators import _ALL_RULES
    rule = next(r for r in _ALL_RULES if r.name == "role_kv_role_consistent")
    ctx = _ctx_with_role(monkeypatch, "decode", kv_role="kv_producer")
    v = rule.check(ctx)
    assert v is not None
    assert "OMNI_PD_ROLE" in v.field_path


def test_role_kv_role_skipped_when_unset(monkeypatch):
    # Skip role validation when no role information is available.
    from omni.configs.validators import _ALL_RULES
    rule = next(r for r in _ALL_RULES if r.name == "role_kv_role_consistent")
    ctx = _ctx_with_role(monkeypatch, None, kv_role="kv_consumer")
    assert rule.check(ctx) is None


def test_cache_consistency_pass(monkeypatch):
    from omni.configs.validators import _ALL_RULES
    rule = next(r for r in _ALL_RULES if r.name == "omni_cache_consistent")
    ctx = _ctx_with_role(monkeypatch, None, add_cfg={"enable_omni_cache": True},
                         omni_cache_env="true")
    assert rule.check(ctx) is None


def test_cache_consistency_fail(monkeypatch):
    from omni.configs.validators import _ALL_RULES
    rule = next(r for r in _ALL_RULES if r.name == "omni_cache_consistent")
    ctx = _ctx_with_role(monkeypatch, None, add_cfg={"enable_omni_cache": True},
                         omni_cache_env="false")
    v = rule.check(ctx)
    assert v is not None
    assert "enable_omni_cache" in v.field_path


def test_cache_consistency_skips_legacy_env_only_configuration(monkeypatch):
    from omni.configs.validators import _ALL_RULES
    rule = next(r for r in _ALL_RULES if r.name == "omni_cache_consistent")
    ctx = _ctx_with_role(
        monkeypatch,
        None,
        add_cfg={},
        omni_cache_env="true",
    )
    assert rule.check(ctx) is None


def _ctx_with_cudagraph(
    capture_sizes,
    *,
    cudagraph_mode="FULL_DECODE_ONLY",
    max_num_seqs=8,
    speculative_method=None,
    num_speculative_tokens=None,
):
    speculative_config = (
        None
        if speculative_method is None
        else SimpleNamespace(
            method=speculative_method,
            num_speculative_tokens=num_speculative_tokens,
        )
    )
    return ValidationContext(
        model_extra_config=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            additional_config={},
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=capture_sizes,
                cudagraph_mode=cudagraph_mode,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            speculative_config=speculative_config,
        ),
    )


def _cudagraph_rule():
    from omni.configs.validators import _ALL_RULES
    return next(
        rule
        for rule in _ALL_RULES
        if rule.name == "cudagraph_capture_sizes_decode_compatible"
    )


def test_mtp_cudagraph_capture_sizes_multiple_pass():
    ctx = _ctx_with_cudagraph(
        [4, 8, 12],
        cudagraph_mode="FULL_DECODE_ONLY",
        speculative_method="mtp",
        num_speculative_tokens=3,
    )
    assert _cudagraph_rule().check(ctx) is None


def test_mtp_cudagraph_capture_sizes_multiple_warn():
    ctx = _ctx_with_cudagraph(
        [4, 6, 8],
        cudagraph_mode="FULL_AND_PIECEWISE",
        speculative_method="mtp",
        num_speculative_tokens=3,
    )
    violation = _cudagraph_rule().check(ctx)
    assert violation is not None
    assert "[6]" in violation.message
    assert "multiples of 4" in violation.message


def test_mtp_full_graph_non_multiple_warn():
    ctx = _ctx_with_cudagraph(
        [3, 5],
        cudagraph_mode="FULL",
        speculative_method="mtp",
        num_speculative_tokens=3,
    )
    violation = _cudagraph_rule().check(ctx)
    assert violation is not None
    assert "[3, 5]" in violation.message


def test_mtp_piecewise_graph_non_multiple_warn():
    ctx = _ctx_with_cudagraph(
        [3, 5],
        cudagraph_mode="PIECEWISE",
        speculative_method="mtp",
        num_speculative_tokens=3,
    )
    violation = _cudagraph_rule().check(ctx)
    assert violation is not None
    assert "[3, 5]" in violation.message


def test_mtp_cudagraph_capture_size_above_decode_limit_warn():
    ctx = _ctx_with_cudagraph(
        [4, 8, 36],
        cudagraph_mode="FULL",
        max_num_seqs=8,
        speculative_method="mtp",
        num_speculative_tokens=3,
    )
    violation = _cudagraph_rule().check(ctx)
    assert violation is not None
    assert "8 * 4 = 32" in violation.message
    assert "[36]" in violation.message


def test_non_mtp_speculative_method_is_skipped():
    ctx = _ctx_with_cudagraph(
        [3, 5],
        speculative_method="eagle",
        num_speculative_tokens=3,
    )
    assert _cudagraph_rule().check(ctx) is None


def test_full_decode_only_capture_sizes_within_max_num_seqs_pass():
    ctx = _ctx_with_cudagraph([1, 4, 8], max_num_seqs=8)
    assert _cudagraph_rule().check(ctx) is None


def test_full_decode_only_capture_sizes_above_max_num_seqs_warn(caplog):
    ctx = _ctx_with_cudagraph([4, 8, 12], max_num_seqs=8)
    rule = _cudagraph_rule()
    with caplog.at_level("WARNING"):
        validate_all(ctx, rules=[rule])
    joined = "\n".join(record.message for record in caplog.records)
    assert rule.name in joined
    assert "[12]" in joined
    assert "max_num_seqs=8" in joined


def test_non_mtp_full_graph_large_capture_size_warn():
    ctx = _ctx_with_cudagraph(
        [8, 16],
        cudagraph_mode="FULL",
        max_num_seqs=8,
    )
    violation = _cudagraph_rule().check(ctx)
    assert violation is not None
    assert "[16]" in violation.message


def test_cudagraph_capture_sizes_are_skipped_in_none_mode():
    ctx = _ctx_with_cudagraph(
        [3, 36],
        cudagraph_mode="NONE",
        max_num_seqs=8,
        speculative_method="mtp",
        num_speculative_tokens=3,
    )
    assert _cudagraph_rule().check(ctx) is None


@pytest.mark.parametrize("capture_sizes", [None, []])
def test_empty_cudagraph_capture_sizes_are_skipped(capture_sizes):
    assert _cudagraph_rule().check(
        _ctx_with_cudagraph(capture_sizes)
    ) is None
