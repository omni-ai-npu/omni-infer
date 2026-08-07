# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Declarative validation across resolved configuration sources.

This registry complements ``loader._validate_config()`` and covers rules for
environment variables, model configuration, cross-source constraints, and
``additional_config``. It runs after ``emit_config_summary`` so startup errors
can be compared with the emitted configuration snapshot.

Add new rules to ``_ALL_RULES`` without changing the execution path.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional

from omni_npu import envs
from omni_npu.configs import OmniAdditionalConfig

logger = logging.getLogger(__name__)

_WARNING_PREFIX = "[Config Validation][WARN]"
_WARNING_BANNER = f"{_WARNING_PREFIX} " + "=" * 64


class Severity(Enum):
    REJECT = "reject"   # Raise ValueError and stop startup.
    WARN = "warn"       # Log the violation and preserve the configured value.


@dataclass
class Violation:
    """Violation details; identity and severity belong to ValidationRule."""
    field_path: str # Example: env var x config field.
    message: str
    original_value: Any = None


@dataclass
class ValidationRule:
    """A rule that returns ``None`` or a ``Violation``."""
    name: str
    description: str
    severity: Severity
    check: Callable[["ValidationContext"], Optional[Violation]]


@dataclass
class ValidationContext:
    """Data sources available to validation rules.

    Rules read environment variables directly from ``omni_npu.envs`` so its
    lazy ``__getattr__`` always observes the latest value.
    """
    model_extra_config: Any                # ModelExtraConfig
    vllm_config: Any                       # VllmConfig for cross-source rules.


# ---------------------------------------------------------------------------
# Cross-source validation rules
# ---------------------------------------------------------------------------
# ROLE and kv_role represent the same concept:
#   ROLE=prefill  <->  kv_role=kv_producer
#   ROLE=decode   <->  kv_role=kv_consumer
# pd_run.sh sets both values, so reject inconsistent combinations.
_ROLE_TO_KV = {"prefill": "kv_producer", "decode": "kv_consumer"}


def _check_role_kv_role_consistent(ctx: ValidationContext) -> Optional[Violation]:
    role = envs.OMNI_PD_ROLE
    kv_cfg = getattr(ctx.vllm_config, "kv_transfer_config", None)
    if role and kv_cfg is not None:
        expected = _ROLE_TO_KV.get(role)
        actual = getattr(kv_cfg, "kv_role", None)
        if expected and actual and expected != actual:
            return Violation(
                field_path="envs.OMNI_PD_ROLE × kv_transfer_config.kv_role",
                message=(
                    f"OMNI_PD_ROLE={role!r} requires "
                    f"kv_transfer_config.kv_role={expected!r}, "
                    f"but got {actual!r}."
                ),
            )
    return None


def _check_omni_cache_consistent(ctx: ValidationContext) -> Optional[Violation]:
    raw_additional_config = getattr(
        ctx.vllm_config, "additional_config", None
    )
    # omniinfer launchers historically configure Omni Cache only through
    # ENABLE_OMNI_CACHE. Preserve that deployment contract when the shared
    # additional_config does not explicitly claim this setting; if both
    # sources are present, they must agree.
    if (
        not isinstance(raw_additional_config, dict)
        or "enable_omni_cache" not in raw_additional_config
    ):
        return None

    env_cache = envs.OMNI_ENABLE_OMNI_CACHE
    add_cache = OmniAdditionalConfig.from_vllm_config(
        ctx.vllm_config
    ).enable_omni_cache
    if env_cache != add_cache:
        return Violation(
            field_path="envs.OMNI_ENABLE_OMNI_CACHE × additional_config.enable_omni_cache",
            message=(
                f"OMNI_ENABLE_OMNI_CACHE={env_cache!r} does not match "
                f"additional_config.enable_omni_cache={add_cache!r}."
            ),
        )
    return None


def _get_positive_int(config: Any, field_name: str) -> Optional[int]:
    value = getattr(config, field_name, None)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        return None
    return value


def _check_mtp_cudagraph_capture_sizes(
    capture_sizes: List[int],
    speculative_config: Any,
    scheduler_config: Any,
) -> Optional[Violation]:
    num_speculative_tokens = _get_positive_int(
        speculative_config, "num_speculative_tokens"
    )
    if num_speculative_tokens is None:
        return None

    decode_query_len = num_speculative_tokens + 1
    non_matching_sizes = [
        size for size in capture_sizes if size % decode_query_len != 0
    ]

    max_num_seqs = _get_positive_int(scheduler_config, "max_num_seqs")
    max_decode_tokens = (
        max_num_seqs * decode_query_len if max_num_seqs is not None else None
    )
    oversized_sizes = (
        [
            size
            for size in capture_sizes
            if size > max_decode_tokens
        ]
        if max_decode_tokens is not None
        else []
    )
    if not non_matching_sizes and not oversized_sizes:
        return None

    details = []
    if non_matching_sizes:
        details.append(
            "Omni recommends CUDA graph capture sizes be multiples of "
            f"{decode_query_len} for decode workloads. Non-matching "
            f"sizes: {non_matching_sizes}."
        )
    if oversized_sizes:
        details.append(
            "Omni recommends the maximum capture size not exceed "
            "max_num_seqs * (num_speculative_tokens + 1) = "
            f"{max_num_seqs} * {decode_query_len} = "
            f"{max_decode_tokens} for decode workloads. Sizes above "
            f"this recommendation: {oversized_sizes}."
        )
    return Violation(
        field_path=(
            "compilation_config.cudagraph_capture_sizes × "
            "speculative_config.num_speculative_tokens × "
            "scheduler_config.max_num_seqs"
        ),
        message=(
            f"MTP processes {decode_query_len} decode tokens per request "
            "(num_speculative_tokens + 1). "
            + " ".join(details)
        ),
        original_value=capture_sizes,
    )


def _check_non_mtp_cudagraph_capture_sizes(
    capture_sizes: List[int],
    scheduler_config: Any,
) -> Optional[Violation]:
    max_num_seqs = _get_positive_int(scheduler_config, "max_num_seqs")
    if max_num_seqs is None:
        return None

    oversized_sizes = [size for size in capture_sizes if size > max_num_seqs]
    if not oversized_sizes:
        return None
    return Violation(
        field_path=(
            "compilation_config.cudagraph_capture_sizes × "
            "scheduler_config.max_num_seqs"
        ),
        message=(
            "Without MTP, a pure decode batch contains at most "
            f"max_num_seqs={max_num_seqs} tokens. Omni recommends "
            "capture sizes not exceed this value for decode workloads. "
            f"Sizes above the recommendation: {oversized_sizes}. Retain "
            "them only when prefill or mixed-batch graph coverage is "
            "required."
        ),
        original_value=capture_sizes,
    )


def _check_cudagraph_capture_sizes_decode_compatible(
    ctx: ValidationContext,
) -> Optional[Violation]:
    """Check Omni capture-size recommendations when CUDA graphs are enabled."""
    compilation_config = getattr(ctx.vllm_config, "compilation_config", None)
    capture_sizes = getattr(
        compilation_config, "cudagraph_capture_sizes", None
    )
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    cudagraph_mode_name = getattr(cudagraph_mode, "name", cudagraph_mode)
    if not capture_sizes or cudagraph_mode_name in (None, "NONE"):
        return None

    speculative_config = getattr(ctx.vllm_config, "speculative_config", None)
    scheduler_config = getattr(ctx.vllm_config, "scheduler_config", None)
    if speculative_config is None:
        return _check_non_mtp_cudagraph_capture_sizes(
            capture_sizes, scheduler_config
        )
    if getattr(speculative_config, "method", None) != "mtp":
        return None
    return _check_mtp_cudagraph_capture_sizes(
        capture_sizes, speculative_config, scheduler_config
    )


# Add a ValidationRule when a startup constraint depends on multiple resolved
# configuration sources (for example, an environment variable and VllmConfig),
# or when it cannot be checked until the complete startup configuration is
# available. Keep single-field type, format, and range validation in the owning
# dataclass or loader validation instead of duplicating it here. To add a rule,
# implement a check that returns None or Violation, register it in _ALL_RULES,
# and cover both valid and invalid combinations in tests. REJECT violations are
# collected and raised together; WARN violations are logged without changing
# the configured value. Automatic configuration fixes are intentionally outside
# this framework.
_ALL_RULES: List[ValidationRule] = [
    ValidationRule(
        name="role_kv_role_consistent",
        description="OMNI_PD_ROLE must match kv_transfer_config.kv_role",
        severity=Severity.REJECT,
        check=_check_role_kv_role_consistent,
    ),
    ValidationRule(
        name="omni_cache_consistent",
        description=(
            "When explicitly configured, additional_config.enable_omni_cache "
            "must match OMNI_ENABLE_OMNI_CACHE"
        ),
        severity=Severity.REJECT,
        check=_check_omni_cache_consistent,
    ),
    ValidationRule(
        name="cudagraph_capture_sizes_decode_compatible",
        description=(
            "When CUDA graphs are enabled, capture sizes should follow Omni "
            "decode-workload recommendations"
        ),
        severity=Severity.WARN,
        check=_check_cudagraph_capture_sizes_decode_compatible,
    ),
]


def _warning_summary_lines(
    warnings: List[tuple[ValidationRule, Violation]],
    reject_count: int,
) -> List[str]:
    startup_rejected = reject_count > 0
    outcome = "reject" if startup_rejected else "continue"
    lines = [
        _WARNING_BANNER,
        (
            f"{_WARNING_PREFIX} #begin count={len(warnings)} "
            f"reject_count={reject_count} outcome={outcome}"
        ),
        f"{_WARNING_PREFIX} !!! OMNI CONFIGURATION WARNING !!!",
    ]
    if startup_rejected:
        lines.append(
            f"{_WARNING_PREFIX} Startup will be rejected because other "
            "configuration errors were found. Review these warnings as well."
        )
    else:
        lines.append(
            f"{_WARNING_PREFIX} Startup will continue with the configured "
            "values. Review them before production deployment."
        )
    lines.append(f"{_WARNING_PREFIX} " + "-" * 64)
    for i, (rule, violation) in enumerate(warnings, 1):
        lines.append(f"{_WARNING_PREFIX} {i}. [{rule.name}]")
        lines.append(
            f"{_WARNING_PREFIX}    Field: {violation.field_path}"
        )
        lines.append(
            f"{_WARNING_PREFIX}    Detail: {violation.message}"
        )
    lines.extend([
        f"{_WARNING_PREFIX} #end",
        _WARNING_BANNER,
    ])
    return lines


def validate_all(ctx: ValidationContext,
                 rules: Optional[List[ValidationRule]] = None) -> None:
    """Validate the resolved startup configuration with registered rules.

    All REJECT violations are collected and raised together. WARN violations
    are collected into one prominent log block and preserve the configured
    value. Unexpected rule failures are logged and skipped so they do not hide
    violations from remaining rules.
    """
    if rules is None:
        rules = _ALL_RULES
    errors: List[tuple[ValidationRule, Violation]] = []
    warnings: List[tuple[ValidationRule, Violation]] = []
    for rule in rules:
        try:
            violation = rule.check(ctx)
        except Exception as e:  # noqa: BLE001 - rule failures are isolated
            logger.warning("[Config Validation] Rule %r raised: %s", rule.name, e)
            continue
        if violation is None:
            continue
        if rule.severity == Severity.REJECT:
            errors.append((rule, violation))
        elif rule.severity == Severity.WARN:
            warnings.append((rule, violation))

    if warnings:
        for line in _warning_summary_lines(warnings, len(errors)):
            logger.warning("%s", line)

    if errors:
        lines = [
            f"[Config Validation] Found {len(errors)} configuration conflict(s):"
        ]
        for i, (rule, violation) in enumerate(errors, 1):
            lines.append(f"  {i}. [{rule.name}] {violation.field_path}")
            lines.append(f"     {violation.message}")
        raise ValueError("\n".join(lines))
