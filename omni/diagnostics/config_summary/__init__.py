# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""OMNI-CONF: startup configuration snapshot emitted as atomic log records.

Implementation lives in :mod:`.collector` (runtime collector/emitter) and
:mod:`.classification` (schema tables, key classification, dot-path codec);
this package only re-exports the public entry points.
"""
from omni_npu.diagnostics.config_summary.collector import (
    build_entries,
    canonical_lines,
    compute_hash,
    emit_config_summary,
    is_enabled,
    reset_once_guard,
)

__all__ = [
    "build_entries",
    "canonical_lines",
    "compute_hash",
    "emit_config_summary",
    "is_enabled",
    "reset_once_guard",
]
