# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""OMNI-DUMP internal constants.

The only deployment-facing knobs are the two environment variables below;
everything else is fixed by design.
"""
import signal

# Forensics signal semantics: "dump and keep running".
DUMP_SIGNAL = signal.SIGUSR1

# Periodic stats snapshot interval (EngineCore role only).
STATS_FLUSH_INTERVAL_SEC = 10.0

# Ring size for per-step duration samples.
STEP_DURATION_RING_SIZE = 64

DUMP_SCHEMA = "omni-dump/1"

# Subdirectory for machine/degraded artifacts (faulthandler stack sink,
# periodic stats snapshot); final dump packages stay at the top level.
RAW_SUBDIR = "raw"

TRIGGER_SIGNAL = "signal"
TRIGGER_ATEXIT = "atexit"

# Marker for sections intentionally not collected (by design, not a failure).
SKIPPED = "skipped"
STACK_SKIPPED = SKIPPED

# The device probe (torch import + mem_get_info) can hang outright when the
# NPU runtime is corrupted; it runs in a disposable timed thread.
DEVICE_MEM_PROBE_TIMEOUT_SEC = 2.0

ROLE_WORKER = "worker"
ROLE_ENGINE = "engine"
ROLE_API = "api"

ENV_ENABLE = "OMNI_DUMP_ENABLE"
ENV_DUMP_DIR = "OMNI_DUMP_DIR"
DEFAULT_DUMP_DIR = "/var/log/omni-npu/dump"
