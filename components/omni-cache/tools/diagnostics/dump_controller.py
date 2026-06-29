# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unified KV dump controller — single env-var gate with gear levels.

One env var: `OMNI_KV_DUMP_GEAR` (alias `OMNI_KV_DUMP`)

  - "transfer" / "1"  — enable 4-stage probes
  - "step" / "2"      — enable per-step decode dumper
  - "all" / "3"       — enable both
  - "off" / "0" / unset — disabled

Other knobs (all prefixed OMNI_KV_DUMP_):
  - OMNI_KV_DUMP_DIR     (default /tmp/kv_dumps)
  - OMNI_KV_DUMP_BRANCH  (default auto)
  - OMNI_KV_DUMP_MAX     (default 0 = ∞)
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

GEAR_ENV = "OMNI_KV_DUMP_GEAR"
GEAR_ALIAS = "OMNI_KV_DUMP"

GEAR_OFF = {"off", "0", "", "false", "no"}
GEAR_TRANSFER = {"transfer", "1", "transmission", "stage", "stages"}
GEAR_STEP = {"step", "2", "decode_step", "per_step"}
GEAR_ALL = {"all", "3", "both", "full"}


def gear() -> str:
    """Return canonical gear: 'off' | 'transfer' | 'step' | 'all'."""
    raw = (
        os.environ.get(GEAR_ENV)
        or os.environ.get(GEAR_ALIAS)
        or ""
    ).strip().lower()
    if raw in GEAR_OFF and raw != "":
        return "off"
    if raw in GEAR_TRANSFER:
        return "transfer"
    if raw in GEAR_STEP:
        return "step"
    if raw in GEAR_ALL:
        return "all"
    return "off"


def gear_is_transfer() -> bool:
    return gear() in ("transfer", "all")


def gear_is_step() -> bool:
    return gear() in ("step", "all")


def dump_dir(default: str = "/tmp/kv_dumps") -> str:
    return os.environ.get("OMNI_KV_DUMP_DIR", default)


def step_dir() -> str:
    return os.path.join(dump_dir(), "_step")


def branch_name() -> str:
    v = (os.environ.get("OMNI_KV_DUMP_BRANCH") or "").strip()
    if v:
        return v
    return ("omnicache"
            if os.environ.get("ENABLE_OMNI_CACHE", "0") == "1"
            else "baseline")


def max_steps() -> int:
    raw = (os.environ.get("OMNI_KV_DUMP_MAX") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def install() -> None:
    """vLLM general-plugins entry point."""
    g = gear()
    logger.info("[tools.diagnostics] dump gear=%s", g)

    if gear_is_step():
        from tools.diagnostics.kv_step_dumper import install as _step_install
        _step_install()
