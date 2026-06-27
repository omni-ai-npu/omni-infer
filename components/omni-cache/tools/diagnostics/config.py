# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Configuration for OmniCache KV diagnostics — fine-grained knobs."""

import os
import threading
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DiagnosticsConfig:
    dump_dir: Optional[str] = None
    verify: bool = False
    max_requests: int = 1
    groups: Optional[List[int]] = None
    layers: Optional[List[int]] = None
    components: Optional[List[str]] = None
    _request_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def active(self) -> bool:
        return self.dump_dir is not None or self.verify

    @property
    def dump_enabled(self) -> bool:
        return self.dump_dir is not None

    def should_capture(self) -> bool:
        if not self.dump_enabled:
            return False
        with self._lock:
            return self._request_count < self.max_requests

    def mark_captured(self) -> None:
        with self._lock:
            self._request_count += 1

    def requests_remaining(self) -> int:
        with self._lock:
            return max(0, self.max_requests - self._request_count)


_singleton: Optional[DiagnosticsConfig] = None
_init_lock = threading.Lock()


def resolve_config() -> DiagnosticsConfig:
    from tools.diagnostics.dump_controller import gear_is_transfer
    if gear_is_transfer():
        dump_dir = os.environ.get("OMNI_KV_DUMP_DIR", "/tmp/kv_dumps")
    else:
        dump_dir = None

    verify = int(os.getenv("OMNI_CACHE_KV_VERIFY", "0")) == 1
    max_requests = int(os.getenv("KV_DUMP_MAX_REQUESTS", "1")) if dump_dir else 1

    groups_str = os.getenv("KV_DUMP_GROUPS")
    groups = (
        [int(g.strip()) for g in groups_str.split(",")]
        if groups_str else None
    )

    layers_str = os.getenv("KV_DUMP_LAYERS")
    layers = (
        [int(l.strip()) for l in layers_str.split(",")]
        if layers_str else None
    )

    components_str = os.getenv("KV_DUMP_COMPONENTS")
    components = (
        [c.strip() for c in components_str.split(",")]
        if components_str else None
    )

    return DiagnosticsConfig(
        dump_dir=dump_dir,
        verify=verify,
        max_requests=max_requests,
        groups=groups,
        layers=layers,
        components=components,
    )


def get_config() -> DiagnosticsConfig:
    global _singleton
    if _singleton is None:
        with _init_lock:
            if _singleton is None:
                _singleton = resolve_config()
    return _singleton


def is_active() -> bool:
    return get_config().active