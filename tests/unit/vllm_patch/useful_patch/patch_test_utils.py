"""Shared helpers for VLLM patch tests."""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager


@contextmanager
def applied_patches(classes):
    """Apply patch classes and restore their targets when the test finishes."""
    saved = []
    owners = {}
    for patch_cls in classes:
        target = patch_cls._target
        if target not in owners:
            owners[target] = dict(
                getattr(target, "_omni_npu_applied_patches", {})
            )
        for name in patch_cls._attr_names_to_apply:
            saved.append((target, name, getattr(target, name)))

    for target in owners:
        target._omni_npu_applied_patches = {}

    try:
        for patch_cls in classes:
            patch_cls.apply()
        yield classes
    finally:
        for target, name, value in saved:
            setattr(target, name, value)
        for target, snapshot in owners.items():
            target._omni_npu_applied_patches = snapshot
