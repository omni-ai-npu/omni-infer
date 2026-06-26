# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Public entrypoints for cache runtime."""

omni_cache = None


def set_omni_cache(cache):
    global omni_cache
    omni_cache = cache


__all__ = [
    "omni_cache",
    "set_omni_cache",
]
