# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import MethodType

from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid import (
    PanguNewKVCacheSpecsPatch,
)

try:
    PanguNewKVCacheSpecsPatch.apply()
except ValueError as exc:
    if "already patched" not in str(exc):
        raise

from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache


class NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _cache(num_lanes=2):
    cache = type("Cache", (), {})()
    cache.req_id_to_idx = {}
    cache.num_max_batch_pool = num_lanes
    cache._lane_lock = NullLock()
    cache.reserve_hbm_lane_for_request = MethodType(
        DecodeOmniCache.reserve_hbm_lane_for_request, cache
    )
    cache.get_hbm_lane_for_request = MethodType(
        DecodeOmniCache.get_hbm_lane_for_request, cache
    )
    cache.release_hbm_lane_for_request = MethodType(
        DecodeOmniCache.release_hbm_lane_for_request, cache
    )
    return cache


def test_reserve_same_request_is_idempotent():
    cache = _cache()

    assert cache.reserve_hbm_lane_for_request("req-a") == 0
    assert cache.reserve_hbm_lane_for_request("req-a") == 0
    assert cache.req_id_to_idx == {"req-a": 0}


def test_reserve_raises_when_lanes_are_exhausted():
    cache = _cache(num_lanes=1)
    cache.reserve_hbm_lane_for_request("req-a")

    try:
        cache.reserve_hbm_lane_for_request("req-b")
    except RuntimeError as exc:
        assert "No free HBM lane" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_release_clears_request_lane_mapping():
    cache = _cache()
    cache.reserve_hbm_lane_for_request("req-a")

    assert cache.release_hbm_lane_for_request("req-a") == 0
    assert cache.req_id_to_idx == {}


def test_release_unknown_request_is_harmless():
    cache = _cache()

    assert cache.release_hbm_lane_for_request("missing") is None
    assert cache.req_id_to_idx == {}
