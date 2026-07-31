# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Patches for vllm.v1.core.kv_cache_utils (Pangu V2 hybrid):

1. OverrideGroupSizePatch — HYBRID_ATTN_GROUP_SIZE env override for group size.
2. MaxMemoryUsageFromGroupsBackport36030 — backport vllm #36030 so hybrid
   startup admission sums the per-request block need over ALL kv-cache groups
   (v0.14.0 only counted groups[0] → EngineCore livelock on near-max requests).
"""

from collections import defaultdict

from typing import Callable

from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import logger, create_kv_cache_group_specs
from vllm.v1.kv_cache_interface import KVCacheSpec

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch


# Create a patched version with HYBRID_ATTN_GROUP_SIZE support
def _get_kv_cache_groups_uniform_page_size_patched(
    kv_cache_spec: dict[str, KVCacheSpec],
):
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Patched to support HYBRID_ATTN_GROUP_SIZE environment variable.
    """

    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.25:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.25 is just a magic number to avoid too many
        # padding layers.
        group_size = max_num_layers
    if (override_group_size := envs.OMNI_HYBRID_ATTN_GROUP_SIZE) > 0:
        logger.warning(
            "Overriding hybrid attention group size from %d to %d.",
            group_size,
            override_group_size,
        )
        group_size = override_group_size
    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


# Register the patch
@register_patch("OverrideGroupSizePatch", kv_cache_utils)
class OverrideGroupSizePatch(VLLMPatch):
    """Patch to add HYBRID_ATTN_GROUP_SIZE support"""

    _attr_names_to_apply = ["_get_kv_cache_groups_uniform_page_size"]

    # Patch start
    _get_kv_cache_groups_uniform_page_size = (
        _get_kv_cache_groups_uniform_page_size_patched
    )
    # patch end


# ---------------------------------------------------------------------------
# Backport of vllm-project/vllm #36030
# v0.14.0 `_max_memory_usage_bytes_from_groups` 的 General case 只用 groups[0]
# 估算一个请求的 KV block 需求，对 hybrid 多 group 模型（DSA + SWA + Mome）系统性
# 低估 → 启动校验 `_check_enough_kv_cache_memory` / auto-fit 放行一个实际装不下
# `max_model_len` 满长请求的配置 → 接近上限的请求被循环调度、EngineCore livelock。
#
# 修复：General case 对【所有 group】求块数再相加，与 concurrency 日志
# `get_max_concurrency_for_kv_cache_config` 的 Σ-over-groups 口径一致（两套公式
# 不再漂移）。UniformTypeKVCacheSpecs 单 group 特例分支本就正确，原样保留。
# ---------------------------------------------------------------------------
def _max_memory_usage_bytes_from_groups_patched(vllm_config, kv_cache_groups):
    """Sum the per-request block requirement over ALL kv-cache groups.

    Backports vllm #36030 onto the v0.14.0 code shape.
    """
    if not kv_cache_groups:
        return 0

    # UniformTypeKVCacheSpecs special case (single group, per-layer specs):
    # already summed correctly upstream — keep as-is.
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, kv_cache_utils.UniformTypeKVCacheSpecs
    ):
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )

    # General case: group_size pools, each shared by one layer per group.
    # #36030 fix: blocks_needed sums over ALL groups (was: only groups[0]).
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = kv_cache_utils.get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    blocks_needed = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )
    return group_size * page_size * blocks_needed


# Register the patch
@register_patch("MaxMemoryUsageFromGroupsBackport36030", kv_cache_utils)
class MaxMemoryUsageFromGroupsBackport36030(VLLMPatch):
    """Backport vllm #36030: sum max-memory over all kv-cache groups.

    Without this, hybrid startup admission under-counts the per-request block
    need (only groups[0]); a config that cannot fit one ``max_model_len``
    request still passes startup checks and the engine livelocks at runtime.
    """

    _attr_names_to_apply = ["_max_memory_usage_bytes_from_groups"]

    # Patch start
    _max_memory_usage_bytes_from_groups = _max_memory_usage_bytes_from_groups_patched
    # patch end


# ---------------------------------------------------------------------------
# 启动 KV cache 容量校验降级为告警
# ---------------------------------------------------------------------------
def _check_enough_kv_cache_memory_patched(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        logger.warning(
            "To serve at least one request with the models's max seq len "
            "(%s), (%s GiB KV "
            "cache is needed, which is larger than the available KV cache "
            "memory (%s GiB). %s"
            "Try increasing `gpu_memory_utilization` or decreasing `max_model_len` "
            "when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details.",
            max_model_len,
            format_gib(needed_memory),
            format_gib(available_memory),
            estimated_msg,
        )


# Register the patch
@register_patch("CheckEnoughKVCacheMemoryWarnOnly", kv_cache_utils)
class CheckEnoughKVCacheMemoryWarnOnly(VLLMPatch):
    """启动 KV cache 容量校验：容量不足时降级为告警而非 raise 退出。

    与 ``MaxMemoryUsageFromGroupsBackport36030`` 配套：后者让容量估算不再低估，
    本 patch 则让“确实装不下”的配置以告警放行、继续启动，而非在启动阶段退出进程。
    """

    _attr_names_to_apply = ["_check_enough_kv_cache_memory"]

    # Patch start
    _check_enough_kv_cache_memory = _check_enough_kv_cache_memory_patched
    # patch end
