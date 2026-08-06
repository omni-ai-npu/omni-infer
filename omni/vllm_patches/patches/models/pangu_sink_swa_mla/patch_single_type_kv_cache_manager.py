# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from vllm.v1.core import single_type_kv_cache_manager
from vllm.v1.core.single_type_kv_cache_manager import (
    SinkFullAttentionManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import SinkMLAAttentionSpec

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


spec_manager_map[SinkMLAAttentionSpec] = SinkFullAttentionManager


@register_patch("single_type_kv_cache_managerPatch", single_type_kv_cache_manager)
class single_type_kv_cache_managerPatch(VLLMPatch):
    _attr_names_to_apply = ['spec_manager_map']

    spec_manager_map = spec_manager_map
