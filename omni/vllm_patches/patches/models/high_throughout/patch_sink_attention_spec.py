# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, fields

from vllm.v1 import kv_cache_interface
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@dataclass(frozen=True, kw_only=True)
class SinkMLAAttentionSpec(MLAAttentionSpec):
    """MLA KV spec that also reserves static-sink tokens.

    Upstream 0.25.1 has SinkFullAttentionSpec but not this MLA variant.
    Same patch name as pangu_sink_swa_mla/patch_kv_cache_interface so the
    model-dir copy overwrites this registration when it loads.
    """

    sink_len: int = 0

    @classmethod
    def merge(cls, specs: list) -> "SinkMLAAttentionSpec":
        merged = MLAAttentionSpec.merge(specs)
        sink_len_set = {getattr(spec, "sink_len", 0) for spec in specs}
        if len(sink_len_set) != 1:
            raise AssertionError(
                "All SinkMLAAttentionSpec layers in the same KV cache group "
                "must use the same sink_len."
            )
        kwargs = {f.name: getattr(merged, f.name) for f in fields(merged)}
        kwargs["sink_len"] = sink_len_set.pop()
        return cls(**kwargs)


@register_patch("SinkAttentionSpecPatch", kv_cache_interface)
class SinkAttentionSpecPatch(VLLMPatch):
    _attr_names_to_apply = ["SinkMLAAttentionSpec"]
    SinkMLAAttentionSpec = SinkMLAAttentionSpec
