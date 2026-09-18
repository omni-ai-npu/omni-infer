# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fix upstream SinkFullAttentionManager.__init__ signature lag: forward
kwargs to the base class, then reserve the sink blocks."""

from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    SinkFullAttentionManager,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("SinkFullAttentionManagerPatch", SinkFullAttentionManager)
class SinkFullAttentionManagerPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__']

    def __init__(
        self,
        kv_cache_spec,
        **kwargs,
    ):
        # Explicit base call: apply() injects a plain function (no __class__ cell).
        SingleTypeKVCacheManager.__init__(self, kv_cache_spec, **kwargs)
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)
