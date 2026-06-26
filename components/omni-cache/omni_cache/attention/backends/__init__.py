# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# omni_cache/attention/backends/__init__.py

import os

# Toggle control: enable extended backend only when VLLM_PLUGINS contains "omni_cache"
# If not included, the extension implementation is not imported, and omni-npu uses the base backend
if int(os.getenv("ENABLE_OMNI_CACHE", "0")):
    from omni_cache.attention.backends.attention_ext import NPUAttentionMetadataBuilderExt
    from omni_cache.attention.backends.dsa_ext import NPUDSABackendExt
    from omni_cache.attention.backends.mla_ext import NPUMLABackendExt
    try:
        from omni_cache.attention.backends.mome_ext import NPUPanguMomeBackendExt
    except ImportError:
        # omni-npu build without the Pangu Mome backend — leave Mome disabled.
        NPUPanguMomeBackendExt = None

    __all__ = [
        "NPUAttentionMetadataBuilderExt",
        "NPUDSABackendExt",
        "NPUMLABackendExt",
        "NPUPanguMomeBackendExt",
    ]
else:
    # When extension is not enabled, export nothing to let omni-npu use the base implementation
    __all__ = []