# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

def plugin() -> str | None:
    """
    Entry point for vLLM to discover the NPU platform plugin.

    Returns the fully-qualified class name of the Platform implementation
    if an NPU environment is detected; otherwise returns None so vLLM can
    fall back to other platforms.
    """
    try:
        import torch
    except Exception:
        return None

    try:
        import torch_npu  # noqa: F401
        # If torch_npu imports, assume NPU platform is intended, even if
        # device_count is 0 at this moment (e.g., container init timing).
        return "omni_npu.platform.NPUPlatform"
    except Exception:
        # Fallback: some builds expose torch.npu without separate torch_npu pkg
        if hasattr(torch, "npu"):
            return "omni_npu.platform.NPUPlatform"
        return None
