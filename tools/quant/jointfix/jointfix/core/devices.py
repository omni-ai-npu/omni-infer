# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Device resolution — imports torch_npu lazily so 'npu' works on Ascend.

torch.device("npu") / .to("npu") only work after `import torch_npu` (it registers
the 'npu' backend). CUDA builds don't have it; CPU-only builds don't either —
hence the best-effort import.
"""
from __future__ import annotations

import torch


def clear_device_cache(device: torch.device) -> None:
    """Release cached allocator memory for an accelerator device."""
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def set_npu_compile_mode() -> None:
    """
    Force precompiled aclnn kernels (jit_compile=False) so NPU ops don't go
    through the online GE/tbe compiler (which can fail to init in-process).

    Idempotent and safe to call repeatedly — needed because some libraries
    (e.g. transformers' trust_remote_code model code) re-import torch_npu and can
    reset the compile mode, so we re-assert it right before the forward loop.
    """
    try:
        import torch_npu
        torch_npu.npu.set_compile_mode(jit_compile=False)
        print("[npu] compile mode: jit_compile=False (aclnn kernels)")
    except (AttributeError, ImportError, RuntimeError) as e:
        print(f"[npu] set_compile_mode skipped: {type(e).__name__}: {e}")


def resolve_device(name: str = "auto") -> torch.device:
    """
    Resolve a device string to a torch.device.

    'auto' -> cuda if available, else npu if available, else cpu.
    'npu'  -> imports torch_npu first (registers the backend).
    """
    name = (name or "auto").lower()
    if name in ("npu", "auto"):
        try:
            import torch_npu  # noqa: F401  registers torch.npu
            # Use precompiled (aclnn) operators instead of online GE/tbe graph
            # compilation (float32 F.conv1d in Pangu MoME otherwise triggers the GE
            # op-compiler, whose tbe init can fail in-process). Re-asserted before
            # the forward loop in the runner (see set_npu_compile_mode).
            set_npu_compile_mode()
        except (ImportError, RuntimeError) as error:
            if name == "npu":
                raise RuntimeError("failed to initialize the NPU backend") from error
            print(f"[npu] auto detection skipped: {type(error).__name__}: {error}")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
        return torch.device("cpu")
    return torch.device(name)


def resolve_devices(name: str = "auto", n: int = 1) -> "list[torch.device]":
    """
    Resolve N devices: [base:0, base:1, ...]. n=1 -> [base]. Multi-device needs
    cuda/npu (cpu is a single device).
    """
    base = resolve_device(name)
    if n <= 1:
        return [base]
    if base.type == "cpu":
        raise ValueError("multi-device requires cuda/npu — cpu is a single device")
    return [torch.device(f"{base.type}:{i}") for i in range(n)]
