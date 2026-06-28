# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Dedicated DSA-only host pool, MMU-aliased into HBM (feature-gated).

Design summary
--------------
The primary decode-side hugetlbfs pool (`omni_cache_decode`) carries
every attention group's KV with a per-layer per-block slot of
`head_size_max` bytes (today 704 = kv_lora 512 + qk_rope 64 +
nope 128 padding for SWA).  OX pulls from prefill into that pool.

DSA attention only needs `kv_lora + qk_rope = 576` bytes per slot.
Keeping the full 704-byte slots MMU-aliased into the NPU consumes
more HBM address space than necessary and forces a wider stride than
the DSA kernel's native layout.

This module owns a SECONDARY hugetlbfs file (default
`/dev/hugepages/omni_cache_decode_dsa`) sized exactly for DSA:
``(num_dsa_layers, num_blocks, block_size, head_size)`` with
``head_size = 576`` by default.  The secondary pool is the one
MMU-aliased to HBM; the primary pool stays host-only.  After each OX
pull the decode side copies the leading ``head_size`` bytes of every
pulled DSA block from the primary to the secondary — a plain host
memcpy, no NPU involvement.

Integration points (all driven from DecodeOmniCache):
  * ``pool = DsaSecondaryHostPool(...); pool.open()`` during decode
    cache construction.
  * ``pool.register_mmu(primary_pool)`` right after the primary's
    existing MMU-bind step (reuses primary's `NPUTensorRegister`).
  * ``pool.copy_from_primary(primary.shared_tensor, dsa_layers,
    block_ids)`` from the post-H2D hook in
    ``connector/decode/process_manager.py::_post_success`` OR
    equivalent Pangu V2 post-pull site.

Feature gating (all default off until verified):
    ENABLE_OMNI_CACHE_DSA_SPLIT=1    turn the whole path on
    OMNI_CACHE_DSA_MMAP_PATH         override hugetlbfs path
    OMNI_CACHE_DSA_HEAD_SIZE         override the default 576
"""

from __future__ import annotations

import mmap as _mmap_mod
import os
from typing import Optional, Sequence

import torch
from vllm.logger import init_logger

try:
    from omni_cache.cache.device_backend.ascend import AscendCLStream
except Exception:
    AscendCLStream = None

import ctypes

from .hugepage_ops import (
    create_memory_mapping,
    create_shared_tensor,
    open_hugepage_file,
)

logger = init_logger("vllm.v1.omni")


ENV_ENABLE = "ENABLE_OMNI_CACHE_DSA_SPLIT"
ENV_PATH = "OMNI_CACHE_DSA_MMAP_PATH"
ENV_FILE = "OMNI_CACHE_DSA_MMAP_FILE"
ENV_ALLOW_UNSAFE_PATH = "OMNI_CACHE_ALLOW_UNSAFE_DSA_MMAP_PATH"

DEFAULT_FILE = "omni_cache_decode_dsa"
HUGEPAGE_ROOT = "/dev/hugepages"
# Fallback head size (kv_lora_rank 512 + qk_rope 64) used only
# when hf_config cannot be resolved at construction time.
_FALLBACK_HEAD_SIZE = 576


def is_enabled() -> bool:
    return os.getenv(ENV_ENABLE, "0") == "1"


def resolve_mmap_path() -> str:
    raw = os.getenv(ENV_PATH)
    if raw:
        if os.getenv(ENV_ALLOW_UNSAFE_PATH, "0") == "1":
            return raw
        return _validate_hugepage_path(raw)
    filename = os.getenv(ENV_FILE, DEFAULT_FILE)
    if os.path.basename(filename) != filename:
        raise ValueError(
            f"{ENV_FILE} must be a file name under {HUGEPAGE_ROOT}, got: {filename!r}"
        )
    return _validate_hugepage_path(os.path.join(HUGEPAGE_ROOT, filename))


def _validate_hugepage_path(path: str) -> str:
    root = os.path.realpath(HUGEPAGE_ROOT)
    resolved = os.path.realpath(path)
    try:
        common = os.path.commonpath([root, resolved])
    except ValueError as exc:
        raise ValueError(f"invalid DSA mmap path: {path!r}") from exc
    if common != root:
        raise ValueError(
            f"DSA mmap path must stay under {HUGEPAGE_ROOT}: {path!r}"
        )
    return resolved


def resolve_head_size(hf_config=None) -> int:
    """Return the DSA-only slot head size.

    This is a pure function of the model's hf_config: the DSA KV
    written per token is kv_lora_rank + qk_rope_head_dim. It is not
    user-configurable; if hf_config is missing or does not expose
    those fields, fall back to the historical Pangu V2 value (576).
    """
    if hf_config is not None:
        kv_lora = getattr(hf_config, "kv_lora_rank", None)
        rope = getattr(hf_config, "qk_rope_head_dim", None)
        if isinstance(kv_lora, int) and isinstance(rope, int) and kv_lora > 0 and rope > 0:
            return kv_lora + rope
    return _FALLBACK_HEAD_SIZE


class DsaSecondaryHostPool:
    """Hugetlbfs-backed DSA-only host pool."""

    def __init__(
        self,
        hugepage_path: str,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        head_size: int = 576,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        if min(num_layers, num_blocks, block_size, head_size) <= 0:
            raise ValueError(
                f"bad DSA pool shape: layers={num_layers} blocks={num_blocks} "
                f"block_size={block_size} head_size={head_size}"
            )
        self.hugepage_path = hugepage_path
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_size = head_size
        self.dtype = dtype
        self.device = device
        self.element_size = torch.zeros((), dtype=dtype).element_size()
        self.slot_bytes = block_size * head_size * self.element_size
        self.layer_bytes = num_blocks * self.slot_bytes
        self.total_bytes = num_layers * self.layer_bytes

        self._fd: Optional[int] = None
        self._mmap: Optional[_mmap_mod.mmap] = None
        self.shared_tensor: Optional[torch.Tensor] = None
        self.device_tensor: Optional[torch.Tensor] = None
        self._ascend_stream = None

    # -- lifecycle -----------------------------------------------------
    def open(self) -> None:
        if self.shared_tensor is not None:
            return
        logger.warning(
            "[DSA-SPLIT] open %s total=%.2f MiB shape=(%d,%d,%d,%d)",
            self.hugepage_path, self.total_bytes / (1 << 20),
            self.num_layers, self.num_blocks, self.block_size, self.head_size,
        )
        self._fd = open_hugepage_file(self.hugepage_path)
        self._mmap = create_memory_mapping(self._fd, self.total_bytes)
        raw = create_shared_tensor(
            self._mmap, self.dtype, self.total_bytes,
            self.element_size, 1, 0,
        )
        self.shared_tensor = raw.view(
            self.num_layers, self.num_blocks, self.block_size, self.head_size,
        )

    def register_mmu(self, primary_pool) -> None:
        if self.shared_tensor is None:
            raise RuntimeError("open() must be called before register_mmu()")
        if not int(os.getenv("ENABLE_HOST_MAPPING", "1")):
            return
        reg = getattr(primary_pool, "npu_tensor_register", None)
        if reg is None:
            raise RuntimeError("primary pool has no npu_tensor_register")
        dev_id = primary_pool.device.index if primary_pool.device is not None else 0
        _, self.device_tensor = reg.host_tensor_register(
            self.shared_tensor, device_id=dev_id,
        )

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except (BufferError, ValueError, OSError):
                pass
            self._mmap = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    # -- post-pull copy ------------------------------------------------
    def copy_from_primary(
        self,
        primary_shared_tensor: torch.Tensor,
        dsa_layer_indices: Sequence[int],
        block_ids: Sequence[int],
        primary_device_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        # Copy the kv region of every pulled DSA block from primary to
        # secondary. When primary_device_tensor (primary pool's NPU-aliased
        # kvi_tensors_swap[0]) is given AND the secondary is MMU-aliased
        # (self.device_tensor is set), use aclrtMemcpyAsync D2D on a
        # private stream; otherwise fall back to a host slice assignment.
        if self.shared_tensor is None or primary_shared_tensor is None:
            return
        if not block_ids or not dsa_layer_indices:
            return
        bs = self.block_size
        h = self.head_size
        esize_p = primary_shared_tensor.element_size()
        esize_s = self.element_size
        kv_bytes = bs * h * esize_p
        block_list = sorted(set(int(b) for b in block_ids))
        layers_list = [int(layer) for layer in dsa_layer_indices]
        self._validate_copy_indices(
            primary_shared_tensor,
            primary_device_tensor,
            layers_list,
            block_list,
        )

        use_async = (
            AscendCLStream is not None
            and self.device_tensor is not None
            and primary_device_tensor is not None
        )
        if use_async:
            p_layer_stride_b = primary_device_tensor.stride(0) * esize_p
            p_block_stride_b = primary_device_tensor.stride(1) * esize_p
            s_layer_stride_b = self.shared_tensor.stride(0) * esize_s
            s_block_stride_b = self.shared_tensor.stride(1) * esize_s
            p_base = primary_device_tensor.data_ptr()
            d_base = self.device_tensor.data_ptr()
            if self._ascend_stream is None:
                _stream = AscendCLStream()
                _stream.create()
                self._ascend_stream = _stream
            stream = self._ascend_stream
            for layer in layers_list:
                base_p = p_base + layer * p_layer_stride_b
                base_d = d_base + layer * s_layer_stride_b
                for blk in block_list:
                    src_ptr = ctypes.c_void_p(base_p + blk * p_block_stride_b)
                    dst_ptr = ctypes.c_void_p(base_d + blk * s_block_stride_b)
                    stream.memcpy_async(dst_ptr, kv_bytes, src_ptr, kv_bytes, 3)
            stream.sync()
            return

        L_total = primary_shared_tensor.shape[0]
        B_total = primary_shared_tensor.shape[1]
        flat = primary_shared_tensor.reshape(L_total, B_total, -1)
        kv_view = flat[..., : bs * h].reshape(L_total, B_total, bs, h)
        idx = torch.as_tensor(block_list, dtype=torch.long)
        if layers_list == list(range(self.num_layers)):
            self.shared_tensor[:, idx, :, :] = kv_view[:, idx, :, :]
        else:
            L = torch.as_tensor(layers_list, dtype=torch.long)
            self.shared_tensor[L[:, None], idx[None, :], :, :] = (
                kv_view[L[:, None], idx[None, :], :, :]
            )

    def _validate_copy_indices(
        self,
        primary_shared_tensor: torch.Tensor,
        primary_device_tensor: Optional[torch.Tensor],
        layers_list: Sequence[int],
        block_list: Sequence[int],
    ) -> None:
        if primary_shared_tensor.dim() < 2:
            raise ValueError(
                f"primary_shared_tensor must have at least 2 dims, got "
                f"{tuple(primary_shared_tensor.shape)}"
            )
        primary_layers = int(primary_shared_tensor.shape[0])
        primary_blocks = int(primary_shared_tensor.shape[1])
        if primary_device_tensor is not None and primary_device_tensor.dim() < 2:
            raise ValueError(
                f"primary_device_tensor must have at least 2 dims, got "
                f"{tuple(primary_device_tensor.shape)}"
            )
        for layer in layers_list:
            if layer < 0 or layer >= self.num_layers or layer >= primary_layers:
                raise ValueError(
                    f"DSA layer out of range: layer={layer}, "
                    f"secondary_layers={self.num_layers}, "
                    f"primary_layers={primary_layers}"
                )
        for blk in block_list:
            if blk < 0 or blk >= self.num_blocks or blk >= primary_blocks:
                raise ValueError(
                    f"DSA block out of range: block={blk}, "
                    f"secondary_blocks={self.num_blocks}, "
                    f"primary_blocks={primary_blocks}"
                )



def maybe_build_from_primary(primary_pool) -> Optional[DsaSecondaryHostPool]:
    """Build the secondary pool when the feature is enabled.

    Returns None when the feature is disabled (safe default).
    """
    if not is_enabled():
        return None
    vllm_cfg = getattr(primary_pool, "vllm_config", None)
    hf_cfg = getattr(vllm_cfg.model_config, "hf_config", None) if vllm_cfg else None
    dsa_layers = getattr(hf_cfg, "dsa_layers", None) if hf_cfg else None
    num_dsa = len(dsa_layers) if dsa_layers else getattr(primary_pool, "num_layers", 0)
    if num_dsa <= 0:
        logger.warning("[DSA-SPLIT] cannot resolve DSA layer count; feature disabled")
        return None
    pool = DsaSecondaryHostPool(
        hugepage_path=resolve_mmap_path(),
        num_layers=num_dsa,
        num_blocks=primary_pool.num_blocks,
        block_size=primary_pool.block_size,
        head_size=resolve_head_size(hf_cfg),
        dtype=torch.bfloat16,
        device=primary_pool.device,
    )
    pool.open()
    return pool
