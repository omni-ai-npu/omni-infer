# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import entry_points
from typing import Callable, Hashable

import numpy as np
import torch
from vllm.distributed import GroupCoordinator, get_tp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

from omni_npu import envs
from omni_npu.layers.utils import named_stream

logger = init_logger(__name__)
NPU_ATTENTION_BACKEND = {}

# NOTE: Cache for entry points to avoid repeated lookups
_PLUGIN_BACKEND_CACHE = None  # type: dict[str, type] | None


def cache_fit_shape(cache: torch.Tensor, mode: str = "3D"):
    KVCACHE_NZ_DIM = 16  # for 16-bit type
    cache = cache.squeeze(2)
    assert cache.dim() == 3
    _, pg, dim = cache.shape
    if mode == "4D":  # [*, pg, 1, D]
        return cache.view(-1, pg, 1, dim)
    elif mode == "3D":  # [*, pg, D]
        return cache.view(-1, pg, dim)
    else:
        raise ValueError(f"cache_fit_shape err: Unreconized {mode}")


def _maybe_padded_raw_tensor_to_strided_caches(
    raw_tensor: torch.Tensor,
    num_blocks: int,
    block_size: int,
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[torch.dtype, ...],
    page_size_bytes: int,
) -> tuple[torch.Tensor, ...]:
    """
    Creates strided views of a raw memory tensor to represent heterogeneous,
    padded KV cache blocks.

    This function maps a flat 1D raw tensor into multiple multi-dimensional
    cache tensors. It assumes the raw tensor is partitioned into `num_blocks`
    pages of `page_size_bytes`. Within each page, the sub-tensors defined by
    `shapes` and `dtypes` are packed sequentially, followed by potential padding
    to fill the rest of the page.

    Memory Layout per Page (Block):
    [ Tensor 0 ] [ Tensor 1 ] ... [ Tensor N ] [ Padding (optional) ]
    |<- bytes ->|<- bytes ->|    |<- bytes ->|
    |<------------------- page_size_bytes ------------------------->|

    Args:
        raw_tensor (torch.Tensor): The underlying 1D memory pool.
        num_blocks (int): The total number of memory blocks (pages) allocated.
        block_size (int): The number of tokens each block can hold.
        shapes (tuple[tuple[int, ...], ...]): A tuple containing the trailing
            dimensions for each sub-tensor (excluding num_blocks and block_size).
        dtypes (tuple[torch.dtype, ...]): The data types corresponding to each shape.
        page_size_bytes (int): The fixed size of each memory block in bytes.

    Returns:
        cache_tensors (tuple[torch.Tensor, ...]): A tuple of strided tensors sharing the same
            underlying storage as `raw_tensor`. Each tensor has the shape:
            (num_blocks, block_size, *shape).

    Raises:
        AssertionError: If shapes and dtypes lengths mismatch, if type sizes don't
            align, if the raw tensor is too small, or if the page size is insufficient.
    """
    assert len(shapes) == len(dtypes), f"Error! {len(shapes)=} while {len(dtypes)=}."

    # Ensure the raw tensor has enough physical memory
    total_required_bytes = num_blocks * page_size_bytes
    actual_bytes = raw_tensor.numel() * raw_tensor.element_size()
    assert actual_bytes >= total_required_bytes, (
        f"Error! Raw tensor has {actual_bytes} bytes, but {total_required_bytes} bytes are required."
    )

    cache_tensors = []
    storage_offset_bytes = 0

    for shape, dtype in zip(shapes, dtypes):
        dtype_size = dtype.itemsize
        assert page_size_bytes % dtype_size == 0, (
            f"Error! Page size {page_size_bytes} is not a multiple of dtype size {dtype_size}."
        )
        assert storage_offset_bytes % dtype_size == 0, (
            f"Error! Offset {storage_offset_bytes} is not aligned for dtype size {dtype_size}."
        )

        num_element_per_page = page_size_bytes // dtype_size
        target_shape = (num_blocks, block_size, *shape)

        # Get contiguous strides. stride[0] will equal the total number
        # of elements this specific tensor occupies within a single block.
        stride = torch.empty(target_shape).stride()

        # Override the 0th stride to jump by the full page size
        target_stride = (num_element_per_page, *stride[1:])

        tensor = torch.as_strided(
            raw_tensor.view(dtype),
            size=target_shape,
            stride=target_stride,
            storage_offset=storage_offset_bytes // dtype_size,
        )
        cache_tensors.append(tensor)

        # Advance the byte offset by the size this tensor takes up in one block
        storage_offset_bytes += stride[0] * dtype_size

    # The crucial missing check: Did we exceed the allocated page size?
    assert storage_offset_bytes <= page_size_bytes, (
        f"Error! Sub-tensors require {storage_offset_bytes} bytes per block, "
        f"which exceeds the allocated page_size_bytes of {page_size_bytes}."
    )

    return tuple(cache_tensors)


def _load_plugin_backends_map() -> dict[str, type]:
    """NOTE: Scan entry points once and build a name -> class map.

    Each entry point under group ``omni_npu.attention_backends`` is loaded
    and indexed by the value returned by its ``get_name()`` method.
    The result is cached so subsequent calls are free.
    """
    global _PLUGIN_BACKEND_CACHE
    if _PLUGIN_BACKEND_CACHE is not None:
        return _PLUGIN_BACKEND_CACHE

    plugin_map: dict[str, type] = {}
    try:
        eps = entry_points(group="omni_npu.attention_backends")
    except Exception:
        _PLUGIN_BACKEND_CACHE = plugin_map
        return plugin_map

    for ep in eps:
        try:
            backend_cls = ep.load()
            name = backend_cls.get_name()
            plugin_map[name] = backend_cls
            logger.debug("Found plugin backend %s from entry point %s", name, ep.name)
        except Exception as e:
            logger.warning("Failed to load backend plugin %s: %s", ep.name, e)

    _PLUGIN_BACKEND_CACHE = plugin_map
    return plugin_map


def _is_plugin_disabled(backend: str) -> bool:
    """NOTE: Check whether a backend name is listed in the
    ``DISABLE_PLUGIN_BACKENDS`` environment variable.

    The env var is a comma-separated list of backend names that should
    **not** be replaced by their plugin counterparts.  For example::

        DISABLE_PLUGIN_BACKENDS=NPUDSA,NPUMLA

    means the decorator will return the original base class for NPUDSA
    and NPUMLA even if a plugin is available.
    """
    disabled = envs.OMNI_DISABLE_PLUGIN_BACKENDS
    if not disabled:
        return False
    disabled_names = [s.strip() for s in disabled.split(",") if s.strip()]
    return backend in disabled_names


def register_attention_backend(backend: str):

    def decorator(cls: type) -> type:
        if not hasattr(cls, "reshape_kv_cache") or not callable(cls.reshape_kv_cache):
            raise NotImplementedError(
                f"Cannot register '{backend}': {cls.__name__} must implement the "
                "`reshape_kv_cache` static method required by NPU backends."
            )

        attn_module = f"{cls.__module__}.{cls.__qualname__}"
        logger.debug("Register attention %s with module %s", backend, attn_module)
        NPU_ATTENTION_BACKEND[backend] = attn_module
        return cls

    return decorator


def get_attention_backend(name: str) -> str | None:
    """Get a registered attention backend path string by name."""
    return NPU_ATTENTION_BACKEND.get(name)


def load_plugin_backends():
    """
    Load attention backend plugins from entry points.

    This should be called during module initialization.
    Plugins can override the base backends by registering with the same name.
    """
    eps = entry_points(group="omni_npu.attention_backends")
    for ep in eps:
        try:
            backend_cls = ep.load()
            name = backend_cls.get_name()
            attn_module = f"{backend_cls.__module__}.{backend_cls.__qualname__}"
            logger.debug("Register attention plugin %s with module %s", name, attn_module)
            logger.info("Loaded attention backend plugin: %s from %s", name, ep.name)
        except Exception as e:
            logger.warning("Failed to load backend plugin %s: %s", ep.name, e)


def get_available_backends() -> list[str]:
    """List all registered backend names."""
    return list(NPU_ATTENTION_BACKEND.keys())


def apply_plugin_overrides():
    """NOTE: Replace registered backends with their plugin counterparts.

    This must be called **after** all base backends have been imported
    (and thus registered), to avoid circular-import issues — plugin
    modules import base classes from omni_npu, so we cannot load them
    while the base modules are still being imported.

    For each registered backend name, we check whether an entry point
    under ``omni_npu.attention_backends`` provides a class whose
    ``get_name()`` matches.  If it does and the backend is not listed
    in the ``DISABLE_PLUGIN_BACKENDS`` environment variable, the
    plugin class replaces the base class in ``NPU_ATTENTION_BACKEND``
    and is also returned so the caller can rebind module-level names.

    Returns:
        A tuple ``(overrides, base_paths)`` where *overrides* maps
        backend name to the plugin class (only for backends that were
        actually overridden) and *base_paths* is the snapshot of
        ``NPU_ATTENTION_BACKEND`` taken before any plugin was loaded.
    """
    overrides: dict[str, type] = {}

    # Snapshot base paths before loading plugins, because loading
    # plugin modules triggers their @register_attention_backend
    # decorators which would overwrite the base paths.
    base_paths = dict(NPU_ATTENTION_BACKEND)

    # Build the plugin map (cached after first call).
    # This loads plugin modules via entry points, which triggers
    # their @register_attention_backend decorators.
    plugin_map = _load_plugin_backends_map()

    for backend_name, base_module in base_paths.items():
        if _is_plugin_disabled(backend_name):
            # If disabled, restore the base path in case a plugin
            # decorator already overwrote it during _load_plugin_backends_map.
            NPU_ATTENTION_BACKEND[backend_name] = base_module
            logger.debug("Plugin override for %s disabled by env var", backend_name)
            continue

        plugin_cls = plugin_map.get(backend_name)
        if plugin_cls is None:
            continue

        plugin_module = f"{plugin_cls.__module__}.{plugin_cls.__qualname__}"
        if plugin_module == base_module:
            continue  # same class, skip

        logger.info(
            "Plugin backend replaces %s: %s -> %s",
            backend_name,
            base_module,
            plugin_module,
        )
        NPU_ATTENTION_BACKEND[backend_name] = plugin_module
        overrides[backend_name] = plugin_cls

    return overrides, base_paths


def lazy_init(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_registered_lazy_init"):
            self._registered_lazy_init = {}

        def initializer():
            func(self, *args, **kwargs)

        self._registered_lazy_init[func.__name__] = initializer

    return wrapper


def depends_on(init_method):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "_registered_lazy_init"):
                initializer = self._registered_lazy_init.pop(init_method.__name__, lambda: None)
                initializer()
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def support_cache(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        name: str = kwargs.pop("cached", None)
        if name:
            cache_name = f"{func.__name__}.{name}"
            if not hasattr(self, "_cached_list"):
                self._cached_list = {}
            if cache_name in self._cached_list:  # no checking input
                return self._cached_list[cache_name]
        y = func(self, *args, **kwargs)
        if name:
            self._cached_list[cache_name] = y
        return y

    return wrapper


class CrossLayerSharedOp:
    """A metadata-producing op whose output is shared across layers
    within a single forward step.

    Holds one persistent buffer per hashable `caller` key. The first layer in a
    step that calls with `recompute=True` runs the underlying op and
    copies the result into the buffer for its caller; later layers in
    the same step call with `recompute=False` and read the buffer
    directly. Buffer addresses are stable across steps so aclgraph /
    cudagraph captures them safely.

    Producer detection lives on the caller (e.g. a per-layer flag like
    `is_fa_metadata_producer`); this class only manages the buffers.

    ps. copy from omni_npu/v1/layers/attention/npu_pangu.py
    """

    def __init__(
        self,
        op: Callable[..., torch.Tensor],
        shape: tuple[int, ...],
        dtype: torch.dtype,
        callers: tuple[Hashable, ...],
        device: str | torch.device = "npu",
    ):
        self._op = op
        self._buffers: dict[Hashable, torch.Tensor] = {
            caller: torch.empty(shape, dtype=dtype, device=device)
            for caller in callers
        }
        self._default_buffer: torch.Tensor = torch.empty(
            shape, dtype=dtype, device=device,
        )

    def __call__(
        self,
        op_args: dict,
        recompute: bool,
        caller: Hashable,
    ) -> torch.Tensor:
        buffer = self._buffers.get(caller, self._default_buffer)
        if buffer is self._default_buffer or recompute:
            buffer.copy_(self._op(**op_args))
        return buffer


class SPManager:
    """
    This module handles sequence parallelism (SP).

    SP here refers to standard SP splitting, i.e., splitting tokens with ceil division
    based on sp_size.

    CP refers to zigzag-form SP splitting, aimed at adjusting query distribution to
    balance attention computation across ranks.
    """

    def __init__(self, sp_group: GroupCoordinator):
        self.sp_group = sp_group
        self.sp_size = sp_group.world_size
        self.sp_rank = sp_group.rank_in_group
        self.sp_comm = sp_group.device_group
        self.cp_mome_max_tail_append_len: int = 0
        assert self.sp_size > 0

    @staticmethod
    def init_sp(tok: int, sp_group: GroupCoordinator = None):
        sp_group = sp_group or get_tp_group()
        sp_manager = SPManager(sp_group)
        sp_manager._scheme_sp_ctrl(tok)
        return sp_manager

    @staticmethod
    def init_cp(
        cumlens: torch.Tensor,  # [B + 1]
        computed_lens: torch.Tensor,  # [B]
        cumlens_np: np.ndarray = None,  # [B + 1]
        sp_group: GroupCoordinator = None,
        page_size: int = 128,
        table_size: int = 128,
        block_table_ref: torch.Tensor = None,
        mome_kernel_width: int = 3,
        has_chunked_context: bool = False,
        num_spec: int = 0,
    ):
        sp_group = sp_group or get_tp_group()
        if cumlens_np is None:
            cumlens_np = np.array(cumlens.tolist(), dtype=np.int32)
        assert cumlens.dim() == 1 and cumlens_np.ndim == 1
        assert cumlens_np.size == cumlens.size(0)
        assert cumlens.size(0) > 1  # at least 2 elements in a valid batch, [B + 1]

        sp_manager = SPManager(sp_group)
        sp_manager._scheme_sp_ctrl(cumlens_np[-1])
        sp_manager._scheme_cp_attn(cumlens, computed_lens, block_table_ref, page_size, table_size)
        sp_manager._scheme_cp_slice(cumlens_np)
        sp_manager._scheme_cp_reorg(cumlens_np)
        if mome_kernel_width > 1:
            seq_lens = np.diff(cumlens_np)
            if all(seq_len >= 4 * sp_manager.sp_size for seq_len in seq_lens) and not has_chunked_context:
                sp_manager._scheme_cp_mome(seq_lens, mome_kernel_width, num_spec)
        return sp_manager

    # ===================== sp_ctrl =====================
    def _get_buffer(self, key, shape, *, dtype, device):
        """Get a cached buffer, reusing allocation when possible.

        On first call for a given key, allocates with torch.empty.
        On subsequent calls with matching shape/dtype, returns the
        same buffer without any initialization — the caller is
        responsible for writing all positions it will read.
        """
        if not hasattr(self, "_buffers"):
            self._buffers = {}
        buf = self._buffers.get(key)
        if buf is not None and buf.shape == torch.Size(shape) and buf.dtype == dtype:
            return buf
        buf = torch.empty(shape, dtype=dtype, device=device)
        self._buffers[key] = buf
        return buf

    @lazy_init
    def _scheme_sp_ctrl(self, tok: int):
        self.tok = int(tok)
        self.sp_len = cdiv(self.tok, self.sp_size)
        self.sp_align = self.sp_len * self.sp_size
        a = min(self.tok, self.sp_rank * self.sp_len)
        b = min(self.tok, a + self.sp_len)
        self.slice_domain = (a, b)

    @depends_on(_scheme_sp_ctrl)
    def align_tokens(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(0) == self.tok
        if self.tok == self.sp_align:
            return x
        y = x.new_zeros(self.sp_align, *x.shape[1:])
        y[: self.tok] = x
        return y

    @depends_on(_scheme_sp_ctrl)
    @support_cache
    def slice_tokens(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(0) >= self.tok, f"x.size(0): {x.size(0)}, self.tok: {self.tok}"
        a, b = self.slice_domain
        if b == a + self.sp_len:
            return x[a:b]
        y = x.new_zeros(self.sp_len, *x.shape[1:])
        if b > a:
            y[: b - a] = x[a:b]
        return y

    @depends_on(_scheme_sp_ctrl)
    def ag_tokens(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(0) == self.sp_len, f"x.size(0): {x.size(0)}, self.sp_len: {self.sp_len}"
        return self.sp_group.all_gather(x, dim=0)[: self.tok]

    # ===================== cp_reorg =====================

    @lazy_init
    def _scheme_cp_reorg(self, cumlens: np.ndarray):  # [B + 1]
        frag_num = self.sp_size * 2
        frag_lens = cdiv(np.diff(cumlens), frag_num)

        ends = cumlens[1:].repeat(2)
        frags = frag_lens.repeat(2)
        frags_base = frags.cumsum() - frags
        sp_len = int(cdiv(cumlens[-1], self.sp_size))
        cp_len = int(frags.sum())

        class slicer:
            def __init__(self, src, dst, cp, raw):
                self.src, self.dst = int(src), int(dst)
                self.cp, self.raw = int(cp), int(raw)
                self.len = 0

            def slice_src(self, x: torch.Tensor):
                return x[self.src:self.src + self.raw]

            def slice_dst(self, x: torch.Tensor):
                return x[self.dst:self.dst + self.raw]

        def _recv(cp: int, sends: list):
            prev = slicer(0, 0, cp, 0)
            sends[0].append(prev)
            left = (
                np.stack(
                    [
                        cumlens[:-1] + cp * frag_lens,
                        cumlens[:-1] + (frag_num - 1 - cp) * frag_lens,
                    ]
                )
                .transpose()
                .flatten()
            )
            right = np.clip(left + frags, a_max=ends, a_min=None)
            for p, a, b in zip(frags_base, left, right):
                while a < b:
                    inc = sp_len - a % sp_len
                    sect = slicer(a % sp_len, p, cp, raw=min(inc, b - a))
                    sends[a // sp_len].append(sect)
                    prev.len = sect.dst - prev.dst
                    prev = sect
                    a += inc
                    p += inc
            prev.len = cp_len - prev.dst

        parallel = range(self.sp_size)
        sends = [[] for sp in parallel]  # [sp, *]
        for cp in parallel:
            _recv(cp, sends)
        sends[0] = [it for it in sends[0] if it.len > 0]

        a2a_map = []  # [sp, cp]
        for sp_sects in sends:
            send_split = [0 for cp in parallel]
            dst = 0
            for sect in sp_sects:
                sect.dst = dst  # post-a2a -> pre-a2a
                dst += sect.len
                send_split[sect.cp] += sect.len
            a2a_map.append(send_split)

        sends = sends[self.sp_rank]
        sp_split = a2a_map[self.sp_rank]
        cp_split = [it[self.sp_rank] for it in a2a_map]

        self.cp_reorg_metadata = (sends, sp_split, cp_split, sp_len, cp_len)

    @depends_on(_scheme_cp_reorg)
    @support_cache
    def sp_to_cp(self, sp: torch.Tensor) -> torch.Tensor:
        sends, sp_split, cp_split, sp_len, cp_len = self.cp_reorg_metadata
        assert sp.size(0) == sp_len
        tmp_shape = (sum(sp_split), *sp.shape[1:])
        tmp = self._get_buffer("sp_to_cp", tmp_shape, dtype=sp.dtype, device=sp.device)
        for it in sends:
            it.slice_dst(tmp).copy_(it.slice_src(sp))
        cp = sp.new_empty(cp_len, *sp.shape[1:])
        # split could be all 0, for send-only or recv-only case
        torch.distributed.all_to_all_single(cp, tmp, cp_split, sp_split, group=self.sp_comm)
        return cp  # output zigzag

    @depends_on(_scheme_cp_reorg)
    @support_cache
    def cp_to_sp(self, cp: torch.Tensor) -> torch.Tensor:
        sends, sp_split, cp_split, sp_len, cp_len = self.cp_reorg_metadata
        assert cp.size(0) == cp_len
        tmp = cp.new_empty(sum(sp_split), *cp.shape[1:])
        # split could be all 0, for send-only or recv-only case
        torch.distributed.all_to_all_single(tmp, cp, sp_split, cp_split, group=self.sp_comm)
        sp_shape = (sp_len, *cp.shape[1:])
        sp = self._get_buffer("cp_to_sp", sp_shape, dtype=cp.dtype, device=cp.device)
        for it in sends:
            it.slice_src(sp).copy_(it.slice_dst(tmp))
        return sp

    def sp_to_tp(self, sp_x: torch.Tensor):  # reorg SP/CP to TP
        assert sp_x.dim() == 2  # [T, nD]
        n = self.sp_size
        D = sp_x.size(1) // n
        sp_x = sp_x.view(-1, n, D)  # [T, n, D]
        sp_x = sp_x.transpose(0, 1).contiguous()  # [n, T, D]
        tp_x = torch.empty_like(sp_x)
        torch.distributed.all_to_all_single(
            tp_x.view(n, -1),
            sp_x.view(n, -1),
            [1] * n,
            [1] * n,
            group=self.sp_comm,
        )
        return tp_x.view(-1, D)  # [nT, D]

    # ===================== cp_slice =====================

    @lazy_init
    def _scheme_cp_slice(self, cumlens: np.ndarray):
        frag_num = self.sp_size * 2
        frag_lens = cdiv(np.diff(cumlens), frag_num)

        ends = cumlens[1:].repeat(2)
        frags = frag_lens.repeat(2)
        frags_base = frags.cumsum() - frags

        left = (
            np.stack(
                [
                    cumlens[:-1] + self.sp_rank * frag_lens,
                    cumlens[:-1] + (frag_num - 1 - self.sp_rank) * frag_lens,
                ]
            )
            .transpose()
            .flatten()
        )
        right = np.clip(left + frags, a_max=ends, a_min=None)

        sects = [(dst, src, end - src) for dst, src, end in zip(frags_base, left, right) if src < end]
        cp_len, tok = int(frags.sum()), int(cumlens[-1])
        self.cp_slice_metadata = (sects, cp_len, tok)

    @depends_on(_scheme_cp_slice)
    @support_cache
    def cp_slice(self, x: torch.Tensor) -> torch.Tensor:
        sects, cp_len, tok = self.cp_slice_metadata
        assert x.size(0) >= tok
        y = x.new_zeros(cp_len, *x.shape[1:])
        for dst, src, n in sects:
            y[dst:dst + n] = x[src:src + n]
        return y

    # ===================== cp_attn =====================

    @lazy_init
    def _scheme_cp_attn(
        self, cumlens: torch.Tensor, computed_lens: torch.Tensor, blk_table_ref: torch.Tensor, pg: int, tab: int
    ):
        frag_num = self.sp_size * 2
        cumlens = cumlens.to(torch.int32)  # [B + 1]
        seq_lens = cumlens.diff()  # [B]
        frag_lens = cdiv(seq_lens, frag_num)  # [B]
        seq_blks = cdiv(seq_lens, pg)  # [B]

        kv_lens_1 = computed_lens + frag_lens * (self.sp_rank + 1)  # [B]
        kv_lens_2 = computed_lens + frag_lens * (frag_num - self.sp_rank)  # [B]
        frag_cumlens = frag_lens.cumsum(dim=0, dtype=torch.int32)  # [B]
        self.cp_attn_metadata_half = (frag_cumlens, frag_cumlens, kv_lens_1, kv_lens_2)

        q_lens = frag_lens.repeat_interleave(2, dim=0)  # [2B]
        q_cumlens = q_lens.cumsum(dim=0, dtype=torch.int32)  # [2B]
        kv_lens = torch.stack([kv_lens_1, kv_lens_2])  # [2, B]
        kv_lens = kv_lens.transpose(0, 1).flatten()  # [2B]

        blk_base = seq_blks.cumsum(dim=0, dtype=torch.int32) - seq_blks
        table0 = torch.arange(tab, dtype=torch.int32, device=cumlens.device)
        blk_table = table0.view(1, -1) + blk_base.repeat_interleave(2, dim=0).view(-1, 1)
        cp_block_table = blk_table_ref.repeat_interleave(2, dim=0)
        self.cp_attn_metadata = (q_cumlens, kv_lens, blk_table, cp_block_table)

        tok = int(cumlens[-1])
        seq_lens = seq_lens.tolist()
        align_lens = (seq_blks * pg).tolist()
        self.blk_align_metadata = (seq_lens, align_lens, pg, tok)

    @depends_on(_scheme_cp_attn)
    def cp_attn_meta(self) -> tuple:  # FA once:
        return self.cp_attn_metadata  # q_cumlens, kv_lens, blk_table

    @depends_on(_scheme_cp_attn)
    def cp_attn_meta_2(self) -> tuple:
        return self.cp_attn_metadata_half

    @depends_on(_scheme_cp_attn)
    def page_align(self, x: torch.Tensor) -> torch.Tensor:
        seq_lens, align_lens, pg, tok = self.blk_align_metadata
        assert x.size(0) == tok
        y = x.new_empty(sum(align_lens), *x.shape[1:])
        torch.split_with_sizes_copy(x, seq_lens, out=[it[:n] for it, n in zip(y.split(align_lens), seq_lens)])
        return y.view(-1, pg, 1, x.size(-1))

    # ===================== cp_mome =====================

    def _scheme_cp_mome(self, seq_lens: np.ndarray, mome_kernel_width: int, num_spec: int = 0):
        """
        TP4, req_num = 2 for example, 8 chunks per request in total
        r0 means req0, c0 means chunk0
        before mome hidden_states layout:
        rank0: | r0 c0 | r0 c7 | r1 c0 | r1 c7 |
        rank1: | r0 c1 | r0 c6 | r1 c1 | r1 c6 |
        before mome suffix exchange suffix layout:
        rank0: | r0 c0 suffix(2) | r0 c7 suffix(2) | r1 c0 suffix(2) | r1 c7 suffix(2) |
        rank1: | r0 c1 suffix(2) | r0 c6 suffix(2) | r1 c1 suffix(2) | r1 c6 suffix(2) |
        after mome suffix exchange suffix layout:
        rank0: | r0 c6 suffix(2) | r1 c6 suffix(2) |
        rank1: | r0 c0 suffix(2) | r0 c5 suffix(2) | r1 c0 suffix(2) | r1 c5 suffix(2) |
        after rank0 mome suffix broadcast hidden_states layout:
        rank0: | r0 c0 | r0 c6 suffix(2) | r0 c7 | r1 c0 | r1 c6 suffix(2) | r1 c7 |
        rank1: | r0 c0 suffix(2) | r0 c1 | r0 c5 suffix(2) | r0 c6 | r0 c7 suffix(4)
        | r1 c0 suffix(2) | r1 c1 | r1 c5 suffix(2) | r1 c6 | r1 c7 suffix(4) |
        after mome suffix exchange hidden_states layout:
        rank0: | r0 c0 | r0 c6 suffix(2) | r0 c7 | r1 c0 | r1 c6 suffix(2) | r1 c7 |
        rank1: | r0 c0 suffix(2) | r0 c1 | r0 c5 suffix(2) | r0 c6 | r0 c7 suffix(4)
        | r1 c0 suffix(2) | r1 c1 | r1 c5 suffix(2) | r1 c6 | r1 c7 suffix(4) |
        rank 0 start_query_loc = [0, (2 *r0_chunk_len + 2),
        (2 * r0_chunk_len + 2) + (2 * r1_chunk_len + 2)]
        rank 1 start_query_loc = [0, (2 * (r0_chunk_len + 2) + 4),
        (2 * (r0_chunk_len + 2) + 4) + (2 * (r1_chunk_len + 2) + 4)]
        after mome restore layout:
        rank0: | r0 c0 | r0 c7 | r1 c0 | r1 c7 |
        rank1: | r0 c1 | r0 c6 | r1 c1 | r1 c6 |
        """

        self.mome_prefix_size = mome_kernel_width - 1
        frag_num = self.sp_size * 2
        cp_query_split_lens = cdiv(seq_lens, frag_num)
        self.cp_mome_phase_split_sizes = tuple(int(req_len) for req_len in cp_query_split_lens.tolist())
        num_reqs = len(self.cp_mome_phase_split_sizes)

        factor = 1 if self.sp_rank == 0 else 2
        core_merged_query_lens = 2 * cp_query_split_lens + factor * self.mome_prefix_size
        # Cache token count m(s) = kernel + max(0, s-1); s=0 is MTP0 (no MTP).
        # tail_append_len = m(s) + prefix_size = kernel + prefix + max(0, s-1).
        # e.g. kernel=3: MTP0 cache 3 / tail 5; MTP1 cache 3 / tail 5; MTP2 cache 4 / tail 6; MTP3 cache 5 / tail 7.
        max_tail_append_len = mome_kernel_width + self.mome_prefix_size + max(0, num_spec - 1)
        self.cp_mome_max_tail_append_len = max_tail_append_len
        # Every rank appends each request's real global tail (up to max_tail_append_len)
        # to the end of phase1, so MOME cache update observes identical tail context.
        req_tail_append_lens = np.minimum(seq_lens, max_tail_append_len)
        merged_query_lens = core_merged_query_lens + req_tail_append_lens
        mome_query_start_loc = np.zeros(num_reqs + 1, dtype=cp_query_split_lens.dtype)
        mome_query_start_loc[1:] = np.cumsum(merged_query_lens)
        self.cp_mome_query_start_loc = torch.tensor(
            mome_query_start_loc,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        self.cp_mome_req_split_sizes = tuple(2 * req_len for req_len in self.cp_mome_phase_split_sizes)
        self.cp_mome_merged_core_split_sizes = tuple(int(merged_len) for merged_len in core_merged_query_lens.tolist())
        self.cp_mome_merged_split_sizes = tuple(int(merged_len) for merged_len in merged_query_lens.tolist())
        self.cp_mome_req_tail_append_lens = tuple(int(append_len) for append_len in req_tail_append_lens.tolist())
        self.cp_mome_seq_lens = tuple(int(seq_len) for seq_len in seq_lens.tolist())
        self.cp_mome_suffix_block_len = num_reqs * self.mome_prefix_size
        self.cp_mome_local_suffix_len = self.cp_mome_suffix_block_len * 2

    def mome_suffix_exchange(self, x: torch.Tensor) -> torch.Tensor:
        # x layout is req-prior: [req0 chunk0][req0 chunk7][req1 chunk0][req1 chunk7]...
        suffix_size = self.mome_prefix_size
        req_chunks = x.split(self.cp_mome_req_split_sizes, dim=0)
        phase0_chunks = [req_chunk[:req_len] for req_chunk, req_len in zip(req_chunks, self.cp_mome_phase_split_sizes)]
        phase1_chunks = [req_chunk[req_len:] for req_chunk, req_len in zip(req_chunks, self.cp_mome_phase_split_sizes)]
        local_suffixes = torch.cat(
            [req_chunk[-suffix_size:] for req_chunk in phase0_chunks]
            + [req_chunk[-suffix_size:] for req_chunk in phase1_chunks],
            dim=0,
        )
        phase0_local_suffix = local_suffixes.narrow(0, 0, self.cp_mome_suffix_block_len)
        all_suffixes = self.sp_group.all_gather(local_suffixes, dim=0)

        local_suffix_len = self.cp_mome_local_suffix_len
        suffix_block_len = self.cp_mome_suffix_block_len

        if self.sp_rank == 0:
            phase0_suffix_chunks = ()
            phase1_suffix = all_suffixes.narrow(0, local_suffix_len + suffix_block_len, suffix_block_len)
        else:
            prev_rank_off = (self.sp_rank - 1) * local_suffix_len
            phase0_suffix = all_suffixes.narrow(0, prev_rank_off, suffix_block_len)
            phase0_suffix_chunks = phase0_suffix.split(suffix_size, dim=0)
            if self.sp_rank == self.sp_size - 1:
                phase1_suffix = phase0_local_suffix
            else:
                next_rank_off = (self.sp_rank + 1) * local_suffix_len + suffix_block_len
                phase1_suffix = all_suffixes.narrow(0, next_rank_off, suffix_block_len)

        phase1_suffix_chunks = phase1_suffix.split(suffix_size, dim=0)
        merged_pieces = []
        for idx, (phase0_chunk, phase1_suffix_chunk, phase1_chunk) in enumerate(
            zip(phase0_chunks, phase1_suffix_chunks, phase1_chunks)
        ):
            if self.sp_rank != 0:
                merged_pieces.append(phase0_suffix_chunks[idx])
            merged_pieces.extend([phase0_chunk, phase1_suffix_chunk, phase1_chunk])
        return torch.cat(merged_pieces, dim=0)

    def append_mome_req_global_tails(
        self,
        x: torch.Tensor,
        cache_key: str = "mome_tail",
    ) -> torch.Tensor:
        tail_len = self.cp_mome_max_tail_append_len
        req_core_sizes = self.cp_mome_merged_core_split_sizes
        req_tail_append_lens = self.cp_mome_req_tail_append_lens
        req_split_sizes = self.cp_mome_phase_split_sizes
        req_seq_lens = self.cp_mome_seq_lens
        frag_num = self.sp_size * 2
        num_reqs = len(req_core_sizes)

        tail_shape = (num_reqs, tail_len, *x.shape[1:])
        local_tail_contrib = self._get_buffer(cache_key, tail_shape, dtype=x.dtype, device=x.device)

        phase0_base = 0 if self.sp_rank == 0 else self.mome_prefix_size
        phase1_base = self.mome_prefix_size if self.sp_rank == 0 else 2 * self.mome_prefix_size
        for req_idx, (req_chunk, req_split_size) in enumerate(zip(x.split(req_core_sizes, dim=0), req_split_sizes)):
            req_tail_len = req_tail_append_lens[req_idx]
            if req_tail_len == 0:
                continue
            local_tail_contrib[req_idx, :req_tail_len].zero_()
            phase0_start = phase0_base
            phase0_end = phase0_start + req_split_size
            phase1_start = phase1_base + req_split_size
            phase1_end = phase1_start + req_split_size
            phase0_chunk = req_chunk[phase0_start:phase0_end]
            phase1_chunk = req_chunk[phase1_start:phase1_end]

            req_seq_len = req_seq_lens[req_idx]
            start_pos = req_seq_len - req_tail_len
            for tail_pos in range(req_tail_len):
                token_pos = start_pos + tail_pos
                chunk_idx = token_pos // req_split_size
                token_off = token_pos - chunk_idx * req_split_size
                if chunk_idx < self.sp_size:
                    owner_rank = chunk_idx
                    if owner_rank == self.sp_rank:
                        local_tail_contrib[req_idx, tail_pos].copy_(phase0_chunk[token_off])
                elif chunk_idx < frag_num:
                    owner_rank = frag_num - 1 - chunk_idx
                    if owner_rank == self.sp_rank:
                        local_tail_contrib[req_idx, tail_pos].copy_(phase1_chunk[token_off])
                else:
                    raise RuntimeError(
                        f"Invalid chunk_idx={chunk_idx} for req_seq_len={req_seq_len}, "
                        f"req_split_size={req_split_size}, frag_num={frag_num}"
                    )

        torch.distributed.all_reduce(
            local_tail_contrib,
            op=torch.distributed.ReduceOp.SUM,
            group=self.sp_comm,
        )

        req_chunks = x.split(req_core_sizes, dim=0)
        merged_pieces = []
        for idx, req_chunk in enumerate(req_chunks):
            req_tail_len = req_tail_append_lens[idx]
            merged_pieces.append(req_chunk)
            if req_tail_len > 0:
                merged_pieces.append(local_tail_contrib[idx, :req_tail_len])
        return torch.cat(merged_pieces, dim=0)

    def mome_split_and_cat(self, merged_output: torch.Tensor) -> torch.Tensor:
        merged_chunks = merged_output.split(self.cp_mome_merged_split_sizes, dim=0)
        restored_chunks = []
        phase0_start = 0 if self.sp_rank == 0 else self.mome_prefix_size
        phase1_base = self.mome_prefix_size if self.sp_rank == 0 else 2 * self.mome_prefix_size
        for req_len, merged_chunk in zip(self.cp_mome_phase_split_sizes, merged_chunks):
            phase0_end = phase0_start + req_len
            phase0_chunk = merged_chunk[phase0_start:phase0_end]
            phase1_start = phase1_base + req_len
            phase1_chunk = merged_chunk[phase1_start:phase1_start + req_len]
            restored_chunks.extend([phase0_chunk, phase1_chunk])
        return torch.cat(restored_chunks, dim=0)


class DummySPManager:
    def __init__(self, sp_group: GroupCoordinator = None):
        sp_group = sp_group or get_tp_group()
        self.sp_size = sp_group.world_size
        # in dummy_run, we assert token_num divisible by sp_size

    def sp_to_cp(self, x: torch.Tensor, cached=None):
        return x

    def cp_to_sp(self, x: torch.Tensor, cached=None):
        return x

    def align_tokens(self, x: torch.Tensor):
        return x

    def page_align(self, x: torch.Tensor):
        return x

    def slice_tokens(self, x: torch.Tensor, cached=None):
        return x[: x.size(0) // self.sp_size].clone()

    def cp_slice(self, x: torch.Tensor, cached=None):
        return x[: x.size(0) // self.sp_size].clone()

    def ag_tokens(self, x: torch.Tensor):
        return torch.cat([x] * self.sp_size)

    def sp_to_tp(self, x: torch.Tensor):
        return x.view(x.size(0) * self.sp_size, -1)

    def cp_attn_meta(self) -> tuple:
        return None, None, None, None

    def mome_suffix_exchange(self, x: torch.Tensor):
        return x

    def append_mome_req_global_tails(self, x: torch.Tensor):
        return x

    def mome_split_and_cat(self, x: torch.Tensor):
        return x


class KVSPMaganer:
    def __init__(
        self,
        q_cumlens: np.ndarray,  # [B+1]
        kv_lens: np.ndarray,  # [B]
        blk_table: torch.Tensor,  # [B, *]
        sp_group: GroupCoordinator = None,
        page_size: int = 128,  # also the interleave size
    ):
        sp_group = sp_group or get_tp_group()

        def as_np(x):
            if isinstance(x, torch.Tensor):
                x = np.array(x.tolist(), dtype=np.int32)
            return x

        q_cumlens = as_np(q_cumlens)
        kv_lens = as_np(kv_lens)
        assert q_cumlens.ndim == 1 and kv_lens.ndim == 1
        assert q_cumlens.size == kv_lens.size + 1, f"{q_cumlens.size} != {kv_lens.size} + 1"
        assert blk_table.size(0) == kv_lens.size, f"{blk_table.shape} {kv_lens}"
        assert page_size > 0

        q_lens = np.diff(q_cumlens)  # [B]
        computed = kv_lens - q_lens  # [B]

        sp_rank = sp_group.rank_in_group
        sp_size = sp_group.world_size
        sp_comm = sp_group.device_group
        sp_len = cdiv(int(q_cumlens[-1]), sp_size)
        cycle = page_size * sp_size  # virtual page size
        tok = q_cumlens[-1]

        self.cfg = {"dtype": torch.int32, "device": blk_table.device}
        self.blank = [blk_table.new_zeros(0)]
        ranks = np.arange(sp_size)

        assert q_cumlens[0] == 0
        assert all(computed >= 0)
        assert blk_table.size(1) * cycle >= kv_lens.max()

        self.blk_table = blk_table
        self.sp_data = (sp_rank, sp_size, sp_comm, ranks)
        self.seq_data = (q_lens, q_cumlens, kv_lens, computed)
        self.size_data = (page_size, cycle, sp_len, tok)

        self._scheme_reorg()
        self._scheme_ag()

    @lazy_init
    def _scheme_reorg(self):
        sp_rank, sp_size, sp_comm, ranks = self.sp_data
        q_lens, q_cumlens, kv_lens, computed = self.seq_data
        pg, cycle, sp_len, tok = self.size_data

        def num_local(rank: np.ndarray, cnt: np.ndarray, loc=None):
            if loc is None:  # count from start
                full = cnt // cycle * pg
                rest = cnt % cycle - rank * pg
                return full + np.clip(rest, 0, pg)
            return num_local(rank, loc + cnt) - num_local(rank, loc)

        def for_rank(rank: int):
            base_ = q_cumlens - sp_len * rank  # [B+1]
            base = np.clip(base_, 0, sp_len)  # [B+1]
            cnts = np.diff(base)  # [B]
            locs = (base - base_)[:-1] + computed  # [B]
            rnks = ranks.reshape(-1, 1)  # [p, 1]
            sends = num_local(rnks, cnts, locs)  # [p, B], broadcast
            return sends, base, locs

        rank_dat = [for_rank(i) for i in ranks]
        a2a_map = [sends.sum(axis=1) for sends, _, _ in rank_dat]
        send_split = [int(a2a_map[sp_rank][i]) for i in ranks]
        recv_split = [int(a2a_map[i][sp_rank]) for i in ranks]

        def local_section(rank: int, cnt: int, base: int, loc: int):
            offset = int(rank + 1) * pg
            loc = int(loc + cycle - offset) % cycle
            loc0 = int(cycle - pg)
            idx, pre = max(0, loc - loc0), max(0, loc0 - loc)

            def slice_fn(template: torch.Tensor):
                return template[idx:idx + cnt] + (base + pre - idx)

            return slice_fn, cdiv(int(idx + cnt), pg)  # required n_cycle

        sects = []
        sends, bases, locs = rank_dat[sp_rank]
        for rank, cnts in enumerate(sends):  # for each rank
            for cnt, base, loc in zip(cnts, bases, locs):  # for each req
                if cnt > 0:
                    sects.append(local_section(rank, cnt, base, loc))

        local_token_counts = num_local(sp_rank, q_lens, computed)  # [B]
        slice_sects = [
            local_section(sp_rank, *args)  # for each req
            for args in zip(local_token_counts, q_cumlens[:-1], computed)
        ]

        n_cycle = max([n for _, n in (sects + slice_sects)] or [0])
        pages = torch.arange(pg, **self.cfg)
        temp_base = torch.arange(n_cycle, **self.cfg) * cycle
        # template [0,1, 8,9, 16,17, ...] for sp=4,pg=2
        temp = (pages + temp_base.view(-1, 1)).flatten()

        select = torch.cat([sect(temp) for sect, _ in sects] or self.blank)
        self.reorg_metadata = (select, send_split, recv_split)

        serial = torch.arange(local_token_counts.max(), **self.cfg)

        def paged_idx(tab, idx):
            return tab[idx // pg] * pg + idx % pg

        slot_sects = [
            paged_idx(tab, serial[:cnt] + loc // sp_size) # for each req
            for tab, cnt, loc in zip(self.blk_table, local_token_counts, computed)
        ]
        local_slots = torch.cat(slot_sects or self.blank).to(torch.int64)
        local_slots_2d = torch.stack([local_slots // pg, local_slots % pg], dim=-1)

        local_idx = torch.cat([sect(temp) for sect, _ in slice_sects] or self.blank)
        self.select_metadata = (local_idx, local_slots, local_slots_2d)

    @depends_on(_scheme_reorg)
    def sp_to_local(self, sp_x: torch.Tensor, seperate=False):
        _, _, sp_comm, _ = self.sp_data
        _, _, sp_len, _ = self.size_data
        select, send_split, recv_split = self.reorg_metadata
        assert sp_x.size(0) == sp_len
        send = sp_x[select]

        def comm():
            recv = sp_x.new_empty(sum(recv_split), *sp_x.shape[1:])
            torch.distributed.all_to_all_single(recv, send, recv_split, send_split, group=sp_comm)
            return recv

        return comm if seperate else comm()

    @depends_on(_scheme_reorg)
    @support_cache
    def select_local(self, x: torch.Tensor):
        *_, tok = self.size_data
        assert x.size(0) == tok
        local_idx, _, _ = self.select_metadata
        return x[local_idx]

    @depends_on(_scheme_reorg)
    def local_slots(self):
        _, slots, slots_2d = self.select_metadata
        return slots, slots_2d

    @lazy_init
    def _scheme_ag(self):
        sp_rank, sp_size, sp_comm, ranks = self.sp_data
        q_lens, q_cumlens, kv_lens, computed = self.seq_data
        pg, cycle, sp_len, tok = self.size_data

        local_blks = cdiv(kv_lens, cycle)  # [B]
        req_pages = [
            tab[:cnt]
            for tab, cnt in zip(self.blk_table, local_blks)  # for each req
        ]
        local_pages = torch.cat(req_pages or self.blank)

        local_blks_ = torch.tensor(local_blks, **self.cfg)  # [B]
        global_blks = local_blks_.repeat(sp_size)  # [B*p]
        bases = global_blks.cumsum(dim=0) - global_blks  # [B*p]
        bases = bases.view(sp_size, -1).transpose(0, 1)  # [B, p]
        table0 = torch.arange(self.blk_table.size(1), **self.cfg)  # [*]
        blk_table = table0.view(1, -1, 1) + bases.unsqueeze(1)  # [B, *, p]
        blk_table = blk_table.reshape(bases.size(0), -1)  # [B, *p]
        blk_table = blk_table.to(torch.int32)  # standard
        blk_table_cp = blk_table.repeat_interleave(2, dim=0)  # for cp

        self.ag_metadata = (local_pages, blk_table, blk_table_cp)

    @depends_on(_scheme_ag)
    def ag_pages(self, cache: torch.Tensor, seperate=False):
        pg, *_ = self.size_data
        _, sp_size, sp_comm, _ = self.sp_data
        local_pages, blk_table, blk_table_cp = self.ag_metadata
        assert cache.dim() in [3, 4] and cache.size(1) == pg
        send = cache[local_pages]

        def comm():
            recv = send.new_empty(send.size(0) * sp_size, *cache.shape[1:])
            torch.distributed.all_gather_into_tensor(recv, send, group=sp_comm)
            return recv

        return (comm if seperate else comm()), blk_table, blk_table_cp


class DummyKVSPMaganer:
    def sp_to_local(self, sp_x: torch.Tensor, seperate=False):
        y = sp_x.new_zeros(0, *sp_x.shape[1:])

        def comm():
            return y

        return comm if seperate else comm()

    def select_local(self, x: torch.Tensor, **kw):
        return x.new_zeros(0, *x.shape[1:])

    def local_slots(self):
        return None, None

    def ag_pages(self, cache: torch.Tensor, seperate=False):
        def comm():
            return None

        return (comm if seperate else comm()), None, None


def get_batch_desc(attn_metadata=None, layer_idx=-1):
    if attn_metadata is None:  # dummy_run
        return slice(0, None), slice(0, 0), True, False

    num_actual_tokens = attn_metadata.num_actual_tokens
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_prefill_tokens = num_actual_tokens - num_decode_tokens

    d_slice = slice(0, num_decode_tokens)
    p_slice = slice(num_decode_tokens, num_actual_tokens)
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    assert has_decode == bool(num_decode_tokens > 0)
    assert has_prefill == bool(num_prefill_tokens > 0)

    slot_mapping_2d = getattr(attn_metadata, "slot_mapping_2d", None)
    if slot_mapping_2d is None and hasattr(attn_metadata, "get_slot_mapping_2d"):
        if layer_idx == -1:
            slot_mapping_2d = attn_metadata.get_slot_mapping_2d()
        else:
            slot_mapping_2d = attn_metadata.get_slot_mapping_2d(layer_idx)

    def slice_slot_mapping(m, s):
        if m is not None:
            m.slot_mapping = attn_metadata.slot_mapping[s]
            if slot_mapping_2d is not None:
                m.slot_mapping_2d = slot_mapping_2d[s]

    slice_slot_mapping(attn_metadata.prefill, p_slice)
    slice_slot_mapping(attn_metadata.decode, d_slice)
    return p_slice, d_slice, has_prefill, has_decode


def lazy_zero_like(like: torch.Tensor):
    class Tensor:
        def __init__(self, like: torch.Tensor):
            self.val = None
            self.shape = like.shape
            self.cfg = {"device": like.device, "dtype": like.dtype}

        def tensor(self):
            if self.val is None:
                self.val = torch.zeros(self.shape, **self.cfg)
            return self.val

        def __setitem__(self, key, value):
            if self.val is None and self.shape == value.shape:
                self.val = value  # directly ref
            else:
                self.tensor()[key] = value

    return Tensor(like)


@contextmanager
def sp_disabled(
    self: "Attention",
    hidden_states: torch.Tensor,
    sp_group: GroupCoordinator = None,
):
    assert hasattr(self, "ena_sp")
    assert hasattr(self, "o_proj")
    out = lazy_zero_like(hidden_states)
    if self.ena_sp:
        self.ena_sp = False
        if hasattr(self.o_proj, "y_transform"):
            x_transform = getattr(self.o_proj, "x_transform", None)
            y_transform = self.o_proj.y_transform
            if (
                y_transform == "ReduceScatter"
                and x_transform != "DP2TPAll2All"
            ):
                self.o_proj.y_transform = "AllReduce"

        sp_group = sp_group or get_tp_group()
        x = sp_group.all_gather(hidden_states, dim=0)
        y = lazy_zero_like(x)
        yield x, y, out
        y = y.tensor().split(hidden_states.size(0))
        out[:] = y[sp_group.rank_in_group]

        self.ena_sp = True
        if hasattr(self.o_proj, "y_transform"):
            self.o_proj.y_transform = y_transform
    else:
        yield hidden_states, out, out


def paged_scatter(
    slot_mapping: torch.Tensor,  # [T]
    cumlens: np.ndarray,  # [B + 1]
    computed: np.ndarray = None,  # [B]
    pg: int = 128,
):
    if isinstance(cumlens, torch.Tensor):
        cumlens = np.array(cumlens.tolist(), dtype=np.int32)
    assert cumlens.ndim == 1 and cumlens.size > 1
    assert slot_mapping.dim() == 1
    assert slot_mapping.size(0) == cumlens[-1]

    seq_lens = np.diff(cumlens, axis=0)

    if computed is not None:
        assert computed.ndim == 1
        assert seq_lens.size() == computed.size()
        pre_lens = computed % pg
    else:
        pre_lens = seq_lens * 0

    pre_odds = (pg - pre_lens) % pg
    pre_odds = np.clip(pre_odds, a_min=0, a_max=seq_lens)
    full_base = cumlens[:-1] + pre_odds
    full_lens = (seq_lens - pre_odds) // pg * pg
    full_ends = full_base + full_lens
    odd_base = np.concatenate(([0], full_ends))
    odd_lens = np.concatenate((full_base, [cumlens[-1]])) - odd_base

    def extraction(base: np.ndarray, lens: np.ndarray):
        return [(int(src), int(src + num)) for src, num in zip(base, lens) if num > 0]

    def extract(x: torch.Tensor, frags: list):
        if len(frags) == 0:
            return x.new_empty(0)
        return torch.cat([x[a:b] for a, b in frags])

    ori_len = int(cumlens[-1])
    odd_frags = extraction(odd_base, odd_lens)
    full_frags = extraction(full_base, full_lens)

    odds_slots = extract(slot_mapping, odd_frags)
    full_slots = extract(slot_mapping, full_frags)
    page_idx = full_slots[::pg].contiguous() // pg

    def full_pages(x: torch.Tensor):  # [T, ...]
        assert x.dim() >= 2 and x.size(0) == ori_len
        x = extract(x, full_frags).view(-1, pg, *x.shape[1:])
        return x, page_idx  # [*, pg, ...], [*]

    def odd_tokens(x: torch.Tensor):  # [T, ...]
        assert x.dim() >= 2 and x.size(0) == ori_len
        return extract(x, odd_frags), odds_slots  # [T, ...], [T]

    return full_pages, odd_tokens


def paged_cache(
    slot_mapping: torch.Tensor,  # [T]
    cumlens: np.ndarray,  # [B + 1]
    computed: np.ndarray = None,  # [B]
    pg: int = 128,
):
    full_pages, odd_tokens = paged_scatter(slot_mapping, cumlens, computed, pg)

    def cache_fn(x: torch.Tensor, cache: torch.Tensor):
        assert x.dim() == 2 and cache.dim() >= 3
        assert cache.size(-1) == x.size(-1)
        D = x.size(-1)
        odd_x, odd_slots = odd_tokens(x)  # [T, D],     [D]
        full_x, page_idx = full_pages(x)  # [*, pg, D], [D]
        odd_x = odd_x.view(-1, D)
        cache.view(-1, D)[odd_slots] = odd_x
        cache.view(-1, pg * D)[page_idx] = full_x.view(-1, pg * D)

    return cache_fn


def simple_conv(
    x: torch.Tensor,
    w: torch.Tensor,
    states: torch.Tensor,
    prefix: torch.Tensor,
    cumlens: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    args = [x, w, states, prefix, cumlens]
    assert [it.dim() for it in args] == [2, 2, 3, 1, 1]
    assert states.size(1) == w.size(0)
    assert len({it.size(-1) for it in args[:3]}) == 1
    assert {it.dtype for it in args[3:]} == {torch.int32}

    return torch.ops.custom.npu_ai_infra_fused_causal_conv1d(
        x,
        w,
        states,
        query_start_loc=cumlens,
        num_computed_tokens=prefix,
        residual_connection=1,
        block_size=256,
        inplace=inplace,
    )


def save_states(
    cache: torch.Tensor,
    index: torch.Tensor,
    states: torch.Tensor,
):
    batch_size, state_len, dim = states.shape
    desc = f"_cache-{states.dtype}-{state_len}"
    if not hasattr(save_states, desc):
        d0 = index.new_zeros(65536)
        d1 = torch._dim_arange(d0[:1024], dim=0).int() * state_len
        setattr(save_states, desc, (d0, d1))
    d0, d1 = getattr(save_states, desc)

    torch.ops.custom.npu_ai_infra_fused_causal_conv1d(
        states.view(-1, dim),
        states.flatten()[: 3 * dim].view(3, dim),
        cache,
        query_start_loc=d1[: batch_size + 1],
        num_computed_tokens=d0[:batch_size],
        cache_indices=index,
        residual_connection=0,
        block_size=state_len,
        inplace=True,
    )


def select_dim0(
    x: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    assert x.dim() > 1 and index.dim() == 1
    assert x[0].is_contiguous()
    buf = x.new_empty(0)
    buf.set_(x.untyped_storage())
    y = buf.view(x.size(0), -1)[index]
    y = y.as_strided(
        [index.size(0), *x.shape[1:]],
        [*x.stride()],
        x.storage_offset(),
    )
    return y.contiguous()


def scheme_conv_sp(
    sp_group: GroupCoordinator,
    cumlens: np.ndarray,
    computed: np.ndarray = None,
    like: torch.Tensor = None,
    block_size: int = 128,
    state_len: int = 3,
    kernel_size: int = 3,
    save_all: bool = True,
) -> tuple[tuple]:
    assert state_len >= kernel_size
    self_rank = sp_group.rank_in_group
    sp_size = sp_group.world_size
    sp_len = int(cdiv(cumlens[-1], sp_size))
    lens = np.diff(cumlens, axis=0)
    batch_size = len(lens)
    pad_cumlens = cumlens
    if sp_len * sp_size > cumlens[-1]:
        pad_cumlens = np.append(cumlens, sp_len * sp_size)
    pad_lens = np.diff(pad_cumlens, axis=0)
    if computed is None:
        computed = lens * 0
    assert all(lens > 0)
    assert computed.size == lens.size

    def token(pos: int, base: int, req: int):
        if pos >= base:
            return pos
        assert pos + state_len >= base
        return (req, pos - base)

    def refer(req: int, pos: int, count: int):
        base = int(cumlens[req])
        start = pos - count
        return [token(start + i, base, req) for i in range(count)]

    def partition(seqs: list, offset: list, unit: int):
        for i, seq in enumerate(seqs):
            pos = int(offset[i])
            end = int(pos + seq)
            while pos < end:
                idx = pos // unit
                nxt = min(end, (idx + 1) * unit)
                is_tail = bool(nxt == end)
                yield i, pos, nxt, idx, is_tail
                pos = nxt

    class Recorder:
        def __init__(self):
            self.convs = []
            self.loads = []
            self.saves = []
            self.refs = defaultdict(lambda: [None])

        def conv(self, toks: tuple, size: int, prefix: int):
            self.loads.extend([self.refs[tok] for tok in toks])
            self.convs.append((size, prefix))

        def save(self, toks: tuple):
            self.saves.extend([self.refs[tok] for tok in toks])

    ranks = [Recorder() for _ in range(sp_size)]
    blocks = [set() for _ in lens]
    saves = deque()

    for req, start, end, rank, _ in partition(pad_lens, pad_cumlens, sp_len):
        if req < batch_size:
            prefix = start + int(computed[req] - cumlens[req])
            ranks[rank].conv(refer(req, start, kernel_size), end - start, prefix)
        else:
            ranks[rank].conv(refer(0, 0, kernel_size), end - start, 0)

    for req, _, end, block, is_tail in partition(lens, computed, block_size):
        if not (is_tail or save_all):
            continue
        tail = int(cumlens[req] - computed[req]) + end
        saves.append(refer(req, tail, state_len))
        blocks[req].add(block)

    a2a_map = []
    sp_blocks = cdiv(len(saves), sp_size)
    for rank, recorder in enumerate(ranks):
        buf = [0] * (batch_size * state_len)
        maps = [{} for _ in ranks]

        for _ in range(sp_blocks):
            recorder.save(saves.popleft() if saves else refer(0, 0, state_len))

        for tok, ref in recorder.refs.items():
            if isinstance(tok, tuple):
                req, offset = tok
                buf[(req + 1) * state_len + offset] = ref
            else:
                maps[tok // sp_len][tok % sp_len] = ref

        recvs = [sorted(it.keys()) for it in maps]
        a2a_map.append(recvs)

        if rank == self_rank:
            for mapping, recv in zip(maps, recvs):
                buf.extend([mapping[key] for key in recv])
            for i, ref in enumerate(buf):
                if isinstance(ref, list):
                    ref[0] = i
            reorg_idx = [ref[0] for ref in recorder.loads + recorder.saves]
            convs = recorder.convs

    send_idx = []
    for dst in a2a_map:
        send_idx.extend(dst[self_rank])
    send_split = [len(dst[self_rank]) for dst in a2a_map]
    recv_split = [len(src) for src in a2a_map[self_rank]]
    save_range = [slice(min(it), max(it) + 1) for it in blocks]

    # Pin on CPU then async H2D. like.new_tensor(list) is a blocking H2D.
    # prefix/cumlens must be int32 for simple_conv.
    send_idx = torch.tensor(send_idx, dtype=torch.int32, device="cpu").pin_memory()
    reorg_idx = torch.tensor(reorg_idx, dtype=torch.int32, device="cpu").pin_memory()
    conv_prefix = torch.tensor(
        [prefix for _, prefix in convs], dtype=torch.int32, device="cpu"
    ).pin_memory()
    conv_cumlens = torch.tensor(
        [0] + [size for size, _ in convs], dtype=torch.int64, device="cpu"
    ).cumsum(0).int().pin_memory()
    if like is not None:
        send_idx = send_idx.to(like.device, non_blocking=True)
        reorg_idx = reorg_idx.to(like.device, non_blocking=True)
        conv_prefix = conv_prefix.to(like.device, non_blocking=True)
        conv_cumlens = conv_cumlens.to(like.device, non_blocking=True)

    a2a_meta = (send_idx, send_split, recv_split)
    conv_meta = (conv_prefix, conv_cumlens, reorg_idx)
    batch_meta = (batch_size, state_len, kernel_size, sp_len, sp_group)
    return a2a_meta, conv_meta, batch_meta, save_range


def conv_sp(
    x: torch.Tensor,
    w: torch.Tensor,
    cache: torch.Tensor,
    init_idx: torch.Tensor,
    save_idx: torch.Tensor,
    metadata: tuple[tuple],
    inplace: bool = False,
):
    current_stream = torch.npu.current_stream()
    sub_stream = named_stream("mome_sp_ag")
    a2a_meta, conv_meta, batch_meta, _ = metadata
    send_idx, send_split, recv_split = a2a_meta
    prefix, cumlens, reorg_idx = conv_meta
    batch_size, state_len, kernel_size, sp_len, sp_group = batch_meta
    dim = x.size(1)

    assert list(x.shape) == [sp_len, dim], f"{x.shape} {[sp_len, dim]}"
    assert list(w.shape) == [kernel_size, dim], f"{w.shape} {[kernel_size, dim]}"
    assert list(cache.shape[1:]) == [state_len, dim], f"{cache.shape[1:]} {[state_len, dim]}"
    assert list(init_idx.shape) == [batch_size], f"{init_idx.shape} {[batch_size]}"

    workspace = x.new_empty(batch_size * state_len + sum(recv_split), dim)
    workspace[: batch_size * state_len] = select_dim0(cache, init_idx).view(-1, dim)
    torch.distributed.all_to_all_single(
        workspace[batch_size * state_len :],
        x[send_idx],
        recv_split,
        send_split,
        group=sp_group.device_group,
    )

    workspace = workspace[reorg_idx]
    load_part = prefix.size(0) * kernel_size
    loads = workspace[:load_part].view(-1, kernel_size, dim)
    saves = workspace[load_part:].view(-1, state_len, dim)
    sub_stream.wait_stream(current_stream)

    y = simple_conv(x, w, loads, prefix, cumlens, inplace)

    with torch.npu.stream(sub_stream):
        gathered_saves = sp_group.all_gather(saves, dim=0)
        current_stream.wait_stream(sub_stream)
    save_states(cache, save_idx, gathered_saves[: save_idx.size(0)])

    return y
