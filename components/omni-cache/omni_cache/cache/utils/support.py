# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


def sync_h2d_event(omni_cache: Any) -> None:
    """Synchronize H2D event to ensure H2D transfer is complete.

    This should be called in pre_attn before the attention kernel runs,
    to ensure the prefetched KV from the previous layer's post_attn
    has landed in device memory.

    Args:
        omni_cache: The omni_cache instance (PrefillOmniCache or DecodeOmniCache)
    """
    if omni_cache is None:
        return

    # Synchronize the H2D stream first (ensures all H2D ops complete)
    h2d_stream = getattr(omni_cache, "h2d_stream", None)
    if h2d_stream is not None:
        h2d_stream.synchronize()

    # Then synchronize the event (belt-and-braces)
    h2d_event = getattr(omni_cache, "h2d_event", None)
    if h2d_event is not None:
        h2d_event.synchronize()


class BlockHashType(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
    """

    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: tuple[int, ...]
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


# The hash seed for the first block of the prefix block sequence.
try:
    from vllm.utils.hashing import sha256
except Exception:
    sha256 = None

if sha256 is not None:
    NONE_HASH = (
        int.from_bytes(os.urandom(32), byteorder="big")
        if os.getenv("PYTHONHASHSEED") is None
        else sha256(os.getenv("PYTHONHASHSEED"))
    )
else:
    NONE_HASH = 0


@dataclass(frozen=True)
class ParallelCacheState:
    tp_rank: int
    tp_local_rank: int
    tp_world_size: int
    dp_world_size: int
    dp_local_rank: int
    dp_world_size_local: int


@dataclass(frozen=True)
class KVSpecState:
    num_spec_token: int
    is_hybrid_attn: bool
    num_layers: int
    block_size: int
    head_sizes: list[int]
    head_size: int
    dtype: object
    block_num: int
    block_bytes: int


def resolve_parallel_cache_state(vllm_config, get_tp_group, get_dp_group, local_dp_size: int) -> ParallelCacheState:
    tp_group = get_tp_group()
    if vllm_config is not None and hasattr(vllm_config, "parallel_config"):
        raw_dp_world_size = vllm_config.parallel_config.data_parallel_size
        raw_dp_rank = vllm_config.parallel_config.data_parallel_rank
    else:
        dp_group = get_dp_group()
        raw_dp_world_size = getattr(dp_group, "world_size", 1)
        raw_dp_rank = getattr(dp_group, "rank", 0)

    dp_world_size = raw_dp_world_size if isinstance(raw_dp_world_size, int) else 1
    dp_rank = raw_dp_rank if isinstance(raw_dp_rank, int) else 0
    dp_local_rank = dp_rank % local_dp_size if dp_world_size > 1 else 0
    dp_world_size_local = min(dp_world_size, local_dp_size)

    return ParallelCacheState(
        tp_rank=tp_group.rank,
        tp_local_rank=tp_group.local_rank,
        tp_world_size=tp_group.world_size,
        dp_world_size=dp_world_size,
        dp_local_rank=dp_local_rank,
        dp_world_size_local=dp_world_size_local,
    )


def resolve_requested_device(requested_device, torch_module):
    if isinstance(requested_device, str) and requested_device.startswith("npu"):
        if not (hasattr(torch_module, "npu") and torch_module.npu.is_available()):
            return "cpu"
    return requested_device


def resolve_num_spec_token(vllm_config) -> int:
    if vllm_config is not None and getattr(vllm_config, "speculative_config", None):
        return vllm_config.speculative_config.num_speculative_tokens
    return 0


def _is_pangu_v2_model(vllm_config) -> bool:
    """Detect Pangu V2 from hf_config.model_type."""
    if vllm_config is None:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return False
    model_type = getattr(hf_config, "model_type", "")
    return "pangu_v2" in model_type


def _detect_hybrid_attention(vllm_config) -> bool:
    if not vllm_config:
        return False
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", "")
    compress_ratios = getattr(hf_config, "compress_ratios", None)
    return "deepseek" in model_type and compress_ratios is not None and len(set(compress_ratios)) > 1


def _pangu_kv_head_dims_from_hf(hf_config):
    """(c_kv, rope, indexer) per-token KV-cache head dims from hf_config.

    In MLA / DSA style models the per-token KV cache stores the
    compressed-KV part (kv_lora_rank, e.g. 512), the rotary part
    (qk_rope_head_dim, e.g. 64), and — for DSA — an indexer slice
    (index_head_dim, e.g. 128). These three are what the original
    hardcoded [512, 64, 128] / [512, 64] literals referred to; we read
    them from hf_config so they track the real model config.
    Falls back to the DeepSeek-V3 defaults when a field is missing.
    """
    kv_lora = getattr(hf_config, "kv_lora_rank", 0) or 512
    rope = getattr(hf_config, "qk_rope_head_dim", 0) or 64
    indexer = getattr(hf_config, "index_head_dim", 0) or getattr(hf_config, "indexer_head_dim", 0) or 128
    return kv_lora, rope, indexer


def resolve_kv_spec_state(
    kv_cache_config,
    vllm_config,
    enable_dsa: bool,
    enable_c8_indexer: bool,
    mla_attention_spec_type,
    force_hybrid_attn=None,
) -> KVSpecState:
    num_spec_token = resolve_num_spec_token(vllm_config)
    is_hybrid_attn = _detect_hybrid_attention(vllm_config)

    if force_hybrid_attn is not None:
        is_hybrid_attn = force_hybrid_attn

    is_pangu = _is_pangu_v2_model(vllm_config)
    if is_pangu:
        is_hybrid_attn = True
        # Prefer an attention spec with the canonical (block_size=32,
        # num_kv_heads=1) layout so omni-cache's host pool uses the same
        # per-block bytes that OX transfers. DSAAttentionSpec /
        # ShareKVSlidingWindowSpec are the right shape; MomeSpec is not
        # (its block_size is the mamba kernel window, not the attention
        # block_size, and its bytes-per-block differ).
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MLAAttentionSpec,
            AttentionSpec,
        )

        attn_spec = None
        # Prefer any attention-family spec (DSA/MLA/SWA) over MambaSpec.
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, AttentionSpec):
                attn_spec = spec
                break
        if attn_spec is None:
            # Fallback to the first group's spec (MomeSpec case).
            attn_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    else:
        attn_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    if is_hybrid_attn:
        num_layers = max(len(group.layer_names) for group in kv_cache_config.kv_cache_groups)
    else:
        num_layers = sum(len(group.layer_names) for group in kv_cache_config.kv_cache_groups)

    if enable_dsa and not is_pangu:
        _n, _r, _i = _pangu_kv_head_dims_from_hf(vllm_config.model_config.hf_config)
        base_head_sizes = [_n, _r, _i]
    elif is_hybrid_attn:
        if is_pangu:
            # For Pangu hybrid: derive head_size so the host pool is large
            # enough for ALL group types (DSA/SWA/MOME).
            #
            # DSAAttentionSpec and ShareKVSlidingWindowSpec override
            # real_page_size_bytes (no 2× K+V factor), which gives the
            # correct per-block byte count.  Fall back to page_size_padded,
            # then page_size_bytes (which has the 2× factor and only works
            # for the fake FullAttentionSpec used during pre_load).
            from vllm.utils.torch_utils import get_dtype_size

            if hasattr(attn_spec, "page_size_bytes"):
                page_size = attn_spec.page_size_bytes
            else:
                psp = getattr(attn_spec, "page_size_padded", None)
                page_size = psp if psp is not None else attn_spec.page_size_bytes
            effective_head_size = page_size // (attn_spec.block_size * get_dtype_size(attn_spec.dtype))
            base_head_sizes = [effective_head_size]
        elif not enable_c8_indexer:
            base_head_sizes = [attn_spec.head_size]
        else:
            base_head_sizes = [attn_spec.head_size, 8]
    elif isinstance(attn_spec, mla_attention_spec_type):
        _n, _r, _ = _pangu_kv_head_dims_from_hf(vllm_config.model_config.hf_config)
        base_head_sizes = [_n, _r]
    else:
        base_head_sizes = [attn_spec.head_size]

    block_num = kv_cache_config.num_blocks
    block_bytes = kv_cache_config.kv_cache_tensors[0].size // block_num

    head_sizes = [size * attn_spec.num_kv_heads for size in base_head_sizes]

    # Pangu V2 PD: OX transfer requires both sides to use the same
    # block_size. The cache_config patch forces 128, but the
    # attention spec may still report a different value (e.g. 64 on
    # TP=8 prefill).  Use the vllm_config's cache_config.block_size
    # directly so prefill and decode agree.
    if is_pangu and vllm_config is not None:
        force_block_size = vllm_config.cache_config.block_size
        logger.warning(
            "[PANGU-V2-BLOCK-SIZE] attn_spec.block_size=%d, cache_config.block_size=%d, forcing=%d",
            attn_spec.block_size,
            vllm_config.cache_config.block_size,
            force_block_size,
        )
    else:
        force_block_size = attn_spec.block_size

    return KVSpecState(
        num_spec_token=num_spec_token,
        is_hybrid_attn=is_hybrid_attn,
        num_layers=num_layers,
        block_size=force_block_size,
        head_sizes=head_sizes,
        head_size=sum(head_sizes),
        dtype=attn_spec.dtype,
        block_num=block_num,
        block_bytes=block_bytes,
    )


def reset_static_forward_context_kv_cache(kv_cache_config, static_forward_context) -> None:
    if hasattr(kv_cache_config, "kv_cache_groups") and len(kv_cache_config.kv_cache_groups) > 1:
        groups = kv_cache_config.kv_cache_groups
    else:
        groups = [kv_cache_config.kv_cache_groups[0]]

    for group in groups:
        for layer_name in group.layer_names:
            static_forward_context[layer_name].kv_cache = [None]


def resolve_layer_name(kv_cache_config, layer_idx: int):
    """Resolve a global model layer index to a layer name.

    Searches all KV cache groups for a layer name whose extracted index
    matches *layer_idx*.  Returns None if no match is found.

    Args:
        kv_cache_config: The KVCacheConfig object.
        layer_idx: Global model layer index (decoder layer index).

    Returns:
        Layer name string or None.
    """
    from vllm.model_executor.models.utils import extract_layer_index

    for group in kv_cache_config.kv_cache_groups:
        logger.warning(f"<<< {group.layer_names=}")
        for name in group.layer_names:
            try:
                if extract_layer_index(name) == layer_idx:
                    return name
            except Exception:
                continue
    return None


def get_active_prefill_cache():
    """Return the active PrefillOmniCache instance, or None."""
    from omni_cache.cache import omni_cache
    from omni_cache.cache.prefill import PrefillOmniCache

    if not isinstance(omni_cache, PrefillOmniCache):
        return None
    return omni_cache


def get_active_decode_cache():
    """Return the active DecodeOmniCache instance, or None."""
    from omni_cache.cache import omni_cache
    from omni_cache.cache.decode import DecodeOmniCache

    if not isinstance(omni_cache, DecodeOmniCache):
        return None
    if getattr(omni_cache, "record_batch_idx_to_req", None) is None:
        return None
    return omni_cache


def should_compute_prefix_meta(vllm_config, metadata) -> bool:
    """Check whether we should compute prefix_meta for APC.

    Args:
        vllm_config: vLLM configuration
        metadata: Attention metadata

    Returns:
        True if prefix_meta should be computed, False otherwise
    """
    if metadata is None:
        return False
    num_prefills = getattr(metadata, "num_prefills", 0)
    if not vllm_config.cache_config.enable_prefix_caching and not vllm_config.scheduler_config.enable_chunked_prefill:
        return False
    if num_prefills <= 0 and vllm_config.cache_config.enable_prefix_caching:
        # NOTE: currently does not support decode-side APC w/ omnicache
        raise RuntimeError("Currently does not support decode-side APC w/ Omnicache")
    cache = get_active_prefill_cache()
    if cache is None:
        return False
    return True


def attach_prefix_meta_to_metadata(vllm_config, metadata, common_attn_metadata) -> bool:
    """Compute and attach prefix_meta to metadata for APC.

    Args:
        vllm_config: vLLM configuration
        metadata: Attention metadata (will be modified in place)
        common_attn_metadata: Common attention metadata with block tables

    Returns:
        True if prefix_meta was attached, False otherwise
    """
    cache = get_active_prefill_cache()
    if cache is None:
        return False

    prefix_meta = cache.compute_prefix_meta_from_metadata(vllm_config, metadata, common_attn_metadata)
    metadata.prefix_meta = prefix_meta
    return prefix_meta is not None


__all__ = [
    "ParallelCacheState",
    "KVSpecState",
    "resolve_parallel_cache_state",
    "resolve_requested_device",
    "resolve_num_spec_token",
    "resolve_kv_spec_state",
    "reset_static_forward_context_kv_cache",
    "resolve_layer_name",
    "get_active_prefill_cache",
    "get_active_decode_cache",
    "should_compute_prefix_meta",
    "attach_prefix_meta_to_metadata",
]
