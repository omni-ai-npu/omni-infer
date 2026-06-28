# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Omni-Cache Attention Plugin Implementations

This module provides specialized plugin implementations for different
attention types to handle KV cache offload operations.

Supported Plugins:
- CompressMQAAttnPlugin: For compressed multi-query attention
- MLAAttnPlugin: For multi-head latent attention
- GeneralMQAAttnPlugin: For standard multi-query attention
"""

import logging
import os
from typing import Optional, Any

import torch
from vllm.forward_context import get_forward_context

from .base import AttentionPlugin
from omni_cache.cache.utils.support import sync_h2d_event

logger = logging.getLogger(__name__)


class CompressMQAAttnPlugin(AttentionPlugin):
    """Plugin for compressed multi-query attention (DeepSeek-V3 style).

    Handles offload for compressed attention with sparse indexing.
    """

    def __init__(self):
        super().__init__()
        self._omni_cache = None
        self._enabled = False

    @property
    def omni_cache(self) -> Any:
        """Lazy load and cache omni_cache instance."""
        if self._omni_cache is None:
            from omni_cache.cache import omni_cache

            self._omni_cache = omni_cache
        return self._omni_cache

    @property
    def enabled(self) -> bool:
        """Check if omni_cache is enabled."""
        if self._enabled:
            return True
        if self.omni_cache is None:
            return False
        if hasattr(self.omni_cache, "enable"):
            self._enabled = self.omni_cache.enable
        else:
            self._enabled = True
        return self._enabled

    def pre_attn(self, *args, **kwargs) -> None:
        """Pre-attention hook: D2H for prefill phase.

        DSA KV is written to HBM before the attention kernel runs, so it is
        safe to copy to host pool here (before the next layer's kernel might
        overwrite the same HBM region). Decode-phase H2D is handled separately.

        Args:
            *args: Positional arguments (instance typically at args[0])
            **kwargs: Keyword arguments including attn_metadata and layer info
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return
        # Extract instance from args (first positional argument)
        instance = args[0] if args else kwargs.get("instance")

        if instance is None:
            raise ValueError("instance is required in pre_attn")

        # Get forward context for full metadata dictionary
        forward_ctx = get_forward_context()
        if forward_ctx is None:
            raise RuntimeError("forward_context is None")
        attn_metadata = forward_ctx.attn_metadata
        if attn_metadata is None:
            return

        # Get metadata from kwargs (passed from _apply_attention)
        win_metadata = kwargs.get("win_metadata")
        cmp_metadata = kwargs.get("cmp_metadata")

        if win_metadata is None:
            raise ValueError("win_metadata is required for prefill phase")

        num_prefills = getattr(win_metadata, "num_prefills", None)
        if num_prefills is None or num_prefills == 0:
            return

        # Prepare D2H transfer for prefill phase
        self._handle_prefill_d2h(instance, win_metadata, cmp_metadata, attn_metadata)

    def _handle_prefill_d2h(
        self,
        instance: Any,
        win_metadata: Any,
        cmp_metadata: Optional[Any],
        attn_metadata: dict,
    ) -> None:
        """Handle D2H transfer for prefill phase.

        This function prepares the attn_names and attn_metadatas for
        omni_cache.synchronize_d2h_hybrid() and triggers the transfer.

        Args:
            instance: The attention layer instance
            win_metadata: Window attention metadata
            cmp_metadata: Compressed attention metadata (may be None)
            attn_metadata: Full attention metadata dictionary
        """
        # Import current_stream from omni_cache to avoid omni_npu dependency
        try:
            from omni_cache.cache.utils.ops import current_stream
        except Exception as e:
            # Fallback to torch.npu.current_stream if omni_cache import fails
            from torch import npu

            current_stream = npu.current_stream
            logger.warning(
                f"Failed to import current_stream from omni_cache.cache.utils: {e}, "
                "using torch.npu.current_stream as fallback"
            )

        # Create event for synchronization
        main_stream = current_stream()
        kv_event = torch.npu.Event(blocking=False, enable_timing=False)
        kv_event.record(main_stream)

        # Construct prefix_list and attn_metadatas for prefill side d2h
        # Following the original logic in npu_compressed_mqa.py:1194-1201

        attn_names = []

        # Always include window attention prefix
        win_prefix = getattr(instance.attn, "win_prefix", None)
        if win_prefix is not None:
            attn_names.append(win_prefix)

        # Include compressed attention prefix if compress_ratio > 1
        compress_ratio = getattr(instance, "compress_ratio", 1)
        if compress_ratio > 1:
            cmp_prefix = getattr(instance.attn, "cmp_prefix", None)
            if cmp_prefix is not None:
                attn_names.append(cmp_prefix)

        # Include indexer prefix if indexer exists
        indexer = getattr(instance, "indexer", None)
        if indexer is not None:
            k_cache = getattr(indexer, "k_cache", None)
            if k_cache is not None:
                k_cache_prefix = getattr(k_cache, "prefix", None)
                if k_cache_prefix is not None:
                    attn_names.append(k_cache_prefix)

        if not attn_names:
            raise ValueError("No valid attn_names constructed for D2H transfer")

        # Construct attn_metadatas corresponding to attn_names
        attn_metadatas = []
        for prefix in attn_names:
            if prefix == win_prefix:
                attn_metadatas.append(win_metadata)
            elif prefix == (getattr(instance.attn, "cmp_prefix", None) if compress_ratio > 1 else None):
                attn_metadatas.append(cmp_metadata)
            elif indexer is not None and prefix == (getattr(k_cache, "prefix", None) if k_cache else None):
                # Get indexer metadata from attn_metadata dictionary
                if prefix not in attn_metadata:
                    raise KeyError(f"Prefix {prefix} not found in attn_metadata")
                attn_metadatas.append(attn_metadata[prefix])

        # Call omni_cache's D2H function
        if not hasattr(self.omni_cache, "synchronize_d2h_hybrid"):
            raise AttributeError("omni_cache does not have synchronize_d2h_hybrid method")
        self.omni_cache.synchronize_d2h_hybrid(attn_names, attn_metadatas, kv_event)
        logger.info("D2H transfer triggered for layers: %s", attn_names)

    def post_attn(self, *args, **kwargs) -> None:
        """Post-attention hook: D2H done in pre_attn."""
        pass


class MLAAttnPlugin(AttentionPlugin):
    """Plugin for multi-head latent attention (DeepSeek-MLA style).

    Handles offload for MLA attention with latent KV.
    """

    def __init__(self):
        super().__init__()
        self._omni_cache = None
        self._enabled = False

    @property
    def omni_cache(self) -> Any:
        if self._omni_cache is None:
            from omni_cache.cache import omni_cache

            self._omni_cache = omni_cache
        return self._omni_cache

    @property
    def enabled(self) -> bool:
        if self._enabled:
            return True
        if self.omni_cache is None:
            return False
        if hasattr(self.omni_cache, "enable"):
            self._enabled = self.omni_cache.enable
        else:
            self._enabled = True
        return self._enabled

    def pre_attn(self, *args, **kwargs) -> None:
        """Pre-attention: H2D sync + D2H for MLA prefill.

        1. Wait for H2D from previous layer's post_attn to complete
        2. D2H for current layer (MLA KV written to HBM before attention)
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        sync_h2d_event(self.omni_cache)

        self._do_mla_d2h(*args, **kwargs)

    def _do_mla_d2h(self, *args, **kwargs) -> None:
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        # h2d_event.synchronize() is called in pre_attn before this method.

        # Extract instance from args
        if type(self.omni_cache).__name__ != "PrefillOmniCache":
            return
        instance = args[0] if args else kwargs.get("instance")
        attn_metadata = kwargs.get("attn_metadata")
        if attn_metadata is None:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata

        if attn_metadata is None or type(attn_metadata).__name__ == "NPUMLADecodeMetadata":
            return
        if hasattr(attn_metadata, "prefill") and attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
            if getattr(attn_metadata, "num_prefills", 0) == 0:
                return
        elif hasattr(attn_metadata, "decode"):
            return
        else:
            metadata = attn_metadata
        from omni_cache.cache.utils.ops import current_stream

        main_stream = current_stream()
        kv_event = torch.npu.Event(blocking=False, enable_timing=False)
        kv_event.record(main_stream)
        mla_layer_name = None
        if instance is not None and hasattr(instance, "attn") and hasattr(instance.attn, "prefix"):
            mla_layer_name = instance.attn.prefix
        elif instance is not None and hasattr(instance, "prefix"):
            mla_layer_name = f"{instance.prefix}.attn"
        if mla_layer_name is None:
            return
        self.omni_cache.synchronize_d2h(
            attn_names=[mla_layer_name],
            attn_metadatas=[metadata],
            kv_event=kv_event,
        )

        logger.debug("MLA D2H transfer completed for layer %s", mla_layer_name)

    def post_attn(self, *args, **kwargs) -> None:
        """Post-attention hook: H2D for next layer's APC prefix cache.

        After current layer's attention is computed, copy next layer's
        prefix cache from host to device.
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        # Extract instance from args
        instance = args[0] if args else kwargs.get("instance")

        # Get attn_metadata from kwargs
        # NOTE: attn_metadata here is NPUMLAPrefillMetadata (child object), not NPUMLAMetadata (parent)
        # This is because npu_mla_forward passes attn_metadata.prefill to _forward_prefill
        attn_metadata = kwargs.get("attn_metadata")
        if attn_metadata is None:
            return

        # Get prefix_meta - it's set on the prefill metadata object
        if hasattr(attn_metadata, "prefix_meta"):
            prefix_meta = attn_metadata.prefix_meta
        else:
            return
        if prefix_meta is None:
            return

        # Get layer name and trigger H2D for NEXT layer
        mla_layer_name = None
        if instance is not None and hasattr(instance, "attn") and hasattr(instance.attn, "prefix"):
            mla_layer_name = instance.attn.prefix
        elif instance is not None and hasattr(instance, "prefix"):
            mla_layer_name = f"{instance.prefix}.attn"

        if mla_layer_name is not None:
            try:
                self.omni_cache.synchronize_h2d(
                    prefix_meta=prefix_meta,
                    attn_names=[mla_layer_name],
                )
            except Exception as e:
                logger.debug("MLAAttnPlugin.post_attn H2D failed: %s", e)


class GeneralMQAAttnPlugin(AttentionPlugin):
    """Plugin for standard multi-query attention without compression.

    A simpler plugin for basic MQA offload.
    """

    def __init__(self):
        super().__init__()
        self._omni_cache = None
        self._enabled = False

    @property
    def omni_cache(self) -> Any:
        if self._omni_cache is None:
            from omni_cache.cache import omni_cache

            self._omni_cache = omni_cache
        return self._omni_cache

    @property
    def enabled(self) -> bool:
        if self._enabled:
            return True
        if self.omni_cache is None:
            return False
        if hasattr(self.omni_cache, "enable"):
            self._enabled = self.omni_cache.enable
        else:
            self._enabled = True
        return self._enabled

    def pre_attn(self, *args, **kwargs) -> None:
        if not self.enabled:
            return
        # Standard MQA H2D/D2H logic
        pass

    def post_attn(self, *args, **kwargs) -> None:
        if not self.enabled:
            return
        # Standard MQA D2H logic
        pass


class DSAAttnPlugin(AttentionPlugin):
    """Plugin for DSA (Dynamic Sparse Attention) in DeepSeek models.

    Handles KV cache offload for DSA attention with separate prefill/decode paths.
    The plugin should decorate the _forward_prefill method to intercept D2H/H2D calls.

    Original code location (npu_dsa.py:470-486):
        if os.environ.get("ENABLE_OMNI_CACHE", "0") == "1" and attn_metadata is not None:
            from omni_cache.cache import omni_cache
            main_stream = current_stream()
            kv_event = torch.npu.Event(blocking=False, enable_timing=False)
            kv_event.record(main_stream)
            kv_states = [kv_a, k_pe, k_indexer]
            omni_cache.synchronize_d2h(
                kv_states,
                self.layer_idx,
                kv_event
            )
    """

    def __init__(self):
        super().__init__()
        self._omni_cache = None
        self._enabled = False

    @property
    def omni_cache(self) -> Any:
        if self._omni_cache is None:
            from omni_cache.cache import omni_cache

            self._omni_cache = omni_cache
        return self._omni_cache

    @property
    def enabled(self) -> bool:
        if self._enabled:
            return True
        if self.omni_cache is None:
            return False
        if hasattr(self.omni_cache, "enable"):
            self._enabled = self.omni_cache.enable
        else:
            self._enabled = True
        return self._enabled

    def pre_attn(self, *args, **kwargs) -> None:
        """Pre-attention: H2D sync + D2H for DSA prefill.

        1. Wait for H2D from previous layer's post_attn to complete
        2. D2H for current layer (KV written by previous layer's forward)
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        if type(self.omni_cache).__name__ == "PrefillOmniCache":
            sync_h2d_event(self.omni_cache)

        return self._do_d2h_common(*args, **kwargs)

    def _do_d2h_common(self, *args, **kwargs):
        """Shared D2H body — called from pre_attn."""
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        # h2d_event.synchronize() is called in pre_attn before this method.

        # Extract instance from args
        instance = args[0] if args else kwargs.get("instance")

        layer_name = kwargs.get("layer_name")
        if layer_name is None and instance is not None:
            layer_name = getattr(instance, "prefix", None)

        attn_metadata = kwargs.get("attn_metadata")
        if attn_metadata is None:
            return
        if hasattr(attn_metadata, "prefill") and attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
            num_prefills = getattr(attn_metadata, "num_prefills", 0)
            if num_prefills == 0:
                raise ValueError("Expected prefill phase but num_prefills is 0")
        elif attn_metadata.__class__.__name__ == "NPUDSADecodeMetadata" or (
            hasattr(attn_metadata, "decode") and attn_metadata.decode is not None
        ):
            decode = getattr(attn_metadata, "decode", None)

            topk_idx = kwargs.get("topk_idx")
            topk_key = "topk_idx"
            if topk_idx is None:
                topk_idx = kwargs.get("topk_indices")
                topk_key = "topk_indices"

            kv_lens = kwargs.get("kv_lens") or (decode.seq_lens if decode else None)
            block_table = kwargs.get("block_table") or (decode.block_table if decode else None)
            query_cumlens = kwargs.get("q_cumlens") or (decode.query_cumlens if decode else None)

            if not self.omni_cache.enable_gs:
                return
            kv_cache = kwargs.get("kv_cache")
            attn_kwargs = {
                "sparse_indices": topk_idx,
                "actual_seq_lengths_kv": kv_lens,
                "block_table": block_table,
                "value": kv_cache[0].unsqueeze(-2),
                "actual_seq_lengths_query": query_cumlens,
            }
            self.omni_cache.gather_selection(
                attn_kwargs,
                self.omni_cache._layer_name_to_group_and_layer_idx(f"{layer_name}.attn")[1],
                attn_type="DSA",
            )
            kv_cache = (attn_kwargs["value"].squeeze(-2),) + kv_cache[1:]
            result = {
                "kv_cache": kv_cache,
                topk_key: attn_kwargs["sparse_indices"],
            }
            if decode is not None:
                decode.block_table = attn_kwargs["block_table"]
            else:
                result["block_table"] = attn_kwargs["block_table"]
            return result

        else:
            metadata = attn_metadata
        if type(self.omni_cache).__name__ != "PrefillOmniCache":
            return
        from omni_cache.cache.utils.ops import current_stream

        main_stream = current_stream()
        kv_event = torch.npu.Event(blocking=False, enable_timing=False)
        kv_event.record(main_stream)
        dsa_layer_name = None
        if instance is not None and hasattr(instance, "attn") and hasattr(instance.attn, "prefix"):
            dsa_layer_name = instance.attn.prefix
        elif instance is not None and hasattr(instance, "prefix"):
            dsa_layer_name = f"{instance.prefix}.attn"
        if dsa_layer_name is None:
            raise ValueError("Cannot determine DSA layer_name")
        self.omni_cache.synchronize_d2h(
            attn_names=[dsa_layer_name],
            attn_metadatas=[metadata],
            kv_event=kv_event,
        )
        logger.debug("DSA D2H transfer completed for %s", dsa_layer_name)

    def post_attn(self, *args, **kwargs) -> None:
        """Post-attention hook: H2D for next layer's APC prefix cache.

        After current layer's attention is computed, copy next layer's
        prefix cache from host to device.
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        if type(self.omni_cache).__name__ != "PrefillOmniCache":
            return
        instance = args[0] if args else kwargs.get("instance")

        # Get attn_metadata from kwargs
        # NOTE: attn_metadata here is NPUDSAPrefillMetadata (child object), not NPUDSAMetadata (parent)
        # This is because npu_dsa_forward passes attn_metadata.prefill to _forward_prefill
        attn_metadata = kwargs.get("attn_metadata")
        if attn_metadata is None:
            return

        # Get prefix_meta - it's set on the prefill metadata object
        if hasattr(attn_metadata, "prefix_meta"):
            prefix_meta = attn_metadata.prefix_meta
        else:
            return

        if prefix_meta is None:
            return

        # Get layer name and trigger H2D for NEXT layer
        dsa_layer_name = None
        if instance is not None and hasattr(instance, "attn") and hasattr(instance.attn, "prefix"):
            dsa_layer_name = instance.attn.prefix
        elif instance is not None and hasattr(instance, "prefix"):
            dsa_layer_name = f"{instance.prefix}.attn"

        if dsa_layer_name is not None:
            try:
                self.omni_cache.synchronize_h2d(
                    prefix_meta=prefix_meta,
                    attn_names=[dsa_layer_name],
                )
            except Exception as e:
                logger.debug("DSAAttnPlugin.post_attn H2D failed: %s", e)


class MOMEAttnPlugin(AttentionPlugin):
    """Plugin for standard multi-query attention without compression."""

    def __init__(self):
        super().__init__()
        self._omni_cache = None
        self._enabled = False

    @property
    def omni_cache(self) -> Any:
        if self._omni_cache is None:
            from omni_cache.cache import omni_cache

            self._omni_cache = omni_cache
        return self._omni_cache

    @property
    def enabled(self) -> bool:
        if self._enabled:
            return True
        if self.omni_cache is None:
            return False
        if hasattr(self.omni_cache, "enable"):
            self._enabled = self.omni_cache.enable
        else:
            self._enabled = True
        return self._enabled

    def pre_attn(self, *args, **kwargs) -> None:
        """Pre-attention: H2D sync for MOME.

        Sync once per layer at qa_conv (first conv) to ensure H2D from
        previous layer's post_attn has completed before this layer reads.
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        sync_h2d_event(self.omni_cache)

    def post_attn(self, *args, **kwargs) -> None:
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return
        if type(self.omni_cache).__name__ != "PrefillOmniCache":
            return

        # apply_mome_conv() is invoked three times per MoMe layer in
        # order: qa_conv, compresskv_conv, o_conv. The D2H of this
        # layer's post-attention conv_state and the H2D prefetch of
        # the next layer's prefix only need to fire once per layer,
        # after o_conv has finished (i.e. after the full layer is
        # done). Gate on the o_conv identity so the qa/compresskv
        # invocations are no-ops here. (See pre_attn for the
        # corresponding qa_conv-gated h2d_event.synchronize.)
        instance = args[0] if args else kwargs.get("instance")
        conv_layer = kwargs.get("conv_layer")
        if not conv_layer:
            conv_layer = kwargs.get("layer")
        if not conv_layer:
            conv_layer = args[2] if len(args) > 2 else None

        if instance is None or conv_layer is None:
            return
        do_d2h = True
        do_h2d = True
        if conv_layer is not getattr(instance, "o_conv", None):
            do_d2h = False
            do_h2d = False

        # Get attn_metadata from kwargs
        attn_metadata = kwargs.get("mome_metadata")
        if attn_metadata is None:
            return

        # Check if this is prefill phase
        if hasattr(attn_metadata, "prefill") and attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
            num_prefills = getattr(attn_metadata, "num_prefills", 0)
            if num_prefills == 0:
                raise ValueError("Expected prefill phase but num_prefills is 0")
        elif hasattr(attn_metadata, "decode"):
            return
        else:
            # already get prefill_metadata from attn_metadata
            metadata = attn_metadata

        # Import current_stream from omni_cache to avoid omni_npu dependency
        from omni_cache.cache.utils.ops import current_stream

        main_stream = current_stream()
        kv_event = torch.npu.Event(blocking=False, enable_timing=False)
        kv_event.record(main_stream)

        # Get MOME layer name
        mome_layer_name = None
        if instance is not None and hasattr(instance, "prefix"):
            mome_layer_name = f"{instance.prefix}.mome"

        if mome_layer_name is None:
            raise ValueError("Cannot determine MOME layer_name")

        # Offload AFTER apply_mome_conv finished updating the device conv_state
        # for this layer.
        if do_d2h:
            self.omni_cache.synchronize_d2h(attn_names=[mome_layer_name], attn_metadatas=[metadata], kv_event=kv_event)

        logger.debug("MOME D2H transfer completed for layer %s", mome_layer_name)

        # H2D prefetch the next layer's prefix cache so the next attention
        # kernel can read it when it runs. Mirrors what MLAAttnPlugin /
        # DSAAttnPlugin do for their layers.
        prefix_meta = getattr(attn_metadata, "prefix_meta", None)
        if do_h2d and prefix_meta is not None:
            self.omni_cache.synchronize_h2d(
                prefix_meta=prefix_meta,
                attn_names=[mome_layer_name],
            )


class ModelForwardPlugin:
    """Fires at model forward boundaries for PD KV/state handoff.

    - `post_model_forward` (prefill side): snapshot MoMe `conv_state`s
      into the host pool so the decode side picks them up via the
      existing OX KV-pull flow (Task #15).
    - `pre_model_forward` (decode side): restore MoMe `conv_state`s
      from the host pool into device kv_cache before the first decode
      step of each request (Task #16). Currently a no-op — the OX pull
      populates the host pool slots, and MoMe decode reads from
      kv_cache which is separate from the host pool, so we still need
      to wire the restore path.
    """

    def pre_model_forward(self, *args, **kwargs) -> None:  # noqa: D401
        # No-op: the OX KV pull + H2D path already places MoMe state
        # in the correct device slots (see
        # omni_cache/cache/transfer_engine/decode.py around the
        # `MoMe data is packed at offset 0 (HEAD) of the host pool
        # slot` block). The legacy `restore_mome_states` re-write
        # was a workaround that masked an unrelated bug; it was
        # buggy and redundant once the H2D copy was correct.
        return

    def post_model_forward(self, *args, **kwargs) -> None:  # noqa: D401
        # E3 diagnostic: dump prefill KV before ENABLE_OMNI_CACHE check
        if int(os.getenv("OMNI_DUMP_PREFILL_KV", "0")):
            from omni_cache.attn_plugins.implementations import _dump_prefill_kv
            import omni_cache.cache as _cache_mod_e3

            _dump_prefill_kv(getattr(_cache_mod_e3, "omni_cache", None))
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return
        import omni_cache.cache as _cache_mod

        oc = getattr(_cache_mod, "omni_cache", None)
        # Diagnostic dump: OMNI_CACHE_DUMP_HOST_POOL=<path_prefix> writes
        # the full host-pool kvi_tensors to disk exactly once per process.
        # Use to diff volatile-swap-on vs volatile-swap-off runs and
        # localize which layer/block has corrupted bytes.
        self._maybe_dump_host_pool(oc)
        # Early ping so we can confirm dispatch even when oc is None or
        # we're on the decode side. Rate-limited on tp_rank=0 only to
        # avoid log spam.
        if int(os.getenv("OMNI_CACHE_MOME_DEBUG", "0")):
            if oc is None or getattr(oc, "tp_rank", 0) == 0:
                logger.warning(
                    "[MOME-HOOK] post_model_forward fired: oc_type=%s",
                    type(oc).__name__ if oc else "None",
                )
        if oc is None:
            return
        if type(oc).__name__ != "PrefillOmniCache":
            return

        flush = getattr(oc, "flush_pending_d2h", None)
        if flush is not None:
            try:
                flush()
            except Exception as e:
                logger.warning("flush_pending_d2h failed: %s", e)

        # KV diagnostics: probe prefill host pool after all D2H threads
        # have completed (flush_pending_d2h ensures stream synchronization).
        from tools.diagnostics.probes import probe_prefill_host, clear_accumulators

        probe_prefill_host(oc, kvi_tensors=oc.host_cache.kvi_tensors, dp_rank=oc.dp_local_rank)
        clear_accumulators()

        # Investigation 2026-04-13: Pangu V2's `conv` (MoMe) and `attn`
        # (DSA) layers share the same raw_tensor (`shared_by` in the
        # KVCacheTensor). Writing MoMe state at an offset within a block
        # slot would collide with attention's bytes at that slot. Skip
        # the custom snapshot until we confirm how vLLM slices shared
        # memory across the 6 layers — the existing DSA D2H path may
        # already be carrying MoMe state through the shared pages.
        # Gated: the snapshot path is correct in isolation (unit tests
        # pass), but prefill and decode host pools have structurally
        # different layouts (prefill 4D `(dp, blocks, block_size, 704)`
        # vs decode 5D `(dies, layers, blocks, block_size, 1408)`), so
        # byte-level OX transfer between them is mis-aligned. Enable
        # only after unifying the pool shapes or implementing a
        # separate MoMe transfer channel.
        import os as _os_snap

        if not int(_os_snap.getenv("OMNI_CACHE_MOME_SNAPSHOT", "0")):
            return
        snap = getattr(oc, "snapshot_mome_states", None)
        if snap is None:
            return
        try:
            snap()
        except Exception as e:
            logger.exception("MoMe snapshot failed: %s", e)

    def _maybe_dump_host_pool(self, oc) -> None:
        """Diagnostic: dump host-pool kvi_tensors each request to disk.

        Enabled by `OMNI_CACHE_DUMP_HOST_POOL=<path_prefix>`.
        Writes `<prefix>.tp<rank>.req<N>.pt` each call."""
        prefix = os.getenv("OMNI_CACHE_DUMP_HOST_POOL")
        if not prefix:
            return
        if oc is None or type(oc).__name__ != "PrefillOmniCache":
            return
        req_idx = getattr(oc, "_dump_req_idx", 0)
        max_dumps = int(os.getenv("OMNI_CACHE_DUMP_MAX", "8"))
        if req_idx >= max_dumps:
            return
        try:
            host_cache = getattr(oc, "host_cache", None)
            kvi = getattr(host_cache, "kvi_tensors", None) if host_cache else None
            if not kvi:
                return
            tp_rank = getattr(oc, "tp_rank", 0)
            path = f"{prefix}.tp{tp_rank}.req{req_idx}.pt"

            def _flat(obj, out):
                if hasattr(obj, "shape") and hasattr(obj, "detach"):
                    out.append(obj)
                elif isinstance(obj, (list, tuple)):
                    for x in obj:
                        _flat(x, out)

            flat = []
            for entry in kvi:
                _flat(entry, flat)
            # Stash of real block IDs (set by pre_prepare_inputs when swap is on)
            stash = None
            try:
                from vllm.platforms import current_platform as _cp

                ib = _cp.get_input_batch() if hasattr(_cp, "get_input_batch") else None
                if ib is not None:
                    stash = getattr(ib, "_real_block_tables_per_group", None)
                    if stash is not None:
                        stash = [list(x) if x is not None else None for x in stash]
            except Exception:
                pass
            # Per-block CRC32. Host pool shape is (dp_or_1, blocks, block_size, hidden).
            # Blocks live on axis 1 (env overridable). We only hash blocks 0..N-1 (env).
            import zlib

            n_raw = int(os.getenv("OMNI_CACHE_DUMP_NBLOCKS", "0"))
            block_axis = int(os.getenv("OMNI_CACHE_DUMP_BLOCK_AXIS", "1"))
            max_blocks = int(os.getenv("OMNI_CACHE_DUMP_MAXBLK", "32"))
            hashes = []
            raw = []
            for t in flat:
                try:
                    tc = t.detach().contiguous().cpu()
                    ax = block_axis if tc.dim() > block_axis else 0
                    n_b = min(tc.shape[ax], max_blocks)
                    # Move block axis to front for easy slicing
                    tp = tc.movedim(ax, 0) if ax != 0 else tc
                    tb = tp[:n_b].contiguous().view(torch.uint8).reshape(n_b, -1)
                    arr = tb.numpy()
                    h = [int(zlib.crc32(arr[i].tobytes()) & 0xFFFFFFFF) for i in range(n_b)]
                    hashes.append(h)
                    raw.append(tp[:n_raw].clone() if n_raw > 0 else None)
                except Exception as e:
                    hashes.append(f"error: {e}")
                    raw.append(None)
            payload = {
                "tp_rank": tp_rank,
                "req_idx": req_idx,
                "num_kvi": len(flat),
                "full_shapes": [tuple(t.shape) for t in flat],
                "dtypes": [str(t.dtype) for t in flat],
                "block_hashes": hashes,
                "raw_blocks": raw,
                "real_block_tables_per_group": stash,
            }
            torch.save(payload, path)
            oc._dump_host_pool_req_count = req_idx + 1
            setattr(oc, "_dump_req_idx", req_idx + 1)
            logger.warning("[HOST-POOL-DUMP] wrote %s", path)
        except Exception as e:
            logger.exception("[HOST-POOL-DUMP] failed: %s", e)


def _dump_prefill_kv(oc) -> None:
    """Dump SHA-256 of prefill HBM KV for each decoder layer. E3 diagnostic.
    Works with oc=None for baseline mode (uses vllm forward context)."""
    import hashlib
    import json

    out_dir = os.environ.get("OMNI_DUMP_PREFILL_KV_DIR", "/tmp/kv_dumps_prefill")
    os.makedirs(out_dir, exist_ok=True)
    tp_rank = getattr(oc, "tp_rank", 0) if oc else 0

    model = None
    if oc is not None:
        model = getattr(getattr(oc, "runner", None), "model", None)
    if model is None:
        # Fallback: try vllm forward context
        try:
            ctx = get_forward_context()
            model = getattr(ctx, "model", None)
        except Exception:
            pass
    if model is None:
        return

    layers = getattr(getattr(model, "model", None), "layers", None) or getattr(model, "layers", None)
    if layers is None:
        return
    for li, layer in enumerate(layers):
        sa = getattr(layer, "self_attn", None)
        if sa is None:
            continue
        kv = getattr(sa, "kv_cache", None)
        if kv is None:
            continue
        if isinstance(kv, (list, tuple)) and kv:
            kv = kv[0]
        if isinstance(kv, (list, tuple)):
            for ci, t in enumerate(kv):
                if hasattr(t, "shape") and t.numel() > 0:
                    h = hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]
                    with open(f"{out_dir}/prefill_L{li:03d}_c{ci}_tp{tp_rank}.json", "w") as f:
                        json.dump(
                            {"layer": li, "component": ci, "tp_rank": tp_rank, "hash": h, "shape": list(t.shape)}, f
                        )
        elif hasattr(kv, "shape") and kv.numel() > 0:
            h = hashlib.sha256(kv.detach().cpu().numpy().tobytes()).hexdigest()[:16]
            with open(f"{out_dir}/prefill_L{li:03d}_tp{tp_rank}.json", "w") as f:
                json.dump({"layer": li, "tp_rank": tp_rank, "hash": h, "shape": list(kv.shape)}, f)


class MoEFFNPlugin(AttentionPlugin):
    """Plugin for MoE FFN layer-boundary KV pipeline sync.

    Registered via entry-point ``omni.moe_ffn_decorators`` and triggered
    by ``moe_ffn_decorator`` after each ``FusedMoE.forward`` call.
    """

    def __init__(self):
        super().__init__()
        self.omni_cache = None

    def pre_attn(self, *args, **kwargs):
        pass

    def post_attn(self, *args, **kwargs):
        from omni_cache.attn_plugins.base import _moe_post_sync

        try:
            _moe_post_sync()
        except Exception:
            pass


class UpdateFromOutputPlugin:
    def pre_update_from_output(self, *args, **kwargs) -> None:
        pass

    def post_update_from_output(self, *args, result=None, **kwargs) -> None:
        if result is None:
            return

        from vllm.v1.outputs import ModelRunnerOutput

        model_runner_output = kwargs.get("model_runner_output") or (args[2] if len(args) > 2 else None)
        if model_runner_output is None or not isinstance(model_runner_output, ModelRunnerOutput):
            return

        reuse_rate = getattr(model_runner_output, "reuse_rate", None)
        eco = result.get(0)
        if eco is not None and hasattr(eco, "scheduler_stats") and eco.scheduler_stats is not None:
            eco.scheduler_stats.reuse_rate = reuse_rate


class ModelOutputPlugin:
    def pre_model_output(self, *args, **kwargs) -> None:
        pass

    def post_model_output(self, *args, result=None, **kwargs) -> None:
        if result is None:
            return
        result = getattr(result, "_model_runner_output", result)

        try:
            import omni_cache.cache as cache_module

            oc = getattr(cache_module, "omni_cache", None)
            if oc is not None and hasattr(oc, "reuse_rate"):
                result.reuse_rate = oc.reuse_rate.tolist()
            else:
                result.reuse_rate = None
        except ImportError:
            result.reuse_rate = None


__all__ = [
    "CompressMQAAttnPlugin",
    "MLAAttnPlugin",
    "GeneralMQAAttnPlugin",
    "DSAAttnPlugin",
    "MOMEAttnPlugin",
    "MoEFFNPlugin",
    "ModelForwardPlugin",
    "UpdateFromOutputPlugin",
    "ExecuteModelPlugin",
]
