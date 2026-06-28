# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Extended MLA backend for omni_cache.

Decode: rewrite the MLA decode ``block_table`` / ``slot_mapping`` so
attention addresses the HBM slots that the decode H2D path filled.
Each request holds one HBM lane ``idx_req`` (claimed in
``transfer_engine/decode.py`` and published via ``record_batch_idx_to_req``
/ ``req_id_to_idx``). Within a lane H2D writes slots
``req_offset*idx_req + 1 .. +N``; slot 0 of each lane is a reserved
rotation sentinel.

Per-row layout for a sliding-window MLA attention group:

    row_i = [0] * num_leading_zeros
          + [base_i + 1, base_i + 2, ...]            (base_i = req_offset * idx_reqs[i])

where ``num_leading_zeros`` is the count of blocks that have rolled out of
the window (``max(0, num_computed - sliding_window + 1) // block_size``).
Slot mapping for the single decode token in row i is

    slot_i = (seq_len_i - 1) - num_leading_zeros * block_size
           + block_size                              # lane blocks start at id 1
           + req_offset * idx_reqs[i] * block_size   # lane base (token offset)

Prefill: swap in PrefillOmniCache's volatile block_table (Task #12) — the
behaviour this file had before the refactor.
"""

import dataclasses
import os
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from omni_npu.attention.backends.mla import NPUMLABackend, NPUMLAMetadataBuilder, NPUMLA
from omni_npu.attention.backends.utils import register_attention_backend
from omni_cache.cache.utils.support import (
    get_active_prefill_cache,
    get_active_decode_cache,
    should_compute_prefix_meta,
    attach_prefix_meta_to_metadata,
)

logger = init_logger(__name__)

ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))
USE_OMNI_INPUT_BATCH = int(os.getenv("USE_OMNI_INPUT_BATCH", "0"))


def _leading_zero_blocks(num_computed: int, sliding_window, block_size: int) -> int:
    if not sliding_window or block_size <= 0:
        return 0
    return max(0, int(num_computed) - int(sliding_window) + 1) // int(block_size)


def _lane_req_offset(spec, vllm_config) -> int:
    from omni_cache.cache.decode.hbm_buffer_utils import _req_offset_for_spec

    return _req_offset_for_spec(spec, max_model_len=vllm_config.model_config.max_model_len)


if ENABLE_OMNI_CACHE and not USE_OMNI_INPUT_BATCH:
    import torch

    from omni_cache.cache import omni_cache
    from omni_cache.cache.prefill import PrefillOmniCache
    from omni_cache.attention.metadata import (
        resolve_batch_idx_reqs,
    )

    class NPUMLAMetadataBuilderExt(NPUMLAMetadataBuilder):
        """MLA metadata builder with HBM-lane awareness for decode."""

        def __init__(
            self,
            kv_cache_spec,
            layer_names: list[str],
            vllm_config,
            device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self.layer_names = layer_names
            slot_capacity = self.vllm_config.scheduler_config.max_num_batched_tokens
            self._slot_map_pinned = torch.full(
                (slot_capacity,),
                PAD_SLOT_ID,
                dtype=torch.int64,
                pin_memory=True,
            )

        # --- decode ---
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):

            cm = common_attn_metadata
            block_table_tensor, slot_mapping = self._rewrite_decode_block_table(
                cm.block_table_tensor,
                cm.seq_lens,
                cm.query_start_loc,
                cm.slot_mapping,
            )
            if block_table_tensor is not None:
                cm = dataclasses.replace(
                    cm,
                    block_table_tensor=block_table_tensor,
                    slot_mapping=slot_mapping,
                )
            # Prefill-side: Force prefill only construct prefill metadata
            if get_active_prefill_cache() is not None and getattr(self, "reorder_batch_threshold", 1) != 0:
                self.reorder_batch_threshold = 0

            metadata = super().build(common_prefix_len, cm, fast_build)

            if metadata.prefill is not None:
                # APC: compute prefix_meta for H2D
                # IMPORTANT: prefix_meta must be set on metadata.prefill (not parent metadata)
                # because npu_mla_forward passes attn_metadata.prefill to _forward_prefill
                # which then passes it to _apply_attention_sink_chunked_prefill decorated by @mla_attn_decorator
                if self._should_add_prefix_meta(metadata):
                    prefix_meta = self._add_prefix_meta(metadata, common_attn_metadata)
                    # Set on prefill (child) object, not parent
                    setattr(metadata.prefill, "prefix_meta", prefix_meta)
            return metadata

        def _should_add_prefix_meta(self, metadata) -> bool:
            """Check whether we should compute prefix_meta for APC."""
            return should_compute_prefix_meta(self.vllm_config, metadata)

        def _add_prefix_meta(self, metadata, common_attn_metadata):
            """Compute prefix_meta for SWA/MLA and attach to metadata.

            Also triggers H2D for Layer 0 - since post_attn prefetches the NEXT layer,
            Layer 0's data must be loaded during build before any forward pass.
            """
            active_cache = get_active_prefill_cache()

            attach_prefix_meta_to_metadata(self.vllm_config, metadata, common_attn_metadata)

            prefix_meta = getattr(metadata, "prefix_meta", None)

            # H2D for Layer 0: Load current layer's KV cache from host to device
            # This is needed because post_attn prefetches NEXT layer, so Layer 0
            # would never get loaded otherwise
            if prefix_meta is not None and active_cache is not None:
                if self.layer_names:
                    first_layer_name = self.layer_names[0]
                    from omni_cache.cache.transfer_engine.synchronize import (
                        synchronize_h2d_prefill,
                    )

                    synchronize_h2d_prefill(active_cache, prefix_meta, first_layer_name, load_next_layer=False)

            return prefix_meta

        def _rewrite_decode_block_table(
            self,
            block_table_tensor,
            seq_lens_device,
            query_start_loc_device,
            slot_mapping,
        ):
            cache = get_active_decode_cache()
            if cache is None:
                return None, None
            num_reqs, max_blocks = block_table_tensor.shape
            idx_reqs = resolve_batch_idx_reqs(cache, num_reqs)
            if idx_reqs is None:
                return None, None

            spec = self.kv_cache_spec
            block_size = getattr(spec, "block_size", 32)
            sliding_window = getattr(spec, "sliding_window", None)
            req_offset = _lane_req_offset(spec, self.vllm_config)
            seq_lens_cpu = seq_lens_device.detach().cpu().tolist()
            query_start_loc_cpu = query_start_loc_device.detach().cpu().tolist()

            # Modify block_table_tensor IN PLACE. In graph mode it IS the
            # persistent tensor the captured kernel reads from; returning
            # a separately allocated `fake` leaves the kernel reading the
            # base-builder-populated address (real vLLM block ids) rather
            # than our lane-remapped values. In eager mode the tensor is
            # rebuilt each step, so zero+write per step is also correct.
            fake = block_table_tensor
            fake.zero_()
            if sliding_window is not None and sliding_window > 0 and block_size > 0:
                win_blocks = max(1, sliding_window // block_size)
            else:
                win_blocks = None

            remapped_slots = self._get_slot_mapping_buffer(slot_mapping)
            remapped_slots.fill_(PAD_SLOT_ID)

            for i in range(num_reqs):
                if idx_reqs[i] < 0:
                    continue  # padded / unresolved row — fake[i] stays zeros
                query_start = query_start_loc_cpu[i]
                query_end = query_start_loc_cpu[i + 1]
                query_len = max(0, query_end - query_start)
                num_computed = max(0, seq_lens_cpu[i] - query_len)
                num_leading = _leading_zero_blocks(num_computed, sliding_window, block_size)
                # Cap valid to this request lane so block IDs do not spill
                # into the next request lane.
                valid = max_blocks - num_leading
                if req_offset > 0:
                    valid = min(valid, req_offset)
                if valid <= 0:
                    continue
                base = req_offset * idx_reqs[i]
                if win_blocks is not None and sliding_window is not None:
                    lb_idx = torch.arange(
                        num_leading,
                        num_leading + valid,
                        dtype=block_table_tensor.dtype,
                        device=block_table_tensor.device,
                    )
                    fake[i, num_leading:num_leading + valid] = (lb_idx % win_blocks) + 1 + base
                else:
                    window = torch.arange(
                        base + 1,
                        base + 1 + valid,
                        dtype=block_table_tensor.dtype,
                        device=block_table_tensor.device,
                    )
                    fake[i, num_leading:num_leading + valid] = window
                start_pos = num_computed
                row_slots = []
                for pos in range(start_pos, start_pos + query_len):
                    logical_block = pos // block_size
                    if logical_block >= max_blocks:
                        row_slots.append(-1)
                        continue
                    block_id = int(fake[i, logical_block].item())
                    if block_id <= 0:
                        row_slots.append(-1)
                        continue
                    row_slots.append(block_id * block_size + (pos % block_size))
                if row_slots:
                    remapped_slots[query_start:query_end] = torch.tensor(row_slots, dtype=remapped_slots.dtype)
            slot_mapping.copy_(
                remapped_slots[: slot_mapping.numel()],
                non_blocking=True,
            )
            return fake, slot_mapping

        def _get_slot_mapping_buffer(self, slot_mapping):
            slots = self._slot_map_pinned
            if slot_mapping.numel() > slots.numel():
                raise RuntimeError(
                    f"slot_mapping has {slot_mapping.numel()} tokens, but persistent buffer only holds {slots.numel()}."
                )
            return slots[: slot_mapping.numel()]

    @register_attention_backend(NPUMLA)
    class NPUMLABackendExt(NPUMLABackend):
        @staticmethod
        def get_builder_cls():
            return NPUMLAMetadataBuilderExt

    logger.info("omni_cache extension enabled, NPUMLABackendExt registered")

else:
    NPUMLABackendExt = NPUMLABackend
    NPUMLAMetadataBuilderExt = NPUMLAMetadataBuilder
    logger.info("omni_cache extension disabled, using base NPUMLABackend")
