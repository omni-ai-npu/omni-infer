# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import vllm.entrypoints.chat_utils as chat_mod
from vllm.entrypoints.chat_utils import (
    AsyncMultiModalItemTracker,
    BaseMultiModalItemTracker,
    MultiModalItemTracker,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_models.models.openpangu_vl.openpangu_vl import VisionChunkVideo

logger = logging.getLogger(__name__)

# ── Add "vision_chunk" to MODALITY_PLACEHOLDERS_MAP ──────────────────────
# Dict insertion cannot use register_patch; runs at import time.
if "vision_chunk" not in chat_mod.MODALITY_PLACEHOLDERS_MAP:
    chat_mod.MODALITY_PLACEHOLDERS_MAP["vision_chunk"] = "<##VIDEO##>"


# ── Helper ───────────────────────────────────────────────────────────────

def _resolve_video_chunks(items, modality_order, mm_processor=None):
    chunks = []
    video_idx = 0

    for inner_modality, item in zip(modality_order, items):
        if item is None:
            chunks.append(item)
        elif inner_modality == "video":
            video_data = item[0] if isinstance(item, tuple) and len(item) >= 1 else item

            if mm_processor is not None and hasattr(mm_processor, "split_video_chunks"):
                try:
                    from vllm.utils import random_uuid
                    video_uuid = random_uuid()
                    video_chunks = mm_processor.split_video_chunks(video_data)

                    for i, vc in enumerate(video_chunks):
                        chunks.append(VisionChunkVideo(
                            video_chunk=vc["video_chunk"],
                            start_frame=vc["start_frame"]
                        ))
                    video_idx += 1
                except Exception as e:
                    logger.warning("Failed to split video chunks: %s", e)
                    chunks.append(VisionChunkVideo(video_chunk=video_data))
            else:
                chunks.append(VisionChunkVideo(video_chunk=video_data))
        else:
            chunks.append(item)
    return chunks


# ── Save originals before patching ───────────────────────────────────────

_orig_base_init = BaseMultiModalItemTracker.__init__
_orig_base_add = BaseMultiModalItemTracker.add
_orig_base_all_mm_uuids = BaseMultiModalItemTracker.all_mm_uuids
_orig_sync_all_mm_data = MultiModalItemTracker.all_mm_data
_orig_async_all_mm_data = AsyncMultiModalItemTracker.all_mm_data


# ── Patch 1: BaseMultiModalItemTracker ───────────────────────────────────

@register_patch("OpenPanguV2VLBaseTrackerPatch", BaseMultiModalItemTracker)
class OpenPanguV2VLBaseTrackerPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__', 'add', 'all_mm_uuids', 'use_unified_vision_chunk_modality']

    def __init__(self, model_config):
        _orig_base_init(self, model_config)
        self._modality_order = defaultdict(list)

    def add(self, modality, item, uuid=None):
        original_modality = modality
        use_unified_vision_chunk = (
            self.use_unified_vision_chunk_modality
            and original_modality == "video"
        )

        if use_unified_vision_chunk:
            input_modality = "vision_chunk"
            num_items = len(self._items_by_modality[input_modality]) + 1

            self.mm_processor.validate_num_items(input_modality, num_items)

            self._items_by_modality[input_modality].append(item)
            self._uuids_by_modality[input_modality].append(uuid)
            self._modality_order[input_modality].append(original_modality)

            return self.model_cls.get_placeholder_str(
                original_modality, num_items
            )
        else:
            return _orig_base_add(self, modality, item, uuid)

    def all_mm_uuids(self):
        if "vision_chunk" in self._uuids_by_modality:
            return None
        return _orig_base_all_mm_uuids(self)

    use_unified_vision_chunk_modality = property(
        lambda self: getattr(self._model_config.hf_config,
                            "use_unified_vision_chunk", False)
    )

# ── Patch 2: MultiModalItemTracker (sync) ───────────────────────────────


@register_patch("OpenPanguV2VLSyncTrackerPatch", MultiModalItemTracker)
class OpenPanguV2VLSyncTrackerPatch(VLLMPatch):
    _attr_names_to_apply = ['all_mm_data']

    def all_mm_data(self):
        if not self._items_by_modality:
            return None

        items_by_modality = dict(self._items_by_modality)

        if "vision_chunk" in items_by_modality:
            mm_inputs = {}
            video_items = items_by_modality.pop("vision_chunk")
            order = getattr(self, "_modality_order", {}).get("vision_chunk", [])
            if order:
                mm_inputs["vision_chunk"] = _resolve_video_chunks(
                    video_items, order, mm_processor=self.mm_processor
                )
            else:
                mm_inputs["vision_chunk"] = video_items

            saved = self._items_by_modality.pop("vision_chunk", None)
            rest = _orig_sync_all_mm_data(self)
            if saved is not None:
                self._items_by_modality["vision_chunk"] = saved
            if rest:
                mm_inputs.update(rest)
            return mm_inputs
        else:
            return _orig_sync_all_mm_data(self)


# ── Patch 3: AsyncMultiModalItemTracker (async) ─────────────────────────

async def _patched_async_all_mm_data(self):
    if not self._items_by_modality:
        return None

    items_by_modality_raw = dict(self._items_by_modality)

    if "vision_chunk" in items_by_modality_raw:
        mm_inputs = {}

        coros = []
        for item in items_by_modality_raw["vision_chunk"]:
            if item is not None:
                coros.append(item)
            else:
                coros.append(asyncio.sleep(0))
        resolved_items = await asyncio.gather(*coros)

        order = getattr(self, "_modality_order", {}).get("vision_chunk", [])
        if order:
            mm_inputs["vision_chunk"] = _resolve_video_chunks(
                resolved_items, order, mm_processor=self.mm_processor)
        else:
            mm_inputs["vision_chunk"] = list(resolved_items)

        saved = self._items_by_modality.pop("vision_chunk", None)
        rest = await _orig_async_all_mm_data(self)
        if saved is not None:
            self._items_by_modality["vision_chunk"] = saved
        if rest:
            mm_inputs.update(rest)
        return mm_inputs
    else:
        result = await _orig_async_all_mm_data(self)
        return result

AsyncMultiModalItemTracker.all_mm_data = _patched_async_all_mm_data
