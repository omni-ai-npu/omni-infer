# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Patch chat_utils to support unified vision_chunk modality mapping.

vLLM 0.16.0 added ``use_unified_vision_chunk`` support in chat_utils.py so
that models declaring only ``{"vision_chunk": None}`` in their supported
modality limits can still receive ``image`` / ``video`` inputs via the
OpenAI-compatible API.  vLLM 0.12.0 lacks this mapping, causing:

    ValueError: At most 0 image(s) may be provided in one prompt.

This patch back-ports the mapping logic:
  1. ``BaseMultiModalItemTracker.add()`` – maps "image"/"video" → "vision_chunk"
     when ``hf_config.use_unified_vision_chunk`` is ``True``.
  2. ``MultiModalItemTracker.all_mm_data()`` / ``AsyncMultiModalItemTracker.all_mm_data()``
     – wraps resolved items in ``VisionChunkImage`` / ``VisionChunkVideo``.
  3. ``MODALITY_PLACEHOLDERS_MAP`` – adds ``"vision_chunk"`` entry.
"""

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

from omni_npu.vllm_patches.patches.models.kimi.kimi_k25 import VisionChunkImage, VisionChunkVideo
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = logging.getLogger(__name__)

# ── Add "vision_chunk" to MODALITY_PLACEHOLDERS_MAP ──────────────────────
# Dict insertion cannot use register_patch; runs at import time.
if "vision_chunk" not in chat_mod.MODALITY_PLACEHOLDERS_MAP:
    chat_mod.MODALITY_PLACEHOLDERS_MAP["vision_chunk"] = "<##IMAGE##>"


# ── Helper ───────────────────────────────────────────────────────────────

def _resolve_vision_chunks(items, modality_order):
    """Wrap resolved items as VisionChunkImage / VisionChunkVideo."""
    chunks = []
    for inner_modality, item in zip(modality_order, items):
        if item is None:
            chunks.append(item)
        elif inner_modality == "image":
            chunks.append(VisionChunkImage(image=item))
        elif inner_modality == "video":
            chunks.append(VisionChunkVideo(video_chunk=item))
        else:
            chunks.append(item)
    return chunks


# ── Save originals before patching ───────────────────────────────────────

_orig_base_init = BaseMultiModalItemTracker.__init__
_orig_base_add = BaseMultiModalItemTracker.add
_orig_sync_all_mm_data = MultiModalItemTracker.all_mm_data
_orig_async_all_mm_data = AsyncMultiModalItemTracker.all_mm_data


# ── Patch 1: BaseMultiModalItemTracker ───────────────────────────────────

@register_patch("KimiBaseTrackerPatch", BaseMultiModalItemTracker)
class KimiBaseTrackerPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__', 'add', 'use_unified_vision_chunk_modality']

    def __init__(self, model_config):
        _orig_base_init(self, model_config)
        self._modality_order = defaultdict(list)

    def add(self, modality, item, uuid=None):
        """Map image/video → vision_chunk when use_unified_vision_chunk."""
        original_modality = modality
        use_vision_chunk = (
            self.use_unified_vision_chunk_modality
            and original_modality in ("image", "video")
        )

        if use_vision_chunk:
            input_modality = "vision_chunk"
            num_items = len(self._items_by_modality[input_modality]) + 1

            self.mm_processor.validate_num_items(input_modality, num_items)

            self._items_by_modality[input_modality].append(item)
            self._uuids_by_modality[input_modality].append(uuid)
            self._modality_order["vision_chunk"].append(original_modality)

            return self.model_cls.get_placeholder_str(
                original_modality, num_items)
        else:
            return _orig_base_add(self, modality, item, uuid)

    use_unified_vision_chunk_modality = property(
        lambda self: getattr(self._model_config.hf_config,
                             "use_unified_vision_chunk", False)
    )


# ── Patch 2: MultiModalItemTracker (sync) ───────────────────────────────

@register_patch("KimiSyncTrackerPatch", MultiModalItemTracker)
class KimiSyncTrackerPatch(VLLMPatch):
    _attr_names_to_apply = ['all_mm_data']

    def all_mm_data(self):
        if not self._items_by_modality:
            return None

        items_by_modality = dict(self._items_by_modality)

        if "vision_chunk" in items_by_modality:
            mm_inputs = {}
            vision_items = items_by_modality.pop("vision_chunk")
            order = getattr(self, "_modality_order", {}).get("vision_chunk", [])
            if order:
                mm_inputs["vision_chunk"] = _resolve_vision_chunks(
                    vision_items, order)
            else:
                mm_inputs["vision_chunk"] = vision_items

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
# NOTE: 不使用 register_patch，因为 KimiBaseTrackerPatch 在
# BaseMultiModalItemTracker 上设置的 _omni_npu_applied_patches dict 会被
# 子类 AsyncMultiModalItemTracker 通过 MRO 继承，导致 patch manager 误判
# all_mm_data "already patched by KimiSyncTrackerPatch"。直接 setattr 绕过。

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
            mm_inputs["vision_chunk"] = _resolve_vision_chunks(
                resolved_items, order)
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
