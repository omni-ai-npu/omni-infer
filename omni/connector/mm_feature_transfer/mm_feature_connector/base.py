# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
import asyncio
import inspect
import threading
import torch
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any, List, Protocol
from collections.abc import Sequence

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.processing.processor import (
    PromptUpdateDetails,
    ResolvedPromptUpdate,
)
from vllm.logger import init_logger

from ..config import ConnectorConfig


logger = init_logger(__name__)


def extract_embed_param(is_embed):
    """Return the embed param captured by a function created via select_token_id."""
    if is_embed is None:
        return None
    # Assuming the closure contains exactly embed_token_id/embed_text
    return is_embed.__closure__[0].cell_contents


def serialize_update_details(update_datails: PromptUpdateDetails) -> dict:
    """Return a pickle-friendly tuple (full, mode, param)."""
    full = update_datails.full
    is_embed = update_datails.is_embed
    
    if is_embed is None:
        return (full, None, None)
    
    try:
        param = extract_embed_param(is_embed)
        if isinstance(param, int):
            return (full, "token_id", param)
        elif isinstance(param, str):
            return (full, "text", param)
        else:
            return (full, None, None)
    except ValueError:
        return (full, None, None)


def deserialize_update_details(data):
    full, mode, param = data
    if mode is None:
        return PromptUpdateDetails.from_seq(full)
    elif mode == "token_id":
        return PromptUpdateDetails.select_token_id(full, param)
    elif mode == "text":
        return PromptUpdateDetails.select_text(full, param)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def serialize_prompt_update(prompt_update: ResolvedPromptUpdate) -> dict:
    """Convert to a picklable dict, skipping unpicklable attributes."""
    # But the 'content' field might contain a callable; we need to replace it.
    # Instead, create a picklable version manually.
    return {
        'modality': prompt_update.modality,
        'item_idx': prompt_update.item_idx,
        'mode': prompt_update.mode,
        'target': prompt_update.target,
        # For content, store its picklable representation:
        'content_info': serialize_update_details(prompt_update.content)
    }


def deserialize_prompt_update(data: dict) -> ResolvedPromptUpdate:
    """Recreate ResolvedPromptUpdate from dict."""
    return ResolvedPromptUpdate(
        modality=data['modality'],
        item_idx=data['item_idx'],
        mode=data['mode'],
        target=data['target'],
        content=deserialize_update_details(data['content_info'])
    )


class BaseMMFeatureConnector(ABC):
    """
    Abstract base class for MM feature connector.

    This connector provides an external cache layer for multimodal features.

    The connector is used by the multimodal processor to:
    - Check if features exist in external storage (across EPD instances)
    - Load features from external storage (when processor cache misses)
    - Save features to external storage (for other instances to use)
    """
    _profile_run_finished: bool = False

    def __init__(self, config: ConnectorConfig) -> None:
        self._config = config

    @property
    def is_producer(self) -> bool:
        """Check if this instance should save features."""
        # (typically encoder/producer instances save, decoder/consumer instances don't)
        return self._config.is_producer

    def is_profile_run(self) -> bool:
        if BaseMMFeatureConnector._profile_run_finished:
            return False
        for frame_info in inspect.stack():
            if 'profile_run' == frame_info.function:
                return True
        BaseMMFeatureConnector._profile_run_finished = True
        return False

    def prepare(
        self, 
        mm_item: "MultiModalKwargsItem", 
        prompt_updates: Sequence[ResolvedPromptUpdate]
    ):
        tensors = {}
        for key, field_elem in mm_item.items():
            if key in self._config.exclude_fields:
                continue
            tensors[key] = field_elem.data

        first_field = next(iter(mm_item.values()))
        modality = first_field.modality

        metadata = {
            "modality": modality,
            "tensor_keys": list(tensors.keys()),
        }

        serialized_updates = [serialize_prompt_update(pu) for pu in prompt_updates]

        return metadata, tensors, serialized_updates

    def reconstitute(
        self, 
        metadata: Dict[str, Any],
        tensors: Dict[str, torch.Tensor],
        serialized_updates: List[Dict[str, Any]], 
    ):
        from vllm.multimodal.inputs import (
            MultiModalBatchedField, MultiModalFieldElem, MultiModalKwargsItem
        )
        field_elems = {}
        for key, tensor in tensors.items():
            field_elems[key] = MultiModalFieldElem(
                data=tensor,
                field=MultiModalBatchedField(),
            )
        mm_item = MultiModalKwargsItem(field_elems)

        prompt_updates = [deserialize_prompt_update(pu) 
                          for pu in serialized_updates]

        return mm_item, prompt_updates
    
    def save_item_with_updates(
        self, 
        mm_hash: str, 
        mm_item: "MultiModalKwargsItem", 
        prompt_updates: Sequence[ResolvedPromptUpdate]
    ) -> None:
        if not self.is_producer:
            return

        raw_data = self.prepare(mm_item, prompt_updates)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.save(mm_hash, *raw_data))
        except RuntimeError:
            # Fallback if no loop is running at all
            threading.Thread(target=lambda: asyncio.run(self.save(mm_hash, *raw_data)), daemon=True).start()

    def load_item_with_updates(
        self, mm_hash: str
    ) -> tuple["MultiModalKwargsItem", Sequence[ResolvedPromptUpdate]] | None:
        """Load both feature and prompt updates. Returns (mm_item, prompt_updates)."""
        raw_data = self.load(mm_hash)
        if raw_data is None:
            return None

        mm_item, prompt_updates = self.reconstitute(*raw_data)
        logger.debug(
            "Loaded MM feature %s mm item %s, prompt updates %s from connector", 
            mm_hash, 
            mm_item, 
            prompt_updates
        )
        return mm_item, prompt_updates

    @abstractmethod
    def has_item(self, mm_hash: str) -> bool:
        """
        Check if a feature exists in external storage.

        Args:
            mm_hash: Hash of the multimodal feature

        Returns:
            True if feature exists in external storage
        """
        pass

    @abstractmethod
    def load(self, mm_hash: str):
        pass

    @abstractmethod
    async def save(
        self, 
        mm_hash: str, 
        metadata: Dict[str, Any], 
        tensors: Dict[str, torch.Tensor], 
        serialized_updates: List[Dict[str, Any]],
    ) -> None:
        pass
