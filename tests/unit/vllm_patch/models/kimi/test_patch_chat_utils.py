# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for kimi patch_chat_utils: unified vision_chunk modality mapping."""

import asyncio
import importlib
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _install_stubbed_chat_utils_env(monkeypatch):
    """Install lightweight vLLM/chat_utils stubs so tests are deterministic."""
    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []
    vllm_entrypoints_pkg = types.ModuleType("vllm.entrypoints")
    vllm_entrypoints_pkg.__path__ = []
    chat_mod = types.ModuleType("vllm.entrypoints.chat_utils")
    chat_mod.MODALITY_PLACEHOLDERS_MAP = {
        "image": "<##IMAGE##>",
        "video": "<##VIDEO##>",
        "audio": "<##AUDIO##>",
    }

    class BaseMultiModalItemTracker:
        def __init__(self, model_config):
            self._model_config = model_config
            self._items_by_modality = defaultdict(list)
            self._uuids_by_modality = defaultdict(list)
            self.mm_processor = types.SimpleNamespace(validate_num_items=lambda *_: None)
            self.model_cls = types.SimpleNamespace(
                get_placeholder_str=lambda modality, num_items: f"<{modality}:{num_items}>"
            )

        def add(self, modality, item, uuid=None):
            items = self._items_by_modality.setdefault(modality, [])
            uuids = self._uuids_by_modality.setdefault(modality, [])
            num_items = len(items) + 1
            self.mm_processor.validate_num_items(modality, num_items)
            items.append(item)
            uuids.append(uuid)
            return self.model_cls.get_placeholder_str(modality, num_items)

        def all_mm_uuids(self):
            if not self._uuids_by_modality:
                return None
            return dict(self._uuids_by_modality)

    class MultiModalItemTracker(BaseMultiModalItemTracker):
        def all_mm_data(self):
            if not self._items_by_modality:
                return None
            return {
                modality: list(items)
                for modality, items in self._items_by_modality.items()
            }

    class AsyncMultiModalItemTracker(BaseMultiModalItemTracker):
        async def all_mm_data(self):
            if not self._items_by_modality:
                return None
            resolved = {}
            for modality, items in self._items_by_modality.items():
                out_items = []
                for item in items:
                    if asyncio.iscoroutine(item) or hasattr(item, "__await__"):
                        out_items.append(await item)
                    else:
                        out_items.append(item)
                resolved[modality] = out_items
            return resolved

    chat_mod.BaseMultiModalItemTracker = BaseMultiModalItemTracker
    chat_mod.MultiModalItemTracker = MultiModalItemTracker
    chat_mod.AsyncMultiModalItemTracker = AsyncMultiModalItemTracker

    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", vllm_entrypoints_pkg)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.chat_utils", chat_mod)

    # Stub VisionChunk types
    kimi_mod = types.ModuleType("omni_npu.vllm_patches.patches.models.kimi.kimi_k25")

    @dataclass(frozen=True)
    class VisionChunkImage:
        image: object
        type: str = "image"

    @dataclass(frozen=True)
    class VisionChunkVideo:
        video_chunk: object
        type: str = "video_chunk"

    kimi_mod.VisionChunkImage = VisionChunkImage
    kimi_mod.VisionChunkVideo = VisionChunkVideo
    monkeypatch.setitem(sys.modules, "omni_npu.vllm_patches.patches.models.kimi.kimi_k25", kimi_mod)

    # Stub omni_npu.vllm_patches.core with a register_patch that applies immediately
    class FakeVLLMPatch:
        _attr_names_to_apply = []

    def fake_register_patch(name, target):
        def decorator(cls):
            cls._target = target
            for attr_name in cls._attr_names_to_apply:
                if attr_name in cls.__dict__:
                    setattr(target, attr_name, cls.__dict__[attr_name])
            return cls
        return decorator

    core_mod = types.ModuleType("omni_npu.vllm_patches.core")
    core_mod.VLLMPatch = FakeVLLMPatch
    core_mod.register_patch = fake_register_patch
    monkeypatch.setitem(sys.modules, "omni_npu.vllm_patches.core", core_mod)

    patch_module = "omni_npu.vllm_patches.patches.models.kimi.patch_chat_utils"
    monkeypatch.delitem(sys.modules, patch_module, raising=False)
    importlib.import_module(patch_module)


def _make_fake_hf_config(use_unified_vision_chunk=True):
    return types.SimpleNamespace(use_unified_vision_chunk=use_unified_vision_chunk)


def _make_fake_model_config(use_unified_vision_chunk=True):
    return types.SimpleNamespace(
        hf_config=_make_fake_hf_config(use_unified_vision_chunk),
        multimodal_config=None,
        allowed_local_media_path=None,
        allowed_media_domains=None,
    )


# ── Tests: patch is applied ─────────────────────────────────────────────


class TestPatchApplied:

    def test_modality_placeholders_map_has_vision_chunk(self):
        """After patch, MODALITY_PLACEHOLDERS_MAP should contain 'vision_chunk'."""
        import omni_npu.vllm_patches.patches.models.kimi.patch_chat_utils  # noqa: F401
        from vllm.entrypoints.chat_utils import MODALITY_PLACEHOLDERS_MAP

        assert "vision_chunk" in MODALITY_PLACEHOLDERS_MAP

    def test_tracker_has_use_unified_vision_chunk_property(self):
        """BaseMultiModalItemTracker should have use_unified_vision_chunk_modality."""
        import omni_npu.vllm_patches.patches.models.kimi.patch_chat_utils  # noqa: F401
        from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker

        assert hasattr(BaseMultiModalItemTracker, "use_unified_vision_chunk_modality")


# ── Tests: add() modality mapping ───────────────────────────────────────


class TestAddModalityMapping:

    def _make_tracker(self, use_unified=True):
        """Create a tracker with mocked dependencies."""
        import omni_npu.vllm_patches.patches.models.kimi.patch_chat_utils  # noqa: F401
        from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker

        model_config = _make_fake_model_config(use_unified)
        tokenizer = MagicMock()

        # We need a concrete subclass to instantiate
        class ConcreteTracker(BaseMultiModalItemTracker):
            def create_parser(self):
                return MagicMock()

        tracker = ConcreteTracker(model_config)

        # Mock mm_processor.validate_num_items to do nothing
        mock_processor = MagicMock()
        mock_processor.validate_num_items = MagicMock()
        tracker.__dict__["mm_processor"] = mock_processor

        # Mock model_cls.get_placeholder_str
        mock_model_cls = MagicMock()
        mock_model_cls.get_placeholder_str = MagicMock(return_value="<|media_pad|>")
        tracker.__dict__["model_cls"] = mock_model_cls

        return tracker

    def test_image_mapped_to_vision_chunk_when_enabled(self):
        tracker = self._make_tracker(use_unified=True)
        sentinel = object()

        tracker.add("image", sentinel, uuid="img-1")

        assert "vision_chunk" in tracker._items_by_modality
        assert sentinel in tracker._items_by_modality["vision_chunk"]
        assert "image" not in tracker._items_by_modality
        tracker.mm_processor.validate_num_items.assert_called_with(
            "vision_chunk", 1)

    def test_video_mapped_to_vision_chunk_when_enabled(self):
        tracker = self._make_tracker(use_unified=True)
        sentinel = object()

        tracker.add("video", sentinel, uuid="vid-1")

        assert "vision_chunk" in tracker._items_by_modality
        assert sentinel in tracker._items_by_modality["vision_chunk"]
        assert "video" not in tracker._items_by_modality

    def test_image_not_mapped_when_disabled(self):
        tracker = self._make_tracker(use_unified=False)
        sentinel = object()

        tracker.add("image", sentinel, uuid="img-1")

        assert "image" in tracker._items_by_modality
        assert sentinel in tracker._items_by_modality["image"]
        assert "vision_chunk" not in tracker._items_by_modality

    def test_modality_order_tracked(self):
        tracker = self._make_tracker(use_unified=True)

        tracker.add("image", object())
        tracker.add("video", object())
        tracker.add("image", object())

        assert tracker._modality_order["vision_chunk"] == [
            "image", "video", "image"
        ]

    def test_uuid_stored_under_vision_chunk(self):
        tracker = self._make_tracker(use_unified=True)

        tracker.add("image", object(), uuid="u1")
        tracker.add("video", object(), uuid="u2")

        assert tracker._uuids_by_modality["vision_chunk"] == ["u1", "u2"]

    def test_placeholder_uses_original_modality(self):
        tracker = self._make_tracker(use_unified=True)

        tracker.add("image", object())

        tracker.model_cls.get_placeholder_str.assert_called_with("image", 1)

    def test_audio_not_affected(self):
        """Audio modality should not be mapped even when vision_chunk enabled."""
        tracker = self._make_tracker(use_unified=True)
        sentinel = object()

        tracker.add("audio", sentinel)

        assert "audio" in tracker._items_by_modality
        assert "vision_chunk" not in tracker._items_by_modality


# ── Tests: all_mm_uuids ─────────────────────────────────────────────────


class TestAllMmUuids:

    def _make_tracker(self):
        import omni_npu.vllm_patches.patches.models.kimi.patch_chat_utils  # noqa: F401
        from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker

        model_config = _make_fake_model_config(True)

        class ConcreteTracker(BaseMultiModalItemTracker):
            def create_parser(self):
                return MagicMock()

        tracker = ConcreteTracker(model_config)
        mock_processor = MagicMock()
        mock_processor.validate_num_items = MagicMock()
        tracker.__dict__["mm_processor"] = mock_processor
        mock_model_cls = MagicMock()
        mock_model_cls.get_placeholder_str = MagicMock(return_value="<|p|>")
        tracker.__dict__["model_cls"] = mock_model_cls
        return tracker

    def test_vision_chunk_uuids_returned(self):
        tracker = self._make_tracker()
        tracker.add("image", object(), uuid="u1")
        tracker.add("image", object(), uuid="u2")

        uuids = tracker.all_mm_uuids()

        assert uuids is not None
        assert "vision_chunk" in uuids
        assert uuids["vision_chunk"] == ["u1", "u2"]

    def test_empty_returns_none(self):
        tracker = self._make_tracker()
        assert tracker.all_mm_uuids() is None


# ── Tests: sync all_mm_data ─────────────────────────────────────────────


class TestSyncAllMmData:

    def _make_sync_tracker(self):
        import omni_npu.vllm_patches.patches.models.kimi.patch_chat_utils  # noqa: F401
        from vllm.entrypoints.chat_utils import MultiModalItemTracker

        model_config = _make_fake_model_config(True)
        tracker = MultiModalItemTracker(model_config)

        mock_processor = MagicMock()
        mock_processor.validate_num_items = MagicMock()
        tracker.__dict__["mm_processor"] = mock_processor
        mock_model_cls = MagicMock()
        mock_model_cls.get_placeholder_str = MagicMock(return_value="<|p|>")
        tracker.__dict__["model_cls"] = mock_model_cls
        return tracker

    def test_image_wrapped_in_vision_chunk_image(self):
        from omni_npu.vllm_patches.patches.models.kimi.kimi_k25 import VisionChunkImage

        tracker = self._make_sync_tracker()
        sentinel = object()
        tracker.add("image", sentinel)

        result = tracker.all_mm_data()

        assert "vision_chunk" in result
        chunks = result["vision_chunk"]
        assert len(chunks) == 1
        assert isinstance(chunks[0], VisionChunkImage)
        assert chunks[0].image is sentinel

    def test_video_wrapped_in_vision_chunk_video(self):
        from omni_npu.vllm_patches.patches.models.kimi.kimi_k25 import VisionChunkVideo

        tracker = self._make_sync_tracker()
        sentinel = object()
        tracker.add("video", sentinel)

        result = tracker.all_mm_data()

        assert "vision_chunk" in result
        chunks = result["vision_chunk"]
        assert len(chunks) == 1
        assert isinstance(chunks[0], VisionChunkVideo)
        assert chunks[0].video_chunk is sentinel

    def test_mixed_image_video_order_preserved(self):
        from omni_npu.vllm_patches.patches.models.kimi.kimi_k25 import VisionChunkImage, VisionChunkVideo

        tracker = self._make_sync_tracker()
        img = object()
        vid = object()
        tracker.add("image", img)
        tracker.add("video", vid)
        tracker.add("image", img)

        result = tracker.all_mm_data()
        chunks = result["vision_chunk"]

        assert len(chunks) == 3
        assert isinstance(chunks[0], VisionChunkImage)
        assert isinstance(chunks[1], VisionChunkVideo)
        assert isinstance(chunks[2], VisionChunkImage)

    def test_empty_returns_none(self):
        tracker = self._make_sync_tracker()
        assert tracker.all_mm_data() is None


# ── Tests: async all_mm_data ────────────────────────────────────────────


class TestAsyncAllMmData:

    def _make_async_tracker(self):
        import omni_npu.vllm_patches.patches.models.kimi.patch_chat_utils  # noqa: F401
        from vllm.entrypoints.chat_utils import AsyncMultiModalItemTracker

        model_config = _make_fake_model_config(True)
        tracker = AsyncMultiModalItemTracker(model_config)

        mock_processor = MagicMock()
        mock_processor.validate_num_items = MagicMock()
        tracker.__dict__["mm_processor"] = mock_processor
        mock_model_cls = MagicMock()
        mock_model_cls.get_placeholder_str = MagicMock(return_value="<|p|>")
        tracker.__dict__["model_cls"] = mock_model_cls
        return tracker

    @pytest.mark.anyio
    async def test_async_image_wrapped(self):
        from omni_npu.vllm_patches.patches.models.kimi.kimi_k25 import VisionChunkImage

        tracker = self._make_async_tracker()
        sentinel = object()

        async def _fetch():
            return sentinel

        tracker.add("image", _fetch())

        result = await tracker.all_mm_data()

        assert "vision_chunk" in result
        chunks = result["vision_chunk"]
        assert len(chunks) == 1
        assert isinstance(chunks[0], VisionChunkImage)
        assert chunks[0].image is sentinel

    @pytest.mark.anyio
    async def test_async_empty_returns_none(self):
        tracker = self._make_async_tracker()
        assert await tracker.all_mm_data() is None
