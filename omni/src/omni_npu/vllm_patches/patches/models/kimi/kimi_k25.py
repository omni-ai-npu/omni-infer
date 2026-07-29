# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Kimi-K2.5 Model Implementation for vLLM on Ascend NPU.

Kimi-K2.5 extends Kimi-K2 with vision support.
This module defines:
- KimiK25ProcessingInfo/KimiK25MultiModalProcessor: Processing logic
- KimiK25ForConditionalGeneration: Main model class
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

import torch
from torch import nn
from transformers import BatchFeature, DeepseekV3Config
from transformers.configuration_utils import PretrainedConfig
from transformers.processing_utils import ProcessorMixin
from vllm.forward_context import get_forward_context

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    HfImageItem,
    HfVideoItem,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from omni_npu.compilation.acl_graph import ACLGraphWrapper
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

# Vision tower (NPU-adapted, all definitions live here)
from omni_npu.vllm_patches.patches.models.kimi.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    KimiK25VisionConfig,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)

logger = init_logger(__name__)
torch.npu.config.allow_internal_format = True


def _apply_kimi_k25_aclgraph_sinklen_patch() -> None:
    """Disable acl_graph aslkv tail extension for Kimi-K2.5 only."""
    if getattr(ACLGraphWrapper, "_kimi_k25_sinklen_patch_applied", False):
        return

    original_call = ACLGraphWrapper.__call__

    def _patched_call(self, *args, **kwargs):
        forward_context = get_forward_context()
        attn_metadata = getattr(forward_context, "attn_metadata", None)
        patched_decodes: list[tuple[object, bool]] = []

        if attn_metadata is not None:
            for metadata in attn_metadata.values():
                decode = getattr(metadata, "decode", None)
                if decode is None:
                    continue
                if getattr(decode, "sink_len", None) is not None:
                    continue
                had_sink_len_attr = hasattr(decode, "sink_len")
                try:
                    decode.sink_len = 0
                    patched_decodes.append((decode, had_sink_len_attr))
                except Exception:
                    continue

        try:
            return original_call(self, *args, **kwargs)
        finally:
            for decode, had_sink_len_attr in patched_decodes:
                try:
                    if had_sink_len_attr:
                        decode.sink_len = None
                    else:
                        delattr(decode, "sink_len")
                except Exception:
                    continue

    ACLGraphWrapper.__call__ = _patched_call
    ACLGraphWrapper._kimi_k25_sinklen_patch_applied = True
    logger.info("Applied Kimi-K2.5 ACLGraph sink_len compatibility patch.")


# ── VisionChunk types ──────────────────────────────────────────────────────
# NOTE: In vLLM 0.16.0, these are TypedDict defined in vllm.multimodal.inputs.
# In vLLM 0.12.0, they don't exist, so we define them locally as frozen
# dataclasses.  The _to_media_tokens_input() helper converts them to dicts
# for the HF media_processor which expects dict-style access.


@dataclass(frozen=True)
class VisionChunkImage:
    """A single image chunk."""
    image: HfImageItem
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class VisionChunkVideo:
    """A video chunk represented as a list of frames."""
    video_chunk: HfVideoItem
    type: Literal["video_chunk"] = "video_chunk"


VisionChunk = Union[VisionChunkImage, VisionChunkVideo]


# NOTE: In vLLM 0.16.0, VisionChunkProcessorItems is in vllm.multimodal.parse.
# In vLLM 0.12.0, it doesn't exist, so we define it locally.
class VisionChunkProcessorItems(ProcessorBatchItems):
    def __init__(self, data: Sequence[VisionChunk] | None) -> None:
        if data is None:
            data = [None]
        super().__init__(data, "vision_chunk")


# ── KimiK25Config ─────────────────────────────────────────────────────────
# NOTE: In vLLM 0.16.0, this is in vllm.transformers_utils.configs.
# In vLLM 0.12.0, it doesn't exist, so we define it locally.


class KimiK25Config(PretrainedConfig):
    """Kimi-K2.5 model configuration (video-chunk multimodal)."""

    model_type = "kimi_k25"

    def __init__(
        self,
        vision_config: dict | KimiK25VisionConfig | None = None,
        text_config: dict | DeepseekV3Config | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        use_unified_vision_chunk: bool = False,
        video_placeholder: str = "<|kimi_k25_video_placeholder|>",
        **kwargs,
    ):
        # Vision config
        if vision_config is None:
            vision_config = KimiK25VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = KimiK25VisionConfig(**vision_config)
        self.vision_config: KimiK25VisionConfig = vision_config

        # Text config
        if text_config is None:
            text_config = DeepseekV3Config()
        elif isinstance(text_config, dict):
            text_config = DeepseekV3Config(**text_config)
        self.text_config: DeepseekV3Config = text_config

        # Set mm_hidden_size to text hidden size if not explicitly set
        if self.vision_config.mm_hidden_size == self.vision_config.hidden_size:
            self.vision_config.mm_hidden_size = self.text_config.hidden_size

        # Other config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder

        # Propagate quantization config from text model
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        """Get hidden size from text config for compatibility."""
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Get vocab size from text config for compatibility."""
        return self.text_config.vocab_size


# ── VisionChunk data parser ───────────────────────────────────────────────
# NOTE: In vLLM 0.16.0, vision_chunk parsing is built into MultiModalDataParser.
# In vLLM 0.12.0, it only has audio/image/video, so we extend it.


class KimiK25DataParser(MultiModalDataParser):
    """Extends vLLM's data parser to support the vision_chunk modality."""

    def _parse_vision_chunk_data(self, data):
        if data is None:
            return VisionChunkProcessorItems(None)
        if self._is_empty(data):
            return None
        if not isinstance(data, list):
            data_items = [data]
        else:
            data_items = [d for d in data if d is not None]
        return VisionChunkProcessorItems(data_items)

    def _get_subparsers(self):
        parsers = super()._get_subparsers()
        parsers["vision_chunk"] = self._parse_vision_chunk_data
        return parsers


# ── Helpers ────────────────────────────────────────────────────────────────
# NOTE: In vLLM 0.16.0, VisionChunk is a TypedDict (i.e. a dict), so it can
# be passed directly to the HF media_processor.  Our frozen dataclass version
# needs conversion to dict for processors that use dict-style access.


def _unwrap_media(obj):
    """Unwrap MediaWithBytes if present (vLLM 0.14.0+)."""
    try:
        from vllm.multimodal.base import MediaWithBytes
        if isinstance(obj, MediaWithBytes):
            return obj.media
    except ImportError:
        pass
    return obj


def _to_media_tokens_input(chunk: VisionChunk) -> Mapping[str, object]:
    """Convert internal VisionChunk dataclass to dict-like processor format."""
    if isinstance(chunk, VisionChunkImage):
        return {"type": "image", "image": _unwrap_media(chunk.image)}
    if isinstance(chunk, VisionChunkVideo):
        return {"type": "video_chunk", "video_chunk": _unwrap_media(chunk.video_chunk)}

    raise TypeError(f"Unsupported vision chunk type: {type(chunk)}")


# Dummy input dimensions for profiling.
@dataclass
class MaxImageTokenMeta:
    width: int = 3000
    height: int = 3000


class KimiK25MediaPixelInputs(TensorSchema):
    """
    Media input schema for K2.5 model.

    Dimensions:
        - np: Number of patches (flattened from all media items)
        - ps: Patch size
        - nm: Number of media items
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    grid_thws: Annotated[torch.Tensor, TensorShape("nm", 3)]


class MoonshotKimiVAutoProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self, media_processor=None, tokenizer=None, media_token_id: int | None = None
    ):
        super().__init__(tokenizer)
        self.media_processor = media_processor
        self.media_token_id = media_token_id
        assert self.media_token_id is not None

    # We do not support str input for text here
    def __call__(
        self,
        vision_chunks: list[VisionChunk] | None = None,
        *,
        text: list[int] | str,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            vision_chunks: List of VisionChunk items to be processed.
                For image: VisionChunkImage with type='image', image=PIL.Image
                For video_chunk: VisionChunkVideo with type='video_chunk',
                    video_chunk=list[PIL.Image]
            text: The token ids to be fed to a model (required).
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- list of token ids to be fed to a model.
            - **pixel_values** -- Pixel values to be fed to a model.
              Returned when `vision_chunks` is not `None`.
            - **grid_thws** -- list of image 3D grid in LLM.
              Returned when `vision_chunks` is not `None`.
        """
        mm_inputs = {}
        input_ids = self.tokenizer.encode(text) if isinstance(text, str) else text
        if vision_chunks is not None:
            assert isinstance(vision_chunks, list)
            # NOTE: Convert frozen dataclasses to dicts for HF media_processor.
            # In vLLM 0.16.0, VisionChunk is TypedDict (already a dict).
            processor_vision_chunks = [
                _to_media_tokens_input(chunk) for chunk in vision_chunks
            ]
            mm_inputs = self.media_processor.preprocess(processor_vision_chunks)

            num_tokens_per_chunk = [
                self.media_processor.media_tokens_calculator(
                    _to_media_tokens_input(chunk)
                )
                for chunk in vision_chunks
            ]

            new_input_ids = []
            for token in input_ids:
                if token == self.media_token_id:
                    new_input_ids.extend(
                        [self.media_token_id] * num_tokens_per_chunk.pop(0)
                    )
                else:
                    new_input_ids.append(token)
            input_ids = new_input_ids

        # XXX: _apply_hf_processor_text_mm will call tolist() on input_ids
        return BatchFeature(
            data={
                "input_ids": torch.tensor([input_ids]),
                **mm_inputs,
            }
        )


class KimiK25ProcessingInfo(BaseProcessingInfo):
    """Processing information for Kimi-K2.5 model.

    Provides configuration and utilities for processing both
    images and video-chunks.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self.hf_config = self.get_hf_config()
        self.media_token_id = self.hf_config.media_placeholder_token_id
        media_processor = cached_get_image_processor(
            self.ctx.model_config.model, trust_remote_code=True
        )
        self.media_processor = media_processor
        self.hf_processor = MoonshotKimiVAutoProcessor(
            media_processor=self.media_processor,
            tokenizer=self.get_tokenizer(),
            media_token_id=self.media_token_id,
        )
        self.media_tokens_calculator = self.media_processor.media_tokens_calculator

    def get_hf_processor(self):
        return self.hf_processor

    def get_hf_config(self):
        return self.ctx.get_hf_config(KimiK25Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"vision_chunk": None}


class KimiK25DummyInputsBuilder(BaseDummyInputsBuilder[KimiK25ProcessingInfo]):
    """Builds dummy inputs for Kimi-K2.5 model profiling."""

    def __init__(self, info: KimiK25ProcessingInfo) -> None:
        super().__init__(info)
        self.media_token_id = self.info.media_token_id
        self.frame_per_chunk = self.info.media_processor.num_frames_per_chunk

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_media = mm_counts.get("vision_chunk", 0)
        return "<|media_pad|>" * num_media

    def get_dummy_mm_items(self):
        dummy_videos = self._get_dummy_images(
            height=MaxImageTokenMeta.height,
            width=MaxImageTokenMeta.width,
            num_images=self.frame_per_chunk,
        )

        video_chunk_dummy_item = VisionChunkVideo(
            type="video_chunk", video_chunk=dummy_videos
        )
        video_chunk_num_tokens = self.info.media_tokens_calculator(
            _to_media_tokens_input(video_chunk_dummy_item)
        )

        image_dummy_item = VisionChunkImage(
            type="image",
            image=self._get_dummy_images(
                height=MaxImageTokenMeta.height,
                width=MaxImageTokenMeta.width,
                num_images=1,
            )[0],
        )
        image_num_tokens = self.info.media_tokens_calculator(
            _to_media_tokens_input(image_dummy_item)
        )
        # return the larger one
        if video_chunk_num_tokens >= image_num_tokens:
            return [video_chunk_dummy_item]
        else:
            return [image_dummy_item]

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        # TODO: Support mm_options for vision_chunk to allow user configuration
        dummy_items = self.get_dummy_mm_items()
        return {"vision_chunk": dummy_items}


class KimiK25MultiModalProcessor(BaseMultiModalProcessor[KimiK25ProcessingInfo]):
    """Multi-modal processor for Kimi-K2.5.

    Handles both image and video-chunk modalities.
    """

    # NOTE: In vLLM 0.12.0, MultiModalDataParser doesn't know about
    # vision_chunk, so we need KimiK25DataParser. In 0.16.0, the default
    # parser already handles vision_chunk and this method is not needed.
    def _get_data_parser(self) -> MultiModalDataParser:
        return KimiK25DataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Indicates how to slice media input into multiple items.

        pixel_values: [N, 3, patch_size, patch_size], all patches collected from B medias
        grid_thws: [B,3], each item: [N_t, N_h ,N_w], indicates the grid size in time/height/width direction
                    for current item.

        by multiplying [N_t, N_h ,N_w], we get the number of patches for each media item, thus we can slice
        pixel_values by pixel_values[start:start + N_t*N_h*N_w] to get patches of one item.

        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "vision_chunk", grid_sizes
            ),
            grid_thws=MultiModalFieldConfig.batched("vision_chunk"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        media_token_id = hf_config.media_placeholder_token_id

        def get_replacement(item_idx: int):
            media = mm_items.get_items("vision_chunk", (VisionChunkProcessorItems,))
            # NOTE: Convert dataclass to dict for media_tokens_calculator.
            # In vLLM 0.16.0, media[item_idx] is already a dict (TypedDict).
            num_media_token = self.info.media_tokens_calculator(
                _to_media_tokens_input(media[item_idx])
            )
            return [media_token_id] * num_media_token

        return [
            PromptReplacement(
                modality="vision_chunk",
                target=[media_token_id],
                replacement=get_replacement,
            ),
        ]

    def split_video_chunks(self, video):
        return self.info.media_processor.split_video_chunks(video)


@MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=KimiK25ProcessingInfo,
    dummy_inputs=KimiK25DummyInputsBuilder,
)
class KimiK25ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant
):
    """Kimi-K2.5 model for conditional generation (Ascend NPU).

    Supports both image and video-chunk modalities.
    Video-chunks are temporal segments (typically 4 frames) that are
    processed with temporal pooling.
    """

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # For legacy NVFP4 checkpoint compatibility:
            # see https://github.com/vllm-project/vllm/pull/33346#issuecomment-3851475033
            "language_model.layers.": "language_model.model.layers.",
            # mm projector
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Kimi-K2.5 uses video_chunk for all media types
        if modality == "image":
            return "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        elif modality == "video":
            # return a placeholder, to be replaced in the future.
            return "<|kimi_k25_video_placeholder|>"

        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _apply_kimi_k25_aclgraph_sinklen_patch()
        model_config = vllm_config.model_config
        config: KimiK25Config = model_config.hf_config
        self.config = config
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()

        # Check for MoonViT config compatibility
        multimodal_config = model_config.multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        # NOTE: In vLLM 0.16.0, vision tower is wrapped with
        # self._mark_tower_model() context manager. In 0.12.0, this API
        # doesn't exist, so we pass use_data_parallel directly.
        self.vision_tower = MoonViT3dPretrainedModel(
            config.vision_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
            use_data_parallel=self.use_data_parallel,
        ).to(device=self.device, dtype=model_config.dtype)

        self.mm_projector = KimiK25MultiModalProjector(
            config=config.vision_config,
            use_data_parallel=self.use_data_parallel,
            prefix=maybe_prefix(prefix, "mm_projector"),
        ).to(device=self.device, dtype=model_config.dtype)

        self.quant_config = vllm_config.quant_config

        # NOTE: In vLLM 0.16.0, language model init is wrapped with
        # self._mark_language_model() and uses sub_vllm_config. In 0.12.0,
        # this API doesn't exist.
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["DeepseekV3ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.media_placeholder: int = self.config.media_placeholder_token_id

    def get_language_model(self):
        return self.language_model

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        # The batch dimension of pixel_values has been flattened into shape[0]
        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, get {type(grid_thws)}"
        )
        # In some cases (e.g. with merger), grid_thws has an extra middle dimension
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, (
            f"unexpected shape for grid_thws: {grid_thws.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def _process_media_input(
        self, media_input: KimiK25MediaPixelInputs
    ) -> list[torch.Tensor]:
        # NOTE(moyan): This forward will automatically batch the forward pass internally
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        # Validate the multimodal input keyword arguments
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_media_input(media_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.language_model.compute_logits(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
