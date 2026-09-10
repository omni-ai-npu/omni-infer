# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Build decoder calibration inputs from real OpenPangu Omni media.

This module deliberately stops at the language-model input embedding boundary:
the vision/audio towers and their projectors run in BF16, their outputs replace
the corresponding placeholder token embeddings, and the existing JointFix
decoder runner remains unchanged.  Modality-aware statistics are a later step.

The JSONL manifest accepts one calibration request per line, for example::

    {"image": "images/cat.jpg", "text": "请描述这张图片。"}
    {"audio": "audio/question.wav", "text": "请转写并回答音频问题。"}
    {"image": "chart.png", "audio": "query.wav", "text": "回答问题。"}

Relative media paths are resolved against the manifest directory.  ``prompt``
is an alias of ``text``.  A plain path line is also accepted and its modality is
inferred from the extension.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from jointfix.core.devices import clear_device_cache
from jointfix.core.modality import CalibrationInputs
from jointfix.multimodal.data import _read_audio, extract_audio_chunks
from jointfix.multimodal.models import (
    AudioLayer,
    AudioState,
    VisionBlock,
    VisionState,
    prepare_audio_state,
    prepare_vision_state,
)
from jointfix.multimodal.runner import CheckpointReader


_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
_AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}


@dataclass(frozen=True)
class OmniCalibrationRecord:
    text: str
    image: str | None = None
    audio: str | None = None


def _resolve_media(value: object, manifest: Path, line_no: int, key: str) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f"{manifest}:{line_no}: only one {key} per calibration record is "
                "supported in the first multimodal calibration version"
            )
        value = value[0]
    if not isinstance(value, str):
        raise TypeError(f"{manifest}:{line_no}: {key} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{manifest}:{line_no}: {key} file not found: {path}")
    return str(path)


def load_omni_manifest(path: str, n_samples: int) -> List[OmniCalibrationRecord]:
    """Read image/audio/text calibration requests from a JSONL manifest."""
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Omni calibration manifest not found: {manifest}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    records: List[OmniCalibrationRecord] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("{"):
                item = json.loads(raw)
                text = item.get("text", item.get("prompt", ""))
                image = _resolve_media(item.get("image"), manifest, line_no, "image")
                audio = _resolve_media(item.get("audio"), manifest, line_no, "audio")
            else:
                media = _resolve_media(raw, manifest, line_no, "media")
                suffix = Path(media).suffix.lower()
                image = media if suffix in _IMAGE_SUFFIXES else None
                audio = media if suffix in _AUDIO_SUFFIXES else None
                text = ""
                if image is None and audio is None:
                    raise ValueError(
                        f"{manifest}:{line_no}: cannot infer image/audio modality "
                        f"from extension {suffix!r}"
                    )
            if not isinstance(text, str):
                raise TypeError(f"{manifest}:{line_no}: text/prompt must be a string")
            if image is None and audio is None and not text.strip():
                raise ValueError(
                    f"{manifest}:{line_no}: calibration record has neither media nor text"
                )
            records.append(OmniCalibrationRecord(text=text, image=image, audio=audio))
            if len(records) >= n_samples:
                break
    if not records:
        raise ValueError(f"no multimodal calibration records found in {manifest}")
    return records


def _segment_batches(segments: Iterable[int], token_budget: int):
    start = batch_start = batch_tokens = 0
    batch_segments: List[int] = []
    for length in segments:
        length = int(length)
        if batch_segments and batch_tokens + length > token_budget:
            yield batch_start, start, batch_segments
            batch_start, batch_segments, batch_tokens = start, [], 0
        batch_segments.append(length)
        batch_tokens += length
        start += length
    if batch_segments:
        yield batch_start, start, batch_segments


def _forward_vision_layer(layer, state: VisionState, device, token_budget: int):
    outputs = []
    for start, end, segments in _segment_batches(state.segments, token_budget):
        outputs.append(layer(
            state.hidden[start:end].to(device),
            cos=state.cos[start:end],
            sin=state.sin[start:end],
            segments=segments,
        ).cpu())
    return torch.cat(outputs, dim=0)


def _forward_audio_layer(layer, state: AudioState, device, token_budget: int):
    outputs = []
    for start, end, segments in _segment_batches(state.segments, token_budget):
        outputs.append(layer(
            state.hidden[start:end].to(device), segments=segments,
        ).cpu())
    return torch.cat(outputs, dim=0)


def _layer_norm(x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    """LayerNorm with production dtype semantics and a CPU-safe fallback."""
    if x.device.type == "cpu":
        return F.layer_norm(
            x.float(), (x.shape[-1],), weight.float(),
            None if bias is None else bias.float(), eps,
        ).to(x.dtype)
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _linear(x: torch.Tensor, tensors: Dict[str, torch.Tensor], base: str,
            device) -> torch.Tensor:
    weight = tensors[f"{base}.weight"].to(device)
    bias = tensors.get(f"{base}.bias")
    return F.linear(x, weight, None if bias is None else bias.to(device))


def _encode_images(
    records: List[OmniCalibrationRecord],
    reader: CheckpointReader,
    model_config: dict,
    device,
    *,
    min_pixels: int,
    max_pixels: int,
    image_use_fast: bool,
    forward_token_budget: int,
) -> Dict[int, torch.Tensor]:
    image_records = [(idx, record.image) for idx, record in enumerate(records) if record.image]
    if not image_records:
        return {}

    from PIL import Image
    from omni_models.models.openpangu_vl.huggingface.imageprocessor_openpangu_vl import (
        OpenPanguVLImageProcessorFast,
    )

    processor = OpenPanguVLImageProcessorFast.from_pretrained(str(reader.root))
    processor.image_use_fast = bool(image_use_fast)
    patches, grids = [], []
    for count, (_, path) in enumerate(image_records, 1):
        with Image.open(path) as image:
            encoded = processor(
                images=[image.convert("RGB")],
                size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
                return_tensors="pt",
                device="cpu",
            )
        patches.append(encoded["pixel_values"].to(torch.bfloat16))
        grids.append(encoded["image_grid_thw"].to(torch.long))
        print(
            f"  [omni-llm/image] {count}/{len(image_records)} {path} "
            f"grid={grids[-1].tolist()}", flush=True,
        )

    vision_config = model_config["vision_config"]
    grid_thw = torch.cat(grids, dim=0)
    frontend = reader.load_exact_prefixes(("visual.patch_embed.", "visual.layernorm_pre."))
    state = prepare_vision_state(
        torch.cat(patches, dim=0), grid_thw, frontend, vision_config, device,
    )
    del frontend, patches
    for layer_idx in range(int(vision_config["depth"])):
        prefix = f"visual.blocks.{layer_idx}"
        weights = reader.load_prefix(prefix + ".")
        layer = VisionBlock(weights, prefix, vision_config, device).eval()
        with torch.no_grad():
            hidden = _forward_vision_layer(layer, state, device, forward_token_budget)
        state = VisionState(hidden, state.cos, state.sin, state.segments)
        del layer, weights, hidden
        clear_device_cache(device)

    tail = reader.load_exact_prefixes(("visual.merger.", "visual.vision_projection."))
    merge = int(vision_config["spatial_merge_size"])
    patch_lengths = [int(t * h * w) for t, h, w in grid_thw.tolist()]
    result: Dict[int, torch.Tensor] = {}
    offset = 0
    with torch.no_grad():
        for (record_idx, _), patch_len in zip(image_records, patch_lengths):
            x = state.hidden[offset:offset + patch_len].to(device)
            offset += patch_len
            x = _layer_norm(
                x,
                tail["visual.merger.ln_q.weight"].to(device),
                tail["visual.merger.ln_q.bias"].to(device)
                if "visual.merger.ln_q.bias" in tail else None,
                float(vision_config.get("rms_norm_eps", 1e-6)),
            )
            x = x.reshape(-1, x.shape[-1] * merge * merge)
            x = F.gelu(_linear(x, tail, "visual.merger.mlp.0", device))
            x = _linear(x, tail, "visual.merger.mlp.2", device)
            if bool(vision_config.get("use_gatedmerger", False)):
                x, gate = torch.chunk(x, 2, dim=-1)
                x = x * F.silu(gate)
            # ProjectionSingle applies SiLU before its only linear.  Keep the
            # base hidden size here; decoder layer 0 expands it into MHC streams.
            x = _linear(F.silu(x), tail, "visual.vision_projection.fc1", device)
            result[record_idx] = x.to(torch.bfloat16).cpu()
    if offset != state.hidden.shape[0]:
        raise RuntimeError(
            f"vision split consumed {offset} patches, tower produced {state.hidden.shape[0]}"
        )
    del state, tail
    clear_device_cache(device)
    return result


def _load_audio_features(records: List[OmniCalibrationRecord], model_dir: str):
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    features, lengths, owners = [], [], []
    for record_idx, record in enumerate(records):
        if not record.audio:
            continue
        waveform = _read_audio(record.audio, extractor.sampling_rate)
        valid = 0
        for feature, length in extract_audio_chunks(extractor, waveform):
            features.append(feature)
            lengths.append(length)
            owners.append(record_idx)
            valid += 1
        print(
            f"  [omni-llm/audio] {record.audio} chunks={valid} "
            f"seconds={len(waveform) / extractor.sampling_rate:.2f}", flush=True,
        )
    if not features:
        return None, None, []
    return torch.cat(features, dim=1), torch.tensor(lengths, dtype=torch.long), owners


def _encode_audio(
    records: List[OmniCalibrationRecord],
    reader: CheckpointReader,
    model_config: dict,
    device,
    *,
    forward_token_budget: int,
) -> Dict[int, torch.Tensor]:
    input_features, feature_lens, owners = _load_audio_features(records, str(reader.root))
    if input_features is None:
        return {}

    audio_config = model_config["audio_config"]
    frontend = reader.load_exact_prefixes(("audio_tower.conv1.", "audio_tower.conv2."))
    state = prepare_audio_state(input_features, feature_lens, frontend, audio_config, device)
    del frontend, input_features
    for layer_idx in range(int(audio_config["encoder_layers"])):
        prefix = f"audio_tower.layers.{layer_idx}"
        weights = reader.load_prefix(prefix + ".")
        layer = AudioLayer(weights, prefix, audio_config, device).eval()
        with torch.no_grad():
            hidden = _forward_audio_layer(layer, state, device, forward_token_budget)
        state = AudioState(hidden, state.segments)
        del layer, weights, hidden
        clear_device_cache(device)

    tail = reader.load_exact_prefixes(("audio_tower.ln_post.", "audio_tower.proj."))
    by_owner: Dict[int, List[torch.Tensor]] = defaultdict(list)
    offset = 0
    with torch.no_grad():
        for owner, length in zip(owners, state.segments):
            x = state.hidden[offset:offset + length].to(device)
            offset += length
            if int(audio_config.get("audio_merge_size", 1)) == 2:
                x = F.avg_pool1d(x.transpose(0, 1), kernel_size=2, stride=2).transpose(0, 1)
            x = _layer_norm(
                x,
                tail["audio_tower.ln_post.weight"].to(device),
                tail["audio_tower.ln_post.bias"].to(device)
                if "audio_tower.ln_post.bias" in tail else None,
                1e-5,
            )
            x = _linear(x, tail, "audio_tower.proj", device)
            by_owner[owner].append(x.to(torch.bfloat16).cpu())
    if offset != state.hidden.shape[0]:
        raise RuntimeError(
            f"audio split consumed {offset} frames, tower produced {state.hidden.shape[0]}"
        )
    result = {owner: torch.cat(chunks, dim=0) for owner, chunks in by_owner.items()}
    del state, tail
    clear_device_cache(device)
    return result


def _default_prompt(record: OmniCalibrationRecord) -> str:
    if record.image and record.audio:
        return "请结合图片和音频内容回答问题。"
    if record.image:
        return "请描述图片内容并回答相关问题。"
    if record.audio:
        return "请理解音频内容并回答相关问题。"
    return "请根据给定文本完成任务。"


def _format_prompt(record: OmniCalibrationRecord) -> str:
    media = ""
    if record.image:
        media += "<|vision_start|><|image_pad|><|vision_end|>"
    if record.audio:
        media += "<|audio_start|><|audio_pad|><|audio_end|>"
    text = record.text.strip() or _default_prompt(record)
    return (
        "<|pangu_text_start|><|message_start|>用户："
        f"{media}{text}<|message_end|><|message_start|>助手："
    )


def _expand_ids(
    tokenizer,
    records: List[OmniCalibrationRecord],
    vision: Dict[int, torch.Tensor],
    audio: Dict[int, torch.Tensor],
    seq_len: int,
) -> List[torch.Tensor]:
    """Create variable-length ids whose placeholder counts match tower outputs."""
    image_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    audio_id = int(tokenizer.convert_tokens_to_ids("<|audio_pad|>"))
    samples: List[torch.Tensor] = []
    for idx, record in enumerate(records):
        base = tokenizer.encode(_format_prompt(record), add_special_tokens=False)
        expanded: List[int] = []
        image_done = audio_done = False
        for token_id in base:
            if token_id == image_id and idx in vision and not image_done:
                expanded.extend([image_id] * int(vision[idx].shape[0]))
                image_done = True
            elif token_id == audio_id and idx in audio and not audio_done:
                expanded.extend([audio_id] * int(audio[idx].shape[0]))
                audio_done = True
            else:
                expanded.append(int(token_id))
        if record.image and not image_done:
            raise RuntimeError(f"record {idx}: image placeholder was not tokenized")
        if record.audio and not audio_done:
            raise RuntimeError(f"record {idx}: audio placeholder was not tokenized")
        if len(expanded) > seq_len and not record.image and not record.audio:
            expanded = expanded[:seq_len]
        if len(expanded) > seq_len:
            raise ValueError(
                f"record {idx}: expanded multimodal prompt has {len(expanded)} tokens, "
                f"larger than --seq-len {seq_len}; increase --seq-len or lower media size"
            )
        samples.append(torch.tensor(expanded, dtype=torch.long))
    return samples


def load_omni_calibration_hidden(
    *,
    model_dir: str,
    manifest: str,
    n_samples: int,
    seq_len: int,
    backend,
    device,
    min_pixels: int = 50176,
    max_pixels: int = 401408,
    image_use_fast: bool = True,
    forward_token_budget: int = 4096,
) -> CalibrationInputs:
    """Return variable-length decoder inputs and aligned modality masks."""
    from transformers import AutoTokenizer

    records = load_omni_manifest(manifest, n_samples)
    reader = CheckpointReader(model_dir)
    model_config = json.loads((Path(model_dir) / "config.json").read_text())
    print(
        f"Omni calibration : {manifest} ({len(records)} requests; "
        f"text_only={sum(not r.image and not r.audio for r in records)}, "
        f"images={sum(bool(r.image) for r in records)}, "
        f"audio={sum(bool(r.audio) for r in records)})",
        flush=True,
    )
    vision = _encode_images(
        records, reader, model_config, device,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_use_fast=image_use_fast,
        forward_token_budget=forward_token_budget,
    )
    audio = _encode_audio(
        records, reader, model_config, device,
        forward_token_budget=forward_token_budget,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    input_ids = _expand_ids(tokenizer, records, vision, audio, seq_len)
    image_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    audio_id = int(tokenizer.convert_tokens_to_ids("<|audio_pad|>"))
    hidden: List[torch.Tensor] = []
    token_is_text: List[torch.Tensor] = []
    for idx, sample_ids in enumerate(input_ids):
        sample_hidden = backend.embed(sample_ids.unsqueeze(0), device)
        for token_id, embeddings, name in (
            (image_id, vision.get(idx), "image"),
            (audio_id, audio.get(idx), "audio"),
        ):
            if embeddings is None:
                continue
            positions = torch.nonzero(sample_ids == token_id, as_tuple=False).flatten()
            if positions.numel() != embeddings.shape[0]:
                raise RuntimeError(
                    f"record {idx}: {name} placeholder count {positions.numel()} "
                    f"!= embedding count {embeddings.shape[0]}"
                )
            sample_hidden[0, positions.to(sample_hidden.device)] = embeddings.to(
                device=sample_hidden.device, dtype=sample_hidden.dtype,
            )
        # The decoder runner distributes samples across devices.  Keep the
        # shared calibration state on CPU instead of pinning every request to
        # the tower device (normally NPU 0).
        hidden.append(sample_hidden.cpu())
        token_is_text.append(
            ((sample_ids != image_id) & (sample_ids != audio_id)).cpu()
        )
    del vision, audio
    clear_device_cache(device)
    print(
        "  built decoder inputs: "
        f"samples={len(hidden)} seq={[x.shape[1] for x in hidden]} "
        f"hidden={hidden[0].shape[-1]} dtype={hidden[0].dtype}",
        flush=True,
    )
    return CalibrationInputs(hidden=hidden, token_is_text=token_is_text)
