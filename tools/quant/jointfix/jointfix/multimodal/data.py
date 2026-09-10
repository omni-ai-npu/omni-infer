# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Manifest readers and production-matching Omni modality preprocessing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch


def _manifest_paths(path: str, keys: tuple[str, ...], limit: int | None) -> List[str]:
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"calibration manifest not found: {manifest}")
    result = []
    with manifest.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("{"):
                item = json.loads(raw)
                value = next((item[k] for k in keys if item.get(k)), None)
                if value is None:
                    raise ValueError(
                        f"{manifest}:{line_no}: expected one of JSON keys {keys}"
                    )
            else:
                value = raw
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = manifest.parent / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"{manifest}:{line_no}: data file not found: {candidate}")
            result.append(str(candidate))
            if limit is not None and len(result) >= limit:
                break
    if not result:
        raise ValueError(f"no calibration entries found in {manifest}")
    return result


def load_vision_calibration(
    model_dir: str,
    manifest: str,
    *,
    n_samples: int,
    min_pixels: int,
    max_pixels: int,
    image_use_fast: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return concatenated flattened patches and per-image ``grid_thw``."""
    from PIL import Image

    # Importing the concrete processor registers its config and exactly matches
    # serving-side resize/normalize/patch ordering.
    from omni_models.models.openpangu_vl.huggingface.imageprocessor_openpangu_vl import (
        OpenPanguVLImageProcessorFast,
    )

    paths = _manifest_paths(manifest, ("image", "path", "file"), n_samples)
    processor = OpenPanguVLImageProcessorFast.from_pretrained(model_dir)
    processor.image_use_fast = bool(image_use_fast)

    patches, grids = [], []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            encoded = processor(
                images=[image],
                # The checkpoint already carries a `size` object; passing only
                # min/max would leave that object in control. Override `size`
                # explicitly so calibration obeys the requested token budget.
                size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
                return_tensors="pt",
                device="cpu",
            )
        patches.append(encoded["pixel_values"].to(torch.bfloat16))
        grids.append(encoded["image_grid_thw"].to(torch.long))
        print(
            f"  [vision-data] {len(patches)}/{len(paths)} {path} "
            f"grid={grids[-1].tolist()} patches={patches[-1].shape[0]}",
            flush=True,
        )
    return torch.cat(patches, dim=0), torch.cat(grids, dim=0)


def _read_audio(path: str, target_rate: int) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != target_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
    if not np.isfinite(audio).all():
        raise ValueError(f"audio contains NaN/Inf: {path}")
    return np.asarray(audio, dtype=np.float32)


def extract_audio_chunks(extractor, waveform: np.ndarray):
    """Extract non-empty log-mel chunks using serving-compatible boundaries."""
    chunks = (
        waveform[start:start + extractor.n_samples]
        for start in range(0, max(len(waveform), 1), extractor.n_samples)
    )
    result = []
    for chunk in chunks:
        encoded = extractor(
            [chunk],
            sampling_rate=extractor.sampling_rate,
            padding="max_length",
            max_length=extractor.n_samples,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        length = int(encoded["attention_mask"][0].sum().item())
        if length > 0:
            feature = encoded["input_features"][0, :, :length].to(torch.bfloat16)
            result.append((feature, length))
    return result


def load_audio_calibration(
    model_dir: str,
    manifest: str,
    *,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return concatenated valid log-mel frames and their individual lengths."""
    from transformers import WhisperFeatureExtractor

    paths = _manifest_paths(manifest, ("audio", "path", "file"), n_samples)
    extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    features, lengths = [], []
    for path in paths:
        waveform = _read_audio(path, extractor.sampling_rate)
        for feature, length in extract_audio_chunks(extractor, waveform):
            features.append(feature)
            lengths.append(length)
        print(
            f"  [audio-data] {len(features)} chunks from {path} "
            f"seconds={len(waveform) / extractor.sampling_rate:.2f}",
            flush=True,
        )
    if not features:
        raise ValueError("audio calibration set produced no non-empty features")
    return torch.cat(features, dim=1), torch.tensor(lengths, dtype=torch.long)
