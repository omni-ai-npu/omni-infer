# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import json

import pytest
import torch

from jointfix.multimodal.llm_calib import (
    OmniCalibrationRecord,
    _expand_ids,
    load_omni_manifest,
)


class _Tokenizer:
    image_id = 101
    audio_id = 102

    def convert_tokens_to_ids(self, token):
        return {
            "<|image_pad|>": self.image_id,
            "<|audio_pad|>": self.audio_id,
        }[token]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = [1]
        if "<|image_pad|>" in text:
            ids.append(self.image_id)
        if "<|audio_pad|>" in text:
            ids.append(self.audio_id)
        ids.extend([2, 3])
        return ids


def test_load_omni_manifest_resolves_relative_media(tmp_path):
    image = tmp_path / "image.jpg"
    audio = tmp_path / "audio.wav"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    manifest = tmp_path / "calib.jsonl"
    manifest.write_text(
        json.dumps({"image": image.name, "text": "describe"}) + "\n" +
        json.dumps({"audio": audio.name, "prompt": "transcribe"}) + "\n"
    )

    records = load_omni_manifest(str(manifest), n_samples=2)

    assert records == [
        OmniCalibrationRecord("describe", image=str(image.resolve())),
        OmniCalibrationRecord("transcribe", audio=str(audio.resolve())),
    ]


def test_load_omni_manifest_accepts_text_only_record(tmp_path):
    manifest = tmp_path / "calib.jsonl"
    manifest.write_text(json.dumps({"text": "a real text calibration sample"}) + "\n")

    assert load_omni_manifest(str(manifest), n_samples=1) == [
        OmniCalibrationRecord("a real text calibration sample")
    ]


def test_load_omni_manifest_rejects_empty_record(tmp_path):
    manifest = tmp_path / "calib.jsonl"
    manifest.write_text("{}\n")

    with pytest.raises(ValueError, match="neither media nor text"):
        load_omni_manifest(str(manifest), n_samples=1)


def test_expand_ids_matches_embedding_counts_without_padding():
    records = [OmniCalibrationRecord("question", image="i", audio="a")]
    vision = {0: torch.randn(3, 4)}
    audio = {0: torch.randn(2, 4)}

    samples = _expand_ids(_Tokenizer(), records, vision, audio, seq_len=16)

    assert len(samples) == 1
    assert int((samples[0] == _Tokenizer.image_id).sum()) == 3
    assert int((samples[0] == _Tokenizer.audio_id).sum()) == 2
    assert samples[0].numel() == 8  # real prompt length; no EOS padding to 16


def test_expand_ids_rejects_media_over_seq_len():
    records = [OmniCalibrationRecord("question", image="i")]
    with pytest.raises(ValueError, match="larger than --seq-len"):
        _expand_ids(_Tokenizer(), records, {0: torch.randn(20, 4)}, {}, seq_len=8)


def test_expand_ids_crops_text_only_record_to_seq_len():
    records = [OmniCalibrationRecord("long text")]

    samples = _expand_ids(_Tokenizer(), records, {}, {}, seq_len=2)

    assert samples[0].tolist() == [1, 2]
