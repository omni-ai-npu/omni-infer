#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Build a reproducible text/image/audio calibration JSONL."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--wikitext", required=True)
    parser.add_argument("--media-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-samples", type=int, default=16)
    parser.add_argument("--image-samples", type=int, default=8)
    parser.add_argument("--audio-samples", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=896)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    frame = pd.read_parquet(args.wikitext, columns=["text"])
    docs = []
    for value in frame["text"].dropna().tolist():
        text = str(value).strip()
        if not text:
            continue
        title = re.fullmatch(r"=+\s*(.*?)\s*=+", text)
        docs.append(title.group(1) if title else text)
    token_pool = tokenizer("\n\n".join(docs), add_special_tokens=False).input_ids
    needed = args.text_samples * args.text_tokens
    if len(token_pool) < needed:
        raise ValueError(f"WikiText has {len(token_pool)} tokens, need at least {needed}")
    stride = max(args.text_tokens, (len(token_pool) - args.text_tokens) // args.text_samples)
    text_records = []
    for index in range(args.text_samples):
        start = min(index * stride, len(token_pool) - args.text_tokens)
        text = tokenizer.decode(
            token_pool[start:start + args.text_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text_records.append({
            "text": "请阅读以下材料并继续完成语言建模任务。\n\n" + text,
        })

    images, audios = [], []
    with Path(args.media_manifest).open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            item = json.loads(raw)
            if item.get("image") and len(images) < args.image_samples:
                images.append(item)
            elif item.get("audio") and len(audios) < args.audio_samples:
                audios.append(item)
    if len(images) != args.image_samples or len(audios) != args.audio_samples:
        raise ValueError(
            f"media manifest supplied images={len(images)}/{args.image_samples}, "
            f"audio={len(audios)}/{args.audio_samples}"
        )

    records = []
    for index in range(max(len(text_records), len(images), len(audios))):
        if index < len(text_records):
            records.append(text_records[index])
        if index < len(images):
            records.append(images[index])
        if index < len(audios):
            records.append(audios[index])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        f"wrote {output}: total={len(records)} text={len(text_records)} "
        f"image={len(images)} audio={len(audios)} text_tokens={args.text_tokens}"
    )


if __name__ == "__main__":
    main()
