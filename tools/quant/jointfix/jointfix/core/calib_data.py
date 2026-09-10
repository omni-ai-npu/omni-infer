# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Calibration data loading — model-agnostic.

Loads 'text' from parquet (single or comma-separated multi-source), tokenizes via
the model's AutoTokenizer, and slices into n_samples × seq_len input_ids.

Takes `tokenizer_path` (the model dir) for the tokenizer but is otherwise
model-independent. GPTQ is most sensitive to the calibration distribution — for
tasks whose distribution differs from generic text (agent rollouts, CoT), a
matching calibration set helps (see examples/data/README.md).

Note: _ensure_parquet_deps() pip-installs pandas/pyarrow if missing — a library
smell kept for offline-cluster parity. pandas/pyarrow are declared in pyproject so
this path normally never fires.
"""
from __future__ import annotations

from typing import List

import torch


def _ensure_parquet_deps() -> None:
    import importlib
    import subprocess
    import sys
    for pkg, mod in [("pandas", "pandas"), ("pyarrow", "pyarrow")]:
        if importlib.util.find_spec(mod) is None:
            print(f"[preflight] {pkg} not found — installing...")
            ret = subprocess.call([
                sys.executable, "-m", "pip", "install", pkg, "-q",
                "--trusted-host", "pypi.org",
                "--trusted-host", "files.pythonhosted.org",
                "--trusted-host", "pypi.python.org",
            ])
            if ret != 0:
                print(f"[preflight] WARNING: failed to install {pkg} (ret={ret}), "
                      f"will try import anyway")


def _load_parquet_texts(calib_path: str) -> List[str]:
    """Read the 'text' column from a parquet, dropping nulls."""
    try:
        import pandas as pd
        return pd.read_parquet(calib_path)["text"].dropna().tolist()
    except ImportError:
        import pyarrow.parquet as pq
        col = pq.read_table(calib_path, columns=["text"]).column("text")
        return [v.as_py() for v in col if v.is_valid]


def _texts_to_samples(texts: List[str], tok, n_samples: int,
                      seq_len: int) -> List[List[int]]:
    """
    Encode `texts`, slice into n_samples × seq_len token lists. Pads by
    repeating if texts have fewer tokens than n_samples × seq_len.
    """
    if n_samples <= 0:
        return []
    needed = n_samples * seq_len + 4096
    all_tokens: List[int] = []
    for text in texts:
        if len(all_tokens) >= needed:
            break
        all_tokens.extend(tok.encode(text, add_special_tokens=False))
    if not all_tokens:
        raise ValueError("No tokens extracted from calib texts (empty parquet?)")
    while len(all_tokens) < needed:
        all_tokens = all_tokens + all_tokens
    return [all_tokens[i * seq_len: (i + 1) * seq_len] for i in range(n_samples)]


def load_calibration_data(calib_path: str, tokenizer_path: str, n_samples: int,
                          seq_len: int, calib_weights: "str | None" = None) -> torch.Tensor:
    """
    Load calibration token ids as a [n_samples, seq_len] LongTensor.

    Supports comma-separated multi-source calib paths:
      - single source: straightforward encode + slice.
      - multi source: split n_samples among sources by `calib_weights` (default
        equal), encode each separately to keep its token stream contiguous, then
        concat sample chunks (preserves per-source token distribution).
    """
    from transformers import AutoTokenizer

    _ensure_parquet_deps()

    paths = [p.strip() for p in calib_path.split(",") if p.strip()]
    if calib_weights is None or len(paths) == 1:
        weights = [1.0 / len(paths)] * len(paths)
    else:
        weights = [float(w) for w in calib_weights.split(",")]
        if len(weights) != len(paths):
            raise ValueError(
                f"--calib-weights count {len(weights)} != --calib-data count {len(paths)}")
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError(f"--calib-weights must sum to 1.0, got {sum(weights)}")

    print(f"Tokenizer        : {tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if len(paths) == 1:
        print(f"Calibration data : {paths[0]}")
        texts = _load_parquet_texts(paths[0])
        samples = _texts_to_samples(texts, tok, n_samples, seq_len)
    else:
        print(f"Calibration data : {len(paths)} sources (multi-source)")
        per_source_n = [int(n_samples * w) for w in weights]
        per_source_n[0] += n_samples - sum(per_source_n)
        samples = []
        for path, w, n_i in zip(paths, weights, per_source_n):
            print(f"  source: {path}  weight={w:.2f}  → {n_i} samples")
            if n_i <= 0:
                continue
            texts_i = _load_parquet_texts(path)
            samples.extend(_texts_to_samples(texts_i, tok, n_i, seq_len))

    input_ids = torch.tensor(samples, dtype=torch.long)
    print(f"  {n_samples} x {seq_len} = {n_samples * seq_len:,} calibration tokens")
    return input_ids
