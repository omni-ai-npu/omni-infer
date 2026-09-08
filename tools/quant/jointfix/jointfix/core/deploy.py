# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Deployment-format assembly — a loadable compressed-tensors model.

The runner writes per-layer quantized tensors (layer_NNNN.safetensors). This
assembles them, plus the original non-layer tensors (embed / norm / lm_head) and
any uncalibrated quantizable weights (RTN), into a drop-in compressed-tensors
checkpoint: the original sharding/filenames are reused, with
model.safetensors.index.json + config.json (carrying the quantization_config) +
copied auxiliary files (tokenizer, modeling code).

Writes the standard compressed-tensors quantization_config schema and
ignore-list semantics.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from jointfix.core.primitives import rtn_quantize, should_quantize


def build_quantization_config(ignore_list, global_compression_ratio=None,
                              kv_cache_scheme=None) -> dict:
    """compressed-tensors W8A8 dynamic config."""
    return {
        "quant_method": "compressed-tensors",
        "quantize": "w8a8_dynamic",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "targets": ["Linear"],
                "weights": {
                    "type": "int", "num_bits": 8, "symmetric": True,
                    "strategy": "channel", "dynamic": False,
                    "observer": "minmax", "actorder": None,
                    "group_size": None, "block_structure": None, "observer_kwargs": {},
                },
                "input_activations": {
                    "type": "int", "num_bits": 8, "symmetric": True,
                    "strategy": "token", "dynamic": True,
                    "actorder": None, "group_size": None,
                    "block_structure": None, "observer": None, "observer_kwargs": {},
                },
                "output_activations": None,
            }
        },
        "ignore": ignore_list,
        "sparsity_config": {}, "transform_config": {},
        "global_compression_ratio": global_compression_ratio,
        "kv_cache_scheme": kv_cache_scheme,
    }


def _orig_weight_map(model_dir: Path) -> dict:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    wmap = {}
    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            wmap[k] = "model.safetensors"
    return wmap


def finalize_model(orig_model_dir, quant_dir, out_dir=None, skip_patterns=None,
                   kv_cache_scheme=None, rtn_uncalibrated=True) -> Path:
    """
    Assemble the deployment model. Returns the output dir.

    quant_dir holds layer_NNNN.safetensors from the runner. Reuses the original
    sharding: each original shard is rewritten with its layers' int8+scale tensors
    (from the per-layer files), RTN for any quantizable weight not covered, and
    passthrough for the rest.

    rtn_uncalibrated: when True (default) any quantizable weight
    NOT in a per-layer file is RTN-quantized — on a PARTIAL run that means RTN-ing
    every uncalibrated layer (slow: tens of GB of CPU work on a 92B). When False,
    only shards that actually contain a per-layer (calibrated) tensor are written —
    fast, for testing the format / partial runs; a full run has no uncalibrated
    weights so the flag is moot.
    """
    orig = Path(orig_model_dir)
    qd = Path(quant_dir)
    out = Path(out_dir) if out_dir else qd
    out.mkdir(parents=True, exist_ok=True)
    skip = list(skip_patterns or [])

    # tensor-name -> per-layer file that holds its quantized version
    layer_map = {}
    for lf in sorted(qd.glob("layer_*.safetensors")):
        with safe_open(str(lf), framework="pt") as f:
            for k in f.keys():
                layer_map[k] = lf

    orig_wmap = _orig_weight_map(orig)
    by_shard = defaultdict(list)
    for name, shard in orig_wmap.items():
        by_shard[shard].append(name)

    shards = sorted(by_shard)
    new_index: dict = {}
    quantized_bases: set = set()   # module bases that got a weight_scale (truly quantized)
    linear_bases: set = set()      # module bases with a 2D .weight (Linear-shaped)
    for si, shard in enumerate(shards):
        names = by_shard[shard]
        has_calibrated = any(n in layer_map for n in names)
        if not rtn_uncalibrated and not has_calibrated:
            print(f"  [finalize {si + 1}/{len(shards)}] skip {shard} (no calibrated layer)")
            continue
        print(f"  [finalize {si + 1}/{len(shards)}] {shard} ...", flush=True)

        out_tensors = {}
        # load the per-layer files this shard needs (cache once each)
        layer_files = set()
        for name in names:
            layer_file = layer_map.get(name)
            if layer_file is not None:
                layer_files.add(layer_file)
        lf_cache = {lf: load_file(str(lf)) for lf in layer_files}
        uncovered = [n for n in names if n not in layer_map]
        orig_t = {}
        if uncovered:
            with safe_open(str(orig / shard), framework="pt") as f:
                for n in uncovered:
                    orig_t[n] = f.get_tensor(n)

        for name in names:
            if name in layer_map:
                layer_file = layer_map[name]
                if layer_file not in lf_cache:
                    raise KeyError(f"missing calibrated layer file for {name}")
                d = lf_cache[layer_file]
                if name not in d:
                    raise KeyError(f"missing calibrated tensor {name}")
                out_tensors[name] = d[name]                       # int8 or passthrough
                sn = name.replace(".weight", ".weight_scale")
                if sn in d:
                    out_tensors[sn] = d[sn]
            else:
                t = orig_t.get(name)
                if t is None:
                    raise KeyError(f"missing original tensor {name}")
                if rtn_uncalibrated and should_quantize(name, t, skip):
                    int8_w, scale = rtn_quantize(t.float())
                    out_tensors[name] = int8_w
                    out_tensors[name.replace(".weight", ".weight_scale")] = scale
                else:
                    out_tensors[name] = t                         # passthrough (bf16)

        save_file(out_tensors, str(out / shard))
        for k, v in out_tensors.items():
            new_index[k] = shard
            if k.endswith(".weight_scale"):
                quantized_bases.add(k.removesuffix(".weight_scale"))
            elif k.endswith(".weight") and getattr(v, "ndim", 0) == 2:
                linear_bases.add(k.removesuffix(".weight"))
        del out_tensors, lf_cache, orig_t

    # copy auxiliary files (config.json, tokenizer, modeling *.py) — not shards/index
    for f in orig.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        dst = out / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    # index.json
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": new_index}, indent=2))

    # config.json + quantization_config. ignore = every un-quantized Linear-shaped
    # weight: a 2D .weight with NO weight_scale sibling is, by definition, not
    # quantized. Data-driven (not name-pattern), so it covers backend skips,
    # method-level skips (e.g. --skip-shared-experts), and plain passthrough alike —
    # the old name-pattern version missed method-level skips and broke vLLM loading.
    ignore = sorted(linear_bases - quantized_bases)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg["quantization_config"] = build_quantization_config(ignore, None, kv_cache_scheme)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    return out
