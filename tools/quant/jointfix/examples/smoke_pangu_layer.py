# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Single-layer smoke test — run on a machine with a real Pangu checkpoint.

Builds one decoder layer from real weights and runs a single forward. No
quantization — this only verifies that the Pangu backend's load_layer_weights +
build_layer + forward path works end-to-end (the interface the runner depends on).

    PYTHONPATH=. python examples/smoke_pangu_layer.py \
        --model /path/to/pangu_92B --layer 0 --seq 64 --device cuda

Expected tail: "SMOKE PASS". If it NaNs or shape-mismatches, the backend's
build/forward seam is wrong and must be fixed before the runner.
"""
import argparse

import torch

from jointfix.backends.pangu import PanguBackend
from jointfix.core.devices import resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="BF16 Pangu model directory")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--device", default="cpu", help="cpu | npu | cuda | auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[0/4] device: {device}")
    backend = PanguBackend(args.model)

    specs = backend.layer_specs()
    spec = specs[args.layer]
    extra = {k: v for k, v in spec.extra.items() if k != "_pangu_spec"}
    print(f"[1/4] layer {spec.layer_idx}: is_moe={spec.is_moe} {extra}")

    print("[2/4] loading layer weights ...")
    weights = backend.load_layer_weights(spec.layer_idx)
    print(f"      {len(weights)} tensors "
          f"(e.g. {next(iter(weights))} {tuple(next(iter(weights.values())).shape)})")

    print("[3/4] building layer (meta -> device) + embedding a random sample ...")
    layer = backend.build_layer(spec, weights, device)
    cfg = backend._config()
    torch.manual_seed(0)
    input_ids = torch.randint(0, int(cfg["vocab_size"]), (1, args.seq))
    hidden = backend.embed(input_ids, device)
    print(f"      hidden: {tuple(hidden.shape)} {hidden.dtype}")

    print("[4/4] forward ...")
    with torch.no_grad():
        out = layer(hidden)

    finite = torch.isfinite(out).all().item()
    bsz, seq = out.shape[0], out.shape[1]
    feat = out.numel() // (bsz * seq)   # features per token
    print(f"\nOUTPUT shape={tuple(out.shape)} dtype={out.dtype}  (feat/token={feat})")
    print(f"  finite={finite}  mean={out.float().mean().item():.4e}  "
          f"std={out.float().std().item():.4e}  absmax={out.float().abs().max().item():.4e}")
    if not finite:
        raise RuntimeError("output has NaN/Inf")

    # The valid inter-layer hidden shape is "whatever the next layer accepts" — so
    # validate it DIRECTLY by feeding the output back through the layer (what layer
    # N+1 does in the runner). MHC carries streams either as 4D [b,s,stream,hidden]
    # (layers with block_post_layernorm) or flattened 3D [b,s,stream*hidden]
    # (without); both are consumed by _flatten_hidden_states, so we assert no fixed
    # dim — only that re-entry is finite and shape-stable.
    print("[+]   re-entry check (inter-layer chaining) ...")
    with torch.no_grad():
        out2 = layer(out)
    if not torch.isfinite(out2).all().item():
        raise RuntimeError("re-entry produced NaN/Inf")
    if out2.shape != out.shape:
        raise RuntimeError(f"re-entry shape changed: {out2.shape} vs {out.shape}")

    h = spec.hidden_size
    ns = spec.extra.get("mhc_num_stream", 1)
    print("\nSMOKE PASS  build_layer + forward + inter-layer chaining work on real weights.")
    if ns > 1 and feat == h * ns:
        form = "4D [b,s,stream,hidden]" if out.dim() == 4 else "flattened-3D [b,s,stream*hidden]"
        print(f"     (MHC: {feat} feats/token = {ns} streams × {h} hidden; {form})")


if __name__ == "__main__":
    main()
