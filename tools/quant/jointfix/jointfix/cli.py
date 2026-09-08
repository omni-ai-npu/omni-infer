# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
jointfix CLI.

    jointfix quantize --backend pangu --method jointfix \\
        --model /path/bf16 --output /path/int8 --calib-data calib.parquet ...

Core args are parsed here; method-specific args are contributed by the method
(method.add_cli_args), so adding a method never touches this file.
"""
from __future__ import annotations

import argparse

import jointfix  # noqa: F401  (populates the registries)
from jointfix.registry import (
    available_backends, available_methods, get_backend, get_method,
)
from jointfix.core.runner import RunConfig, calibrate_and_quantize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jointfix")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quantize", help="quantize a model")
    q.add_argument("--backend", required=True, choices=available_backends())
    q.add_argument("--method", required=True, choices=available_methods())
    q.add_argument("--model", required=True, help="BF16 model directory")
    q.add_argument("--output", required=True, help="output directory")
    q.add_argument(
        "--calib-data", required=True,
        help="text parquet (default) or an Omni JSONL manifest with --calib-format omni",
    )
    q.add_argument(
        "--calib-format", choices=["text", "omni"], default="text",
        help="text: tokenize parquet; omni: run BF16 media towers/projectors and "
             "replace decoder placeholder embeddings",
    )
    q.add_argument("--n-samples", type=int, default=128)
    q.add_argument("--seq-len", type=int, default=2048)
    q.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "npu"])
    q.add_argument("--num-devices", type=int, default=1)
    q.add_argument("--start-layer", type=int, default=0)
    q.add_argument("--end-layer", type=int, default=None)
    mm = q.add_argument_group("Omni LLM calibration")
    mm.add_argument("--mm-min-pixels", type=int, default=50176)
    mm.add_argument("--mm-max-pixels", type=int, default=401408)
    mm.add_argument(
        "--mm-image-use-fast", action=argparse.BooleanOptionalAction, default=True,
    )
    mm.add_argument("--mm-forward-token-budget", type=int, default=4096)
    q.add_argument("--no-finalize", action="store_true", default=False,
                   help="stop after per-layer quantization; skip assembling the "
                        "deployable compressed-tensors model (run `jointfix finalize` "
                        "later). Default: quantize finalizes in one step.")

    # Let the chosen method inject its own argument group.
    method_name = _peek(["--method"])
    if method_name:
        get_method(method_name)().add_cli_args(q)

    fin = sub.add_parser("finalize",
                         help="assemble per-layer outputs into a loadable compressed-tensors model")
    fin.add_argument("--model", required=True, help="original BF16 model dir")
    fin.add_argument("--quantized", required=True, help="dir with layer_*.safetensors")
    fin.add_argument("--output", default=None, help="output dir (default: in-place in --quantized)")
    fin.add_argument("--backend", default="pangu", choices=available_backends())
    fin.add_argument(
        "--method", default="jointfix", choices=available_methods(),
        help="method used for quantization; jointfix-mdmixq preserves MTP as BF16",
    )
    fin.add_argument("--calibrated-only", action="store_true",
                     help="only write shards with a calibrated layer (fast; skips RTN of "
                          "uncalibrated layers — use for testing / partial runs)")
    fin.add_argument("--skip-shared-experts", action="store_true", default=False,
                     help="keep mlp.shared_experts in BF16 — MUST match the quantize run's "
                          "flag, else uncalibrated shared experts (the MTP layers) get "
                          "RTN-quantized despite it")

    return p


def _peek(flags) -> "str | None":
    """Pre-scan argv to know which method's args to register (a tiny two-pass)."""
    import sys
    for i, a in enumerate(sys.argv):
        if a in flags and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        for f in flags:
            if a.startswith(f + "="):
                return a.split("=", 1)[1]
    return None


def _deploy_skip_patterns(backend, args):
    """
    Build finalize skips without changing the legacy JointFix defaults.

    ``--skip-shared-experts`` applies to every method as before. Full-layer MTP
    preservation is added only for ``jointfix-mdmixq``; plain ``jointfix`` keeps
    the historical RTN handling of uncalibrated MTP tensors.
    """
    skip = backend.skip_patterns()
    if (getattr(args, "method", "jointfix") == "jointfix-mdmixq"
            and hasattr(backend, "mtp_skip_patterns")):
        skip = skip + backend.mtp_skip_patterns()
    if getattr(args, "skip_shared_experts", False):
        skip = skip + ["mlp.shared_experts"]
    return skip


def _finalize_deploy(model_dir, quant_dir, skip_patterns, *, enabled):
    """
    Assemble the deployable compressed-tensors model in-place into `quant_dir`
    (the run's --output), so a single `quantize` yields a loadable model. Any
    quantizable weight not covered by a per-layer file is RTN-quantized. Returns the
    deploy dir, or None when disabled (--no-finalize -> stop at per-layer artifacts).
    """
    if not enabled:
        return None
    from jointfix.core.deploy import finalize_model
    return finalize_model(model_dir, quant_dir, None,
                          skip_patterns=skip_patterns, rtn_uncalibrated=True)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.cmd == "finalize":
        from jointfix.core.deploy import finalize_model
        backend = get_backend(args.backend)(args.model)
        out = finalize_model(args.model, args.quantized, args.output,
                             skip_patterns=_deploy_skip_patterns(backend, args),
                             rtn_uncalibrated=not args.calibrated_only)
        print(f"deployment model written to: {out}")
        return

    backend = get_backend(args.backend)(args.model)
    method = get_method(args.method)()
    method.configure(args)            # apply parsed method args onto its config
    if args.method == "jointfix-mdmixq" and args.calib_format != "omni":
        raise ValueError("jointfix-mdmixq requires --calib-format omni")
    cfg = RunConfig(
        model_dir=args.model, output_dir=args.output, calib_data=args.calib_data,
        n_samples=args.n_samples, seq_len=args.seq_len, device=args.device,
        num_devices=args.num_devices, start_layer=args.start_layer,
        end_layer=args.end_layer,
        calib_format=args.calib_format,
        mm_min_pixels=args.mm_min_pixels,
        mm_max_pixels=args.mm_max_pixels,
        mm_image_use_fast=args.mm_image_use_fast,
        mm_forward_token_budget=args.mm_forward_token_budget,
    )
    calibrate_and_quantize(backend, method, cfg)

    # One-step by default: assemble the deployable model right after quantize.
    deploy = _finalize_deploy(args.model, args.output,
                              _deploy_skip_patterns(backend, args),
                              enabled=not args.no_finalize)
    if deploy is not None:
        print(f"deployment model written to: {deploy}")
    else:
        print(f"per-layer outputs written to: {args.output}\n"
              f"  (--no-finalize set — assemble later with: jointfix finalize "
              f"--model {args.model} --quantized {args.output} --output <deploy_dir> "
              f"--method {args.method})")


if __name__ == "__main__":
    main()
