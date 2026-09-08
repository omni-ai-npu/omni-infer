#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Calibrate OpenPangu Omni ViT/Audio with JointFix and merge the artifacts."""
from __future__ import annotations

import argparse

from jointfix.multimodal.deploy import (
    finalize_text_artifacts,
    merge_multimodal_artifacts,
    validate_multimodal_artifacts,
    validate_text_base_checkpoint,
)
from jointfix.multimodal.runner import quantize_omni_multimodal


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="run real multimodal JointFix calibration")
    calibrate.add_argument("--model", required=True, help="original BF16 Omni checkpoint")
    calibrate.add_argument("--output", required=True, help="empty artifact directory")
    calibrate.add_argument("--modality", choices=["vision", "audio", "both"], default="both")
    calibrate.add_argument("--vision-manifest")
    calibrate.add_argument("--audio-manifest")
    calibrate.add_argument("--n-vision-samples", type=int, default=32)
    calibrate.add_argument("--n-audio-samples", type=int, default=32)
    calibrate.add_argument("--min-pixels", type=int, default=50176)
    calibrate.add_argument("--max-pixels", type=int, default=2500000)
    calibrate.add_argument("--image-use-fast", action=argparse.BooleanOptionalAction, default=True)
    calibrate.add_argument("--device", choices=["cpu", "cuda", "npu", "auto"], default="npu")
    calibrate.add_argument("--sample-rows", type=int, default=512)
    calibrate.add_argument("--forward-token-budget", type=int, default=4096)
    calibrate.add_argument("--gptq-block-size", type=int, default=128)
    calibrate.add_argument("--gptq-damp", type=float, default=0.01)
    calibrate.add_argument("--smooth-scale-min", type=float, default=0.25)
    calibrate.add_argument("--smooth-scale-max", type=float, default=4.0)
    calibrate.add_argument("--max-output-rel-rmse", type=float, default=0.03)
    calibrate.add_argument("--min-output-cosine", type=float, default=0.9995)
    calibrate.add_argument("--max-update-rel-rmse", type=float, default=0.15)
    calibrate.add_argument("--min-update-cosine", type=float, default=0.98)

    text_base = sub.add_parser(
        "finalize-text", help="assemble LLM layer artifacts; preserve ViT/Audio BF16"
    )
    text_base.add_argument("--model", required=True, help="original BF16 Omni checkpoint")
    text_base.add_argument("--text-artifacts", required=True)
    text_base.add_argument("--output", required=True, help="deployable LLM+MTP INT8 base")

    check_text = sub.add_parser(
        "validate-text", help="validate a deployable LLM+MTP base and BF16 modalities"
    )
    check_text.add_argument("--model", required=True)

    merge = sub.add_parser("merge", help="merge calibrated artifacts; never quantizes uncovered weights")
    merge.add_argument("--base-quant", required=True, help="working language+MTP quantized checkpoint")
    merge.add_argument("--artifacts", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--require-vision", action=argparse.BooleanOptionalAction, default=True)
    merge.add_argument("--require-audio", action=argparse.BooleanOptionalAction, default=True)
    merge.add_argument("--link-unchanged-shards", action=argparse.BooleanOptionalAction, default=True)

    validate = sub.add_parser("validate", help="verify all modality artifacts are strict-GPTQ")
    validate.add_argument("--artifacts", required=True)
    validate.add_argument(
        "--model",
        help="checkpoint config root; defaults to metadata.model when still accessible",
    )
    validate.add_argument("--require-vision", action=argparse.BooleanOptionalAction, default=True)
    validate.add_argument("--require-audio", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "calibrate":
        quantize_omni_multimodal(
            model_dir=args.model,
            output_dir=args.output,
            vision_manifest=args.vision_manifest,
            audio_manifest=args.audio_manifest,
            modality=args.modality,
            n_vision_samples=args.n_vision_samples,
            n_audio_samples=args.n_audio_samples,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            image_use_fast=args.image_use_fast,
            device_name=args.device,
            sample_rows=args.sample_rows,
            forward_token_budget=args.forward_token_budget,
            gptq_block_size=args.gptq_block_size,
            gptq_damp=args.gptq_damp,
            smooth_scale_min=args.smooth_scale_min,
            smooth_scale_max=args.smooth_scale_max,
            max_output_rel_rmse=args.max_output_rel_rmse,
            min_output_cosine=args.min_output_cosine,
            max_update_rel_rmse=args.max_update_rel_rmse,
            min_update_cosine=args.min_update_cosine,
        )
    elif args.command == "finalize-text":
        finalize_text_artifacts(args.model, args.text_artifacts, args.output)
    elif args.command == "validate-text":
        validate_text_base_checkpoint(args.model)
    elif args.command == "merge":
        merge_multimodal_artifacts(
            args.base_quant,
            args.artifacts,
            args.output,
            require_vision=args.require_vision,
            require_audio=args.require_audio,
            link_unchanged_shards=args.link_unchanged_shards,
        )
    else:
        validate_multimodal_artifacts(
            args.artifacts,
            require_vision=args.require_vision,
            require_audio=args.require_audio,
            model_dir=args.model,
        )


if __name__ == "__main__":
    main()
