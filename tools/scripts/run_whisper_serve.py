# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCH_COMPILE_GE", "False")
os.environ["VLLM_PLUGINS"] = "omni-npu,omni_npu_patches"

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--max-model-len", type=int, default=448)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1500)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--served-model-name", default="whisper")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra_args = list(args.vllm_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return args, extra_args


def main() -> None:
    args, extra_args = parse_args()

    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = [
        "vllm",
        "serve",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--limit-mm-per-prompt",
        '{"audio":1}',
        #"--enforce-eager",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        *extra_args,
    ]

    if args.served_model_name:
        sys.argv.extend(["--served-model-name", args.served_model_name])

    if args.trust_remote_code:
        sys.argv.append("--trust-remote-code")

    vllm_main()


if __name__ == "__main__":
    main()
