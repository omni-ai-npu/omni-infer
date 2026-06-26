#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Offline inference test for Pangu V2 prefill node with Automatic Prefix Caching (APC)
and optional OmniCache.

This script demonstrates:
1. Loading Pangu V2 model on Ascend NPU with TP=8
2. Enabling APC for prefix caching
3. Optional OmniCache integration (requires kv_transfer_config for prefill role)
4. Running multiple queries with shared prefixes to verify APC effectiveness

NOTE: This is a PREFILL-ONLY test. No PD disaggregation or KV transfer to decode node.
When OmniCache is enabled, it acts as kv_producer for local cache management only.

Environment variables (have defaults, can be overridden):
    MODEL_PATH: Path to Pangu V2 model (default: /data/models/PanguV2/92B_iter_0000180/)
    TP_SIZE: Tensor parallel size (default: 8)
    DP_SIZE: Data parallel size (default: 1)
    MAX_LEN: Maximum model length (default: 8192)
    BSZ: Batch size / max num seqs (default: 64)
    NUM_QUERIES: Number of test queries (default: 4)
    PREFIX_TOKENS: Length of shared prefix in tokens (default: 512)
    WARM_UP: Number of warm-up rounds (default: 1)
    LOAD_FORMAT: Model load format - "auto" or "dummy" (default: auto)
    ASCEND_RT_VISIBLE_DEVICES: NPU device visibility (default: 0,1,2,3,4,5,6,7)

Usage:
    # Run with APC enabled, OmniCache disabled (default)
    python offline_prefill_omnicache.py

    # Run with APC disabled
    python offline_prefill_omnicache.py --disable-apc

    # Run with both APC and OmniCache enabled
    python offline_prefill_omnicache.py --enable-omnicache

    # Run with OmniCache only (APC disabled)
    python offline_prefill_omnicache.py --disable-apc --enable-omnicache

    # Customize prefix length, number of queries, and warm-up
    python offline_prefill_omnicache.py -n 8 -p 1024 -w 2

    # Test different prefix lengths to see APC scaling
    python offline_prefill_omnicache.py -p 256   # short prefix
    python offline_prefill_omnicache.py -p 1024  # medium prefix
    python offline_prefill_omnicache.py -p 4096  # long prefix

    # Skip warm-up for cold start measurement
    python offline_prefill_omnicache.py -w 0

    # Skip weight loading (for faster testing without actual inference)
    python offline_prefill_omnicache.py --dummy

    # Override configuration via environment variables
    MODEL_PATH=/path/to/model TP_SIZE=4 MAX_LEN=4096 python offline_prefill_omnicache.py
"""

import os
import time
import argparse
import socket
import random
import string
from dataclasses import dataclass

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Test configuration."""
    model_path: str = "/data/models/PanguV2/92B_iter_0000180/"
    tp_size: int = 8
    dp_size: int = 1
    max_len: int = 8192 * 16
    bsz: int = 64
    kv_parallel_size: int = 2
    enable_apc: bool = True
    enable_omnicache: bool = False
    num_queries: int = 4  # Number of test queries to run
    prefix_tokens: int = 512  # Length of shared prefix in tokens
    warm_up: int = 1  # Number of warm-up rounds before measurement
    load_format: str = "auto"  # "auto" or "dummy" (skip weight loading)


def get_config_from_env() -> Config:
    """Create config from environment variables."""
    return Config(
        model_path=os.environ.get("MODEL_PATH", Config.model_path),
        tp_size=int(os.environ.get("TP_SIZE", str(Config.tp_size))),
        dp_size=int(os.environ.get("DP_SIZE", str(Config.dp_size))),
        max_len=int(os.environ.get("MAX_LEN", str(Config.max_len))),
        bsz=int(os.environ.get("BSZ", str(Config.bsz))),
        kv_parallel_size=int(os.environ.get("KV_PARALLEL_SIZE", str(Config.kv_parallel_size))),
        num_queries=int(os.environ.get("NUM_QUERIES", str(Config.num_queries))),
        prefix_tokens=int(os.environ.get("PREFIX_TOKENS", str(Config.prefix_tokens))),
        warm_up=int(os.environ.get("WARM_UP", str(Config.warm_up))),
        load_format=os.environ.get("LOAD_FORMAT", Config.load_format),
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_local_ip() -> str:
    """Get the local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pangu V2 offline inference test with optional APC and OmniCache"
    )
    parser.add_argument(
        "--enable-apc",
        action="store_true",
        default=None,
        help="Enable Automatic Prefix Caching (default: enabled)"
    )
    parser.add_argument(
        "--disable-apc",
        action="store_true",
        help="Disable Automatic Prefix Caching"
    )
    parser.add_argument(
        "--enable-omnicache",
        action="store_true",
        help="Enable OmniCache (requires kv_transfer_config as kv_producer)"
    )
    parser.add_argument(
        "--disable-omnicache",
        action="store_true",
        help="Disable OmniCache (default)"
    )
    parser.add_argument(
        "-n", "--num-queries",
        type=int,
        default=None,
        help="Number of test queries to run (default: 4)"
    )
    parser.add_argument(
        "-p", "--prefix-tokens",
        type=int,
        default=None,
        help="Length of shared prefix in tokens (default: 512)"
    )
    parser.add_argument(
        "-w", "--warm-up",
        type=int,
        default=None,
        help="Number of warm-up rounds before measurement (default: 1)"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy load format (skip weight loading for faster testing)"
    )
    return parser.parse_args()


def setup_base_environment():
    """Set up base environment variables for Pangu V2."""
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    os.environ.setdefault("OMNI_NPU_PATCHES_DIR", "pangu_v2_hybrid")
    os.environ.setdefault("OMNI_NPU_VLLM_PATCHES", "ALL")
    os.environ.setdefault("HYBRID_ATTN_GROUP_SIZE", "16")
    os.environ.setdefault("GLOO_SOCKET_IFNAME","enp23s0f3")
    os.environ.setdefault("HCCL_SOCKET_IFNAME","enp23s0f3")

def setup_omnicache_environment():
    """Set up OmniCache-specific environment variables."""
    os.environ["ENABLE_OMNI_CACHE"] = "1"
    os.environ.setdefault("P_NODE_LIST", get_local_ip())
    os.environ.setdefault("OMNI_CACHE_MMAP_FILE", "omni_cache_p")
    os.environ.setdefault("OMNI_CACHE_MMAP_PATH", "/dev/hugepages/omni_cache_p")

    # Add omni-cache to plugins
    base_plugins = "omni-npu,omni_npu_patches,omni_custom_models"
    os.environ["VLLM_PLUGINS"] = f"{base_plugins},omni-cache"


def setup_no_omnicache_environment():
    """Set up environment without OmniCache."""
    os.environ["ENABLE_OMNI_CACHE"] = "0"
    os.environ["VLLM_PLUGINS"] = "omni-npu,omni_npu_patches,omni_custom_models"


def setup_environment(config: Config):
    """Set up all environment variables based on config."""
    setup_base_environment()
    if config.enable_omnicache:
        setup_omnicache_environment()
    else:
        setup_no_omnicache_environment()


def print_config(config: Config):
    """Print configuration summary."""
    print("=" * 60)
    print("Pangu V2 Prefill APC Test")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"TP_SIZE: {config.tp_size}, DP_SIZE: {config.dp_size}")
    print(f"MAX_LEN: {config.max_len}, BSZ: {config.bsz}")
    print(f"ENABLE_APC: {config.enable_apc}")
    print(f"ENABLE_OMNICACHE: {config.enable_omnicache}")
    print(f"NUM_QUERIES: {config.num_queries}")
    print(f"PREFIX_TOKENS: {config.prefix_tokens}")
    print(f"WARM_UP: {config.warm_up}")
    print(f"LOAD_FORMAT: {config.load_format}")
    if config.enable_omnicache:
        print(f"P_NODE_LIST: {os.environ.get('P_NODE_LIST')}")
        print(f"OMNI_CACHE_MMAP_PATH: {os.environ.get('OMNI_CACHE_MMAP_PATH')}")
    print(f"VLLM_PLUGINS: {os.environ.get('VLLM_PLUGINS')}")
    print("=" * 60)


# =============================================================================
# LLM Creation
# =============================================================================

def create_llm(config: Config) -> LLM:
    """Create and return LLM instance."""
    print("\nInitializing LLM...")
    init_start = time.time()

    llm_kwargs = dict(
        model=config.model_path,
        tensor_parallel_size=config.tp_size,
        data_parallel_size=config.dp_size,
        max_model_len=config.max_len,
        max_num_seqs=config.bsz,
        enable_prefix_caching=config.enable_apc,
        block_size=128,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        enable_expert_parallel=True,
        disable_hybrid_kv_cache_manager=False,  # Required for Pangu V2 hybrid attention
        load_format=config.load_format,
    )

    if config.enable_omnicache:
        llm_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OmniCacheConnector",
            kv_role="kv_producer",
            kv_rank=0,
            kv_parallel_size=config.kv_parallel_size,
        )

    llm = LLM(**llm_kwargs)

    init_time = time.time() - init_start
    print(f"LLM initialized in {init_time:.2f} seconds")

    return llm


# =============================================================================
# Query Execution
# =============================================================================

def generate_shared_prefix(num_tokens: int) -> str:
    """Generate a shared prefix of approximately the requested token count.

    Uses repetitive but meaningful content to reach the target token count.
    Approximation: ~4 characters per token for Chinese/English mixed text.
    """
    base_text = """You are a helpful assistant analyzing tabular data. Here is a table of employee information:

| ID  | Name          | Age | Department    | Role          | Years Exp | Salary  |
|-----|---------------|-----|---------------|---------------|-----------|---------|
| 1   | Alice Wang    | 32  | Engineering   | Senior Dev    | 8         | 180000  |
| 2   | Bob Chen      | 28  | ML Team       | ML Engineer   | 5         | 165000  |
| 3   | Carol Liu     | 35  | Engineering   | Tech Lead     | 12        | 220000  |
| 4   | David Zhang   | 26  | ML Team       | Junior Dev    | 3         | 120000  |
| 5   | Emma Li       | 30  | Research      | Researcher    | 6         | 175000  |
| 6   | Frank Wu      | 33  | Engineering   | Senior Dev    | 9         | 185000  |
| 7   | Grace Huang   | 29  | ML Team       | ML Engineer   | 4         | 155000  |
| 8   | Henry Zhao    | 31  | Research      | Senior Res    | 7         | 190000  |

"""
    # Estimate: ~4 chars per token
    target_chars = num_tokens * 4

    # Repeat the table content to reach target length
    prefix = base_text
    while len(prefix) < target_chars:
        prefix += f"\nAdditional context line {len(prefix) // len(base_text) + 1}: "
        prefix += "This is additional padding text to reach the desired prefix length. "
        prefix += "The model should cache this prefix for efficient reuse. "

    return prefix[:target_chars]


def generate_test_queries(num_queries: int) -> list:
    """Generate test query suffixes."""
    query_templates = [
        "Question: What is the average age of employees in the ML Team?",
        "Question: Who has the highest salary in Engineering?",
        "Question: How many years of experience does the Research team have combined?",
        "Question: List all employees under 30 years old.",
        "Question: What is the total salary of all employees?",
        "Question: Which department has the most employees?",
        "Question: What is the average years of experience?",
        "Question: Who is the youngest employee?",
    ]
    # Cycle through templates if more queries needed
    return [query_templates[i % len(query_templates)] for i in range(num_queries)]


def run_single_query(llm: LLM, sampling_params: SamplingParams, prompt: str, label: str) -> tuple:
    """Run a single query and return timing and output."""
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    end_time = time.time()

    generated_text = outputs[0].outputs[0].text
    elapsed = end_time - start_time

    print("-" * 50)
    print(f"[{label}] Generation time: {elapsed:.3f} seconds")
    output_display = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
    print(f"[{label}] Output: {output_display}")
    print("-" * 50)

    return elapsed, generated_text


def run_queries(llm: LLM, config: Config) -> tuple:
    """Run warm-up queries and then measured test queries."""
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Generate dynamic prefix and queries for measurement
    shared_prefix = generate_shared_prefix(config.prefix_tokens)
    test_queries = generate_test_queries(config.num_queries)

    # Warm-up phase - use random strings to avoid polluting APC cache
    if config.warm_up > 0:
        print("\n" + "=" * 60)
        print(f"Warm-up: Running {config.warm_up} round(s) to warm up the system...")
        print("(Using random strings to avoid polluting APC cache)")
        print("=" * 60)

        for i in range(config.warm_up):
            # Generate random string (~4 chars per token), capped at max_len
            random_chars = string.ascii_letters + string.digits + " "
            warm_up_len = min(config.prefix_tokens * 4, config.max_len - 100)  # Leave room for query suffix
            warm_up_prompt = ''.join(random.choices(random_chars, k=warm_up_len))
            warm_up_prompt += f" Question {i+1}: This is a warm-up query."
            run_single_query(llm, sampling_params, warm_up_prompt, label=f"Warm-up {i+1}")
            time.sleep(0.3)

        print("Warm-up complete.\n")

    # Measurement phase
    print("\n" + "=" * 60)
    if config.enable_apc:
        print(f"Running {config.num_queries} queries with {config.prefix_tokens}-token shared prefix")
        print("(APC enabled - first query caches prefix, subsequent queries hit cache)")
    else:
        print(f"Running {config.num_queries} queries with {config.prefix_tokens}-token shared prefix")
        print("(APC disabled - no prefix caching)")
    print("=" * 60)

    times = []
    outputs = []

    for i, query in enumerate(test_queries, start=1):
        full_prompt = shared_prefix + query
        elapsed, output = run_single_query(llm, sampling_params, full_prompt, label=f"Query {i}")
        times.append(elapsed)
        outputs.append(output)
        time.sleep(0.5)  # Small delay between queries

    return times, outputs


# =============================================================================
# Results Reporting
# =============================================================================

def print_summary(times: list, outputs: list, config: Config):
    """Print performance summary."""
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"APC enabled: {config.enable_apc}")
    print(f"OmniCache enabled: {config.enable_omnicache}")
    print(f"Prefix length: {config.prefix_tokens} tokens")
    print(f"Number of queries: {config.num_queries}")
    print(f"Warm-up rounds: {config.warm_up}")

    # Show individual query times and outputs
    print("\nQuery times and outputs:")
    for i, (t, out) in enumerate(zip(times, outputs), start=1):
        print(f"  Query {i}: {t:.3f}s")
        output_display = out[:150] + "..." if len(out) > 150 else out
        print(f"           Output: {output_display}")

    avg_time = sum(times) / len(times)
    print(f"\nAverage query time: {avg_time:.3f}s")

    if not config.enable_apc:
        print("APC is disabled - all queries are cold (no prefix caching).")
    if len(times) > 1:
        print_speedup_analysis(times)


def print_speedup_analysis(times: list):
    """Print speedup analysis."""
    print(f"\nFirst query (after warm-up):    {times[0]:.3f}s")
    print(f"Subsequent queries (w/ same prefix hit):")

    for i, t in enumerate(times[1:], start=2):
        speedup = times[0] / t if t > 0 else 0
        print(f"  Query {i}: {t:.3f}s ({speedup:.2f}x faster than cold)")

    avg_cached = sum(times[1:]) / len(times[1:])
    overall_speedup = times[0] / avg_cached if avg_cached > 0 else 0

    print(f"\nAverage cached query time: {avg_cached:.3f}s")
    print(f"Overall speedup: {overall_speedup:.2f}x")

    if overall_speedup > 1.1:
        print("\nSignificant Speed up! ")
    else:
        print("\nNo significant speed up.")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    config = get_config_from_env()

    # Apply CLI args to config
    if args.disable_apc:
        config.enable_apc = False
    elif args.enable_apc is not None:
        config.enable_apc = True

    if args.enable_omnicache:
        config.enable_omnicache = True
    elif args.disable_omnicache:
        config.enable_omnicache = False

    if args.num_queries is not None:
        config.num_queries = args.num_queries

    if args.prefix_tokens is not None:
        config.prefix_tokens = args.prefix_tokens

    if args.warm_up is not None:
        config.warm_up = args.warm_up

    if args.dummy:
        config.load_format = "dummy"

    # Setup and run
    setup_environment(config)
    print_config(config)

    llm = create_llm(config)
    times, outputs = run_queries(llm, config)
    print_summary(times, outputs, config)

    print("\nDone.")


if __name__ == "__main__":
    main()