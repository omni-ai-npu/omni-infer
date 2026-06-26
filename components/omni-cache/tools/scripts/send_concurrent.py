#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Send N concurrent requests with deterministic request IDs.

Each request gets a predictable ID (``test-req-0000``, ``test-req-0001``, …)
set via the ``X-Request-Id`` HTTP header, which the router forwards to vLLM.

Usage::

    # 10 concurrent requests, 50 tokens each (defaults)
    python tools/scripts/send_concurrent.py

    # 5 requests, 100 tokens, custom endpoint, save responses
    python tools/scripts/send_concurrent.py -n 5 --max-tokens 100 \\
        --url http://10.0.0.1:7077/v1/chat/completions \\
        --model my-model --save-dir /tmp/responses
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


SHORT_PROMPT = (
    "The capital of China is Beijing. The capital of France is Paris. "
    "The capital of Japan is Tokyo. The largest ocean is the Pacific."
)

_VARIED_PROMPTS = [
    "What is the capital of China? Explain its history briefly.",
    "Describe the theory of relativity in simple terms.",
    "Write a short poem about artificial intelligence.",
    "List 5 major achievements of the Renaissance period.",
    "Explain how climate change affects ocean ecosystems.",
    "What are the key differences between Python and C++?",
    "Describe the process of photosynthesis in plants.",
    "What is machine learning? Give a concise overview.",
    "Explain the significance of the Turing test in AI history.",
    "How does blockchain technology work? Keep it brief.",
]


def build_prompt(long: bool = False) -> str:
    if long:
        paragraphs = []
        for i in range(20):
            paragraphs.append(
                f"Paragraph {i+1}: The field of artificial intelligence has "
                f"undergone remarkable transformations since its inception."
            )
        return " ".join(paragraphs)
    return SHORT_PROMPT


def build_prompts(n: int, vary: bool = False, long: bool = False) -> list[str]:
    """Build `n` prompts, cycling through varied ones if `vary` is set."""
    if not vary:
        return [build_prompt(long=long)] * n
    out = []
    for i in range(n):
        p = _VARIED_PROMPTS[i % len(_VARIED_PROMPTS)]
        if long:
            p = build_prompt(long=True) + " " + p
        out.append(p)
    return out


def load_prompts_from_file(path: str, key: str, n: int,
                           seed: int = 42, no_shuffle: bool = False) -> list[str]:
    """Load `n` prompts from a JSONL file, picking from field `key`.

    Uses a fixed random seed so the same prompts are selected every time
    across baseline and omnicache runs.  Set ``no_shuffle=True`` to take
    the first *n* entries in file order.
    """
    import random
    import json as _json
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            val = obj.get(key, "")
            if isinstance(val, str) and val.strip():
                entries.append(val.strip())
    if len(entries) < n:
        n = len(entries)
    if no_shuffle:
        return entries[:n]
    rng = random.Random(seed)
    return rng.sample(entries, n)


def send_request(idx: int, url: str, model: str, prompt: str,
                 max_tokens: int, id_prefix: str, temperature: float = 0.0,
                 ignore_eos: bool = False, logprobs: bool = False,
                 save_dir: str | None = None):
    req_id = f"{id_prefix}-{idx:04d}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if logprobs:
        payload["logprobs"] = True
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": req_id,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=1200000) as resp:
            data = json.loads(resp.read())
            rid = data["id"]
            tokens = data["usage"]["completion_tokens"]
            if save_dir:
                fname = os.path.join(save_dir, f"{req_id}.json")
                with open(fname, "w") as f:
                    json.dump(data, f, indent=2)
            return idx, "ok", rid, tokens
    except Exception as e:
        return idx, "error", str(e), 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, required=True,
                    help="Number of concurrent requests (must match server --max-num-seqs)")
    ap.add_argument("--max-tokens", type=int, default=50,
                    help="Max tokens per request (default: 50)")
    ap.add_argument("--url", default="http://localhost:7077/v1/chat/completions",
                    help="API endpoint URL")
    ap.add_argument("--model", default="deepseek",
                    help="Model name (default: deepseek)")
    ap.add_argument("--id-prefix", default="test-req",
                    help="Request ID prefix (default: test-req)")
    ap.add_argument("--long", action="store_true",
                    help="Use long prompt (20 paragraphs, default: short)")
    ap.add_argument("--vary", action="store_true",
                    help="Use varied prompts per request instead of identical")
    ap.add_argument("--prompt", default=None,
                    help="Custom prompt text (overrides --long and --vary)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (default: 0.0)")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="Disable EOS token stopping")
    ap.add_argument("--logprobs", action="store_true",
                    help="Request token logprobs from the API")
    ap.add_argument("--save-dir", default=None,
                    help="Directory to save response JSON per request")
    ap.add_argument("--prompts-file", default=None,
                    help="JSONL file to load prompts from")
    ap.add_argument("--prompts-key", default="prompt",
                    help="Key in JSONL objects (default: prompt)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for prompt selection (default: 42)")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="Take prompts in file order instead of random")
    ap.add_argument("--delay-ms", type=int, default=0,
                    help="Delay between requests in ms (0 = concurrent, >0 = sequential)")
    ap.add_argument("--serial", action="store_true",
                    help="Wait for each request to finish before sending the next")
    args = ap.parse_args()

    if args.prompts_file:
        prompts = load_prompts_from_file(
            args.prompts_file, args.prompts_key, args.n, seed=args.seed,
            no_shuffle=args.no_shuffle)
    elif args.prompt:
        prompts = [args.prompt] * args.n
    else:
        prompts = build_prompts(n=args.n, vary=args.vary, long=args.long)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    delay = args.delay_ms / 1000.0 if args.delay_ms > 0 else 0
    mode = "staggered" if delay > 0 else "concurrent"
    print(f"Sending {args.n} {mode} requests "
          f"(max_tokens={args.max_tokens}, prefix={args.id_prefix}, "
          f"vary={args.vary}, long={args.long}"
          + (f", delay={args.delay_ms}ms" if delay > 0 else "")
          + ")...")

    import time as _time
    if args.serial:
        for i in range(args.n):
            idx, status, rid, tokens = send_request(
                i, args.url, args.model, prompts[i], args.max_tokens,
                args.id_prefix, args.temperature, args.ignore_eos,
                args.logprobs, args.save_dir)
            print(f"  req {idx}: {status} server_id={rid} tokens={tokens}")
            if delay > 0 and i < args.n - 1:
                _time.sleep(delay)
    else:
        with ThreadPoolExecutor(max_workers=args.n) as pool:
            futs = []
            for i in range(args.n):
                futs.append(pool.submit(
                    send_request, i, args.url, args.model,
                    prompts[i], args.max_tokens, args.id_prefix,
                    args.temperature, args.ignore_eos,
                    args.logprobs, args.save_dir))
                if delay > 0 and i < args.n - 1:
                    _time.sleep(delay)
            for f in as_completed(futs):
                idx, status, rid, tokens = f.result()
                print(f"  req {idx}: {status} server_id={rid} tokens={tokens}")
    print("Done")


if __name__ == "__main__":
    main()
