#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Repro: fire the same prompt concurrently, dump full response bodies.

Usage: python3 repro_repeat.py --n 8 --prompt-id music --out /tmp/repro
"""
import argparse
import json
import threading
import time
from urllib import request as urlreq

PROMPTS = {
    "music": dict(
        max_tokens=256,
        prompt="请从社会、技术、政治三个角度分析20世纪流行音乐的发展脉络，重点讨论爵士、摇滚、朋克、嘻哈和电子音乐。",
    ),
    "dna": dict(
        max_tokens=120,
        prompt="用一两句话说明 DNA 双螺旋结构由谁提出、在哪一年。",
    ),
}


def call(url, prompt, max_tokens, timeout=400):
    body = json.dumps(dict(
        model="deepseek", max_tokens=max_tokens, temperature=0, stream=False,
        messages=[{"role": "user", "content": prompt}],
    )).encode()
    req = urlreq.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlreq.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://0.0.0.0:7077/v1/chat/completions")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prompt-id", choices=list(PROMPTS.keys()), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = PROMPTS[args.prompt_id]
    results = [None] * args.n

    def worker(i):
        results[i] = call(args.url, p["prompt"], p["max_tokens"])

    threads = [threading.Thread(target=worker, args=(i,), name=f"w{i}") for i in range(args.n)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0

    with open(args.out, "w") as f:
        f.write(f"N={args.n}, prompt_id={args.prompt_id}, elapsed={elapsed:.1f}s\n")
        f.write(f"PROMPT: {p['prompt']}\n")
        f.write("=" * 80 + "\n")
        for i, r in enumerate(results):
            f.write(f"--- response {i}, len={len(r)} chars ---\n")
            f.write(r + "\n\n")
    print(f"wrote {args.out}, N={args.n} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
