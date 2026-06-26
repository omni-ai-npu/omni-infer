#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Steady-state N-in-flight benchmark for the Pangu PD proxy.

N worker threads each fire the next request as soon as theirs returns,
so there are always N in-flight. Prompts vary in length and topic.
Each prompt has a `required` keyword list (the answer must hit at
least one) and a `forbidden` keyword list (drawn from OTHER prompts,
so a hit means cross-request KV contamination).
"""
import argparse
import json
import random
import threading
import time
from urllib import request as urlreq
from urllib.error import URLError

PROMPTS = [
    dict(label="music_long_zh", max_tokens=256,
         prompt="请从社会、技术、政治三个角度分析20世纪流行音乐的发展脉络，讨论爵士、摇滚、朋克、嘻哈、电子舞曲等流派。",
         required=["音乐", "爵士", "摇滚", "嘻哈", "朋克", "电子"],
         forbidden=["会计分录", "融资租赁", "Fibonacci", "def "]),
    dict(label="quantum_med_zh", max_tokens=256,
         prompt="请解释量子叠加和量子纠缠的基本原理，并给出一个应用例子。",
         required=["量子", "叠加", "纠缠"],
         forbidden=["爵士", "摇滚", "Fibonacci", "def "]),
    dict(label="integral_short_en", max_tokens=96,
         prompt="Compute the definite integral of x^2 from 0 to 1. Show brief steps.",
         required=["1/3", "0.333", "x^3"],
         forbidden=["quantum", "爵士", "Fibonacci", "rock"]),
    dict(label="frev_med_en", max_tokens=256,
         prompt="Summarise the main causes of the French Revolution of 1789 in a short paragraph.",
         required=["France", "Bastille", "revolution", "1789", "king"],
         forbidden=["quantum", "integral", "Fibonacci", "爵士"]),
    dict(label="fib_code_en", max_tokens=200,
         prompt="Write a short Python function that returns the nth Fibonacci number using memoization.",
         required=["def ", "fib", "return"],
         forbidden=["French Revolution", "quantum"]),
    dict(label="dna_short_zh", max_tokens=120,
         prompt="请用一两句话说明 DNA 双螺旋结构由谁提出、在哪一年。",
         required=["沃森", "克里克", "1953", "Watson", "Crick"],
         forbidden=["Fibonacci", "French Revolution", "量子", "爵士"]),
    dict(label="mul_tiny_en", max_tokens=24,
         prompt="What is 17 times 23? Answer with only the number.",
         required=["391"],
         forbidden=[]),
    dict(label="ack_tiny_en", max_tokens=24,
         prompt="Please reply with only the exact phrase: ACK-7X2.",
         required=["ACK-7X2"],
         forbidden=["Fibonacci", "French Revolution", "量子"]),
    dict(label="recipe_short_en", max_tokens=160,
         prompt="List the ingredients and simple steps for scrambled eggs for 2 people.",
         required=["egg", "butter", "salt", "pan", "whisk"],
         forbidden=["Fibonacci", "French Revolution", "quantum", "爵士"]),
    dict(label="ethics_med_en", max_tokens=320,
         prompt="In two short paragraphs, contrast utilitarianism with Kantian deontological ethics using one example.",
         required=["utilitarian", "Kant", "deontolog"],
         forbidden=["Fibonacci", "French Revolution", "integral of x^2"]),
]

# The first PROMPTS entry was built with a kwargs-style `prompt=` but I
# realised I left a trailing-comma issue; the list literal is fine.
# Normalise `required` -> `required_any` as canonical key.
for _p in PROMPTS:
    _p["required_any"] = _p.pop("required", _p.get("required_any", []))
    _p.setdefault("forbidden", [])


def call(url, model, prompt, max_tokens, timeout):
    body = json.dumps(dict(
        model=model, max_tokens=max_tokens, temperature=0, stream=False,
        messages=[{"role": "user", "content": prompt}],
    )).encode("utf-8")
    req = urlreq.Request(url, data=body,
                         headers={"Content-Type": "application/json"},
                         method="POST")
    t0 = time.monotonic()
    try:
        with urlreq.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
            http = r.status
        return dict(ok=True, http=http, body=text,
                    elapsed=time.monotonic() - t0)
    except URLError as e:
        return dict(ok=False, http=None, body="",
                    elapsed=time.monotonic() - t0, err=f"URLError: {e}")
    except Exception as e:
        return dict(ok=False, http=None, body="",
                    elapsed=time.monotonic() - t0, err=f"{type(e).__name__}: {e}")


def extract_content(text):
    try:
        return json.loads(text)["choices"][0]["message"]["content"]
    except Exception:
        return ""


def score(content, task):
    lc = content.lower()
    req = task.get("required_any", [])
    forb = task.get("forbidden", [])
    has_req = any(k.lower() in lc for k in req) if req else True
    has_forb = any(k.lower() in lc for k in forb)
    return has_req, has_forb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="deepseek")
    ap.add_argument("-n", "--n-inflight", type=int, default=2)
    ap.add_argument("--total", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = [rng.choice(PROMPTS) for _ in range(args.total)]

    idx = [0]
    idx_lock = threading.Lock()
    results = []
    results_lock = threading.Lock()
    outf = open(args.out, "w", buffering=1)

    t_start = time.monotonic()

    def worker(wid):
        while True:
            with idx_lock:
                if idx[0] >= args.total:
                    return
                i = idx[0]
                idx[0] += 1
            task = tasks[i]
            r = call(args.url, args.model, task["prompt"],
                     task["max_tokens"], args.timeout)
            content = extract_content(r["body"]) if r["ok"] else ""
            has_req, has_forb = score(content, task)
            verdict = "PASS" if (r["ok"] and has_req and not has_forb) else "FAIL"
            rec = dict(i=i, worker=wid, label=task["label"],
                       prompt_len=len(task["prompt"]),
                       max_tokens=task["max_tokens"],
                       ok=r["ok"], http=r.get("http"),
                       elapsed_s=round(r["elapsed"], 2),
                       resp_len=len(content),
                       has_required=has_req, has_forbidden=has_forb,
                       verdict=verdict, preview=content[:200])
            with results_lock:
                results.append(rec)
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                outf.flush()
            print(f"[{i+1:02d}/{args.total}] w{wid} {task['label']:<18s} "
                  f"{rec['elapsed_s']:>5.1f}s resp={rec['resp_len']:>4d}c "
                  f"req={has_req!s:<5} forb={has_forb!s:<5} {verdict}",
                  flush=True)

    threads = [threading.Thread(target=worker, args=(w,), name=f"w{w}")
               for w in range(args.n_inflight)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall = time.monotonic() - t_start
    n = len(results)
    passes = sum(1 for r in results if r["verdict"] == "PASS")
    http_fail = sum(1 for r in results if not r["ok"])
    fail_missing = sum(1 for r in results if r["ok"] and not r["has_required"])
    fail_contam = sum(1 for r in results if r["ok"] and r["has_forbidden"])

    print("")
    print(f"=== done: total={n} pass={passes} fail={n-passes}  "
          f"wall={wall:.1f}s  throughput={n/wall:.2f} req/s ===")
    print(f"http_fail={http_fail}  missing_required_kw={fail_missing}  "
          f"cross_topic_contamination={fail_contam}")
    outf.close()


if __name__ == "__main__":
    main()
