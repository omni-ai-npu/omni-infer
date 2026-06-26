# Pangu V2 hybrid — PD launcher scripts

Reference launch scripts for running a prefill / decode disaggregated
deployment of a Pangu V2 hybrid attention model with the OmniCache
host-backed KV transfer.

## Layout

| File | Role |
|------|------|
| `launch_prefill.sh` | Prefill server (TP=8 on dies 0–7 by default). Launches `vllm serve` with `kv_role=kv_producer` plus the Pangu V2 hybrid patches, and reserves the prefill-side hugetlbfs file. |
| `launch_decode.sh` | Decode servers: `DECODE_DP_SIZE` (default 8) parallel `vllm serve` processes with `kv_role=kv_consumer`, one per die 8–15. Reserves the decode-side hugetlbfs file. |

There is **no intermediate wrapper script** — both launchers construct
the full `vllm serve` command inline (each with ~20 CLI flags) and exec
it in the foreground. If you used to invoke `serve-single-instance-*.sh`
or `serve-pd-disaggregate-*.sh`, those have been replaced by these two
self-contained launchers.

The launchers also **do not start a reverse proxy**. In the reference
test harness an external nginx instance (container `yyx-container-c0`,
port `7077`) routes all traffic to decode DP3 (port `8085`) as a
single-DP stress test. If you want real load-balancing across DPs,
configure nginx yourself or hit the individual DP ports
(`PORT_BASE..PORT_BASE+DECODE_DP_SIZE-1`, default `8082..8089`).

## Concurrency status

The prior concurrent-KV corruption (fixed in `908cb94`) is guarded by a
reproducible N-in-flight benchmark — see
[`../../docs/NINFLIGHT_BENCH_REPORT.md`](../../docs/NINFLIGHT_BENCH_REPORT.md)
for the full report. Summary on `dev_support_pangu`:

| N in-flight | Total reqs | PASS | Cross-topic contamination | Throughput |
|---|---|---|---|---|
| 2 | 20 | 20/20 | 0 | 0.06 req/s |
| 4 | 24 | 24/24 | 0 | 0.17 req/s |
| 8 | 40 | 40/40 | 0 | 0.24 req/s |
| 12 | 40 | 40/40 | 0 | 0.34 req/s |
| 16 | 40 | 40/40 | 0 | 0.38 req/s |

N=16 hits the `max_num_seqs=16` ceiling of a single decode DP. Reproduce
the bench with:

```bash
python3 tools/benchmark/bench_nflight.py \
    --url http://0.0.0.0:7077/v1/chat/completions \
    -n 16 --total 40 --timeout 800 \
    --out /tmp/bench_n16.jsonl
```

Additional fixes landed with the benchmark campaign:

- `fea5e74` — `VLLM_LOGGING_LEVEL` default `DEBUG → INFO`. DEBUG forces
  tensor-repr via `logger.debug(f"...{retval=}")`, which dispatches
  `aclnnMaskedSelect` on the NPU every decode step and trips aivec
  errors at N≥4.
- `a323545` — `LOCAL_DP_SIZE: 16 → 8`. Each role instance owns 8
  dies in this PD deployment, not 16, so the per-die num_blocks was
  half of what it should be.
- `d808f4d` — `--num-gpu-blocks-override 1400` on the decode `vllm
  serve` command. vLLM's scheduler was allocating block ids beyond
  the DSA HBM pool size; capping the scheduler keeps them in range.

## Prerequisites

- Hugetlbfs mounted at `/dev/hugepages` with enough 2 MiB pages to back
  the mmap file (`MAP_SIZE_BYTES / 2 MiB`). Both launchers invoke
  `tools/setup/setup_hugetlbfs_2MB.sh` to reserve pages and create
  the file. Default `MAP_SIZE_BYTES=500 GiB` needs 256 000 pages.
- Ascend toolkit sourced in the invoking shell (`/root/.bashrc` on the
  reference image already does this). Run via `bash -lc` if launching
  from a non-login shell.
- Model weights present at `$MODEL_PATH`.
- OX binary present at `omni_cache/connector/backends/ox/ox` (sometimes
  gitignored — restore from a prior artifact if missing).

## Usage

```bash
# Defaults: prefill on port 8000 (dies 0–7), decode on ports 8082..8089
# (dies 8–15), HM=1.
bash examples/pangu_v2_pd/launch_prefill.sh
bash examples/pangu_v2_pd/launch_decode.sh
```

Every knob is an environment-variable override:

```bash
# Different model + smaller batch
MODEL_PATH=/data/models/my_model BSZ=8 bash launch_prefill.sh

# HOST_MAPPING=0 decode (no host-mmap aliasing, KV stays in HBM)
ENABLE_HOST_MAPPING=0 bash launch_decode.sh

# Smaller host pool (100 GiB instead of 500 GiB)
MAP_SIZE_BYTES=107374182400 bash launch_prefill.sh

# Change the vLLM scheduler block cap (must stay < DSA HBM pool size)
NUM_GPU_BLOCKS_OVERRIDE=1200 bash launch_decode.sh

# Lower per-layer HBM footprint on decode (e.g., for tighter budgets)
OMNI_CACHE_LAYER_BYTES=536870912 bash launch_decode.sh  # 512 MiB
```

See the top of each launcher for the complete variable list.

## Common overrides

| Variable | Default | Side | Meaning |
|---|---|---|---|
| `MODEL_PATH` | `/data/models/92B_DSA_iter_0000180/` | both | Model weight directory |
| `MAX_LEN` | `8192` | both | `--max-model-len` / `--max-num-batched-tokens` |
| `BSZ` | `32` | both | `--max-num-seqs` (per-DP concurrent requests) |
| `PORT` | `8000` | prefill | Prefill HTTP port |
| `PORT_BASE` | `8082` | decode | First decode DP port (DP_i at `PORT_BASE + i`) |
| `TP_SIZE` | `8` | prefill | Prefill tensor parallel |
| `DP_SIZE` | `1` | prefill | Prefill data parallel |
| `DECODE_TP_SIZE` | `1` | decode | Tensor parallel per decode DP |
| `DECODE_DP_SIZE` | `8` | decode | Number of decode DP instances |
| `DEVICE_START` | `0` / `8` | prefill / decode | First NPU die index |
| `ENABLE_OMNI_CACHE` | `1` | both | `1`: `OmniCacheConnector` (host-backed KV + hugetlbfs). `0`: vLLM's `LLMDataDistConnector` (no hugetlbfs). |
| `ENABLE_HOST_MAPPING` | `1` | decode | `1`: DSA indexer on HBM, rest read from host via NPU MMU. `0`: full HBM kv_cache. |
| `OMNI_CACHE_LAYER_BYTES` | `4294967296` (4 GiB) | decode | Per-layer HBM buffer budget; determines `num_blocks` per die |
| `MAP_SIZE_BYTES` | `536870912000` (500 GiB) | both | Hugetlbfs file size |
| `NUM_GPU_BLOCKS_OVERRIDE` | `1400` | decode | `--num-gpu-blocks-override`. Caps vLLM scheduler below the DSA HBM pool size (default ≈ 1489 under default `OMNI_CACHE_LAYER_BYTES`). |
| `VLLM_LOGGING_LEVEL` | `INFO` | both | Keep at `INFO` under concurrency; `DEBUG` triggers aivec errors via tensor-repr. |

## DSA-split secondary host pool (optional)

Pangu V2 hybrid lays DSA attention KV out per block as two sectioned
sub-regions:

    [ block_size * head_size       ] bf16  kv  (kv_lora + k_pe, 576 bytes/token)
    [ block_size * indexer_head_dim ] bf16  indexer (128 bytes/token)

Primary host pool stores both (704 bytes/token including SWA alignment),
MMU-aliased into HBM. When you turn on `ENABLE_OMNI_CACHE_DSA_SPLIT=1`,
decode additionally reserves a **secondary** hugetlbfs file sized only
for the kv part of DSA blocks. After every OX pull, decode copies the
kv section of each pulled DSA block from the primary to the secondary
pool via `aclrtMemcpyAsync(kind=D2D)` on a dedicated ACL stream. The
attention kernel then reads kv from the secondary alias and the indexer
from its HBM buffer — no indexer padding on the kv read path.

Why this is useful:

- Narrower per-slot read on the DSA attention kernel (576 B vs. 704 B);
  HBM-aligned with the model's natural layout.
- Copy is async on a private stream, overlaps with other NPU work.
- No change to the OX pull path — primary still receives prefill blocks
  exactly as before.

Enable / tune:

```bash
# turn on
ENABLE_OMNI_CACHE_DSA_SPLIT=1 bash launch_decode.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `ENABLE_OMNI_CACHE_DSA_SPLIT` | `0` | Master toggle for the DSA-only secondary pool + post-pull copy. |
| `OMNI_CACHE_DSA_MMAP_FILE` | `omni_cache_decode_dsa` | Filename under `/dev/hugepages` for the secondary pool. |
| `OMNI_CACHE_DSA_MMAP_PATH` | `/dev/hugepages/${OMNI_CACHE_DSA_MMAP_FILE}` | Full hugetlbfs path. Override if you mount elsewhere. |
| `OMNI_CACHE_DSA_MAP_SIZE_BYTES` | `MAP_SIZE_BYTES * 80 / 100` | Hugetlbfs reservation for the secondary file (auto-sized relative to the primary pool). |

The head size itself is not user-configurable — it is derived from
the model's `hf_config` (`kv_lora_rank + qk_rope_head_dim`, e.g. 576
for Pangu V2 DSA). When the feature is off, behaviour is identical to
before — the secondary file is neither reserved nor opened.

## Other OmniCache tuning

| Variable | Default | Side | Meaning |
|---|---|---|---|
| `OMNI_REUSE_PREFILLED_TOKENS` | `0` | decode | `1` re-uses the last prefilled token on decode step 0 (saves one forward) — off by default because MoMe D2H timing needed `post_attn` first; now safe either way. |
| `OMNI_SKIP_DECODE_TOKENIZE` | `0` | decode | `1` skips server-side detokenize in the hotpath — off by default for compatibility. |
| `DUMP_ATTN_META` / `DUMP_KV_CACHE` | `0` | both | Debug: dump attention metadata / kv cache tensors to disk. Verbose, only for diagnosis. |

## Logs

- Prefill server log → `examples/pangu_v2_pd/logs/prefill/serving.log`
- Decode DP logs → `examples/pangu_v2_pd/logs/decode/decode_{0..DECODE_DP_SIZE-1}.log`
- Launcher stdout (hugepages reservation, rank boot ordering) → the
  `launch_p.log` / `launch_d.log` alongside the per-DP logs.

Override `LOG_DIR` to redirect.
