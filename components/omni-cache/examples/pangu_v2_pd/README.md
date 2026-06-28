# Pangu V2 Hybrid — 1P-1D Disaggregated Serving

1 Prefill + 1 Decode deployment on a single Huawei NPU A3 node (16 cards),
with an omni-proxy for APC-aware request scheduling.

## Topology

```
Single node (16 NPU cards):
  P0     container omnicache_pangu_p0  cards 0-7   TP=8  port=8000
  D0     container omnicache_pangu_d0  cards 8-15  DP=8  TP=1  ports=8082-8089
  Proxy  runs in omnicache_pangu_p0                port=7150
```

## Quick Start

Two steps: create containers first, then launch services.

### Step 1: Create containers

```bash
DOCKER_IMAGE_ID=<image> \
  LAUNCH_MODE=all \
  bash examples/pangu_v2_pd/launch_containers.sh
```

### Step 2: Launch services

```bash
# Launch prefill + decode + proxy
LAUNCH_MODE=all \
  bash examples/pangu_v2_pd/launch_pd.sh

# With config profile
CONFIG_PROFILE=high-throughput \
  LAUNCH_MODE=all \
  bash examples/pangu_v2_pd/launch_pd.sh
```

## Launch modes

`launch_pd.sh` supports the following `LAUNCH_MODE` values:

| Mode | Starts | Use case |
|------|--------|----------|
| `both` (default) | prefill + decode | Basic PD without proxy |
| `all` | prefill + decode + proxy | Full deployment with proxy |
| `prefill` | prefill only | Debug prefill independently |
| `decode` | decode only | Debug decode independently |
| `proxy` | proxy only | Restart proxy without touching P/D |

## Proxy configuration

| Variable | Default | Meaning |
|---|---|---|
| `PROXY_LISTEN_PORT` | `7150` | Proxy HTTP listen port |
| `PROXY_CORE_NUM` | `4` | Number of CPU cores for proxy |
| `PROXY_START_CORE_INDEX` | `16` | Starting CPU core index |
| `PROXY_LOG_LEVEL` | `info` | Log verbosity |
| `PROXY_PD_POLICY` | `sequential` | PD scheduling policy |
| `PROXY_MAX_BATCH_NUM_TOKEN` | `512000` | Max batch num tokens |
| `PROXY_PREFILL_MAX_NUM_SEQS` | `2` | Max concurrent prefill sequences |
| `PROXY_DECODE_MAX_NUM_SEQS` | `1024` | Max concurrent decode sequences |
| `PROXY_PREFILL_STARVATION_TIMEOUT` | `400` | Prefill starvation timeout (ms) |
| `PROXY_SCHEDULE_ALGO` | `default` | Scheduling algorithm |
| `PROXY_STREAM_OPS` | `off` | Stream operations toggle |
| `PROXY_PREFILL_POD_SIZE` | `1` | Prefill pod size |
| `PROXY_DECODE_POD_SIZE` | `1` | Decode pod size |
| `REBUILD` | `0` | Force rebuild omni-proxy (`1` to force) |

## Common overrides

| Variable | Default | Side | Meaning |
|---|---|---|---|
| `MODEL_PATH` | `/data/models/iter_0011840-W8A8-0509-skip_shared_experts/` | both | Model weight directory |
| `SERVED_MODEL_NAME` | `pangu_ultra_moe` | both | Served model name |
| `MAX_LEN` | `8192` | both | `--max-model-len` / `--max-num-batched-tokens` |
| `BSZ` | `8` | both | `--max-num-seqs` (per-DP concurrent requests) |
| `PORT` | `8000` | prefill | Prefill HTTP port |
| `PORT_BASE` | `8082` | decode | First decode DP port (DP_i at `PORT_BASE + i`) |
| `TP_SIZE` | `8` | prefill | Prefill tensor parallel |
| `DP_SIZE` | `1` | prefill | Prefill data parallel |
| `DECODE_TP_SIZE` | `1` | decode | Tensor parallel per decode DP |
| `DECODE_DP_SIZE` | `8` | decode | Number of decode DP instances |
| `DEVICE_START` | `0` / `8` | prefill / decode | First NPU die index |
| `ENABLE_OMNI_CACHE` | `1` | both | `1`: `OmniCacheConnector` (host-backed KV + hugetlbfs). `0`: vLLM's `LLMDataDistConnector` (no hugetlbfs). |
| `ENABLE_HOST_MAPPING` | `1` | decode | `1`: DSA indexer on HBM, rest read from host via NPU MMU. `0`: full HBM kv_cache. |
| `OMNI_CACHE_LAYER_BYTES` | `17179869184` (16 GiB) | decode | Per-layer HBM buffer budget; determines `num_blocks` per die |
| `MAP_SIZE_BYTES` | `536870912000` (500 GiB) | both | Hugetlbfs file size |
| `NUM_GPU_BLOCKS_OVERRIDE` | `1400` | decode | `--num-gpu-blocks-override`. Caps vLLM scheduler below the DSA HBM pool size. |
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

Enable / tune:

```bash
ENABLE_OMNI_CACHE_DSA_SPLIT=1 bash launch_decode.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `ENABLE_OMNI_CACHE_DSA_SPLIT` | `0` | Master toggle for the DSA-only secondary pool + post-pull copy. |
| `OMNI_CACHE_DSA_MMAP_FILE` | `omni_cache_decode_dsa` | Filename under `/dev/hugepages` for the secondary pool. |
| `OMNI_CACHE_DSA_MMAP_PATH` | `/dev/hugepages/${OMNI_CACHE_DSA_MMAP_FILE}` | Full hugetlbfs path. |
| `OMNI_CACHE_DSA_MAP_SIZE_BYTES` | `MAP_SIZE_BYTES * 80 / 100` | Hugetlbfs reservation for the secondary file. |

## Logs

All logs are under `examples/pangu_v2_pd/logs/`:

```
logs/
  prefill/        # P0
    serving.log
    prefill_launch.log
  decode/         # D0
    decode_{0..7}.log
    decode_launch.log
  proxy/          # omni-proxy
    nginx_error.log
    nginx_access.log
    launch_config.log
    proxy_cmd.log
    proxy_launch.log
```