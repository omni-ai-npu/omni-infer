# OmniCache Configuration Reference

## 1. Core Switches

| Variable | Values | Default | Side | Description |
|----------|--------|---------|------|-------------|
| `ENABLE_OMNI_CACHE` | `0`, `1` | `1` | P / D | Master toggle. `1` enables OmniCacheConnector + hugetlbfs host pool; `0` falls back to vanilla vLLM LLMDataDistConnector (HBM only, no hugepages) |
| `ENABLE_HOST_MAPPING` | `0`, `1` | P: `0`, D: `1` | P / D | Host memory mmap aliasing. When `1`, the DSA indexer stays on HBM while the rest of KV data is read from host via NPU MMU. Prefill does not use this mapping |
| `P_NODE_LIST` | string | P: `192.168.0.148`<br>D: `192.168.0.148` | P / D | Prefill node IP list. Comma-separated within a machine, semicolon-separated across machines. Decode connects to Prefill via this IP over ZMQ/OX |
| `OMNI_CACHE_LOCAL_DP_SIZE` | int | `8` | P / D | Local DP parallelism per machine — the number of NPU dies on a single machine allocated to the current role. Used to calculate the host KV Cache size and block count allocated to each DP rank |

**IP format:**

```bash
# Single machine, multiple instances
P_NODE_LIST="192.168.1.10,192.168.1.11"

# Multi-machine
P_NODE_LIST="192.168.1.10,192.168.1.11;192.168.2.10,192.168.2.11"
```

---

## 2. kv-transfer-config

OmniCache receives its configuration through vLLM's `--kv-transfer-config` CLI parameter as a JSON string that specifies the connector type, role, and network topology.

### Parameter Structure

```json
{
    "kv_connector": "<connector_name>",
    "kv_role": "<role>",
    "kv_rank": <rank>,
    "kv_parallel_size": <parallel_size>,
    "kv_connector_extra_config": {
        "p_node_list": ["<ip>", ...],
        "kv_producer_dp_size": <dp_size>
    }
}
```

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `kv_connector` | string | Connector name. OmniCache: `OmniCacheConnector`; baseline: `LLMDataDistConnector` |
| `kv_role` | string | Node role: `kv_producer` (Prefill) or `kv_consumer` (Decode) |
| `kv_rank` | int | Rank within `kv_parallel_size`. Prefill is always `0`; Decode DP_i is `i + 1` |
| `kv_parallel_size` | int | Total KV parallel group size. OmniCache: `1`; baseline: determined by slices of KV cache in among TP ranks |
| `kv_connector_extra_config` | object | Connector-specific configuration passed into OmniCacheConnector |

### kv_connector_extra_config Fields

| Field | Type | Description |
|-------|------|-------------|
| `p_node_list` | list[string] | Prefill node IP list. The semicolon/comma-delimited string in env var `P_NODE_LIST` is expanded into a flat list. Example: `["192.168.1.10", "192.168.1.11"]` |
| `kv_producer_dp_size` | int | Prefill-side DP count, typically `1` |

### Prefill Example (OmniCache)

```
--kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["192.168.0.148"],"kv_producer_dp_size":1}}'
```

### Decode Example (OmniCache, DECODE_DP_SIZE=8)

```
--kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["127.0.0.1"],"kv_producer_dp_size":1}}'
```

For multiple DPs, increment `kv_rank` per instance (2, 3, ..., 8) while keeping other fields identical.

### Baseline Mode

```
--kv-transfer-config '{"kv_connector":"LLMDataDistConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

---

## 3. Memory & HugePages

### 2.1 HugePage Basics

| Variable | Default | Side | Description |
|----------|---------|------|-------------|
| `OMNI_CACHE_MMAP_FILE` | P: `omni_cache_p`<br>D: `omni_cache_d` | P / D | Memory-mapped filename under `/dev/hugepages/` |
| `OMNI_CACHE_MMAP_PATH` | `/dev/hugepages/${OMNI_CACHE_MMAP_FILE}` | P / D | Full hugetlbfs path. Override if mounted elsewhere |
| `MAP_SIZE_BYTES` | `536870912000` (500 GiB) | P / D | Hugepage file reservation size in bytes. Determines total host KV pool capacity |

### 2.2 Per-Layer HBM Budget

| Variable | Default | Side | Description |
|----------|---------|------|-------------|
| `OMNI_CACHE_LAYER_BYTES` | `17179869184` (16 GiB) | P / D | Per-layer host KV cache buffer budget in bytes. Affects `num_blocks` per die. Reduce on decode for tighter HBM budgets |
| `NUM_GPU_BLOCKS_OVERRIDE` | P: `50000`, D: `11800` | P / D | vLLM scheduler block cap. Must be less than `OMNI_CACHE_LAYER_BYTES / (DP_SIZE_LOCAL * nbytes(kv_cache_block))` |

---

## 4. HBM Layout Mode (Prefill Only)

| Variable | Values | Default | Side | Description |
|----------|--------|---------|------|-------------|
| `OMNI_CACHE_PACKED_HBM` | `0`, `1` | `1` | Prefill | PACKED_HBM layout mode. When `1`, device KV cache blocks on HBM are managed independently from host block IDs. When `0`, the number of blocks on host and device must be strictly equal |

---

## 5. DSA Split Secondary Pool (Decode Only)

In Pangu V2 hybrid, each DSA attention block contains two regions:

```
[ block_size * head_size ]       bf16  kv (kv_lora + k_pe, 576 bytes/token)
[ block_size * indexer_head_dim ] bf16  indexer (128 bytes/token)
```

When DSA Split is enabled, decode reserves an additional secondary hugepage file holding only the DSA KV portion. After each OX pull, the KV segment is asynchronously copied from the primary pool to the secondary pool via `aclrtMemcpyAsync`, allowing the attention kernel to read narrower KV data.

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `ENABLE_OMNI_CACHE_DSA_SPLIT` | `0`, `1` | `0` | Master toggle for the DSA-only secondary pool + post-pull async copy |
| `OMNI_CACHE_DSA_MMAP_FILE` | string | `omni_cache_decode_dsa` | Secondary pool filename under `/dev/hugepages/` |
| `OMNI_CACHE_DSA_MMAP_PATH` | path | `/dev/hugepages/${OMNI_CACHE_DSA_MMAP_FILE}` | Full secondary pool path |
| `OMNI_CACHE_DSA_MAP_SIZE_BYTES` | bytes | `MAP_SIZE_BYTES * 80 / 100` | Secondary pool hugepage reservation size (auto-sized to 80% of the primary pool) |

---

## 6. Networking

| Variable | Default | Side | Description |
|----------|---------|------|-------------|
| `BASE_PORT` | `16077` | P / D | Base port for inter-node communication |
| `ZMQ_BASE_PORT` | `16555` | P / D | Base port for ZMQ transport |

---

## 7. Diagnostics & Debugging

These variables are for development and correctness verification only. They are not needed in production.

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `OMNI_KV_DUMP_GEAR` | `off`, `transfer`, `step` | `off` | KV dump mode. `transfer`: dump at 4 transfer probe points; `step`: dump at each decode step |
| `OMNI_KV_DUMP_BRANCH` | string | empty | Branch label for dump files (e.g. `baseline` / `omnicache`) |
| `OMNI_KV_DUMP_MAX` | int | `0` | Max dump count. `0` = unlimited |
| `OMNI_KV_DUMP_TARGET_REQ` | string | empty | Only dump requests whose ID begins with this prefix |
| `OMNI_MOCK_SCHEDULE` | `0`, `1`, `2` | `0` | Mock scheduler: `0` normal, `1` batch gate + sort, `2` additionally swaps InputBatch rows at step 5 |
| `OMNI_CACHE_VERIFY_TRANSFER` | `0`, `1` | `0` | Verify KV transfer byte integrity |
| `OMNI_CACHE_SKIP_OX_PULL` | `0`, `1` | `0` | Skip OX pull (decode-only test without prefill) |
| `OMNI_CACHE_MOME_DEBUG` | `0`, `1` | `0` | MoME state transfer debug logging |
| `ENABLE_MOCK_P` | `0`, `1` | `0` | Mock prefill mode for testing |
| `OMNI_CACHE_TEST_BLOCK_STATUS` | `0`, `1` | `0` | Test mode: initialize block status mappings |
| `OMNI_CACHE_DEBUG` | `0`, `1` | `0` | Debug logging for module loading and execution details |
