# OmniCache Diagnostics Tools

Tools for debugging KV cache correctness in the PD disaggregation pipeline.

## Deterministic Request IDs

To compare the same request across branches (e.g. baseline vs omnicache) under
concurrent load, assign deterministic request IDs from the client side using the
`X-Request-Id` HTTP header:

```bash
curl -H "X-Request-Id: test-req-0001" \
     http://localhost:7077/v1/chat/completions \
     -d '{"model":"deepseek","messages":[...]}'
```

The nginx router's `ngx_http_set_request_id_module.so` forwards this header to
vLLM, which uses it as the `chatcmpl-<id>` request identifier.  The ID then
appears in KV dump filenames, enabling pairwise comparison across branches.

> **Note:** The `request_id` field in the JSON body does *not* propagate through
> the nginx router — you must use the HTTP header.

## Tools

### `send_concurrent.py` — concurrent request sender

Sends N concurrent identical requests with deterministic IDs (`test-req-0000`,
`test-req-0001`, …).

```bash
# 10 requests, 50 tokens (defaults)
python tools/scripts/send_concurrent.py

# 5 requests, 100 tokens, custom endpoint
python tools/scripts/send_concurrent.py -n 5 --max-tokens 100 \
    --url http://10.0.0.1:7077/v1/chat/completions

# custom ID prefix for a specific test run
python tools/scripts/send_concurrent.py -n 20 --id-prefix run42
```

### `kv_dump_compare.py` — cross-branch KV comparison

Compares KV cache tensors between two branches (baseline vs omnicache) to detect
transfer corruption or divergence.

**Step mode** (per-decode-step `.pt` dumps):

```bash
# auto-detect mode
python tools/kv_dump/kv_dump_compare.py --dump-dir /path/to/dumps

# compare ALL requests with matching deterministic IDs
python tools/kv_dump/kv_dump_compare.py --mode step --dump-dir /path/to/dumps --all-requests

# compare a single request pair
python tools/kv_dump/kv_dump_compare.py --mode step \
    --baseline-id chatcmpl-test-req-0001 \
    --omni-id chatcmpl-test-req-0001
```

**Transfer mode** (4-stage probe data — prefill_hbm → prefill_host → decode_host
→ decode_hbm):

```bash
python tools/kv_dump/kv_dump_compare.py --mode transfer --request-id chatcmpl-XXXX
```

# default: exclude last block (it legitimately changes each step)

### `kv_dump_compare.py` — unified KV comparison (transfer, step, self-consistency)

Offline checker that compares KV tensors at each hop of the transfer pipeline
(prefill HBM → prefill host → decode host → decode HBM).

```bash
python tools/kv_dump/kv_dump_compare.py --mode transfer --dump-dir /tmp/kv_dumps \
    --request-id chatcmpl-XXXX \
    --compare-pairs prefill_hbm:prefill_host prefill_host:decode_host \
                    decode_host:decode_hbm prefill_hbm:decode_hbm
```

## Typical Workflow

1. Start PD servers (prefill + decode + router)
2. Send a warmup request (first request initializes caches — skip it)
3. Enable KV dump env vars and restart if needed
4. Send concurrent requests with deterministic IDs:
   ```bash
   python tools/scripts/send_concurrent.py -n 5 --id-prefix test-req
   ```
5. Copy dumps to persistent storage (they may be in `/dev/shm`)
6. Run self-consistency check per branch:
   ```bash
   python tools/kv_dump/kv_dump_compare.py --mode self-consistency /path/to/dumps/omnicache
   python tools/kv_dump/kv_dump_compare.py --mode self-consistency /path/to/dumps/baseline
   ```
7. Run cross-branch comparison:
   ```bash
   python tools/kv_dump/kv_dump_compare.py --dump-dir /path/to/dumps --all-requests
   ```
