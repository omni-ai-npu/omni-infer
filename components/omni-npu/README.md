# omni-npu (vLLM NPU Plugin)

A vLLM (0.14.0) out-of-tree platform plugin that enables running vLLM on NPU (Ascend/torch_npu).

- Loaded via vLLM plugin entry points (no code changes to vLLM required).
- Provides a minimal NPU Platform, Worker, and a standalone NPU ModelRunner adapter.
- Uses vLLM's existing serving APIs unchanged.

## Requirements

- Python >= 3.12
- vLLM == 0.14.0
- torch and torch_npu for your platform (vendor-specific install)

## Install (order matters)

```bash
# 1) Install vLLM first (pin to 0.14.0 for compatibility)
pip install vllm==0.14.0

# 2) Install vendor runtime (example: torch_npu for Ascend)
# Follow your vendor instructions; example:
# pip install torch==<compatible> torch_npu==<compatible>

# 3) Install omni-npu plugin (this project)
pip install .
# or
pip install -e .
```

## How it works

- The plugin registers under the vLLM entry point group `vllm.platform_plugins`.
- vLLM discovers `omni_npu.platform.NPUPlatform` when `torch_npu` is available.
- The platform sets `device_type=npu`, configures worker class, and uses HCCL.
- The NPU worker constructs a standalone `NPUModelRunner` that adapts vLLM's model runner logic to NPU without subclassing GPUModelRunner.

## Usage (serve via vLLM API)

- OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model your/model \
  --device npu \
  --port 8000 \
  --trust-remote-code
```

- Python API:

```python
from vllm import LLM
llm = LLM(model="your/model", device="npu")
print(llm.generate(["hello world"]))
```

Notes:
- Keep using vLLM’s parameters; only change `--device npu`.
- Ensure `torch_npu` is installed and NPUs are visible (`ASCEND_RT_VISIBLE_DEVICES`).

## Troubleshooting

- Plugin not detected: ensure `pip show vllm omni-npu` lists both and `torch_npu` is importable.
- Distributed backend: HCCL is used; configure env (MASTER_ADDR/PORT) per your cluster.
- Memory issues: adjust `--gpu-memory-utilization` or `--kv-cache-memory`.

## LLMDataDistConnector Communication Matrix (IP / Ports)

### Endpoint matrix

| # | Protocol | Address (IP + port / path) | Port | Trigger |
| --- | --- | --- | --- | --- |
| 1 | ZMQ `PUSH`/`PULL` / TCP | `tcp://<prefill_ip>:<port + prefill_dp>` | default `5568`; override via env `VLLM_LLMDATADIST_ZMQ_PORT` or `kv_transfer_config.kv_port` | Decode KV pulled → `["remote_request_id"]`; heartbeat every 5 s → `["decode_hb:<cluster_id>"]`; `request.status == FINISHED_ABORTED` |
| 2 | ZMQ `PUB` / TCP | `tcp://<prefill_ip>:port` | default `LLMDATADIST_BASE_PORT - 1` = `15566`; override via env `VLLM_LLMDATADIST_HEARTBEAT_PORT` | Prefill every 5 s sends `prefill_hb:<host_cluster_id>` |
| 3 | ZMQ `SUB` / TCP | `tcp://<prefill_ip>:<port + prefill_dp>` | row 2 port + prefill `dp_rank` | Decode `heartbeat_timer_func` polls |
| 4 | ZMQ `PUSH`/`PULL` / IPC | `ipc://<HEARTBEAT_IPC_PATH>_<rank>` | `HEARTBEAT_IPC_PATH` (= `ipc:///tmp/prefill_llmdatadist_connector_ipc`) + `_<rank>` | Prefill rank 0 detects remote heartbeat timeout → other ranks receive `force_unlink <cluster_id>` |
| 5 | ZMQ `PUB` / IPC | `ipc:///tmp/sched-pub-<kv_rank>-<dp_rank_local>` | hardcoded path | `async_pull_kv=True`; `build_connector_metadata` with `scheduler_output is None` and non-empty `metadata.requests`; payload = `pickle.dumps(metadata)` |
| 6 | `llm_datadist` RoCE / TCP | `<prefill_ip>:<port + prefill_local_rank>` | `LLMDATADIST_BASE_PORT` (default `15567`) + `local_rank` | Decode first `pull_kv` → `register_link` (`link_clusters`) → `pull_blocks`; prefill listens via `listen_ip_info` |

### Notes

- Local IP is discovered by default route of the OS system.

## License

MIT
