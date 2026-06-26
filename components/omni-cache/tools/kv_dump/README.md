# KV Dump 诊断与验证工具

KV cache 字节一致性比对、logP 分析、HTML 报告生成。

---

## 文件说明

### `kv_dump_compare.py` — 统一比对引擎

KV dump 的统一 CLI 入口，支持多种模式：

```
# 自动检测模式
python tools/kv_dump/kv_dump_compare.py --dump-dir /tmp/kv_dumps

# Transfer 模式（4 阶段探针比对）
python tools/kv_dump/kv_dump_compare.py --mode transfer --dump-dir /tmp/kv_dumps --request-id <id>

# Step 模式（逐步比对）
python tools/kv_dump/kv_dump_compare.py --mode step --dump-dir /tmp/kv_dumps_step --all-requests

# Self-consistency 模式（单分支内 KV 稳定性检查）
python tools/kv_dump/kv_dump_compare.py --mode self-consistency /path/to/dumps/omnicache
python tools/kv_dump/kv_dump_compare.py --mode self-consistency /path/to/dumps/omnicache --include-last-block
```

**模式说明**：

| 模式 | 用途 | 输入 |
|------|------|------|
| `auto` | 自动检测 transfer 或 step 模式 | dump 目录 |
| `transfer` | 对比 P/D 各阶段的 KV 字节一致性 | transfer dump (`.npz`) |
| `step` | 逐步对比 baseline vs omnicache KV | step dump (`.pt`) |
| `self-consistency` | 单分支内并发请求 KV 稳定性 | 单分支 step dump |

### `kv_dump_analyze.py` — HTML 报告生成

```
# KV 比对报告（默认）
python tools/kv_dump/kv_dump_analyze.py --dump-dir /tmp/kv_dumps_cmp --output /tmp/report.html

# logP 失配分析
python tools/kv_dump/kv_dump_analyze.py --mode logp \
    --baseline-dir /tmp/kv_responses/baseline \
    --omnicache-dir /tmp/kv_responses/omnicache \
    --output /tmp/logp_report.html
```

### `run_kv_verification.py` — 一键编排

```
# 发送测试请求
python tools/kv_dump/run_kv_verification.py send -n 8 --max-tokens 20

# 比对结果
python tools/kv_dump/run_kv_verification.py compare --id-prefix humaneval

# 完整流程（发送 + 比对 + 报告）
python tools/kv_dump/run_kv_verification.py send -n 8 --max-tokens 20 --ignore-eos
python tools/kv_dump/run_kv_verification.py compare --id-prefix humaneval --analyze
```

## 依赖

```
pip install numpy torch
```
