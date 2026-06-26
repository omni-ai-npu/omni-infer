# Diagnostics 诊断工具

KV cache 一致性与精度诊断工具集，用于 PD 分离模式下的 KV 数据传输验证与调试。

---

## 模块概览

| 模块 | 用途 |
|------|------|
| `probes.py` | KV 探针，在 D2H/H2D 各阶段采集 KV tensor 快照 |
| `kv_step_dumper.py` | 逐步 KV dump，支持 transfer 模式和 step 模式 |
| `dump_controller.py` | KV dump 的安装入口与环境变量 gate |
| `config.py` | 诊断配置解析（gear、branch、max_steps） |
| `normalizer.py` | KV tensor 标准化，用于跨阶段/跨进程比对 |
| `snapshot.py` | KV tensor 快照的哈希计算与存储 |
| `mock_schedule.py` | Mock 调度器，批次排序确定性化（`OMNI_MOCK_SCHEDULE=1`） |
| `input_swap.py` | 输入批次行交换（`OMNI_MOCK_SCHEDULE=2`） |

---

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OMNI_KV_DUMP_GEAR` | `off` | KV dump 模式：`transfer`（传输探针）、`step`（逐步 dump） |
| `OMNI_KV_DUMP_DIR` | `/tmp/kv_dumps` | Dump 文件输出目录 |
| `OMNI_KV_DUMP_MAX` | `0` | 最大 dump 步数（0=无限制） |
| `OMNI_KV_DUMP_TARGET_REQ` | 不设置 | 仅 dump 指定 request ID |
| `OMNI_KV_DUMP_PRE_STEP_MODE` | `sliced` | Pre-step dump 模式：`full` 或 `sliced` |
| `OMNI_MOCK_SCHEDULE` | `0` | `1`=确定性批次排序，`2`=输入批次交换 |

---

## 使用方式

### Transfer 模式（传输探针）

在 D2H/H2D 各阶段自动采集 KV 数据，用于跨阶段字节一致性验证。

```bash
# Launch with transfer probes
OMNI_KV_DUMP_GEAR=transfer OMNI_MOCK_SCHEDULE=1 bash launch_pd.sh

# Send request and compare
python tools/kv_dump/kv_dump_compare.py --mode transfer --request-id <id>
```

### Step 模式（逐步 dump）

每个 decode step 自动 dump KV 数据，用于逐步 logP 一致性分析。

```bash
OMNI_KV_DUMP_GEAR=step OMNI_MOCK_SCHEDULE=1 bash launch_pd.sh
```

### Mock 调度

`OMNI_MOCK_SCHEDULE=1` 使 Scheduler 按 request_id 排序批次，确保跨运行的批次组成一致，消除时序噪声。

---

## 设计原则

- **Probe 与业务代码松耦合**：probe 函数通过函数参数接收数据，不修改业务逻辑
- **环境变量 gate**：所有诊断功能通过环境变量控制，默认关闭
- **子进程安全**：probe 支持从 multiprocessing 子进程调用（如 address subprocess）
