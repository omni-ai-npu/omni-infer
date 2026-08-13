# KV Dump 使用文档

本文档说明 `omni_npu.connector.kv_dump` 模块的使用方法。该模块在 vLLM NPU 推理过程中对 KV cache 做 MD5 哈希，有两大用途：

1. **KV cache 传输校验**：在 producer/consumer 模式下验证 KV cache 跨节点传输的正确性。
2. **KV cache 踩踏检测**：在 decode 阶段每一步检测 KV cache 是否被意外覆写（踩踏），一旦发现立即抛异常。

> ⚠️ **重要提示：当前暂不支持 MTP（Multi-Token Prediction）。**
>
> `maybe_dump_kv` 的钩子挂在 `execute_model` 上，而 MTP 的 drafter 层在 `execute_model` 返回后的 **sampling 阶段**才执行，钩子无法覆盖 MTP 层。开启 MTP 时工具仍可运行，但 MTP 层的 KV dump 会显示不匹配。后续计划扩展钩子至 sampling 阶段以完整支持 MTP。详见[第 8 节](#8-mtp-支持说明)。

## 1. 功能概述

- **Prefill 阶段**：对所有新分配的 KV cache 块做哈希，记录完整快照。
- **Decode 阶段**：每步增量校验已计算块与历史一致（检测 KV cache 踩踏），并对新增请求 dump 初始哈希。若发现块内容被意外覆写，通过 `assert torch.equal` 直接抛出异常。
- **输出**：按 rank 分别输出 JSON 文件，key 为 `req_id`，value 为嵌套的 block hash 数据。

## 2. 启用方式

通过环境变量 `KV_DUMP_PATH` 控制，不设置或为空字符串则不启用。

```bash
export KV_DUMP_PATH=/path/to/output/
```

## 3. 代码接入

在 `GPUModelRunner.execute_model` 入口处包裹：

```python
from omni_npu.connector.kv_dump import maybe_dump_kv

# 原函数
def execute_model(self, scheduler_output, ...):
    ...

# 包裹后
execute_model = maybe_dump_kv(execute_model)
```

`maybe_dump_kv` 是一个装饰器工厂：若 `KV_DUMP_PATH` 为空则原样返回函数；否则返回 wrapper，在每次 `execute_model` 调用前后执行 hash 与 dump。

## 4. 输出说明

- **目录结构**：`{KV_DUMP_PATH}/{YYYYMMDD-HHMM}/`
- **文件命名**：`rank0.json`、`rank1.json` …（按 `get_world_group().rank`）
- **写入方式**：后台线程每秒轮询队列，以流式 JSON 追加写入，避免全量加载。

## 5. 生产者 / 消费者角色

模块根据 `kv_transfer_config.kv_role` 区分行为：

- **`kv_producer`（生产者）**：真正执行 dump，将 KV cache 哈希写入 JSON。
- **`kv_consumer`（消费者）**：不输出文件，但执行两项校验：
  - `_check_blks`：将已满块的 KV cache 与上一轮保存的 `hashed_batch` 逐块对比，`assert torch.equal(a, b)`。
  - `_check_toks`：将未满块中已计算的 token 与 `hist_hash` 中记录的历史对比。
  
  两者任一不一致即抛出 `AssertionError`，从而捕获 KV cache 踩踏问题。

## 6. 关键日志

运行时可在日志中看到耗时信息：

```
[Xms] hash blocks
[Xms] check_blks ok: blocks=N
[Xms] check_toks ok: blocks=N
[Xms] save json: /path/to/rank0.json
```

## 7. 性能影响

> ⚠️ **此工具会显著降低推理性能，仅限调试场景使用，不可常态部署。**

开启后性能劣化情况：

| 场景 | 性能劣化 |
|------|---------|
| KV cache 传输（producer dump） | 约 **3 倍** |
| Decode 逐 step 校验 | 约 **5 倍** |

劣化主要来源于：每步对全部活跃块的哈希计算、CPU/GPU 间数据搬运（`.contiguous().cpu()`）、以及后台 JSON 序列化写入。

## 8. MTP 支持说明

> ⚠️ **当前版本暂不支持 MTP（Multi-Token Prediction）。**

### 原因

`maybe_dump_kv` 的钩子挂载在 `NPUModelRunner.execute_model` 上，其上下文管理器在模型前向计算前后捕获 KV cache 并做哈希。而 MTP（多 token 预测）的 drafter 层是在 `execute_model` 返回之后的 **sampling（采样）阶段**才执行的，此时钩子已经退出，无法覆盖 MTP 层产生的 KV cache。

```
execute_model（钩子覆盖范围）
  └── 主模型 forward → KV cache 被 hash/dump
        ↓ execute_model 返回，钩子退出
sampling 阶段（钩子未覆盖）
  └── MTP drafter 层 forward → MTP 层 KV cache 未被 hash/dump
```

### 当前行为

模块在开启 MTP 时**仍然可以运行**，不会报错。但由于 MTP 层的 KV cache 未被钩子捕获，dump 结果中 MTP 对应层的 hash 会显示为**不匹配**。

### 后续计划

后续会将钩子扩展至 sampling 阶段，以完整覆盖 MTP 层的 KV cache，届时将正式支持 MTP 场景。

## 9. 其它注意事项

- 依赖 vLLM 内部 API（`GPUModelRunner`、`SchedulerOutput`、`extract_layer_index` 等），升级 vLLM 版本时需关注兼容性。
