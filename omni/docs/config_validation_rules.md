# Omni-NPU ValidationRule 开发规范

本文约定如何新增、实现、注册和测试 `ValidationRule`。它是启动校验框架的
独立开发规范，不定义环境变量、`additional_config` 或
`ModelExtraConfig` 配置项本身；配置项开发参见
[配置开发规范](config_dev_guide.md)。

源码入口：

- [`validators.py`](../src/omni_npu/configs/validators.py)：
  规则类型、注册表和执行器。
- [`npu_worker.py`](../src/omni_npu/worker/npu_worker.py)：启动校验调用点。
- [`test_validators.py`](../tests/config/test_validators.py)：框架和内置规则
  测试。

## 1. 定位与执行时机

`ValidationRule` 对完整、已解析的启动状态执行只读声明式校验。它用于补充
而不是替代字段 owner、vLLM 和 loader 已有的校验。

当前 NPU worker 内的顺序是：

```text
distributed initialization
  -> load_model_extra_config()
  -> emit_config_summary()
  -> validate_all()
  -> remaining worker initialization
```

因此：

- 规则看到的是 loader 处理后的 `model_extra_config` 和已解析的
  `vllm_config`。
- 校验在每个 NPU worker 中执行，不是整个集群只执行一次；warning 可能在
  多个 worker 中重复。
- 配置摘要先于校验输出，启动失败时可结合 OMNI-CONF 快照排查。
- 环境变量应通过 `omni_npu.envs` 懒读取。
- `additional_config` 不在 `ValidationContext` 中重复保存。需要时调用
  `OmniAdditionalConfig.from_vllm_config(ctx.vllm_config)` 现场解析。

## 2. 什么时候应该添加规则

仅在以下至少一种情况成立时添加 `ValidationRule`：

- 约束依赖多个来源，例如环境变量与 `VllmConfig` 必须一致。
- 约束依赖 loader 处理后的 `ModelExtraConfig`。
- 只有完整的 `VllmConfig` 解析完成后才能确认当前模式和最终值。
- vLLM 的字段本身合法，但在 Omni-NPU 的特定组合中会失效、被忽略或产生
  明确风险。

以下情况不要添加规则：

- 单字段类型、格式、范围或默认值校验。
- 单个配置类在构造时即可判断的内部约束。
- vLLM 或字段 owner 已有等价校验。
- 需要修改、补全、裁剪或覆盖配置值。
- 只基于猜测、尚不能证明成立的限制。

单字段校验放在所属 parser 或 dataclass，单个 `ModelExtraConfig` 子对象的
内部约束放在所属 dataclass 或 loader；需要修改配置的逻辑放在 loader 或
具体 feature。

## 3. 核心对象的职责

| 对象 | 职责 |
| --- | --- |
| `ValidationContext` | 只提供最终的 `model_extra_config` 和 `vllm_config` |
| `ValidationRule` | `name`、`description`、`severity` 和 `check` 的唯一来源 |
| `Violation` | 保存 `field_path`、`message` 和可选 `original_value` |
| `Severity.REJECT` | 聚合后抛出 `ValueError`，阻止启动 |
| `Severity.WARN` | 输出醒目的 warning 摘要，保留原值并继续启动 |

框架没有 `FIX` 语义。validator 不能修改配置，也不能在 warning 路径中进行
隐式修复。

当前一个 `check()` 最多返回一个 `Violation`。如果一次检查发现多个同类
值不符合要求，应将相关值聚合到同一条 violation 中；不同语义的约束应拆分
成多个规则。

## 4. 选择 REJECT 或 WARN

| severity | 使用条件 | 当前示例 |
| --- | --- | --- |
| `REJECT` | 配置确定矛盾，继续启动会使功能、正确性或安全性无法成立 | `role_kv_role_consistent`、`omni_cache_consistent` |
| `WARN` | 仍可正确启动，但配置会被忽略、由上游调整、影响性能，或明显偏离用户意图 | `cudagraph_capture_sizes_decode_compatible` |

选择原则：

- 只有证据充分、继续运行没有合理语义时才使用 `REJECT`。
- 性能建议和 vLLM 能够安全调整的值通常使用 `WARN`。
- 不能因为“无法确认”就选择 `REJECT`。
- `WARN` 只报告问题，不能修改配置值。

## 5. 实现 check 函数

规则函数必须满足：

- 确定性、幂等、只读、无 I/O、无副作用。
- 不在函数中记录正常违规日志、修改配置或执行自动修复。
- 不适用于当前模式时尽早返回 `None`，避免误报。
- 配置缺失时区分“功能未启用”和“缺失本身构成违规”。
- 正常通过返回 `None`，正常违规返回 `Violation`。
- 一个规则只表达一个长期不变量。

最小结构如下，其中 `OMNI_EXAMPLE_*` 和 `example_limit` 是占位名称：

```python
def _check_example_limit_consistent(
    ctx: ValidationContext,
) -> Optional[Violation]:
    from omni_npu import envs

    scheduler_config = getattr(ctx.vllm_config, "scheduler_config", None)
    if not envs.OMNI_EXAMPLE_ENABLE_FAST_PATH or scheduler_config is None:
        return None

    actual = envs.OMNI_EXAMPLE_REQUEST_LIMIT
    expected_max = getattr(scheduler_config, "example_limit", None)
    if expected_max is None or actual <= expected_max:
        return None

    return Violation(
        field_path=(
            "envs.OMNI_EXAMPLE_REQUEST_LIMIT × "
            "scheduler_config.example_limit"
        ),
        message=(
            f"OMNI_EXAMPLE_REQUEST_LIMIT={actual!r} exceeds "
            f"scheduler_config.example_limit={expected_max!r}. "
            "Reduce the environment value before deployment."
        ),
        original_value=actual,
    )
```

读取 `additional_config` 时不要扩展 `ValidationContext`：

```python
additional_config = OmniAdditionalConfig.from_vllm_config(
    ctx.vllm_config
)
```

规则应基于上游的真实解析语义设置适用条件。例如某约束只对特定
`cudagraph_mode`、speculative method 或 PD role 成立时，必须先排除其他
模式，不能将局部约束推广到所有启动方式。

## 6. 命名和诊断消息

`ValidationRule`：

- `name` 使用稳定、唯一的 `snake_case`，不添加 `_check_` 前缀。
- 名称会进入日志和测试，发布后不要随意修改。
- 当前框架不检查重名，必须由开发者和评审者保证唯一。
- `description` 使用简洁英文描述长期不变量，而不是某一次违规。

`Violation`：

- `field_path` 使用来源限定的稳定路径；跨来源可使用现有的
  `source.field × other.field` 格式。
- `message` 必须是英文单行文本，说明 actual、expected、影响，并在可行时
  给出修正动作。
- 不在 `message` 中重复 rule name、severity 或手写 `[WARNING]`。
- `original_value` 只保存相关的小型原始值，不能保存 tensor、大型对象或
  敏感信息。
- 当前 formatter 不展示 `description` 和 `original_value`；需要让用户看到
  的关键信息必须写入 `message`。

不要将 token、secret、password、credential、完整请求内容或其他敏感信息
放入 `field_path`、`message` 或 `original_value`。

## 7. 注册规则

将规则加入 [`validators.py`](../src/omni_npu/configs/validators.py)
的 `_ALL_RULES`：

```python
ValidationRule(
    name="example_limit_consistent",
    description="Example request limit must fit the resolved scheduler limit",
    severity=Severity.REJECT,
    check=_check_example_limit_consistent,
),
```

注册顺序就是诊断输出顺序。新增规则不应修改 `validate_all()` 的执行路径。

当前内置规则为：

| rule | severity | 约束 |
| --- | --- | --- |
| `role_kv_role_consistent` | `REJECT` | `OMNI_PD_ROLE` 与 KV role 必须一致 |
| `omni_cache_consistent` | `REJECT` | `additional_config` 显式设置 Omni Cache 时必须与环境变量一致；兼容仅使用环境变量的旧 omniinfer 启动方式 |
| `cudagraph_capture_sizes_decode_compatible` | `WARN` | 启用 CUDAGraph 时，capture size 应符合 Omni decode workload 建议；MTP 场景还需满足整数倍和最大 decode token 数约束 |

## 8. 执行器和日志语义

`validate_all()` 会执行所有规则：

- 所有 `WARN` 按注册顺序聚合到同一个
  `[Config Validation][WARN]` 日志块中，并保留配置原值。
- 所有 `REJECT` 聚合后通过一个 `ValueError` 抛出。
- `WARN` 与 `REJECT` 同时出现时，先输出 warning，日志标记
  `outcome=reject`，随后抛出所有 reject。
- `check()` 意外抛出 `Exception` 时，executor 会记录一条普通 warning，
  跳过该规则并继续其他规则。

最后一种行为是 fail-open：即使规则标记为 `REJECT`，规则自身的实现异常也
不会阻止启动。因此正常违规绝不能通过抛异常表达，REJECT 规则必须重点覆盖
属性缺失、类型异常和不适用模式，避免实现错误使规则失效。

warning block 的每一行是独立 LogRecord，并保持相同前缀，便于容器日志和
日志平台检索。规则自身不要输出 ANSI 颜色、内嵌换行或重复的日志级别前缀。
