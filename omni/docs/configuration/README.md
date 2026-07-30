# Omni-NPU 配置开发规范

本文约定两类配置开发行为：

1. 如何命名、注册、读取和测试新的 Omni-NPU 环境变量。
2. 如何在 `OmniAdditionalConfig` 中增加结构化启动配置。

跨配置源或依赖完整启动状态的只读约束，请参阅独立的
[ValidationRule 开发规范](../validation/README.md)。本文不定义 validator
的实现、severity 或测试要求。

`ModelExtraConfig` 字段和模型最佳实践 JSON 的开发及使用，请直接参阅已有的
[模型配置项自动加载使用说明](../../src/omni_npu/model_config/README.md)。

本文是开发规范，不是配置项清单。真实行为以以下代码为准：

- [`envs.py`](../../src/omni_npu/envs.py)：Omni-NPU 自有环境变量的唯一注册表。
- [`additional_config.py`](../../src/omni_npu/configs/additional_config.py)：
  `vllm_config.additional_config` 中 Omni-NPU 字段的类型和默认值。

## 1. 先确定配置应放在哪里

新增配置前先判断其所有权和生命周期，不要默认使用环境变量。

| 配置需求 | 应放置的位置 |
| --- | --- |
| vLLM 已提供等价参数或字段 | 直接使用 vLLM CLI 或 `VllmConfig`，不创建 Omni-NPU 副本 |
| Omni-NPU 自有、需要结构化传入且随服务配置确定 | `--additional-config`，并在 `OmniAdditionalConfig` 中声明 |
| 与模型、硬件、量化方式或部署形态绑定的最佳实践参数 | [ModelExtraConfig 使用说明](../../src/omni_npu/model_config/README.md) |
| 进程启动前就要生效的部署开关，或诊断、调试、实验开关 | Omni-NPU 环境变量 |
| 单字段的类型、格式、范围或默认值校验 | 字段所属的 parser、dataclass 或 loader |
| 单个配置对象构造时即可判断的内部约束 | 所属 dataclass 的 `__post_init__` 或 loader |
| 需要根据最终运行模式修改、归一化或覆盖模型配置值 | loader 或具体 feature |

如果字段由 vLLM、HCCL、Ascend 或 PyTorch 所有，应保留所有者的命名和读取
方式，例如 `VLLM_*`、`HCCL_*`、`ASCEND_*`、`PYTORCH_*`。不要为其再创建
一个 `OMNI_*` 副本。

## 2. 环境变量命名

### 2.1 基本格式

所有新建的 Omni-NPU 自有环境变量必须使用：

```text
OMNI_<SUBSYSTEM>_<SEMANTIC_NAME>[_<UNIT>]
```

具体要求：

- 使用大写蛇形命名，统一以 `OMNI_` 开头。
- 优先按语义域分组，例如 `OMNI_PD_*`、`OMNI_PROFILE_*`、
  `OMNI_PROFILE_*`。
- 只有语义明确属于 NPU 实现时才使用 `OMNI_NPU_*`；不能仅因为变量位于
  omni-npu 仓库就机械添加 `NPU`。
- 布尔量使用正向、可读的动词，例如 `ENABLE`、`DISABLE`、`USE`、
  `SKIP`、`VALIDATE`。
- 数值存在明确单位时将单位写入名称，例如 `_MS`、`_BYTES`、`_MB`。
- 名称描述配置语义，而不是某个 consumer 的文件名或临时实现细节。

新代码不得继续引入 `ROLE`、`PROFILER_STOP_STEP` 这类无所有者前缀的公开名称。

### 2.2 新增变量不设置旧名称

新增变量调用 `get_env_with_fallback()` 时，`old_names` 必须为 `None`：

```python
"OMNI_EXAMPLE_ENABLE_FAST_PATH":
lambda: get_env_with_fallback(
    "OMNI_EXAMPLE_ENABLE_FAST_PATH",
    None,
    False,
    _as_bool,
),
```

未设置新增变量时，直接返回其默认值。

## 3. 注册和读取环境变量

### 3.1 添加类型声明和注册项

在 `if TYPE_CHECKING:` 的对应分组中添加准确的运行时类型：

```python
if TYPE_CHECKING:
    OMNI_EXAMPLE_ENABLE_FAST_PATH: bool
    OMNI_EXAMPLE_REQUEST_LIMIT: int
```

再在 `# begin-env-vars-definition` 和 `# end-env-vars-definition` 之间添加
唯一注册项。注册表 key 必须与 `get_env_with_fallback()` 的 `new_name`
完全相同。

默认值必须已经是目标 Python 类型。parser 只解析真实环境变量中的字符串，
不会解析默认值：

```python
# Correct: the parsed value and default are both int.
"OMNI_EXAMPLE_REQUEST_LIMIT":
lambda: get_env_with_fallback(
    "OMNI_EXAMPLE_REQUEST_LIMIT",
    None,
    64,
    _as_int,
),
```

不要将整数默认值写成字符串 `"64"`。

### 3.2 选择 parser 和默认值

| parser | 当前精确行为 | 适用场景 |
| --- | --- | --- |
| 无 parser | 返回原始字符串；显式 `""` 也视为已设置 | 路径、枚举字符串、JSON 字符串等 |
| `_as_bool` | 去除首尾空白并转小写；仅 `1`、`true` 为 `True`，其他字符串均为 `False` | 普通布尔开关 |
| `_as_exact_one` | 仅原始字符串精确等于 `1` 时为 `True`；不忽略空白、不接受大小写变体 | 需要保持既有“仅 `1` 启用”接口语义的兼容开关 |
| `_as_int` | 调用 `int(raw)`；非法输入在属性访问时抛出 `ValueError` | 非法值应使启动失败的整数 |
| `_as_bool_or_all` | 在 `_as_bool` 基础上额外接受 `all` | 现有 benchmark gate；普通功能不要使用 |
| `_as_int_or_default` | 非法整数记录 warning 并使用默认值 | 现有非生产 benchmark 阈值；普通生产配置不要使用 |

不要因为多个值都是 falsy 就混用 `None`、`""`、`0` 和 `False`：

- `None`：未设置本身是一种状态。
- `""`：consumer 明确将空字符串视为关闭或空路径。
- `0`：整数零是有效值，或需要与未设置区分。
- `False`：明确关闭布尔功能。

如果现有 parser 的容错语义不符合新变量，应定义命名明确的新 parser，并为
非法输入策略添加测试，不能在 consumer 中重复解析。

### 3.3 保持懒读取

`envs.py` 通过模块级 `__getattr__` 在每次访问时解析变量。consumer 必须读取
模块属性：

```python
from omni_npu import envs

if envs.OMNI_EXAMPLE_ENABLE_FAST_PATH:
    enable_fast_path()
```

禁止以下新代码：

```python
# Bypasses the central registry.
value = os.environ.get("OMNI_EXAMPLE_ENABLE_FAST_PATH")

# Binds one resolved value and loses later environment changes.
from omni_npu.envs import OMNI_EXAMPLE_ENABLE_FAST_PATH
```

注册表是懒求值的，因此非法值可能在第一次访问时才报错。

### 3.4 编写变量注释

每个注册项前使用英文注释，并至少说明：

- 值域、类型和单位。
- unset、空字符串和默认值的准确语义。
- 生效阶段、触发条件，以及依赖的模型、部署模式或 patch。
- 与其他配置的优先级、互斥或一致性约束。
- 非法值是终止启动还是告警后回退。
- 真实 consumer 的路径和符号。
- 调试或 benchmark 开关是否禁止用于生产。

注释必须描述当前真实行为。规划中的语义需要明确标注为 staged，不能写成
已经生效。

### 3.5 配置摘要和敏感信息

新的 `OMNI_*` 变量会被 OMNI-CONF 的前缀白名单自动采集。

变量包含 token、secret、password、credential 或其他敏感信息时，必须确认
摘要脱敏规则能够命中。非标准敏感名称需要显式补充 mask 规则和测试。

## 4. 环境变量测试

测试放在 [`test_envs.py`](../../tests/config/test_envs.py)，并保持为不依赖
Torch 的纯标准库测试。每个变量根据实际语义覆盖：

- 未设置时的默认值及准确 Python 类型。
- 新名称合法值的解析。
- 非法值是抛异常还是告警后使用默认值。
- `None`、`""`、`0`、`False` 等需要区分的状态。
- 变量能够通过 `dir(envs)` 或注册表被发现。
- 懒读取是否需要观察运行期间的环境变化。
- consumer 的实际功能效果。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
pytest -q -p no:cacheprovider tests/config/test_envs.py
```

## 5. OmniAdditionalConfig 规范

### 5.1 定位与读取方式

vLLM 将 `--additional-config '<JSON>'` 解析到共享字典
`vllm_config.additional_config`。Omni-NPU 自有字段统一声明在
`OmniAdditionalConfig` 中：

```bash
vllm serve ... \
  --additional-config '{"enable_low_latency":true}'
```

consumer 不应直接散落调用
`vllm_config.additional_config.get("field", default)`，也不应把解析结果复制
到另一个 context。每次使用时从当前 `vllm_config` 解析：

```python
from omni_npu.configs import OmniAdditionalConfig

additional_config = OmniAdditionalConfig.from_vllm_config(vllm_config)
if additional_config.enable_low_latency:
    ...
```

当前解析是 per-call 的；后续调用能够看到
`vllm_config.additional_config` 的最新值。不要在模块导入时缓存解析结果。

### 5.2 新增字段

新增 Omni-NPU 自有字段时：

1. 在 `OmniAdditionalConfig` 中添加准确类型和类型正确的默认值。
2. 使用英文注释说明含义、默认行为、生效条件和真实 consumer。
3. 如果属于现有同类约束，将字段同步加入 `_BOOL_FIELDS`、
   `_POSITIVE_INT_FIELDS` 等校验组；其他类型或范围在 `__post_init__`
   中显式检查。
4. consumer 只从 `OmniAdditionalConfig.from_vllm_config()` 的结果读取。
5. 为默认值、合法值、非法类型、边界和 consumer 行为添加测试。

dataclass 类型注解不会自动执行运行时类型校验。需要拒绝 JSON 字符串
`"true"`、整数 `1` 或非法范围时，必须写显式检查，不能只写类型注解。

`additional_config` 是 vLLM 和其他插件共享的字典，因此未知 key 当前会被
忽略。这个行为允许其他所有者共用字典，但也意味着 Omni-NPU 字段拼写错误
会静默回退默认值。新增字段必须用实际解析和 consumer 测试证明它生效，不能
只检查启动 JSON 文本。

### 5.3 默认值与所有权

- 默认值必须在没有该字段的所有既有部署中保持安全、可运行。
- 新的 opt-in 功能通常默认关闭，但应以兼容性和真实语义为准。
- 不要在 `additional_config` 中复制已有 vLLM 字段。
- 不要声明其他插件所有的 key。
- 结构化子配置应使用 `field(default_factory=...)`，避免共享可变默认值。
- 单字段及同一配置对象内部的依赖在 `__post_init__` 校验。

测试放在
[`test_additional_config.py`](../../tests/config/test_additional_config.py)：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
pytest -q -p no:cacheprovider tests/config/test_additional_config.py
```
