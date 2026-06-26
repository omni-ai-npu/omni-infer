# 添加自定义vLLM补丁到omni-npu中

本模块参考自：https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html

本指南用于说明如何在`omni-npu`中添加、注册和执行自定义vLLM补丁。

## 1.准备补丁文件

将补丁文件放置在`src/omni_npu/vllm_patches/patches`目录下。

补丁文件至少需要定义一个继承自`VLLMPatch`的补丁类，并通过`@register_patch(name, target)`完成注册，同时定义`_attr_names_to_apply`，表示目标类或目标模块中需要新增或替换的属性。

`@register_patch(name, target)`中的：

- `name`表示补丁在`PatchManager`中的注册名
- `target`表示vLLM中被打补丁的类或模块

可以参考`src/omni_npu/vllm_patches/patches/examples/llm_engine_hello_world.py`中的`LLMEngineHelloWorldPatch`：

### 类函数拓展示例

```python
from vllm.v1.engine.llm_engine import LLMEngine

@register_patch("LLMEngineHelloWorld", LLMEngine)
class LLMEngineHelloWorldPatch(VLLMPatch):
    """
    Makes LLMEngines print 'Hello World' when get supported tasks.
    """

    _attr_names_to_apply = ["print_hello_world", "get_supported_tasks"]

    @staticmethod
    def print_hello_world():
        print("Hello World")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        self.print_hello_world()
        return self.engine_core.get_supported_tasks()
```

上述例子中，`LLMEngineHelloWorldPatch`注册名为`LLMEngineHelloWorld`，目标类为`vllm.v1.engine.llm_engine.LLMEngine`。补丁通过`_attr_names_to_apply`声明要新增或替换`print_hello_world`和`get_supported_tasks`。

### 模块级函数拓展示例

```python
import vllm.engine.arg_utils as arg_utils

@register_patch("GetKwargsHelloWorld", arg_utils)
class GetKwargsHelloWorldPatch(VLLMPatch):
    _attr_names_to_apply = ["get_kwargs"]

    def get_kwargs(cls):
        logger.info(">>> Hello World: get_kwargs is called for %s", cls)
        return copy.deepcopy(_compute_kwargs(cls))
```

上述例子中，`GetKwargsHelloWorldPatch`的目标模块是`vllm.engine.arg_utils`，补丁会替换模块中的`get_kwargs`函数。

---

## 2.运行时指定执行补丁

补丁文件需要经过两个环节：

- 注册：补丁文件被导入后，`@register_patch`会把补丁注册到`PatchManager`
- 执行：通过`OMNI_NPU_VLLM_PATCHES`决定实际应用哪些已注册补丁

### 补丁文件注册

`/patches/common`用于存放公共补丁，`/patches/models/xxxmodel`用于存放模型相关补丁。

模型补丁支持两种注册方式：

- 手动注册：通过`OMNI_NPU_PATCHES_DIR`直接指定目录
- 自动注册：通过模型`config.json`中的`model_type`自动匹配目录

手动注册优先于自动注册。

#### 手动注册

当设置`OMNI_NPU_PATCHES_DIR=xxxmodel`时，会导入`patches/models/xxxmodel`目录下的补丁文件。

```bash
export OMNI_NPU_PATCHES_DIR="pangu72b-vl"
# 对应目录：patches/models/pangu72b-vl
```

`OMNI_NPU_PATCHES_DIR`支持逗号分隔多个目录，目录会按顺序依次加载：

```bash
export OMNI_NPU_PATCHES_DIR="pangu_v2_base,pangu_sink_swa_mla"
# 先加载pangu_v2_base/，再加载pangu_sink_swa_mla/
```

#### 自动注册

服务启动时指定`VLLM_PLUGINS="omni-npu,omni_npu_patches"`后，会自动导入`/patches/common`和匹配到的`/patches/models/xxxmodel`目录。

自动匹配流程如下：

```python
model_path = Path(sys.argv[2])
model_type = get_model_type_from_config(model_path)
models_root = patches_root / "models"
model_dirs = _find_patch_dir_fuzzy(model_type, models_root)
```

示例：

```bash
model="/data/models/DeepSeek-V3.2-INT8"
VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models" vllm serve "$model"
```

`model_type`与目录名称的匹配优先级如下：

1. 映射表匹配：在`src/omni_npu/vllm_patches/__init__.py`中维护映射关系
2. 前缀匹配：目录名是`model_type`前缀
3. 包含匹配：目录名是`model_type`子串

其中映射表支持多目录映射，例如：

```python
"openpangu_v2": "pangu_v2_base,pangu_sink_swa_mla"
```

### 补丁文件执行

通过环境变量`OMNI_NPU_VLLM_PATCHES`指定具体执行哪些补丁。

默认行为如下：

- 未设置`OMNI_NPU_VLLM_PATCHES`时，默认执行所有已注册补丁
- 设置为空字符串时，默认执行所有已注册补丁
- 设置为`"ALL"`时，默认执行所有已注册补丁

简化用法如下：

```bash
VLLM_PLUGINS="omni-npu,omni_npu_patches" vllm serve /path/to/model
```

显式指定执行全部补丁：

```bash
VLLM_PLUGINS="omni-npu,omni_npu_patches" \
OMNI_NPU_VLLM_PATCHES="ALL" \
vllm serve /path/to/model
```

只执行指定补丁：

```bash
VLLM_PLUGINS="omni-npu,omni_npu_patches" \
OMNI_NPU_VLLM_PATCHES="PatchA,PatchB" \
vllm serve /path/to/model
```

其中`PatchA`和`PatchB`是补丁在`PatchManager`中的注册名。

---

## 3.模型族通用补丁与模型特有补丁

### 目录职责

- `patches/models/pangu_v2_base`：Pangu V2模型族通用补丁目录
- `patches/models/pangu_sink_swa_mla`：`openpangu_v2`特有补丁目录
- `patches/models/openpangu_ultra_omni`：`openpangu_ultra_omni`特有补丁目录

### 生效范围

`pangu_v2_base`目前只对以下两个模型生效：

- `openpangu_v2`
- `openpangu_ultra_omni`

对应加载顺序如下：

- `openpangu_v2`→`pangu_v2_base`→`pangu_sink_swa_mla`
- `openpangu_ultra_omni`→`pangu_v2_base`→`openpangu_ultra_omni`

这意味着：

- 两个模型共享的补丁应放在`pangu_v2_base`
- 只对单个模型生效的补丁应放在各自模型目录
- 其他模型不受`pangu_v2_base`影响

### 什么时候放到`pangu_v2_base`

满足以下条件时，建议放到`pangu_v2_base`：

- 补丁逻辑同时适用于`openpangu_v2`和`openpangu_ultra_omni`
- 两个模型目标符号一致，行为一致
- 提取后能减少重复代码和重复维护

如果补丁只服务其中一个模型，或两个模型的实现细节不同，就不要放到`pangu_v2_base`。

### 后续如何新增补丁文件

如果要给Pangu V2模型族新增通用补丁，直接在`patches/models/pangu_v2_base/`下新增`patch_*.py`文件即可，文件会随目录一起按文件名顺序导入。

目录说明统一放在`patches/models/pangu_v2_base/patch_pangu_v2_common.py`文件头注释中。后续维护这个目录时，可以优先查看这个文件。

建议遵循以下原则：

- 文件名使用`patch_xxx.py`
- 共享逻辑放在`pangu_v2_base`
- 模型特有逻辑留在各自模型目录
- 不要把只对单个模型生效的补丁放进`pangu_v2_base`

---

## 4.补丁生效范围

补丁通过`vllm.plugins.load_general_plugins`加载。只有在补丁执行之后再被import的函数、类或模块，才会使用补丁后的实现。

---

## 5.使用示例

### 自动应用所有补丁

```bash
VLLM_PLUGINS="omni-npu,omni_npu_patches" vllm serve /data/models/DeepSeek-V3.2-INT8
```

效果：自动加载并应用所有common补丁和deepseek模型补丁。

### 手动指定单个模型补丁目录

```bash
VLLM_PLUGINS="omni-npu,omni_npu_patches" \
OMNI_NPU_PATCHES_DIR="deepseek" \
vllm serve /data/models/DeepSeek-V3.2-INT8
```

效果：精确匹配`patches/models/deepseek`目录并加载对应补丁。

### 手动指定多个模型补丁目录

```bash
VLLM_PLUGINS="omni-npu,omni_npu_patches" \
OMNI_NPU_PATCHES_DIR="pangu_v2_base,pangu_sink_swa_mla" \
vllm serve /path/to/openpangu_v2_model
```

效果：按顺序加载两个目录的补丁，先加载`pangu_v2_base/`，再加载`pangu_sink_swa_mla/`。

### 只应用特定补丁

```bash
VLLM_PLUGINS="omni-npu,omni_npu_patches" \
OMNI_NPU_VLLM_PATCHES="EngineArgsConfig,DSV32Indexer" \
vllm serve /data/models/DeepSeek-V3.2-INT8
```

效果：只应用`EngineArgsConfig`和`DSV32Indexer`两个补丁。

---

## 6.常见问题

### 补丁没有生效

请确认以下几点：

1. `VLLM_PLUGINS`包含`omni-npu`和`omni_npu_patches`
2. 补丁文件已正确放在`/patches/common/`或`/patches/models/xxxmodel/`
3. 补丁类已使用`@register_patch`
4. 日志中有对应的补丁注册或补丁应用日志

### 如何查看已注册补丁

启动服务时，日志会打印已注册补丁，例如：

```text
INFO omni-npu[patch_manager.py:21] patch class XXXPatch registered as PatchName
```

### 补丁是否可以重复应用

不可以。重复应用的补丁会被跳过，并在日志中输出告警。

