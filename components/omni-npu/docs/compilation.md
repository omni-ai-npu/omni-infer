# Pass Manager 使用文档

本文档说明 omni-npu 的图优化 pass 整体链路、现有 pass 的开启方法，以及新增 pass 的流程。

## 1. 整体链路

- `NPUPlatform.get_pass_manager_cls()` 返回自定义 pass manager：
  - `omni_npu.compilation.pass_manager.GraphOptiPassManager`
- vLLM 在编译阶段会把 pass manager 注入到 `post_grad_custom_post_pass`。
- `NpuGraphExAdaptor` 会将该对象继续传给 `torchair.CompilerConfig`。
- `GraphOptiPassManager.configure()` 从 `additional_config["npugraph_ex_config"]` 读取开关并注册 pass。

## 2. 启用已有 pass（例如merge_dynamic_quant）

在启动命令中通过 `--additional-config` 开启：

```bash
--additional-config '{"npugraph_ex_config":{"enable": true, "merge_dynamic_quant": true}}'
```

确保编译后端为 `npugraph_ex`，否则相关优化不会生效。

## 3. 新增 pass 流程

### 步骤 1：新增 pass 文件

在 <code>omni_npu/compilation/</code> 下新增文件（例如 `my_new_pass.py`），实现 `VllmInductorPass`子类。

- 若是 pattern 替换型 pass：在 `__init__` 中完成 `register_replacement`
- 若是 FX 改写型 pass：在 `__call__(graph)` 中直接操作图

### 步骤 2：接入 pass_manager 配置
在 `omni_npu/compilation/pass_manager.py` 的 `configure()` 中增加开关逻辑：

```python
if self.npugraph_ex_config.get("my_new_pass", False):
    from omni_npu.compilation.my_new_pass import MyNewPass
    self.passes.append(MyNewPass(config))
```

### 步骤 3：启动时开启开关
```bash
--additional-config '{"npugraph_ex_config":{"enable": true, "my_new_pass": true}}'
```
