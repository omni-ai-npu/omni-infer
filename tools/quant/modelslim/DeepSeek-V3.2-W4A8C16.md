# DeepSeek-V3.2-W4A8C16量化方式

## 操作步骤
### 一、FP8→BF16权重转换
#### 1. 下载原始权重
官网发布模型为fp8混合精度模型，量化前需要将模型权重转化到bf16格式。
#### 2. 进入[容器](../README.md#环境配置)
#### 3. 转化原有权重到bf16
进入[格式转换文件目录下](../)执行：
`python fp8_cast_bf16.py --input-fp8-hf-path {fp8权重路径} --output-bf16-hf-path {bf16权重路径}`
将config.json中的"quantization_config"字段删除
注意：转换完成后，需要将原模型文件中**除权重文件**以外的其他文件拷贝到新生成的bf16权重路径下

### 二、BF16→W4A8C16量化
#### 1. pip install -U transformers 升级transformers
#### 2. **安装modelslim工具**: 
`bash install.sh`
#### 3. 量化
- 1. 将量化[配置文件](./quant_config/deepseek32_w4a8.yaml)拷贝到自己的目录下
- 2. 执行量化命令
```bash
msmodelslim quant --model_path {bf16权重路径} \
--save_path {量化权重路径} \
--device npu \
--model_type DeepSeek-V3.2 \
--config_path {配置文件路径} \
--trust_remote_code True
```
#### 4. 将权重转换为omniinfer所需格式
运行[ascendv1_to_omni_format.py](./quant_config/ascendv1_to_omni_format.py)，以转换为omniinfer所需格式
```bash
python ascendv1_to_omni_format.py \
--input_path {w4a8量化权重路径} \
--output_path {转换完格式后的量化权重路径} \
--device npu
```
注意：转换完成后，需要将新生成的转换完格式后的量化权重文件覆盖拷贝到w4a8量化权重路径下
#### 5. 修改配置
修改config.json与组网一致，/path/to/save/quantized/model/config.json中增加"quantization_config"字段：
```json
"quantization_config": {
    "config_groups": {
      "group_0": {
        "input_activations": {
          "actorder": null,
          "block_structure": null,
          "dynamic": true,
          "group_size": null,
          "num_bits": 8,
          "observer": "memoryless",
          "observer_kwargs": {},
          "strategy": "token",
          "symmetric": true,
          "type": "int"
        },
        "output_activations": null,
        "targets": [
          "Linear"
        ],
        "weights": {
          "actorder": null,
          "block_structure": null,
          "dynamic": false,
          "group_size": null,
          "num_bits": {
            "self_attn.kv_a_proj_with_mqa": 16,
            "self_attn.q_a_proj": 16,
            "self_attn.indexer.wk": 16,
            "self_attn.indexer.weights_proj": 16,
            "self_attn.indexer.wq_b": 8,
            "self_attn.q_b_proj": 8,
            "self_attn.o_proj": 8,
            "mlp.down_proj": 8,
            "mlp.gate_up_proj": 8,
            "mlp.shared_experts": 8,
            "mlp.experts": 4
          },
          "observer": "minmax",
          "observer_kwargs": {},
          "strategy": "channel",
          "symmetric": true,
          "type": "int"
        }
      }
    },
    "format": "int-quantized",
    "ignore": [
      "model.layers.0.self_attn.kv_b_proj",
      "model.layers.1.self_attn.kv_b_proj",
      "model.layers.2.self_attn.kv_b_proj",
      "model.layers.3.self_attn.kv_b_proj",
      "model.layers.4.self_attn.kv_b_proj",
      "model.layers.5.self_attn.kv_b_proj",
      "model.layers.6.self_attn.kv_b_proj",
      "model.layers.7.self_attn.kv_b_proj",
      "model.layers.8.self_attn.kv_b_proj",
      "model.layers.9.self_attn.kv_b_proj",
      "model.layers.10.self_attn.kv_b_proj",
      "model.layers.11.self_attn.kv_b_proj",
      "model.layers.12.self_attn.kv_b_proj",
      "model.layers.13.self_attn.kv_b_proj",
      "model.layers.14.self_attn.kv_b_proj",
      "model.layers.15.self_attn.kv_b_proj",
      "model.layers.16.self_attn.kv_b_proj",
      "model.layers.17.self_attn.kv_b_proj",
      "model.layers.18.self_attn.kv_b_proj",
      "model.layers.19.self_attn.kv_b_proj",
      "model.layers.20.self_attn.kv_b_proj",
      "model.layers.21.self_attn.kv_b_proj",
      "model.layers.22.self_attn.kv_b_proj",
      "model.layers.23.self_attn.kv_b_proj",
      "model.layers.24.self_attn.kv_b_proj",
      "model.layers.25.self_attn.kv_b_proj",
      "model.layers.26.self_attn.kv_b_proj",
      "model.layers.27.self_attn.kv_b_proj",
      "model.layers.28.self_attn.kv_b_proj",
      "model.layers.29.self_attn.kv_b_proj",
      "model.layers.30.self_attn.kv_b_proj",
      "model.layers.31.self_attn.kv_b_proj",
      "model.layers.32.self_attn.kv_b_proj",
      "model.layers.33.self_attn.kv_b_proj",
      "model.layers.34.self_attn.kv_b_proj",
      "model.layers.35.self_attn.kv_b_proj",
      "model.layers.36.self_attn.kv_b_proj",
      "model.layers.37.self_attn.kv_b_proj",
      "model.layers.38.self_attn.kv_b_proj",
      "model.layers.39.self_attn.kv_b_proj",
      "model.layers.40.self_attn.kv_b_proj",
      "model.layers.41.self_attn.kv_b_proj",
      "model.layers.42.self_attn.kv_b_proj",
      "model.layers.43.self_attn.kv_b_proj",
      "model.layers.44.self_attn.kv_b_proj",
      "model.layers.45.self_attn.kv_b_proj",
      "model.layers.46.self_attn.kv_b_proj",
      "model.layers.47.self_attn.kv_b_proj",
      "model.layers.48.self_attn.kv_b_proj",
      "model.layers.49.self_attn.kv_b_proj",
      "model.layers.50.self_attn.kv_b_proj",
      "model.layers.51.self_attn.kv_b_proj",
      "model.layers.52.self_attn.kv_b_proj",
      "model.layers.53.self_attn.kv_b_proj",
      "model.layers.54.self_attn.kv_b_proj",
      "model.layers.55.self_attn.kv_b_proj",
      "model.layers.56.self_attn.kv_b_proj",
      "model.layers.57.self_attn.kv_b_proj",
      "model.layers.58.self_attn.kv_b_proj",
      "model.layers.59.self_attn.kv_b_proj",
      "model.layers.60.self_attn.kv_b_proj",
      "model.layers.61.self_attn.kv_b_proj",
      "lm_head"
    ],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed"
  }
```
