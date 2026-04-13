# kimi2.5-W4A8C16量化方式

## 操作步骤
### 一、INT4→BF16权重转换
#### 1. 下载原始权重
官网发布模型为int4量化权重。考虑到 msmodelslim 的格式要求，量化前需要将模型权重转化到bf16格式。
#### 2. 进入[容器](../README.md#环境配置)
#### 3. 转化原有权重到bf16
- 1. 昇腾（Ascend）不支持flash_attn库
将权重文件夹中modeling_deepseek.py文件中is_flash_attn_2_available相关代码注释掉。
- 2. 修改config
将原始权重的config.json中的"quantization_config"字段删掉。
- 3. 拉取转换脚本，执行int4→bf16转换
```bash
git clone https://gitcode.com/libarry/msit_9400.git
cd msit_9400
git checkout k2
cd msmodelslim/example/DeepSeek
python convert_int4_to_bf16.py \
  --input-int4-hf-path ${原始权重路径} \
  --output-bf16-hf-path ${bf16权重路径}
```
注意：转换完成后，需要将原模型文件中除权重文件以外的其他文件拷贝到新生成的bf16权重路径下

### 二、BF16→W4A8C16量化
#### 1. pip install -U transformers 升级transformers
#### 2. **拉取量化代码**: 
代码仓需要用私仓分支的代码：
```bash
git clone https://gitcode.com/Liziqi77/msmodelslim
cd msmodelslim
git checkout -b kimi2.5 origin/kimi2.5
```
#### 3. **修改量化代码**
- 1. 拷贝相关代码
将模型权重文件夹下的configuration_deepseek.py、configuration_kimi_k25.py、kimi_k25_processor.py、kimi_k25_vision_processing.py、media_utils.py、modeling_deepseek.py、modeling_kimi_k25.py、tokenization_kimi.py、tool_declaration_ts.py九个文件拷贝到./msmodelslim/msmodelslim/model/kimi_k2_5目录下。搜索`@torch.compile(dynamic=True)`，将此字段删掉。
- 2. 修改代码
1) 修改model_adapter.py文件中get_model_from_pretrained参数为：
```python
model = SafeGenerator.get_model_from_pretrained(model_path=str(self.model_path),
                                                                config=self.config,
                                                                trust_remote_code=self.trust_remote_code,
                                                                torch_dtype="auto",
                                                                low_cpu_mem_usage=True,
                                                                attn_implementation='eager')
```
2） 在modeling_kimi_k25.py文件中，`self.blocks = nn.ModuleList`前加上`self.use_deterministic_attn = False`
- 3. **安装modelslim工具**: 
`bash install.sh`
#### 4. 量化
- 1. 将量化[配置文件](./quant_config/kimi2.5_w4a8c16.yaml)拷贝到自己的目录下
- 2. 执行量化命令
```bash
msmodelslim quant --model_path {bf16权重路径} \
--save_path {量化权重路径} \
--device npu \
--model_type Kimi-K2.5 \
--config_path {配置文件路径} \
--trust_remote_code True
```
#### 5. 将权重转换为omniinfer所需格式
运行[ascendv1_to_omni_format.py](./quant_config/ascendv1_to_omni_format.py)，以转换为omniinfer所需格式
```bash
python ascendv1_to_omni_format.py \
--input_path {w4a8量化权重路径} \
--output_path {转换完格式后的量化权重路径} \
--device npu
```
注意：转换完成后，需要将新生成的转换完格式后的量化权重文件覆盖拷贝到w4a8量化权重路径下
#### 6. 修改配置
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
			                      "self_attn.fused_qkv_a_proj": 8,
                            "self_attn.kv_a_proj_with_mqa": 8,
                            "self_attn.q_a_proj": 8,
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
            "global_compression_ratio": 1.5943962512751308,
            "ignore": [
                "language_model.layers.0.self_attn.kv_b_proj",
                "language_model.layers.1.self_attn.kv_b_proj",
                "language_model.layers.2.self_attn.kv_b_proj",
                "language_model.layers.3.self_attn.kv_b_proj",
                "language_model.layers.4.self_attn.kv_b_proj",
                "language_model.layers.5.self_attn.kv_b_proj",
                "language_model.layers.6.self_attn.kv_b_proj",
                "language_model.layers.7.self_attn.kv_b_proj",
                "language_model.layers.8.self_attn.kv_b_proj",
                "language_model.layers.9.self_attn.kv_b_proj",
                "language_model.layers.10.self_attn.kv_b_proj",
                "language_model.layers.11.self_attn.kv_b_proj",
                "language_model.layers.12.self_attn.kv_b_proj",
                "language_model.layers.13.self_attn.kv_b_proj",
                "language_model.layers.14.self_attn.kv_b_proj",
                "language_model.layers.15.self_attn.kv_b_proj",
                "language_model.layers.16.self_attn.kv_b_proj",
                "language_model.layers.17.self_attn.kv_b_proj",
                "language_model.layers.18.self_attn.kv_b_proj",
                "language_model.layers.19.self_attn.kv_b_proj",
                "language_model.layers.20.self_attn.kv_b_proj",
                "language_model.layers.21.self_attn.kv_b_proj",
                "language_model.layers.22.self_attn.kv_b_proj",
                "language_model.layers.23.self_attn.kv_b_proj",
                "language_model.layers.24.self_attn.kv_b_proj",
                "language_model.layers.25.self_attn.kv_b_proj",
                "language_model.layers.26.self_attn.kv_b_proj",
                "language_model.layers.27.self_attn.kv_b_proj",
                "language_model.layers.28.self_attn.kv_b_proj",
                "language_model.layers.29.self_attn.kv_b_proj",
                "language_model.layers.30.self_attn.kv_b_proj",
                "language_model.layers.31.self_attn.kv_b_proj",
                "language_model.layers.32.self_attn.kv_b_proj",
                "language_model.layers.33.self_attn.kv_b_proj",
                "language_model.layers.34.self_attn.kv_b_proj",
                "language_model.layers.35.self_attn.kv_b_proj",
                "language_model.layers.36.self_attn.kv_b_proj",
                "language_model.layers.37.self_attn.kv_b_proj",
                "language_model.layers.38.self_attn.kv_b_proj",
                "language_model.layers.39.self_attn.kv_b_proj",
                "language_model.layers.40.self_attn.kv_b_proj",
                "language_model.layers.41.self_attn.kv_b_proj",
                "language_model.layers.42.self_attn.kv_b_proj",
                "language_model.layers.43.self_attn.kv_b_proj",
                "language_model.layers.44.self_attn.kv_b_proj",
                "language_model.layers.45.self_attn.kv_b_proj",
                "language_model.layers.46.self_attn.kv_b_proj",
                "language_model.layers.47.self_attn.kv_b_proj",
                "language_model.layers.48.self_attn.kv_b_proj",
                "language_model.layers.49.self_attn.kv_b_proj",
                "language_model.layers.50.self_attn.kv_b_proj",
                "language_model.layers.51.self_attn.kv_b_proj",
                "language_model.layers.52.self_attn.kv_b_proj",
                "language_model.layers.53.self_attn.kv_b_proj",
                "language_model.layers.54.self_attn.kv_b_proj",
                "language_model.layers.55.self_attn.kv_b_proj",
                "language_model.layers.56.self_attn.kv_b_proj",
                "language_model.layers.57.self_attn.kv_b_proj",
                "language_model.layers.58.self_attn.kv_b_proj",
                "language_model.layers.59.self_attn.kv_b_proj",
                "language_model.layers.60.self_attn.kv_b_proj",
                "language_model.layers.61.self_attn.kv_b_proj"
            ],
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed"
        }
```
