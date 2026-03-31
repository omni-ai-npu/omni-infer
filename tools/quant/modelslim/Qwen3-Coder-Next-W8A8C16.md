# Qwen3-Coder-Next-W8A8C16量化方式

## 操作步骤
1. 进入[容器](../README.md#环境配置)
2. 将transformers升级到最新版
3. **再次安装modelslim工具**: bash install.sh
4. 将量化[配置文件](./quant_config/qwen3-next-80b-a3b-w8a8.yaml)拷贝到自己的目录下
5. 执行量化命令
`msmodelslim quant --model_path {bf16权重路径} --save_path {量化权重路径} --device npu --model_type Qwen3-Next-80B-A3B-Instruct --config_path {配置文件路径}`
6. 将下面[drop_offset.py](./quant_config/drop_offset.py)复制到/path/to/save/quantized/model目录下运行以去掉offset
7. 修改config.json与组网一致，/path/to/save/quantized/model/config.json中增加"quantization_config"字段
```bash
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
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": true,
                    "type": "int"
                }
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": null,
        "ignore_layers": [],
        "ignore": [
            "re:model\\.layers\\.\\d+\\.linear_attn",
            "re:model\\.layers\\.\\d+\\.self_attn",
            "re:model\\.layers\\.\\d+\\.mlp.gate",
            "re:model\\.layers\\.\\d+\\.input_layernorm",
            "re:model\\.layers\\.\\d+\\.post_attention_layernorm",
            "re:model\\.layers\\.\\d+\\.mlp.shared_expert_gate",
            "model.embed_tokens",
            "lm_head"
            ],
        "kv_cache_scheme": null,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }
```

