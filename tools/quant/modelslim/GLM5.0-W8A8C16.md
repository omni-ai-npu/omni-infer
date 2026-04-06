# GLM5.0-W8A8C16量化方式

## 操作步骤
1. 进入[容器](../README.md#环境配置)
2. 将transformers升级到最新版
3. **再次安装modelslim工具**: bash install.sh
4. 将量化[配置文件](./quant_config/glm5_w8a8c16.yaml)拷贝到自己的目录下
5. 执行量化命令
`msmodelslim quant --model_path {bf16权重路径} --save_path {量化权重路径} --device npu --model_type GLM-5 --config_path {配置文件路径}`
6. 将下面[drop_offset.py](./quant_config/drop_offset.py)复制到/path/to/save/quantized/model目录下运行以去掉offset
执行drop_offset.py，将rot.safetensors移动到optional文件夹内
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
            "mlp.experts": 8
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
      "model.layers.62.self_attn.kv_b_proj",
      "model.layers.63.self_attn.kv_b_proj",
      "model.layers.64.self_attn.kv_b_proj",
      "model.layers.65.self_attn.kv_b_proj",
      "model.layers.66.self_attn.kv_b_proj",
      "model.layers.67.self_attn.kv_b_proj",
      "model.layers.68.self_attn.kv_b_proj",
      "model.layers.69.self_attn.kv_b_proj",
      "model.layers.70.self_attn.kv_b_proj",
      "model.layers.71.self_attn.kv_b_proj",
      "model.layers.72.self_attn.kv_b_proj",
      "model.layers.73.self_attn.kv_b_proj",
      "model.layers.74.self_attn.kv_b_proj",
      "model.layers.75.self_attn.kv_b_proj",
      "model.layers.76.self_attn.kv_b_proj",
      "model.layers.77.self_attn.kv_b_proj",
      "model.layers.78.self_attn.kv_b_proj",
      "lm_head"
    ],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed"
  }
```

