# GLM5.0-W4A8C16量化方式

## 操作步骤
1. 进入[容器](../README.md#环境配置)
2. 将transformers升级到最新版
3. **再次安装modelslim工具**: bash install.sh
4. 将量化[配置文件](./quant_config/glm5_w4a8c16.yaml)拷贝到自己的目录下
5. 执行量化命令
`msmodelslim quant --model_path {bf16权重路径} --save_path {量化权重路径} --device npu --model_type GLM-5 --config_path {配置文件路径}`
6. 运行[ascendv1_to_omni_format.py](./quant_config/ascendv1_to_omni_format.py)，以转换为omniinfer所需格式
```bash
python ascendv1_to_omni_format.py \
--input_path {w4a8量化权重路径} \
--output_path {转换完格式后的量化权重路径} \
--device npu
```

注意：转换完成后，需要将新生成的转换完格式后的量化权重文件覆盖拷贝到w4a8量化权重路径下

7. 将[rename_rot](./quant_config/rename_rot.py)脚本拷贝到量化权重目录下，执行`python rename_rot.py`，以修改quant_model_weights.safetensors.index.json文件。
8. 将config.json改为：
```json
{
  "architectures": [
    "DeepseekV32ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "torch_dtype": "bfloat16",
  "eos_token_id": [
    154820,
    154827,
    154829
  ],
  "ep_size": 1,
  "first_k_dense_replace": 3,
  "hidden_act": "silu",
  "head_dim": 64,
  "hidden_size": 6144,
  "index_head_dim": 128,
  "index_n_heads": 32,
  "index_topk": 2048,
  "indexer_rope_interleave": true,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "kv_lora_rank": 512,
  "max_position_embeddings": 202752,
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "model_type": "deepseek_v32",
  "n_group": 1,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 64,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 78,
  "num_key_value_heads": 64,
  "num_nextn_predict_layers": 1,
  "pad_token_id": 154820,
  "pretraining_tp": 1,
  "q_lora_rank": 2048,
  "qk_head_dim": 256,
  "qk_nope_head_dim": 192,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-05,
  "rope_interleave": true,
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
    },
  "rope_parameters": {
    "rope_theta": 1000000,
    "rope_type": "default"
  },
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "tie_word_embeddings": false,
  "topk_group": 1,
  "topk_method": "noaux_tc",
  "transformers_version": "5.0.2.dev0",
  "use_cache": true,
  "v_head_dim": 256,
  "vocab_size": 154880,
  "apply_mtp_rot": true
}
```

9. 将tokenizer_config.json改为：
```json
{
  "tokenizer_class": "PreTrainedTokenizerFast",
  "clean_up_tokenization_spaces": false,
  "do_lower_case": false,
  "eos_token": "<|endoftext|>",
  "pad_token": "<|endoftext|>",
  "padding_side": "left",
  "model_max_length": 202752,
  "model_specific_special_tokens": {},
  "is_local": true,
  "remove_space": false
}
```
10. 需要chat_template.jinja文件，否则chat/completions接口报错。可从官网下载。