# MiniMax-2.5-W8A8C16量化方式

## 操作步骤
### 一、FP8→BF16权重转换
#### 1. 下载原始权重
官网发布模型为fp8混合精度模型。考虑到 msmodelslim 的格式要求，量化前需要将模型权重转化到bf16格式。
#### 2. 进入[容器](../README.md#环境配置)
#### 3. 转化原有权重到bf16
考虑到量化过程中fp8存储权重格式，将[转换脚本](./quant_config/minimax_fp8_to_bf16.py)保存到自己目录，修改"MODEL_DIR"、"OUTPUT_DIR"路径，将权重转换为bf16格式。

#### 4. 删除模型默认的初始量化配置，删除后模型配置文件为：
```bash
{
  "architectures": [
    "MiniMaxM2ForCausalLM"
  ],
  "attn_type_list": [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
  ],
  "auto_map": {
    "AutoConfig": "configuration_minimax_m2.MiniMaxM2Config",
    "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM"
  },
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 3072,
  "intermediate_size": 1536,
  "max_position_embeddings": 196608,
  "model_type": "minimax_m2",
  "mtp_transformer_layers": 1,
  "num_attention_heads": 48,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 62,
  "num_key_value_heads": 8,
  "num_local_experts": 256,
  "num_mtp_modules": 3,
  "qk_norm_type": "per_layer",
  "rms_norm_eps": 1e-06,
  "rope_theta": 5000000,
  "rotary_dim": 64,
  "scoring_func": "sigmoid",
  "shared_intermediate_size": 0,
  "tie_word_embeddings": false,
  "transformers_version": "4.46.1",
  "use_cache": true,
  "use_mtp": true,
  "use_qk_norm": true,
  "use_routing_bias": true,
  "vocab_size": 200064,
  "torch_dtype": "bfloat16"
```
至此，待量化模型文件准备就绪。

### 二、BF16→W8A8C16量化
#### 1. pip install transformers==4.57.1
#### 2. **再次安装modelslim工具**: 
bash install.sh
#### 3. **修复 pad_token (重要!)**
BF16 模型的 tokenizer 默认没有 pad_token，msmodelslim 校准数据对齐时会报错。到BF16权重目录下执行下面脚本，以添加"pad_token"到tokenizer_config.json
```bash
import json, os, sys

path = './tokenizer_config.json'
with open(path) as f:
    cfg = json.load(f)

# 复用 eos_token 作为 pad_token
if 'pad_token' not in cfg or cfg['pad_token'] is None:
    cfg['pad_token'] = cfg.get('eos_token', '</s>')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f'[OK] pad_token set to: {cfg[\"pad_token\"]}')
else:
    print(f'[OK] pad_token already exists: {cfg[\"pad_token\"]}')
```
#### 4. 量化
采用 msmodelslim 默认的 model_adapter 进行量化，安装msmodelslim 后需要准备两个文件，分别为run.sh 以及 minimax_quant.yaml，run.sh 文件指定model_adapter，minimax_quant.yaml 中描述模型量化具体细节，文件参照如下：
- minimax_quant.yaml参照：
```bash
# minimax_quant.yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"      # 激活值按 token 动态缩放，对 W8A8 精度至关重要
          dtype: "int8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_channel"    # 权重按通道缩放
          dtype: "int8"
          symmetric: true
          method: "minmax"
      include: [ "*" ]
      exclude: [
        # 1. 核心路由与偏置保护 (MiniMax 2.5 特有高敏感层)
        "*block_sparse_moe.gate*",
        "*block_sparse_moe.e_score_correction_bias*",

        # 2. MTP (Multi-Token Prediction) 模块保护
        "*mtp_modules*",
        "*mtp_transition*",

        # 3. 注意力归一化层保护 (config中 use_qk_norm: true)
        "*self_attn.q_norm*",
        "*self_attn.k_norm*",

        # 4. 词表与嵌入层保护 (200K超大词表)
        "lm_head",
        "model.embed_tokens"
      ]

  save:
    - type: "ascendv1_saver"
```
- run.sh 参照
```bash
#!/bin/bash
# run.sh

# 环境变量设置 (根据你的 Ascend 环境实际情况调整)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Starting W8A8 Quantization for MiniMax 2.5..."

# 启动 modelslim 量化任务
# 注意：确保 yaml 文件路径和模型路径正确
msmodelslim quant \
    --model_path /opt/data/models/MiniMax-M2.5-BF16/ \
    --save_path /opt/data/models/MiniMax-M2.5_W8A8_v3 \
    --model_type default \
    --config_path ./minimax_quant.yaml \
    --device npu \
    --trust_remote_code True

echo "Quantization finished! Check /opt/data/models/MiniMax-M2.5_W8A8 for the output."
```
- 拷贝相关文件
将下面脚本保存到自己目录下，修改"--input_dir"，"--output_file"，"--index_file"为对应路径并执行.py文件
```bash
import os
import json
import argparse
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="/docker/models/GLM-5")
parser.add_argument("--output_file", type=str, default="/docker/models/GLM-5-w8a8c16/quant_model_weights-e_score.safetensors")
parser.add_argument("--index_file", type=str, default="/docker/models/GLM-5-w8a8c16/quant_model_weights.safetensors.index.json")

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_FILE = args.output_file
INDEX_FILE = args.index_file

collected_tensors = {}

input_path = Path(INPUT_DIR)

if not input_path.exists():
    raise FileNotFoundError(f"{INPUT_DIR} does not exist")

safetensor_files = sorted(input_path.glob("*.safetensors"))

print(f"Found {len(safetensor_files)} safetensor files")

for file in safetensor_files:
    print(f"Reading: {file}")
    tensors = load_file(str(file))

    for key, value in tensors.items():
        if "e_score_correction_bias" in key:
            print(f"  -> Matched key: {key}, shape={tuple(value.shape)}")
            collected_tensors[key] = value.clone()

print(f"\nTotal matched tensors: {len(collected_tensors)}")

if len(collected_tensors) == 0:
    print("WARNING: No matching keys found!")

output_dir = Path(OUTPUT_FILE).parent
output_dir.mkdir(parents=True, exist_ok=True)

save_file(collected_tensors, OUTPUT_FILE)

print(f"\nSaved to: {OUTPUT_FILE}")
print("Done.")

# ===== 更新 index.json =====
with open(INDEX_FILE, "r") as f:
    index = json.load(f)

new_shard_name = Path(OUTPUT_FILE).name

for key in collected_tensors:
    index["weight_map"][key] = new_shard_name

with open(INDEX_FILE, "w") as f:
    json.dump(index, f, indent=2)

print(f"Updated index.json with {len(collected_tensors)} new keys")
```
#### 5. 修改配置
从 BF16 模型复制缺失的辅助文件（tokenizer, modeling 代码等），并将上面生成的W8A8C16权重的config.json文件替换为：
```bash
{
  "architectures": [
    "MiniMaxM2ForCausalLM"
  ],
  "_quantization_config": {
    "quant_method": "ascend",
    "bits": 8,
    "sym": true,
    "group_size": -1,
    "act_quant_method": "per_token",
    "weight_quant_method": "per_channel",
    "modules_to_not_convert": [
      "gate",
      "e_score_correction_bias",
      "q_norm",
      "k_norm",
      "mtp_modules",
      "mtp_transition",
      "lm_head",
      "embed_tokens"
    ]
  },
  "quantization_config": {
    "config_groups": {
        "group_0": {
            "input_activations": {
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
            "targets": ["Linear"],
            "weights": {
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
    "ignore": [
        "e_score_correction_bias",
        "q_norm",
        "k_norm",
        "mtp_modules",
        "mtp_transition",
        "embed_tokens",
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
  "torch_dtype": "bfloat16",
  "attn_type_list": [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
  ],
  "auto_map": {
    "AutoConfig": "configuration_minimax_m2.MiniMaxM2Config",
    "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM"
  },
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 3072,
  "intermediate_size": 1536,
  "max_position_embeddings": 196608,
  "model_type": "minimax_m2",
  "mtp_transformer_layers": 1,
  "num_attention_heads": 48,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 62,
  "num_key_value_heads": 8,
  "num_local_experts": 256,
  "num_mtp_modules": 3,
  "qk_norm_type": "per_layer",
  "rms_norm_eps": 1e-06,
  "rope_theta": 5000000,
  "rotary_dim": 64,
  "scoring_func": "sigmoid",
  "shared_intermediate_size": 0,
  "tie_word_embeddings": false,
  "transformers_version": "4.46.1",
  "use_cache": true,
  "use_mtp": true,
  "use_qk_norm": true,
  "use_routing_bias": true,
  "vocab_size": 200064
}
```
