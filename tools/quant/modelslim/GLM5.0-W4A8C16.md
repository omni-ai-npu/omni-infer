# GLM5.0-W4A8C16量化方式

## 操作步骤
1. 进入[容器](../README.md#环境配置)
2. 将transformers升级到最新版
3. **再次安装modelslim工具**: bash install.sh
4. 将量化[配置文件](./quant_config/glm5_w4a8c16.yaml)拷贝到自己的目录下
5. 执行量化命令
`msmodelslim quant --model_path {bf16权重路径} --save_path {量化权重路径} --device npu --model_type GLM-5 --config_path {配置文件路径}`
6. 运行下面的ascendv1_to_omni_format.py，以转换为omniinfer所需格式
```bash
# ascendv1_to_omni_format.py

import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import torch
try:
    import torch_npu
except:
    pass
from safetensors.torch import load_file, save_file


def main(args, bf16_path, int8_path, model_name="deepseek-ai/DeepSeek-R1"):

    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(int8_path, exist_ok=True)
    model_index_file = os.path.join(bf16_path, "quant_model_weights.safetensors.index.json")
    new_model_index_file = os.path.join(int8_path, "quant_model_weights.safetensors.index.json")

    
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
   
    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    if args.file_count:
        safetensor_files = safetensor_files[:args.file_count]

    new_weight_map = {}

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)

        state_dict = load_file(safetensor_file, device=args.device)
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            if "weight_offset" in weight_name:
                print(weight_name, "drop!")
                continue
            elif "scale_bias" in weight_name:
                new_weight_name = weight_name.replace("scale_bias", "weight_bias")
                print(new_weight_name, weight.dtype)
                weight = weight.sum(dim=1, keepdim=True)
                weight = weight.view(weight.numel())
                new_state_dict[new_weight_name] = weight
                new_weight_map[new_weight_name] = file_name
            elif "weight_scale" in weight_name:
                if weight_name.replace("weight_scale", "scale_bias") in weight_map:
                    bestScale = weight.to(torch.float32).view(torch.int32)
                    bestScaleInt64 = torch.zeros(bestScale.shape, dtype=torch.int64).view(torch.int32).reshape(-1, 2)
                    bestScaleInt64[:, 0] = bestScale.reshape(-1)
                    bestScaleInt64 = bestScaleInt64.view(torch.int64).reshape(bestScale.shape)
                    bestScaleInt64 = bestScaleInt64.view(1, -1)
                    new_weight_name = weight_name.replace("weight_scale", "weight_int4_scale")
                    print(new_weight_name, bestScaleInt64.dtype)
                    new_state_dict[new_weight_name] = bestScaleInt64
                    new_weight_map[new_weight_name] = file_name
                else:
                    print(weight_name, weight.dtype)
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
            elif ".offset" in weight_name:
                print(weight_name, "drop!")
                continue
            else:
                print(weight_name, weight.dtype)
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
                continue

        new_safetensor_file = os.path.join(int8_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

    # modify model.safetensors.index.json
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_weight_map
    with open(new_model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"model.safetensors.index.json modified and saved to {model_index_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument('--file_count', type=int, default=0, help="Layer count when loading model")

    args = parser.parse_args()
    main(args, args.input_path, args.output_path)
    print("done")
```
执行
```bash
python ascendv1_to_omni_format.py \
--input_path {w4a8量化权重路径} \
--output_path {转换完格式后的量化权重路径} \
--device npu
```
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
    }
```