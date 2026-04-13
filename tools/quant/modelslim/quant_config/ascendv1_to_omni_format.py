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