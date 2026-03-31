import os
import json
from glob import glob
from tqdm import tqdm
import torch
from safetensors.torch import load_file, save_file


input_path = "."
output_path = "."

safetensor_files = list(glob(os.path.join(input_path, "*.safetensors")))
safetensor_files.sort()

new_weight_map = {}
for safetensor_file in tqdm(safetensor_files):
    file_name = os.path.basename(safetensor_file)
    state_dict = load_file(safetensor_file, device="cpu")
    new_state_dict = {}
    for weight_name, weight in state_dict.items():
        if "weight_offset" in weight_name or "rot" in weight_name:
            continue
        new_state_dict[weight_name] = weight
        new_weight_map[weight_name] = file_name
    if new_state_dict:
        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

model_index_file = os.path.join(input_path, "quant_model_weights.safetensors.index.json")
with open(model_index_file, "r") as f:
    model_index = json.load(f)
model_index["weight_map"] = new_weight_map
with open(model_index_file, "w", encoding="utf-8") as f:
    json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)