# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from safetensors import safe_open
from safetensors.torch import save_file
import os
import json

new_tensors = {}
input_path = "."

with safe_open("rot.safetensors", framework="pt") as f:
    for k in f.keys():
        tensor = f.get_tensor(k)
        if k == "rot.weight":
            new_tensors["model.layers.78.rot.weight"] = tensor
        else:
            new_tensors[k] = tensor

save_file(new_tensors, "rot.safetensors")

model_index_file = os.path.join(input_path, "quant_model_weights.safetensors.index.json")
with open(model_index_file, "r") as f:
    model_index = json.load(f)

weight_map = model_index["weight_map"]
if "rot.weight" in weight_map:
    weight_map["model.layers.78.rot.weight"] = weight_map.pop("rot.weight")

model_index["weight_map"] = weight_map
with open(model_index_file, "w", encoding="utf-8") as f:
    json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)