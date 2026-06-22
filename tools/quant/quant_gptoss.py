# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import optiquant.gpt_oss_int8 as qint8
import optiquant.faquant as faquant
from argparse import ArgumentParser
import json
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True, help="bf16 weight path")
    parser.add_argument("--output-path", type=str, required=True, help="quantized weight path")
    parser.add_argument("--device", type=str, required=True, help="support cpu and npu")
    parser.add_argument("--file_count", type=int, default=0, help="File count when loading model")
    parser.add_argument("--model-type", choices=["120b", "20b"], required=True, help="support gpt-oss-120b and gpt-oss-20b")

    args = parser.parse_args()

    qint8.main(args, args.input_bf16_hf_path, args.output_path, args.model_type)
    num_bits = 8

    quant_config = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int"
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": {
                        "self_attn": 16,
                        "mlp.router": 16,
                        "mlp.experts": 8
                    },
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int"
                }
            }
        },
        "format": "int-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }

    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["quantization_config"] = quant_config

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
