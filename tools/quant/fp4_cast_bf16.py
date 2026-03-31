import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import math
import torch
from safetensors.torch import load_file, save_file

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def main(args):
    fp4_path = args.input_fp4_hf_path
    bf16_path = args.output_bf16_hf_path
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp4_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    fp4_weight_names = []

    def mxfp4_weight_dequant(fp4_weight, fp4_scale, dtype=torch.bfloat16, rows_per_chunk=16384 * 512):
        blocks = fp4_weight
        scales = fp4_scale.to(torch.int32) - 127
        assert blocks.shape[:-1] == scales.shape, (
            f"{blocks.shape=} does not match {scales.shape=}"
        )
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total   = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp
        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    safetensor_files = list(glob(os.path.join(fp4_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            weight_path = weight_map[weight_name]
            print(weight_name)
            if weight_name.endswith("_scales"):
                weight_map.pop(weight_name)
            elif weight_name.endswith("_blocks"):
                weight_map.pop(weight_name)
                bf16_weight_name = weight_name.replace("_blocks", ".weight")
                fp4_weight = weight
                fp4_scale = current_state_dict[weight_name.replace("_blocks", "_scales")]
                bf16_weight = mxfp4_weight_dequant(fp4_weight, fp4_scale)
                if args.gpt_oss and "mlp.experts" in bf16_weight_name:
                    bf16_weight = bf16_weight.transpose(-2,-1).contiguous()
                new_state_dict[bf16_weight_name] = bf16_weight
                weight_map[bf16_weight_name] = weight_path
                print(bf16_weight_name, new_state_dict[bf16_weight_name].shape, new_state_dict[bf16_weight_name].dtype)
            elif weight_name.endswith("_bias"):
                weight_map.pop(weight_name)
                bf16_weight_name = weight_name.replace("_bias", ".bias")
                new_state_dict[bf16_weight_name] = weight
                weight_map[bf16_weight_name] = weight_path
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            # torch.cuda.empty_cache()
    
    # # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp4-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    parser.add_argument("--gpt-oss", default=False, action="store_true", help="fp4 to bf16 cast for gpt-oss")
    args = parser.parse_args()
    main(args)