import os
import sys
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Simple curl")

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--key-token", type=int, required=True)
    parser.add_argument("--mask-delta", type=int, default=0)

    args = parser.parse_args()
    return parser, args

def main(data_dir, key_token, mask_delta):
    files = os.listdir(data_dir)
    for i, file in enumerate(files):
        filepath = os.path.join(data_dir, file)
        data = torch.load(filepath)
        pos = (data['input_ids'] == key_token).to(torch.int32).argmax()
        data['loss_mask'] = torch.zeros_like(data['input_ids'])
        data['loss_mask'][pos + mask_delta:] = 1
        torch.save(data, filepath)

        print(f"{filepath} rewrited! {i + 1} / {len(files)}", flush=True)

    print("Done!")

if __name__ == "__main__":
    parser, args = parse_args()
    main(args.data_dir, args.key_token, args.mask_delta)