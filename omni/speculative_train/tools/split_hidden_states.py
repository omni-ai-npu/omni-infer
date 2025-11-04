import os
import sys
import time
import torch

def main(input_file, output_dir):
    data = torch.load(input_file)

    for i, item in enumerate(data):
        output_file = os.path.join(output_dir, f"{i:08d}.pt")
        pos = (item['input_ids'] == 151667).to(torch.int32).argmax()
        item['loss_mask'] = torch.zeros_like(item['input_ids'])
        item['loss_mask'][pos + 2:] = 1
        torch.save(item, output_file)
        print(f"{output_file} saved!", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])