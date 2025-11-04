import argparse
import requests
import json
from multiprocessing import Pool
import os
import torch
import functools

def parse_args():
    parser = argparse.ArgumentParser(description="Simple curl")

    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    return parser, args


def call_one(file, input_dir, model_name, url, headers):
    input = torch.load(os.path.join(input_dir, file))
    data = {
        "model": model_name,
        "max_tokens": 2,
        "include_stop_str_in_output": True,
        "prompt": input["prompt_token_ids"] + input['output_token_ids']
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        return {"Error": response.status_code}

def run_requests(
        max_concurrency, ip, port, input_dir, model_name):
    url = f"http://{ip}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    partial_call_one = functools.partial(call_one, input_dir=input_dir, model_name=model_name, url=url, headers=headers)

    files = os.listdir(input_dir)
    
    with Pool(max_concurrency) as pool:
        results = pool.map(partial_call_one, files)
    
    return results

def main():
    parser, args = parse_args()
    results = run_requests(
        max_concurrency=args.max_concurrency,
        ip=args.ip,
        port=args.port,
        input_dir=args.input_dir,
        model_name=args.model_name,
    )
    if args.output is not None:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f)
        

if __name__ == "__main__":
    main()

