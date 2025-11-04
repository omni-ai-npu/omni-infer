import argparse
import requests
import json
from multiprocessing import Pool
import time
import functools

def parse_args():
    parser = argparse.ArgumentParser(description="Simple curl")

    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--datafile", type=str, required=True)
    parser.add_argument("--input-key", type=str, default='input')
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    return parser, args

def call_one(input, model_name, temperature, max_tokens, input_key, url, headers):
    data = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "include_stop_str_in_output": True,
        "messages":[{"role": "user", "content": input[input_key]}],
        "ignore_eos": True,
        "skip_special_tokens": False,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        return {"Error": response.status_code}


def run_requests(
        max_concurrency, ip, port, datafile, input_key, model_name, temperature, max_tokens):
    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    partial_call_one = functools.partial(
        call_one,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        input_key=input_key,
        url=url,
        headers=headers
    )

    with open(datafile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with Pool(max_concurrency) as pool:
        results = pool.map(partial_call_one, data)
    
    return results

def main():
    parser, args = parse_args()
    results = run_requests(
        max_concurrency=args.max_concurrency,
        ip=args.ip,
        port=args.port,
        datafile=args.datafile,
        input_key=args.input_key,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.output is not None:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f)
        

if __name__ == "__main__":
    main()
