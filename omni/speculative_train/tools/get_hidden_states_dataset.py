import argparse
import requests
import json
from multiprocessing import Pool
from transformers import AutoTokenizer
import datasets
import functools
from tdqm import tdqm

def parse_args():
    parser = argparse.ArgumentParser(description="Construct offline traning dataset from dataset like sharegpt")

    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--send-length", type=int, default=1024)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--overlap-length", type=int, default=32)

    args = parser.parse_args()
    return parser, args

def call_one(
    item,
    model_name, url, headers,
    tokenizer, send_length, threshold, overlap_length,
):
    messages = []
    for one in item["conversations"]:
        role = one['from']
        content = one['value']
        if role == 'gpt':
            role = 'assistant'
        elif role == 'human':
            role = 'user'
        messages.append({'role': role, 'content': content})
    tokens = tokenizer.apply_chat_template(messages)

    ntokens = len(tokens)
    results = []
    delta = send_length - overlap_length
    for j in range(0, ntokens, delta):
        prompt = tokens[j:j + send_length]
        if len(prompt) > threshold:
            data = {
                "model": model_name,
                "max_tokens": 2,
                "include_stop_str_in_output": True,
                "prompt": prompt,
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                results.append(response.json())
            else:
                results.append({"Error": response.status_code})

    return results

def run_requests(
    max_concurrency, ip, port, model_name,
    dataset, send_length, threshold, overlap_length, tokenizer,
):
    url = f"http://{ip}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    partial_call_one = functools.partial(
        call_one,
        model_name=model_name, url=url, headers=headers,
        tokenizer=tokenizer, send_length=send_length, threshold=threshold, overlap_length=overlap_length,
    )

    with Pool(max_concurrency) as pool:
        results = list(tdqm(pool.imap(partial_call_one, dataset))))
    
    return results

def main():
    parser, args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = datasets.load_dataset(args.dataset, split='train')

    results = run_requests(
        max_concurrency=args.max_concurrency,
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        dataset=dataset,
        send_length=args.send_length,
        threshold=args.threshold,
        overlap_length=args.overlap_length,
        tokenizer=tokenizer,
    )

    if args.output is not None:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f)

if __name__ == "__main__":
    main()
