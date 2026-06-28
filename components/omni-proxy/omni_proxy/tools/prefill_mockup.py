# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from multiprocessing import Process
import uvicorn
import asyncio
import time
import uuid
import argparse

app = FastAPI()


@app.api_route("/v1/chat/completions", methods=["POST", "GET"])
async def generate(request: Request):
    response_data = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": None,
                "content": "This is prefill node."
            },
            "logprobs": None,
            "finish_reason": "length",
            "stop_reason": None
        }],
        "usage": {
            "prompt_tokens": 114,
            "total_tokens": 115,
            "completion_tokens": 1,
            "prompt_tokens_details": None
        },
        "prompt_logprobs": None,
        "kv_transfer_params": {
            "remote_block_ids": [1],
            "remote_cluster_id": 0,
            "remote_host_ip": "tcp://localhost:5568",
            "spec_token_ids": [],
            "remote_dp_rank": 0,
            "remote_request_id": f"chatcmpl-{uuid.uuid4().hex}",
        }
    }
    return JSONResponse(content=response_data)


def run_server(port):
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start FastAPI server on specified ports')
    parser.add_argument('--ports', nargs='+', type=int, default=[9000],
                       help='List of ports to run servers on (default: [9000])')
    
    args = parser.parse_args()
    ports = args.ports
    
    print(f"Starting servers on ports: {ports}")
    
    processes = []
    for port in ports:
        p = Process(target=run_server, args=(port,))
        p.start()
        processes.append(p)
        print(f"Server started on port {port}")

    for p in processes:
        p.join()