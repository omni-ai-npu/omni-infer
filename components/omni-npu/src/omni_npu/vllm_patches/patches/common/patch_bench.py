# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import re
import sys
import uuid
import json
import time
import random
import asyncio
import aiohttp
import argparse
import traceback
import contextlib
from typing import Any, Literal
from datetime import datetime
from collections.abc import AsyncGenerator, Iterable

import numpy as np
from vllm.utils.import_utils import PlaceholderModule
try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

# import vllm.benchmarks.serve
import vllm
from vllm.benchmarks.datasets import CustomDataset
from vllm.tokenizers import get_tokenizer
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.network_utils import join_host_port
from vllm.benchmarks.lib.ready_checker import wait_for_endpoint
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
    _validate_api_url,
    _get_chat_content,
    _update_payload_common,
    _update_headers_common,
    StreamedResponseHandler,
)
from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
from vllm.benchmarks.serve import BenchmarkMetrics, TaskType, calculate_metrics, calculate_metrics_for_embeddings, get_request, check_goodput_args, save_to_pytorch_benchmark_format

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("BenchEndpointPatch", vllm.benchmarks.lib.endpoint_request_func)
class BenchEndpointPatch(VLLMPatch):
    _attr_names_to_apply = ['async_request_openai_chat_completions']

    async def async_request_openai_chat_completions(
        request_func_input: RequestFuncInput,
        session: aiohttp.ClientSession,
        pbar: tqdm | None = None,
        mm_position: Literal["first", "last"] = "last",
    ) -> RequestFuncOutput:
        api_url = request_func_input.api_url
        _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

        content = _get_chat_content(request_func_input, mm_position=mm_position)

        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        _update_payload_common(payload, request_func_input)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        _update_headers_common(headers, request_func_input)

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    handler = StreamedResponseHandler()
                    async for chunk_bytes in response.content.iter_any():
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        messages = handler.add_chunk(chunk_bytes)
                        for message in messages:
                            # NOTE: SSE comments (often used as pings) start with
                            # a colon. These are not JSON data payload and should
                            # be skipped.
                            if message.startswith(":"):
                                continue

                            chunk = message.removeprefix("data: ")

                            if chunk != "[DONE]":
                                timestamp = time.perf_counter()
                                data = json.loads(chunk)

                                if choices := data.get("choices"):
                                    content = choices[0]["delta"].get("content")
                                    # First token
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.itl.append(timestamp - most_recent_timestamp)

                                    generated_text += content or ""
                                elif usage := data.get("usage"):
                                    output.output_tokens = usage.get("completion_tokens")

                                most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


@register_patch("CustomDatasetPatch", CustomDataset)
class CustomDatasetPatch(VLLMPatch):
    _attr_names_to_apply = ['load_data', 'sample']

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        # self.data will be a list of dictionaries
        # e.g., [{"prompt": "What is the capital of India?"}, ...]
        # This will be the standardized format which load_data()
        # has to convert into depending on the filetype of dataset_path.
        # sample() will assume this standardized format of self.data
        self.data = []

        # Load the JSONL file
        if self.dataset_path.endswith(".jsonl"):
            self.custom_data_type = "jsonl"
            jsonl_data = pd.read_json(path_or_buf=self.dataset_path, lines=True)

            # check if the JSONL file has a 'prompt' column
            if "prompt" not in jsonl_data.columns:
                raise ValueError("JSONL file must contain a 'prompt' column.")

            # Convert each row to a dictionary and append to self.data
            # This will convert the DataFrame to a list of dictionaries
            # where each dictionary corresponds to a row in the DataFrame.
            # This is the standardized format we want for self.data
            for _, row in jsonl_data.iterrows():
                self.data.append(row.to_dict())
        elif self.dataset_path.endswith(".json"):
            self.custom_data_type = "json"
            with open(self.dataset_path, 'r', encoding='utf-8') as file:
                # Load the JSON data
                json_data = json.load(file)

                # Check if the JSON data is a list of dictionaries
                if not isinstance(json_data, list):
                    raise ValueError("JSON file must contain a list of dictionaries.")

                # Check if each dictionary in the list has a 'input' key
                for item in json_data:
                    if not isinstance(item, dict):
                        raise ValueError("Each item in the JSON list must be a dictionary.")
                    item["prompt"] = item.get("input", item.get("prompt", None))
                    if item["prompt"] is None:
                        raise ValueError("Each item in the JSON list must be a dictionary with a 'input' key or a 'prompt' key.")
                    if "prompt_len" in item:
                        item.pop("prompt_len")
            # Append each dictionary to self.data
            self.data.extend(json_data)
        else:
            raise NotImplementedError(
                "Only JSONL format is supported for CustomDataset."
            )

        random.seed(self.random_seed)
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: str | None = None,
        max_loras: int | None = None,
        output_len: int | None = None,
        enable_multimodal_chat: bool = False,
        skip_chat_template: bool = False,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs,
    ) -> list:
        # load all data if needed
        self.num_available_samples = len(self.data)
        if num_requests <= 0:
            num_requests = self.num_available_samples
            print(
                "num_requests is set to 0 or negative, "
                "so using all available samples: %d",
                num_requests,
            )

        sampled_requests = []
        if self.custom_data_type == "jsonl":
            for i, item in enumerate(self.data):
                if len(sampled_requests) >= num_requests:
                    break
                prompt = item["prompt"]

                # apply template
                if not skip_chat_template:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )

                prompt_len = len(tokenizer(prompt).input_ids)
                sampled_requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                        request_id=request_id_prefix + str(i),
                    )
                )
            self.maybe_oversample_requests(
                sampled_requests, num_requests, request_id_prefix, no_oversample
            )
        else:
            for i in range(num_requests):
                if not getattr(self, "disable_shuffle", False): # Sample data from self.data with replacement
                    item = random.choice(self.data)
                else: # Sample data from self.data in order cyclically
                    item = self.data[i % len(self.data)]                    
                prompt = item["prompt"]
                if "prompt_len" in item:
                    prompt_len = item["prompt_len"]
                else:
                    prompt_len = len(tokenizer(prompt).input_ids)
                    item["prompt_len"] = prompt_len # avoid recomputing prompt_len

                output_len = item.get("output_tokens", item.get("output_len", output_len))
                if output_len is not None:
                    try:
                        output_len = int(output_len)
                    except:
                        raise ValueError(f"output_len of prompt: {prompt} must can be transfer to an integer.")
                else:
                    raise ValueError(f"output_len of prompt: {prompt} is None, add output_len in json file or specify via --fixed-output-len.")

                sampled_requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                        request_id=request_id_prefix + str(i),
                    ))

        return sampled_requests


@register_patch("BenchServePatch", vllm.benchmarks.serve)
class BenchServePatch(VLLMPatch):
    _attr_names_to_apply = ['get_http_session', 'process_benchmark_results', 'benchmark', 'worker_process', 'initial_test', 'warm_up', 'start_profile', 'stop_profile', 'add_cli_args', 'main_async']

    def get_http_session(max_concurrency: int, api_url: str) -> aiohttp.ClientSession:
        """
        Create an HTTP session with appropriate settings.
        """
        # Reuses connections across requests to reduce TLS handshake overhead.
        connector = aiohttp.TCPConnector(
            limit=max_concurrency or 0,
            limit_per_host=max_concurrency or 0,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False,
            ssl=("https://" in api_url),
        )

        session = aiohttp.ClientSession(
            connector=connector,
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
        )
        return session

    def process_benchmark_results(
        outputs: list[RequestFuncOutput],
        input_requests: list[SampleRequest],
        task_type: TaskType,
        tokenizer: PreTrainedTokenizerBase,
        benchmark_duration: float,
        request_rate: float,
        selected_percentiles: list[float],
        selected_percentile_metrics: list[str],
        goodput_config_dict: dict[str, float],
        max_concurrency: int | None,
        rps_change_events: list[dict[str, Any]],
    ):
        if task_type == TaskType.GENERATION:
            metrics, actual_output_lens = calculate_metrics(
                input_requests=input_requests,
                outputs=outputs,
                dur_s=benchmark_duration,
                tokenizer=tokenizer,
                selected_percentiles=selected_percentiles,
                goodput_config_dict=goodput_config_dict,
            )
        else:
            metrics = calculate_metrics_for_embeddings(
                outputs=outputs,
                dur_s=benchmark_duration,
                selected_percentiles=selected_percentiles,
            )
            actual_output_lens = 0

        print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
        print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
        print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
        if max_concurrency is not None:
            print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
        if request_rate != float("inf"):
            print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
        if isinstance(metrics, BenchmarkMetrics):
            print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
        print(
            "{:<40} {:<10.2f}".format(
                "Request throughput (req/s):", metrics.request_throughput
            )
        )
        if goodput_config_dict:
            print(
                "{:<40} {:<10.2f}".format(
                    "Request goodput (req/s):", metrics.request_goodput
                )
            )
        if isinstance(metrics, BenchmarkMetrics):
            print(
                "{:<40} {:<10.2f}".format(
                    "Output token throughput (tok/s):", metrics.output_throughput
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Peak concurrent requests:", metrics.max_concurrent_requests
                )
            )
        print(
            "{:<40} {:<10.2f}".format(
                "Total Token throughput (tok/s):", metrics.total_token_throughput
            )
        )

        if isinstance(metrics, BenchmarkMetrics):
            result = {
                "duration": benchmark_duration,
                "completed": metrics.completed,
                "failed": metrics.failed,
                "total_input_tokens": metrics.total_input,
                "total_output_tokens": metrics.total_output,
                "request_throughput": metrics.request_throughput,
                "request_goodput": metrics.request_goodput if goodput_config_dict else None,
                "output_throughput": metrics.output_throughput,
                "total_token_throughput": metrics.total_token_throughput,
                "input_lens": [output.prompt_len for output in outputs],
                "output_lens": actual_output_lens,
                "ttfts": [output.ttft for output in outputs],
                "itls": [output.itl for output in outputs],
                "generated_texts": [output.generated_text for output in outputs],
                "errors": [output.error for output in outputs],
                "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
                "max_concurrent_requests": metrics.max_concurrent_requests,
            }
        else:
            result = {
                "duration": benchmark_duration,
                "completed": metrics.completed,
                "total_input_tokens": metrics.total_input,
                "request_throughput": metrics.request_throughput,
                "total_token_throughput": metrics.total_token_throughput,
                "input_lens": [output.prompt_len for output in outputs],
                "errors": [output.error for output in outputs],
            }

        if rps_change_events:
            result["rps_change_events"] = rps_change_events

        def process_one_metric(
            # E.g., "ttft"
            metric_attribute_name: str,
            # E.g., "TTFT"
            metric_name: str,
            # E.g., "Time to First Token"
            metric_header: str,
        ):
            # This function prints and adds statistics of the specified
            # metric.
            if metric_attribute_name not in selected_percentile_metrics:
                return
            print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
            print(
                "{:<40} {:<10.2f}".format(
                    f"Mean {metric_name} (ms):",
                    getattr(metrics, f"mean_{metric_attribute_name}_ms"),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    f"Median {metric_name} (ms):",
                    getattr(metrics, f"median_{metric_attribute_name}_ms"),
                )
            )
            result[f"mean_{metric_attribute_name}_ms"] = getattr(
                metrics, f"mean_{metric_attribute_name}_ms"
            )
            result[f"median_{metric_attribute_name}_ms"] = getattr(
                metrics, f"median_{metric_attribute_name}_ms"
            )
            result[f"std_{metric_attribute_name}_ms"] = getattr(
                metrics, f"std_{metric_attribute_name}_ms"
            )
            for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
                p_word = str(int(p)) if int(p) == p else str(p)
                print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
                result[f"p{p_word}_{metric_attribute_name}_ms"] = value

        if task_type == TaskType.GENERATION:
            process_one_metric("ttft", "TTFT", "Time to First Token")
            process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
            process_one_metric("itl", "ITL", "Inter-token Latency")
        process_one_metric("e2el", "E2EL", "End-to-end Latency")

        print("=" * 50)
        return result

    async def benchmark(
        request_func,
        api_url: str,
        model_id: str,
        model_name: str,
        input_requests: list[SampleRequest],
        logprobs: int | None,
        request_rate: float,
        burstiness: float,
        disable_tqdm: bool,
        ignore_eos: bool,
        max_concurrency: int | None,
        lora_modules: Iterable[str] | None,
        extra_headers: dict | None,
        extra_body: dict | None,
        ramp_up_strategy: Literal["linear", "exponential"] | None = None,
        ramp_up_start_rps: int | None = None,
        ramp_up_end_rps: int | None = None,
        process_id: int = -1,
        tqdm_bar_q = None,
        bench_start_barrier = None,
    ):
        get_http_session_func = sys.modules['vllm.benchmarks.serve'].get_http_session
        session = get_http_session_func(max_concurrency, api_url)

        if lora_modules:
            # For each input request, choose a LoRA module at random.
            lora_modules = iter(
                [random.choice(lora_modules) for _ in range(len(input_requests))]
            )

        distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

        if ramp_up_strategy is not None:
            print(f"benchmark process <{process_id}>: Traffic ramp-up strategy: {ramp_up_strategy}.")
            print(
                f"benchmark process <{process_id}>: Will increase RPS from {ramp_up_start_rps} to "
                f"{ramp_up_end_rps} RPS over the duration of the benchmark."
            )
        else:
            print(f"benchmark process <{process_id}>: Traffic request rate: {request_rate}")

        print(f"benchmark process <{process_id}>: Burstiness factor: {burstiness} ({distribution})")
        print(f"benchmark process <{process_id}>: Maximum request concurrency: {max_concurrency}")

        pbar = None
        semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )

        async def limited_request_func(request_func_input, session, pbar, process_id, tqdm_bar_q):
            async with semaphore:
                res = await request_func(
                    request_func_input=request_func_input, session=session, pbar=pbar
                )
            if not disable_tqdm:
                tqdm_bar_q.put((process_id, 1))
            return res

        benchmark_start_time = time.perf_counter()
        tasks: list[asyncio.Task] = []

        rps_change_events = []
        last_int_rps = -1
        if ramp_up_strategy is not None and ramp_up_start_rps is not None:
            last_int_rps = ramp_up_start_rps
            rps_change_events.append(
                {
                    "rps": last_int_rps,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        bench_start_barrier.wait()
        async for request, current_request_rate in get_request(
            input_requests,
            request_rate,
            burstiness,
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
        ):
            if ramp_up_strategy is not None:
                current_int_rps = int(current_request_rate)
                if current_int_rps > last_int_rps:
                    timestamp = datetime.now().isoformat()
                    for rps_val in range(last_int_rps + 1, current_int_rps + 1):
                        rps_change_events.append({"rps": rps_val, "timestamp": timestamp})
                    last_int_rps = current_int_rps
            prompt, prompt_len, output_len, mm_content, request_id = (
                request.prompt,
                request.prompt_len,
                request.expected_output_len,
                request.multi_modal_data,
                request.request_id,
            )
            req_model_id, req_model_name = model_id, model_name
            if lora_modules:
                req_lora_module = next(lora_modules)
                req_model_id, req_model_name = req_lora_module, req_lora_module

            request_func_input = RequestFuncInput(
                model=req_model_id,
                model_name=req_model_name,
                prompt=prompt,
                api_url=api_url,
                prompt_len=prompt_len,
                output_len=output_len,
                logprobs=logprobs,
                multi_modal_content=mm_content,
                ignore_eos=ignore_eos,
                extra_headers=extra_headers,
                extra_body=extra_body,
                request_id=request_id,
            )
            tasks.append(
                asyncio.create_task(
                    limited_request_func(
                        request_func_input=request_func_input, session=session, pbar=pbar,
                        process_id=process_id, tqdm_bar_q=tqdm_bar_q
                    )
                )
            )
        outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
        benchmark_end_time = time.perf_counter()

        if not disable_tqdm:
            tqdm_bar_q.put((process_id, None))

        await session.close()
        return outputs, rps_change_events, benchmark_start_time, benchmark_end_time

    def worker_process(
        request_func,
        args: argparse.Namespace,
        api_url: str,
        model_name: str,
        extra_headers,
        extra_body,
        sub_requests,
        sub_max_concurrency,
        sub_request_rate,
        sub_ramp_up_start_rps,
        sub_ramp_up_end_rps,
        benchmark_outputs,
        process_id,
        bench_start_barrier,
        tqdm_bar_q,
    ):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        benchmark_func = sys.modules['vllm.benchmarks.serve'].benchmark
        sub_result = loop.run_until_complete(
            benchmark_func(
                request_func=request_func,
                api_url=api_url,
                model_id=args.model,
                model_name=model_name,
                input_requests=sub_requests,
                logprobs=args.logprobs,
                request_rate=sub_request_rate,
                burstiness=args.burstiness,
                disable_tqdm=args.disable_tqdm,
                ignore_eos=args.ignore_eos,
                max_concurrency=sub_max_concurrency,
                lora_modules=args.lora_modules,
                extra_headers=extra_headers,
                extra_body=extra_body,
                ramp_up_strategy=args.ramp_up_strategy,
                ramp_up_start_rps=sub_ramp_up_start_rps,
                ramp_up_end_rps=sub_ramp_up_end_rps,
                process_id=process_id,
                tqdm_bar_q=tqdm_bar_q,
                bench_start_barrier=bench_start_barrier,
            )
        )
        benchmark_outputs[process_id] = sub_result

    async def initial_test(
        request_func,
        api_url: str,
        test_input: SampleRequest,
        ready_check_timeout_sec: int = 600,
    ):
        get_http_session_func = sys.modules['vllm.benchmarks.serve'].get_http_session
        session = get_http_session_func(1, api_url)
        print("Starting initial single prompt test run...")
        if ready_check_timeout_sec > 0:
            test_output = await wait_for_endpoint(
                request_func,
                test_input,
                session,
                timeout_seconds=ready_check_timeout_sec,
            )
            if not test_output.success:
                raise ValueError(
                    "Initial test run failed - Please make sure benchmark "
                    "arguments are correctly specified. "
                    f"Error: {test_output.error}"
                )
            else:
                print("Initial test run completed. Starting main benchmark run...")
        else:
            print("Skipping endpoint ready check.")

        await session.close()

    async def warm_up(
        request_func,
        api_url: str,
        num_warmups: int,
        disable_tqdm: bool,
        max_concurrency: int,
        test_input: SampleRequest,
    ):
        print(f"Warming up with {num_warmups} requests...")
        get_http_session_func = sys.modules['vllm.benchmarks.serve'].get_http_session
        session = get_http_session_func(max_concurrency, api_url)

        warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
        warmup_semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )
        warmup_tasks = []

        async def warmup_limited_request_func():
            async with warmup_semaphore:
                return await request_func(
                    request_func_input=test_input, session=session, pbar=warmup_pbar
                )

        for _ in range(num_warmups):
            request_task = asyncio.create_task(warmup_limited_request_func())
            warmup_tasks.append(request_task)
        _ = await asyncio.gather(*warmup_tasks)

        if warmup_pbar is not None:
            warmup_pbar.close()
        await session.close()
        print("Warmup run completed.")

    async def start_profile(
        request_func,
        api_url: str,
        base_url: str,
        model_id: str,
        model_name: str,
        logprobs: bool,
        ignore_eos: bool,
        extra_headers: dict | None,
        extra_body: dict | None,
        input_requests: list[SampleRequest],
    ):
        print("Starting profiler...")
        get_http_session_func = sys.modules['vllm.benchmarks.serve'].get_http_session
        session = get_http_session_func(1, api_url)
        test_prompt, test_prompt_len, test_output_len, test_mm_content = (
            input_requests[0].prompt,
            input_requests[0].prompt_len,
            input_requests[0].expected_output_len,
            input_requests[0].multi_modal_data,
        )
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        profile_output = await request_func(request_func_input=profile_input, session=session)
        if profile_output.success:
            print("Profiler started")
        await session.close()

    async def stop_profile(
        request_func,
        api_url: str,
        base_url: str,
        model_id: str,
        logprobs: bool,
        input_requests: list[SampleRequest],
    ):
        print("Stopping profiler...")
        get_http_session_func = sys.modules['vllm.benchmarks.serve'].get_http_session
        session = get_http_session_func(1, api_url)
        test_prompt, test_prompt_len, test_output_len, test_mm_content = (
            input_requests[0].prompt,
            input_requests[0].prompt_len,
            input_requests[0].expected_output_len,
            input_requests[0].multi_modal_data,
        )
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input, session=session)
        if profile_output.success:
            print("Profiler stopped")
        await session.close()

    def add_cli_args(parser: argparse.ArgumentParser):
        add_dataset_parser(parser)
        parser.add_argument(
            "--label",
            type=str,
            default=None,
            help="The label (prefix) of the benchmark results. If not specified, "
            "the value of '--backend' will be used as the label.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="openai",
            choices=list(ASYNC_REQUEST_FUNCS.keys()),
            help="The type of backend or endpoint to use for the benchmark.",
        )
        parser.add_argument(
            "--base-url",
            type=str,
            default=None,
            help="Server or API base url if not using http host and port.",
        )
        # Use 127.0.0.1 here instead of localhost to force the use of ipv4
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument(
            "--endpoint",
            type=str,
            default="/v1/completions",
            help="API endpoint.",
        )
        parser.add_argument(
            "--header",
            metavar="KEY=VALUE",
            nargs="*",
            help="Key-value pairs (e.g, --header x-additional-info=0.3.3) "
            "for headers to be passed with each request. These headers override "
            "per backend constants and values set via environment variable, and "
            "will be overridden by other arguments (such as request ids).",
        )
        parser.add_argument(
            "--max-concurrency",
            type=int,
            default=None,
            help="Maximum number of concurrent requests. This can be used "
            "to help simulate an environment where a higher level component "
            "is enforcing a maximum number of concurrent requests. While the "
            "--request-rate argument controls the rate at which requests are "
            "initiated, this argument will control how many are actually allowed "
            "to execute at a time. This means that when used in combination, the "
            "actual request rate may be lower than specified with --request-rate, "
            "if the server is not processing requests fast enough to keep up.",
        )

        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Name of the model.",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
        )
        parser.add_argument("--use-beam-search", action="store_true")
        parser.add_argument(
            "--logprobs",
            type=int,
            default=None,
            help=(
                "Number of logprobs-per-token to compute & return as part of "
                "the request. If unspecified, then either (1) if beam search "
                "is disabled, no logprobs are computed & a single dummy "
                "logprob is returned for each token; or (2) if beam search "
                "is enabled 1 logprob per token is computed"
            ),
        )
        parser.add_argument(
            "--request-rate",
            type=float,
            default=float("inf"),
            help="Number of requests per second. If this is inf, "
            "then all the requests are sent at time 0. "
            "Otherwise, we use Poisson process or gamma distribution "
            "to synthesize the request arrival times.",
        )
        parser.add_argument(
            "--burstiness",
            type=float,
            default=1.0,
            help="Burstiness factor of the request generation. "
            "Only take effect when request_rate is not inf. "
            "Default value is 1, which follows Poisson process. "
            "Otherwise, the request intervals follow a gamma distribution. "
            "A lower burstiness value (0 < burstiness < 1) results in more "
            "bursty requests. A higher burstiness value (burstiness > 1) "
            "results in a more uniform arrival of requests.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code from huggingface",
        )
        parser.add_argument(
            "--disable-tqdm",
            action="store_true",
            help="Specify to disable tqdm progress bar.",
        )
        parser.add_argument(
            "--num-warmups",
            type=int,
            default=0,
            help="Number of warmup requests.",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with "
            "VLLM_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--save-result",
            action="store_true",
            help="Specify to save benchmark results to a json file",
        )
        parser.add_argument(
            "--save-detailed",
            action="store_true",
            help="When saving the results, whether to include per request "
            "information such as response, error, ttfs, tpots, etc.",
        )
        parser.add_argument(
            "--append-result",
            action="store_true",
            help="Append the benchmark result to the existing json file.",
        )
        parser.add_argument(
            "--metadata",
            metavar="KEY=VALUE",
            nargs="*",
            help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
            "for metadata of this run to be saved in the result JSON file "
            "for record keeping purposes.",
        )
        parser.add_argument(
            "--result-dir",
            type=str,
            default=None,
            help="Specify directory to save benchmark json results."
            "If not specified, results are saved in the current directory.",
        )
        parser.add_argument(
            "--result-filename",
            type=str,
            default=None,
            help="Specify the filename to save benchmark json results."
            "If not specified, results will be saved in "
            "{label}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  # noqa
            " format.",
        )
        parser.add_argument(
            "--ignore-eos",
            action="store_true",
            help="Set ignore_eos flag when sending the benchmark request."
            "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
        )
        parser.add_argument(
            "--percentile-metrics",
            type=str,
            default=None,
            help="Comma-separated list of selected metrics to report percentiles. "
            "This argument specifies the metrics to report percentiles. "
            'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
            'If not specified, defaults to "ttft,tpot,itl" for generative models '
            'and "e2el" for pooling models.',
        )
        parser.add_argument(
            "--metric-percentiles",
            type=str,
            default="99",
            help="Comma-separated list of percentiles for selected metrics. "
            'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
            'Default value is "99".'
            'Use "--percentile-metrics" to select metrics.',
        )
        parser.add_argument(
            "--range-metrics",
            type=str,
            default="0%,100%",
            help="Calculate metrics for requests within a specified time range. The "
            "parameter format is \"start%,end%\", where start and end must be integers <= 100 "
            "with start < end. This indicates that only requests initiated between the first start% "
            "and first end% of the benchmark will be considered for metric calculation. "
            "The default values for start and end are 0 and 100, meaning that by default, "
            "metrics are calculated for all requests throughout the entire benchmark duration."
        )
        parser.add_argument(
            "--goodput",
            nargs="+",
            required=False,
            help='Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is in "
            'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
            "separated by spaces. Allowed request level metric names are "
            '"ttft", "tpot", "e2el". For more context on the definition of '
            "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
            "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
        )
        parser.add_argument(
            "--request-id-prefix",
            type=str,
            required=False,
            default=f"bench-{uuid.uuid4().hex[:8]}-",
            help="Specify the prefix of request id.",
        )
        parser.add_argument(
            "--num-processes",
            type=int,
            default=1,
            help="Number of processes to send requests.",
        )

        sampling_group = parser.add_argument_group("sampling parameters")
        sampling_group.add_argument(
            "--top-p",
            type=float,
            default=None,
            help="Top-p sampling parameter. Only has effect on openai-compatible backends.",
        )
        sampling_group.add_argument(
            "--top-k",
            type=int,
            default=None,
            help="Top-k sampling parameter. Only has effect on openai-compatible backends.",
        )
        sampling_group.add_argument(
            "--min-p",
            type=float,
            default=None,
            help="Min-p sampling parameter. Only has effect on openai-compatible backends.",
        )
        sampling_group.add_argument(
            "--temperature",
            type=float,
            default=None,
            help="Temperature sampling parameter. Only has effect on "
            "openai-compatible backends. If not specified, default to greedy "
            "decoding (i.e. temperature==0.0).",
        )
        sampling_group.add_argument(
            "--frequency-penalty",
            type=float,
            default=None,
            help="Frequency penalty sampling parameter. Only has effect on "
            "openai-compatible backends.",
        )
        sampling_group.add_argument(
            "--presence-penalty",
            type=float,
            default=None,
            help="Presence penalty sampling parameter. Only has effect on "
            "openai-compatible backends.",
        )
        sampling_group.add_argument(
            "--repetition-penalty",
            type=float,
            default=None,
            help="Repetition penalty sampling parameter. Only has effect on "
            "openai-compatible backends.",
        )

        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default="auto",
            choices=["auto", "slow", "mistral", "custom"],
            help='The tokenizer mode.\n\n* "auto" will use the '
            'fast tokenizer if available.\n* "slow" will '
            "always use the slow tokenizer. \n* "
            '"mistral" will always use the `mistral_common` tokenizer. \n*'
            '"custom" will use --tokenizer to select the preregistered tokenizer.',
        )

        parser.add_argument(
            "--served-model-name",
            type=str,
            default=None,
            help="The model name used in the API. "
            "If not specified, the model name will be the "
            "same as the `--model` argument. ",
        )

        parser.add_argument(
            "--lora-modules",
            nargs="+",
            default=None,
            help="A subset of LoRA module names passed in when "
            "launching the server. For each request, the "
            "script chooses a LoRA module at random.",
        )

        parser.add_argument(
            "--ramp-up-strategy",
            type=str,
            default=None,
            choices=["linear", "exponential"],
            help="The ramp-up strategy. This would be used to "
            "ramp up the request rate from initial RPS to final "
            "RPS rate (specified by --ramp-up-start-rps and "
            "--ramp-up-end-rps.) over the duration of the benchmark.",
        )
        parser.add_argument(
            "--ramp-up-start-rps",
            type=int,
            default=None,
            help="The starting request rate for ramp-up (RPS). "
            "Needs to be specified when --ramp-up-strategy is used.",
        )
        parser.add_argument(
            "--ramp-up-end-rps",
            type=int,
            default=None,
            help="The ending request rate for ramp-up (RPS). "
            "Needs to be specified when --ramp-up-strategy is used.",
        )
        parser.add_argument(
            "--ready-check-timeout-sec",
            type=int,
            default=600,
            help="Maximum time to wait for the endpoint to become ready "
            "in seconds (default: 600 seconds / 10 minutes). If set to 0, "
            "the ready check will be skipped.",
        )

        parser.add_argument(
            "--extra-body",
            help="A JSON string representing extra body parameters to include "
            "in each request."
            'Example: \'{"chat_template_kwargs":{"enable_thinking":false}}\'',
            type=json.loads,
            default=None,
        )

    async def main_async(args: argparse.Namespace) -> dict[str, Any]:
        print(args)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Validate ramp-up arguments
        if args.ramp_up_strategy is not None:
            if args.request_rate != float("inf"):
                raise ValueError(
                    "When using ramp-up, do not specify --request-rate. "
                    "The request rate will be controlled by ramp-up parameters. "
                    "Please remove the --request-rate argument."
                )
            if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
                raise ValueError(
                    "When using --ramp-up-strategy, both --ramp-up-start-rps and "
                    "--ramp-up-end-rps must be specified"
                )
            if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
                raise ValueError("Ramp-up start and end RPS must be non-negative")
            if args.ramp_up_start_rps > args.ramp_up_end_rps:
                raise ValueError("Ramp-up start RPS must be less than end RPS")
            if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
                raise ValueError("For exponential ramp-up, start RPS cannot be 0.")
            if args.ramp_up_start_rps < args.num_processes:
                raise ValueError(
                    "When using --ramp-up-strategy, make sure --ramp-up-end-rps >= --num-processes"
                )

        # Validate range-metrics arguments
        start_range, end_range = 0, 0
        pattern = r'^(\d+)%,(\d+)%$'
        match = re.match(pattern, args.range_metrics)
        if not match:
            raise ValueError(
                "Invalid --range-metrics format. Expected format: 'start%,end%', start and end must be integers <= 100 and >=0, with a < b.")
        else:
            start_range = int(match.group(1))
            end_range = int(match.group(2))
            if not (0 <= start_range and start_range <= 100 and 0 <= end_range and end_range <= 100):
                raise ValueError(
                    "Invalid --range-metrics format. Expected format: 'start%,end%', start and end must be integers <= 100 and >=0, with a < b.")

            if start_range >= end_range:
                raise ValueError(
                    "Invalid --range-metrics format. Expected format: 'start%,end%', start and end must be integers <= 100 and >=0, with a < b.")

        label = args.label
        model_id = args.model
        model_name = args.served_model_name if args.served_model_name else args.model
        tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
        tokenizer_mode = args.tokenizer_mode

        if args.base_url is not None:
            api_url = f"{args.base_url}{args.endpoint}"
            base_url = f"{args.base_url}"
        else:
            host_port = join_host_port(args.host, args.port)
            api_url = f"http://{host_port}{args.endpoint}"
            base_url = f"http://{host_port}"

        # Headers
        headers = None
        if args.header:
            headers = {}
            for item in args.header:
                if "=" in item:
                    kvstring = item.split("=", 1)
                    headers[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError("Invalid header format. Please use KEY=VALUE format.")

        tokenizer = get_tokenizer(
            tokenizer_id,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=args.trust_remote_code,
        )

        if args.dataset_name is None:
            raise ValueError(
                "Please specify '--dataset-name' and the corresponding "
                "'--dataset-path' if required."
            )

        # when using random datasets, default to ignoring EOS
        # so generation runs to the requested length
        if (
            args.dataset_name in ("random", "random-mm")
            and args.backend in OPENAI_COMPATIBLE_BACKENDS
        ):
            args.ignore_eos = True

        # Load the dataset.
        input_requests = get_samples(args, tokenizer)
        goodput_config_dict = check_goodput_args(args)

        backend = args.backend
        task_type = (
            TaskType.POOLING
            if "embeddings" in backend or "rerank" in backend
            else TaskType.GENERATION
        )

        # Collect the sampling parameters.
        if task_type == TaskType.GENERATION:
            sampling_params = {
                k: v
                for k, v in {
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "temperature": args.temperature,
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty,
                    "repetition_penalty": args.repetition_penalty,
                }.items()
                if v is not None
            }

            # Sampling parameters are only supported by openai-compatible backend.
            if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
                raise ValueError(
                    "Sampling parameters are only supported by openai-compatible backends."
                )

            if "temperature" not in sampling_params:
                sampling_params["temperature"] = 0.0  # Default to greedy decoding.

            default_percentile_metrics = "ttft,tpot,itl"
        else:
            sampling_params = {}
            default_percentile_metrics = "e2el"

        extra_body = args.extra_body or {}
        extra_body = {**sampling_params, **extra_body}

        percentile_metrics: str = args.percentile_metrics or default_percentile_metrics

        # Avoid GC processing "static" data - reduce pause times.
        freeze_gc_heap()

        # Get request function
        try:
            ASYNC_REQUEST_FUNCS["openai-chat"] = sys.modules['vllm.benchmarks.lib.endpoint_request_func'].async_request_openai_chat_completions
            request_func = ASYNC_REQUEST_FUNCS[backend]
        except KeyError:
            raise ValueError(f"Unknown backend: {backend}") from None

        # Get test_input for initial test and warm up
        test_prompt, test_prompt_len, test_output_len, test_mm_content = (
            input_requests[0].prompt,
            input_requests[0].prompt_len,
            input_requests[0].expected_output_len,
            input_requests[0].multi_modal_data,
        )

        assert (
            test_mm_content is None
            or isinstance(test_mm_content, dict)
            or (
                isinstance(test_mm_content, list)
                and all(isinstance(item, dict) for item in test_mm_content)
            )
        ), "multi_modal_data must be a dict or list[dict]"
        test_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=args.logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=args.ignore_eos,
            extra_headers=headers,
            extra_body=extra_body,
        )

        # Run initial test
        initial_test_func = sys.modules['vllm.benchmarks.serve'].initial_test
        await initial_test_func(request_func, api_url, test_input, args.ready_check_timeout_sec
        )

        # warm up
        if args.num_warmups > 0:
            warm_up_func = sys.modules['vllm.benchmarks.serve'].warm_up
            await warm_up_func(request_func, api_url, args.num_warmups, args.disable_tqdm,
                        args.max_concurrency, test_input)

        # Start profiler if needed
        if args.profile:
            start_profile_func = sys.modules['vllm.benchmarks.serve'].start_profile
            await start_profile_func(request_func, api_url, base_url, model_id, model_name,
                args.logprobs, args.ignore_eos, headers, extra_body, input_requests)

        print(f"Starting multi-process benchmark with {args.num_processes} processes.")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Allocate the input_requests, max_concurrency, request_rate, ramp_up_start_rps, and
        # ramp_up_end_rps that each benchmark process should have.
        chunk_size = len(input_requests) // args.num_processes
        sub_input_requests_list = [
            input_requests[i*chunk_size : (i+1)*chunk_size]
            for i in range(args.num_processes)
        ]
        if len(input_requests) % args.num_processes != 0:
            sub_input_requests_list[-1].extend(input_requests[args.num_processes * chunk_size:])

        if args.max_concurrency is None:
            sub_max_concurrency_list = [None] * args.num_processes
        else:
            chunk_size = args.max_concurrency // args.num_processes
            sub_max_concurrency_list = [chunk_size] * (args.num_processes - 1)
            sub_max_concurrency_list.append(args.max_concurrency - chunk_size * (args.num_processes - 1))

        if args.request_rate == float("inf"):
            sub_request_rate_list = [float("inf")] * args.num_processes
        else:
            # Allocate the request_rate for each process in proportion to the number of requests of
            # each process in the sub_input_requests_list.
            sub_request_rate_list = []
            for x in sub_input_requests_list:
                sub_request_rate_list.append(args.request_rate * len(x) / len(input_requests))

        if args.ramp_up_start_rps is None:
            sub_ramp_up_start_rps_list = [None] * args.num_processes
        else:
            chunk_size = args.ramp_up_start_rps // args.num_processes
            sub_ramp_up_start_rps_list = [chunk_size] * (args.num_processes - 1)
            sub_ramp_up_start_rps_list.append(args.ramp_up_start_rps - chunk_size * (args.num_processes - 1))
            sub_ramp_up_start_rps_list = sorted(sub_ramp_up_start_rps_list)

        if args.ramp_up_end_rps is None:
            sub_ramp_up_end_rps_list = [None] * args.num_processes
        else:
            chunk_size = args.ramp_up_end_rps // args.num_processes
            sub_ramp_up_end_rps_list = [chunk_size] * (args.num_processes - 1)
            sub_ramp_up_end_rps_list.append(args.ramp_up_end_rps - chunk_size * (args.num_processes - 1))
            sub_ramp_up_end_rps_list = sorted(sub_ramp_up_end_rps_list)

        import multiprocessing
        from multiprocessing import Manager, Queue
        manager = Manager()
        tqdm_bar_q = Queue()
        bench_start_barrier = manager.Barrier(args.num_processes + 1)
        benchmark_outputs = manager.dict()
        process_list = []

        worker_process_func = sys.modules['vllm.benchmarks.serve'].worker_process
        for i in range(args.num_processes):
            p = multiprocessing.Process(
                target=worker_process_func,
                args=(request_func, args, api_url, model_name, headers, extra_body, sub_input_requests_list[i],
                    sub_max_concurrency_list[i], sub_request_rate_list[i], sub_ramp_up_start_rps_list[i],
                    sub_ramp_up_end_rps_list[i], benchmark_outputs, i, bench_start_barrier, tqdm_bar_q
                )
            )
            process_list.append(p)
            p.start()

        bench_start_barrier.wait()
        finished_tasks = 0
        progress_bars = []
        if not args.disable_tqdm:
            for i in range(args.num_processes):
                bar = tqdm(
                    total=len(sub_input_requests_list[i]),
                    desc=f"benchmark process <{i}>",
                    position=i,
                    leave=True,
                    delay=1,
                )
                progress_bars.append(bar)
            while finished_tasks < args.num_processes:
                if not tqdm_bar_q.empty():
                    process_id, delta = tqdm_bar_q.get()
                    if delta is None:
                        finished_tasks += 1
                        progress_bars[process_id].close()
                    else:
                        progress_bars[process_id].update(delta)

        for i, p in enumerate(process_list):
            p.join()

        # Stop profiler if needed
        if args.profile:
            stop_profile_func = sys.modules['vllm.benchmarks.serve'].stop_profile
            await stop_profile_func(request_func, api_url, base_url, model_id,
                            args.logprobs, input_requests)

        all_outputs = []
        all_rps_change_events = []
        benchmark_start_time_list = []
        benchmark_end_time_list = []
        for i in range(args.num_processes):
            res = benchmark_outputs[i]
            all_outputs.extend(res[0])
            if res[1]:
                all_rps_change_events.extend(res[1])
            benchmark_start_time_list.append(res[2])
            benchmark_end_time_list.append(res[3])

        benchmark_duration = max(benchmark_end_time_list) - min(benchmark_start_time_list)
        benchmark_clip_start_time = (start_range / 100) * benchmark_duration + min(benchmark_start_time_list)
        benchmark_clip_end_time = (end_range / 100) * benchmark_duration + min(benchmark_start_time_list)
        benchmark_duration = benchmark_clip_end_time - benchmark_clip_start_time
        tmp_all_outputs = []
        tmp_input_request_idxs = []
        tmp_input_requests = []
        for idx, output in enumerate(all_outputs):
            if output.start_time >= benchmark_clip_start_time and output.start_time <= benchmark_clip_end_time:
                tmp_all_outputs.append(output)
                tmp_input_request_idxs.append(idx)
        for tmp_input_request_idx in tmp_input_request_idxs:
            tmp_input_requests.append(input_requests[tmp_input_request_idx])
        all_outputs = tmp_all_outputs
        input_requests = tmp_input_requests

        # Calculate metrics and display results
        process_benchmark_results_func = sys.modules['vllm.benchmarks.serve'].process_benchmark_results
        benchmark_result = process_benchmark_results_func(
            outputs=all_outputs,
            input_requests=input_requests,
            task_type=task_type,
            tokenizer=tokenizer,
            benchmark_duration=benchmark_duration,
            request_rate=args.request_rate,
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            selected_percentile_metrics=percentile_metrics.split(","),
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            rps_change_events=all_rps_change_events,
        )

        # Save config and results to json
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["endpoint_type"] = args.backend  # for backward compatibility
        result_json["backend"] = args.backend
        result_json["label"] = label
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts
        result_json["num_processes"] = args.num_processes

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=", 1)
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        if args.ramp_up_strategy is not None:
            result_json["ramp_up_strategy"] = args.ramp_up_strategy
            result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
            result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        if not args.save_detailed:
            # Remove fields with too many data points
            for field in [
                "input_lens",
                "output_lens",
                "ttfts",
                "itls",
                "generated_texts",
                "errors",
            ]:
                if field in result_json:
                    del result_json[field]
                if field in benchmark_result:
                    del benchmark_result[field]

            # Save to file
        if args.save_result or args.append_result:
            base_model_id = model_id.split("/")[-1]
            max_concurrency_str = (
                f"-concurrency{args.max_concurrency}"
                if args.max_concurrency is not None
                else ""
            )
            label = label or args.backend
            if args.ramp_up_strategy is not None:
                file_name = f"{label}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
            else:
                file_name = f"{label}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
            if args.result_filename:
                file_name = args.result_filename
            if args.result_dir:
                os.makedirs(args.result_dir, exist_ok=True)
                file_name = os.path.join(args.result_dir, file_name)
            with open(
                file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
            ) as outfile:
                # Append a newline.
                if args.append_result and outfile.tell() != 0:
                    outfile.write("\n")
                json.dump(result_json, outfile)
            save_to_pytorch_benchmark_format(args, result_json, file_name)

        return result_json
