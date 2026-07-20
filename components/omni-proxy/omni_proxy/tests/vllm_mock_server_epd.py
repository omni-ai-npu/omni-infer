# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import argparse
import asyncio
import json

import logging
import multiprocessing
import time
import uuid
import zmq
import zmq.asyncio
import msgspec
import random
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

import ctypes, random, time
import numpy as np
import zlib
import base64

import random, time, ctypes, logging, msgspec
# from vllm.distributed.kv_events import (
#     KVEventBatch,
#     BlockStored,
#     BlockRemoved,
#     AllBlocksCleared,
# )

MODEL_LAYER_NUM = 46
TOPK_NUM = 8

## 每个进程最多只支持400并发的情况下能保证实时性
# ==========================================================
# Python 对应的 msgspec.Struct（字段完全对齐 C 结构体）
# ==========================================================
class BlockStored(msgspec.Struct, array_like=False):
    block_hashes: List[int]
    block_hashes_count: int
    parent_block_hash: int
    token_ids: List[int]
    token_ids_count: int
    block_size: int
    lora_id: int


class BlockRemoved(msgspec.Struct, array_like=False):
    block_hashes: List[int]
    block_hashes_count: int


class KVCacheEvent(msgspec.Struct, array_like=False):
    type: int
    data: dict  # {"block_stored": {...}} 或 {"block_removed": {...}}


class KVEventBatch(msgspec.Struct, array_like=False):
    ts: float
    events: List[KVCacheEvent]
    events_count: int


class AllBlocksCleared(msgspec.Struct, array_like=False):
    pass


# ======================================================
# ZMQ 事件发布器
# ======================================================
class ZMQDirectPublisher:
    """模仿 vLLM 的 ZMQ+msgspec 发布行为"""

    def __init__(self, bind_addr: str, topic: str = "kv_events"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(bind_addr)
        self.topic = topic
        self.topic_bytes = topic.encode()
        self._seq = 0
        self._pack = msgspec.msgpack.Encoder()
        logging.info(f"[ZMQPublisher] Bound to {bind_addr}")

    def _next_seq(self):
        self._seq += 1
        return self._seq

    def publish(self, event_batch: KVEventBatch):
        """发送符合 C 端 parse_kv_event_batch() 的 multipart 编码"""
        try:
            seq = self._next_seq()
            payload = self._pack.encode(event_batch)
            seq_bytes = seq.to_bytes(8, "big")
            self.socket.send_multipart((self.topic_bytes, seq_bytes, payload))
            logging.info(
                f"[ZMQPublisher] seq={seq}, {event_batch.events_count} events, size={len(payload)}B"
            )
        except Exception as e:
            logging.error(f"[ZMQPublisher] Failed to send event: {e}")

    def close(self):
        self.socket.close()
        self.context.term()

class OmniCompatiblePublisher:
    def __init__(self, zmq_socket, topic_bytes: bytes, encoder=None):
        self.socket = zmq_socket
        self.topic_bytes = topic_bytes
        self._seq = 0
        self.encoder = encoder or msgspec.msgpack.Encoder()

    def _next_seq(self):
        self._seq += 1
        return self._seq

    def publish_as_omni(self, batch: KVEventBatch):
        """将 vLLM 的 KVEventBatch 转为 [ts, [[Tag, {...}], ...]] 数组"""
        omni_events = []

        for ev in batch.events:
            if isinstance(ev, BlockStored):
                tag = "BlockStored"
                data = msgspec.structs.asdict(ev)
            elif isinstance(ev, BlockRemoved):
                tag = "BlockRemoved"
                data = msgspec.structs.asdict(ev)
            elif isinstance(ev, AllBlocksCleared):
                tag = "AllBlocksCleared"
                data = {}
            else:
                continue

            omni_events.append([tag, data])

        payload_array = [batch.ts, omni_events]
        payload = self.encoder.encode(payload_array)

        seq = self._next_seq()
        seq_bytes = seq.to_bytes(8, "big")
        self.socket.send_multipart((self.topic_bytes, seq_bytes, payload))

        logging.info(
            f"[OmniPublisher] Published {len(omni_events)} event(s), seq={seq}, bytes={len(payload)}"
        )

import threading
import queue
import asyncio
import time
import logging
import uuid
import csv
import atexit
import signal
import os

class PrefillBatchExecutor:
    """
    Prefill 批处理模拟执行器（线程安全、带日志导出）
    ----------------------------------------------
    - 使用同步队列聚合请求；
    - 每次批处理记录请求入队时间、批开始与结束时间；
    - 退出时自动写出 CSV；
    - 支持 Ctrl+C / SIGTERM 安全收尾。
    """

    def __init__(self, max_batch_size=8, base_time_per_req=0.05, wait_time=0.02, executor_id=None, exec_time_multiplier=1.0):
        # ==============================
        # 基本参数
        # ==============================
        self.queue = queue.Queue()
        self.max_batch_size = max_batch_size
        self.base_time_per_req = base_time_per_req
        self.wait_time = wait_time
        self.executor_id = executor_id or f"prefill-{uuid.uuid4().hex[:8]}"
        self.exec_time_multiplier = exec_time_multiplier
        self.loop = None

        # ==============================
        # 运行状态
        # ==============================
        self._running = threading.Event()
        self._thread = None
        self.records = []  # 批处理日志
        self._csv_path = f"prefill_executor_{self.executor_id}.csv"
        self._record_lock = threading.Lock()

        # ==============================
        # 注册退出与信号回调
        # ==============================
        atexit.register(self._dump_csv)
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

        logging.info(f"[PrefillBatchExecutor] created ID={self.executor_id}")

    # ============================================================
    # 生命周期控制
    # ============================================================

    def start(self, loop: asyncio.AbstractEventLoop):
        """启动后台批处理线程"""
        if self._thread is None or not self._thread.is_alive():
            self.loop = loop
            self._running.set()
            self._thread = threading.Thread(target=self._batch_loop, daemon=True)
            self._thread.start()
            logging.info(f"[PrefillBatchExecutor-{self.executor_id}] started background thread")

    def stop(self):
        """安全停止"""
        logging.warning(f"[PrefillBatchExecutor-{self.executor_id}] stopping ...")
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1)
        self._dump_csv()

    def _handle_exit(self, signum, frame):
        """捕获信号，自动写出 CSV"""
        logging.warning(f"[PrefillBatchExecutor-{self.executor_id}] Caught signal {signum}, dumping CSV...")
        try:
            self._dump_csv()
        except Exception as e:
            logging.error(f"[PrefillBatchExecutor-{self.executor_id}] dump_csv on signal failed: {e}")
        finally:
            self._running.clear()

    # ============================================================
    # 请求接口
    # ============================================================

    def enqueue(self, req_id, msg):
        """主线程调用：请求入队"""
        fut = self.loop.create_future()
        enqueue_time = time.time()
        logging.info(f"[enqueue-{self.executor_id}] req={req_id} enqueued at {enqueue_time:.4f}")
        self.queue.put({
            "id": req_id,
            "msg": msg,
            "future": fut,
            "enqueue_time": enqueue_time,
        })
        return fut

    # ============================================================
    # 主循环逻辑
    # ============================================================

    def _batch_loop(self):
        """同步死循环线程"""
        logging.info(f"[PrefillBatchExecutor-{self.executor_id}] batch loop running...")
        while self._running.is_set():
            try:
                # 阻塞等待至少一个请求
                req = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [req]
            collect_start = time.time()

            # 收集更多请求（时间窗口内）
            while time.time() - collect_start < self.wait_time and len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    time.sleep(0.001)
                    continue

            # 执行模拟
            exec_start = time.time()

            batch_size = len(batch)
            exec_time_map = [0, 300, 400, 500]
            if batch_size < len(exec_time_map):
                exec_time = exec_time_map[batch_size]
            else:
                exec_time = exec_time_map[-1] + (batch_size - len(exec_time_map) + 1) * 100

            # Apply multiplier to the original exec_time
            exec_time = exec_time * self.exec_time_multiplier

            #jitter = random.uniform(-5, 15)
            #exec_time += jitter

            logging.info(
                f"[PrefillBatchExecutor-{self.executor_id}] 执行 batch={len(batch)}, 耗时≈{exec_time:.3f}s"
            )
            time.sleep(exec_time/1000)
            exec_end = time.time()

            # 记录日志（线程安全）
            with self._record_lock:
                for r in batch:
                    record = {
                        "RequestID": r["id"],
                        "P_NODE": self.executor_id,
                        "PD api server get request": f"{r['enqueue_time']:.4f}",
                        "Prefill start execute_model": f"{exec_start:.4f}",
                        "Prefill done execute_model": f"{exec_end:.4f}",
                        "Start to send output in prefill stage" : f"{exec_end:.4f}",
                        "Finish prefill pickle and start response" : f"{exec_end:.4f}",
                    }
                    self.records.append(record)

            # Future 回调回主事件循环
            for r in batch:
                fut = r["future"]
                if not fut.done():
                    self.loop.call_soon_threadsafe(
                        fut.set_result,
                        {
                            "status": "done",
                            "batch_size": len(batch),
                            "exec_time": round(exec_end - exec_start, 3),
                        },
                    )

            # ✅ 周期性落盘（防止意外丢数据）
            if len(self.records) % 100 == 0:
                self._dump_csv(partial=True)

        logging.info(f"[PrefillBatchExecutor-{self.executor_id}] loop stopped.")

    # ============================================================
    # CSV 导出逻辑
    # ============================================================
    def _dump_csv(self, partial=False):
        """保存日志为 CSV 文件（支持增量写入）"""
        if not self.records:
            return

        try:
            with self._record_lock:
                file_exists = os.path.exists(self._csv_path)
                write_header = not file_exists

                # ✅ 追加写入模式
                with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "RequestID",
                        "P_NODE",
                        "PD api server get request",
                        "Prefill start execute_model",
                        "Prefill done execute_model",
                        "Start to send output in prefill stage",
                        "Finish prefill pickle and start response",
                    ])

                    # 首次创建才写 header
                    if write_header:
                        writer.writeheader()

                    writer.writerows(self.records)

                count = len(self.records)
                self.records.clear()  # ✅ 清空缓存，防止重复写入

            logging.info(
                f"[PrefillBatchExecutor-{self.executor_id}] CSV {'(partial)' if partial else '(final)'} "
                f"append -> {self._csv_path} (+{count} records)"
            )

        except Exception as e:
            logging.error(f"[PrefillBatchExecutor-{self.executor_id}] Failed to save CSV: {e}")
            
# ======================================================
# vLLM 模拟节点
# ======================================================
class VLLMNode:
    def __init__(self, role: str, port: int, model_path: str, kv_offset: int, pd_policy: str,
                 max_batch_size: int = 16, base_time_per_req: float = 0.05, export_expert_id=False,
                 prefill_exec_time_multiplier: float = 1.0, decode_delay_ms: int = 0):
        self.role = role
        self.port = port
        self.model_path = model_path
        self.kv_port = port + kv_offset
        self.pd_policy = pd_policy
        self.start_time = time.time()

        #ctx = zmq.Context()
        #sock = ctx.socket(zmq.PUB)
        #sock.bind(f"tcp://0.0.0.0:{self.kv_port}")
        #topic = "kv_events".encode()
        #self.omni_publisher = OmniCompatiblePublisher(sock, topic)
        self.kv_publisher = ZMQDirectPublisher(f"tcp://0.0.0.0:{self.kv_port}")
        self.app = self._create_app()
        if self.role == "prefill":
            self.prefill_executor = PrefillBatchExecutor(
                max_batch_size=8, base_time_per_req=0.05,
                executor_id=port, exec_time_multiplier=prefill_exec_time_multiplier
            )
        self.export_expert_id = export_expert_id  # 新增功能开关
        self.decode_delay_ms = decode_delay_ms  # E2E 测试用：让 decode mock 在响应前 sleep 触发 read_timeout
        logging.info(f"[VLLMNode] {role} started on {port}, kv_port={self.kv_port}")

    def generate_3d_list(self, dim1, dim2, dim3):
        """
        生成一个指定维度大小的三维列表，使用0到100之间的随机整数填充。

        参数:
        - dim1: 第一维的大小
        - dim2: 第二维的大小
        - dim3: 第三维的大小

        返回:
        - 一个根据给定维度大小生成的三维列表
        """
        # 使用列表推导式生成三维列表
        #start_ts = time.time_ns()
        #expert_id_list = [[[random.randint(0, 100) for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]
        total = dim1 * dim2 * dim3
        expert_id_list = (np.arange(total, dtype=np.int16) % 384).reshape(dim1, dim2, dim3)
        #end_ts = time.time_ns()
        #diff = (end_ts - start_ts) / 1000
        #print(f"gen list cost time: {diff} us")
        return expert_id_list

    def add_ndarray_info_to_dict(self, data: np.ndarray, routed_experts_dict: dict):
        shape = data.shape
        dtype = str(data.dtype)
        raw_bytes = data.tobytes()
        compressed = zlib.compress(raw_bytes, level=1)
        b64_data = base64.b64encode(compressed).decode('utf-8')
        print(f"prefill_expert_id {len(b64_data)=}")
        routed_experts_dict["routed_experts_shape"] = shape
        routed_experts_dict["routed_experts_dtype"] = dtype
        routed_experts_dict["routed_experts_str_len"] = len(b64_data)
        routed_experts_dict["routed_experts_str"] = b64_data

    def _create_app(self):
        app = FastAPI(title=f"Mock vLLM {self.role}")

        @app.get("/v1/health")
        async def health():
            return {
                "status": "ok",
                "role": self.role,
                "port": self.port,
                "kv_port": self.kv_port,
                "pd_policy": self.pd_policy,
                "uptime_sec": round(time.time() - self.start_time, 2),
            }

        @app.get("/omni_proxy/health")
        async def omni_proxy_health():
            return JSONResponse({"status": "ok"}, status_code=200)

        @app.get("/health")
        async def health_simple():
            return JSONResponse({"status": "ok"}, status_code=200)
        
        @app.post("/v1/completions")
        @app.post("/v1/chat/completions")
        async def create_chat_completion(request: Request):
            try:
                body = await request.json()
            except json.JSONDecodeError:
                body = {}

            # 基础请求信息
            request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            stream_mode = body.get("stream", False)
            messages = body.get("messages", [])
            message_text = messages[0].get("content", "") if messages else ""
            decode_interval_us = int(body.get("decode_interval_us", 10000))
            max_tokens = int(body.get("max_tokens", 10))
            print(f"{max_tokens=}")
            model = body.get("model", "deepseek")

            server_host, server_port = request.scope['server']

            prefill_tokens = len(message_text) // 2
            prefill_delay_us = len(message_text) * 0  # 每字符 10μs

            current_time = time.time()
            base_id = f"chatcmpl-{request_id[:12]}"
            if self.export_expert_id:
                kv_transfer_params = body.get("kv_transfer_params", {})
                #if kv_transfer_params:
                    #expert_id_list = kv_transfer_params.get("prefill_experts_id", "")


            if self.role == "prefill":
                stream_mode = False
                await self.prefill_executor.enqueue(request_id, message_text)

            if self.role == "encode":
                stream_mode = False

            #return JSONResponse(
            #    status_code=403,
            #    content={"error": "Forbidden", "message": "Access denied for this request."}
            #)
            
            if stream_mode:
                async def event_generator():
                    print(f"[{time.strftime('%H:%M:%S')}] [Request {request_id}] stream start, "
                        f"{prefill_tokens=}, {max_tokens=}")
                    # decode 阶段延迟（E2E 测试用：mock 慢响应触发 read_timeout）
                    if self.role == "decode" and self.decode_delay_ms > 0:
                        print(f"[{time.strftime('%H:%M:%S')}] [Request {request_id}] decode_delay {self.decode_delay_ms} ms")
                        await asyncio.sleep(self.decode_delay_ms / 1000.0)
                    # prefill 阶段延迟
                    if prefill_delay_us > 0:
                        print(f"[{time.strftime('%H:%M:%S')}] [Request {request_id}] prefill_delay {prefill_delay_us} us")
                        await asyncio.sleep(prefill_delay_us / 1_000_000.0)

                    prompt_tokens = prefill_tokens
                    reasoning_tokens = 0
                    prompt_tokens_details = None

                    for t in range(max_tokens):
                        completion_tokens = t + 1
                        total_tokens = prompt_tokens + completion_tokens
                        token_timestamp = time.time()

                        chunk_obj = {
                            "id": base_id,
                            "object": "chat.completion.chunk",
                            "created": int(token_timestamp),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant" if t == 0 else None,
                                        "content": f"tok{t}",
                                        "reasoning_content": None,
                                        "tool_calls": None
                                    },
                                    "logprobs": None,
                                    "finish_reason": None,
                                    "matched_stop": None
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "total_tokens": total_tokens,
                                "completion_tokens": completion_tokens,
                                "prompt_tokens_details": prompt_tokens_details,
                                "reasoning_tokens": reasoning_tokens
                            }
                        }
                        
                        # 添加 routed_experts 字段（当开关打开时）
                        if self.export_expert_id:
                            if t == 0:
                                #chunk_obj["routed_experts"] = self.generate_3d_list(1, MODEL_LAYER_NUM, TOPK_NUM)
                                if kv_transfer_params:
                                    #expert_id_list = kv_transfer_params.get("prefill_experts_id", "")
                                    required_keys = [
                                        "routed_experts_shape",
                                        "routed_experts_dtype",
                                        "routed_experts_str_len",
                                        "routed_experts_str"
                                    ]

                                    if all(key in kv_transfer_params for key in required_keys):
                                        chunk_obj["routed_experts"] = {
                                            key: kv_transfer_params[key]
                                            for key in required_keys
                                        }
                            else:
                                arr = self.generate_3d_list(1, MODEL_LAYER_NUM, TOPK_NUM)
                                expert_info_dict = {}
                                self.add_ndarray_info_to_dict(arr, expert_info_dict)
                                chunk_obj["routed_experts"] = expert_info_dict

                        # 打印 token 进度 + 时间戳
                        if (total_tokens % 500 == 0) or (total_tokens == max_tokens):
                            print(f"[{time.strftime('%H:%M:%S')}] [Request {request_id}] "
                                f"stream token {t+1}/{max_tokens}, "
                                f"timestamp={token_timestamp:.6f}, total_tokens={total_tokens}")
                                
                        yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                        if decode_interval_us > 0:
                            await asyncio.sleep(decode_interval_us / 1_000_000.0)

                    # 最后一个 chunk
                    final_chunk = {
                        "id": base_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": None,
                                    "content": "",
                                    "reasoning_content": None,
                                    "tool_calls": None
                                },
                                "logprobs": None,
                                "finish_reason": "length",
                                "matched_stop": None
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "total_tokens": prompt_tokens + max_tokens,
                            "completion_tokens": max_tokens,
                            "prompt_tokens_details": prompt_tokens_details,
                            "reasoning_tokens": reasoning_tokens
                        }
                    }
                    if self.export_expert_id:
                        arr = self.generate_3d_list(1, MODEL_LAYER_NUM, TOPK_NUM)
                        expert_info_dict = {}
                        self.add_ndarray_info_to_dict(arr, expert_info_dict)
                        final_chunk["routed_experts"] = expert_info_dict

                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

                    print(f"[{time.strftime('%H:%M:%S')}] [Request {request_id}] stream completed, "
                        f"total_tokens={prompt_tokens + max_tokens}")

                    yield "data: [DONE]\n\n"

                return StreamingResponse(event_generator(), media_type="text/event-stream")

            else:
                start_ts = time.time_ns()
                # decode 阶段延迟（非流式路径同样支持，E2E 测试触发 read_timeout）
                if self.role == "decode" and self.decode_delay_ms > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] [Request {request_id}] decode_delay {self.decode_delay_ms} ms")
                    await asyncio.sleep(self.decode_delay_ms / 1000.0)
                # Non-stream: 总延迟
                #total_sleep = (prefill_delay_us + decode_interval_us * max_tokens * 0) / 1_000_000.0
                total_sleep = 0
                if total_sleep > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] [Request {request_id}] non-stream total_sleep {total_sleep:.6f}s")
                    await asyncio.sleep(total_sleep)

                full_obj = {
                    "id": base_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(f"tok{i}" for i in range(max_tokens))
                            },
                            "finish_reason": "length"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prefill_tokens,
                        "completion_tokens": max_tokens,
                        "total_tokens": prefill_tokens + max_tokens
                    }
                }
                if self.role == "prefill":
                    full_obj["kv_transfer_params"] = {"remote_block_ids":[1], "remote_cluster_id": 0, "remote_host_ip":f"tcp://127.0.0.1:{self.port}"}
                    full_obj["trace_obj"] = None
                    if self.export_expert_id:
                        arr = self.generate_3d_list(prefill_tokens, MODEL_LAYER_NUM, TOPK_NUM)
                        self.add_ndarray_info_to_dict(arr, full_obj["kv_transfer_params"])
                        print(f"{full_obj['kv_transfer_params']=}")
                elif self.role == "encode":
                    full_obj.pop("choices", None)
                    full_obj["ec_transfer_params"] = {
                        "remote_host_ip": f"tcp://127.0.0.1:{self.port}",
                        "mm_hash": f"hash_{request_id[:8]}",
                        "mm_hash_data": f"data_{request_id[:8]}",
                    }
                    full_obj["usage"] = {
                        "prompt_tokens": prefill_tokens,
                        "completion_tokens": 1,
                        "total_tokens": prefill_tokens + 1
                    }
                    full_obj["message"] = message_text
                else:
                    if self.export_expert_id:
                        print(f"{time.strftime('%H:%M:%S')}, {prefill_tokens=}")
                        arr = self.generate_3d_list(prefill_tokens + max_tokens, MODEL_LAYER_NUM, TOPK_NUM)
                        expert_info_dict = {}
                        self.add_ndarray_info_to_dict(arr, expert_info_dict)
                        full_obj["routed_experts"] = expert_info_dict
                        print(f"{full_obj['routed_experts']=}")

                print(f"[{time.strftime('%H:%M:%S')}] [Request {request_id}] non-stream completed, "
                    f"total_tokens={prefill_tokens + max_tokens}")

                # Add traceparent headers for prefill role to trigger update_traceparent_header in omni-proxy
                if self.role == "prefill":
                    trace_headers = {
                        "Traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
                        "Start_time_ns": "1234567890"
                    }
                    js_resp = JSONResponse(content=full_obj, headers=trace_headers)
                else:
                    js_resp = JSONResponse(content=full_obj)
                end_ts = time.time_ns()
                diff = (end_ts - start_ts) / 1000
                print(f"full_obj finish cost time: {diff} us")
                return js_resp
        
        
        return app
    
    async def _simulate_kv_events(self, request_id: str):
        ts = time.time()
        tag = random.choice(["BlockStored"])
        #tag = random.choice(["BlockStored", "BlockRemoved", "AllBlocksCleared"])

        if tag == "BlockStored":
            ev = BlockStored(
                block_hashes=[random.getrandbits(63) for _ in range(3)],
                parent_block_hash=random.getrandbits(63),
                token_ids=[random.randint(1, 5000) for _ in range(3)],
                block_size=random.choice([2048, 4096, 8192]),
                lora_id=random.randint(0, 10),
            )
        elif tag == "BlockRemoved":
            ev = BlockRemoved(block_hashes=[random.getrandbits(63) for _ in range(3)])
        else:
            ev = AllBlocksCleared()

        batch = KVEventBatch(ts=ts, events=[ev])
        self.omni_publisher.publish_as_omni(batch)

        logging.info(f"[{self.role}] Published KVEvent {tag} for req {request_id}")


    async def _simulate_kv_events(self, request_id: str):
        """模拟 vLLM 旧版 array 格式 KVEventBatch"""
        ts = time.time()
        events = []

        #event_type = random.choice(["store", "remove", "clear"])
        event_type = random.choice(["store"])

        if event_type == "store":
            block_hashes = [random.getrandbits(64) for _ in range(random.randint(1, 4))]
            token_ids = [random.randint(1, 10000) for _ in range(random.randint(1, 6))]
            event = [
                "BlockStored",
                block_hashes,
                random.getrandbits(64),   # parent_block_hash
                token_ids,
                random.choice([2048, 4096, 8192]),  # block_size
                random.randint(0, 5),  # lora_id
                "GPU",
                "mock_lora",
                random.randint(0, 3),
            ]
            events.append(event)

        elif event_type == "remove":
            block_hashes = [random.getrandbits(64) for _ in range(random.randint(1, 3))]
            event = ["BlockRemoved", block_hashes, "GPU", random.randint(0, 3)]
            events.append(event)

        else:
            event = ["AllBlocksCleared"]
            events.append(event)

        # 构造符合 nginx 预期的批次
        payload = [ts, events]

        # msgpack 编码
        encoded = msgspec.msgpack.encode(payload)

        # multipart 发送
        seq = random.randint(1, 1 << 30)
        seq_bytes = seq.to_bytes(8, "big")

        self.kv_publisher.socket.send_multipart(
            [self.kv_publisher.topic_bytes, seq_bytes, encoded]
        )

        logging.info(
            f"[{self.role}] Published legacy KVEvent: {event_type}, len={len(encoded)} bytes, ts={ts:.6f}"
        )
    async def run(self):
        if self.role == "prefill":
            loop = asyncio.get_running_loop()  # 🧠 获取主线程事件循环
            self.prefill_executor.start(loop)  # 传入 loop

        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


# ======================================================
# 主程序入口
# ======================================================
def set_cpu_affinity(worker_id: int):
    """
    将当前进程绑定到指定的 CPU 核心。
    绑核策略：worker_id % cpu_count
    """
    try:
        # 获取系统总 CPU 核心数（逻辑核）
        cpu_count = os.cpu_count()
        if cpu_count is None:
            print(f"[WARN] Cannot detect CPU count, skip affinity setting.")
            return

        # 计算目标核心编号（循环分配）
        target_core = worker_id % cpu_count
        # sched_setaffinity 需要传入可迭代的核心列表
        os.sched_setaffinity(0, [target_core])  # pid=0 表示当前进程

        current_pid = os.getpid()
        print(f"[INFO] Worker-{worker_id} (PID={current_pid}) bound to CPU core {target_core}")
    except AttributeError:
        # Windows 或旧版 Python 不支持
        print(f"[WARN] CPU affinity not supported on this platform.")
    except OSError as e:
        print(f"[ERROR] Failed to set CPU affinity for worker {worker_id}: {e}")

def run_node(role, port, model_path, kv_offset, pd_policy, worker_id=0, export_expert_id=False, prefill_exec_time_multiplier=1.0, decode_delay_ms=0):
    #set_cpu_affinity(worker_id)
    node = VLLMNode(role, port, model_path, kv_offset, pd_policy, export_expert_id=export_expert_id,
                    prefill_exec_time_multiplier=prefill_exec_time_multiplier,
                    decode_delay_ms=decode_delay_ms)
    asyncio.run(node.run())

def run_encode_node(role, port, model_path, kv_offset, pd_policy, worker_id=0, export_expert_id=False, prefill_exec_time_multiplier=1.0, decode_delay_ms=0):
    node = VLLMNode(role, port, model_path, kv_offset, pd_policy, export_expert_id=export_expert_id,
                    prefill_exec_time_multiplier=prefill_exec_time_multiplier,
                    decode_delay_ms=decode_delay_ms)
    asyncio.run(node.run())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode-endpoints", type=str, default="127.0.0.1:17001,127.0.0.1:17002,127.0.0.1:17003,127.0.0.1:17004")
    parser.add_argument("--prefill-endpoints", type=str, default="127.0.0.1:18001,127.0.0.1:18002,127.0.0.1:18003,127.0.0.1:18004,127.0.0.1:18005,127.0.0.1:18006,127.0.0.1:18007,127.0.0.1:18008,127.0.0.1:18009,127.0.0.1:18010,127.0.0.1:18011,127.0.0.1:18012")
    parser.add_argument("--decode-endpoints", type=str, default="127.0.0.1:19001,127.0.0.1:19002,127.0.0.1:19003,127.0.0.1:19004,127.0.0.1:19005,127.0.0.1:19006,127.0.0.1:19007,127.0.0.1:19008,127.0.0.1:19009,127.0.0.1:19010,127.0.0.1:19011,127.0.0.1:19012")
    parser.add_argument("--omni-proxy-model-path", type=str, default="/data/models/QwQ-32B")
    parser.add_argument("--omni_proxy_kv_port_offset", type=int, default=100)
    parser.add_argument("--omni-proxy-pd-policy", type=str, default="sequential")
    # 新增参数：功能开关
    parser.add_argument("--export-expert-id", action="store_true", default=False, help="Enable export of routed_experts")
    parser.add_argument("--prefill-exec-time-multiplier", type=float, default=1.0,
                        help="Prefill execution time multiplier (default 1.0, set to 5.0 for overflow test)")
    parser.add_argument("--decode-delay-ms", type=int, default=0,
                        help="When role == 'decode', delay response by N ms before sending first byte. "
                             "Used by E2E tests to exercise the omni_proxy main upstream read_timeout.")
    args = parser.parse_args()
    all_procs = []

    for ep in args.encode_endpoints.split(","):
        if not ep:
            continue
        _, port = ep.split(":")
        p = multiprocessing.Process(target=run_encode_node,
                                    args=("encode", int(port),
                                          args.omni_proxy_model_path,
                                          args.omni_proxy_kv_port_offset,
                                          args.omni_proxy_pd_policy,
                                          len(all_procs),
                                          args.export_expert_id,
                                          1.0,
                                          args.decode_delay_ms))
        p.start()
        all_procs.append(p)

    for ep in args.prefill_endpoints.split(","):
        _, port = ep.split(":")
        p = multiprocessing.Process(target=run_node,
                                    args=("prefill", int(port),
                                          args.omni_proxy_model_path,
                                          args.omni_proxy_kv_port_offset,
                                          args.omni_proxy_pd_policy,
                                          len(all_procs),
                                          args.export_expert_id,
                                          args.prefill_exec_time_multiplier,
                                          args.decode_delay_ms))
        p.start()
        all_procs.append(p)

    for ep in args.decode_endpoints.split(","):
        _, port = ep.split(":")
        p = multiprocessing.Process(target=run_node,
                                    args=("decode", int(port),
                                          args.omni_proxy_model_path,
                                          args.omni_proxy_kv_port_offset,
                                          args.omni_proxy_pd_policy,
                                          len(all_procs),
                                          args.export_expert_id,
                                          1.0,  # decode doesn't need multiplier
                                          args.decode_delay_ms))
        p.start()
        all_procs.append(p)

    print(f"[Main] Started {len(all_procs)} mock vLLM instances.")

    try:
        for p in all_procs:
            p.join()
    except KeyboardInterrupt:
        for p in all_procs:
            p.terminate()
        print("[Main] All processes terminated.")
