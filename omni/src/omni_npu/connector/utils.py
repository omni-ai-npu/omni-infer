# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import zmq
import torch
import time
import json
import queue
import socket
import threading
import numpy as np

from vllm.config import VllmConfig, ParallelConfig
from vllm.distributed.parallel_state import (
    get_world_group,
    get_pp_group,
    get_pcp_group,
    get_dp_group,
    get_tp_group,
    get_dcp_group,
)
from vllm.logger import init_logger


logger = init_logger(__name__)


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def get_kv_port(vllm_config: VllmConfig) -> int:
    def from_cfg():
        cfg = vllm_config.kv_transfer_config
        return getattr(cfg, "kv_port", None)
    base = from_cfg() or 5568  # default kv port
    return base + get_world_group().local_rank


def get_kv_role(vllm_config: VllmConfig):
    return {
        "kv_producer": True,
        "kv_consumer": False,
    }[vllm_config.kv_transfer_config.kv_role]


def serial_brief(s: list[int]):
    msg, last, head = "[", None, None
    for x in s + [None]:
        if x is None or x - 1 != last:
            if head != last and head is not None:
                msg += f"~{last}"
            if x is not None:
                if head is not None:
                    msg += ", "
                msg += f"{x}"
            head = x
        last = x
    return msg + "]"


def start_daemon(task, feedback=None):
    feedback = feedback or queue.Queue()
    threading.Thread(target=task, daemon=True, args=(feedback,)).start()
    return feedback


def calm_down(case, interval=0.05, tol=0.99):
    if not hasattr(calm_down, "_ts"):
        calm_down._ts = {}
        calm_down._lock = threading.Lock()
    now, case = time.time(), str(case)
    with calm_down._lock:
        dt = now - calm_down._ts.get(case, now - interval)
        calm_down._ts[case] = now
    if dt < interval * tol:
        time.sleep(interval - dt)


class ParallelDesc:

    def __init__(self, config=None, rank=None):
        if isinstance(config, (VllmConfig, ParallelConfig)):
            if isinstance(config, VllmConfig):
                config = config.parallel_config
            self.pp = config.pipeline_parallel_size
            self.dp = config.data_parallel_size
            self.pcp = config.prefill_context_parallel_size
            self.tp = config.tensor_parallel_size
            self.dcp = config.decode_context_parallel_size
        elif isinstance(config, (list, tuple)):
            if len(config) != 5:
                raise ValueError(f"parallel config must have 5 dims, got {len(config)}")
            for it in config:
                if not isinstance(it, int) or it <= 0:
                    raise ValueError(f"parallel dim must be positive int, got {it}")
            self.pp, self.dp, self.pcp, self.tp, self.dcp = tuple(config)
        elif config is None:  # local runtime
            self.pp = get_pp_group().world_size
            self.dp = get_dp_group().world_size
            self.pcp = get_pcp_group().world_size
            self.tp = get_tp_group().world_size
            self.dcp = get_dcp_group().world_size
            rank = get_world_group().rank_in_group
        else:
            raise ValueError("invalid parallel input")

        self.pcp_stride = self.tp
        self.dp_stride = self.pcp * self.pcp_stride
        self.pp_stride = self.dp * self.dp_stride
        self.size = self.pp * self.pp_stride
        if config is None:  # local runtime
            world_size = get_world_group().world_size
            if world_size != self.size:
                raise ValueError(f"world_size {world_size} != {self.size}")
        if rank is not None:
            self.apply_rank(rank)

    def apply_rank(self, rank: int):
        if not (0 <= rank < self.size):
            raise ValueError(f"invalid rank: {rank}, size={self.size}")
        self.rank = rank
        self.pp_rank = (rank // self.pp_stride) % self.pp
        self.dp_rank = (rank // self.dp_stride) % self.dp
        self.pcp_rank = (rank // self.pcp_stride) % self.pcp
        self.tp_rank = rank % self.tp
        self.dcp_rank = rank % self.dcp

    def to_list(self):
        return [self.pp, self.dp, self.pcp, self.tp, self.dcp]


class SimpleServer:  # single-thread

    def __init__(self, addr: str):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.bind(addr)

    def __del__(self):
        self.sock.close(linger=0)
        self.ctx.term()

    def addr(self) -> str:
        return self.sock.getsockopt(zmq.LAST_ENDPOINT).decode()

    def todo(self, loop=False):  # yield req, rep
        def reply(client_id, msg: dict | str):  # str for error
            msg = {"error": str(msg)} if not isinstance(msg, dict) else msg
            self.sock.send_multipart([client_id, json.dumps(msg).encode()])

        while True:  # loop until no more req
            try:  # recv req
                client_id, raw = self.sock.recv_multipart(
                    flags=0 if loop else zmq.NOBLOCK)
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:  # no more req
                    if loop:
                        continue
                    return
                raise  # other error, terminate
            try:  # parse req
                recv = json.loads(raw.decode())
                req_id, req = recv["req_id"], recv["req"]
                if not isinstance(req, dict):
                    raise TypeError("req must be dict")
            except (TypeError, KeyError, json.JSONDecodeError, ValueError):  # bad req
                reply(client_id, "bad_req")
                continue
            try:  # proc req
                rep = {}  # user modify
                yield req, rep
            except:
                reply(client_id, "internal_error")
                raise  # internal error, terminate
            reply(client_id, {"req_id": req_id, "rep": rep})

    def handle(self, loop=None, **op_list):
        def _unknown_op(req):
            return {"err": "unknown op"}

        for req, rep in self.todo(loop):
            op = req.get("op", None)
            op = op_list.get(op, _unknown_op)
            rep.update(op(req))


class SimpleClient:

    def __init__(self, addr: str):
        self.ctx = zmq.Context()
        self.sock_lock = threading.Lock()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.connect(addr)
        self.lock = threading.Lock()
        self.cb = {}  # req_id -> (cb, ts)
        self.id_cnt = 0

    def _clear(self, timeout):
        with self.lock:
            now = time.time()
            timeout_list = [(req_id, cb)
                for req_id, (cb, ts) in self.cb.items()
                if now - ts > timeout]
            for req_id, _ in timeout_list:
                del self.cb[req_id]
            rest = len(self.cb)
        for _, cb in timeout_list:
            cb(None)  # None for timeout
        return rest

    def routine(self, timeout=5.0):
        while True:
            try:  # recv any msg
                with self.sock_lock:
                    if self.sock is None:
                        raise RuntimeError("called after stop()")
                    recv = self.sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    break  # no more msg
                raise e  # other error
            try:  # parse reply
                recv = json.loads(recv)
                req_id, rep = recv["req_id"], recv["rep"]
                if not isinstance(req_id, int) or not isinstance(rep, dict):
                    raise TypeError("reply req_id must be int, rep must be dict")
            except (TypeError, KeyError, json.JSONDecodeError, ValueError):  # bad reply
                continue
            with self.lock:
                cb, _ = self.cb.pop(req_id, (None, None))
            if cb is not None:
                cb(rep)
        return self._clear(timeout)  # -> rest cb

    def flush(self, polling=0.2, timeout=5.0):
        while self.routine(timeout) > 0:
            time.sleep(polling)

    def close(self, linger=0):
        with self.sock_lock:
            if self.sock is not None:
                self.sock.close(linger=linger)
                self.sock = None
        self._clear(0.0)
        self.ctx.term()

    # user side method, safe in multi-thread
    def send_json(self, req: dict, cb=None):
        if not isinstance(req, dict):
            raise TypeError("req must be dict")
        with self.lock:
            req_id = self.id_cnt
            self.id_cnt += 1
            msg = json.dumps({"req_id": req_id, "req": req})
            if callable(cb):
                self.cb[req_id] = (cb, time.time())
        with self.sock_lock:
            if self.sock is not None:
                return self.sock.send_string(msg)
        return self._clear(0.0)  # when sock is None

    def query(self, op: str, cb=None, **kwargs):
        self.send_json(dict(op=op, **kwargs), cb=cb)


class TP_Convertor:

    def __init__(self, remote_tp_size: int):
        tp_group = get_tp_group()
        self.tp_comm = tp_group.device_group
        self.tp_size = tp_group.world_size
        self.tp_rank = tp_group.rank_in_group
        self.transfer_done = False

        # Initialize attributes that will be set in scheme_reorg
        self.tail_blk = None
        self.kv_group = None
        self.send_domain = None
        self.recv_domain = None
        self.recv_num = None
        self.send_split = None
        self.recv_split = None

        self.remote_tp_size = remote_tp_size
        if self.tp_size % remote_tp_size != 0:
            raise ValueError(f"tp_size ({self.tp_size}) must be divisible by remote_tp_size ({remote_tp_size})")
        self.stride = self.tp_size // remote_tp_size
        self.offset = self.tp_rank % self.stride

    def scheme_reorg(self,
        token_num: int,
        tail_blk: int,
        kv_group: list[list[torch.Tensor]],
        local_block_ids: list[int],
        remote_block_ids: list[int],
    ) -> list[int]:  # adjusted remote_blk_ids
        if self.stride == 1:
            return remote_block_ids

        tails = [TP_Convertor.tail_blk_num( # tok_num of tail block
            token_num, i, self.tp_size, self.remote_tp_size,
        ) for i in range(self.tp_size)]     # of all ranks
        before, after = tails[self.tp_rank] # tok_num before/after a2a
        end = max(before, after)

        self.tail_blk = tail_blk
        self.kv_group = kv_group
        self.send_domain = (after, end)
        self.recv_domain = (before, end)
        self.recv_num = end - before

        a2a_map = TP_Convertor.a2a_mapper([a - b for a, b in tails])
        self.send_split = a2a_map[self.tp_rank].tolist()
        self.recv_split = a2a_map[:, self.tp_rank].tolist()

        TP_Convertor.scheduled_list().append(self)

        n_remote = len(remote_block_ids)
        return [
            remote_block_ids[
                (self.offset + i * self.stride) % n_remote
            ] for i, _ in enumerate(local_block_ids)
        ]

    # call after scheme_reorg()
    def token_reorg(self):
        if self.stride == 1:
            return
        for kvs in self.kv_group:
            send = TP_Convertor.extract_kv(kvs, self.tail_blk, self.send_domain)
            recv = send.new_empty(self.recv_num, *send.shape[1:])
            torch.distributed.all_to_all_single(
                recv, send,
                self.recv_split,
                self.send_split,
                self.tp_comm,
            )
            TP_Convertor.store_kv(recv, kvs, self.tail_blk, self.recv_domain)

    # ======================= utils =======================

    @classmethod
    def scheduled_list(cls):
        rank = get_tp_group().rank_in_group
        attr_name = f"scheduled_reorg_{rank}"
        if not hasattr(cls, attr_name):
            setattr(cls, attr_name, [])
        return getattr(cls, attr_name)

    @classmethod
    def do_scheduled_kv_reorg(cls):
        # called by MLA builder
        # len(sched) always be the same in all ranks
        sched: list[TP_Convertor] = cls.scheduled_list()
        if len(sched) == 0:
            return
        fin_num = 0
        for it in sched:
            if not it.transfer_done:
                break
            fin_num += 1

        dev = sched[0].kv_group[0][0].device
        cfg = {"dtype": torch.int32, "device": dev}
        local_num = torch.tensor([fin_num], **cfg)
        global_num = get_tp_group().all_gather(local_num)
        all_fin_num = min(global_num.tolist())
        for _ in range(all_fin_num):
            sched.pop(0).token_reorg()

    @staticmethod
    def tail_blk_num(num, d_rank, d_size, p_size, pg=128):
        if d_size < p_size:
            raise ValueError(f"d_size ({d_size}) must be >= p_size ({p_size})")
        stride = d_size // p_size
        p_num = num // p_size + int(d_rank // stride < num % p_size)
        d_num = num // d_size + int(d_rank < num % d_size)
        before = p_num % (pg * stride) - pg * (d_rank % stride)
        after = d_num % pg
        return min(pg, max(before, 0)), after

    @staticmethod
    def link_to_remote(tp_rank, tp_size, remote_tp_size):
        if tp_size % remote_tp_size != 0:
            raise ValueError(f"{tp_size=} % {remote_tp_size=} != 0")
        return tp_rank // (tp_size // remote_tp_size)

    @staticmethod
    def a2a_mapper(movement: list[int]):
        give, take = [], []
        for i, d in enumerate(movement):
            if d > 0:
                give.append([d, i])
            if d < 0:
                take.append([-d, i])

        num = len(movement)
        a2a_map = np.zeros((num, num), np.int32)
        while len(give) > 0 and len(take) > 0:
            a, b = give[-1], take[-1]
            d = min(a[0], b[0])
            a2a_map[a[1]][b[1]] = d
            if a[0] <= d:
                give.pop()
            else:
                a[0] -= d
            if b[0] <= d:
                take.pop()
            else:
                b[0] -= d

        if len(give) + len(take) != 0:
            raise RuntimeError("a2a_mapper failed: give and take should be empty")
        return a2a_map  # np.int32[num, num]

    @staticmethod
    def extract_kv(
        kvs: list[torch.Tensor],
        blk_i: int,
        domain: tuple[int, int],
    ):
        def frag_of(kv):
            if not isinstance(kv, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(kv)}")
            if kv.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {kv.dim()}D")
            return kv[blk_i][domain[0]:domain[1]]
        return torch.stack(
            [frag_of(it) for it in kvs]
        ).transpose(0, 1).contiguous()  # [T, N, D]

    @staticmethod
    def store_kv(
        x: torch.Tensor,
        kvs: list[torch.Tensor],
        blk_i: int,
        domain: tuple[int, int],
    ):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
        if x.size(0) != max(0, domain[1] - domain[0]):
            raise ValueError(f"x.size(0) mismatch: {x.size(0)} vs {domain[1] - domain[0]}")
        if x.size(1) != len(kvs):
            raise ValueError(f"x.size(1) mismatch: {x.size(1)} vs {len(kvs)}")
        x = x.transpose(0, 1)  # [T, N, D] -> [N, T, D]
        for dst, src in zip(kvs, x):
            if not isinstance(dst, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(dst)}")
            if dst.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {dst.dim()}D")
            if dst.size(2) != x.size(2):
                raise ValueError(f"D size mismatch: {dst.size(2)} vs {x.size(2)}")
            dst[blk_i][domain[0]:domain[1]] = src
