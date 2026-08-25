# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import time
import queue
import threading
import itertools
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.v1.request import Request, RequestStatus
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.core.kv_cache_manager import KVCacheBlock
from vllm.logger import init_logger

from .llmdatadist_manager_v1 import CacheManager
from .utils import (
    ParallelDesc,
    SimpleServer,
    SimpleClient,
    get_local_ip,
    get_kv_port,
    get_kv_role,
    start_daemon,
    calm_down,
)
from omni_npu.diagnostics.metrics import kv_transfer


logger = init_logger(__name__)


class Metadata(KVConnectorMetadata):
    def __init__(self, scheduler_addr=None):
        self.scheduler_addr = scheduler_addr
        self.req_params: dict[str, dict] = {}  # req_id -> params


class LLMDataDistConnector(KVConnectorBase_V1, SupportsHMA):

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        self.is_prefill = is_prefill = get_kv_role(vllm_config)
        self._connector_metadata = None
        if role == KVConnectorRole.SCHEDULER:
            cls = PrefillScheduler if is_prefill else DecodeScheduler
            self.scheduler = cls(vllm_config)
            self.worker = None
        else:
            cls = PrefillWorker if is_prefill else DecodeWorker
            self.worker = cls(vllm_config)
            self.scheduler = None

    # ================= SupportsHMA overrides =================

    def request_finished_all_groups(self, request, block_ids):
        # Hybrid KV, block_ids will be in multi groups
        return self.request_finished(request, block_ids)

    # ================= Scheduler overrides =================

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        def parse_block(b: KVCacheBlock) -> int:
            not_skip = self.is_prefill or b.block_hash is None
            return b.block_id if not_skip else -1
        block_ids = [  # -1 for decode hits prefix cache
            [parse_block(b) for b in group]
            for group in blocks.blocks]
        return self.scheduler.update_state_after_alloc(request, block_ids)

    def build_connector_meta(self, scheduler_output):
        return self.scheduler.build_connector_metadata()

    def request_finished(self, request, block_ids: list[int] | tuple[list[int]]):
        if isinstance(block_ids, tuple):  # hybrid kv
            block_ids = [group for group in block_ids]
        else:
            block_ids = [block_ids]  # only one group
        return self.scheduler.request_finished(request, block_ids)

    def update_connector_output(self, connector_output):
        if isinstance(self.scheduler, PrefillScheduler):
            self.scheduler.update_connector_output(connector_output)

    def get_finished_count(self):
        # Prefill pull_done is reported by a single TP rank (send barrier).
        return self.get_finished_send_count()

    def get_finished_send_count(self):
        # prefill only requires one of ranks to report finished_sending
        return 1 if self.is_prefill else None

    def get_finished_recv_count(self):
        # None => Aggregator uses world_size (needed with Offloading loads).
        return None

    # ================= Worker overrides =================

    def register_kv_caches(self, kv_caches):
        self.worker.manager.weakup()
        self.worker.manager.register(kv_caches)

    def unregister_kv_caches(self):
        self.worker.manager.sleep()

    def get_finished(self, finished_req_ids):
        if not isinstance(self._connector_metadata, Metadata):
            raise TypeError("_connector_metadata must be Metadata")
        return self.worker.get_finished(self._connector_metadata)

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.worker is None:
            return set()
        return self.worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context, **kwargs):
        if not isinstance(self._connector_metadata, Metadata):
            raise TypeError("_connector_metadata must be Metadata")
        self.worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    # ============ C 线指标（走 KVConnectorStats 通道，无 multiprocess）========

    def get_kv_connector_stats(self):
        # model-runner 每步调用，汇总本周期 KV 失败 + worker 显存采样（空则 None）。
        # 仅 worker 角色传 rank 采显存：scheduler 角色跑在 EngineCore 进程，那儿的显存无意义。
        rank = None
        if self.worker is not None:
            if getattr(self, "_rank", None) is None:
                try:
                    self._rank = get_world_group().rank_in_group
                except Exception:  # noqa: BLE001 - 取不到 rank 就不采显存，不影响失败上报
                    self._rank = -1
            rank = self._rank
        return kv_transfer.collect(rank=rank)

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        # 聚合解析：从 .data 字典重建 stats 对象（跨 worker aggregate 用）。
        return kv_transfer.build_stats(data)

    @classmethod
    def build_prom_metrics(cls, vllm_config, metric_types, labelnames,
                           per_engine_labelvalues):
        # 前端单进程：注册 kv_transfer_failures + worker_mem_* 并 observe，无需 multiprocess。
        return kv_transfer.build_prom_metrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues)


class PrefillScheduler:
    POLL_INTERVAL = 0.2
    TIMEOUT_WAIT_PULL_DONE = 600.0

    def __init__(self, vllm_config: VllmConfig):
        self.parallel = desc = ParallelDesc(vllm_config)
        self.engines = [None] * desc.size
        self.workers = [None] * desc.size
        self.pp_layers = [None] * desc.pp
        self.collected = False

        self.looping = True
        self.addr = start_daemon(self._task_collect_workers).get()
        self.finish_time: dict[str, float] = {}              # req_id -> finish_time

    def __del__(self):
        self.looping = False

    def _task_collect_workers(self, feedback: queue.Queue):
        sock = SimpleServer(f"tcp://{get_local_ip()}:0")
        feedback.put(sock.addr())  # any available port

        def on_register(req: dict):
            try:  # parse req
                rank: int = req["rank"]
                w_addr: str = req["worker_addr"]
                e_addr: str = req["engine_addr"]
                layer_ids: list = req["layer_ids"]
                if not isinstance(layer_ids, list) or not isinstance(rank, int):
                    raise TypeError("layer_ids must be list, rank must be int")
                if not isinstance(w_addr, str) or not isinstance(e_addr, str):
                    raise TypeError("worker_addr and engine_addr must be str")
                if not (0 <= rank < len(self.engines)):
                    raise ValueError(f"invalid rank: {rank}")
                pp_rank = rank // self.parallel.pp_stride
            except (TypeError, ValueError, KeyError):
                return {"error": "bad request"}
            if str(layer_ids) != str(self.pp_layers[pp_rank]):
                logger.info(f"pp_layers[{pp_rank}]={layer_ids}")
            self.pp_layers[pp_rank] = layer_ids
            self.workers[rank] = w_addr
            self.engines[rank] = e_addr
            logger.info(f"worker[{rank}] at {w_addr}")
            if None not in self.engines:
                self.collected = True
            return {"registered": "ok"}

        while self.looping:
            sock.handle(register=on_register)
            time.sleep(self.POLL_INTERVAL)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks):
        # chunked prefill allocates more than once
        # thus we handle blocks in "request_finished"
        pass

    def build_connector_metadata(self):
        now = time.time()
        timeout = {req_id
            for req_id, ts in self.finish_time.items()
            if now - ts > self.TIMEOUT_WAIT_PULL_DONE}
        pending = set(self.finish_time.keys()) - timeout
        metadata = Metadata(self.addr)
        metadata.req_params = {"pending": pending, "timeout": timeout}
        return metadata

    def update_connector_output(self, connector_output):
        for req_id in connector_output.finished_sending or ():
            self.finish_time.pop(req_id, None)

    def request_finished(self, request: Request, block_ids: list[list[int]]):
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None  # aborted, immediately free
        if sum(len(group) for group in block_ids) == 0:
            return False, None  # empty block, immediately free

        while not self.collected:
            logger.warning(f"collecting workers...")
            time.sleep(0.5)

        req_id = request.request_id
        self.finish_time[req_id] = time.time()
        return True, dict(  # delay free
            prefill_req_id=req_id,
            prefill_blocks=block_ids,
            parallel=self.parallel.to_list(),
            workers=self.workers.copy(),
            engines=self.engines.copy(),
            pp_layers=self.pp_layers.copy(),
            # TODO: deprecated params for omni-proxy validation
            remote_cluster_id="deprecated",
            remote_host_ip="deprecated",
            remote_block_ids="deprecated",
        )


class PrefillWorker:
    POLL_INTERVAL = 0.2

    def __init__(self, vllm_config: VllmConfig):
        self.manager = CacheManager(get_kv_port(vllm_config), is_prefill=True)
        self.send_done = queue.Queue()
        self.looping = True
        self.addr = start_daemon(self._task_cnnt_manager).get()
        self.scheduler_addr = None
        self.pending: set[str] = set()
        self.reported: set[str] = set()

    def __del__(self):
        self.looping = False

    def _task_cnnt_manager(self, feedback: queue.Queue):
        in_progress = {}  # req_id -> (done, parts)
        sock = SimpleServer(f"tcp://{get_local_ip()}:0")
        feedback.put(sock.addr())  # bind on any available port

        def on_pull_done(req: dict):
            try:  # parse request
                req_id: str = req["req_id"]
                _parts: int = req["parts"]
                if not isinstance(req_id, str) or not isinstance(_parts, int):
                    raise TypeError("req_id must be str, parts must be int")
            except (TypeError, ValueError, KeyError):  # bad request
                return {"error": "bad request"}
            done, parts = in_progress.pop(req_id, (0, 0))
            parts = max(_parts, parts)
            done = done + 1
            if done >= parts:  # all parts done
                self.send_done.put(req_id)
            else:  # store unfinished req
                in_progress[req_id] = (done, parts)
            return {"pull_done": "ok"}

        while self.looping:
            sock.handle(pull_done=on_pull_done)
            time.sleep(self.POLL_INTERVAL)

    def start_load_kv(self, metadata: Metadata):
        # prefill does not pull kv
        # just register worker to scheduler
        addr = metadata.scheduler_addr
        if addr is None:
            raise ValueError("scheduler_addr is None")
        if self.scheduler_addr == addr:
            calm_down(self.start_load_kv)  # yield
            return

        def cb(rep: dict | None):
            if rep is not None:
                self.scheduler_addr = addr  # succeeded
        sock = SimpleClient(addr)
        layer_ids = self.manager.register(None)  # query layer_ids
        sock.query("register", cb=cb,
            rank=get_world_group().rank_in_group,
            worker_addr=self.addr,
            engine_addr=self.manager.engine_addr,
            layer_ids=layer_ids)
        sock.flush()
        sock.close()

    def get_finished(self, metadata: Metadata):
        send_done, missed = set(), set()
        pending = metadata.req_params["pending"]
        timeout = metadata.req_params["timeout"]
        while not self.send_done.empty():
            self.pending.add(self.send_done.get())
        for req_id in self.pending:
            if req_id in pending:
                send_done.add(req_id)
            elif req_id not in timeout:
                missed.add(req_id)
        self.pending = missed
        if timeout and get_world_group().rank_in_group == 0:
            send_done |= timeout
        send_done -= self.reported
        self.reported = (self.reported | send_done) & (pending | timeout)
        if not send_done:
            calm_down(self.get_finished)  # yield
        return send_done, None  # only send

    def get_block_ids_with_load_errors(self) -> set[int]:
        # Prefill workers never pull KV blocks.
        return set()


class DecodeScheduler:

    def __init__(self, vllm_config: VllmConfig):
        self.block_size = vllm_config.cache_config.block_size
        self.pending_build: dict[str, tuple[Request, list]] = {}  # req_id -> (req, blocks)
        self.processed: set[str] = set()

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens):
        if num_computed_tokens % self.block_size != 0:
            raise ValueError(
                f"num_computed_tokens {num_computed_tokens} not aligned "
                f"to block_size {self.block_size}")
        if request.request_id in self.processed:
            return 0, False
        if request.kv_transfer_params is None:
            logger.warning(f"request without kv_transfer_params: {request.request_id}")
            return 0, False
        count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
        return count, count > 0

    def update_state_after_alloc(self, request: Request, blocks: list[list[int]]):
        if request.kv_transfer_params is None:
            return  # marked as built, skip
        self.processed.add(request.request_id)
        self.pending_build[request.request_id] = (request, blocks)

    def build_connector_metadata(self):
        metadata = Metadata()
        for req_id, (req, blocks) in self.pending_build.items():
            metadata.req_params[req_id] = dict(
                token_num=req.num_prompt_tokens,
                decode_blocks=blocks,
                **req.kv_transfer_params,
            )
            req.kv_transfer_params = None  # mark as built
        self.pending_build.clear()
        return metadata

    def request_finished(self, request: Request, block_ids):
        req_id = request.request_id
        params = request.kv_transfer_params
        if (request.status == RequestStatus.FINISHED_ABORTED
            and params is not None):
            if "prefill_req_id" in params:
                logger.warning(f"Request aborted without built: {req_id}")
                sock = SimpleClient(params["workers"][0])  # always send to worker0
                sock.query("pull_done", req_id=params["prefill_req_id"], parts=1)
                sock.close(linger=-1)
        self.processed.discard(req_id)
        return False, None  # immediately free


class DecodeWorker:
    POLL_INTERVAL = 0.2
    NUM_PARALLEL_PULL = 1
    SOCK_RECYCLE_DELAY = 60

    def __init__(self, vllm_config: VllmConfig):
        self.manager = CacheManager(get_kv_port(vllm_config), is_prefill=False)  # client
        self.executor = ThreadPoolExecutor(self.NUM_PARALLEL_PULL)
        self.pull_done = queue.Queue()
        self.send_pull_done = queue.Queue()
        self._invalid_block_ids: set[int] = set()
        self._invalid_lock = threading.Lock()
        self.looping = True
        kv_transfer.maybe_selftest()  # OMNI_METRICS_KV_TRANSFER_SELFTEST=1 时注入合成失败，端到端自测
        start_daemon(self._task_send_pull_done)

    def __del__(self):
        self.looping = False

    def _task_send_pull_done(self, feedback):
        socks: dict[str, SimpleClient] = {}
        times: dict[str, float] = {}
        broken: set[str] = set()

        def on_send(addr, req_id, parts):
            if addr in socks:
                sock = socks[addr]
            else:
                socks[addr] = sock = SimpleClient(addr)

            def cb(rep):
                if rep is None:  # timeout or sock error
                    broken.add(addr)
                    logger.error(f"err send pull done {addr}")
                    kv_transfer.record_failure("send_done")
            sock.query("pull_done", cb=cb, req_id=req_id, parts=parts)
            times[addr] = time.time()

        def recycle(now):
            expired = {addr for addr, ts in times.items()
                if now - ts > self.SOCK_RECYCLE_DELAY}
            for addr in broken | expired:
                socks.pop(addr).close()
                del times[addr]
            broken.clear()

        while self.looping:
            time.sleep(self.POLL_INTERVAL)
            while not self.send_pull_done.empty():
                on_send(*self.send_pull_done.get())
            for sock in socks.values():
                sock.routine()
            recycle(time.time())

    def _pull_kv(self, req_id: str, params: dict):
        layer_ids = self.manager.register(None)  # query layer_ids
        pull = SchemePull(params, layer_ids)
        targets = pull.targets()

        def task_pull_kv():
            failed_blocks: set[int] = set()
            for addr, target in targets.items():
                try:
                    ok = self.manager.pull_blocks(
                        addr,
                        target.p_blocks,
                        target.d_blocks,
                        target.layer_ids,
                    )
                except Exception as error:  # noqa: BLE001
                    logger.error(f"err pull_blocks: {addr} {target}: {error}")
                    ok = False
                if not ok:
                    failed_blocks.update(
                        block_id
                        for block_id in target.d_blocks
                        if block_id not in SchemePull.SKIP_BLOCK_ID
                    )
            if failed_blocks:
                kv_transfer.record_failure("pull")
                with self._invalid_lock:
                    self._invalid_block_ids |= failed_blocks
            # A failed pull must also leave WAITING_FOR_REMOTE_KVS. The invalid
            # block IDs let the scheduler recompute or fail the request based
            # on kv_load_failure_policy.
            self.pull_done.put(req_id)            # local
            self.send_pull_done.put(pull.done())  # remote

        def handle_exception(future):
            if future.exception():
                logger.error(f"err in task_pull_kv: {future.exception()}")
                kv_transfer.record_failure("pull")
                raise future.exception()

        task = self.executor.submit(task_pull_kv)
        task.add_done_callback(handle_exception)

    def start_load_kv(self, metadata: Metadata):
        for req_id, params in metadata.req_params.items():
            self._pull_kv(req_id, params)

    def get_finished(self, metadata: Metadata):
        recv_done = set()  # req_ids
        while not self.pull_done.empty():
            recv_done.add(self.pull_done.get())
        return None, recv_done  # only recv

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._invalid_lock:
            invalid_block_ids = self._invalid_block_ids
            self._invalid_block_ids = set()
        return invalid_block_ids


class SchemePull:
    SKIP_BLOCK_ID = [-1, 0]

    @dataclass
    class Target:
        p_blocks: list[int]          # remote block ids
        d_blocks: list[int]          # local block ids
        layer_ids: list[int] = None  # None for select all layers

    def __init__(
        self,
        kv_transfer_params: dict,
        local_layer_ids: list[int],
    ):
        def get(name, typ):
            if name not in kv_transfer_params:
                raise KeyError(f"missing kv_transfer param: {name}")
            val = kv_transfer_params[name]
            if not isinstance(val, typ):
                raise TypeError(f"{name} must be {typ.__name__}, got {type(val)}")
            return val

        self.local_layer_ids = local_layer_ids
        self.tok_num: int = get("token_num", int)
        self.workers: list[str] = get("workers", list)
        self.engines: list[str] = get("engines", list)
        self.p_req_id: str = get("prefill_req_id", str)
        self.pp_layers: list[list] = get("pp_layers", list)
        self.d_blocks: list[list[int]] = get("decode_blocks", list)
        self.p_blocks: list[list[int]] = get("prefill_blocks", list)
        self.remote_parallel = ParallelDesc(get("parallel", list))
        self.local_parallel = ParallelDesc(None)

    def _parse_blocks(self):
        d_blocks, p_blocks = [], []
        for p_group, d_group in zip(self.p_blocks, self.d_blocks):
            if len(p_group) != len(d_group):
                if len(p_group) != len(d_group) + 1:
                    raise ValueError(
                        f"block group length mismatch: {len(p_group)}:{len(d_group)}")
                p_group = p_group[:-1]
            for p_block, d_block in zip(p_group, d_group):
                if d_block not in self.SKIP_BLOCK_ID:  # ensure not hit prefix cache
                    if p_block not in self.SKIP_BLOCK_ID:
                        p_blocks.append(p_block)
                        d_blocks.append(d_block)
        return p_blocks, d_blocks

    def _dcp_broadcast(self, p_pp_offset: int):
        p, d = self.remote_parallel, self.local_parallel
        p_dcp_base = p_pp_offset // p.dcp * p.dcp
        targets = defaultdict(lambda: ([], []))

        def append_block(d_idx, d_block, p_group):
            idx = d_idx * d.dcp + d.dcp_rank
            p_block = p_group[idx // p.dcp]
            pp_offset = p_dcp_base + idx % p.dcp
            if p_block in self.SKIP_BLOCK_ID:
                raise ValueError(f"prefill block {p_block} in skip")
            p_blocks, d_blocks = targets[pp_offset]
            p_blocks.append(p_block)
            d_blocks.append(d_block)

        for p_group, d_group in zip(self.p_blocks, self.d_blocks):
            if len(p_group) * p.dcp > (len(d_group) + 1) * d.dcp:
                raise ValueError(
                    f"dcp block count mismatch: {len(p_group)}*{p.dcp} vs "
                    f"({len(d_group)}+1)*{d.dcp}")
            for d_idx, d_block in enumerate(d_group):
                if d_block not in self.SKIP_BLOCK_ID:
                    append_block(d_idx, d_block, p_group)

        return {pp_offset: self.Target(p_blocks, d_blocks)
            for pp_offset, (p_blocks, d_blocks) in targets.items()}

    def targets(self) -> dict[str, Target]:
        p, d = self.remote_parallel, self.local_parallel
        if len(self.engines) != p.size:
            raise ValueError(f"engines size {len(self.engines)} != {p.size}")
        if len(self.pp_layers) != p.pp:
            raise ValueError(f"pp_layers size {len(self.pp_layers)} != {p.pp}")
        if p.dp != 1:
            raise ValueError(f"remote dp must be 1, got {p.dp}")
        if p.pp % d.pp != 0:
            raise ValueError(f"p.pp {p.pp} not divisible by d.pp {d.pp}")
        if p.pcp != 1 or d.pcp != 1:
            raise ValueError("pcp not supported")
        if d.tp % d.dcp != 0 or p.tp % p.dcp != 0:
            raise ValueError(
                f"tp must be divisible by dcp: d({d.tp}/{d.dcp}) p({p.tp}/{p.dcp})")

        # scheme 1v1 connection for load balance
        d_pp_offset = d.rank % d.pp_stride
        p_pp_offset = d_pp_offset * p.pp_stride // d.pp_stride

        if p.dcp != d.dcp:  # need kvsp reorg
            # caches must be interleaved with block size
            targets = self._dcp_broadcast(p_pp_offset)
        else:  # simple 1v1 connection scheme
            p_blocks, d_blocks = self._parse_blocks()
            targets = {p_pp_offset: self.Target(p_blocks, d_blocks)}

        return self._pp_broadcast(targets)

    def _pp_broadcast(self, targets: dict[int, Target]) -> dict[str, Target]:
        def full_cover(a: set, b: set):
            outside = b - a       # B's items outside A
            inside = b - outside  # others are inside A
            if inside and outside:
                raise ValueError("forbid partial coverage")
            return inside

        p = self.remote_parallel
        pp_targets, local_ids = {}, set(self.local_layer_ids)

        for pp_rank, layer_ids in enumerate(self.pp_layers):
            if full_cover(local_ids, set(layer_ids)):
                for pp_offset, t in targets.items():
                    if pp_offset >= p.pp_stride:
                        raise ValueError(f"invalid pp_offset: {pp_offset}")
                    rank = pp_offset + pp_rank * p.pp_stride
                    addr = self.engines[rank]
                    pp_targets[addr] = self.Target(t.p_blocks, t.d_blocks, layer_ids)
        return pp_targets

    def done(self):  # send pull_done to which worker
        p, d = self.remote_parallel, self.local_parallel
        parts = d.size // d.dp
        w_addr = self.workers[d.dp_rank % p.size]
        return w_addr, self.p_req_id, parts
