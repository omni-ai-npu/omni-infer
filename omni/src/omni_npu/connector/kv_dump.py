# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import json
import time
import torch
import queue
import hashlib
import threading
from collections import defaultdict
from contextlib import contextmanager

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger

logger = init_logger(__name__)
_kv_dump_path: str = os.environ.get("KV_DUMP_PATH", "")


def _make_run_id() -> str:
    return str(int(time.time()))


def gen_hash(buf: torch.Tensor):
    if buf.dtype != torch.uint8:
        raise ValueError(f"buf dtype must be uint8, got {buf.dtype}")
    if buf.dim() != 1:
        raise ValueError(f"buf dim must be 1, got {buf.dim()}")

    def tensor_desc(x: torch.Tensor):
        dtype = f"{x.dtype}".replace("torch.", "")
        shape = ",".join(str(i) for i in x.shape)
        return f"{dtype}({shape})"

    kind = f"{tensor_desc(buf)}.blk"
    dat = buf.cpu().numpy().tobytes()
    digest = hashlib.md5(dat).hexdigest()
    return kind, digest


def append_items(path: str, items: dict):
    if not isinstance(items, dict):
        raise TypeError(f"items must be dict, got {type(items)}")
    if not items:
        return
    if not os.path.exists(path):
        open(path, "w").close()
    with open(path, "r+") as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            f.write("{\n")
        else:
            f.seek(file_size - 2, 0)
            f.write(",\n")

        prefix = ""
        for k, v in items.items():
            f.write(f'{prefix}"{k}": {v}')
            prefix = ",\n"

        f.write("\n}")
        f.truncate()


def load_layers(vllm_caches: dict | list):
    def parse_layers(layers):
        if isinstance(layers, dict):
            for name, layer in layers.items():
                yield name, extract_layer_index(name), layer
        elif isinstance(layers, (tuple, list)):
            for i, layer in enumerate(layers):
                yield str(i), i, layer
        else:
            raise TypeError("invalid format")

    def parse_layer(name, layer_id, layer):
        if not isinstance(layer, (tuple, list)):
            layer = [layer]
        for i, x in enumerate(layer):
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{name}.{i} must be torch.Tensor")
            yield f"{name}.{i}", layer_id, x

    block_num = set()

    def to_buffer(x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("layer cache must be torch.Tensor")
        if x.dim() not in [2, 3, 4]:
            raise ValueError(f"tensor dim must be 2/3/4, got {x.dim()}")
        block_num.add(x.size(0))
        buf = x.untyped_storage()
        y = x.new_empty(0, dtype=torch.uint8)
        size = x.size(0) * x.stride(0) * x.element_size()
        if x.is_contiguous():
            offset = x.data_ptr() - buf.data_ptr()
            buf = buf[offset:offset + size]
        else:
            if not x[0].is_contiguous():
                raise ValueError("first block must be contiguous")
            if buf.size() != size:
                raise ValueError(f"untrimed storage required: {buf.size()} != {size}")
        y.set_(buf)
        return y.view(x.size(0), -1)

    def sort_by_key(s: dict):
        return [s[k] for k in sorted(s.keys())]

    buffers = sort_by_key({
        name: (i, to_buffer(x))
        for layer in parse_layers(vllm_caches)
        for name, i, x in parse_layer(*layer)
    })
    if len(block_num) != 1:
        raise ValueError(f"inconsistent block_num: {block_num}")

    layers = defaultdict(list)
    for i, buf in buffers:
        layers[i].append(buf)

    shapes = defaultdict(lambda: (list(), list()))
    for i in sorted(layers.keys()):
        for buf in dict(layers)[i]:
            ptrs, bufs = shapes[str(buf.shape)]
            if buf.data_ptr() not in ptrs:
                ptrs.append(buf.data_ptr())
                bufs.append(buf)
    shapes = [shapes[k] for k in sorted(shapes.keys())]
    shapes = [bufs for ptrs, bufs in shapes]
    shapes: list[list[torch.Tensor]]
    return shapes


def parse_remote_reqs(scheduler_output: SchedulerOutput) -> dict:
    from omni_npu.connector.llmdatadist_connector_v1 import Metadata
    meta: Metadata = scheduler_output.kv_connector_metadata
    return {req_id: params["prefill_req_id"]
        for req_id, params in meta.req_params.items()}


class PrefillReq:

    def __init__(self, new: NewRequestData):
        self.req_id = new.req_id
        self.tok_num = len(new.prompt_token_ids)
        self.blocks = [[idx for idx in group] for group in new.block_ids]
        self.block_set = {idx for blk in self.blocks for idx in blk}

    def dump(self, shapes: list[list[torch.Tensor]], remote_req=None):
        def gen(idx_: int) -> dict:
            idx = abs(idx_)
            attrs = {"block_id": idx}
            if idx == 0:
                return attrs
            if idx_ < 0:
                attrs["hashed"] = True
            for shape in shapes:
                layer = []
                for i, cache in enumerate(shape):
                    kind, digest = gen_hash(cache[idx])
                    layer.append(f"{i}:{digest}")
                attrs[kind] = layer
            return attrs

        t0 = time.monotonic()
        data = {f"group{i}": [gen(idx) for idx in group]
            for i, group in enumerate(self.blocks)}
        dt = int((time.monotonic() - t0) * 1000)
        logger.info("[%dms] hash blocks", dt)
        if remote_req is not None:
            data["remote_req_id"] = remote_req
        return data

    def append_blocks(self, new_blks: tuple[list[int]] | None):
        if new_blks is not None:
            for group, blocks in zip(self.blocks, new_blks):
                for idx in blocks:
                    if idx in self.block_set:
                        raise ValueError(f"duplicate block id: {idx}")
                    self.block_set.add(idx)
                    group.append(idx)
        return self


class DecodeReq(PrefillReq):

    def __init__(self, new: NewRequestData):
        super().__init__(new)
        self.hist_hash: dict[str, tuple] = defaultdict(lambda: (None, None, None))
        self.hashed_blocks: dict[int, list[list[str]]] = {}
        self.hashed_blks: list[int] = []
        self.hashed_batch: torch.Tensor = None

    def _check_blks(self, full: int, shapes: list[list[torch.Tensor]]):
        idxs = []
        for group in self.blocks:
            idxs.extend(group[:min(full, len(group) - 1)])

        hashed_num = len(self.hashed_blks)
        reorg = self.hashed_blks.copy()
        for idx in reorg:
            if idx not in idxs:
                raise ValueError(f"hashed block {idx} not in active blocks {idxs}")
        for idx in idxs:
            if idx not in reorg:
                reorg.append(idx)

        idx_ = shapes[0][0].new_tensor(reorg, dtype=torch.int)

        if self.hashed_batch is None:
            self.hashed_batch = [[None for cache in shape] for shape in shapes]

        def make_cache(cache: torch.Tensor, cmp: torch.Tensor | None):
            blks = cache[idx_].contiguous()
            if cmp is not None:
                a = blks[:hashed_num]
                b = cmp.to(device=a.device)
                if not torch.equal(a, b):
                    raise ValueError("KV block hash mismatch in decode check")
            return blks.cpu()

        batch = [[make_cache(*cache) for cache in zip(*layer)]
            for layer in zip(shapes, self.hashed_batch)]

        self.hashed_batch = batch
        self.hashed_blks = reorg

    def _check_toks(self, full, rest, groups: list[dict[str, tuple[torch.Tensor]]], pg):
        if rest <= 0:
            return

        def active_blocks():
            for group_cache, group_blk in zip(groups, self.blocks):
                active_id = group_blk[min(full, len(group_blk) - 1)]
                for name, layer in group_cache.items():
                    for i, cache in enumerate(layer):
                        if cache.size(1) == pg:
                            yield f"{name}.{i}", cache[active_id]

        for name, blk in active_blocks():
            sect = blk[:rest].contiguous()
            _full, _rest, _sect = self.hist_hash[name]
            if _full == full:
                if _rest >= rest:
                    raise ValueError(f"token slice rest must grow: {_rest} >= {rest}")
                a = sect[:_rest]
                b = _sect.to(device=a.device)
                if not torch.equal(a, b):
                    raise ValueError(f"KV token slice mismatch at {name}: full={full}")
            self.hist_hash[name] = (full, rest, sect.cpu())

    def update(self,
        tok_num: int,
        new_blks: tuple[list[int]] | None,
        groups: list[dict[str, tuple[torch.Tensor]]],
        shapes: list[list[torch.Tensor]],
        mtp: int = 0,
        pg: int = 128,
    ):
        self.append_blocks(new_blks)
        self.tok_num = tok_num
        stable_tok = tok_num - mtp
        full = stable_tok // pg
        rest = stable_tok % pg

        t0 = time.monotonic()
        self._check_blks(full, shapes)
        dt = int((time.monotonic() - t0) * 1000)
        logger.info("[%dms] check_blks ok: blocks=%d", dt, full)

        t0 = time.monotonic()
        self._check_toks(full, rest, groups, pg)
        dt = int((time.monotonic() - t0) * 1000)
        logger.info("[%dms] check_toks ok: blocks=%d", dt, full)
        return self

    def finalize(self):
        self.hashed_batch = None


class Dump:

    def __init__(self, path: str):
        self.shapes: list[list[torch.Tensor]] = None  # [types, layers]
        self.groups: list[dict[str, tuple[torch.Tensor]]] = None  # [group, layers, caches]
        self.reqs: dict[str, DecodeReq | PrefillReq] = {}
        self.remote_reqs: dict = {}

        run_id = _make_run_id()
        self.path = f"{path}/{run_id}/"
        self.rank = None
        self.change = queue.Queue()
        threading.Thread(target=self._task_save_json, daemon=True).start()

    def _task_save_json(self):
        while self.rank is None:
            time.sleep(1.0)

        outfile = f"rank{self.rank}.json"
        os.makedirs(self.path, exist_ok=True)

        while True:
            time.sleep(1.0)
            if self.change.empty():
                continue
            items = {}
            while not self.change.empty():
                req_id, data = self.change.get()
                t0 = time.monotonic()
                items[req_id] = json.dumps(data, indent=2)
                dt = int((time.monotonic() - t0) * 1000)
                logger.info("[%dms] save json: %s", dt, self.path + outfile)
            append_items(self.path + outfile, items)

    def _register(self, model_runner: GPUModelRunner):
        if self.shapes is not None:
            return
        layers = model_runner.compilation_config.static_forward_context
        groups = model_runner.kv_cache_config.kv_cache_groups

        def validate(x):
            if not isinstance(x, list):
                raise TypeError(f"kv_cache must be list, got {type(x)}")
            if not isinstance(x[0], tuple):
                raise TypeError(f"kv_cache[0] must be tuple, got {type(x[0])}")
            return x[0]

        self.groups = [{
            name: validate(layers[name].kv_cache)
            for name in group.layer_names}
            for group in groups]

        flatten = {k: v for group in self.groups for k, v in group.items()}
        self.shapes = load_layers(flatten)

    @contextmanager
    def update(self,
        model_runner: GPUModelRunner,
        scheduler_output: SchedulerOutput,
    ):
        self.rank = get_world_group().rank
        self._register(model_runner)

        role = {"kv_producer": True, "kv_consumer": False}
        cfg = model_runner.vllm_config.kv_transfer_config
        kv_role = cfg.kv_role
        if kv_role not in role:
            raise ValueError(f"unsupported kv_role: {kv_role}")
        is_prefill = role[kv_role]

        news = scheduler_output.scheduled_new_reqs
        reqs = scheduler_output.scheduled_cached_reqs
        req_ids = reqs.req_ids
        blocks = reqs.new_block_ids or [None] * len(req_ids)

        if is_prefill:
            self._prefill(req_ids, blocks, news)
            yield
            for req in self.reqs.values():
                req.data = req.dump(self.shapes)
        else:
            computed = reqs.num_computed_tokens
            mtp = model_runner.num_spec_tokens
            remote_reqs = parse_remote_reqs(scheduler_output)

            self._decode(req_ids, computed, blocks, news, mtp, remote_reqs)
            yield

    def _decode(self, req_ids, computed, blocks, news, mtp, remote_reqs):
        self.remote_reqs.update(remote_reqs)

        for req_id, req in self.reqs.items():
            if req_id not in req_ids:
                req.finalize()
                self.remote_reqs.pop(req_id, None)

        self.reqs = {req_id: self.reqs[req_id].update(
            tok_num, new_blks, self.groups, self.shapes, mtp,
        ) for req_id, tok_num, new_blks in zip(req_ids, computed, blocks)}

        for new in news:
            req_id = new.req_id
            req = DecodeReq(new)
            self.reqs[req_id] = req
            data = req.dump(self.shapes, self.remote_reqs[req_id])
            self.change.put((req_id, data))

    def _prefill(self, req_ids, blocks, news):
        for req_id, req in self.reqs.items():
            if req_id not in req_ids:
                self.change.put((req_id, req.data))

        self.reqs = {req_id: self.reqs[req_id].append_blocks(new_blks)
            for req_id, new_blks in zip(req_ids, blocks)}

        for new in news:
            self.reqs[new.req_id] = PrefillReq(new)


def maybe_dump_kv(fn_execute_model):
    def wrapper(
        self: GPUModelRunner,
        scheduler_output: SchedulerOutput,
        intermediate_tensors=None,
    ):
        def get_instance(name="dump_prefill") -> Dump:
            if not hasattr(maybe_dump_kv, name):
                setattr(maybe_dump_kv, name, Dump(_kv_dump_path))
            return getattr(maybe_dump_kv, name)

        if not _kv_dump_path:
            return fn_execute_model(self, scheduler_output, intermediate_tensors)
        with get_instance().update(self, scheduler_output):
            return fn_execute_model(self, scheduler_output, intermediate_tensors)
    return wrapper
