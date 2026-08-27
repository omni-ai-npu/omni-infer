# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import time
import zlib
import uuid
import json
import torch
import queue
import socket
import threading
from collections import defaultdict

import llm_datadist as datadist
from llm_datadist import LLMStatusCode

from omni_npu.v1.utils import on_ascend950
from vllm.model_executor.models.utils import extract_layer_index
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger

from .utils import (
    get_local_ip,
    serial_brief,
    start_daemon,
)


logger = init_logger(__name__)


class DatadistEngine:
    LINK_TIMEOUT = int(os.environ.get("LINK_TIMEOUT", "120000"))  # ms
    SYNC_KV_TIMEOUT = int(os.environ.get("SYNC_KV_TIMEOUT", "120000"))  # ms
    LINK_RECYCLE_DELAY = 300.0
    HEARTBEAT_INTERVAL = 1.0

    def __init__(self, port: int, is_prefill: bool):
        self.on_ascend950 = on_ascend950()
        self.local_rank = get_world_group().local_rank
        self.cid = uuid.uuid4().int & 0xffffffffffffffff
        self.ip = get_local_ip()
        port = port + self.local_rank
        if not (0 < port < 65536):
            raise ValueError(f"invalid port: {port}")

        hixl_backend = False
        if self.on_ascend950:
            # Load local_comm_res from JSON file for hixl backend
            local_comm_res = self._get_local_comm_res()
            if local_comm_res:
                hixl_backend = True
        else:
            local_comm_res = ""  # do new_datadist_link

        cfg = datadist.LLMConfig()
        cfg.device_id = self.local_rank
        cfg.sync_kv_timeout = self.SYNC_KV_TIMEOUT  # RoCE
        cfg.enable_remote_cache_accessible = True
        cfg.local_comm_res = local_comm_res

        # Determine listening address configuration based on deployment mode
        if is_prefill:
            # Prefill service: bind to specified fixed port for stable connection access
            cfg.listen_ip_info = f"{self.ip}:{port}"
        elif hixl_backend:
            # HIXL backend service: use port 0, system automatically assign available random port
            cfg.listen_ip_info = f"{self.ip}:0"
            
        if hixl_backend:
            cfg.transfer_backend = "hixl"
            logger.info(f"[hixl] {cfg.listen_ip_info=}")

        role = (datadist.LLMRole.PROMPT if is_prefill
            else datadist.LLMRole.DECODER)
        self.engine = datadist.LLMDataDist(role, self.cid)
        self.cfg = cfg.generate_options()
        self.engine.init(self.cfg)

        self.data_lock = threading.Lock()
        self.caches = {}  # uuid -> cache
        self.links = {}  # cid -> ts
        self.is_prefill = is_prefill
        self.port = port
        self.inited = True

    def sleep(self):
        if self.inited:
            self._clear()
            self.unregister()
            with self.data_lock:
                self.engine.finalize()
            self.inited = False

    def weakup(self):
        if not self.inited:
            with self.data_lock:
                self.engine.init(self.cfg)
            self.inited = True

    # ================= link =================

    def _make_info(self, cid, ip=None, port=None):
        info = datadist.LLMClusterInfo()
        info.remote_cluster_id = cid
        if None not in [ip, port]:
            info.append_remote_ip_info(ip, port)
            info.append_local_ip_info(self.ip, 0)
        return info

    def _link(self, cid, ip=None, port=None) -> bool:
        with self.data_lock:
            if None not in [ip, port] and cid not in self.links:
                info = self._make_info(cid, ip, port)
                ret, _ = self.engine.link_clusters(
                    [info], timeout=self.LINK_TIMEOUT)
                if ret != LLMStatusCode.LLM_SUCCESS:
                    logger.error(f"link failed: {ret}")
                    return False
            if cid not in self.links:
                logger.info(f"dynamic link: {cid}")
            self.links[cid] = time.time()
            return True

    def _unlink(self, cid, ip=None, port=None):
        with self.data_lock:
            info = self._make_info(cid, ip, port)
            self.links.pop(cid, None)
            self.engine.unlink_clusters([info],
                timeout=self.LINK_TIMEOUT, force=True)

    def _clear(self, timeout=None):
        now = time.time()
        with self.data_lock:
            expired = [cid for cid, ts in self.links.items()
                if (timeout is None or now - ts > timeout)]
            for cid in expired:
                del self.links[cid]
            if expired:
                if timeout is not None:
                    logger.info(f"recycle links (>{timeout}s) {expired}")
                infos = [self._make_info(cid) for cid in expired]
                self.engine.unlink_clusters(infos,
                    timeout=self.LINK_TIMEOUT, force=True)

    # ================= memory =================

    def register(self, bufs: list[torch.Tensor]) -> int:
        if not self.inited:
            raise RuntimeError("engine not inited")
        if not all(it.dtype == torch.uint8 for it in bufs):
            raise TypeError("bufs must be torch.uint8 tensors")
        if len({str(it.shape) for it in bufs}) != 1:
            raise ValueError("bufs must share the same shape")
        desc = datadist.CacheDesc(
            num_tensors=len(bufs),
            shape=tuple(bufs[0].shape),
            data_type=datadist.DataType.DT_UINT8,
        )
        addrs = [int(it.data_ptr()) for it in bufs]
        cache_id = uuid.uuid4().int
        with self.data_lock:
            key = datadist.BlocksCacheKey(
                self.engine.cluster_id,
                model_id=len(self.caches),
            ) if self.is_prefill else None  # only server needs key
            cache = self.engine.cache_manager.register_blocks_cache(desc, addrs, key)
            self.caches[cache_id] = cache
        return cache_id

    def unregister(self):
        if not self.inited:
            raise RuntimeError("engine not inited")
        with self.data_lock:
            for _, cache in self.caches.items():
                self.engine.cache_manager.unregister_cache(cache.cache_id)
            self.caches.clear()

    # ================= A5 hixl =================

    def _get_host_physical_device_id(self, logical_device_id: int) -> int:
        """
        Translate a container-local device index into the host physical device id.

        Container runtimes (ModelArts, Ascend docker runtime, k8s device plugin) hand a
        subset of the host devices to the container and publish it in ASCEND_VISIBLE_DEVICES.
        The container-local index follows the physical ids in ascending order, which is not
        necessarily the order they are written in, so the list is sorted before indexing.
        On bare metal the variable is absent and the two numbering spaces already coincide.

        Example: ASCEND_VISIBLE_DEVICES="7,4,5,6" -> [4,5,6,7], logical 0 -> host physical 4.
        """
        host_devices = os.getenv("ASCEND_VISIBLE_DEVICES")
        if not host_devices:
            return logical_device_id
        try:
            mapping = sorted(int(d) for d in host_devices.split(",") if d.strip())
        except ValueError:
            logger.warning(
                f"cannot parse ASCEND_VISIBLE_DEVICES={host_devices}, "
                f"using logical device {logical_device_id} as the physical id"
            )
            return logical_device_id
        if logical_device_id < len(mapping):
            host_device_id = mapping[logical_device_id]
            if host_device_id != logical_device_id:
                logger.info(
                    f"container logical device {logical_device_id} maps to "
                    f"host physical device {host_device_id}"
                )
            return host_device_id
        logger.warning(
            f"logical device {logical_device_id} out of range in "
            f"ASCEND_VISIBLE_DEVICES={host_devices}, using it as the physical id"
        )
        return logical_device_id

    def _get_physical_device_id(self) -> int:
        """
        Get physical device ID from ASCEND_RT_VISIBLE_DEVICES environment variable.
        Falls back to local_rank if not set.

        The mapping logic:
        - ASCEND_RT_VISIBLE_DEVICES="0,1,2,3", local_rank=0 -> physical_device_id=0
        - ASCEND_RT_VISIBLE_DEVICES="4,5,6,7", local_rank=0 -> physical_device_id=4
        - ASCEND_RT_VISIBLE_DEVICES="4,5,6,7", local_rank=1 -> physical_device_id=5

        The value selected above is a container-local index, so it is translated once more
        through ASCEND_VISIBLE_DEVICES to reach the host physical id that
        /etc/hccl_rootinfo.json is keyed by.

        Returns:
            int: Physical device ID.
        """
        visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
        if visible_devices is not None:
            # ASCEND_RT_VISIBLE_DEVICES can be a comma-separated list like "0,1,2,3"
            # The list represents visible devices, indexed by local_rank
            devices = [d.strip() for d in visible_devices.split(",") if d.strip()]
            if devices:
                # Use the device corresponding to local_rank
                if self.local_rank < len(devices):
                    return self._get_host_physical_device_id(int(devices[self.local_rank]))
                # Fallback: local_rank exceeds device count, log warning
                logger.warning(
                    f"local_rank ({self.local_rank}) >= visible device count ({len(devices)}), "
                    f"falling back to first device {devices[0]}"
                )
                return self._get_host_physical_device_id(int(devices[0]))
        return self._get_host_physical_device_id(self.local_rank)

    def _get_cluster_device_id(self) -> int:
        """
        Map physical device ID to cluster device ID via /etc/hccl_rootinfo.json.

        Uses rank_list[physical_device_id].device_id from the HCCL root info file.
        Falls back to physical_device_id if the file is missing or mapping fails.
        """
        physical_device_id = self._get_physical_device_id()
        hccl_rootinfo_path = "/etc/hccl_rootinfo.json"
        try:
            with open(hccl_rootinfo_path, "r") as f:
                rootinfo = json.load(f)
            rank_list = rootinfo.get("rank_list", [])
            if physical_device_id < len(rank_list):
                cluster_device_id = rank_list[physical_device_id].get("device_id")
                if cluster_device_id is not None:
                    return int(cluster_device_id)
            logger.warning(
                f"cluster device mapping failed for physical_device_id={physical_device_id} "
                f"(rank_list length={len(rank_list)}), falling back to physical_device_id"
            )
        except FileNotFoundError:
            logger.warning(
                f"hccl_rootinfo file not found: {hccl_rootinfo_path}, "
                f"falling back to physical_device_id={physical_device_id}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load cluster device id from {hccl_rootinfo_path}: {e}, "
                f"falling back to physical_device_id={physical_device_id}"
            )
        return physical_device_id

    def _get_local_comm_res(self) -> str:
        """
        Read local_comm_res from JSON file based on cluster device ID.
        Following push_blocks_sample.py pattern: load from ub_endpoint_npu_*.json

        The base path can be configured via HIXLP_ENDPOINT_PATH environment variable.
        If not set, defaults to /etc/hixlep.

        Controlled by HIXL_LOCAL_COMM_RES_ENABLE environment variable:
        - "1", "true", "yes" (default): enable loading
        - "0", "false", "no": disable loading, return empty string

        Returns:
            str: JSON string for hixl backend, or empty string if not found/disabled.
        """
        # Check if local_comm_res loading is enabled
        local_comm_res_enable_env = os.getenv("HIXL_LOCAL_COMM_RES_ENABLE", "false").lower()
        if local_comm_res_enable_env in ("0", "false", "no"):
            logger.info("HIXL_LOCAL_COMM_RES_ENABLE is disabled, skipping local_comm_res loading")
            return ""

        cluster_device_id = self._get_cluster_device_id()

        # Get base path from environment variable, default to /etc/hixlep
        hixlp_base_path = os.getenv("HIXLP_ENDPOINT_PATH", "/etc/hixlep")

        # Build file path based on cluster_device_id
        comm_res_file_path = f"{hixlp_base_path}/ub_endpoint_npu_{cluster_device_id}.json"

        try:
            with open(comm_res_file_path, 'r') as f:
                json_content = json.load(f)
                local_comm_res = json.dumps(json_content, separators=(',', ':'))
                logger.info(f"Loaded local_comm_res from {comm_res_file_path}, cluster_device_id={cluster_device_id}")
                return local_comm_res
        except FileNotFoundError:
            logger.warning(f"local_comm_res file not found: {comm_res_file_path}, using empty string")
        except Exception as e:
            logger.warning(f"Failed to load local_comm_res from {comm_res_file_path}: {e}, using empty string")

        return ""


class ServerEngine(DatadistEngine):
    POLL_INTERVAL = 0.2

    def __init__(self, port: int):
        super().__init__(port, is_prefill=True)
        self.looping = True
        hb_port = start_daemon(self._task_recycle).get()
        self.addr = f"{self.ip}:{hb_port}:{self.port}:{self.cid}"

    def __del__(self):
        self.looping = False

    def _task_recycle(self, feedback: queue.Queue):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.ip, 0))  # any available port
        sock.setblocking(False)
        _, hb_port = sock.getsockname()
        feedback.put(hb_port)

        def flush_msg():
            while True:
                try:
                    data, _ = sock.recvfrom(1024)
                except BlockingIOError:  # no msg
                    break
                except Exception as e:  # other err
                    logger.error(f"sock err: {e}")
                    continue
                try:  # parse and validate cid
                    msg = data.decode()
                    cid, crc32 = tuple(msg.split("#"))
                    if zlib.crc32(cid.encode()) != int(crc32):
                        raise ValueError(f"invalid crc32: {crc32}")
                    cid = int(cid)
                except (ValueError, TypeError, IndexError):  # invalid data
                    continue
                self._link(cid)  # fresh timestamp only
            return self.POLL_INTERVAL

        while self.looping:
            flush_msg()
            self._clear(self.LINK_RECYCLE_DELAY)
            time.sleep(self.HEARTBEAT_INTERVAL)


class ClientEngine(DatadistEngine):
    LINK_RETRIES = 2  # newly reg cache may require relink
    PULL_RETRIES = 2
    RETRY_DELAY = 2.0

    def __init__(self, port: int):
        super().__init__(port, is_prefill=False)
        self.send_hb = queue.Queue()
        self.looping = True
        start_daemon(self._task_recycle)

    def __del__(self):
        self.looping = False

    def _task_recycle(self, feedback):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        hb_data = f"{self.cid}#{zlib.crc32(str(self.cid).encode())}".encode()

        def send_hb():
            addrs = set()
            while not self.send_hb.empty():
                addrs.add(self.send_hb.get())
            for ip, port in addrs:
                try:  # send heartbeat
                    sock.sendto(hb_data, (ip, port))
                except Exception as e:
                    logger.error(f"err send hb to {ip}:{port}, {e}")

        while self.looping:
            send_hb()
            self._clear(self.LINK_RECYCLE_DELAY)
            time.sleep(self.HEARTBEAT_INTERVAL)

    def pull_blocks(
        self,
        addr: str,
        model_id: int,
        cache_id: int,
        p_blocks: list[int],
        d_blocks: list[int],
    ) -> bool:
        if len(p_blocks) != len(d_blocks):
            raise ValueError(
                f"p_blocks and d_blocks length mismatch: "
                f"{len(p_blocks)} vs {len(d_blocks)}")
        if len(p_blocks) == 0:
            return True

        try:  # parse and validate addr
            ip, s_port, e_port, cid = tuple(addr.split(":"))
            s_port, e_port, cid = int(s_port), int(e_port), int(cid)
        except (ValueError, TypeError):
            logger.error(f"invalid addr: {addr}")
            return False

        def pull_once() -> str:
            try:
                with self.data_lock:
                    if not self.inited:
                        logger.error(f"engine not inited")
                        return "fail"
                    cache = self.caches.get(cache_id, None)
                    if cache is None:
                        logger.error(f"invalid {cache_id=}")
                        return "fail"
                    self.engine.cache_manager.pull_blocks(
                        datadist.BlocksCacheKey(cid, model_id),
                        cache, p_blocks, d_blocks)
                return "ok"
            except datadist.LLMException as e:
                logger.error(f"err pull blocks: {e}")
                return "retry" if e.status_code in [
                    LLMStatusCode.LLM_REPEAT_REQUEST,
                    LLMStatusCode.LLM_CLUSTER_NUM_EXCEED_LIMIT,
                    LLMStatusCode.LLM_PROCESSING_LINK,  # Building chain is in progress
                    LLMStatusCode.LLM_DEVICE_OUT_OF_MEMORY,
                    LLMStatusCode.LLM_TIMEOUT,
                    LLMStatusCode.LLM_WAIT_PROCESS_TIMEOUT,
                    LLMStatusCode.LLM_LINK_BUSY,
                ] else "fail"

        def pull_with_retry() -> bool:
            for attempt in range(self.PULL_RETRIES + 1):
                if attempt > 0:
                    time.sleep(self.RETRY_DELAY)
                res = pull_once()
                if res != "retry":
                    break
            return res == "ok"

        self.send_hb.put((ip, s_port))
        for attempt in range(self.LINK_RETRIES + 1):
            if attempt > 0:
                time.sleep(self.RETRY_DELAY)
            if self._link(cid, ip, e_port):
                if pull_with_retry():
                    return True  # succeeded
            self._unlink(cid, ip, e_port)
        return False  # never succeeded


class CacheManager:  # safe in multi-thread

    def __init__(self, port: int, is_prefill: bool):
        if not isinstance(port, int):
            raise TypeError(f"port must be int, got {type(port)}")
        if is_prefill:
            self.engine = ServerEngine(port)
            self.engine_addr = self.engine.addr
        else:
            self.engine = ClientEngine(port)

        # layer_ids could be not consective
        self.loaded_layer_ids: list[int] = []
        self.loaded_layers: dict[int, list] = {}  # layer_id -> buffer
        self.loaded_groups: dict[str, list] = {}  # leyer_ids -> cache_ids
        self.data_lock = threading.Lock()

    def sleep(self):
        self.engine.sleep()  # auto unregister

    def weakup(self):
        self.engine.weakup()

    # ================= memory =================

    def _load_layers(self, vllm_caches: dict | list):
        if self.loaded_layers:
            raise RuntimeError("layers already loaded")

        def parse_layers(layers):  # -> name, layer_id, layer
            if isinstance(layers, dict):  # layer_name -> layer_cache
                for name, layer in layers.items():
                    yield name, extract_layer_index(name), layer
            elif isinstance(layers, (tuple, list)):  # consecutive layer idx
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
                buf = buf[offset : offset + size]
            else:  # blocks with stride
                if not x[0].is_contiguous():
                    raise ValueError("first block must be contiguous")
                if buf.size() != size:
                    raise ValueError(
                        f"untrimed storage required: {buf.size()} != {size}")
            y.set_(buf)
            return y.view(x.size(0), -1)

        def sort_by_key(s: dict):
            return [s[k] for k in sorted(s.keys())]

        buffers = sort_by_key({name: (i, to_buffer(x))
            for layer in parse_layers(vllm_caches)
            for name, i, x in parse_layer(*layer)
        })  # flatten, sort by name, for stable order
        if len(block_num) != 1:
            raise ValueError(f"inconsistent block_num: {block_num}")

        layers = defaultdict(list)
        for i, buf in buffers:
            layers[i].append(buf)  # stable order per layer
        self.loaded_layers = dict(layers)  # layer_id -> buffer_list
        self.loaded_layer_ids = sorted(layers.keys())  # sort for stable order

    def _make_group(self, layer_ids: list[int] | None):
        if not self.loaded_layers:
            raise RuntimeError("layers not loaded")
        shapes = defaultdict(lambda: (list(), list()))  # shape -> (ptrs, bufs)
        # grouped by shape, merge same buffer, stable order
        for i in (layer_ids or self.loaded_layer_ids):
            for buf in self.loaded_layers[i]:
                ptrs, bufs = shapes[str(buf.shape)]
                if buf.data_ptr() not in ptrs:
                    ptrs.append(buf.data_ptr())
                    bufs.append(buf)
        shapes = [shapes[k] for k in sorted(shapes.keys())]
        shapes = [bufs for _, bufs in shapes]  # only bufs
        return [self.engine.register(bufs) for bufs in shapes]

    def register(self, vllm_caches: dict | list | None) -> list[int]:
        if vllm_caches is None:  # None for only query
            return self.loaded_layer_ids
        self.unregister()
        with self.data_lock:
            self._load_layers(vllm_caches)
            if isinstance(self.engine, ServerEngine):
                self._make_group(None)  # all layers
            return self.loaded_layer_ids

    def unregister(self):
        self.engine.unregister()
        with self.data_lock:
            self.loaded_groups.clear()
            self.loaded_layers.clear()
            self.loaded_layer_ids.clear()

    # ================= pull =================

    def _get_group(self, layers: list[int] | None) -> list:
        if not self.loaded_layers:
            raise RuntimeError("layers not loaded")
        layers = layers or self.loaded_layer_ids  # None for all

        def valid(i):
            if i not in self.loaded_layers:
                raise KeyError(f"layer {i} not loaded")
            return i
        # validate, deduplicate and sort
        layers = sorted({valid(i) for i in layers})
        key = serial_brief(layers)

        # lazy register caches by selection
        if key not in self.loaded_groups:
            logger.info(f"dynamic register group: {key}")
            self.loaded_groups[key] = self._make_group(layers)
        return self.loaded_groups[key]

    def pull_blocks(
        self,
        addr: str,
        p_blocks: list[int],
        d_blocks: list[int],
        layer_ids: list[int],  # None for all
    ) -> bool:
        if not isinstance(self.engine, ClientEngine):
            raise TypeError("pull_blocks requires ClientEngine")
        with self.data_lock:
            cache_ids = self._get_group(layer_ids)
        for model_id, cache_id in enumerate(cache_ids):
            if not self.engine.pull_blocks(
                addr, model_id, cache_id, p_blocks, d_blocks):
                return False
        return True
