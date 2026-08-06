# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

import time
import pytest
import torch
import queue
import uuid
from types import SimpleNamespace
from dataclasses import dataclass
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, mock_open

from omni.connector.llmdatadist_manager_v1 import (
    DatadistEngine,
    ServerEngine,
    ClientEngine,
    CacheManager,
)


def _stop_daemon(*objs):
    """Stop the background daemon thread(s) these objects started via start_daemon(),
    so they don't leak into (and pollute) later tests.

    ServerEngine / ClientEngine (and the connector's PrefillScheduler / PrefillWorker /
    DecodeWorker) start a daemon in __init__ and expose the stop switch as ``looping``;
    an LLMDataDistConnector exposes it on its ``scheduler`` / ``worker``. The daemon's
    ``target`` is a bound method holding its owner, so the owner is never GC'd and its
    ``__del__`` (which clears ``looping``) never runs -- the ``while self.looping`` loop
    would otherwise spin for the whole process. Flip the flag so the thread exits.
    """
    for obj in objs:
        for owner in (obj, getattr(obj, "scheduler", None), getattr(obj, "worker", None)):
            if owner is not None and hasattr(owner, "looping"):
                owner.looping = False


@contextmanager
def _patch_module(module, **patches):
    olds = {}
    for name, new in patches.items():
        olds[name] = getattr(module, name, None)
        setattr(module, name, new)
    yield
    for name, old in olds.items():
        setattr(module, name, old)


@contextmanager
def _mock_hccl_rootinfo(rootinfo=None, open_side_effect=None, load_side_effect=None):
    open_mock = mock_open() if open_side_effect is None else MagicMock(side_effect=open_side_effect)
    load_kwargs = {}
    if load_side_effect is not None:
        load_kwargs["side_effect"] = load_side_effect
    elif rootinfo is not None:
        load_kwargs["return_value"] = rootinfo
    with patch("builtins.open", open_mock):
        with patch("omni.connector.llmdatadist_manager_v1.json.load", **load_kwargs):
            yield


class _LogCompare:

    def __init__(self):
        self._runtime = []
        self._manual = []

    def runtime(self, log: str):
        self._runtime.append(str(log))

    def manual(self, log: str):
        self._manual.append(str(log))

    def diff(self):
        text = "\n--------- runtime ----------"
        for it in self._runtime:
            text += f"\n{it}"
        text += "\n--------- manual ----------"
        for it in self._manual:
            text += f"\n{it}"
        text += "\n----------------------------"
        return text

    def compare(self):
        if len(self._runtime) != len(self._manual):
            raise RuntimeError(self.diff())
        for a, b in zip(self._runtime, self._manual):
            if a != b:
                raise RuntimeError(self.diff())

    def clear(self):
        self._runtime.clear()
        self._manual.clear()

    def compare_and_clear(self):
        self.compare()
        self.clear()

class SupportLogCompare():
    _queue = queue.Queue()

    @classmethod
    def clear_common(cls):
        while not cls._queue.empty():
            cls._queue.get()

    @classmethod
    def get_instance(cls):
        assert not cls._queue.empty()
        log, core = cls._queue.get()
        return log, core

    def _init_log_compare(self):
        self.log = _LogCompare()
        assert SupportLogCompare._queue.empty()
        SupportLogCompare._queue.put((self.log, self))


@contextmanager
def _mock_parallel(pp=1, dp=1, pcp=1, tp=1, dcp=1, rank=0):
    @dataclass
    class GroupCoordinator:
        world_size: int = 1
        rank_in_group: int = 0
        local_rank: int = 0

    world_size = pp * dp * pcp * tp
    assert rank >= 0 and rank < world_size
    with patch.multiple(
            "vllm.distributed.parallel_state",
            _WORLD=GroupCoordinator(world_size, rank, rank % 16),
            _PP=GroupCoordinator(pp, rank // (dp * pcp * tp) % pp),
            _DP=GroupCoordinator(dp, rank // (pcp * tp) % dp),
            _PCP=GroupCoordinator(pcp, rank // tp % pcp),
            _TP=GroupCoordinator(tp, rank % tp),
            _DCP=GroupCoordinator(dcp, rank % dcp),
        ):
        yield


@contextmanager
def _mock_llm_datadist():
    from llm_datadist import (
        LLMRole,
        CacheDesc,
        BlocksCacheKey,
        LLMStatusCode,
    )

    @dataclass
    class LLMConfig:
        device_id = None
        local_comm_res = None
        sync_kv_timeout = None
        enable_remote_cache_accessible = None
        listen_ip_info = None
        transfer_backend = None

        def generate_options(self) -> dict:
            return {
                "device_id": self.device_id,
                "local_comm_res": self.local_comm_res,
                "sync_kv_timeout": self.sync_kv_timeout,
                "enable_remote_cache_accessible": self.enable_remote_cache_accessible,
                "listen_ip_info": self.listen_ip_info,
                "transfer_backend": self.transfer_backend,
            }

    @dataclass
    class Cache:
        cache_id: int = None

    @dataclass
    class LLMClusterInfo:
        remote_cluster_id: int = None
        _remote = None
        _local = None
        def append_remote_ip_info(self, ip, port):
            self._remote = (ip, port)
        def append_local_ip_info(self, ip, port):
            self._local = (ip, port)

    def valid_addr(ip, port):
        assert type(ip) is str
        assert type(port) is int
        s = ip.split(".")
        assert len(s) == 4 # ipv4
        for it in s:
            assert 0 <= int(it) < 256

    _servers = {}

    class LLMDataDist(SupportLogCompare):

        def __init__(self, role: LLMRole, cluster_id: int):
            self._init_log_compare()
            assert role in [LLMRole.PROMPT, LLMRole.DECODER]
            assert type(cluster_id) is int
            self.cluster_id = cluster_id
            self.inited = False
            self.keys = {} # key -> id
            self.caches = {} # id -> key
            self.links = {} # cid -> link

            self.log.runtime(f"__init__: {role == LLMRole.PROMPT}")

        def _manual___init__(self, is_prefill: bool):
            self.log.manual(f"__init__: {is_prefill}")

        def init(self, options: dict):
            assert not self.inited
            assert type(options) is dict
            assert type(options["device_id"]) is int
            assert type(options["local_comm_res"]) is str
            assert type(options["sync_kv_timeout"]) is int
            assert type(options["enable_remote_cache_accessible"]) is bool
            ip_info = options["listen_ip_info"]
            if ip_info is not None:
                assert type(ip_info) is str
                self.addr = f"{ip_info}:{self.cluster_id}"
                assert self.addr not in _servers
                _servers[self.addr] = self
            else:
                self.addr = None

            self.inited = True
            self.log.runtime(f"init {ip_info}")

        def _manual_init(self, ip_info: str):
            self.log.manual(f"init {ip_info}")

        def finalize(self):
            assert self.inited
            if self.addr is not None:
                assert self.addr in _servers
                _servers.pop(self.addr)
            self.inited = False
            self.log.runtime(f"finalize")

        def _manual_finalize(self):
            self.log.manual(f"finalize")

        def link_clusters(self, infos: list, timeout: int, force=False) -> bool:
            assert type(infos) is list
            assert type(timeout) is int
            assert not force
            for info in infos:
                assert type(info) is LLMClusterInfo
                assert type(info.remote_cluster_id) is int
                cid = info.remote_cluster_id
                assert cid not in self.links
                valid_addr(*info._remote)
                valid_addr(*info._local)
                ip, port = info._remote
                addr = f"{ip}:{port}:{cid}"
                if addr not in _servers:
                    return LLMStatusCode.LLM_FAILED, None
                server: LLMDataDist = _servers[addr]
                server.links[self.cluster_id] = "passive link"
                self.links[cid] = addr
                self.log.runtime(f"link: {addr}")
            return LLMStatusCode.LLM_SUCCESS, None

        def _manual_link_clusters(self, addr):
            self.log.manual(f"link: {addr}")

        def unlink_clusters(self, infos: list, timeout: int, force=False):
            assert type(infos) is list
            assert type(timeout) is int
            for info in infos:
                assert type(info) is LLMClusterInfo
                assert type(info.remote_cluster_id) is int
                cid = info.remote_cluster_id
                if not force:
                    assert cid in self.links
                self.links.pop(cid, None)
                self.log.runtime(f"unlink: {cid}")
            return LLMStatusCode.LLM_SUCCESS, None

        def _manual_unlink_clusters(self, cid):
            self.log.manual(f"unlink: {cid}")

        @property
        def cache_manager(self):
            return SimpleNamespace(
                register_blocks_cache=self._register_blocks_cache,
                unregister_cache=self._unregister_cache,
                pull_blocks=self._pull_blocks,
            )

        def _pull_blocks(self,
            src_cache_key: BlocksCacheKey,
            dst_cache: Cache,
            src_blocks: list[int],
            dst_blocks: list[int],
        ):
            assert type(src_cache_key) is BlocksCacheKey
            assert type(dst_cache) is Cache
            assert type(src_blocks) is list
            assert type(src_blocks) is list
            assert len(src_blocks) == len(dst_blocks)
            assert dst_cache.cache_id in self.caches
            cid = src_cache_key.prompt_cluster_id
            model_id = src_cache_key.model_id
            assert cid in self.links
            addr = self.links[cid]
            assert addr in _servers
            server: LLMDataDist = _servers[addr]
            key = f"{model_id}:{cid}"
            assert key in server.keys
            self.log.runtime(f"pull: {cid}")

        def _manual_pull_blocks(self, cid):
            self.log.manual(f"pull: {cid}")

        def _register_blocks_cache(self,
            cache_desc: CacheDesc,
            addrs: list[int],
            blocks_cache_key: BlocksCacheKey = None,
        ):
            assert self.inited
            assert type(cache_desc) is CacheDesc
            assert type(addrs) is list
            for it in addrs:
                assert type(it) is int
            cache_id = uuid.uuid4().int

            if blocks_cache_key is not None:
                assert type(blocks_cache_key) is BlocksCacheKey
                assert self.cluster_id == blocks_cache_key.cluster_id
                key = f"{blocks_cache_key.model_id}:{self.cluster_id}"
                self.keys[key] = cache_id
            else:
                key = None

            self.caches[cache_id] = key
            self.log.runtime(f"register: {cache_desc.num_tensors}")
            return Cache(cache_id)

        def _manual_register_blocks_cache(self, num_tensors: int):
            self.log.manual(f"register: {num_tensors}")

        def _unregister_cache(self, cache_id: int):
            assert cache_id in self.caches
            key = self.caches.pop(cache_id)
            if key is not None:
                assert self.keys.pop(key) == cache_id
            self.log.runtime(f"unregister")

        def _manual_unregister_cache(self):
            self.log.manual(f"unregister")

    import llm_datadist as module
    with _patch_module(module,
        Cache=Cache,
        LLMConfig=LLMConfig,
        LLMClusterInfo=LLMClusterInfo,
        LLMDataDist=LLMDataDist,
    ):
        SupportLogCompare.clear_common()
        def get_instance() -> tuple[_LogCompare, LLMDataDist]:
            log, core = SupportLogCompare.get_instance()
            return log, core
        yield get_instance


class TestDatadistEngine:

    def test_link_and_sync_kv_timeout_defaults(self):
        """Class-level timeouts come from LINK_TIMEOUT / SYNC_KV_TIMEOUT env (default 120000)."""
        assert DatadistEngine.LINK_TIMEOUT == int(
            __import__("os").environ.get("LINK_TIMEOUT", "120000")
        )
        assert DatadistEngine.SYNC_KV_TIMEOUT == int(
            __import__("os").environ.get("SYNC_KV_TIMEOUT", "120000")
        )

    def test_init_with_port(self, rank=7, port=10000):
        from omni.connector.utils import get_local_ip
        ip = get_local_ip()
        with (
            _mock_parallel(tp=16, rank=rank),
            _mock_llm_datadist() as get_instance,
        ):
            engine = DatadistEngine(port, is_prefill=True)
            log, core = get_instance()
            core._manual___init__(is_prefill=True)
            core._manual_init(f"{ip}:{port + rank}")
            log.compare_and_clear()

    def test_init_prefill_with_hixl_backend(self, rank=7, port=10000):
        """Prefill + hixl: listen_ip_info uses fixed port (prefill takes priority)."""
        from omni.connector.utils import get_local_ip
        ip = get_local_ip()
        with (
            patch("omni.connector.llmdatadist_manager_v1.on_ascend950", return_value=True),
            patch.object(DatadistEngine, "_get_local_comm_res", return_value='{"k":"v"}'),
            _mock_parallel(tp=16, rank=rank),
            _mock_llm_datadist() as get_instance,
        ):
            engine = DatadistEngine(port, is_prefill=True)
            log, core = get_instance()
            core._manual___init__(is_prefill=True)
            # prefill branch: fixed port
            core._manual_init(f"{ip}:{port + rank}")
            log.compare_and_clear()
            # verify transfer_backend is set to hixl
            assert engine.cfg["transfer_backend"] == "hixl"

    def test_init_decoder_with_hixl_backend(self, rank=7, port=10000):
        """Decoder + hixl: listen_ip_info uses port 0 (random assignment)."""
        from omni.connector.utils import get_local_ip
        ip = get_local_ip()
        with (
            patch("omni.connector.llmdatadist_manager_v1.on_ascend950", return_value=True),
            patch.object(DatadistEngine, "_get_local_comm_res", return_value='{"k":"v"}'),
            _mock_parallel(tp=16, rank=rank),
            _mock_llm_datadist() as get_instance,
        ):
            engine = DatadistEngine(port, is_prefill=False)
            log, core = get_instance()
            core._manual___init__(is_prefill=False)
            # hixl backend (non-prefill) branch: port 0
            core._manual_init(f"{ip}:0")
            log.compare_and_clear()
            # verify transfer_backend is set to hixl
            assert engine.cfg["transfer_backend"] == "hixl"

    def test_init_decoder_without_hixl_backend(self, rank=7, port=10000):
        """Decoder + no hixl: listen_ip_info is None (no binding)."""
        with (
            patch("omni.connector.llmdatadist_manager_v1.on_ascend950", return_value=False),
            _mock_parallel(tp=16, rank=rank),
            _mock_llm_datadist() as get_instance,
        ):
            engine = DatadistEngine(port, is_prefill=False)
            log, core = get_instance()
            core._manual___init__(is_prefill=False)
            core._manual_init(ip_info=None)
            log.compare_and_clear()
            # no hixl backend, transfer_backend should be None
            assert engine.cfg["transfer_backend"] is None

    def test_register_unregister_sleep_weakup(self, cnt = 16, port=10000):
        bufs = [torch.zeros(10, 100, dtype=torch.uint8)] * cnt

        with (
            _mock_parallel(tp=16, rank=7),
            _mock_llm_datadist() as get_instance,
        ):
            engine = DatadistEngine(port, is_prefill=False)
            log, core = get_instance()
            core._manual___init__(is_prefill=False)
            core._manual_init(ip_info=None)
            log.compare_and_clear()

            cache_id = engine.register(bufs)
            assert type(cache_id) is int
            core._manual_register_blocks_cache(num_tensors=cnt)
            log.compare_and_clear()

            engine.unregister()
            core._manual_unregister_cache()
            log.compare_and_clear()

            engine.register(bufs)
            log.clear()

            engine.sleep()
            core._manual_unregister_cache()
            core._manual_finalize()
            log.compare_and_clear()

            engine.weakup()
            core._manual_init(ip_info=None)
            log.compare_and_clear()

    def test_pull_and_recycle(self, port=10000, cnt=16):
        buf1 = [torch.zeros(10, 100, dtype=torch.uint8)] * cnt
        buf2 = [torch.zeros(20, 200, dtype=torch.uint8)] * cnt
        buf3 = [torch.zeros(20, 200, dtype=torch.uint8)] * cnt

        with (
            _patch_module(DatadistEngine,
                LINK_RECYCLE_DELAY = 2.0,
                HEARTBEAT_INTERVAL = 0.2,
            ), # smaller timeout for faster test
            _patch_module(ClientEngine,
                RETRY_DELAY = 0.3,
            ), # smaller delay for faster test
            _mock_parallel(tp=16, rank=7),
            _mock_llm_datadist() as get_instance,
        ):
            server = ServerEngine(port)
            server_log, server_core = get_instance()
            server_log.clear()

            client = ClientEngine(port + 1000)
            client_log, client_core = get_instance()
            client_log.clear()

            # register caches sequently
            server.register(buf1) # model_id=0
            server.register(buf2) # model_id=1
            server.register(buf3) # model_id=2
            server_core._manual_register_blocks_cache(num_tensors=cnt)
            server_core._manual_register_blocks_cache(num_tensors=cnt)
            server_core._manual_register_blocks_cache(num_tensors=cnt)
            server_log.compare_and_clear()

            cache_id1 = client.register(buf3)
            client_core._manual_register_blocks_cache(num_tensors=cnt)
            client_log.compare_and_clear()

            time.sleep(0.5) # wait for engine ready

            # fail1: invalid addr
            pull_ok = client.pull_blocks(
                addr="invalid addr",
                model_id=0,
                cache_id=cache_id1,
                p_blocks=[0, 1, 2],
                d_blocks=[0, 1, 2],
            )
            assert not pull_ok

            # fail2: invalid cache_id
            pull_ok = client.pull_blocks(
                addr=server.addr,
                model_id=0,
                cache_id=236647523762, # invalid cache_id
                p_blocks=[0, 1, 2],
                d_blocks=[0, 1, 2],
            )
            assert not pull_ok

            # fail3: invalid server
            pull_ok = client.pull_blocks(
                addr="192.168.0.1:1234:4321:38274528734649", # invalid server
                model_id=0,
                cache_id=cache_id1,
                p_blocks=[0, 1, 2],
                d_blocks=[0, 1, 2],
            )
            assert not pull_ok

            # reset status
            time.sleep(0.5)
            client_log.clear()

            # pull blocks and trigger dynamic link
            pull_ok = client.pull_blocks(
                addr=server.addr,
                model_id=0,
                cache_id=cache_id1,
                p_blocks=[0, 1, 2],
                d_blocks=[0, 1, 2],
            )
            assert pull_ok
            # only client links, server makes passive link
            client_core._manual_link_clusters(addr=server_core.addr)
            client_core._manual_pull_blocks(cid=server_core.cluster_id)
            server_log.compare_and_clear()
            client_log.compare_and_clear()

            time.sleep(1.0) # before timeout (<2s), assert no operation
            server_log.compare_and_clear()
            client_log.compare_and_clear()

            # pull again, link exists, hb freshed
            pull_ok = client.pull_blocks(
                addr=server.addr,
                model_id=1,
                cache_id=cache_id1,
                p_blocks=[4, 5, 6],
                d_blocks=[6, 7, 8],
            )
            client_core._manual_pull_blocks(cid=server_core.cluster_id)
            server_log.compare_and_clear()
            client_log.compare_and_clear()

            time.sleep(3.0) # after link timeout(>2s), assert unlink
            server_core._manual_unlink_clusters(cid=client_core.cluster_id)
            client_core._manual_unlink_clusters(cid=server_core.cluster_id)
            server_log.compare_and_clear()
            client_log.compare_and_clear()

            # Stop the ServerEngine/ClientEngine recycle daemons so their threads
            # don't leak into later tests (e.g. test_kv_dump's global time.sleep patch).
            _stop_daemon(server, client)


@contextmanager
def _mock_datadist_engine():

    class DatadistEngine(SupportLogCompare):
        def __init__(self, port, is_prefill):
            self.is_server = False
            if is_prefill: # is server
                assert type(port) is int
                assert port > 0 and port < 65536
                self.is_server = True
            self.caches = set()
            self.inited = True

        def sleep(self):
            self.inited = False
            self.log.runtime("sleep")

        def weakup(self):
            self.inited = True
            self.log.runtime("weakup")

        def register(self, bufs: list[torch.Tensor]) -> int:
            assert self.inited
            assert all(it.dtype == torch.uint8 for it in bufs)
            assert len({str(it.shape) for it in bufs}) == 1
            cache_id = uuid.uuid4().int
            self.caches.add(cache_id)
            self.log.runtime(f"register {[it.data_ptr() for it in bufs]}")
            return cache_id

        def unregister(self):
            assert self.inited
            self.log.runtime("unregister")
            self.caches.clear()

    class ServerEngine(DatadistEngine):
        def __init__(self, port: int):
            self._init_log_compare()
            super().__init__(port, is_prefill=True)
            self.addr = "[server-addr]"

    class ClientEngine(DatadistEngine):
        def __init__(self, port: int):
            self._init_log_compare()
            super().__init__(port, is_prefill=False)
            self._pull_ok = True # edittable

        def pull_blocks(
            self,
            addr: str,
            model_id: int,
            cache_id: int,
            p_blocks: list[int],
            d_blocks: list[int],
        ) -> bool:
            assert len(p_blocks) == len(d_blocks)
            if not self.inited:
                return False
            if cache_id not in self.caches:
                return False
            try: # parse and validate addr
                ip, s_port, e_port, cid = tuple(addr.split(":"))
                s_port, e_port, cid = int(s_port), int(e_port), int(cid)
            except:
                return False
            self.log.manual(f"pull_blocks: {model_id}")
            return self._pull_ok # edittable

    import omni.connector.llmdatadist_manager_v1 as module
    with _patch_module(module,
        DatadistEngine=DatadistEngine,
        ServerEngine=ServerEngine,
        ClientEngine=ClientEngine,
    ):
        SupportLogCompare.clear_common()
        def get_instance() -> tuple[_LogCompare, ServerEngine | ClientEngine]:
            log, core = SupportLogCompare.get_instance()
            return log, core
        yield get_instance


class TestCacheManager:

    def test_system(self):
        # simplest one-layer one-cache
        vllm_caches = [(torch.zeros(100, 128, 512, dtype=torch.bfloat16), )]
        with (
            _mock_parallel(tp=16, rank=7),
            _mock_datadist_engine() as get_instance,
        ):
            server = CacheManager(23424, is_prefill=True)
            server_log, server_core = get_instance()
            client = CacheManager(25424, is_prefill=False)
            client_log, client_core = get_instance()
            server_log.clear()
            client_log.clear()

            server.sleep()
            server_log.manual("sleep")
            server_log.compare_and_clear()

            client.sleep()
            client_log.manual("sleep")
            client_log.compare_and_clear()

            # should not register or unregister during sleep
            with pytest.raises(Exception):
                server.unregister()
            with pytest.raises(Exception):
                client.unregister()
            with pytest.raises(Exception):
                server.register(vllm_caches)
            with pytest.raises(Exception):
                client.register(vllm_caches)
            server_log.clear()
            client_log.clear()

            server.weakup()
            server_log.manual("weakup")
            server_log.compare_and_clear()

            client.weakup()
            client_log.manual("weakup")
            client_log.compare_and_clear()

            server.unregister()
            client.unregister()
            server.register(vllm_caches)
            client.register(vllm_caches)

            pull_args = dict(
                addr=server.engine_addr,
                p_blocks=[0],
                d_blocks=[0],
                layer_ids=[0],
            )
            client.pull_blocks(**pull_args)

            # server should not pull
            with pytest.raises(Exception):
                server.pull_blocks(**pull_args)

    def test_register_list(self):
        pass

    def test_register_dict(self):
        pass

    def test_pull_selected_layers(self):
        pass


class TestGetPhysicalDeviceId:
    """Tests for LLMDataDistManager._get_physical_device_id."""

    def _make_manager(self, local_rank=0):
        """Create a minimal manager without calling __init__."""
        mgr = DatadistEngine.__new__(DatadistEngine)
        mgr.local_rank = local_rank
        return mgr

    def test_returns_local_rank_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
        mgr = self._make_manager(local_rank=2)
        assert mgr._get_physical_device_id() == 2

    def test_maps_local_rank_to_visible_device(self, monkeypatch):
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
        mgr = self._make_manager(local_rank=1)
        assert mgr._get_physical_device_id() == 5

    def test_first_device_when_local_rank_zero(self, monkeypatch):
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
        mgr = self._make_manager(local_rank=0)
        assert mgr._get_physical_device_id() == 0

    def test_fallback_to_first_device_when_local_rank_exceeds_count(self, monkeypatch):
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5")
        mgr = self._make_manager(local_rank=5)
        assert mgr._get_physical_device_id() == 4

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", " 4 , 5 , 6 ")
        mgr = self._make_manager(local_rank=1)
        assert mgr._get_physical_device_id() == 5

    def test_empty_string_env_returns_local_rank(self, monkeypatch):
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
        mgr = self._make_manager(local_rank=3)
        assert mgr._get_physical_device_id() == 3


class TestGetClusterDeviceId:
    """Tests for DatadistEngine._get_cluster_device_id."""

    def _make_manager(self, local_rank=0):
        mgr = DatadistEngine.__new__(DatadistEngine)
        mgr.local_rank = local_rank
        return mgr

    def test_maps_physical_index_to_cluster_device_id(self, monkeypatch):
        rootinfo = {
            "rank_list": [
                {"device_id": 100},
                {"device_id": 101},
                {"device_id": 102},
            ]
        }
        with _mock_hccl_rootinfo(rootinfo):
            monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2")
            mgr = self._make_manager(local_rank=1)
            assert mgr._get_cluster_device_id() == 101

    def test_maps_visible_device_through_hccl_rootinfo(self, monkeypatch):
        rootinfo = {"rank_list": [{"device_id": i * 10} for i in range(8)]}
        with _mock_hccl_rootinfo(rootinfo):
            monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
            mgr = self._make_manager(local_rank=1)
            assert mgr._get_physical_device_id() == 5
            assert mgr._get_cluster_device_id() == 50

    def test_fallback_when_file_not_found(self, monkeypatch):
        with _mock_hccl_rootinfo(open_side_effect=FileNotFoundError):
            monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
            mgr = self._make_manager(local_rank=3)
            assert mgr._get_cluster_device_id() == 3

    def test_fallback_when_index_out_of_range(self, monkeypatch):
        rootinfo = {"rank_list": [{"device_id": 0}, {"device_id": 1}]}
        with _mock_hccl_rootinfo(rootinfo):
            monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
            mgr = self._make_manager(local_rank=5)
            assert mgr._get_cluster_device_id() == 5

    def test_fallback_when_device_id_missing(self, monkeypatch):
        rootinfo = {"rank_list": [{"local_id": 0}, {"device_id": 7}]}
        with _mock_hccl_rootinfo(rootinfo):
            monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
            mgr = self._make_manager(local_rank=0)
            assert mgr._get_cluster_device_id() == 0

    def test_fallback_on_invalid_json(self, monkeypatch):
        with _mock_hccl_rootinfo(load_side_effect=ValueError("invalid json")):
            monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
            mgr = self._make_manager(local_rank=2)
            assert mgr._get_cluster_device_id() == 2

    def test_fallback_when_rank_list_empty(self, monkeypatch):
        with _mock_hccl_rootinfo({"rank_list": []}):
            monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
            mgr = self._make_manager(local_rank=1)
            assert mgr._get_cluster_device_id() == 1


class TestGetLocalCommRes:
    """Tests for LLMDataDistManager._get_local_comm_res."""

    def _make_manager(self, local_rank=0):
        mgr = DatadistEngine.__new__(DatadistEngine)
        mgr.local_rank = local_rank
        return mgr

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("HIXL_LOCAL_COMM_RES_ENABLE", raising=False)
        mgr = self._make_manager()
        assert mgr._get_local_comm_res() == ""

    def test_disabled_when_false(self, monkeypatch):
        monkeypatch.setenv("HIXL_LOCAL_COMM_RES_ENABLE", "false")
        mgr = self._make_manager()
        assert mgr._get_local_comm_res() == ""

    def test_disabled_when_zero(self, monkeypatch):
        monkeypatch.setenv("HIXL_LOCAL_COMM_RES_ENABLE", "0")
        mgr = self._make_manager()
        assert mgr._get_local_comm_res() == ""

    def test_disabled_when_no(self, monkeypatch):
        monkeypatch.setenv("HIXL_LOCAL_COMM_RES_ENABLE", "no")
        mgr = self._make_manager()
        assert mgr._get_local_comm_res() == ""

    def test_enabled_loads_json_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIXL_LOCAL_COMM_RES_ENABLE", "true")
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("HIXLP_ENDPOINT_PATH", str(tmp_path))

        json_file = tmp_path / "ub_endpoint_npu_0.json"
        json_file.write_text('{"key": "value"}')

        mgr = self._make_manager()
        result = mgr._get_local_comm_res()
        assert result == '{"key":"value"}'

    def test_enabled_missing_file_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIXL_LOCAL_COMM_RES_ENABLE", "1")
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("HIXLP_ENDPOINT_PATH", str(tmp_path))

        mgr = self._make_manager()
        assert mgr._get_local_comm_res() == ""

    def test_enabled_corrupt_json_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIXL_LOCAL_COMM_RES_ENABLE", "yes")
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("HIXLP_ENDPOINT_PATH", str(tmp_path))

        json_file = tmp_path / "ub_endpoint_npu_0.json"
        json_file.write_text("not valid json{")

        mgr = self._make_manager()
        assert mgr._get_local_comm_res() == ""

    def test_uses_physical_device_id_in_filename(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIXL_LOCAL_COMM_RES_ENABLE", "1")
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
        monkeypatch.setenv("HIXLP_ENDPOINT_PATH", str(tmp_path))

        # local_rank=1 -> physical_device_id=5
        json_file = tmp_path / "ub_endpoint_npu_5.json"
        json_file.write_text('{"device": 5}')

        mgr = self._make_manager(local_rank=1)
        result = mgr._get_local_comm_res()
        assert result == '{"device":5}'

    def test_default_base_path(self, monkeypatch):
        monkeypatch.setenv("HIXL_LOCAL_COMM_RES_ENABLE", "1")
        monkeypatch.delenv("HIXLP_ENDPOINT_PATH", raising=False)
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")

        mgr = self._make_manager()
        # /etc/hixlep won't exist in test, so should return ""
        assert mgr._get_local_comm_res() == ""
