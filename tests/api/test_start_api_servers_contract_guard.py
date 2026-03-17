# test_start_api_servers_contract_guard.py
#
# start_api_servers.py 接口契约看护测试 （包括RL新增接口）
#
# 目的：
#   - 锁定 start_api_servers 脚本层的稳定导出接口与调用契约
#   - 防止参数名 / 默认值 / 命令拼装 / 环境变量约定发生破坏性变更
#   - 作为 omni-RL / 部署脚本 与 start_api_servers 之间的接口边界约束
#
# 说明：
#   - 本测试不启动真实 vLLM 服务
#   - 本测试不依赖真实模型
#   - 本测试不验证推理行为
#   - 本测试仅关注脚本层 contract，不覆盖 vLLM 自身公共 API
#
# 变更说明：
#   - 本文件为脚本级接口看护测试
#   - 如需修改本测试或其覆盖的接口，请先联系框架组负责人（王锐：00580756 李泽宇：00959921）评估


import inspect
import io
import os
import subprocess
import types
import weakref
import signal
import pytest
import importlib.util
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    raise FileNotFoundError("Cannot locate repo root from current test file path.")


def _load_module():
    repo_root = _find_repo_root(Path(__file__).parent)
    
    target = repo_root / "tools" / "scripts" / "start_api_servers.py"
    assert target.exists(), f"target file not found: {target}"

    spec = importlib.util.spec_from_file_location("start_api_servers", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# ==============================================================================
# 1. 模块导出入口（只锁脚本层核心函数 / 类）
# ==============================================================================

def test_module_exports_contract():
    for name in [
        "is_port_available",
        "find_available_port",
        "ProcessManager",
        "process_space_split",
        "process_extra_args",
        "start_single_node_api_servers",
        "signal_handler",
    ]:
        assert hasattr(mod, name), f"missing export: {name}"


# ==============================================================================
# 2. is_port_available（工具函数：锁签名）
# ==============================================================================

def test_is_port_available_contract():
    sig = inspect.signature(mod.is_port_available)
    params = sig.parameters

    assert list(params.keys()) == ["port", "host"]
    assert params["host"].default == "0.0.0.0"


# ==============================================================================
# 3. find_available_port（工具函数：锁签名 + 扫描语义）
# ==============================================================================

def test_find_available_port_contract():
    sig = inspect.signature(mod.find_available_port)
    params = sig.parameters

    assert list(params.keys()) == ["base_port", "max_attempts", "host"]
    assert params["max_attempts"].default == 10
    assert params["host"].default == "0.0.0.0"


def test_find_available_port_returns_first_available(monkeypatch):
    calls = []

    def fake_is_port_available(port, host="0.0.0.0"):
        calls.append((port, host))
        return port == 9002

    monkeypatch.setattr(mod, "is_port_available", fake_is_port_available)

    port = mod.find_available_port(9000, max_attempts=5, host="127.0.0.1")
    assert port == 9002
    assert calls == [
        (9000, "127.0.0.1"),
        (9001, "127.0.0.1"),
        (9002, "127.0.0.1"),
    ]


def test_find_available_port_raises_when_no_port_available(monkeypatch):
    monkeypatch.setattr(mod, "is_port_available", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError) as exc:
        mod.find_available_port(9000, max_attempts=3)

    assert "No available port found between 9000 and 9002" in str(exc.value)


# ==============================================================================
# 4. ProcessManager（轻量包装对象：锁构造签名 + 成员承载）
# ==============================================================================

def test_process_manager_contract():
    assert inspect.isclass(mod.ProcessManager)

    sig = inspect.signature(mod.ProcessManager.__init__)
    assert list(sig.parameters.keys()) == ["self", "processes"]

    pm = mod.ProcessManager(processes=[("p", "l")])
    assert hasattr(pm, "processes")
    assert pm.processes == [("p", "l")]


# ==============================================================================
# 5. process_space_split / process_extra_args（额外参数解析：锁语义）
# ==============================================================================

def test_process_space_split_contract():
    sig = inspect.signature(mod.process_space_split)
    assert list(sig.parameters.keys()) == ["arg_temp", "out_list"]


def test_process_extra_args_contract():
    sig = inspect.signature(mod.process_extra_args)
    assert list(sig.parameters.keys()) == ["extra_args"]


def test_process_space_split_for_compilation_config():
    out = []
    ret = mod.process_space_split(
        '--compilation-config {"level":1,"backend":"inductor"}',
        out,
    )
    assert ret is out
    assert out == [
        "--compilation-config",
        '{"level":1,"backend":"inductor"}',
    ]


def test_process_space_split_general_case():
    out = []
    ret = mod.process_space_split("--tensor-parallel-size 4", out)
    assert ret is out
    assert out == ["--tensor-parallel-size", "4"]


def test_process_extra_args_general_case():
    s = "--enable-expert-parallel --max-num-seqs 256 --disable-log-requests"
    out = mod.process_extra_args(s)
    assert out == [
        "--enable-expert-parallel",
        "--max-num-seqs", "256",
        "--disable-log-requests",
    ]


def test_process_extra_args_keeps_compilation_config_value_intact():
    s = '--foo bar --compilation-config {"mode":3,"use_inductor":true} --baz qux'
    out = mod.process_extra_args(s)
    assert out == [
        "--foo", "bar",
        "--compilation-config", '{"mode":3,"use_inductor":true}',
        "--baz", "qux",
    ]


def test_process_extra_args_empty_string():
    assert mod.process_extra_args("") == []


# ==============================================================================
# 6. start_single_node_api_servers（核心：锁签名）
# ==============================================================================

def test_start_single_node_api_servers_signature_contract():
    sig = inspect.signature(mod.start_single_node_api_servers)
    params = sig.parameters

    assert list(params.keys()) == [
        "num_servers",
        "model_path",
        "base_api_port",
        "master_ip",
        "master_port",
        "total_dp_size",
        "gpu_util",
        "served_model_name",
        "server_offset",
        "block_size",
        "tp",
        "pp",
        "distributed_executor_backend",
        "kv_transfer_config",
        "log_dir",
        "max_port_attempts",
        "max_tokens",
        "load_format",
        "extra_args",
        "additional_config",
        "enable_mtp",
        "no_enable_prefix_caching",
        "num_speculative_tokens",
        "no_enable_chunked_prefill",
    ]

    assert params["server_offset"].default == 0
    assert params["block_size"].default == 128
    assert params["tp"].default == 1
    assert params["pp"].default == 1
    assert params["distributed_executor_backend"].default is None
    assert params["kv_transfer_config"].default is None
    assert params["log_dir"].default == "logs"
    assert params["max_port_attempts"].default == 10
    assert params["max_tokens"].default == 4096
    assert params["load_format"].default == "auto"
    assert params["extra_args"].default is None
    assert params["additional_config"].default is None
    assert params["enable_mtp"].default is False
    assert params["no_enable_prefix_caching"].default is False
    assert params["num_speculative_tokens"].default == 1
    assert params["no_enable_chunked_prefill"].default is False


# ==============================================================================
# 7. start_single_node_api_servers（参数校验：additional_config JSON）
# ==============================================================================

def test_start_single_node_api_servers_rejects_invalid_additional_config():
    with pytest.raises(ValueError) as exc:
        mod.start_single_node_api_servers(
            num_servers=1,
            model_path="/fake/model",
            base_api_port=9000,
            master_ip="127.0.0.1",
            master_port=8000,
            total_dp_size=1,
            gpu_util=0.9,
            served_model_name="fake-model",
            additional_config='{"bad_json": ]',
        )

    assert "additional_config must be a valid JSON string" in str(exc.value)


# ==============================================================================
# 8. start_single_node_api_servers（命令拼装 + 环境变量约定）
# ==============================================================================

class _DummyProc:
    def __init__(self, pid=12345, poll_result=None):
        self.pid = pid
        self._poll_result = poll_result
        self.terminate_called = 0
        self.kill_called = 0
        self.wait_calls = []

    def poll(self):
        return self._poll_result

    def terminate(self):
        self.terminate_called += 1

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return 0

    def kill(self):
        self.kill_called += 1


def test_start_single_node_api_servers_basic_command_and_env(monkeypatch, tmp_path):
    spawned = []
    finalized = []

    def fake_makedirs(path, exist_ok=False):
        assert path == str(tmp_path)
        assert exist_ok is True

    def fake_find_available_port(base_port, max_attempts=10, host="0.0.0.0"):
        assert max_attempts == 7
        return 9100

    def fake_open(path, mode):
        assert mode == "w"
        return io.StringIO()

    def fake_popen(cmd, env, stdout, stderr):
        proc = _DummyProc(pid=20001, poll_result=None)
        spawned.append(
            {
                "cmd": cmd,
                "env": env,
                "stdout": stdout,
                "stderr": stderr,
                "proc": proc,
            }
        )
        return proc

    def fake_finalize(obj, func):
        finalized.append((obj, func))
        return object()

    monkeypatch.setattr(mod.os, "makedirs", fake_makedirs)
    monkeypatch.setattr(mod, "find_available_port", fake_find_available_port)
    monkeypatch.setattr(mod, "open", fake_open, raising=False)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod.weakref, "finalize", fake_finalize)
    monkeypatch.setattr(mod.os, "environ", {})

    processes, process_manager, ports = mod.start_single_node_api_servers(
        num_servers=1,
        model_path="/fake/model",
        base_api_port=9000,
        master_ip="10.0.0.1",
        master_port=8000,
        total_dp_size=4,
        gpu_util=0.85,
        served_model_name="my-model",
        server_offset=2,
        block_size=256,
        tp=2,
        pp=3,
        distributed_executor_backend="mp",
        kv_transfer_config='{"kv_connector":"p2p"}',
        log_dir=str(tmp_path),
        max_port_attempts=7,
        max_tokens=8192,
        load_format="safetensors",
    )

    assert len(processes) == 1
    assert isinstance(process_manager, mod.ProcessManager)
    assert ports == {"server_0": 9100}

    assert len(spawned) == 1
    item = spawned[0]
    cmd = item["cmd"]
    env = item["env"]

    assert cmd[:3] == ["vllm", "serve", "/fake/model"]
    assert "--trust-remote-code" in cmd
    assert "--pipeline-parallel-size" in cmd
    assert "3" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.85" in cmd
    assert "--block-size" in cmd
    assert "256" in cmd
    assert "--tensor-parallel-size" in cmd
    assert "2" in cmd
    assert "--data-parallel-address" in cmd
    assert "10.0.0.1" in cmd
    assert "--data-parallel-rpc-port" in cmd
    assert "8000" in cmd
    assert "--port" in cmd
    assert "9100" in cmd
    assert "--served-model-name" in cmd
    assert "my-model" in cmd
    assert "--max-model-len" in cmd
    assert "8192" in cmd
    assert "--load-format" in cmd
    assert "safetensors" in cmd
    assert "--distributed-executor-backend" in cmd
    assert "mp" in cmd
    assert "--kv-transfer-config" in cmd
    assert '{"kv_connector":"p2p"}' in cmd

    assert env["VLLM_DP_SIZE"] == "4"
    assert env["VLLM_DP_RANK"] == "1"          # rank=0, server_offset=2, tp=2 => 0 + 2//2 = 1
    assert env["VLLM_DP_RANK_LOCAL"] == "1"
    assert env["VLLM_DP_MASTER_IP"] == "10.0.0.1"
    assert env["VLLM_DP_MASTER_PORT"] == "8000"
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"

    assert len(finalized) == 1
    assert finalized[0][0] is process_manager
    assert callable(finalized[0][1])


def test_start_single_node_api_servers_decode_role_adds_dp_flags(monkeypatch, tmp_path):
    spawned = []

    def fake_find_available_port(*args, **kwargs):
        return 9200

    def fake_open(*args, **kwargs):
        return io.StringIO()

    def fake_popen(cmd, env, stdout, stderr):
        proc = _DummyProc(pid=20002, poll_result=None)
        spawned.append({"cmd": cmd, "env": env, "proc": proc})
        return proc

    monkeypatch.setattr(mod.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "find_available_port", fake_find_available_port)
    monkeypatch.setattr(mod, "open", fake_open, raising=False)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod.weakref, "finalize", lambda obj, func: object())
    monkeypatch.setattr(mod.os, "environ", {"ROLE": "decode"})

    mod.start_single_node_api_servers(
        num_servers=1,
        model_path="/fake/model",
        base_api_port=9000,
        master_ip="10.0.0.1",
        master_port=8000,
        total_dp_size=8,
        gpu_util=0.9,
        served_model_name="my-model",
        server_offset=4,
        tp=2,
    )

    cmd = spawned[0]["cmd"]
    assert "--data-parallel-size" in cmd
    assert "8" in cmd
    assert "--data-parallel-rank" in cmd
    assert "2" in cmd   # rank=0 + 4//2


def test_start_single_node_api_servers_optional_flags(monkeypatch, tmp_path):
    spawned = []

    def fake_find_available_port(*args, **kwargs):
        return 9300

    def fake_open(*args, **kwargs):
        return io.StringIO()

    def fake_popen(cmd, env, stdout, stderr):
        proc = _DummyProc(pid=20003, poll_result=None)
        spawned.append(cmd)
        return proc

    monkeypatch.setattr(mod.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "find_available_port", fake_find_available_port)
    monkeypatch.setattr(mod, "open", fake_open, raising=False)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod.weakref, "finalize", lambda obj, func: object())
    monkeypatch.setattr(mod.os, "environ", {})

    mod.start_single_node_api_servers(
        num_servers=1,
        model_path="/fake/model",
        base_api_port=9000,
        master_ip="127.0.0.1",
        master_port=8000,
        total_dp_size=1,
        gpu_util=0.9,
        served_model_name="my-model",
        enable_mtp=True,
        num_speculative_tokens=3,
        no_enable_prefix_caching=True,
        no_enable_chunked_prefill=True,
        extra_args="--enable-expert-parallel --max-num-seqs 128",
        additional_config='{"foo":"bar"}',
    )

    cmd = spawned[0]

    assert "--speculative_config" in cmd
    assert '{"method": "mtp", "num_speculative_tokens": 3}' in cmd

    assert "--no-enable-prefix-caching" in cmd
    assert "--no-enable-chunked-prefill" in cmd

    assert "--enable-expert-parallel" in cmd
    assert "--max-num-seqs" in cmd
    assert "128" in cmd

    assert "--additional-config" in cmd
    assert '{"foo":"bar"}' in cmd


def test_start_single_node_api_servers_skips_distributed_backend_when_none_string(monkeypatch):
    spawned = []

    monkeypatch.setattr(mod.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "find_available_port", lambda *args, **kwargs: 9400)
    monkeypatch.setattr(mod, "open", lambda *args, **kwargs: io.StringIO(), raising=False)
    monkeypatch.setattr(
        mod.subprocess,
        "Popen",
        lambda cmd, env, stdout, stderr: spawned.append(cmd) or _DummyProc(pid=20004),
    )
    monkeypatch.setattr(mod.weakref, "finalize", lambda obj, func: object())
    monkeypatch.setattr(mod.os, "environ", {})

    mod.start_single_node_api_servers(
        num_servers=1,
        model_path="/fake/model",
        base_api_port=9000,
        master_ip="127.0.0.1",
        master_port=8000,
        total_dp_size=1,
        gpu_util=0.9,
        served_model_name="my-model",
        distributed_executor_backend="None",
    )

    cmd = spawned[0]
    assert "--distributed-executor-backend" not in cmd


# ==============================================================================
# 9. start_single_node_api_servers（端口分配 / 多实例语义）
# ==============================================================================

def test_start_single_node_api_servers_multi_server_port_progression(monkeypatch):
    spawned = []
    port_calls = []

    returned_ports = [9500, 9502]

    def fake_find_available_port(base_port, max_attempts=10, host="0.0.0.0"):
        port_calls.append((base_port, max_attempts, host))
        return returned_ports[len(port_calls) - 1]

    monkeypatch.setattr(mod.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "find_available_port", fake_find_available_port)
    monkeypatch.setattr(mod, "open", lambda *args, **kwargs: io.StringIO(), raising=False)
    monkeypatch.setattr(
        mod.subprocess,
        "Popen",
        lambda cmd, env, stdout, stderr: spawned.append((cmd, env)) or _DummyProc(pid=30000 + len(spawned)),
    )
    monkeypatch.setattr(mod.weakref, "finalize", lambda obj, func: object())
    monkeypatch.setattr(mod.os, "environ", {})

    _, _, ports = mod.start_single_node_api_servers(
        num_servers=2,
        model_path="/fake/model",
        base_api_port=9500,
        master_ip="127.0.0.1",
        master_port=8000,
        total_dp_size=2,
        gpu_util=0.9,
        served_model_name="my-model",
        max_port_attempts=20,
    )

    assert port_calls == [
        (9500, 20, "0.0.0.0"),
        (9501, 20, "0.0.0.0"),
    ]
    assert ports == {
        "server_0": 9500,
        "server_1": 9502,
    }

    assert len(spawned) == 2
    assert spawned[0][1]["VLLM_DP_RANK"] == "0"
    assert spawned[1][1]["VLLM_DP_RANK"] == "1"


# ==============================================================================
# 10. start_single_node_api_servers（端口耗尽时应退出）
# ==============================================================================

def test_start_single_node_api_servers_exits_when_no_port_found(monkeypatch):
    def fake_find_available_port(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "find_available_port", fake_find_available_port)

    with pytest.raises(SystemExit) as exc:
        mod.start_single_node_api_servers(
            num_servers=1,
            model_path="/fake/model",
            base_api_port=9000,
            master_ip="127.0.0.1",
            master_port=8000,
            total_dp_size=1,
            gpu_util=0.9,
            served_model_name="my-model",
        )

    assert exc.value.code == 1


# ==============================================================================
# 11. weakref.finalize 注册的 cleanup（资源清理语义）
# ==============================================================================

def test_start_single_node_api_servers_registers_cleanup_that_terminates_and_closes(monkeypatch):
    finalized = []

    log1 = io.StringIO()
    log2 = io.StringIO()
    procs = [_DummyProc(pid=101, poll_result=None), _DummyProc(pid=102, poll_result=0)]
    open_calls = {"n": 0}

    def fake_open(*args, **kwargs):
        open_calls["n"] += 1
        return log1 if open_calls["n"] == 1 else log2

    def fake_popen(cmd, env, stdout, stderr):
        return procs.pop(0)

    def fake_finalize(obj, func):
        finalized.append((obj, func))
        return object()

    monkeypatch.setattr(mod.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "find_available_port", lambda *args, **kwargs: 9600 if open_calls["n"] == 0 else 9601)
    monkeypatch.setattr(mod, "open", fake_open, raising=False)
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod.weakref, "finalize", fake_finalize)
    monkeypatch.setattr(mod.os, "environ", {})

    processes, process_manager, _ = mod.start_single_node_api_servers(
        num_servers=2,
        model_path="/fake/model",
        base_api_port=9600,
        master_ip="127.0.0.1",
        master_port=8000,
        total_dp_size=2,
        gpu_util=0.9,
        served_model_name="my-model",
    )

    assert len(processes) == 2
    assert len(finalized) == 1

    cleanup = finalized[0][1]
    cleanup()

    # 第一个进程仍活着，应 terminate + wait
    p0 = processes[0][0]
    assert p0.terminate_called == 1
    assert p0.wait_calls == [5]

    # 第二个进程已退出，不应再次 terminate
    p1 = processes[1][0]
    assert p1.terminate_called == 0

    # 日志应关闭
    assert log1.closed is True
    assert log2.closed is True


# ==============================================================================
# 12. signal_handler（锁签名 + 清理语义）
# ==============================================================================

def test_signal_handler_contract():
    sig = inspect.signature(mod.signal_handler)
    assert list(sig.parameters.keys()) == ["sig", "frame"]


def test_signal_handler_terminates_processes_and_exits(monkeypatch):
    log1 = io.StringIO()
    log2 = io.StringIO()

    p1 = _DummyProc(pid=111, poll_result=None)
    p2 = _DummyProc(pid=222, poll_result=0)

    fake_pm = types.SimpleNamespace(
        processes=[
            (p1, log1),
            (p2, log2),
        ]
    )

    monkeypatch.setattr(mod, "process_manager", fake_pm, raising=False)

    with pytest.raises(SystemExit) as exc:
        mod.signal_handler(signal.SIGINT, None)

    assert exc.value.code == 0

    assert p1.terminate_called == 1
    assert p1.wait_calls == [5]
    assert p2.terminate_called == 0

    assert log1.closed is True
    assert log2.closed is True

