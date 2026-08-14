# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_vllm_v1_api_contract_guard.py
#
# vLLM v0.12.0（V1 API）接口契约看护测试
#
# 目的：
#   - 锁定 vLLM V1 对外接口的导包路径与调用契约
#   - 防止构造函数 / 方法签名发生破坏性变更
#   - 作为 omni-RL 与 vLLM 之间的接口边界约束
#
# 说明：
#   - 本测试不加载模型
#   - 本测试不启动 EngineCore / worker 进程
#   - 本测试仅关注接口契约，不验证推理行为
#
# 变更说明：
#   - 本文件为框架级接口看护测试
#   - 如需修改本测试或其覆盖的接口，请先联系框架组负责人（王锐：00580756 李泽宇：00959921）评估


import inspect
import importlib
import pytest


def _import(path: str):
    return importlib.import_module(path)


# ==============================================================================
# 1. WorkerWrapperBase（内部接口，仅锁最小稳定面）
# ==============================================================================

def test_worker_wrapper_base_contract():
    mod = _import("vllm.v1.worker.worker_base")
    cls = mod.WorkerWrapperBase

    assert inspect.isclass(cls)

    # 锁定构造函数签名（参数名 + 默认值 + 注解）
    sig = inspect.signature(cls.__init__)
    assert list(sig.parameters.keys()) == ["self", "rpc_rank", "global_rank"]

    p = sig.parameters
    assert p["rpc_rank"].default == 0
    assert p["global_rank"].default is None
    if p["rpc_rank"].annotation is not inspect._empty:
        assert p["rpc_rank"].annotation is int

    # 仅锁最小稳定方法集合（v1 中 sleep/wake_up/load_model 已移除）
    for name in [
        "init_worker",
        "execute_model",
        "reset_mm_cache",
        "shutdown",
    ]:
        assert hasattr(cls, name)

    # 锁部分关键方法签名（避免调用方崩溃）
    init_sig = inspect.signature(cls.init_worker)
    assert list(init_sig.parameters.keys()) == ["self", "all_kwargs"]

    # 明确“这些接口在 v1 中不再属于 WorkerWrapperBase contract”
    for removed in ["sleep", "wake_up", "load_model", "execute_model_async"]:
        assert not hasattr(cls, removed)


# ==============================================================================
# 2. LLM（同步推理入口，稳定对外 API）
# ==============================================================================

def test_llm_contract():
    from vllm import LLM

    sig = inspect.signature(LLM)
    params = sig.parameters

    assert "model" in params
    assert params["model"].annotation in (str, inspect._empty)

    assert "tensor_parallel_size" in params
    assert params["tensor_parallel_size"].default == 1
    if params["tensor_parallel_size"].annotation is not inspect._empty:
        assert params["tensor_parallel_size"].annotation is int

    # 关键公共方法存在性 + 返回类型（若有注解则锁）
    assert hasattr(LLM, "generate")
    gsig = inspect.signature(LLM.generate)
    assert list(gsig.parameters.keys())[:2] == ["self", "prompts"]
    if gsig.return_annotation is not inspect._empty:
        # vLLM 0.12.0: -> list[vllm.outputs.RequestOutput]
        assert "RequestOutput" in str(gsig.return_annotation)

    assert hasattr(LLM, "sleep")
    ssig = inspect.signature(LLM.sleep)
    # vllm 0.25.1: LLM.sleep gained mode param
    _sleep_keys = list(ssig.parameters.keys())
    assert _sleep_keys[:2] == ["self", "level"]  # mode added in 0.25.1
    assert ssig.parameters["level"].default == 1

    assert hasattr(LLM, "wake_up")
    wsig = inspect.signature(LLM.wake_up)
    assert list(wsig.parameters.keys()) == ["self", "tags"]
    assert wsig.parameters["tags"].default is None

    assert hasattr(LLM, "reset_prefix_cache")
    rsig = inspect.signature(LLM.reset_prefix_cache)
    assert "reset_running_requests" in rsig.parameters
    assert rsig.parameters["reset_running_requests"].default is False
    if rsig.return_annotation is not inspect._empty:
        assert rsig.return_annotation is bool


# ==============================================================================
# 3. AsyncLLM（异步推理入口）
# ==============================================================================

def test_async_llm_contract():
    from vllm.v1.engine.async_llm import AsyncLLM

    assert inspect.isclass(AsyncLLM)

    # from_vllm_config 的关键参数（语义反转点）
    f_sig = inspect.signature(AsyncLLM.from_vllm_config)
    f_params = f_sig.parameters
    assert "vllm_config" in f_params
    assert "enable_log_requests" in f_params
    assert f_params["enable_log_requests"].default is False
    assert "disable_log_stats" in f_params

    # generate 的返回是 AsyncGenerator[RequestOutput, None]
    assert hasattr(AsyncLLM, "generate")
    g_sig = inspect.signature(AsyncLLM.generate)
    assert list(g_sig.parameters.keys())[:4] == ["self", "prompt", "sampling_params", "request_id"]
    if g_sig.return_annotation is not inspect._empty:
        assert "AsyncGenerator" in str(g_sig.return_annotation) or "collections.abc.AsyncGenerator" in str(
            g_sig.return_annotation
        )

    # 关键公共方法存在性 + 默认值
    assert hasattr(AsyncLLM, "reset_mm_cache")
    assert inspect.signature(AsyncLLM.reset_mm_cache).return_annotation in (None, inspect._empty)

    assert hasattr(AsyncLLM, "reset_prefix_cache")
    r_sig = inspect.signature(AsyncLLM.reset_prefix_cache)
    assert r_sig.parameters["reset_running_requests"].default is False

    assert hasattr(AsyncLLM, "wait_for_requests_to_drain")
    d_sig = inspect.signature(AsyncLLM.wait_for_requests_to_drain)
    assert d_sig.parameters["drain_timeout"].default == 300


# ==============================================================================
# 4. SamplingParams（kwargs 驱动；锁“语义字段存在”，不锁构造签名）
# ==============================================================================

def test_sampling_params_contract():
    from vllm import SamplingParams

    # vLLM 0.12.0: __init__(self, /, *args, **kwargs)
    sig = inspect.signature(SamplingParams.__init__)
    kinds = [p.kind for p in sig.parameters.values()]
    assert inspect.Parameter.VAR_POSITIONAL in kinds
    assert inspect.Parameter.VAR_KEYWORD in kinds

    # 语义字段：logprobs（int | None）是 RL 依赖点
    sp = SamplingParams(max_tokens=1, logprobs=None)
    assert hasattr(sp, "logprobs")
    assert sp.logprobs is None

    sp2 = SamplingParams(max_tokens=1, logprobs=1)
    assert sp2.logprobs is None or isinstance(sp2.logprobs, int)

    # 异常输入（轻量）：get_tcp_uri 才做更严格，SamplingParams 不做强约束避免过拟合


# ==============================================================================
# 5. Compilation / LoRA Config（配置对象，锁存在性 + 模式枚举）
# ==============================================================================

def test_compilation_and_lora_config_contract():
    from vllm.config.compilation import CompilationConfig, CompilationMode
    from vllm.config import LoRAConfig

    assert inspect.isclass(CompilationConfig)
    assert inspect.isclass(CompilationMode)
    assert inspect.isclass(LoRAConfig)

    # CompilationMode 为 IntEnum（语义稳定）
    import enum
    assert issubclass(CompilationMode, enum.IntEnum)

    # PydanticDataclass 风格：构造签名不锁字段名，避免字段演进导致误报
    cc_sig = inspect.signature(CompilationConfig.__init__)
    assert cc_sig.parameters is not None

    lc_sig = inspect.signature(LoRAConfig.__init__)
    assert lc_sig.parameters is not None


# ==============================================================================
# 6. AsyncEngineArgs（稳定对外配置入口：锁关键参数+关键方法签名）
# ==============================================================================

def test_async_engine_args_contract():
    from vllm.engine.arg_utils import AsyncEngineArgs

    sig = inspect.signature(AsyncEngineArgs.__init__)
    params = sig.parameters
    assert "model" in params
    assert params["model"].default == "Qwen/Qwen3-0.6B"

    assert hasattr(AsyncEngineArgs, "from_cli_args")
    from_sig = inspect.signature(AsyncEngineArgs.from_cli_args)
    assert list(from_sig.parameters.keys()) == ["args"]

    assert hasattr(AsyncEngineArgs, "create_engine_config")
    ce_sig = inspect.signature(AsyncEngineArgs.create_engine_config)
    assert list(ce_sig.parameters.keys()) == ["self", "usage_context", "headless"]
    assert ce_sig.parameters["headless"].default is False


# ==============================================================================
# 7. OpenAI API Server bootstrap（锁 init_app_state 三参契约）
# ==============================================================================

def test_api_server_bootstrap_contract():
    from vllm.entrypoints.openai.api_server import build_app, init_app_state

    b_sig = inspect.signature(build_app)
    assert list(b_sig.parameters.keys()) == ["args", "supported_tasks", "model_config"]

    i_sig = inspect.signature(init_app_state)
    assert list(i_sig.parameters.keys()) == ["engine_client", "state", "args"]

    assert callable(build_app)
    assert callable(init_app_state)


# ==============================================================================
# 8. TokensPrompt（输入数据结构：只锁“可构造 + 仍为类类型”）
# ==============================================================================

def test_tokens_prompt_contract():
    from vllm.inputs import TokensPrompt

    # vLLM 0.12.0 中为类（kwargs 驱动），不锁字段名（字段可演进）
    assert inspect.isclass(TokensPrompt)

    tsig = inspect.signature(TokensPrompt.__init__)
    kinds = [p.kind for p in tsig.parameters.values()]
    assert inspect.Parameter.VAR_POSITIONAL in kinds
    assert inspect.Parameter.VAR_KEYWORD in kinds


# ==============================================================================
# 9. RequestOutput（输出结构：锁关键构造参数存在性）
# ==============================================================================

def test_request_output_contract():
    from vllm.outputs import RequestOutput

    sig = inspect.signature(RequestOutput.__init__)
    params = sig.parameters

    # RL 强依赖字段：request_id / outputs
    assert "request_id" in params
    assert "outputs" in params

    # finished/metrics 常见依赖点（不锁类型细节）
    assert "finished" in params


# ==============================================================================
# 10. UsageContext（枚举：锁关键成员存在）
# ==============================================================================

def test_usage_context_contract():
    from vllm.usage.usage_lib import UsageContext

    assert inspect.isclass(UsageContext)
    assert hasattr(UsageContext, "OPENAI_API_SERVER")
    assert hasattr(UsageContext, "ENGINE_CONTEXT")


# ==============================================================================
# 11. FlexibleArgumentParser（CLI glue：锁类存在 + 继承 ArgumentParser）
# ==============================================================================

def test_flexible_argument_parser_contract():
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    import argparse

    assert inspect.isclass(FlexibleArgumentParser)
    assert issubclass(FlexibleArgumentParser, argparse.ArgumentParser)


# ==============================================================================
# 12. get_tcp_uri
# ==============================================================================

def test_get_tcp_uri_contract():
    from vllm.utils.network_utils import get_tcp_uri

    sig = inspect.signature(get_tcp_uri)
    assert list(sig.parameters.keys()) == ["ip", "port"]

    # 只验证“约定格式”，不锁异常行为
    uri = get_tcp_uri("127.0.0.1", 8000)
    assert uri == "tcp://127.0.0.1:8000"


# ==============================================================================
# 13. EngineCoreProc（核心重点：锁 __init__ + run_engine_core 签名/staticmethod）
# ==============================================================================

def test_engine_core_proc_contract():
    from vllm.v1.engine.core import EngineCoreProc

    assert inspect.isclass(EngineCoreProc)

    # __init__ 签名（参数名 + 默认值）
    sig = inspect.signature(EngineCoreProc.__init__)
    assert list(sig.parameters.keys()) == [
        "self",
        "vllm_config",
        "local_client",
        "handshake_address",
        "executor_class",
        "log_stats",
        "client_handshake_address",
        "tensor_queue",
        "engine_index",
    ]
    p = sig.parameters
    assert p["client_handshake_address"].default is None
    assert p["engine_index"].default == 0

    # run_engine_core：必须是 staticmethod，且签名固定为：
    # (*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs)
    assert hasattr(EngineCoreProc, "run_engine_core")
    raw = EngineCoreProc.__dict__["run_engine_core"]
    assert isinstance(raw, staticmethod)

    fn = EngineCoreProc.run_engine_core
    fsig = inspect.signature(fn)
    assert list(fsig.parameters.keys()) == ["args", "dp_rank", "local_dp_rank", "kwargs"]
    fp = fsig.parameters
    assert fp["dp_rank"].default == 0
    assert fp["local_dp_rank"].default == 0
    assert fp["dp_rank"].annotation is int
    assert fp["local_dp_rank"].annotation is int


# ==============================================================================
# 14. CoreEngineProcManager（进程管理：锁构造签名 + 注解）
# ==============================================================================

def test_core_engine_proc_manager_contract():
    from vllm.v1.engine.utils import CoreEngineProcManager

    assert inspect.isclass(CoreEngineProcManager)

    sig = inspect.signature(CoreEngineProcManager.__init__)
    assert list(sig.parameters.keys()) == [
        "self",
        "local_engine_count",
        "start_index",
        "local_start_index",
        "vllm_config",
        "local_client",
        "handshake_address",
        "executor_class",
        "log_stats",
        "client_handshake_address",
        "tensor_queue",
    ]

    p = sig.parameters
    assert p["client_handshake_address"].default is None
    # 关键注解（若缺失则不强制）
    if p["local_engine_count"].annotation is not inspect._empty:
        assert p["local_engine_count"].annotation is int
    if p["handshake_address"].annotation is not inspect._empty:
        assert p["handshake_address"].annotation is str


# ==============================================================================
# 15. Executor（扩展点：可继承 + 关键方法签名）
# ==============================================================================

def test_executor_contract():
    from vllm.v1.executor.abstract import Executor

    assert inspect.isclass(Executor)

    # 必须存在工厂方法 get_class(vllm_config)
    assert hasattr(Executor, "get_class")
    sig = inspect.signature(Executor.get_class)
    assert list(sig.parameters.keys()) == ["vllm_config"]

    # 扩展点：必须可被继承
    class _TestExecutor(Executor):
        def _init_executor(self):
            pass

        def collective_rpc(self, *args, **kwargs):
            pass

    # 构造签名在 v0.12.0: __init__(self, vllm_config) -> None
    init_sig = inspect.signature(Executor.__init__)
    assert list(init_sig.parameters.keys()) == ["self", "vllm_config"]


# ==============================================================================
# 16. parallel_state（全局状态模块：锁最小稳定入口）
# ==============================================================================

def test_parallel_state_contract():
    from vllm.distributed import parallel_state

    assert hasattr(parallel_state, "initialize_model_parallel")
    assert hasattr(parallel_state, "destroy_model_parallel")
    assert hasattr(parallel_state, "get_tp_group")

    # 锁 initialize_model_parallel 的关键默认值（不锁全部参数，避免过拟合）
    sig = inspect.signature(parallel_state.initialize_model_parallel)
    assert "tensor_model_parallel_size" in sig.parameters
    assert sig.parameters["tensor_model_parallel_size"].default == 1


# ==============================================================================
# 17. LoRARequest
# ==============================================================================

def test_lora_request_contract():
    from vllm.lora.request import LoRARequest

    # 1. 必须是可用的请求对象类型
    assert inspect.isclass(LoRARequest)

    sig = inspect.signature(LoRARequest)

    # 2. 明确这是一个强约束构造对象（存在必填参数）
    required_params = [
        p for p in sig.parameters.values()
        if p.default is inspect._empty and p.name != "self"
    ]
    assert len(required_params) > 0

    # 3. 必须支持被继承（用于 RL / verl patch / typing）
    class _TestLoRARequest(LoRARequest):
        pass

    assert issubclass(_TestLoRARequest, LoRARequest)