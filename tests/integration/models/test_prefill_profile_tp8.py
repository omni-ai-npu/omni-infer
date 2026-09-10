# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Heavyweight real-NPU prefill profiling test.

This test is skipped by default. Run it manually on the NPU profiling host:

OMNI_NPU_PREFILL_PROFILE_RUN=1 \
OMNI_NPU_PREFILL_MODEL=/path/to/model \
pytest -s tests/integration/models/test_prefill_profile_tp8.py

Default workload:
  - tensor parallel size: 8
  - batch size: 8
  - prompt length: 100000 tokens
  - warmup iterations: 20
  - measured iterations: 100
  - output length: 1 token
  - expert parallel: enabled
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pytest


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _load_vocab_size(model_path: str) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        text_config = getattr(config, "text_config", None)
        vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError(
            "Could not infer vocab_size from model config. Set "
            "OMNI_NPU_PREFILL_VOCAB_SIZE explicitly.")
    return int(vocab_size)


def _build_random_prompts(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> list[dict[str, list[int]]]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    return [{"prompt_token_ids": row.tolist()} for row in token_ids]


def _make_sampling_params(output_len: int) -> Any:
    from vllm import SamplingParams

    try:
        return SamplingParams(
            max_tokens=output_len,
            temperature=0.0,
            ignore_eos=True,
        )
    except TypeError:
        return SamplingParams(max_tokens=output_len, temperature=0.0)


@contextlib.contextmanager
def _vllm_model_argv(model_path: str):
    original_argv = sys.argv[:]
    sys.argv = ["vllm", "serve", model_path]
    try:
        yield
    finally:
        sys.argv = original_argv


def _create_llm(config: dict[str, Any]) -> Any:
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": config["model"],
        "skip_tokenizer_init": config["skip_tokenizer_init"],
        "tensor_parallel_size": config["tp_size"],
        "dtype": config["dtype"],
        "trust_remote_code": config["trust_remote_code"],
        "max_model_len": config["max_model_len"],
        "max_num_batched_tokens": config["max_num_batched_tokens"],
        "max_num_seqs": config["batch_size"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "distributed_executor_backend": config["distributed_executor_backend"],
        "enable_expert_parallel": config["enable_expert_parallel"],
    }
    if config["enforce_eager"]:
        llm_kwargs["enforce_eager"] = True
    llm_kwargs.update(config["extra_engine_kwargs"])
    with _vllm_model_argv(config["model"]):
        return LLM(**llm_kwargs)


def _call_profile_hook(llm: Any, hook_name: str) -> None:
    candidates = [llm, getattr(llm, "llm_engine", None)]
    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is not None:
        candidates.append(getattr(llm_engine, "model_executor", None))

    for candidate in candidates:
        if candidate is None:
            continue
        hook = getattr(candidate, hook_name, None)
        if hook is not None:
            hook()
            return

    raise RuntimeError(
        f"vLLM object does not expose {hook_name}(). "
        "Profiling requires a vLLM build with profiler hooks enabled.")


def _synchronize_npu() -> None:
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
    except Exception:
        pass


def _generate(llm: Any, prompts: list[dict[str, list[int]]],
              sampling_params: Any) -> Any:
    try:
        return llm.generate(prompts, sampling_params, use_tqdm=False)
    except TypeError:
        return llm.generate(prompts, sampling_params)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _summarize_timings(timings_s: list[float]) -> dict[str, float]:
    return {
        "min_s": min(timings_s),
        "max_s": max(timings_s),
        "mean_s": statistics.fmean(timings_s),
        "median_s": statistics.median(timings_s),
        "p90_s": _percentile(timings_s, 0.90),
        "p99_s": _percentile(timings_s, 0.99),
    }


def _write_result(result_dir: Path, payload: dict[str, Any]) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "prefill_profile_tp8_result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                           encoding="utf-8")
    return result_path


def _build_config() -> dict[str, Any]:
    model = os.getenv("OMNI_NPU_PREFILL_MODEL")
    if not model:
        pytest.skip("Set OMNI_NPU_PREFILL_MODEL to run this test.")

    batch_size = _env_int("OMNI_NPU_PREFILL_BATCH_SIZE", 8)
    seq_len = _env_int("OMNI_NPU_PREFILL_SEQ_LEN", 100_000)
    output_len = _env_int("OMNI_NPU_PREFILL_OUTPUT_LEN", 1)
    max_model_len = _env_int("OMNI_NPU_PREFILL_MAX_MODEL_LEN",
                             seq_len + output_len)
    max_num_batched_tokens = _env_int(
        "OMNI_NPU_PREFILL_MAX_NUM_BATCHED_TOKENS",
        batch_size * seq_len,
    )
    extra_engine_kwargs = json.loads(
        os.getenv("OMNI_NPU_PREFILL_ENGINE_KWARGS_JSON", "{}"))

    profile_dir = Path(
        os.getenv("OMNI_NPU_PREFILL_PROFILER_DIR",
                  "prefill_profile_tp8/profiler")).resolve()
    result_dir = Path(
        os.getenv("OMNI_NPU_PREFILL_RESULT_DIR",
                  "prefill_profile_tp8")).resolve()

    os.environ.setdefault("VLLM_TORCH_PROFILER_DIR", str(profile_dir))
    os.environ.setdefault("VLLM_TORCH_PROFILER_RECORD_SHAPES", "True")
    os.environ.setdefault("VLLM_TORCH_PROFILER_WITH_STACK", "False")
    os.environ.setdefault("VLLM_TORCH_PROFILER_WITH_FLOPS", "False")

    return {
        "model": model,
        "tp_size": _env_int("OMNI_NPU_PREFILL_TP_SIZE", 8),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "output_len": output_len,
        "warmup_iters": _env_int("OMNI_NPU_PREFILL_WARMUP_ITERS", 20),
        "measure_iters": _env_int("OMNI_NPU_PREFILL_MEASURE_ITERS", 100),
        "seed": _env_int("OMNI_NPU_PREFILL_SEED", 20260417),
        "vocab_size": _env_int("OMNI_NPU_PREFILL_VOCAB_SIZE", 0),
        "dtype": os.getenv("OMNI_NPU_PREFILL_DTYPE", "bfloat16"),
        "skip_tokenizer_init": _env_bool(
            "OMNI_NPU_PREFILL_SKIP_TOKENIZER_INIT", True),
        "enable_expert_parallel": _env_bool(
            "OMNI_NPU_PREFILL_ENABLE_EXPERT_PARALLEL", True),
        "trust_remote_code": _env_bool("OMNI_NPU_PREFILL_TRUST_REMOTE_CODE",
                                       True),
        "distributed_executor_backend": os.getenv(
            "OMNI_NPU_PREFILL_DIST_BACKEND", "mp"),
        "gpu_memory_utilization": _env_float(
            "OMNI_NPU_PREFILL_GPU_MEMORY_UTILIZATION", 0.90),
        "enforce_eager": _env_bool("OMNI_NPU_PREFILL_ENFORCE_EAGER", False),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "extra_engine_kwargs": extra_engine_kwargs,
        "profile_dir": str(profile_dir),
        "result_dir": str(result_dir),
    }


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.multi_device
def test_prefill_random_100k_tp8_profile() -> None:
    if not _env_bool("OMNI_NPU_PREFILL_PROFILE_RUN", False):
        pytest.skip("Set OMNI_NPU_PREFILL_PROFILE_RUN=1 to run this test.")
    if not _npu_available():
        pytest.skip("NPU hardware and torch_npu are required.")

    config = _build_config()
    if config["vocab_size"] <= 0:
        config["vocab_size"] = _load_vocab_size(config["model"])

    prompts = _build_random_prompts(
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        seed=config["seed"],
    )
    sampling_params = _make_sampling_params(config["output_len"])
    llm = _create_llm(config)

    for _ in range(config["warmup_iters"]):
        outputs = _generate(llm, prompts, sampling_params)
        assert len(outputs) == config["batch_size"]

    _synchronize_npu()
    _call_profile_hook(llm, "start_profile")
    timings_s: list[float] = []
    try:
        for _ in range(config["measure_iters"]):
            _synchronize_npu()
            start_s = time.perf_counter()
            outputs = _generate(llm, prompts, sampling_params)
            _synchronize_npu()
            elapsed_s = time.perf_counter() - start_s
            assert len(outputs) == config["batch_size"]
            timings_s.append(elapsed_s)
    finally:
        _call_profile_hook(llm, "stop_profile")

    result = {
        "config": {key: value for key, value in config.items()
                   if key != "extra_engine_kwargs"},
        "extra_engine_kwargs": config["extra_engine_kwargs"],
        "timings_s": timings_s,
        "summary": _summarize_timings(timings_s),
    }
    result_path = _write_result(Path(config["result_dir"]), result)
    print(f"prefill profile result: {result_path}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))

    assert len(timings_s) == config["measure_iters"]