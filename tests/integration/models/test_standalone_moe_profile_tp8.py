# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone real-NPU MoE AGRS profiling test.

This is a lighter alternative to the full vLLM.LLM prefill profile. It avoids
model loading and KV cache allocation, and directly exercises the MoE path with
random hidden states and synthetic top-k routing.

Run manually on an 8-NPU host:

OMNI_NPU_STANDALONE_MOE_RUN=1 \
torchrun --nproc_per_node=8 -m pytest -s \
    tests/integration/models/test_standalone_moe_profile_tp8.py

Default workload:
  - global batch size: 8
  - logical sequence length: 100000
  - requested active tokens per MoE step: 100001, matching server_profiles.yml prefill max-num-batched-tokens
  - padded active tokens per MoE step with TP/EP=8: 100008
  - local active tokens per rank with TP/EP=8: 12501
  - hidden size: 3072
  - intermediate size: 1536
  - global experts: 256
  - local experts per rank: 32
  - top-k: 8
  - warmup iterations: 20
  - measured iterations: 100

The logical batch/sequence length is kept in the result metadata, but the
standalone MoE operator sees only one chunk worth of active tokens. This mirrors
the prefill profile in server_profiles.yml, which uses --enable-chunked-prefill
and max-num-batched-tokens 100001. The token count is padded to be divisible
by WORLD_SIZE before being split across ranks. Set
OMNI_NPU_STANDALONE_MOE_GLOBAL_TOKENS explicitly to stress a different chunk.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
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


def _require_torchrun() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so RANK/WORLD_SIZE/LOCAL_RANK are set.")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.getenv("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


class _TorchEPGroup:
    def __init__(self, rank: int, world_size: int, group: Any):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = world_size
        self.device_group = group

    def all_gather(self, tensor, dim: int = 0):
        import torch
        import torch.distributed as dist

        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self.device_group)
        return torch.cat(gathered, dim=dim)

    def reduce_scatter(self, tensor, dim: int = 0):
        import torch
        import torch.distributed as dist

        chunks = [chunk.contiguous() for chunk in torch.chunk(tensor, self.world_size, dim=dim)]
        output = torch.empty_like(chunks[self.rank])
        dist.reduce_scatter(output, chunks, group=self.device_group)
        return output


@contextlib.contextmanager
def _npu_profiler(profile_dir: Path, enabled: bool):
    if not enabled:
        yield
        return

    import torch_npu

    profile_dir.mkdir(parents=True, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(profile_dir)
        ),
    )
    profiler.start()
    try:
        yield
    finally:
        profiler.stop()


def _patch_prepare_module_for_torchrun(ep_group: _TorchEPGroup):
    import omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize as module

    module.get_ep_group = lambda: ep_group
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
    return module


def _build_layer(config: dict[str, Any], device: str):
    import torch

    local_experts = config["local_experts"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    weight_low = config["weight_low"]
    weight_high = config["weight_high"]

    layer = SimpleNamespace()
    layer.global_num_experts = config["global_experts"]
    layer.quant_config = SimpleNamespace(name="standalone_w8a8")
    layer.quant_method = SimpleNamespace(
        moe_quant_config=SimpleNamespace(use_hifloat8_w8a8=False)
    )
    layer.moe_parallel_config = SimpleNamespace(use_ep=True)
    layer.shared_experts = None

    layer.w13_weight = torch.randint(
        weight_low,
        weight_high,
        (local_experts, hidden_size, 2 * intermediate_size),
        dtype=torch.int8,
        device=device,
    )
    layer.w2_weight = torch.randint(
        weight_low,
        weight_high,
        (local_experts, intermediate_size, hidden_size),
        dtype=torch.int8,
        device=device,
    )
    layer.w13_weight_scale = torch.ones(
        local_experts,
        2 * intermediate_size,
        dtype=torch.float32,
        device=device,
    )
    layer.w2_weight_scale = torch.ones(
        local_experts,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    return layer


def _build_inputs(config: dict[str, Any], rank: int, device: str):
    import torch

    local_tokens = config["local_tokens"]
    hidden_size = config["hidden_size"]
    top_k = config["top_k"]
    global_experts = config["global_experts"]
    seed = config["seed"] + rank

    torch.manual_seed(seed)
    hidden_states = torch.randn(
        local_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    ) * config["hidden_scale"]

    topk_ids = torch.arange(
        rank * local_tokens * top_k,
        (rank + 1) * local_tokens * top_k,
        dtype=torch.int32,
        device=device,
    ).view(local_tokens, top_k) % global_experts
    topk_weights = torch.rand(
        local_tokens,
        top_k,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return hidden_states, topk_ids, topk_weights


def _run_complete_agrs_moe_iteration_from_init_routing(
    strategy,
    expert_method,
    layer,
    hidden_states,
    topk_ids,
    topk_weights,
    metadata_stream,
):
    """Run the complete AGRS MoE body starting at init routing.

    Gating/top-k selection is synthetic in this standalone test. The measured
    operator sequence starts with prepare_permute(), which calls
    npu_moe_init_routing_v2, then overlaps finalize metadata preparation with
    expert GMMs, and finally runs finalize routing plus reduce_scatter.
    """
    import torch

    cur_stream = torch.npu.current_stream()
    with torch.autograd.profiler.record_function("moe_init_routing_prepare_permute"):
        prepare_result = strategy.prepare_permute(layer, hidden_states, topk_ids)

    metadata_stream.wait_stream(cur_stream)
    with torch.npu.stream(metadata_stream):
        with torch.autograd.profiler.record_function("moe_finalize_metadata_overlap"):
            finalize_metadata = strategy.prepare_finalize_metadata(
                layer, topk_weights, prepare_result
            )

    with torch.autograd.profiler.record_function("moe_experts_grouped_matmul"):
        expert_output = expert_method.apply_experts(
            layer=layer,
            prepare_permute_result=prepare_result,
            activation="silu",
        )

    cur_stream.wait_stream(metadata_stream)
    with torch.autograd.profiler.record_function("moe_finalize_routing_reduce_scatter"):
        return strategy.unpermute_finalize(
            layer=layer,
            hidden_states=expert_output,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            agrs_prepare_permute_result=prepare_result,
            finalize_metadata=finalize_metadata,
        )


def _build_config(world_size: int) -> dict[str, Any]:
    batch_size = _env_int("OMNI_NPU_STANDALONE_MOE_BATCH_SIZE", 8)
    seq_len = _env_int("OMNI_NPU_STANDALONE_MOE_SEQ_LEN", 100_000)
    max_num_batched_tokens = _env_int(
        "OMNI_NPU_STANDALONE_MOE_MAX_NUM_BATCHED_TOKENS", 100_001
    )
    requested_global_tokens = _env_int(
        "OMNI_NPU_STANDALONE_MOE_GLOBAL_TOKENS", max_num_batched_tokens
    )
    local_tokens = _env_int(
        "OMNI_NPU_STANDALONE_MOE_LOCAL_TOKENS",
        (requested_global_tokens + world_size - 1) // world_size,
    )
    global_tokens = local_tokens * world_size

    global_experts = _env_int("OMNI_NPU_STANDALONE_MOE_GLOBAL_EXPERTS", 256)
    if global_experts % world_size != 0:
        raise ValueError(
            f"global_experts={global_experts} must be divisible by world_size={world_size}."
        )

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "requested_global_tokens": requested_global_tokens,
        "global_tokens": global_tokens,
        "local_tokens": local_tokens,
        "hidden_size": _env_int("OMNI_NPU_STANDALONE_MOE_HIDDEN_SIZE", 3072),
        "intermediate_size": _env_int(
            "OMNI_NPU_STANDALONE_MOE_INTERMEDIATE_SIZE", 1536
        ),
        "global_experts": global_experts,
        "local_experts": _env_int(
            "OMNI_NPU_STANDALONE_MOE_LOCAL_EXPERTS",
            global_experts // world_size,
        ),
        "top_k": _env_int("OMNI_NPU_STANDALONE_MOE_TOP_K", 8),
        "warmup_iters": _env_int("OMNI_NPU_STANDALONE_MOE_WARMUP_ITERS", 20),
        "measure_iters": _env_int("OMNI_NPU_STANDALONE_MOE_MEASURE_ITERS", 100),
        "seed": _env_int("OMNI_NPU_STANDALONE_MOE_SEED", 20260417),
        "hidden_scale": float(os.getenv("OMNI_NPU_STANDALONE_MOE_HIDDEN_SCALE", "0.01")),
        "weight_low": _env_int("OMNI_NPU_STANDALONE_MOE_WEIGHT_LOW", -2),
        "weight_high": _env_int("OMNI_NPU_STANDALONE_MOE_WEIGHT_HIGH", 2),
    }


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.multi_device
def test_standalone_agrs_moe_profile_tp8() -> None:
    if not _env_bool("OMNI_NPU_STANDALONE_MOE_RUN", False):
        pytest.skip("Set OMNI_NPU_STANDALONE_MOE_RUN=1 to run this test.")

    rank, world_size, local_rank = _require_torchrun()

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    expected_world_size = _env_int("OMNI_NPU_STANDALONE_MOE_WORLD_SIZE", 8)
    if world_size != expected_world_size:
        pytest.skip(
            f"Expected WORLD_SIZE={expected_world_size}, got {world_size}."
        )
    if torch.npu.device_count() <= local_rank:
        pytest.skip(f"NPU local_rank={local_rank} is not available.")

    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=os.getenv("OMNI_NPU_STANDALONE_MOE_DIST_BACKEND", "hccl")
        )

    device = f"npu:{local_rank}"
    config = _build_config(world_size)
    ep_group = _TorchEPGroup(rank, world_size, dist.group.WORLD)
    prepare_module = _patch_prepare_module_for_torchrun(ep_group)

    from omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        NPUCompressedTensorsW8A8Int8MoEMethod,
    )

    layer = _build_layer(config, device)
    strategy = prepare_module.AGRSPrepPmtAndUnpmtFinal(layer)
    expert_method = NPUCompressedTensorsW8A8Int8MoEMethod.__new__(
        NPUCompressedTensorsW8A8Int8MoEMethod
    )
    expert_method.moe = SimpleNamespace(has_bias=False)

    hidden_states, topk_ids, topk_weights = _build_inputs(config, rank, device)
    metadata_stream = torch.npu.Stream()

    result_dir = Path(
        os.getenv("OMNI_NPU_STANDALONE_MOE_RESULT_DIR", "standalone_moe_profile_tp8")
    ).resolve()
    profile_root = Path(
        os.getenv(
            "OMNI_NPU_STANDALONE_MOE_PROFILER_DIR",
            "standalone_moe_profile_tp8/profiler",
        )
    ).resolve()
    profile_dir = profile_root / f"rank_{rank}"

    for _ in range(config["warmup_iters"]):
        routed_output = _run_complete_agrs_moe_iteration_from_init_routing(
            strategy,
            expert_method,
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            metadata_stream,
        )
        assert routed_output.shape == hidden_states.shape

    torch.npu.synchronize()
    dist.barrier()

    timings_s: list[float] = []
    with _npu_profiler(
        profile_dir,
        _env_bool("OMNI_NPU_STANDALONE_MOE_PROFILE", True),
    ):
        for _ in range(config["measure_iters"]):
            dist.barrier()
            start_s = time.perf_counter()
            routed_output = _run_complete_agrs_moe_iteration_from_init_routing(
                strategy,
                expert_method,
                layer,
                hidden_states,
                topk_ids,
                topk_weights,
                metadata_stream,
            )
            torch.npu.synchronize()
            dist.barrier()
            timings_s.append(time.perf_counter() - start_s)
            assert routed_output.shape == hidden_states.shape

    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "config": config,
        "timings_s": timings_s,
        "summary": _summarize_timings(timings_s),
        "profiler_dir": str(profile_dir),
    }
    result_path = result_dir / f"standalone_moe_profile_rank_{rank}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if rank == 0:
        print(f"standalone MoE profile result dir: {result_dir}")
        print(f"standalone MoE profiler dir: {profile_root}")
        print(json.dumps(result["summary"], indent=2, sort_keys=True))

    assert len(timings_s) == config["measure_iters"]
