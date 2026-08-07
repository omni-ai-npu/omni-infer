# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import math
import os
import sys
import types
import fcntl
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from .distributed_test_common import distributed_worker_pool

import torch_npu

pytestmark = pytest.mark.skipif(
    torch_npu is None
    or not hasattr(torch, "npu")
    or not torch.npu.is_available()
    or torch.npu.device_count() < 2,
    reason="requires at least 2 NPUs with torch_npu",
)

TEST_SEED = 0
NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 1024
DTYPE = torch.bfloat16
FUSED_MOE_GOLDEN_ATOL = 2e-3
FUSED_MOE_GOLDEN_RTOL = 2e-3
'''
Golden update usage:
1) Update all test cases:
   UPDATE_FUSED_MOE_GOLDEN=1 pytest tests/unit/layers/st/test_fused_moe.py -v
2) Update one specific case:
   UPDATE_FUSED_MOE_GOLDEN=1 pytest tests/unit/layers/st/test_fused_moe.py -k "test_w4a8_all2all" -v
3) Update with a custom golden directory:
   FUSED_MOE_GOLDEN_DIR=/your/golden/path UPDATE_FUSED_MOE_GOLDEN=1 pytest tests/unit/layers/st/test_fused_moe.py -v
'''
FUSED_MOE_GOLDEN_DIR = Path(os.getenv("FUSED_MOE_GOLDEN_DIR","/data/models/ut_pt",))
UPDATE_FUSED_MOE_GOLDEN = os.getenv("UPDATE_FUSED_MOE_GOLDEN", "0") == "1"
MULTI_DP_REQUIRED_NPUS = 2
TWO_CARD_EP_RUNTIME_CONFIG = {
    "world_size": 2,
    "tp_size": 2,
    "dp_size": 1,
    "enable_expert_parallel": True,
}
# Multi-DP paths: tp=1, dp=2 (world_size=2).
TWO_CARD_TP1_DP2_RUNTIME_CONFIG = {
    "world_size": 2,
    "tp_size": 1,
    "dp_size": 2,
    "enable_expert_parallel": True,
}


@dataclass(frozen=True)
class DummyAttnMetadata:
    decode_threshold: int
    decode: object | None = None
    num_prefills: int = 1


class DummyForwardContext:
    def __init__(self, attn_metadata):
        self.attn_metadata = attn_metadata
        self.virtual_engine = 0
        self.batch_descriptor = None
        self.no_compile_layers = {}
        self.capturing = False


@dataclass(frozen=True)
class MoERegressionCase:
    num_tokens: int
    expected_strategy: str
    quant_mode: str = "bf16"
    shared_tp_size: int | None = None
    enable_omni_custom_models: bool = False
    runtime_config: dict[str, int | bool] | None = None


class TrackingSharedExperts(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        device: torch.device,
        *,
        tp_size: int,
        tp_world_size: int,
    ) -> None:
        super().__init__()
        torch.manual_seed(TEST_SEED + 11)
        weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32) / math.sqrt(
            hidden_size
        )
        self.weight = torch.nn.Parameter(
            weight.to(device=device, dtype=DTYPE), requires_grad=False
        )
        # Prefetch path expects shared_experts.{gate_up_proj,down_proj}.weight.
        # Keep a lightweight linear-like carrier so ST mock stays compatible with
        # real layer contracts without changing reference math.
        self.gate_up_proj = SimpleNamespace(tp_size=tp_size, weight=self.weight)
        self.down_proj = SimpleNamespace(weight=self.weight)
        self._runtime_scale = 1.0 / tp_world_size if tp_size > 1 else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x.float(), self.weight.float())
        return (output * self._runtime_scale).to(dtype=x.dtype)


def _build_hidden_states(
    num_tokens: int,
    hidden_size: int,
    device: torch.device,
    batch_group_id: int,
) -> torch.Tensor:
    torch.manual_seed(TEST_SEED + batch_group_id)
    base = torch.randn(hidden_size, num_tokens * 2, dtype=torch.float32)
    hidden_states = base[:, ::2].t().to(device=device, dtype=DTYPE)
    assert hidden_states.shape == (num_tokens, hidden_size)
    return hidden_states


def _build_router_logits(
    num_tokens: int,
    num_experts: int,
    device: torch.device,
    batch_group_id: int,
) -> torch.Tensor:
    logits = torch.full((num_tokens, num_experts), -9.0, dtype=torch.float32)
    for token_idx in range(num_tokens):
        first = (token_idx + batch_group_id) % num_experts
        second = (token_idx + batch_group_id + 2) % num_experts
        logits[token_idx, first] = 8.0 + token_idx * 0.01
        logits[token_idx, second] = 4.0 + token_idx * 0.01
    return logits.to(device=device)


def _build_full_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(TEST_SEED + 1)
    w13 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.float32,
    ) / math.sqrt(hidden_size)
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.float32,
    ) / math.sqrt(intermediate_size)
    return w13.to(device=device, dtype=DTYPE), w2.to(device=device, dtype=DTYPE)


def _quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0
    quantized = torch.round(weight.float() / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale.to(torch.float32)


def _quantize_per_channel_int4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 7.0
    quantized = torch.round(weight.float() / scale).clamp(-8, 7).to(torch.int8)
    return quantized, scale.to(torch.float32)


def _pack_int4_pairs_along_dim0(q_int4: torch.Tensor) -> torch.Tensor:
    if q_int4.shape[0] % 2 != 0:
        raise ValueError(f"int4 pack expects even dim0, got shape={tuple(q_int4.shape)}")
    low = q_int4[0::2].to(torch.int16)
    high = q_int4[1::2].to(torch.int16)
    packed_u8 = ((low & 0x0F) | ((high & 0x0F) << 4)).to(torch.uint8)
    return packed_u8.view(torch.int8).contiguous()


def _assert_moe_output_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    input_summary: str,
    expected_strategy: str,
    actual_strategy: str,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"dtype mismatch: actual={actual.dtype} expected={expected.dtype}"
        )
    if torch.allclose(actual, expected, atol=atol, rtol=rtol):
        return

    diff = (actual.float() - expected.float()).abs()
    flat_idx = diff.view(-1).argmax().item()
    max_abs_diff = diff.view(-1)[flat_idx].item()
    mean_abs_diff = diff.mean().item()
    max_idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx), diff.shape))
    raise AssertionError(
        "moe output mismatch: "
        f"max_abs_diff={max_abs_diff} "
        f"mean_abs_diff={mean_abs_diff} "
        f"max_idx={max_idx} "
        f"shape={tuple(actual.shape)} "
        f"dtype={actual.dtype} "
        f"input_summary={input_summary} "
        f"expected_strategy={expected_strategy} "
        f"actual_strategy={actual_strategy}"
    )


def _load_local_expert_weights(
    layer,
    full_w13: torch.Tensor,
    full_w2: torch.Tensor,
) -> None:
    with torch.no_grad():
        for global_expert_idx, local_expert_idx in enumerate(layer.expert_map.tolist()):
            if local_expert_idx < 0:
                continue
            if layer.quant_config is None:
                layer.w13_weight.data[local_expert_idx].copy_(full_w13[global_expert_idx])
                layer.w2_weight.data[local_expert_idx].copy_(full_w2[global_expert_idx])
                continue

            if hasattr(layer, "w13_weight_int4_scale"):
                q_w13_i4, w13_scale = _quantize_per_channel_int4(full_w13[global_expert_idx])
                q_w2_i4, w2_scale = _quantize_per_channel_int4(full_w2[global_expert_idx])
                layer.w13_weight.data[local_expert_idx].copy_(
                    _pack_int4_pairs_along_dim0(q_w13_i4)
                )
                layer.w2_weight.data[local_expert_idx].copy_(
                    _pack_int4_pairs_along_dim0(q_w2_i4)
                )
                encoded_w13_scale = torch_npu.npu_trans_quant_param(
                    w13_scale.squeeze(-1).to(dtype=torch.float32, device=layer.w13_weight.device)
                )
                encoded_w2_scale = torch_npu.npu_trans_quant_param(
                    w2_scale.squeeze(-1).to(dtype=torch.float32, device=layer.w2_weight.device)
                )
                layer.w13_weight_int4_scale.data[local_expert_idx].copy_(
                    encoded_w13_scale.to(layer.w13_weight_int4_scale.dtype).view(
                        layer.w13_weight_int4_scale.data[local_expert_idx].shape
                    )
                )
                layer.w2_weight_int4_scale.data[local_expert_idx].copy_(
                    encoded_w2_scale.to(layer.w2_weight_int4_scale.dtype).view(
                        layer.w2_weight_int4_scale.data[local_expert_idx].shape
                    )
                )
                layer.w13_weight_bias.data[local_expert_idx].zero_()
                layer.w2_weight_bias.data[local_expert_idx].zero_()
                continue

            q_w13, w13_scale = _quantize_per_channel(full_w13[global_expert_idx])
            q_w2, w2_scale = _quantize_per_channel(full_w2[global_expert_idx])
            layer.w13_weight.data[local_expert_idx].copy_(q_w13)
            layer.w2_weight.data[local_expert_idx].copy_(q_w2)
            layer.w13_weight_scale.data[local_expert_idx].copy_(w13_scale)
            layer.w2_weight_scale.data[local_expert_idx].copy_(w2_scale)
        layer.quant_method.process_weights_after_loading(layer)


def _make_forward_context_for_strategy(expected_strategy: str) -> DummyForwardContext:
    decode_threshold = 0 if expected_strategy == "all2all" else 1024
    return DummyForwardContext(
        {0: DummyAttnMetadata(decode_threshold=decode_threshold)}
    )


def _configure_strategy_selector_for_case(case: MoERegressionCase) -> None:
    from omni_npu.model_config.config_loader.loader import model_extra_config

    model_extra_config.operator_opt_config.decode_moe_dispatch_combine = (
        case.expected_strategy in ("dispatch_combine", "all2all")
    )
    model_extra_config.operator_opt_config.enable_moe_agrs = (
        case.expected_strategy == "agrs"
    )
    model_extra_config.parall_config.ena_seq_parallel = False


def _runtime_config_for_case(case: MoERegressionCase) -> dict[str, int | bool]:
    if case.runtime_config is not None:
        return dict(case.runtime_config)
    return dict(TWO_CARD_EP_RUNTIME_CONFIG)


def _fused_moe_case_golden_path(
    case: MoERegressionCase,
    *,
    runtime_config: dict[str, int | bool],
    quant_mode: str,
) -> Path:
    plug_tag = "high" if case.enable_omni_custom_models else "base"
    return FUSED_MOE_GOLDEN_DIR / (
        f"{case.expected_strategy}"
        f"_tp{runtime_config['tp_size']}_dp{runtime_config['dp_size']}"
        f"_{quant_mode}"
        f"_stp{case.shared_tp_size if case.shared_tp_size is not None else 'none'}"
        f"_{plug_tag}"
        ".pt"
    )


def _fused_moe_golden_key(*, local_rank: int, batch_group_id: int) -> str:
    return f"rank{local_rank}_bg{batch_group_id}"


def _skip_if_missing_fused_moe_golden(
    case: MoERegressionCase,
    *,
    runtime_config: dict[str, int | bool],
) -> None:
    if UPDATE_FUSED_MOE_GOLDEN:
        return
    golden_path = _fused_moe_case_golden_path(
        case,
        runtime_config=runtime_config,
        quant_mode=case.quant_mode,
    )
    if not golden_path.exists():
        pytest.skip(
            f"missing fused_moe golden file: {golden_path}. "
            "Please generate with UPDATE_FUSED_MOE_GOLDEN=1."
        )


def _load_or_update_fused_moe_golden(
    actual: torch.Tensor,
    case: MoERegressionCase,
    *,
    local_rank: int,
    runtime_config: dict[str, int | bool],
    batch_group_id: int,
    quant_mode: str,
) -> torch.Tensor:
    golden_path = _fused_moe_case_golden_path(
        case,
        runtime_config=runtime_config,
        quant_mode=quant_mode,
    )
    golden_key = _fused_moe_golden_key(
        local_rank=local_rank,
        batch_group_id=batch_group_id,
    )
    should_update = UPDATE_FUSED_MOE_GOLDEN
    if should_update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = golden_path.with_suffix(".lock")
        try:
            with open(lock_path, "w", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                payload: dict[str, torch.Tensor] = {}
                if golden_path.exists():
                    existing = torch.load(golden_path, map_location="cpu")
                    if isinstance(existing, dict):
                        payload = dict(existing)
                payload[golden_key] = actual.detach().cpu()
                tmp_path = golden_path.with_suffix(".tmp")
                torch.save(payload, tmp_path)
                os.replace(tmp_path, golden_path)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass
        return actual
    if not golden_path.exists():
        raise RuntimeError(
            f"missing fused_moe golden file: {golden_path}. "
            "Please generate with UPDATE_FUSED_MOE_GOLDEN=1."
        )
    golden_payload = torch.load(golden_path, map_location="cpu")
    if not isinstance(golden_payload, dict):
        raise AssertionError(
            f"invalid fused_moe golden payload in {golden_path}: expected dict with rank keys."
        )
    if golden_key not in golden_payload:
        raise AssertionError(
            f"missing key {golden_key} in {golden_path}. "
            f"available keys={sorted(golden_payload.keys())}"
        )
    golden = golden_payload[golden_key]
    return golden.to(device=actual.device, dtype=actual.dtype)


def _build_w8a8_quant_config():
    from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import (
        NPUCompressedTensorsConfig,
    )

    quant_config = NPUCompressedTensorsConfig.__new__(NPUCompressedTensorsConfig)
    quant_config.quant_format = "activation"
    quant_config.target_scheme_map = {
        "Linear": {
            "weights": SimpleNamespace(
                num_bits=8,
                strategy="channel",
                dynamic=False,
                symmetric=True,
            ),
            "input_activations": SimpleNamespace(
                num_bits=8,
                strategy="token",
                dynamic=True,
                symmetric=True,
            ),
        }
    }
    quant_config.ignore = []
    quant_config.packed_modules_mapping = {}
    quant_config.get_name = lambda: "npu-compressed-tensors"
    return quant_config


def _build_w4a8_quant_config():
    from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import (
        NPUCompressedTensorsConfig,
    )

    quant_config = NPUCompressedTensorsConfig.__new__(NPUCompressedTensorsConfig)
    quant_config.quant_format = "activation"
    quant_config.target_scheme_map = {
        "Linear": {
            "weights": SimpleNamespace(
                num_bits={"mlp.experts": 4},
                strategy="channel",
                dynamic=False,
                symmetric=True,
            ),
            "input_activations": SimpleNamespace(
                num_bits=8,
                strategy="token",
                dynamic=True,
                symmetric=True,
            ),
        }
    }
    quant_config.ignore = []
    quant_config.packed_modules_mapping = {}
    quant_config.get_name = lambda: "npu-compressed-tensors"
    return quant_config


def _install_omni_layer_packages() -> None:
    base = Path(__file__).resolve().parents[4] / "omni"

    layers_pkg = types.ModuleType("omni_npu.layers")
    layers_pkg.__path__ = [str(base / "layers")]
    sys.modules["omni_npu.layers"] = layers_pkg

    fused_moe_pkg = types.ModuleType("omni_npu.layers.fused_moe")
    fused_moe_pkg.__path__ = [str(base / "layers" / "fused_moe")]
    sys.modules["omni_npu.layers.fused_moe"] = fused_moe_pkg

    quant_pkg = types.ModuleType("omni_npu.layers.quantization")
    quant_pkg.__path__ = [str(base / "layers" / "quantization")]
    sys.modules["omni_npu.layers.quantization"] = quant_pkg

    ct_pkg = types.ModuleType("omni_npu.layers.quantization.compressed_tensors")
    ct_pkg.__path__ = [str(base / "layers" / "quantization" / "compressed_tensors")]
    sys.modules["omni_npu.layers.quantization.compressed_tensors"] = ct_pkg


def _run_moe_output_regression(
    device: int,
    local_rank: int,
    world_size: int,
    case: MoERegressionCase,
) -> None:
    _install_omni_layer_packages()

    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_dp_group
    from omni_npu.layers.fused_moe.layer import NPUFusedMoE, NPUSharedFusedMoE
    import omni_npu.layers.fused_moe.layer as moe_layer_module
    import omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize as prepare_module

    torch.manual_seed(TEST_SEED)
    torch.npu.set_device(device)
    npu_device = torch.device(f"npu:{device}")
    runtime_config = _runtime_config_for_case(case)

    vllm_config = get_current_vllm_config()
    vllm_config.parallel_config.enable_expert_parallel = True
    if getattr(vllm_config, "model_config", None) is None:
        vllm_config.model_config = SimpleNamespace()
    if getattr(vllm_config.model_config, "hf_config", None) is None:
        vllm_config.model_config.hf_config = SimpleNamespace()
    if getattr(vllm_config.model_config, "dtype", None) is None:
        vllm_config.model_config.dtype = DTYPE
    if getattr(vllm_config.model_config, "enable_return_routed_experts", None) is None:
        vllm_config.model_config.enable_return_routed_experts = False
    dp_group = get_dp_group()
    batch_group_id = dp_group.rank_in_group if dp_group.world_size > 1 else 0

    hidden_states = _build_hidden_states(
        case.num_tokens, HIDDEN_SIZE, npu_device, batch_group_id
    )
    router_logits = _build_router_logits(
        case.num_tokens, NUM_EXPERTS, npu_device, batch_group_id
    )
    full_w13, full_w2 = _build_full_weights(
        NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, npu_device
    )
    quant_mode = case.quant_mode

    shared_experts = None
    if case.shared_tp_size is not None:
        shared_experts = TrackingSharedExperts(
            HIDDEN_SIZE,
            npu_device,
            tp_size=case.shared_tp_size,
            tp_world_size=runtime_config["tp_size"],
        ).to(device=npu_device)

    if quant_mode == "w8a8":
        quant_config = _build_w8a8_quant_config()
    elif quant_mode == "w4a8":
        quant_config = _build_w4a8_quant_config()
    else:
        quant_config = None
    layer_cls = NPUSharedFusedMoE if shared_experts is not None else NPUFusedMoE
    layer_kwargs = {
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "params_dtype": DTYPE,
        "reduce_results": False,
        "renormalize": True,
        "quant_config": quant_config,
        "prefix": f"model.layers.0.mlp.experts.{case.expected_strategy}.{local_rank}",
    }
    if shared_experts is not None:
        layer_kwargs["shared_experts"] = shared_experts
    layer = layer_cls(**layer_kwargs).to(device=npu_device)
    _load_local_expert_weights(layer, full_w13, full_w2)

    forward_ctx = _make_forward_context_for_strategy(case.expected_strategy)
    forward_ctx.no_compile_layers[layer.layer_name] = layer
    _configure_strategy_selector_for_case(case)
    env_patch = (
        patch.dict(os.environ, {"VLLM_PLUGINS": "omni_custom_models"})
        if case.enable_omni_custom_models
        else patch.dict(os.environ, {}, clear=False)
    )
    with env_patch, patch.object(
        moe_layer_module, "get_forward_context", return_value=forward_ctx
    ), patch.object(
        prepare_module, "get_forward_context", return_value=forward_ctx
    ):
        actual_strategy, _ = layer.quant_method.select_communication_strategy(case.num_tokens)
        assert actual_strategy == case.expected_strategy

        topk_weights, topk_ids = NPUFusedMoE.select_experts(
            router_logits=router_logits,
            top_k=TOP_K,
            use_grouped_topk=False,
            renormalize=True,
            routed_scaling_factor=1.0,
        )
        row_sums = topk_weights.sum(dim=-1)
        assert torch.allclose(
            row_sums.float(),
            torch.ones_like(row_sums.float()),
            atol=1e-4,
            rtol=1e-4,
        )

        actual = layer(hidden_states, router_logits)
        actual_output = actual[-1] if isinstance(actual, tuple) else actual

    final_expected = _load_or_update_fused_moe_golden(
        actual_output,
        case,
        local_rank=local_rank,
        runtime_config=runtime_config,
        batch_group_id=batch_group_id,
        quant_mode=quant_mode,
    )
    atol, rtol = FUSED_MOE_GOLDEN_ATOL, FUSED_MOE_GOLDEN_RTOL
    _assert_moe_output_close(
        actual=actual_output,
        expected=final_expected,
        atol=atol,
        rtol=rtol,
        input_summary=(
            f"num_tokens={case.num_tokens},hidden_size={HIDDEN_SIZE},"
            f"top_k={TOP_K},world_size={world_size},rank={local_rank},"
            f"batch_group_id={batch_group_id},"
            f"quant_mode={quant_mode},"
            f"tp_size={runtime_config['tp_size']},dp_size={runtime_config['dp_size']},"
            f"shared_tp_size={case.shared_tp_size}"
        ),
        expected_strategy=case.expected_strategy,
        actual_strategy=actual_strategy,
    )


def _run_case(distributed_worker_pool, case: MoERegressionCase) -> None:
    runtime_config = _runtime_config_for_case(case)
    _skip_if_missing_fused_moe_golden(case, runtime_config=runtime_config)
    distributed_worker_pool(
        _run_moe_output_regression,
        case,
        config={},
        runtime_config=runtime_config,
    )


def test_bf16_dispatch_combine(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33, 
            expected_strategy="dispatch_combine",
            quant_mode="bf16",
        ),
    )


def test_w8a8_dispatch_combine(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33,
            expected_strategy="dispatch_combine",
            quant_mode="w8a8",
        ),
    )


def test_w4a8_dispatch_combine(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33,
            expected_strategy="dispatch_combine",
            quant_mode="w4a8",
        ),
    )


def test_bf16_all2all(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=131, 
            expected_strategy="all2all",
            quant_mode="bf16",
        ),
    )


def test_w8a8_all2all(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=131,
            expected_strategy="all2all",
            quant_mode="w8a8",
        ),
    )


def test_w4a8_all2all(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=131,
            expected_strategy="all2all",
            quant_mode="w4a8",
        ),
    )


@pytest.mark.skipif(
    torch_npu is None
    or not hasattr(torch, "npu")
    or not torch.npu.is_available()
    or torch.npu.device_count() < MULTI_DP_REQUIRED_NPUS,
    reason="requires 2 NPUs for tp=1 dp=2 all2all",
)
def test_bf16_all2all_tp1_dp2(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=131,
            expected_strategy="all2all",
            quant_mode="bf16",
            runtime_config=TWO_CARD_TP1_DP2_RUNTIME_CONFIG,
        ),
    )


@pytest.mark.skipif(
    torch_npu is None
    or not hasattr(torch, "npu")
    or not torch.npu.is_available()
    or torch.npu.device_count() < MULTI_DP_REQUIRED_NPUS,
    reason="requires 2 NPUs for tp=1 dp=2 all2all",
)
def test_w8a8_all2all_tp1_dp2(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=131,
            expected_strategy="all2all",
            quant_mode="w8a8",
            runtime_config=TWO_CARD_TP1_DP2_RUNTIME_CONFIG,
        ),
    )


@pytest.mark.skipif(
    torch_npu is None
    or not hasattr(torch, "npu")
    or not torch.npu.is_available()
    or torch.npu.device_count() < MULTI_DP_REQUIRED_NPUS,
    reason="requires 2 NPUs for tp=1 dp=2 agrs",
)
def test_bf16_agrs_tp1_dp2(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33,
            expected_strategy="dispatch_combine",
            quant_mode="bf16",
            runtime_config=TWO_CARD_TP1_DP2_RUNTIME_CONFIG,
        ),
    )


@pytest.mark.skipif(
    torch_npu is None
    or not hasattr(torch, "npu")
    or not torch.npu.is_available()
    or torch.npu.device_count() < MULTI_DP_REQUIRED_NPUS,
    reason="requires 2 NPUs for tp=1 dp=2 agrs",
)
def test_w8a8_agrs_tp1_dp2(distributed_worker_pool) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33,
            expected_strategy="dispatch_combine",
            quant_mode="w8a8",
            runtime_config=TWO_CARD_TP1_DP2_RUNTIME_CONFIG,
        ),
    )


def test_bf16_shared_experts_tp_eq_1_highlayer(
    distributed_worker_pool,
) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33,
            expected_strategy="dispatch_combine",
            quant_mode="bf16",
            shared_tp_size=1,
            enable_omni_custom_models=True,
        ),
    )


def test_w8a8_shared_experts_tp_eq_1_highlayer(
    distributed_worker_pool,
) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33,
            expected_strategy="dispatch_combine",
            quant_mode="w8a8",
            shared_tp_size=1,
            enable_omni_custom_models=True,
        ),
    )


def test_bf16_shared_experts_tp_gt_1_baselayer(
    distributed_worker_pool,
) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33,
            expected_strategy="dispatch_combine",
            quant_mode="bf16",
            shared_tp_size=2,
            enable_omni_custom_models=False,
        ),
    )


def test_w8a8_shared_experts_tp_gt_1_baselayer(
    distributed_worker_pool,
) -> None:
    _run_case(
        distributed_worker_pool,
        MoERegressionCase(
            num_tokens=33,
            expected_strategy="dispatch_combine",
            quant_mode="w8a8",
            shared_tp_size=2,
            enable_omni_custom_models=False,
        ),
    )
