# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Reload consistency for RL weights after sleep/wake: within a single test case, sequentially
exercise high-performance ``PanguUltraMoEForCausalLM`` and baselayer ``OpenPanguModel``.

Both parts use a real ``NPUWorker.init_device`` (HCCL, ``NPUModelRunner``).

Run with::

    pytest tests/unit/layers/st/test_pangu_v1_update_weight.py
"""
from __future__ import annotations

import os
import re
import tempfile
import time
from collections.abc import Mapping
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch_npu

import vllm.model_executor.model_loader.base_loader as base_loader
import vllm.model_executor.models.openpangu as openpangu_mod
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader import utils as loader_utils

from omni.model_config.config_loader.loader import model_extra_config
from omni.v1.models.pangu import pangu_ultra_moe as pangu_ultra_moe_mod
from omni.worker.npu_worker import NPUWorker
from tests.unit.platform.utils import DeviceConfig, create_vllm_config

ParamLayoutSnapshot = tuple[tuple[int, ...], torch.dtype, str, int]
_BF16_MODEL_CFG_STUB = SimpleNamespace(dtype=torch.bfloat16, quantization=None)


def _teardown_distributed() -> None:
    """Fully tear down vLLM's distributed state, not just the torch PG.

    Calling only ``dist.destroy_process_group()`` kills the underlying torch
    process groups but leaves vLLM's cached ``_WORLD`` / model-parallel groups
    pointing at the now-dead groups. A later test running in the same worker
    (e.g. pangumtp_st's ``_init_hccl_dist``) would then re-initialize on top of
    that stale state and crash with "process group is not initialized in the
    world group map". Destroy vLLM's wrappers while the torch PGs are still
    alive so no dangling references survive.
    """
    from vllm.distributed import (destroy_distributed_environment,
                                  destroy_model_parallel)
    destroy_model_parallel()
    destroy_distributed_environment()


def _require_same_dict_keys(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    context: str,
) -> None:
    a, b = set(first), set(second)
    if a != b:
        raise AssertionError(
            f"{context}: key mismatch. "
            f"missing_in_second={sorted(a - b)}, missing_in_first={sorted(b - a)}"
        )


def _snapshot_parameters_from_model(model: torch.nn.Module) -> dict[str, ParamLayoutSnapshot]:
    return {
        name: (
            tuple(param.shape),
            param.dtype,
            str(param.device),
            torch_npu.get_npu_format(param),
        )
        for name, param in model.named_parameters()
    }


def _assert_parameter_layout_equal(
    first: dict[str, ParamLayoutSnapshot],
    second: dict[str, ParamLayoutSnapshot],
) -> None:
    _require_same_dict_keys(first, second, context="Layout snapshots")
    mismatches: list[tuple[str, ParamLayoutSnapshot, ParamLayoutSnapshot]] = []
    for key in sorted(first):
        if first[key] != second[key]:
            mismatches.append((key, first[key], second[key]))
    if mismatches:
        lines = []
        for name, a, b in mismatches:
            ash, adt, adev, afmt = a
            bsh, bdt, bdev, bfmt = b
            lines.append(
                f"  {name}: first shape={ash} dtype={adt} {adev=} {afmt=} | "
                f"second shape={bsh} dtype={bdt} {bdev=} {bfmt=}"
            )
        raise AssertionError(
            "Parameter layout mismatch (shape/dtype/device/npu_format):\n" + "\n".join(lines)
        )


def _snapshot_values_from_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _snapshot_attn_impl_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collect W_UK_T/W_UV from modules that directly own `impl`."""
    snapshot: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        impl = getattr(module, "impl", None)
        if impl is None:
            continue
        key_prefix = f"{module_name}.impl" if module_name else "impl"
        w_uk_t = getattr(impl, "W_UK_T", None)
        w_uv = getattr(impl, "W_UV", None)
        if isinstance(w_uk_t, torch.Tensor):
            snapshot[f"{key_prefix}.W_UK_T"] = w_uk_t.detach().clone()
        if isinstance(w_uv, torch.Tensor):
            snapshot[f"{key_prefix}.W_UV"] = w_uv.detach().clone()
    if not snapshot:
        raise AssertionError(
            "No W_UK_T/W_UV tensors were found under module.impl."
        )
    return snapshot


def _assert_value_snapshots_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    _require_same_dict_keys(first, second, context="Value snapshots")
    for name in sorted(first):
        a, b = first[name].detach(), second[name].detach()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(
                f"{name}: shape/dtype mismatch first={a.shape}/{a.dtype} "
                f"second={b.shape}/{b.dtype}"
            )
        if a.device != b.device:
            raise AssertionError(
                f"{name}: device mismatch first={a.device} second={b.device}"
            )
        if not torch.allclose(a, b, rtol=1e-5, atol=1e-8):
            raise AssertionError(f"{name}: tensors not close enough on device={a.device}")


def _build_random_weights(
    schema: dict[str, tuple[tuple[int, ...], torch.dtype]], seed: int
) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    out: list[tuple[str, torch.Tensor]] = []
    for name, (shape, dtype) in schema.items():
        out.append((name, torch.randn(shape, dtype=dtype, device="cpu", generator=generator)))
    return out


def _patch_npu_layer_static_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "omni.layers.fused_moe.prepare_permute_unpermute_finalize.current_platform",
        SimpleNamespace(device_type="npu"),
    )
    monkeypatch.setattr(
        "omni.layers.fused_moe.layer.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "omni.layers.fused_moe.layer.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "omni.v1.layers.vocab_parallel_embedding.get_local_world_group",
        lambda: SimpleNamespace(world_size=1),
    )

class _PanguUltraMoEModelRegistry:
    def resolve_model_cls(self, _architectures, model_config=None):
        del model_config
        return (
            pangu_ultra_moe_mod.PanguUltraMoEForCausalLM,
            "PanguUltraMoEForCausalLM",
        )


def _build_ut_hf_config_mla_and_dsa(architectures: list[str]) -> SimpleNamespace:
    num_hidden_layers = 20
    dsa_layers = [0, 2, 4, 6, 8]
    hidden_size = 1024
    num_attention_heads = 16
    return SimpleNamespace(
        architectures=architectures,
        model_type="openpangu_v2",
        pad_token_id=0,
        vocab_size=8192,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
        first_k_dense_replace=1,
        intermediate_size=1024,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=512,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        router_enable_expert_bias=False,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=0,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=16,
        index_topk=64,
        index_n_heads=4,
        index_head_dim=32,
        dsa_layers=dsa_layers,
        param_sink_number=0,
        use_cache=True,
        torch_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
    )

def _apply_common_model_config(mc) -> None:
    mc._get_transformers_backend_cls = lambda: "__none__"
    mc.convert_type = "none"
    mc.model_impl = "vllm"
    mc.runner_type = "generate"
    mc.seed = 0
    mc.trust_remote_code = False
    mc.dtype = torch.bfloat16
    mc.quantization = None
    mc.device = "npu"
    mc.enforce_eager = True


class _OpenPanguModelRegistry:
    def resolve_model_cls(self, _architectures, model_config=None):
        del model_config
        return openpangu_mod.OpenPanguModel, "OpenPanguModel"


def _build_model_specs() -> list[_ModelRunSpec]:
    return [
        SimpleNamespace(
            model_name="ut-pangu-ultra-moe",
            seed=7,
            registry=_PanguUltraMoEModelRegistry(),
            hf_config_factory=lambda: _build_ut_hf_config_mla_and_dsa(
                ["PanguUltraMoEForCausalLM"]
            ),
            split_gate_up_proj=True,
        ),
        SimpleNamespace(
            model_name="ut-openpangu",
            seed=11,
            registry=_OpenPanguModelRegistry(),
            hf_config_factory=lambda: _build_ut_hf_config_mla_and_dsa(["OpenPanguModel"]),
            split_gate_up_proj=False,
        ),
    ]


def _create_real_worker() -> tuple[NPUWorker, str]:
    if dist.is_initialized():
        raise RuntimeError("torch.distributed already initialized; run this test in isolation.")
    vllm_cfg = create_vllm_config()
    vllm_cfg.device_config = DeviceConfig("npu")
    vllm_cfg.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    mc = vllm_cfg.model_config
    _apply_common_model_config(mc)
    # Shared worker only initializes once; real model fields are overwritten per case.
    mc.registry = _PanguUltraMoEModelRegistry()
    mc.hf_config = _build_ut_hf_config_mla_and_dsa(["PanguUltraMoEForCausalLM"])
    mc.model = "ut-shared-worker"
    fd, rendezvous_path = tempfile.mkstemp(prefix="omni_ut_dist_")
    os.close(fd)
    init_method = f"file://{rendezvous_path}"
    try:
        worker = NPUWorker(
            vllm_config=vllm_cfg,
            local_rank=0,
            rank=0,
            distributed_init_method=init_method,
            is_driver_worker=True,
        )
        worker.init_device()
        return worker, rendezvous_path
    except Exception:
        if dist.is_initialized():
            _teardown_distributed()
        if os.path.isfile(rendezvous_path):
            try:
                os.unlink(rendezvous_path)
            except OSError:
                pass
        raise

def _bind_runner_model(worker: NPUWorker, model: torch.nn.Module) -> None:
    """Attach the loaded module to the real runner so sleep/wake can move weights."""
    worker.model_runner.model = model


def _init_model(
    worker: NPUWorker,
    monkeypatch: pytest.MonkeyPatch,
    spec: _ModelRunSpec,
) -> torch.nn.Module:
    _patch_npu_layer_static_imports(monkeypatch)
    mc = worker.vllm_config.model_config
    _apply_common_model_config(mc)
    mc.registry = spec.registry
    mc.hf_config = spec.hf_config_factory()
    mc.model = spec.model_name
    worker.vllm_config.load_config = LoadConfig(load_format="dummy")
    model_loader = get_model_loader(worker.vllm_config.load_config)
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        lambda *_a, **_k: None,
    )
    worker.vllm_config.compilation_config.static_forward_context.clear()
    return model_loader.load_model(
        vllm_config=worker.vllm_config,
        model_config=worker.vllm_config.model_config,
    )


_EXPERT_W13_RE = re.compile(r"^(.*\.mlp\.experts)\.w13_weight$")
_EXPERT_W2_RE = re.compile(r"^(.*\.mlp\.experts)\.w2_weight$")
_GATE_UP_PROJ_RE = re.compile(r"^(.*)\.gate_up_proj\.weight$")


def _build_ckpt_schema(
    model: torch.nn.Module,
    spec: _ModelRunSpec,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    schema: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        dtype = param.dtype
        m_w13 = _EXPERT_W13_RE.match(name)
        if m_w13:
            if len(shape) != 3:
                raise ValueError(f"{name} must be rank-3, got shape={shape}")
            num_experts, double_intermediate, hidden = shape
            if double_intermediate % 2 != 0:
                raise ValueError(
                    f"{name} shape={shape} has non-even middle dim for w13 split"
                )
            intermediate = double_intermediate // 2
            prefix = m_w13.group(1)
            for expert_id in range(num_experts):
                schema[f"{prefix}.{expert_id}.gate_proj.weight"] = (
                    (intermediate, hidden),
                    dtype,
                )
                schema[f"{prefix}.{expert_id}.up_proj.weight"] = (
                    (intermediate, hidden),
                    dtype,
                )
            continue

        m_w2 = _EXPERT_W2_RE.match(name)
        if m_w2:
            if len(shape) != 3:
                raise ValueError(f"{name} must be rank-3, got shape={shape}")
            if spec.model_name == "ut-pangu-ultra-moe":
                if getattr(param, "is_weight_transposed", False):
                    num_experts, hidden, intermediate = shape
                else:
                    num_experts, intermediate, hidden = shape
                down_proj_shape = (intermediate, hidden)
            elif spec.model_name == "ut-openpangu":
                num_experts, hidden, intermediate = shape
                down_proj_shape = (hidden, intermediate)
            else:
                raise ValueError(f"Unsupported model_name for w2 parse: {spec.model_name}")
            prefix = m_w2.group(1)
            for expert_id in range(num_experts):
                schema[f"{prefix}.{expert_id}.down_proj.weight"] = (down_proj_shape, dtype)
            continue

        if spec.split_gate_up_proj:
            m_gate_up = _GATE_UP_PROJ_RE.match(name)
            if m_gate_up:
                if len(shape) != 2:
                    raise ValueError(f"{name} must be rank-2, got shape={shape}")
                double_intermediate, hidden = shape
                if double_intermediate % 2 != 0:
                    raise ValueError(
                        f"{name} shape={shape} has non-even first dim for gate/up split"
                    )
                intermediate = double_intermediate // 2
                prefix = m_gate_up.group(1)
                schema[f"{prefix}.gate_proj.weight"] = ((intermediate, hidden), dtype)
                schema[f"{prefix}.up_proj.weight"] = ((intermediate, hidden), dtype)
                continue

        schema[name] = (shape, dtype)
    return schema


def _run_model_case(
    worker: NPUWorker,
    monkeypatch: pytest.MonkeyPatch,
    spec: _ModelRunSpec,
) -> None:
    # build weights
    model = _init_model(worker, monkeypatch, spec)
    schema = _build_ckpt_schema(model, spec)
    weights = _build_random_weights(schema, seed=spec.seed)

    # Auto weights loader
    model_auto = _init_model(worker, monkeypatch, spec)
    model_auto.load_weights(iter(weights))
    post_weight_load = getattr(model_auto, "post_weight_load", None)
    if post_weight_load:
        post_weight_load()
    loader_utils.process_weights_after_loading(
        model_auto, _BF16_MODEL_CFG_STUB, torch.device("npu")
    )
    first_layout = _snapshot_parameters_from_model(model_auto)
    first_values = _snapshot_values_from_model(model_auto)
    first_attn_impl_weights = _snapshot_attn_impl_weights(model_auto)

    # simulate RL weight update process: dummy_load + update_weights.
    # but when open mtp RL : auto + update_weight
    model_rl = _init_model(worker, monkeypatch, spec)
    model_rl.load_weights(iter(weights))
    post_weight_load = getattr(model_rl, "post_weight_load", None)
    if post_weight_load:
        post_weight_load()
    loader_utils.process_weights_after_loading(
        model_rl, _BF16_MODEL_CFG_STUB, torch.device("npu")
    )
    _bind_runner_model(worker, model_rl)
    free_before_sleep = torch.npu.mem_get_info()[0]
    worker.sleep(level=1)
    free_after_sleep = torch.npu.mem_get_info()[0]
    first_time_sleep_free = free_after_sleep - free_before_sleep
    
    worker.wake_up(tags=["weights"])
    time.sleep(1)
    model_rl.load_weights(iter(weights))
    if post_weight_load:
        post_weight_load()
    second_layout = _snapshot_parameters_from_model(model_rl)
    second_values = _snapshot_values_from_model(model_rl)
    second_attn_impl_weights = _snapshot_attn_impl_weights(model_rl)
    worker.wake_up(tags=["kv_cache"])

    _assert_parameter_layout_equal(first_layout, second_layout)
    _assert_value_snapshots_equal(first_values, second_values)
    _assert_value_snapshots_equal(first_attn_impl_weights, second_attn_impl_weights)


def test_high_player_update_weight(monkeypatch: pytest.MonkeyPatch):
    """Share one real worker, load Ultra/OpenPangu per model spec in order, and verify reload consistency."""
    monkeypatch.setattr(
        "omni.layers.fused_moe.layer.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "omni.layers.fused_moe.layer.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    high_performance_spec, base_layer_spec = _build_model_specs()
    rendezvous_path: str | None = None
    try:
        worker, rendezvous_path = _create_real_worker()
        _run_model_case(worker, monkeypatch, high_performance_spec)
        _run_model_case(worker, monkeypatch, base_layer_spec)
    finally:
        if dist.is_initialized():
            _teardown_distributed()
        if rendezvous_path and os.path.isfile(rendezvous_path):
            try:
                os.unlink(rendezvous_path)
            except OSError:
                pass