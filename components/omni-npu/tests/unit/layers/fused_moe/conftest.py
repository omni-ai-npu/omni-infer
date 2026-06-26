# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _ensure_module(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _register_oot(cls, subcls):
    return subcls


class _Registerable:
    register_oot = classmethod(_register_oot)


@pytest.fixture(autouse=True)
def _stub_fused_moe_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", SimpleNamespace(config=SimpleNamespace()), raising=False)
    if not hasattr(torch.npu, "config"):
        torch.npu.config = SimpleNamespace()
    if not hasattr(torch.npu.config, "allow_internal_format"):
        try:
            torch.npu.config.allow_internal_format = False
        except Exception:
            pass
    if not hasattr(torch.npu, "Event"):
        try:
            torch.npu.Event = type("Event", (), {})
        except Exception:
            pass
    if not hasattr(torch.npu, "current_stream"):
        try:
            torch.npu.current_stream = lambda: SimpleNamespace(
                wait_stream=lambda _other: None,
            )
        except Exception:
            pass
    if not hasattr(torch.npu, "stream"):
        class _NoopStream:
            def __init__(self, _stream=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False
        try:
            torch.npu.stream = _NoopStream
        except Exception:
            pass

    _ensure_module(monkeypatch, "torch_npu")

    omni_npu_pkg = _ensure_module(monkeypatch, "omni_npu")
    omni_npu_pkg.__path__ = [
        str(Path(__file__).resolve().parents[4] / "src" / "omni_npu")
    ]
    model_config_pkg = _ensure_module(monkeypatch, "omni_npu.model_config")
    model_config_pkg.__path__ = []
    config_loader_pkg = _ensure_module(monkeypatch, "omni_npu.model_config.config_loader")
    config_loader_pkg.__path__ = []
    loader_module = _ensure_module(
        monkeypatch, "omni_npu.model_config.config_loader.loader"
    )
    loader_module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            shared_expert_parallel_schedule="with_finalize",
            decode_moe_dispatch_combine=False,
            enable_moe_agrs=False,
            prefill_grouped_matmul_finalize_routing=False,
            enable_prefetch=False,
            shared_expert_multi_stream=False,
            enable_agrs_finalize_metadata_overlap=False,
            gmm_nz=True,
            router_gating_in_fp32=False,
        ),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    layers_utils_module = _ensure_module(monkeypatch, "omni_npu.layers.utils")
    layers_utils_module.named_stream = lambda _name: SimpleNamespace(
        wait_stream=lambda _other: None,
    )

    vllm_module = _ensure_module(monkeypatch, "vllm")
    vllm_module.__path__ = []
    logger_module = _ensure_module(monkeypatch, "vllm.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    logger_module.logger = logger_module.init_logger("vllm")

    distributed_module = _ensure_module(monkeypatch, "vllm.distributed")
    distributed_module.get_tp_group = lambda: SimpleNamespace(all_gather=lambda x, dim=0: x)
    distributed_module.get_ep_group = lambda: SimpleNamespace(rank=0, world_size=1)
    distributed_module.tensor_model_parallel_all_reduce = lambda x: x
    distributed_module.tensor_model_parallel_all_gather = lambda x, dim=0: x
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    distributed_module.get_tensor_model_parallel_rank = lambda: 0

    platforms_module = _ensure_module(monkeypatch, "vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu", dispatch_key="cpu")

    forward_context_module = _ensure_module(monkeypatch, "vllm.forward_context")
    forward_context_module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None, virtual_engine=0)

    model_executor_module = _ensure_module(monkeypatch, "vllm.model_executor")
    model_executor_module.__path__ = []
    layers_module = _ensure_module(monkeypatch, "vllm.model_executor.layers")
    layers_module.__path__ = []
    model_executor_utils_module = _ensure_module(monkeypatch, "vllm.model_executor.utils")
    model_executor_utils_module.set_weight_attrs = (
        lambda param, attrs: [setattr(param, k, v) for k, v in attrs.items()]
    )

    vllm_utils_module = _ensure_module(monkeypatch, "vllm.utils")
    vllm_utils_module.__path__ = []
    vllm_utils_module.random_uuid = lambda: "stub-uuid"
    math_utils_module = _ensure_module(monkeypatch, "vllm.utils.math_utils")
    math_utils_module.cdiv = lambda a, b: (a + b - 1) // b

    fused_moe_pkg = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe")
    fused_moe_pkg.__path__ = []

    class FusedMoEConfig:
        def __init__(self, ep_size=1, ep_rank=0, num_experts=1):
            self.ep_size = ep_size
            self.ep_rank = ep_rank
            self.num_experts = num_experts

    fused_moe_pkg.FusedMoEConfig = FusedMoEConfig

    fused_moe_layer_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe.layer")

    class FusedMoE(_Registerable):
        pass

    class UnquantizedFusedMoEMethod(_Registerable):
        pass

    import enum
    class FusedMoeWeightScaleSupported(enum.Enum):
        TENSOR = "tensor"
        CHANNEL = "channel"
        GROUP = "group"
        BLOCK = "block"

    fused_moe_layer_module.FusedMoE = FusedMoE
    fused_moe_layer_module.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod
    fused_moe_layer_module.FusedMoeWeightScaleSupported = FusedMoeWeightScaleSupported

    fused_moe_config_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe.config")

    fused_moe_config_module.FusedMoEConfig = FusedMoEConfig

    class FusedMoEQuantConfig:
        def __init__(self, use_int8_w8a8: bool = False):
            self.use_int8_w8a8 = use_int8_w8a8

    fused_moe_config_module.FusedMoEQuantConfig = FusedMoEQuantConfig

    routed_experts_capturer_module = _ensure_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer",
    )

    class RoutedExpertsCapturer:
        @classmethod
        def get_instance(cls):
            return None

        def capture(self, layer_id, topk_ids):
            return None

    routed_experts_capturer_module.RoutedExpertsCapturer = RoutedExpertsCapturer

    shared_fused_moe_module = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe.shared_fused_moe")

    class SharedFusedMoE(_Registerable):
        pass

    shared_fused_moe_module.SharedFusedMoE = SharedFusedMoE

    compressed_tensors_module = _ensure_module(
        monkeypatch,
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors",
    )

    class NPUCompressedTensorsConfig:
        pass

    compressed_tensors_module.NPUCompressedTensorsConfig = NPUCompressedTensorsConfig

    layers_pkg = _ensure_module(monkeypatch, "omni_npu.layers")
    layers_pkg.__path__ = [
        str(Path(__file__).resolve().parents[4] / "src" / "omni_npu" / "layers")
    ]
    omni_npu_v1_pkg = _ensure_module(monkeypatch, "omni_npu.v1")
    omni_npu_v1_pkg.__path__ = []
    omni_npu_v1_utils_module = _ensure_module(monkeypatch, "omni_npu.v1.utils")
    omni_npu_v1_utils_module.on_ascend950 = lambda: False
    omni_npu_compilation_pkg = _ensure_module(monkeypatch, "omni_npu.compilation")
    omni_npu_compilation_pkg.__path__ = []
    acl_graph_module = _ensure_module(monkeypatch, "omni_npu.compilation.acl_graph")
    acl_graph_module.set_aclgraph_recapture = lambda _flag: None
    
    vllm_module.logger = logger_module
    vllm_module.distributed = distributed_module
    vllm_module.platforms = platforms_module
    vllm_module.forward_context = forward_context_module
    vllm_module.model_executor = model_executor_module
    vllm_module.utils = vllm_utils_module
    model_executor_module.layers = layers_module
    model_executor_module.utils = model_executor_utils_module
    layers_module.fused_moe = fused_moe_pkg
    fused_moe_pkg.layer = fused_moe_layer_module
    fused_moe_pkg.shared_fused_moe = shared_fused_moe_module


@pytest.fixture(autouse=True)
def _force_cpu_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_arange = torch.arange
    orig_tensor = torch.tensor
    orig_ones = torch.ones
    orig_zeros = torch.zeros
    orig_full = torch.full
    orig_empty = torch.empty
    orig_rand = torch.rand
    orig_randn = torch.randn

    def _coerce_device(kwargs):
        dev = kwargs.get("device")
        if dev is None:
            try:
                if orig_tensor([]).device.type == "npu":
                    kwargs["device"] = "cpu"
            except Exception:
                pass
            return
        if isinstance(dev, torch.device):
            if dev.type == "npu":
                kwargs["device"] = "cpu"
        elif dev == "npu":
            kwargs["device"] = "cpu"

    def _wrap(fn):
        def _inner(*args, **kwargs):
            _coerce_device(kwargs)
            return fn(*args, **kwargs)
        return _inner

    monkeypatch.setattr(torch, "arange", _wrap(orig_arange))
    monkeypatch.setattr(torch, "tensor", _wrap(orig_tensor))
    monkeypatch.setattr(torch, "ones", _wrap(orig_ones))
    monkeypatch.setattr(torch, "zeros", _wrap(orig_zeros))
    monkeypatch.setattr(torch, "full", _wrap(orig_full))
    monkeypatch.setattr(torch, "empty", _wrap(orig_empty))
    monkeypatch.setattr(torch, "rand", _wrap(orig_rand))
    monkeypatch.setattr(torch, "randn", _wrap(orig_randn))
