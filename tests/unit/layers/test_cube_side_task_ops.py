# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


pytestmark = pytest.mark.unit


def _make_module(monkeypatch, name, is_package=False):
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def cube_side_task_ops(monkeypatch):
    """Stub vllm and omni_npu deps just enough to import cube_side_task_ops."""
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "src"))

    # vllm.forward_context — stub get_forward_context to return a mutable namespace
    fwctx = SimpleNamespace(
        no_compile_layers={},
        additional_kwargs={},
    )
    forward_context_module = _make_module(monkeypatch, "vllm.forward_context")
    forward_context_module.get_forward_context = lambda: fwctx
    forward_context_module.is_forward_context_available = lambda: True

    # vllm.utils.torch_utils — stub direct_register_custom_op to a no-op so we
    # can import without a real torch_npu / vllm_lib environment.
    utils_pkg = _make_module(monkeypatch, "vllm.utils", is_package=True)
    utils_pkg.__path__ = []
    torch_utils_module = _make_module(monkeypatch, "vllm.utils.torch_utils")
    torch_utils_module.direct_register_custom_op = lambda **_kw: None

    # omni_npu.layers.utils — supply CubeSideTask + key constant.
    _make_module(monkeypatch, "omni_npu", is_package=True)
    _make_module(monkeypatch, "omni_npu.layers", is_package=True)
    layer_utils_module = _make_module(monkeypatch, "omni_npu.layers.utils")
    from dataclasses import dataclass
    from typing import Callable, Optional

    @dataclass
    class _CubeSideTask:
        fn: Callable[[], None]
        done_event: Optional[object] = None

    layer_utils_module.CubeSideTask = _CubeSideTask
    layer_utils_module.CUBE_SIDE_TASKS_KEY = "cube_side_tasks"
    layer_utils_module.CUBE_SIDE_STREAM_NAME = "cube_side_task"
    layer_utils_module.named_stream = lambda _name: SimpleNamespace(
        wait_stream=lambda *_a: None,
        wait_event=lambda *_a: None,
    )
    mhc_pkg = _make_module(monkeypatch, "omni_npu.layers.mhc", is_package=True)
    mhc_pkg.__path__ = [str(repo_root / "omni" / "layers" / "mhc")]

    # Stub torch.npu.{current_stream, stream} so the closure can run without NPU.
    class _NoopStreamCtx:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    main_stream = SimpleNamespace(
        record_stream=lambda *_a: None,
    )

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: main_stream,
            stream=lambda _s: _NoopStreamCtx(),
        ),
        raising=False,
    )
    # Tensor.record_stream needs a real Stream. For tests, no-op it.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_a: None)

    module_name = "omni_npu.layers.mhc.cube_side_task_ops"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    return importlib.reload(module), fwctx


def test_register_then_fetch_runs_closure_and_returns_post_value(
    cube_side_task_ops,
):
    module, fwctx = cube_side_task_ops

    # Set up a fake layer with an mhc_module whose mhc_sinkhorn flips a flag and
    # returns a recognisable tensor.
    flag = {"called": False}
    post_value = torch.tensor([42.0])

    class _StubMhc:
        def mhc_sinkhorn(self, x):
            flag["called"] = True
            return post_value

    fake_layer = SimpleNamespace(mhc_module=_StubMhc())
    fwctx.no_compile_layers["my.layer"] = fake_layer

    h_res = torch.zeros(2, 3)
    returned = module.mhc_register("my.layer", "task.key", h_res)

    # mhc_register returns the input tensor unchanged so Dynamo sees a tensor flow.
    assert returned is h_res
    # Task and holder are recorded in additional_kwargs.
    assert "task.key" in fwctx.additional_kwargs[module.CUBE_SIDE_TASKS_KEY]
    assert "task.key" in fwctx.additional_kwargs[module.MHC_HOLDER_KEY]

    # Simulate the runner: invoke the registered task's fn().
    fwctx.additional_kwargs[module.CUBE_SIDE_TASKS_KEY]["task.key"].fn()
    assert flag["called"] is True

    # mhc_fetch pops the holder and returns the closure-mutated value.
    fetched = module.mhc_fetch("task.key", torch.tensor([0.0]))
    assert fetched is post_value
    assert "task.key" not in fwctx.additional_kwargs[module.MHC_HOLDER_KEY]


def test_register_returns_h_res_when_layer_has_no_mhc_module(cube_side_task_ops):
    module, fwctx = cube_side_task_ops

    fwctx.no_compile_layers["bare.layer"] = SimpleNamespace()
    h_res = torch.ones(4)
    returned = module.mhc_register("bare.layer", "k", h_res)
    assert returned is h_res
    # No registration happened.
    assert module.MHC_HOLDER_KEY not in fwctx.additional_kwargs


def test_fetch_returns_fallback_when_not_registered(cube_side_task_ops):
    module, fwctx = cube_side_task_ops
    fallback = torch.tensor([99.0])
    out = module.mhc_fetch("missing.key", fallback)
    assert out is fallback


def test_fake_impls_pass_through(cube_side_task_ops):
    module, _ = cube_side_task_ops
    h_res = torch.zeros(2)
    assert module.mhc_register_fake("layer", "key", h_res) is h_res
    fallback = torch.zeros(3)
    assert module.mhc_fetch_fake("key", fallback) is fallback


def _bind_torch_ops_vllm(monkeypatch, module):
    """Wire torch.ops.vllm.{mhc_register,mhc_fetch} to the module's Python
    functions. Needed because the fixture stubs direct_register_custom_op,
    so the real ops aren't registered in tests."""
    ops_ns = SimpleNamespace(
        mhc_register=module.mhc_register,
        mhc_fetch=module.mhc_fetch,
    )
    monkeypatch.setattr(torch.ops, "vllm", ops_ns, raising=False)


def test_maybe_register_passes_through_when_task_key_is_none(cube_side_task_ops):
    module, fwctx = cube_side_task_ops
    h_res = torch.ones(3)
    out = module.maybe_register_mhc_task("holder", None, h_res)
    assert out is h_res
    # No task or holder registered when task_key is None.
    assert module.CUBE_SIDE_TASKS_KEY not in fwctx.additional_kwargs
    assert module.MHC_HOLDER_KEY not in fwctx.additional_kwargs


def test_maybe_register_delegates_when_task_key_set(cube_side_task_ops, monkeypatch):
    module, fwctx = cube_side_task_ops
    _bind_torch_ops_vllm(monkeypatch, module)

    class _Stub:
        def mhc_sinkhorn(self, x):
            return x

    fwctx.no_compile_layers["layer"] = SimpleNamespace(mhc_module=_Stub())
    h_res = torch.zeros(2)
    out = module.maybe_register_mhc_task("layer", "task_k", h_res)
    assert out is h_res
    assert "task_k" in fwctx.additional_kwargs[module.CUBE_SIDE_TASKS_KEY]
    assert "task_k" in fwctx.additional_kwargs[module.MHC_HOLDER_KEY]


def test_resolve_runs_sinkhorn_when_task_key_is_none(cube_side_task_ops):
    module, _ = cube_side_task_ops
    seen = {"called": False}

    class _Stub:
        def mhc_sinkhorn(self, x):
            seen["called"] = True
            return x + 1

    h_res = torch.zeros(2)
    out = module.resolve_mhc_h_res(_Stub(), None, h_res)
    assert seen["called"] is True
    assert torch.equal(out, h_res + 1)


def test_resolve_fetches_when_task_key_set(cube_side_task_ops, monkeypatch):
    module, fwctx = cube_side_task_ops
    _bind_torch_ops_vllm(monkeypatch, module)

    post = torch.tensor([42.0])
    fwctx.additional_kwargs[module.MHC_HOLDER_KEY] = {"k": [post]}
    fallback = torch.zeros(1)
    out = module.resolve_mhc_h_res(MagicMock(), "k", fallback)
    assert out is post


def test_cube_side_run_no_op_when_no_task_registered(cube_side_task_ops):
    module, fwctx = cube_side_task_ops
    x = torch.zeros(2)
    out = module.cube_side_run("layer_x", x)
    assert out is x
    # No pending event stashed when no task fires.
    assert module.CUBE_SIDE_PENDING_KEY not in fwctx.additional_kwargs


def test_cube_side_run_fires_task_and_stashes_event(cube_side_task_ops):
    module, fwctx = cube_side_task_ops
    flag = {"called": False}

    class _DummyEvent:
        def __init__(self):
            self.recorded = False

        def record(self):
            self.recorded = True

    # Replace torch.npu.Event with our recorder so we can assert it was used.
    import torch as _torch
    _torch.npu.Event = _DummyEvent  # fixture's torch.npu is a SimpleNamespace

    def _fn():
        flag["called"] = True

    task = module.CubeSideTask(fn=_fn)
    fwctx.additional_kwargs[module.CUBE_SIDE_TASKS_KEY] = {"key_a": task}

    x = torch.ones(3)
    out = module.cube_side_run("key_a", x)

    assert out is x
    assert flag["called"] is True
    # Task popped from the registration dict.
    assert "key_a" not in fwctx.additional_kwargs[module.CUBE_SIDE_TASKS_KEY]
    # Event stashed for the matching wait op.
    pending = fwctx.additional_kwargs[module.CUBE_SIDE_PENDING_KEY]
    assert "key_a" in pending
    assert isinstance(pending["key_a"], _DummyEvent)
    assert pending["key_a"].recorded is True


def test_cube_side_wait_no_op_when_no_pending_event(cube_side_task_ops):
    module, _ = cube_side_task_ops
    y = torch.zeros(2)
    assert module.cube_side_wait("missing", y) is y


def test_cube_side_wait_consumes_pending_event(cube_side_task_ops):
    module, fwctx = cube_side_task_ops
    waited_on = []

    class _FakeStream:
        def wait_event(self, event):
            waited_on.append(event)

    fake_event = object()
    fwctx.additional_kwargs[module.CUBE_SIDE_PENDING_KEY] = {"key_b": fake_event}

    import torch as _torch
    _torch.npu.current_stream = lambda: _FakeStream()

    y = torch.ones(2)
    out = module.cube_side_wait("key_b", y)
    assert out is y
    assert waited_on == [fake_event]
    # Event popped from the pending dict.
    assert "key_b" not in fwctx.additional_kwargs[module.CUBE_SIDE_PENDING_KEY]


def test_cube_side_fake_impls_pass_through(cube_side_task_ops):
    module, _ = cube_side_task_ops
    x = torch.zeros(2)
    assert module.cube_side_run_fake("k", x) is x
    y = torch.zeros(3)
    assert module.cube_side_wait_fake("k", y) is y
