# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for patch_health (vLLM 0.25.1 adaptation).

Every vLLM, fastapi and omni_npu dependency is stubbed, so the test body itself
needs nothing but the standard library -- no vllm, no torch, no NPU.  It still
sits next to its siblings under tests/unit/, whose conftest imports torch, so a
torch-less environment skips the whole directory rather than this file alone.
Run either way::

    pytest tests/unit/vllm_patch/useful_patch/test_patch_health.py -v
    python  tests/unit/vllm_patch/useful_patch/test_patch_health.py
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

# <repo>/tests/unit/vllm_patch/useful_patch/ -> parents[3] is <repo>/tests,
# parents[4] is the repo root.  Note the spelling difference that the directory
# names carry: the test tree says "useful_patch", the source tree "usefull_patch".
PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "omni/vllm_patches/usefull_patch/common/patch_health.py"
)

HANG_SEC = 60.0


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


class _VLLMPatch:
    """Faithful stand-in for omni_npu.vllm_patches.core.VLLMPatch.

    Targets here are a mix of classes and one *module*; the real apply() does
    plain setattr either way, so the same stub covers both.
    """

    _attr_names_to_apply: list[str] = []

    @classmethod
    def apply(cls):
        target = cls._target
        if not hasattr(target, "_omni_npu_applied_patches"):
            target._omni_npu_applied_patches = {}
        for name in cls._attr_names_to_apply:
            if name in target._omni_npu_applied_patches:
                raise ValueError(
                    f"{target.__name__}.{name} already patched by "
                    f"{target._omni_npu_applied_patches[name]}"
                )
            target._omni_npu_applied_patches[name] = cls.__name__
            setattr(target, name, cls.__dict__[name])


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _Recorder:
    """Collects the ordered trace of everything the patch does."""

    def __init__(self):
        self.calls = []
        self.registered = []
        # heartbeat.stalled_engines answers with this; tests rewrite it.
        self.stalled = []
        self.stalled_args = []

    def register_patch(self, name, target):
        def decorator(cls):
            cls._target = target
            self.registered.append((name, target, tuple(cls._attr_names_to_apply)))
            return cls

        return decorator


class _JSONResponse:
    def __init__(self, status_code=None, content=None):
        self.status_code = status_code
        self.content = content


def _install_stubs(recorder):
    """(Re)install the stub tree; returns the fake targets under test."""

    class AsyncLLM:
        async def check_health(self):
            recorder.calls.append(("orig_check_health",))

    class OutputProcessor:
        def __init__(self, unfinished=0):
            self._unfinished = unfinished

        def get_num_unfinished_requests(self):
            return self._unfinished

        def add_request(self, *args, **kwargs):
            recorder.calls.append(("orig_add_request", args, kwargs))
            return "orig-result"

    def register_instrumentator_api_routers(app):
        recorder.calls.append(("orig_register_routers", app))

    _module("vllm")
    _module("vllm.logger", init_logger=lambda _name: types.SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    ))
    _module("vllm.v1")
    _module("vllm.v1.engine")
    _module("vllm.v1.engine.async_llm", AsyncLLM=AsyncLLM)
    _module("vllm.v1.engine.output_processor", OutputProcessor=OutputProcessor)

    # `import vllm.entrypoints.serve.instrumentator as m` resolves the name by
    # walking parent attributes, so every level has to be wired up, not just
    # dropped into sys.modules.
    entrypoints = _module("vllm.entrypoints")
    serve = _module("vllm.entrypoints.serve")
    instrumentator = _module(
        "vllm.entrypoints.serve.instrumentator",
        register_instrumentator_api_routers=register_instrumentator_api_routers,
    )
    sys.modules["vllm"].entrypoints = entrypoints
    entrypoints.serve = serve
    serve.instrumentator = instrumentator

    fastapi = _module("fastapi")
    fastapi.responses = _module("fastapi.responses", JSONResponse=_JSONResponse)

    heartbeat = _module(
        "omni_npu.diagnostics.watchdog.heartbeat",
        stalled_engines=lambda sec: (
            recorder.stalled_args.append(sec) or recorder.stalled
        ),
        snapshot=lambda: {0: 61.5, 1: 62.5},
        mark_busy=lambda: recorder.calls.append(("mark_busy",)),
    )
    envs = _module("omni_npu.envs", OMNI_HEALTH_HANG_SEC=HANG_SEC)
    _module("omni_npu", envs=envs)
    _module("omni_npu.diagnostics")
    _module("omni_npu.diagnostics.watchdog", heartbeat=heartbeat)
    _module("omni_npu.vllm_patches")
    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=_VLLMPatch,
        register_patch=recorder.register_patch,
    )

    return AsyncLLM, OutputProcessor, instrumentator


def _load_patch_module():
    recorder = _Recorder()
    targets = _install_stubs(recorder)
    spec = importlib.util.spec_from_file_location("_patch_health", PATCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, recorder, targets


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_registers_the_three_mounts():
    """Detection, reporting and the busy-edge stamp, one mount each."""
    _mod, rec, (AsyncLLM, OutputProcessor, instrumentator) = _load_patch_module()

    assert [(n, attrs) for n, _t, attrs in rec.registered] == [
        ("HealthHangPatch", ("check_health",)),
        ("HealthExceptionHandlerPatch", ("register_instrumentator_api_routers",)),
        ("BusySinceAddRequestPatch", ("add_request",)),
    ]
    assert [t for _n, t, _a in rec.registered] == [
        AsyncLLM,
        instrumentator,
        OutputProcessor,
    ]


def test_handler_mount_is_the_instrumentator_package():
    """Regression guard for the 0.25.1 breakage.

    The handler used to be registered from instrumentator.health.attach_router,
    which 0.25.1 deleted by inlining it into the package-level
    register_instrumentator_api_routers. Mounting on the vanished name raises
    AttributeError at *import* time, which import_patches_from_dir does not
    catch -- it would take down apply_patches() as a whole, not just this file.
    """
    _mod, rec, (_AsyncLLM, _OutputProcessor, instrumentator) = _load_patch_module()

    handler_mounts = [
        (t, attrs) for n, t, attrs in rec.registered if "ExceptionHandler" in n
    ]
    assert handler_mounts == [
        (instrumentator, ("register_instrumentator_api_routers",))
    ]
    assert "attach_router" not in sys.modules["vllm.entrypoints.serve.instrumentator"].__dict__


def test_hang_error_keeps_the_server_alive():
    """EngineHangError must not look like a dead engine.

    Inheriting EngineDeadError (or setting `errored`) would route the hang into
    vLLM's shutdown path; the whole point is that the server stays up and
    recovers on its own once the engine moves again.
    """
    mod, _rec, _targets = _load_patch_module()

    assert mod.EngineHangError.__bases__ == (Exception,)
    exc = mod.EngineHangError([0], {0: 61.5}, 3)
    assert (exc.engines, exc.stalled_sec, exc.in_flight) == ([0], {0: 61.5}, 3)


def test_check_health_raises_when_stalled_and_busy():
    """Both halves of the verdict, and the community check still runs first."""
    mod, rec, _targets = _load_patch_module()
    mod.HealthHangPatch.apply()

    engine = mod.HealthHangPatch._target()
    engine.output_processor = sys.modules[
        "vllm.v1.engine.output_processor"
    ].OutputProcessor(unfinished=3)
    rec.stalled = [0]

    try:
        asyncio.run(engine.check_health())
    except mod.EngineHangError as exc:
        assert exc.engines == [0]
        assert exc.in_flight == 3
        assert exc.stalled_sec == {0: 61.5}  # read back from heartbeat.snapshot()
    else:
        raise AssertionError("stalled engine with work in flight must raise")

    assert rec.calls == [("orig_check_health",)]
    assert rec.stalled_args == [HANG_SEC]  # threshold comes from envs, as a float


def test_check_health_stays_quiet_when_idle():
    """stalled_engines() is a pure clock check, so idle engines look stalled.

    in_flight is the second, independent source that keeps a healthy idle engine
    from being reported -- without it every quiet deployment would flip to 503.
    """
    mod, rec, _targets = _load_patch_module()
    mod.HealthHangPatch.apply()

    engine = mod.HealthHangPatch._target()
    engine.output_processor = sys.modules[
        "vllm.v1.engine.output_processor"
    ].OutputProcessor(unfinished=0)
    rec.stalled = [0]

    asyncio.run(engine.check_health())  # must not raise
    assert rec.calls == [("orig_check_health",)]


def test_handler_registration_wraps_the_original():
    """The community routers still get registered, ours is added afterwards."""
    mod, rec, (_AsyncLLM, _OutputProcessor, instrumentator) = _load_patch_module()
    mod.HealthExceptionHandlerPatch.apply()

    handlers = {}

    class _App:
        def add_exception_handler(self, exc_type, handler):
            handlers[exc_type] = handler

    app = _App()
    instrumentator.register_instrumentator_api_routers(app)

    assert rec.calls == [("orig_register_routers", app)]
    assert list(handlers) == [mod.EngineHangError]


def test_handler_answers_503_with_a_body():
    """/health (and SageMaker /ping, same app) report why they are unhealthy."""
    mod, _rec, _targets = _load_patch_module()

    exc = mod.EngineHangError([1], {1: 62.5}, 2)
    response = asyncio.run(mod._engine_hang_handler(object(), exc))

    assert response.status_code == 503
    assert response.content == {
        "fault message": "engine hang",
        "engines": [1],
        "in_flight": 2,
    }


def test_busy_is_stamped_only_on_the_idle_edge():
    """Stamp the 0 -> >0 transition, not every request.

    Detecting the edge request-side is what frees hang detection from needing a
    health probe to land during the idle->busy window; stamping on every request
    would instead keep pushing the busy start forward and mask a real hang.
    """
    mod, rec, _targets = _load_patch_module()
    mod.BusySinceAddRequestPatch.apply()

    OutputProcessor = mod.BusySinceAddRequestPatch._target

    idle = OutputProcessor(unfinished=0)
    assert idle.add_request("req-1", kw=1) == "orig-result"
    assert rec.calls == [("mark_busy",), ("orig_add_request", ("req-1",), {"kw": 1})]

    rec.calls.clear()
    busy = OutputProcessor(unfinished=2)
    busy.add_request("req-2")
    assert rec.calls == [("orig_add_request", ("req-2",), {})]


def test_original_callables_are_captured_before_apply():
    """The relay must chain the *unpatched* callables, never itself."""
    mod, _rec, (AsyncLLM, OutputProcessor, instrumentator) = _load_patch_module()

    assert mod._orig_check_health is AsyncLLM.__dict__["check_health"]
    assert mod._orig_add_request is OutputProcessor.__dict__["add_request"]
    assert (
        mod._orig_register_routers
        is instrumentator.register_instrumentator_api_routers
    )

    # After apply the attribute is the wrapper, but the captured reference still
    # points at the original -- otherwise infinite recursion.
    mod.HealthExceptionHandlerPatch.apply()
    assert instrumentator.register_instrumentator_api_routers is not (
        mod._orig_register_routers
    )


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("=" * 60)
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
