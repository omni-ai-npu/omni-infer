# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for patch_dump (vLLM 0.25.1 adaptation).

Every vLLM and omni_npu dependency is stubbed, so the test body itself needs
nothing but the standard library -- no vllm, no torch, no NPU.  It still sits
next to its siblings under tests/unit/, whose conftest imports torch, so a
torch-less environment skips the whole directory rather than this file alone.
Run either way::

    pytest tests/unit/vllm_patch/useful_patch/test_patch_dump.py -v
    python  tests/unit/vllm_patch/useful_patch/test_patch_dump.py
"""

import importlib.util
import sys
import types
from pathlib import Path

# <repo>/tests/unit/vllm_patch/useful_patch/ -> parents[3] is <repo>/tests,
# parents[4] is the repo root.  Note the spelling difference that the directory
# names carry: the test tree says "useful_patch", the source tree "usefull_patch".
PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "omni/vllm_patches/usefull_patch/common/patch_dump.py"
)


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


class _VLLMPatch:
    """Faithful stand-in for omni_npu.vllm_patches.core.VLLMPatch.

    The hasattr() bookkeeping lookup is reproduced exactly: subclasses inherit
    it, which is the whole trap. A stub that just called setattr would pass
    regardless.
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

    def register_patch(self, name, target):
        def decorator(cls):
            cls._target = target
            self.registered.append((name, target, tuple(cls._attr_names_to_apply)))
            return cls

        return decorator


def _install_stubs(enable_dump, recorder):
    """(Re)install the stub tree; returns the fake classes under test."""

    class AsyncLLM:
        def __init__(self, vllm_config=None, **kwargs):
            recorder.calls.append(("orig_api_init", kwargs))
            # Only set *after* the original initialiser runs; on_api_init reads it.
            self.output_processor = object()

    class EngineCoreProc:
        def __init__(self, *args, **kwargs):
            self.model_executor = object()
            self.scheduler = object()

        def run_busy_loop(self):
            recorder.calls.append(("orig_run_busy_loop", type(self).__name__))

    # Mirrors 0.25.1: the DP variant reimplements run_busy_loop instead of
    # calling super(), which is why it needs its own mount.
    class DPEngineCoreProc(EngineCoreProc):
        def run_busy_loop(self):
            recorder.calls.append(("orig_dp_run_busy_loop", type(self).__name__))

    class NPUWorker:
        def init_device(self):
            recorder.calls.append(("orig_init_device",))
            self.local_rank = 3

    _module("vllm")
    _module("vllm.logger", init_logger=lambda _name: types.SimpleNamespace(
        warning=lambda *a, **k: None, info=lambda *a, **k: None
    ))
    _module("vllm.v1")
    _module("vllm.v1.engine")
    _module("vllm.v1.engine.async_llm", AsyncLLM=AsyncLLM)
    _module(
        "vllm.v1.engine.core",
        EngineCoreProc=EngineCoreProc,
        DPEngineCoreProc=DPEngineCoreProc,
    )

    envs = _module("omni_npu.envs", OMNI_DUMP_ENABLE=enable_dump)
    hooks = _module(
        "omni_npu.diagnostics.dump.hooks",
        on_api_init=lambda obj: recorder.calls.append(
            ("on_api_init", hasattr(obj, "output_processor"))
        ),
        on_engine_init=lambda obj: recorder.calls.append(
            ("on_engine_init", hasattr(obj, "model_executor"), hasattr(obj, "scheduler"))
        ),
        on_worker_init=lambda obj: recorder.calls.append(
            ("on_worker_init", getattr(obj, "local_rank", None))
        ),
    )
    _module("omni_npu", envs=envs)
    _module("omni_npu.diagnostics")
    _module("omni_npu.diagnostics.dump", hooks=hooks)
    _module("omni_npu.vllm_patches")
    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=_VLLMPatch,
        register_patch=recorder.register_patch,
    )
    _module("omni_npu.worker")
    _module("omni_npu.worker.npu_worker", NPUWorker=NPUWorker)

    return AsyncLLM, EngineCoreProc, DPEngineCoreProc, NPUWorker


def _load_patch_module(enable_dump=True):
    recorder = _Recorder()
    targets = _install_stubs(enable_dump, recorder)
    spec = importlib.util.spec_from_file_location("_patch_dump", PATCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, recorder, targets


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_registers_a_mount_per_role_plus_the_dp_engine_variant():
    """One mount per process role, and a second engine mount for the DP variant."""
    _mod, rec, (AsyncLLM, Proc, DPProc, NPUWorker) = _load_patch_module(enable_dump=True)

    assert [(n, attrs) for n, _t, attrs in rec.registered] == [
        ("ExitDumpApiPatch", ("__init__",)),
        ("ExitDumpEnginePatch", ("run_busy_loop",)),
        ("ExitDumpDPEnginePatch", ("run_busy_loop",)),
        ("ExitDumpWorkerPatch", ("init_device",)),
    ]
    assert [t for _n, t, _a in rec.registered] == [AsyncLLM, Proc, DPProc, NPUWorker]


def test_engine_mount_is_not_engine_core_init():
    """Regression guard for the spawn failure.

    vLLM loads general plugins from inside EngineCore.__init__, so under spawn
    the patch lands while that call is already on the stack and a hook at its
    end never runs. The mount must therefore be something entered after
    construction returns.
    """
    _mod, rec, _targets = _load_patch_module(enable_dump=True)
    engine_mounts = [attrs for n, _t, attrs in rec.registered if "Engine" in n]
    assert engine_mounts == [("run_busy_loop",), ("run_busy_loop",)]
    assert all("__init__" not in attrs for attrs in engine_mounts)


def test_gate_off_registers_nothing_at_all():
    """OMNI_DUMP_ENABLE=0 must leave the hot path without any wrapper."""
    _mod, rec, _targets = _load_patch_module(enable_dump=False)
    assert rec.registered == []
    assert rec.calls == []


def test_api_hook_runs_after_the_original_init():
    """on_api_init reads async_llm.output_processor, which __init__ creates."""
    mod, rec, _targets = _load_patch_module(enable_dump=True)
    mod.ExitDumpApiPatch.apply()

    # 0.25.1's from_vllm_config calls AsyncLLM(**kwargs) with keywords only, and
    # no longer passes use_cached_outputs; the *args/**kwargs relay is agnostic.
    instance = mod.ExitDumpApiPatch._target(vllm_config="cfg", log_stats=True)

    assert rec.calls == [
        ("orig_api_init", {"log_stats": True}),
        ("on_api_init", True),  # output_processor already present
    ]
    assert hasattr(instance, "output_processor")


def test_engine_hook_runs_before_the_busy_loop():
    """The hook must precede the loop so wrap_executor covers the first step.

    on_engine_init reads model_executor and scheduler; __init__ has already
    returned by the time run_busy_loop is entered, so both exist.
    """
    mod, rec, _targets = _load_patch_module(enable_dump=True)
    mod.ExitDumpEnginePatch.apply()

    engine = mod.ExitDumpEnginePatch._target("vllm_config")
    engine.run_busy_loop()

    assert rec.calls == [
        ("on_engine_init", True, True),
        ("orig_run_busy_loop", "EngineCoreProc"),
    ]


def test_dp_engine_mount_applies_alongside_its_parent():
    """Both engine mounts must take effect, in either application order.

    VLLMPatch.apply() finds its "already patched" bookkeeping with hasattr(),
    which DPEngineCoreProc inherits from EngineCoreProc. Without the patch
    module giving the subclass its own dict, whichever mount applied second
    would raise "already patched by ..." -- swallowed by apply_patch into one
    log line, leaving that deployment shape silently without forensics.
    """
    for order in (
        ("ExitDumpEnginePatch", "ExitDumpDPEnginePatch"),
        ("ExitDumpDPEnginePatch", "ExitDumpEnginePatch"),
    ):
        mod, rec, _targets = _load_patch_module(enable_dump=True)
        for patch_name in order:
            getattr(mod, patch_name).apply()  # must not raise

        dp_engine = mod.ExitDumpDPEnginePatch._target("vllm_config")
        dp_engine.run_busy_loop()

        assert rec.calls == [
            ("on_engine_init", True, True),
            ("orig_dp_run_busy_loop", "DPEngineCoreProc"),
        ], f"DP mount ineffective when applied in order {order}"


def test_worker_hook_runs_after_init_device():
    """on_worker_init resolves the rank, which init_device assigns."""
    mod, rec, _targets = _load_patch_module(enable_dump=True)
    mod.ExitDumpWorkerPatch.apply()

    worker = mod.ExitDumpWorkerPatch._target()
    worker.init_device()

    assert rec.calls == [("orig_init_device",), ("on_worker_init", 3)]


def test_original_callables_are_captured_before_apply():
    """The relay must chain the *unpatched* callables, never itself."""
    mod, _rec, (AsyncLLM, Proc, DPProc, NPUWorker) = _load_patch_module(enable_dump=True)

    assert mod._orig_api_init is AsyncLLM.__dict__["__init__"]
    assert mod._orig_engine_loop is Proc.__dict__["run_busy_loop"]
    assert mod._orig_dp_engine_loop is DPProc.__dict__["run_busy_loop"]
    assert mod._orig_init_device is NPUWorker.__dict__["init_device"]

    # After apply the class attribute is the wrapper, but the captured
    # reference still points at the original -- otherwise infinite recursion.
    mod.ExitDumpApiPatch.apply()
    assert AsyncLLM.__init__ is not mod._orig_api_init


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
