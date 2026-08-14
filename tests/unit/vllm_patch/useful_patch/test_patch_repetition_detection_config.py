# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for patch_repetition_detection_config (vLLM 0.25.1).

Pure-stdlib: every vLLM and omni_npu dependency is stubbed, so this needs no
vllm, torch, pydantic or NPU. Deliberately placed outside tests/unit/ for the
same reason as tests/config/: that conftest imports torch at module level.
Run with pytest or directly.

The stubs reproduce the three 0.25.1 facts the patch leans on:
repetition_detection is a real SamplingParams field, EngineArgs.from_cli_args
rebuilds from dataclass fields only, and process_inputs takes params as its 3rd
positional argument and clones it internally.
"""

import argparse
import importlib.util
import logging
import sys
import types
from dataclasses import dataclass, fields
from pathlib import Path

# <repo>/tests/unit/vllm_patch/useful_patch/ -> parents[4] is <repo>
PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "omni/vllm_patches/usefull_patch/patch_repetition_detection_config.py"
)

REGISTERED = []


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


@dataclass
class _RepetitionDetectionParams:
    """Mirrors vllm/sampling_params.py:146, including its validation."""

    max_pattern_size: int = 0
    min_pattern_size: int = 0
    min_count: int = 0

    def __post_init__(self):
        if (
            self.max_pattern_size < 0
            or self.min_pattern_size < 0
            or self.min_pattern_size > self.max_pattern_size
        ):
            raise ValueError("bad pattern sizes")
        if self.max_pattern_size > 0 and self.min_count < 2:
            raise ValueError("min_count must be >= 2")


class _SamplingParams:
    def __init__(self, repetition_detection=None):
        self.repetition_detection = repetition_detection


class _PoolingParams:
    pass


@dataclass
class _VllmConfig:
    """Stands in for the pydantic dataclass.  A real dataclass, so _vllm_replace
    below can run against it."""

    seen_usage_context: object = None
    seen_headless: bool = False


def _vllm_replace(instance, /, **kwargs):
    """Clone of vllm/config/utils.py `replace()`: an undeclared key raises
    ValueError, it is not dropped."""
    cls = type(instance)
    names = {f.name for f in fields(cls)}

    def _is_init_field(name):
        if name not in names:
            raise ValueError(f"Field '{name}' not found in {cls.__name__}.")
        return True

    merged = {k: v for k, v in instance.__dict__.items() if _is_init_field(k)}
    merged.update(kwargs)
    return cls(**merged)


@dataclass
class _EngineArgs:
    """Only the bits the patch touches: a real dataclass field plus classmethods.

    ``repetition_detection`` is deliberately *not* a field -- that is exactly the
    situation on real EngineArgs, and the reason from_cli_args needs patching.
    """

    model: str = "dummy"

    @staticmethod
    def add_cli_args(parser):
        # Guarded so the "registered twice" test isolates the patch's own guard
        # instead of tripping over this stub's --model first.
        try:
            parser.add_argument("--model", default="dummy")
        except argparse.ArgumentError:
            pass
        return parser

    @classmethod
    def from_cli_args(cls, args):
        attrs = [f.name for f in fields(cls)]
        return cls(**{a: getattr(args, a) for a in attrs if hasattr(args, a)})

    def create_engine_config(self, usage_context=None, headless=False):
        cfg = _VllmConfig()
        cfg.seen_usage_context = usage_context
        cfg.seen_headless = headless
        return cfg


class _InputProcessor:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def process_inputs(self, request_id, prompt, params, **kwargs):
        # 0.25.1 clones the params (input_processor.py:314); the clone is what
        # reaches the engine core, so the test asserts on it.
        if isinstance(params, _SamplingParams):
            clone = _SamplingParams(params.repetition_detection)
        else:
            clone = params
        return {"request_id": request_id, "params": clone, "kwargs": kwargs}


class _VLLMPatch:
    """Faithful mini-version of vllm_patches/core.py::VLLMPatch.apply()."""

    _attr_names_to_apply = []

    @classmethod
    def apply(cls):
        target = cls._target
        if not hasattr(target, "_omni_npu_applied_patches"):
            target._omni_npu_applied_patches = {}
        for name in cls._attr_names_to_apply:
            if name not in cls.__dict__:
                raise ValueError(f"cannot find {name} in PatchClass {cls.__name__}")
            if name in target._omni_npu_applied_patches:
                raise ValueError(f"{target}.{name} already patched")
            target._omni_npu_applied_patches[name] = cls.__name__
            setattr(target, name, cls.__dict__[name])


def _register_patch(name, target):
    def decorator(cls):
        cls._target = target
        # vllm 0.25.1 A-tier fix: _register_patch dedup — _load_patch() is invoked twice
        # (module-level line 238 + module fixture line 530); dedupe by patch name
        # so REGISTERED doesn't double-fill and trip apply() 'already patched' guard.
        if not any(existing_name == name for existing_name, _, _ in REGISTERED):
            REGISTERED.append((name, target, cls))
        return cls

    return decorator


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _Envs:
    OMNI_REPETITION_DETECTION_CONFIG = None


ENVS = _Envs()


def _install_stubs():
    for pkg in (
        "vllm",
        "vllm.v1",
        "vllm.v1.engine",
        "omni_npu",
        "omni_npu.vllm_patches",
    ):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))

    sys.modules["vllm"].EngineArgs = _EngineArgs
    _module("vllm.config", VllmConfig=_VllmConfig)
    _module(
        "vllm.sampling_params",
        RepetitionDetectionParams=_RepetitionDetectionParams,
        SamplingParams=_SamplingParams,
        PoolingParams=_PoolingParams,
    )
    _module("vllm.logger", init_logger=lambda name: logging.getLogger(name))
    _module("vllm.v1.engine.input_processor", InputProcessor=_InputProcessor)
    sys.modules["omni_npu"].envs = ENVS
    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=_VLLMPatch,
        register_patch=_register_patch,
    )


def _load_patch():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("patch_rep_det", PATCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["patch_rep_det"] = mod
    spec.loader.exec_module(mod)
    return mod


_STUBBED_MODULE_NAMES = (
    "vllm",
    "vllm.config",
    "vllm.sampling_params",
    "vllm.logger",
    "vllm.v1",
    "vllm.v1.engine",
    "vllm.v1.engine.input_processor",
    "omni_npu",
    "omni_npu.vllm_patches",
    "omni_npu.vllm_patches.core",
)
_MISSING = object()
_saved_modules = {
    name: sys.modules.get(name, _MISSING) for name in _STUBBED_MODULE_NAMES
}
_saved_vllm_engine_args = getattr(
    sys.modules.get("vllm"), "EngineArgs", _MISSING
)
_saved_omni_envs = getattr(sys.modules.get("omni_npu"), "envs", _MISSING)

PATCH = _load_patch()
PATCH = None  # _load_patch() moved into _restore_sys_modules fixture to avoid collection-time pollution

for _name, _module_value in _saved_modules.items():
    if _module_value is _MISSING:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module_value

_vllm_module = sys.modules.get("vllm")
if _vllm_module is not None:
    if _saved_vllm_engine_args is _MISSING:
        delattr(_vllm_module, "EngineArgs")
    else:
        _vllm_module.EngineArgs = _saved_vllm_engine_args

_omni_module = sys.modules.get("omni_npu")
if _omni_module is not None:
    if _saved_omni_envs is _MISSING:
        delattr(_omni_module, "envs")
    else:
        _omni_module.envs = _saved_omni_envs


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

GOOD_JSON = '{"max_pattern_size":10,"min_pattern_size":2,"min_count":3}'


def _reset():
    """Undo per-process caches so each test starts from a clean slate."""
    PATCH._PROCESS_DEFAULT = None
    PATCH._env_default = PATCH._ENV_DEFAULT_UNRESOLVED
    ENVS.OMNI_REPETITION_DETECTION_CONFIG = None
    # vllm 0.25.1 A-tier fix: clear stub VLLMPatch.apply()'s 'already patched' tracking so re-apply works across tests
    for _, target, cls in REGISTERED:
        if hasattr(target, "_omni_npu_applied_patches"):
            target._omni_npu_applied_patches.clear()
    # vllm 0.25.1 A-tier fix: clear stub VLLMPatch.apply()'s 'already patched' tracking so re-apply works across tests
    for _, target, cls in REGISTERED:
        if hasattr(target, "_omni_npu_applied_patches"):
            target._omni_npu_applied_patches.clear()


def _parser():
    parser = argparse.ArgumentParser()
    return PATCH.EngineArgsRepetitionDetectionPatch.add_cli_args(parser)


def _engine_args_from(argv):
    args = _parser().parse_args(argv)
    return PATCH.EngineArgsRepetitionDetectionPatch.from_cli_args.__func__(
        _EngineArgs, args
    )


def _config_from(argv):
    ea = _engine_args_from(argv)
    return PATCH.EngineArgsRepetitionDetectionPatch.create_engine_config(ea)


def _process(processor, params):
    proc = PATCH.InputProcessorRepetitionDetectionPatch.process_inputs
    return proc(processor, "req-0", "hello", params, supported_tasks=("generate",))


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_every_declared_attr_exists():
    """VLLMPatch.apply() raises if a name in _attr_names_to_apply is missing."""
    for name, target, cls in REGISTERED:
        for attr in cls._attr_names_to_apply:
            assert attr in cls.__dict__, f"{cls.__name__} declares missing {attr!r}"
        assert cls._target is target


def test_patch_targets():
    names = {name: target for name, target, _ in REGISTERED}
    assert names["OmniRepetitionDetectionVllmConfigPatch"] is _VllmConfig
    assert names["OmniRepetitionDetectionEngineArgsPatch"] is _EngineArgs
    assert names["OmniRepetitionDetectionInputProcessorPatch"] is _InputProcessor


def test_cli_flag_parses_json():
    _reset()
    ea = _engine_args_from(["--repetition-detection", GOOD_JSON])
    assert ea.repetition_detection == _RepetitionDetectionParams(10, 2, 3)


def test_cli_absent_leaves_it_off():
    _reset()
    _config_from([])
    assert PATCH._PROCESS_DEFAULT is None


def test_create_engine_config_carries_cli_value_and_forwards_args():
    _reset()
    cfg = _config_from(["--repetition-detection", GOOD_JSON])
    assert PATCH._PROCESS_DEFAULT == _RepetitionDetectionParams(10, 2, 3)
    assert "repetition_detection" not in cfg.__dict__
    # the original create_engine_config still received its own arguments
    assert cfg.seen_usage_context is None and cfg.seen_headless is False


def test_env_fallback_when_cli_absent():
    _reset()
    ENVS.OMNI_REPETITION_DETECTION_CONFIG = '{"max_pattern_size":4,"min_count":2}'
    _config_from([])
    assert PATCH._PROCESS_DEFAULT == _RepetitionDetectionParams(4, 0, 2)


def test_env_beats_cli():
    """Same order as the 0.14 backport: a valid env value overwrites the CLI."""
    _reset()
    ENVS.OMNI_REPETITION_DETECTION_CONFIG = '{"max_pattern_size":4,"min_count":2}'
    _config_from(["--repetition-detection", GOOD_JSON])
    assert PATCH._PROCESS_DEFAULT == _RepetitionDetectionParams(4, 0, 2)


def test_bad_env_leaves_the_cli_value_in_place():
    """A malformed env value must not silently disable a working CLI config."""
    _reset()
    ENVS.OMNI_REPETITION_DETECTION_CONFIG = "{not json"
    _config_from(["--repetition-detection", GOOD_JSON])
    assert PATCH._PROCESS_DEFAULT == _RepetitionDetectionParams(10, 2, 3)


def test_bad_cli_json_fails_the_launch():
    _reset()
    try:
        _parser().parse_args(["--repetition-detection", "{not json"])
    except SystemExit:
        pass  # argparse turns the ValueError from type= into exit(2)
    else:
        raise AssertionError("a malformed --repetition-detection must not be ignored")


def test_cli_rejects_semantically_invalid_config():
    """min_count < 2 with detection enabled is caught by __post_init__."""
    _reset()
    try:
        _parser().parse_args(
            ["--repetition-detection", '{"max_pattern_size":5,"min_count":1}']
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("invalid min_count must not be accepted")


def test_bad_env_json_is_logged_and_ignored():
    _reset()
    ENVS.OMNI_REPETITION_DETECTION_CONFIG = "{not json"
    _config_from([])
    assert PATCH._PROCESS_DEFAULT is None  # no raise: node still starts


def test_flag_registered_twice_does_not_abort():
    _reset()
    parser = _parser()
    PATCH.EngineArgsRepetitionDetectionPatch.add_cli_args(parser)
    assert parser.parse_args(["--repetition-detection", GOOD_JSON])


def test_injection_fills_in_the_server_default():
    _reset()
    cfg = _config_from(["--repetition-detection", GOOD_JSON])
    out = _process(_InputProcessor(cfg), _SamplingParams())
    assert out["params"].repetition_detection == _RepetitionDetectionParams(10, 2, 3)


def test_request_value_wins_over_server_default():
    _reset()
    cfg = _config_from(["--repetition-detection", GOOD_JSON])
    asked = _RepetitionDetectionParams(6, 1, 5)
    out = _process(_InputProcessor(cfg), _SamplingParams(asked))
    assert out["params"].repetition_detection == asked


def test_no_default_configured_is_a_no_op():
    _reset()
    cfg = _config_from([])
    out = _process(_InputProcessor(cfg), _SamplingParams())
    assert out["params"].repetition_detection is None


def test_injection_survives_a_config_that_lost_the_attribute():
    """dataclasses.replace()/reconstruction elsewhere would drop the plain attr."""
    _reset()
    _config_from(["--repetition-detection", GOOD_JSON])  # populates _PROCESS_DEFAULT
    out = _process(_InputProcessor(_VllmConfig()), _SamplingParams())
    assert out["params"].repetition_detection == _RepetitionDetectionParams(10, 2, 3)


def test_config_stays_reconstructible_for_spec_decode():
    """The default must not be parked on the instance: 0.25.1's spec-decode path
    rebuilds the config through replace(), which raises on undeclared keys and
    took down every MTP worker at load_model."""
    _reset()
    cfg = _config_from(["--repetition-detection", GOOD_JSON])
    assert "repetition_detection" not in cfg.__dict__

    draft = _vllm_replace(cfg)  # must not raise

    out = _process(_InputProcessor(draft), _SamplingParams())
    assert out["params"].repetition_detection == _RepetitionDetectionParams(10, 2, 3)


def test_replace_stub_rejects_unknown_keys():
    """Guard for the test above: if the stub silently dropped extra keys instead
    of raising, that regression test would pass no matter what the patch does."""
    cfg = _VllmConfig()
    cfg.injected_by_someone = 1
    try:
        _vllm_replace(cfg)
    except ValueError as exc:
        assert "injected_by_someone" in str(exc)
    else:
        raise AssertionError("the replace() stub no longer models 0.25.1")


def test_env_only_process_still_injects():
    """A process that never ran create_engine_config falls back to the env var."""
    _reset()
    ENVS.OMNI_REPETITION_DETECTION_CONFIG = GOOD_JSON
    out = _process(_InputProcessor(_VllmConfig()), _SamplingParams())
    assert out["params"].repetition_detection == _RepetitionDetectionParams(10, 2, 3)


def test_pooling_params_are_left_alone():
    _reset()
    cfg = _config_from(["--repetition-detection", GOOD_JSON])
    pooling = _PoolingParams()
    out = _process(_InputProcessor(cfg), pooling)
    assert out["params"] is pooling
    assert not hasattr(pooling, "repetition_detection")


def test_original_process_inputs_still_receives_everything():
    _reset()
    cfg = _config_from([])
    out = _process(_InputProcessor(cfg), _SamplingParams())
    assert out["request_id"] == "req-0"
    assert out["kwargs"]["supported_tasks"] == ("generate",)


def test_apply_actually_installs_the_attributes():
    """End-to-end through the real apply() semantics from core.py."""
    _reset()
    for _, _, cls in REGISTERED:
        cls.apply()
    assert _VllmConfig.repetition_detection is None
    assert _EngineArgs.repetition_detection is None
    # add_cli_args resolves to the patched staticmethod, and the patched
    # EngineArgs.from_cli_args copies the flag across.
    args = _EngineArgs.add_cli_args(argparse.ArgumentParser()).parse_args(
        ["--repetition-detection", GOOD_JSON]
    )
    ea = _EngineArgs.from_cli_args(args)
    assert ea.repetition_detection == _RepetitionDetectionParams(10, 2, 3)
    vllm_config = ea.create_engine_config()
    assert PATCH._PROCESS_DEFAULT == _RepetitionDetectionParams(10, 2, 3)
    assert "repetition_detection" not in vllm_config.__dict__
    out = _InputProcessor(vllm_config).process_inputs(
        "req-1", "hi", _SamplingParams(), supported_tasks=("generate",)
    )
    assert out["params"].repetition_detection == _RepetitionDetectionParams(10, 2, 3)

try:
    import pytest
except ImportError:
    pytest = None

@pytest.fixture(scope="module", autouse=True)
def _restore_sys_modules():
    # snapshot sys.modules *before* _load_patch() pollutes them
    keys = (
        "vllm", "vllm.v1", "vllm.v1.engine", "vllm.config",
        "vllm.sampling_params", "vllm.logger",
        "vllm.v1.engine.input_processor",
        "omni_npu", "omni_npu.vllm_patches", "omni_npu.vllm_patches.core",
        "patch_rep_det",
    )
    saved_modules = {k: sys.modules.get(k) for k in keys}
    vllm_now = sys.modules.get("vllm")
    omni_now = sys.modules.get("omni_npu")
    saved_vllm_EngineArgs = (hasattr(vllm_now, "EngineArgs"),
                             getattr(vllm_now, "EngineArgs", None)) if vllm_now else (False, None)
    saved_omni_envs = (hasattr(omni_now, "envs"),
                       getattr(omni_now, "envs", None)) if omni_now else (False, None)
    global PATCH
    PATCH = _load_patch()
    yield
    # teardown: restore from snapshot taken before _load_patch()
    for k, v in saved_modules.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    vllm_now = sys.modules.get("vllm")
    if vllm_now is not None:
        had, val = saved_vllm_EngineArgs
        if had:
            vllm_now.EngineArgs = val
        else:
            try: delattr(vllm_now, "EngineArgs")
            except AttributeError: pass
    omni_now = sys.modules.get("omni_npu")
    if omni_now is not None:
        had, val = saved_omni_envs
        if had:
            omni_now.envs = val
        else:
            try: delattr(omni_now, "envs")
            except AttributeError: pass

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
