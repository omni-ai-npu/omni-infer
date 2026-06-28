# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# SPDX-License-Identifier: MIT
"""Unit tests for the OMNI-CONF runtime collector (synthetic configs, no NPU).

Class names below intentionally mirror real vLLM config class names so the
three-state init=False projection tables key correctly.
"""

import dataclasses
import enum
import logging
import sys
import types
from dataclasses import dataclass, field

import pytest

from omni_npu.diagnostics import classification as cls
from omni_npu.diagnostics import config_summary as summ


_CONFIG_SUMMARY_LOGGER = "omni_npu.diagnostics.config_summary"


class _Poison:
    """Sentinel: ANY interaction must crash the test if the walker touches it."""

    def __getattr__(self, name):  # pragma: no cover - must never run
        raise AssertionError("walker touched runtime state!")

    def __repr__(self):  # pragma: no cover - must never run
        raise AssertionError("walker repr'd runtime state!")


@dataclass
class CompilationConfig:
    mode: int = 3
    backend: str = "eager"
    cudagraph_capture_sizes: list = field(default_factory=lambda: [16])
    # runtime fields (RUNTIME_FIELD_EXCLUDE)
    static_forward_context: dict = field(default_factory=dict, init=False)
    compilation_time: float = field(default=0.0, init=False)
    # deployment vLLM 0.14.0+empty field (real-NPU finding): post_init-derived
    # cudagraph padding table, RUNTIME_FIELD_EXCLUDE
    bs_to_padded_graph_size: list = field(default_factory=list, init=False)


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    data_parallel_size: int = 32
    data_parallel_rank: int = 7
    enable_expert_parallel: bool = True
    # per-worker / per-process / per-node seat numbers (init=True, IDENTITY_KEYS)
    rank: int = 0
    _api_process_rank: int = 0
    node_rank: int = 0
    world_size: int = field(default=32, init=False)        # DERIVED_FIELD_INCLUDE
    totally_new_runtime_thing: int = field(default=0, init=False)  # unclassified!


@dataclass
class SchedulerConfig:
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 128


class _FakeHF:
    """PretrainedConfig-like: not a dataclass, exposes to_dict()."""

    def __init__(self):
        self.model_type = "openpangu_v2"
        self.num_key_value_heads = 8
        self.hidden_size = 7168

    def to_dict(self):
        return dict(vars(self))


@dataclass
class ModelConfig:
    model: str = "/mnt/model/path"
    tokenizer_mode: str = "auto"
    hf_config: object = field(default_factory=_FakeHF)


@dataclass
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    speculative_config: object = None


@pytest.fixture(autouse=True)
def _reset_guard():
    summ.reset_once_guard()
    yield
    summ.reset_once_guard()


@pytest.fixture()
def caplog_config_summary(caplog):
    """Capture config_summary logs even when omni_npu propagation is disabled."""
    logger = logging.getLogger(_CONFIG_SUMMARY_LOGGER)
    added_handler = caplog.handler not in logger.handlers
    if added_handler:
        logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=_CONFIG_SUMMARY_LOGGER):
            yield caplog
    finally:
        if added_handler:
            logger.removeHandler(caplog.handler)


@pytest.fixture()
def cfg():
    c = VllmConfig()
    c.compilation_config.static_forward_context["layer.0"] = _Poison()
    return c


class TestProjection:
    def test_runtime_fields_never_walked(self, cfg):
        entries = summ.build_entries(cfg, scope="engine")
        assert not any("static_forward_context" in k for k in entries)
        assert not any("compilation_time" in k for k in entries)

    def test_derived_config_included(self, cfg):
        entries = summ.build_entries(cfg, scope="engine")
        assert entries["vllm.parallel.world_size"] == 32

    def test_bs_to_padded_graph_size_excluded_no_warning(
            self, cfg, caplog_config_summary):
        """Real-NPU finding: deployment vLLM exposes
        CompilationConfig.bs_to_padded_graph_size (init=False, post_init-derived
        cudagraph padding table). Must be excluded WITHOUT an unclassified
        warning (17 such warnings appeared in the real prefill log)."""
        entries = summ.build_entries(cfg, scope="engine")
        assert not any("bs_to_padded_graph_size" in k for k in entries)
        assert not any("bs_to_padded_graph_size" in r.message
                       for r in caplog_config_summary.records)

    def test_rank_seat_numbers_do_not_enter_hash(self, cfg):
        """Real-NPU finding: w0-w15 (same prefill role) had 16 distinct hashes
        solely because vllm.parallel.rank 0..15 leaked into the shared hash.
        rank / _api_process_rank / node_rank are seat numbers, not config."""
        base = summ.build_entries(cfg, scope="worker", rank=0, local_rank=0)
        base["meta.ts"] = "T"
        hashes = set()
        for r in range(16):
            e = dict(base)
            e["vllm.parallel.rank"] = r
            e["vllm.parallel._api_process_rank"] = r
            e["vllm.parallel.node_rank"] = r // 8
            e["meta.ts"] = "T"
            hashes.add(summ.compute_hash(e))
        assert len(hashes) == 1, "rank seat numbers must not change the hash"
        # but a genuine config change still must
        drift = dict(base)
        drift["vllm.scheduler.max_num_batched_tokens"] = 99999
        assert summ.compute_hash(drift) != summ.compute_hash(base)

    def test_rank_fields_are_identity(self):
        for k in ("vllm.parallel.rank", "vllm.parallel._api_process_rank",
                  "vllm.parallel.node_rank"):
            assert cls.classify_key(k) == cls.CLASS_IDENTITY, k

    def test_unclassified_init_false_warned_and_skipped(
            self, cfg, caplog_config_summary):
        entries = summ.build_entries(cfg, scope="engine")
        assert not any("totally_new_runtime_thing" in k for k in entries)
        assert any("unclassified init=False" in r.message
                   for r in caplog_config_summary.records)

    def test_no_underscore_wildcard_exemption(self, caplog_config_summary):
        """impl-review round-2 P2 minimal repro: a future `_foo` init=False
        field must trip the unclassified warning - never a silent skip via
        name-based wildcard."""
        @dataclass
        class FutureConfig:
            normal: int = 1
            _future_config_toggle: bool = field(default=False, init=False)

        @dataclass
        class Cfg:
            future_config: FutureConfig = field(default_factory=FutureConfig)

        summ.reset_once_guard()
        entries = summ.build_entries(Cfg(), scope="engine")
        assert not any("_future_config_toggle" in k for k in entries)
        assert any("FutureConfig._future_config_toggle" in r.message
                   for r in caplog_config_summary.records)

    def test_hf_expanded_not_repr(self, cfg):
        entries = summ.build_entries(cfg, scope="engine")
        assert entries["model.hf.num_key_value_heads"] == 8
        assert entries["model.hf.model_type"] == "openpangu_v2"
        # never a repr-degraded blob
        assert "model.hf" not in entries

    def test_named_configs_present(self, cfg):
        entries = summ.build_entries(cfg, scope="engine")
        assert entries["vllm.scheduler.enable_chunked_prefill"] is True
        assert entries["vllm.scheduler.max_num_batched_tokens"] == 128
        assert entries["vllm.parallel.tensor_parallel_size"] == 1
        assert entries["vllm.speculative"] is None

    def test_audit_all_init_false_classified_in_fakes(self):
        """Mini schema audit over the synthetic classes (real one runs on CI
        with vllm importable; see test_schema_audit below)."""
        unclassified = []
        for klass in (CompilationConfig, ParallelConfig):
            for f in dataclasses.fields(klass):
                if f.init:
                    continue
                fq = f"{klass.__name__}.{f.name}"
                if fq not in cls.DERIVED_FIELD_INCLUDE and fq not in cls.RUNTIME_FIELD_EXCLUDE:
                    unclassified.append(fq)
        assert unclassified == ["ParallelConfig.totally_new_runtime_thing"]


class TestMaskAndHash:
    def test_env_credentials_masked(self, cfg, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
        entries = summ.build_entries(cfg, scope="engine")
        joined = "\n".join(summ.canonical_lines(entries))
        assert "sk-secret" not in joined
        assert 'env.OPENAI_API_KEY="***"' in joined
        assert 'env.ENABLE_OMNI_CACHE="1"' in joined  # real config survives

    def test_render_masks_credentials_at_value_layer(self):
        # codex round-5 leak pair: masked regardless of collection source
        # (locally vllm.envs is unavailable; on NPU/CI these ARE collected)
        assert summ._render_value("env.S3_SECRET_ACCESS_KEY", "sssh") == '"***"'
        assert summ._render_value("env.S3_ACCESS_KEY_ID", "AKIA") == '"***"'
        assert summ._render_value("env.VLLM_API_KEY", "k") == '"***"'

    def test_requirement_targets_never_masked(self, cfg):
        lines = "\n".join(summ.canonical_lines(summ.build_entries(cfg, scope="engine")))
        assert "model.hf.num_key_value_heads=8" in lines
        assert "vllm.scheduler.max_num_batched_tokens=128" in lines
        assert 'vllm.model.tokenizer_mode="auto"' in lines

    def test_hash_ignores_identity_but_not_shared(self, cfg):
        e1 = summ.build_entries(cfg, scope="worker", rank=0, local_rank=0)
        e2 = summ.build_entries(cfg, scope="worker", rank=7, local_rank=7)
        # strip volatile meta noise that differs run-to-run (ts/pid identical here)
        for e in (e1, e2):
            e["meta.ts"] = "T"
        assert summ.compute_hash(e1) == summ.compute_hash(e2)
        cfg2 = VllmConfig()
        cfg2.scheduler_config.max_num_batched_tokens = 256
        e3 = summ.build_entries(cfg2, scope="worker", rank=0, local_rank=0)
        e3["meta.ts"] = "T"
        assert summ.compute_hash(e3) != summ.compute_hash(e1)

    def test_data_parallel_rank_is_identity(self):
        assert cls.classify_key("vllm.parallel.data_parallel_rank") == cls.CLASS_IDENTITY


class _ListLogger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)

    def warning(self, *a, **k):
        self.lines.append(f"WARNING:{a}")


class TestEmit:
    def test_emit_format_and_once_guard(self, cfg):
        log = _ListLogger()
        assert summ.emit_config_summary(cfg, scope="worker", rank=0, local_rank=0,
                                        log=log) is True
        assert log.lines[0].startswith("[OMNI-CONF:w0] #begin scope=worker")
        assert log.lines[-1].startswith("[OMNI-CONF:w0] #end n=")
        n = int(log.lines[-1].split("n=")[1].split()[0])
        assert n == len(log.lines) - 2
        assert "config_hash=sha256:" in log.lines[-1]
        # every payload line is single, marker-prefixed, k=v
        for ln in log.lines[1:-1]:
            assert ln.startswith("[OMNI-CONF:w0] ") and "=" in ln
        # once-guard
        assert summ.emit_config_summary(cfg, scope="worker", rank=0, log=log) is False

    def test_hash_only_single_line(self, cfg):
        log = _ListLogger()
        assert summ.emit_config_summary(cfg, scope="worker", rank=3, local_rank=1,
                                        hash_only=True, log=log) is True
        assert len(log.lines) == 1
        assert "#hashonly" in log.lines[0] and "config_hash=sha256:" in log.lines[0]

    def test_disabled_by_env(self, cfg, monkeypatch):
        monkeypatch.setenv("OMNI_CONFIG_SUMMARY", "0")
        log = _ListLogger()
        assert summ.emit_config_summary(cfg, scope="engine", log=log) is False
        assert log.lines == []

    def test_never_crashes_on_poisoned_config(self):
        class Exploding:
            @property
            def model_config(self):
                raise RuntimeError("boom")

        log = _ListLogger()
        # must not raise, degrades to warning
        summ.emit_config_summary(Exploding(), scope="engine", log=log)


def test_ast_audit_omni_custom_configs():
    """AST-level init=False audit over omni-npu's OWN config sources
    (no imports needed -> always runs, even without torch). Catches e.g.
    the custom ReasoningConfig._tool_call_start_token_id (impl-review r3)."""
    import ast
    import pathlib

    src_root = pathlib.Path(__file__).parent.parent.parent / "src" / "omni_npu"
    unclassified = []
    for py in (src_root / "v1" / "config").glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not (isinstance(stmt, ast.AnnAssign) and stmt.value is not None
                        and isinstance(stmt.target, ast.Name)):
                    continue
                if "init=False" not in ast.unparse(stmt.value):
                    continue
                fq = f"{node.name}.{stmt.target.id}"
                if (fq not in cls.DERIVED_FIELD_INCLUDE
                        and fq not in cls.RUNTIME_FIELD_EXCLUDE
                        and fq not in cls.SPECIAL_FIELD_HANDLED):
                    unclassified.append(f"{fq} ({py.name})")
    assert not unclassified, (
        f"unclassified init=False fields in omni-npu configs: {unclassified}")


def test_schema_audit_real_vllm():
    """Full init=False audit over real vLLM configs (runs where vllm imports;
    auto-skips on boxes without vllm/torch - covered on NPU CI)."""
    pytest.importorskip("vllm.config")
    import vllm.config as vc

    unclassified = []
    for name in dir(vc):
        obj = getattr(vc, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        for f in dataclasses.fields(obj):
            if f.init:
                continue
            fq = f"{obj.__name__}.{f.name}"
            if (fq not in cls.DERIVED_FIELD_INCLUDE
                    and fq not in cls.RUNTIME_FIELD_EXCLUDE
                    and fq not in cls.SPECIAL_FIELD_HANDLED):
                unclassified.append(fq)
    assert not unclassified, (
        f"unclassified init=False fields (add to DERIVED_FIELD_INCLUDE or "
        f"RUNTIME_FIELD_EXCLUDE): {sorted(set(unclassified))}")


# ==========================================================================
# Walker internals. The module contract is "a config dump must NEVER break
# serving", so every degrade-not-crash branch of _walk has to be exercised
# explicitly (depth guard, cycle guard, enums, empty/odd containers, per-field
# isolation, repr fallback).
# ==========================================================================


class _RaisingField:
    """Data descriptor whose read always raises a non-AttributeError.

    Installed onto a dataclass *after* @dataclass has built __init__, so the
    field still appears in dataclasses.fields() but every attribute read blows
    up. Because it defines __set__ it is a data descriptor and wins over the
    instance __dict__, so the __init__ assignment is silently dropped and the
    getattr inside the walker/projector raises - exactly the per-field failure
    the try/except is meant to isolate.
    """

    def __set__(self, obj, value):  # pragma: no cover - assignment is a no-op
        pass

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        raise RuntimeError("field read boom")


class TestWalker:
    @staticmethod
    def _walk(value, segments=("x",)):
        out: dict = {}
        summ._walk(out, list(segments), value)
        return out

    def test_depth_guard_truncates(self):
        nested = 0
        for _ in range(summ._MAX_DEPTH + 5):
            nested = [nested]
        out = self._walk(nested)
        assert any(v == "…trunc(depth)" for v in out.values())

    def test_enum_rendered_as_name(self):
        class Color(enum.Enum):
            RED = 1
            BLUE = 2

        assert self._walk(Color.BLUE)["x"] == "BLUE"

    def test_cycle_guard_truncates_dict(self):
        d: dict = {}
        d["self"] = d
        assert any(v == "…trunc(cycle)" for v in self._walk(d).values())

    def test_cycle_guard_truncates_list(self):
        lst: list = []
        lst.append(lst)
        assert any(v == "…trunc(cycle)" for v in self._walk(lst).values())

    def test_empty_containers_preserved(self):
        assert self._walk({})["x"] == {}
        assert self._walk([])["x"] == []
        assert self._walk(())["x"] == []

    def test_set_is_sorted_deterministically(self):
        assert self._walk({3, 1, 2}) == {"x[0]": 1, "x[1]": 2, "x[2]": 3}

    def test_set_with_unsortable_elements_falls_back(self):
        # sorted(key=repr) raises when repr() itself raises -> the walker must
        # degrade to an unsorted list rather than propagating the error.
        @dataclass(frozen=True)
        class Hashable:
            v: int = 7

            def __repr__(self):
                raise RuntimeError("no repr")

        assert self._walk(frozenset({Hashable()})) == {"x[0].v": 7}

    def test_dataclass_walked_fieldwise(self):
        @dataclass
        class Inner:
            a: int = 1
            b: str = "z"

        assert self._walk(Inner()) == {"x.a": 1, "x.b": "z"}

    def test_dataclass_per_field_error_isolated(self):
        @dataclass
        class Tricky:
            good: int = 1
            bad: int = 0

        Tricky.bad = _RaisingField()
        out = self._walk(Tricky())
        assert out["x.good"] == 1
        assert out["x.bad"] == "…error(RuntimeError)"

    def test_config_carrier_expanded_via_field_name(self):
        # The adapter chain applies ONLY to fields named in CONFIG_OBJECT_FIELDS
        # (here: quantization_config), not to arbitrary objects.
        class Quant:
            def to_dict(self):
                return {"bits": 4}

        out = self._walk(Quant(), segments=("vllm", "quantization_config"))
        assert out["vllm.quantization_config.bits"] == 4

    def test_unknown_object_degrades_to_repr(self):
        assert self._walk(object())["x"].startswith("<object object at")

    def test_shared_ref_walked_twice_not_cycle(self):
        # review (omni_ci !1557): a diamond (same object referenced from two
        # siblings) is NOT a cycle. `seen` is path-scoped - the oid is added
        # before descending and discarded in the finally afterwards - so a
        # shared-but-acyclic ref is fully walked on BOTH paths. A global
        # visited-set keyed on id() would wrongly truncate the second path.
        shared = {"v": 1}
        out = self._walk({"a": shared, "b": shared})
        assert out["x.a.v"] == 1
        assert out["x.b.v"] == 1
        assert not any(val == "…trunc(cycle)" for val in out.values())


class TestAdaptConfigObject:
    def test_to_dict_failure_falls_back_to_vars(self):
        class C:
            def __init__(self):
                self.x = 5

            def to_dict(self):
                raise RuntimeError("boom")

        assert summ._adapt_config_object(C()) == {"x": 5}

    def test_model_dump_used_when_to_dict_absent(self):
        class C:
            def model_dump(self):
                return {"k": "v"}

        assert summ._adapt_config_object(C()) == {"k": "v"}

    def test_no_adapter_and_no_vars_returns_none(self):
        # object() has neither to_dict/model_dump nor a usable __dict__
        assert summ._adapt_config_object(object()) is None

    def test_slots_object_without_dict_projected(self):
        # review (omni_ci !1557): __slots__ objects have no __dict__, so
        # dict(vars(obj)) raises TypeError. The adapter must still project the
        # declared slots rather than giving up (and rather than letting the
        # TypeError escape to a caller that uses vars() directly).
        class Slotted:
            __slots__ = ("a", "b")

            def __init__(self):
                self.a = 1
                self.b = 2

        assert summ._adapt_config_object(Slotted()) == {"a": 1, "b": 2}

    def test_slots_object_with_unset_slot_skipped(self):
        # an unset slot has no value; project only the slots actually assigned.
        class Slotted:
            __slots__ = ("a", "b")

            def __init__(self):
                self.a = 1  # b left unset

        assert summ._adapt_config_object(Slotted()) == {"a": 1}


class TestCollectors:
    def test_collect_vllm_none_returns_empty(self):
        assert summ.collect_vllm(None) == {}

    def test_collect_vllm_duck_typed_non_dataclass_field(self):
        class Duck:
            def __init__(self):
                self.some_config = {"key": "val"}  # non-dataclass -> _walk path
                self._private = "skip me"

        out = summ.collect_vllm(Duck())
        assert out["vllm.some.key"] == "val"
        assert not any("_private" in k for k in out)  # underscore fields skipped

    def test_collect_vllm_npu_compilation_dataclass(self):
        @dataclass
        class NpuCC:
            enable_graph: bool = True

        cfg = VllmConfig()
        cfg.npu_compilation_config = NpuCC()  # attached dynamically by NPUPlatform
        out = summ.collect_vllm(cfg)
        assert out["vllm.npu_compilation.enable_graph"] is True

    def test_collect_vllm_npu_compilation_non_dataclass(self):
        class NpuCC:
            def __init__(self):
                self.level = 2

        cfg = VllmConfig()
        cfg.npu_compilation_config = NpuCC()
        out = summ.collect_vllm(cfg)
        assert out["vllm.npu_compilation.level"] == 2

    def test_collect_vllm_duck_typed_slots_no_dict(self):
        # review (omni_ci !1557): the duck-typed fallback (tests/exotic configs)
        # must not blow up on a __slots__ object with no __dict__. vars() would
        # raise TypeError and, uncaught at this site, degrade the WHOLE snapshot
        # to a single WARNING. Route through the slots-aware adapter instead.
        class SlotCfg:
            __slots__ = ("scheduler_config",)

            def __init__(self):
                self.scheduler_config = {"max_num_batched_tokens": 64}

        out = summ.collect_vllm(SlotCfg())
        assert out["vllm.scheduler.max_num_batched_tokens"] == 64

    def test_collect_vllm_npu_compilation_slots_no_dict(self):
        # review (omni_ci !1557): npu_compilation_config attached as a __slots__
        # object (no __dict__) must project its declared fields, not degrade to
        # an opaque repr blob.
        class NpuCC:
            __slots__ = ("level",)

            def __init__(self):
                self.level = 2

        cfg = VllmConfig()
        cfg.npu_compilation_config = NpuCC()
        out = summ.collect_vllm(cfg)
        assert out["vllm.npu_compilation.level"] == 2

    def test_project_subconfig_per_field_error_isolated(self):
        # declared-field projection must isolate a field whose read raises (a
        # property/descriptor blowing up mid-init) into an "…error(...)" marker
        # rather than aborting the whole sub-config projection.
        @dataclass
        class SubCfg:
            ok: int = 1
            bad: int = 0

        SubCfg.bad = _RaisingField()
        out: dict = {}
        summ._project_subconfig(out, ["vllm", "sub"], SubCfg())
        assert out["vllm.sub.ok"] == 1
        assert out["vllm.sub.bad"] == "…error(RuntimeError)"

    def test_collect_model_hf_unadaptable_degrades_to_repr(self):
        @dataclass
        class MC:
            hf_config: object = field(default_factory=object)

        @dataclass
        class Cfg:
            model_config: object = field(default_factory=MC)

        out = summ.collect_model_hf(Cfg())
        assert out["model.hf"].startswith("<object object at")

    def test_collect_env_includes_vllm_declared(self, monkeypatch):
        fake_vllm = types.ModuleType("vllm")
        fake_envs = types.ModuleType("vllm.envs")
        fake_envs.environment_variables = {"ZZ_DECLARED_BY_VLLM": None}
        fake_vllm.envs = fake_envs
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
        monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)
        monkeypatch.setenv("ZZ_DECLARED_BY_VLLM", "yes")
        # This name is NOT covered by the static whitelist, so its presence in
        # the output proves the runtime vllm.envs-declared branch actually ran.
        assert cls.env_is_collected("ZZ_DECLARED_BY_VLLM") is False
        out = summ.collect_env()
        assert out["env.ZZ_DECLARED_BY_VLLM"] == "yes"

    def test_collect_omni_walks_loaded_config(self, monkeypatch):
        @dataclass
        class MEC:
            dispatch: str = "all2all"

        fake_mod = types.ModuleType("omni_npu.model_config.config_loader")
        fake_mod.loader = types.SimpleNamespace(model_extra_config=MEC())
        monkeypatch.setitem(
            sys.modules, "omni_npu.model_config.config_loader", fake_mod)
        out = summ.collect_omni()
        assert out["model.omni.dispatch"] == "all2all"

    def test_collect_omni_missing_loader_degrades_to_warning(
            self, caplog_config_summary, monkeypatch):
        # collect_omni must degrade to a warning (never raise, never emit a
        # partial section) when the config_loader import fails. Force that
        # DETERMINISTICALLY by pinning the package entry in sys.modules to None
        # (CPython raises ModuleNotFoundError on a None entry, halting the import
        # at the package level before any submodule/attribute is resolved).
        #
        # NB: injecting a fake module WITHOUT a `loader` attribute is NOT enough.
        # On a real NPU worker `config_loader` is a PACKAGE and `loader` is a
        # SUBMODULE (config_loader/loader.py), so `from ...config_loader import
        # loader` bypasses the fake parent, imports the real loader.py submodule
        # from sys.modules, and silently succeeds -> collect_omni returns the
        # real model.omni.* tree and `assert out == {}` blows up (real-env CI).
        # The None sentinel is package/submodule/attribute-shape agnostic.
        monkeypatch.setitem(
            sys.modules, "omni_npu.model_config.config_loader", None)
        out = summ.collect_omni()
        assert out == {}
        assert any("model.omni section unavailable" in r.message
                   for r in caplog_config_summary.records)

    def test_collect_meta_includes_dp_rank_only_when_given(self):
        assert summ.collect_meta(
            "worker", rank=0, local_rank=0, dp_rank=3)["meta.dp_rank"] == 3
        assert "meta.dp_rank" not in summ.collect_meta("worker", rank=0)

    def test_detect_role_from_env(self, monkeypatch):
        monkeypatch.setenv("ROLE", "prefill")
        assert summ._detect_role() == "prefill"
        monkeypatch.setenv("ROLE", "decode")
        assert summ._detect_role() == "decode"
        monkeypatch.setenv("ROLE", "sidecar")  # unknown role -> namespaced
        assert summ._detect_role() == "other:sidecar"
        monkeypatch.delenv("ROLE", raising=False)
        assert summ._detect_role() == "hybrid"


class TestRenderAndBuild:
    def test_render_value_unserialisable_degrades_to_repr(self):
        # mixed int/str keys make json.dumps(sort_keys=True) raise TypeError; the
        # renderer must fall back to a repr string instead of propagating.
        rendered = summ._render_value("vllm.some.weird", {1: "a", "b": 2})
        assert rendered.startswith('"') and "1" in rendered

    def test_build_entries_include_omni_toggle(self, cfg, monkeypatch):
        @dataclass
        class MEC:
            attn: str = "mla"

        fake_mod = types.ModuleType("omni_npu.model_config.config_loader")
        fake_mod.loader = types.SimpleNamespace(model_extra_config=MEC())
        monkeypatch.setitem(
            sys.modules, "omni_npu.model_config.config_loader", fake_mod)
        with_omni = summ.build_entries(cfg, scope="worker", include_omni=True)
        assert with_omni["model.omni.attn"] == "mla"
        without = summ.build_entries(cfg, scope="worker", include_omni=False)
        assert not any(k.startswith("model.omni") for k in without)
