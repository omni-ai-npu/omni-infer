# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""OMNI-CONF runtime collector/emitter.

Emits a startup configuration snapshot as atomic single-line records:

    [OMNI-CONF:w77] #begin scope=worker rank=77 ... schema=1
    [OMNI-CONF:w77] vllm.parallel.tensor_parallel_size=1
    ...
    [OMNI-CONF:w77] #end n=187 config_hash=sha256:ab12cd34

Emission point (solution doc §3.3):
  Worker scope: NPUWorker.init_device after load_model_extra_config.

HARD CONSTRAINTS:
  * stdlib-only module-level imports (vllm / torch only inside guarded calls);
  * a config dump must NEVER break serving: every public entry point is fully
    wrapped, failure degrades to one WARNING line;
  * declared-field projection, never ``dataclasses.asdict`` on VllmConfig
    (asdict deepcopies non-dataclass leaves -> model layers / tensors).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import socket
import time

from omni_npu import envs
from omni_npu.diagnostics.config_summary import classification as cls

logger = logging.getLogger("omni_npu.diagnostics.config_summary")

_MAX_DEPTH = 32
_REPR_LIMIT = 256

# In-process "emit once" guard. This is per-process BY DESIGN, not a global
# cross-process lock: every worker process is *meant* to emit its own block
# (local_rank 0 full, others hash-only - decided at the call site via the
# hash_only flag, see npu_worker.py). This set only stops the SAME process from
# emitting the SAME scope twice (e.g. a retried init_device). The guard_key
# embeds os.getpid() on purpose so a forked child still emits its own block
# instead of inheriting (copy-on-write) the parent's already-seen keys and
# going silent. Cross-worker de-dup is therefore intentionally NOT this guard's
# job and must not be "fixed" with a file lock / multiprocessing shared set.
_EMITTED_SCOPES: set[str] = set()


def is_enabled() -> bool:
    return envs.OMNI_CONFIG_SUMMARY


# --------------------------------------------------------------------------
# value walker (no deepcopy, projection boundary enforced by callers)
# --------------------------------------------------------------------------

def _adapt_config_object(obj):
    """Adapter chain for known config-carrier objects (and only those)."""
    for attr in ("to_dict", "model_dump"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:  # noqa: BLE001 - adapter chain falls through
                pass
    try:
        return dict(vars(obj))
    except TypeError:
        pass
    # __slots__ / no-__dict__ objects: project the declared slots that are set.
    # Without this, vars() raises TypeError and any caller using it directly
    # would lose the whole section (collect_vllm fallback / npu_compilation).
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        if isinstance(slots, str):
            slots = (slots,)
        got = {s: getattr(obj, s) for s in slots if hasattr(obj, s)}
        if got:
            return got
    return None


def _walk(out: dict, segments: list, value, depth: int = 0, seen: set | None = None):
    """Flatten *value* into out[encoded-path] = json-able scalar."""
    if seen is None:
        seen = set()
    path = cls.encode_segments(segments)
    if depth > _MAX_DEPTH:
        out[path] = "…trunc(depth)"
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        out[path] = value
        return
    # enums -> name
    name = getattr(value, "name", None)
    if name is not None and value.__class__.__module__ != "builtins" and hasattr(
            value.__class__, "__members__"):
        out[path] = name
        return
    # cycle guard: `seen` tracks ONLY the current recursion PATH (add before we
    # descend, discard in the `finally` blocks below) — it is NOT a global
    # visited-set. Any oid present in `seen` therefore belongs to an object the
    # live call stack still references, so it cannot be GC'd and its id() cannot
    # be reused while flagged. A plain id() key is correct here precisely because
    # of this scoping; a diamond (shared, non-cyclic ref) is walked twice, not
    # mistaken for a cycle.
    oid = id(value)
    if oid in seen:
        out[path] = "…trunc(cycle)"
        return
    if isinstance(value, dict):
        if not value:
            out[path] = {}
            return
        seen.add(oid)
        try:
            for k, v in value.items():
                _walk(out, segments + [k if isinstance(k, (str, int)) else str(k)],
                      v, depth + 1, seen)
        finally:
            seen.discard(oid)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            out[path] = []
            return
        seen.add(oid)
        try:
            for i, v in enumerate(value):
                _walk(out, segments + [i], v, depth + 1, seen)
        finally:
            seen.discard(oid)
        return
    if isinstance(value, (set, frozenset)):
        try:
            items = sorted(value, key=repr)
        except Exception:  # noqa: BLE001
            items = list(value)
        _walk(out, segments, items, depth, seen)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(oid)
        try:
            for f in dataclasses.fields(value):
                try:
                    _walk(out, segments + [f.name], getattr(value, f.name, None),
                          depth + 1, seen)
                except Exception as e:  # noqa: BLE001 - per-field isolation
                    out[cls.encode_segments(segments + [f.name])] = f"…error({e.__class__.__name__})"
        finally:
            seen.discard(oid)
        return
    # known config carriers get the adapter chain via field name (last segment)
    if segments and str(segments[-1]) in cls.CONFIG_OBJECT_FIELDS:
        adapted = _adapt_config_object(value)
        if adapted is not None:
            _walk(out, segments, adapted, depth, seen)
            return
    out[path] = repr(value)[:_REPR_LIMIT]


# --------------------------------------------------------------------------
# section collectors
# --------------------------------------------------------------------------

def _project_subconfig(out: dict, ns_segments: list, sub) -> None:
    """Declared-field projection with the rev6 three-state init=False rule."""
    cname = type(sub).__name__
    for f in dataclasses.fields(sub):
        fq = f"{cname}.{f.name}"
        if cname == "ModelConfig" and f.name in cls.MODEL_CONFIG_HF_SKIP:
            continue  # projected via the dedicated model.hf.* channel
        if not f.init:
            if fq in cls.RUNTIME_FIELD_EXCLUDE or fq in cls.SPECIAL_FIELD_HANDLED:
                continue
            if fq not in cls.DERIVED_FIELD_INCLUDE:
                logger.warning(
                    "OMNI-CONF: unclassified init=False field %s - skipped from "
                    "snapshot; add it to DERIVED_FIELD_INCLUDE or "
                    "RUNTIME_FIELD_EXCLUDE (schema audit will fail until then)",
                    fq,
                )
                continue
        try:
            _walk(out, ns_segments + [f.name], getattr(sub, f.name, None))
        except Exception as e:  # noqa: BLE001
            out[cls.encode_segments(ns_segments + [f.name])] = f"…error({e.__class__.__name__})"


def collect_vllm(vllm_config) -> dict:
    out: dict = {}
    if vllm_config is None:
        return out
    if dataclasses.is_dataclass(vllm_config):
        top_fields = [(f.name, getattr(vllm_config, f.name, None))
                      for f in dataclasses.fields(vllm_config)]
    else:  # duck-typed fallback (tests, exotic configs)
        top_fields = [(k, v) for k, v in (_adapt_config_object(vllm_config) or {}).items()
                      if not k.startswith("_")]
    for name, sub in top_fields:
        ns = name[:-len("_config")] if name.endswith("_config") else name
        if sub is None:
            out[cls.encode_segments(["vllm", ns])] = None
        elif dataclasses.is_dataclass(sub) and not isinstance(sub, type):
            _project_subconfig(out, ["vllm", ns], sub)
        else:
            _walk(out, ["vllm", ns], sub)
    # NPU compilation config is attached dynamically by NPUPlatform
    npu_cc = getattr(vllm_config, "npu_compilation_config", None)
    if npu_cc is not None and "vllm.npu_compilation" not in out:
        if dataclasses.is_dataclass(npu_cc) and not isinstance(npu_cc, type):
            _project_subconfig(out, ["vllm", "npu_compilation"], npu_cc)
        else:
            adapted = _adapt_config_object(npu_cc)
            _walk(out, ["vllm", "npu_compilation"],
                  adapted if adapted is not None else repr(npu_cc)[:_REPR_LIMIT])
    return out


def collect_model_hf(vllm_config) -> dict:
    out: dict = {}
    model_config = getattr(vllm_config, "model_config", None)
    hf = getattr(model_config, "hf_config", None)
    if hf is None:
        return out
    adapted = _adapt_config_object(hf)
    if adapted is None:
        out["model.hf"] = repr(hf)[:_REPR_LIMIT]
        return out
    _walk(out, ["model", "hf"], adapted)
    return out


def collect_env() -> dict:
    out: dict = {}
    collected = {k: v for k, v in os.environ.items() if cls.env_is_collected(k)}
    # vllm-declared envs that are explicitly set (best effort, runtime only)
    try:
        from vllm import envs as vllm_envs  # noqa: PLC0415
        for name in getattr(vllm_envs, "environment_variables", {}):
            if name in os.environ:
                collected.setdefault(name, os.environ[name])
    except Exception:  # noqa: BLE001 - not available offline / in tests
        pass
    for k in sorted(collected):
        out[cls.encode_segments(["env", k])] = collected[k]
    return out


def collect_omni() -> dict:
    """ModelExtraConfig (worker side only; loaded state, pure dataclass tree)."""
    out: dict = {}
    try:
        from omni_npu.model_config.config_loader import loader  # noqa: PLC0415
        mec = getattr(loader, "model_extra_config", None)
        if mec is not None:
            _walk(out, ["model", "omni"], mec)
    except Exception:  # noqa: BLE001
        logger.warning("OMNI-CONF: model.omni section unavailable", exc_info=True)
    return out


def collect_meta(scope: str, rank=None, local_rank=None, dp_rank=None) -> dict:
    meta = {
        "meta.schema": cls.SCHEMA_VERSION,
        "meta.scope": scope,
        "meta.ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "meta.pid": os.getpid(),
        "meta.hostname": socket.gethostname(),
        "meta.role": _detect_role(),
    }
    if rank is not None:
        meta["meta.rank"] = rank
    if local_rank is not None:
        meta["meta.local_rank"] = local_rank
    if dp_rank is not None:
        meta["meta.dp_rank"] = dp_rank
    for mod, key in (("vllm", "meta.vllm_version"), ("omni_npu", "meta.omni_npu_version")):
        try:
            import importlib.metadata as _md  # noqa: PLC0415
            meta[key] = _md.version(mod.replace("_", "-"))
        except Exception:  # noqa: BLE001
            pass
    return meta


def _detect_role() -> str:
    role = envs.OMNI_PD_ROLE
    if role:
        return role if role in ("prefill", "decode") else f"other:{role}"
    return "hybrid"


# --------------------------------------------------------------------------
# canonicalisation / hash / emit
# --------------------------------------------------------------------------

def _render_value(path: str, value) -> str:
    if cls.is_sensitive(path):
        value = cls.MASK
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return json.dumps(repr(value)[:_REPR_LIMIT], ensure_ascii=False)


def canonical_lines(entries: dict) -> list[str]:
    return [f"{p}={_render_value(p, entries[p])}" for p in sorted(entries)]


def compute_hash(entries: dict) -> str:
    shared = [f"{p}={_render_value(p, entries[p])}"
              for p in sorted(entries)
              if cls.classify_key(p) == cls.CLASS_SHARED]
    h = hashlib.sha256()
    h.update(f"schema={cls.SCHEMA_VERSION}\n".encode())
    h.update("\n".join(shared).encode())
    return f"sha256:{h.hexdigest()[:16]}"


def build_entries(vllm_config=None, scope: str = "engine", rank=None,
                  local_rank=None, dp_rank=None, include_omni: bool = False) -> dict:
    entries: dict = {}
    entries.update(collect_meta(scope, rank=rank, local_rank=local_rank, dp_rank=dp_rank))
    entries.update(collect_env())
    entries.update(collect_vllm(vllm_config))
    entries.update(collect_model_hf(vllm_config))
    if include_omni:
        entries.update(collect_omni())
    return entries


def emit_config_summary(vllm_config=None, scope: str = "engine", rank=None,
                        local_rank=None, dp_rank=None, include_omni: bool = False,
                        hash_only: bool = False, log=None) -> bool:
    """Emit one OMNI-CONF block. Returns True when something was emitted.

    Never raises: any internal failure degrades to a single WARNING.
    """
    out = log or logger
    try:
        if not is_enabled():
            return False
        token = f"{'w' if scope == 'worker' else 'e'}{rank if rank is not None else 0}"
        guard_key = f"{scope}:{token}:{os.getpid()}"
        if guard_key in _EMITTED_SCOPES:
            return False
        _EMITTED_SCOPES.add(guard_key)

        entries = build_entries(vllm_config, scope=scope, rank=rank,
                                local_rank=local_rank, dp_rank=dp_rank,
                                include_omni=include_omni)
        cfg_hash = compute_hash(entries)
        marker = f"[OMNI-CONF:{token}]"
        role = entries.get("meta.role", "hybrid")

        if hash_only:
            out.info("%s #hashonly scope=%s rank=%s role=%s schema=%s config_hash=%s",
                     marker, scope, rank, role, cls.SCHEMA_VERSION, cfg_hash)
            return True

        lines = canonical_lines(entries)
        out.info("%s #begin scope=%s rank=%s role=%s schema=%s vllm=%s omni_npu=%s",
                 marker, scope, rank, role, cls.SCHEMA_VERSION,
                 entries.get("meta.vllm_version", "?"),
                 entries.get("meta.omni_npu_version", "?"))
        for line in lines:
            out.info("%s %s", marker, line)
        out.info("%s #end n=%d config_hash=%s", marker, len(lines), cfg_hash)
        return True
    except Exception:  # noqa: BLE001 - dump must never break serving
        out.warning("OMNI-CONF: config summary emission failed", exc_info=True)
        return False


def reset_once_guard() -> None:
    """Test hook."""
    _EMITTED_SCOPES.clear()
