# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""OMNI-CONF schema constants: key classification, field projection tables,
sensitive-env masking and the dot-path codec.

This module is the single source of truth shared by the runtime emitter
(``omni_npu.diagnostics.config_summary.collector``).

HARD CONSTRAINT: stdlib-only imports. The offline CLI must run on maintenance
boxes without torch / torch_npu / vllm installed (covered by an import-isolation
test).

Classification semantics (solution doc §3.2, rev6):
  - shared   : real configuration semantics. Only shared keys enter the
               config_hash, and only shared diffs drive the default diff
               exit code.
  - identity : per-rank/per-process "seat number" (rank, visible devices...).
               Always dumped, never hashed, grouped separately in diff output.
  - volatile : per-deployment-instance noise (ports, lease-ish values).
               Always dumped, never hashed, grouped separately in diff output.
  - UNKNOWN KEYS DEFAULT TO shared: an unclassified naturally-varying key
    surfaces loudly as a (false) drift alarm and then gets classified
    explicitly - we never silently hide a real config difference.
"""

from __future__ import annotations

import json
import re

SCHEMA_VERSION = 1

CLASS_SHARED = "shared"
CLASS_IDENTITY = "identity"
CLASS_VOLATILE = "volatile"

# --------------------------------------------------------------------------
# Key classification (exact dot-paths; NO broad suffix regexes - they were
# proven to swallow real config such as ENABLE_OMNI_CACHE / KV_CACHE_MODE /
# CUSTOM_MODEL_CONFIG_PATH / HCCL_IF_BASE_PORT, codex review round-2).
# --------------------------------------------------------------------------

IDENTITY_KEYS = frozenset({
    "meta.ts",
    "meta.pid",
    "meta.rank",
    "meta.local_rank",
    "meta.dp_rank",
    "meta.hostname",
    "meta.world_ranks",
    "env.LOCAL_RANK",
    "env.RANK",
    "env.RANK_ID",
    "env.ASCEND_RT_VISIBLE_DEVICES",
    "env.NPU_VISIBLE_DEVICES",
    "env.ASCEND_VISIBLE_DEVICES",
    'env["NPU-VISIBLE-DEVICES"]',   # k8s-style hyphen twin (bootstrap dumps)
    "vllm.parallel.data_parallel_rank",
    "vllm.parallel.data_parallel_rank_local",
    "vllm.parallel.data_parallel_index",
    # per-worker / per-process / per-node seat numbers: NOT config, must not
    # enter the hash (real-NPU finding: w0-w15 same prefill role had 16 distinct
    # hashes solely because vllm.parallel.rank 0..15 leaked into shared).
    # node_rank / _api_process_rank were =0 in the single-node single-DP capture
    # but are identity by the same semantics and would break multi-node hashes.
    "vllm.parallel.rank",
    "vllm.parallel._api_process_rank",
    "vllm.parallel.node_rank",
    "vllm.kv_transfer.kv_rank",
    "vllm.kv_transfer.engine_id",
})

VOLATILE_KEYS = frozenset({
    "vllm.parallel.data_parallel_master_ip",
    "vllm.parallel.data_parallel_master_port",
    "vllm.parallel.data_parallel_address",
    "vllm.parallel.data_parallel_rpc_port",
    "vllm.kv_transfer.kv_ip",
    "vllm.kv_transfer.kv_port",
    "env.ASCEND_PROCESS_LOG_PATH",
    "env.ASCEND_WORK_PATH",
})


def classify_key(path: str) -> str:
    """Classify a canonical dot-path. Unknown keys default to shared."""
    if path in IDENTITY_KEYS:
        return CLASS_IDENTITY
    if path in VOLATILE_KEYS:
        return CLASS_VOLATILE
    return CLASS_SHARED


# --------------------------------------------------------------------------
# init=False field projection tables (solution doc §2, rev6 three-state rule).
# Keyed by "<ConfigClassName>.<field_name>".
#   - init=True fields are always projected.
#   - init=False fields MUST be listed in exactly one of the two tables below.
#   - An unlisted init=False field triggers a WARNING at emit time and fails
#     the schema audit test (never silently skipped, never silently walked).
# --------------------------------------------------------------------------

DERIVED_FIELD_INCLUDE = frozenset({
    # derived but genuine configuration - diff-relevant
    "ParallelConfig.world_size",
    "ParallelConfig.world_size_across_dp",
    "ParallelConfig.data_parallel_index",   # rank-like; key classified identity
    "DeviceConfig.device_type",
    "SchedulerConfig.max_num_encoder_input_tokens",
    "SchedulerConfig.encoder_cache_size",
    "CacheConfig.user_specified_block_size",
    "CacheConfig.user_specified_mamba_block_size",
})

# init=False fields that are neither derived-config nor runtime state because
# they are projected through a DEDICATED channel (model.hf.* adapter chain).
SPECIAL_FIELD_HANDLED = frozenset({
    "ModelConfig.hf_config",
    "ModelConfig.hf_text_config",
})

# NOTE (impl-review round-2 P2): NO wildcard exemptions. Private/underscore
# fields are listed EXPLICITLY by fully-qualified name below - a future
# `_foo` init=False field must trip the unclassified warning + schema audit,
# that is the whole point of the three-state invariant.

RUNTIME_FIELD_EXCLUDE = frozenset({
    # runtime state mutated after/around model load - NEVER walk these
    # (static_forward_context holds live model layer objects, vllm itself
    # excludes it from serialization: vllm/config/compilation.py:775/790)
    "CompilationConfig.static_forward_context",
    "CompilationConfig.enabled_custom_ops",
    "CompilationConfig.disabled_custom_ops",
    "CompilationConfig.traced_files",
    "CompilationConfig.compilation_time",
    "CompilationConfig.encoder_compilation_time",
    "CompilationConfig.static_all_moe_layers",
    "CompilationConfig.local_cache_dir",
    # post_init-derived lookup table (list[int] up to max_capture_size) mapping
    # batch size -> padded cudagraph size; deployment vLLM 0.14.0+empty exposes
    # it (upstream confirms init=False, derived from cudagraph_capture_sizes via
    # VllmConfig.__post_init__). Excluded as redundant: the SOURCE
    # cudagraph_capture_sizes is already captured, and dumping the full table
    # would add hundreds of derived lines. Same rationale as tokenizer token_ids.
    "CompilationConfig.bs_to_padded_graph_size",
    # runtime-backfilled KV sizing (core.py:273), deliberately outside the
    # snapshot contract (solution doc §3.3)
    "CacheConfig.num_gpu_blocks",
    "CacheConfig.num_cpu_blocks",
    "CacheConfig.num_gpu_blocks_override",
    # private resolution state, EXPLICIT fq-names (no underscore wildcard)
    "CacheConfig._block_size_resolved",
    "ReasoningConfig._enabled",
    # tokenizer-derived token IDs, set by initialize_token_ids at runtime:
    # excluding them keeps the hash free of tokenizer-init-phase drift noise.
    # Keyed by ClassName.field, so these cover BOTH vllm.config.ReasoningConfig
    # (reasoning.py:29-40) and the omni-npu custom ReasoningConfig
    # (src/omni_npu/v1/config/reasoning.py:36-47, which adds tool-call fields).
    "ReasoningConfig._reasoning_start_token_ids",
    "ReasoningConfig._reasoning_end_token_ids",
    "ReasoningConfig._tool_call_start_token_id",
    "ReasoningConfig._tool_call_end_token_id",
})

# Object-valued attributes that are *known config carriers*: the adapter chain
# (to_dict -> model_dump -> vars) applies ONLY to these, it is not an escape
# hatch for arbitrary runtime objects (codex round-4/5).
CONFIG_OBJECT_FIELDS = frozenset({
    "hf_config",
    "quantization_config",
})

# Attributes of ModelConfig that are huge HF objects handled separately via
# the model.hf.* namespace (or intentionally skipped duplicates).
MODEL_CONFIG_HF_SKIP = frozenset({
    "hf_config",
    "hf_text_config",
    "hf_image_processor_config",
})

# --------------------------------------------------------------------------
# Sensitive masking (solution doc §3.4, rev6).
# Masking applies ONLY to the env.* namespace, by FULL variable name.
# Other namespaces use the explicit SENSITIVE_PATHS table (initially empty).
# --------------------------------------------------------------------------

SENSITIVE_ENV_EXACT = frozenset({
    "S3_ACCESS_KEY_ID",        # present in vllm.envs (envs.py:29/676)
    "S3_SECRET_ACCESS_KEY",    # present in vllm.envs (envs.py:30/677)
    "VLLM_API_KEY",
    "OPENAI_API_KEY",
})

SENSITIVE_ENV_PATTERN = re.compile(
    r".*(_API_KEY|_ACCESS_KEY(_ID)?|_TOKEN|_SECRET[A-Z_]*|_PASSWORD|_PASSWD|_CREDENTIALS?)$"
    r"|.*CREDENTIAL.*"  # e.g. CREDENTIAL_PROFILES_FILE (points at secrets)
)

SENSITIVE_PATHS: frozenset[str] = frozenset()

MASK = "***"


def env_name_of(path: str) -> str | None:
    """Return env variable name for env-namespace paths.

    Handles both:
    - env.NAME
    - env["k8s-style-name"] produced by codec for non-identifier names
      (e.g. NPU-VISIBLE-DEVICES)
    """
    if not path.startswith("env"):
        return None
    try:
        segments = decode_path(path)
    except ValueError:
        return None
    if len(segments) == 2 and segments[0] == "env" and isinstance(segments[1], str):
        return segments[1]
    return None


def is_sensitive(path: str) -> bool:
    """Whether the value at this canonical path must be masked."""
    if path in SENSITIVE_PATHS:
        return True
    name = env_name_of(path)
    if name is not None:
        if name in SENSITIVE_ENV_EXACT:
            return True
        return SENSITIVE_ENV_PATTERN.fullmatch(name) is not None
    return False


# --------------------------------------------------------------------------
# Env collection whitelist (solution doc §4.2): prefix rules + the exact
# names actually consumed by omni-npu source (grep evidence in doc).
# --------------------------------------------------------------------------

ENV_PREFIX_WHITELIST = (
    "ASCEND_", "HCCL_", "NPU_", "OMNI_", "VLLM_", "RANK_TABLE",
    "TORCH_", "PYTORCH_", "RAY_", "GLOO_",
)

ENV_EXACT_WHITELIST = frozenset({
    "ROLE", "PREFILL_POD_NUM", "DECODE_POD_NUM", "LOCAL_RANK", "RANK",
    "RANK_ID", "KV_CACHE_MODE", "MOE_DISPATCH_COMBINE",
    "HYBRID_ATTN_GROUP_SIZE", "NUM_DIE_PER_MACH", "ENABLE_OMNI_CACHE",
    "CUSTOM_MODEL_CONFIG_PATH", "PREFILL_PROCESS", "USE_HA", "HA_SERVER_IP",
    "HA_PORT", "SYNC_KV_TIMEOUT", "LINK_TIMEOUT", "BLOCK_RELEASE_DELAY",
    "MAX_DISPATCH_COMBINE_THRESHOLD", "FROZEN_PARAMETER_DISABLED",
    "TOKENIZERS_PARALLELISM", "CAPTURE_MODE", "REPLAY_MODE", "RANDOM_MODE",
    "OPENAI_API_KEY",  # consumed by omni-npu source; collected AND masked
})


def env_is_collected(name: str) -> bool:
    if name in ENV_EXACT_WHITELIST:
        return True
    return any(name.startswith(p) for p in ENV_PREFIX_WHITELIST)


# --------------------------------------------------------------------------
# Dot-path codec (solution doc §2, rev4 encoding):
#   - identifier-safe string segments are written bare, joined by '.'
#   - any other string segment is written as a JSON-string bracket segment,
#     e.g. ["self_attn.o_proj"]  (JSON escaping handles ] " \ etc.)
#   - integer array indices are unquoted brackets: [0]
#   Quoted bracket = string key, unquoted bracket = array index: no ambiguity.
# --------------------------------------------------------------------------

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def encode_segments(segments: list) -> str:
    parts: list[str] = []
    for seg in segments:
        if isinstance(seg, int):
            parts.append(f"[{seg}]")
        elif isinstance(seg, str) and _IDENT_RE.match(seg):
            parts.append(f".{seg}" if parts else seg)
        else:
            parts.append(f"[{json.dumps(str(seg), ensure_ascii=False)}]")
    return "".join(parts)


def decode_path(path: str) -> list:
    """Inverse of :func:`encode_segments`. Raises ValueError on malformed input."""
    segments: list = []
    i, n = 0, len(path)
    while i < n:
        ch = path[i]
        if ch == ".":
            # '.' is a separator: the encoder emits it ONLY between segments and
            # always before a bare identifier (bracket segments attach with no
            # leading dot). Reject leading / trailing / doubled dots so malformed
            # input fails loudly instead of silently dropping an empty segment.
            nxt = path[i + 1] if i + 1 < n else ""
            if not segments or not (nxt.isalpha() or nxt == "_"):
                raise ValueError(f"unexpected '.' separator in {path!r}")
            i += 1
            continue
        if ch == "[":
            if i + 1 < n and path[i + 1] == '"':
                # JSON string segment: find its true end via raw_decode
                decoder = json.JSONDecoder()
                try:
                    value, end = decoder.raw_decode(path, i + 1)
                except ValueError as exc:  # json.JSONDecodeError is a ValueError
                    raise ValueError(
                        f"invalid JSON string segment in {path!r}") from exc
                if end >= n or path[end] != "]":
                    raise ValueError(f"unterminated string segment in {path!r}")
                segments.append(value)
                i = end + 1
            else:
                try:
                    j = path.index("]", i)
                except ValueError:
                    raise ValueError(
                        f"unterminated array index in {path!r}") from None
                idx_str = path[i + 1:j]
                try:
                    segments.append(int(idx_str))
                except ValueError as exc:
                    raise ValueError(
                        f"invalid array index {idx_str!r} in {path!r}") from exc
                i = j + 1
            continue
        # bare identifier
        if not (ch.isalpha() or ch == "_"):
            raise ValueError(f"malformed path at offset {i}: {path!r}")
        j = i
        while j < n and (path[j].isalnum() or path[j] == "_"):
            j += 1
        segments.append(path[i:j])
        i = j
    return segments
