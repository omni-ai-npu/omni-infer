# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Central registry for omni-npu environment variables.

Adapted from vllm-project/vllm/vllm/envs.py:
  - module-level ``env_variables`` lambdas with lazy ``__getattr__`` access;
  - ``# begin/end-env-vars-definition`` markers for documentation generation;
  - ``get_env_with_fallback`` for new name -> legacy name -> default lookup,
    with a deprecation warning whenever a legacy name is used.

Each variable's inline comment documents its *real* behavior:
value domain, default semantics, trigger conditions, invariants,
and the main consumer. Cross-references point at the source of truth.
"""
import logging
import os
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Static declarations for type checkers and IDE completion. Runtime values are
# resolved lazily by ``__getattr__``. These annotations do NOT affect runtime
# behavior — the actual parsing happens in the lambdas below.
if TYPE_CHECKING:
    # PD disaggregation
    OMNI_PD_ROLE: Optional[str]
    OMNI_PD_PREFILL_POD_NUM: int
    OMNI_PD_DECODE_POD_NUM: int
    # Cache
    OMNI_ENABLE_OMNI_CACHE: bool
    # Mock-compatible device topology
    OMNI_NO_NPU_MOCK: bool
    # Profiler
    OMNI_TRACE_OUTPUT_DIRECTORY: Optional[str]
    OMNI_PROFILE_TOKEN_THRESHOLD: Optional[int]
    OMNI_PROFILE_STOP_STEP: int
    OMNI_ENABLE_PREFILL_PROFILER: bool
    OMNI_PROFILE_SKIP_REQUESTS: int
    # Patch / Attention
    OMNI_DISABLE_PLUGIN_BACKENDS: str
    OMNI_HYBRID_ATTN_GROUP_SIZE: int
    OMNI_REPETITION_DETECTION_CONFIG: Optional[str]
    OMNI_REASONING_CONFIG: Optional[str]
    OMNI_STRUCTURED_OUTPUT_CONFIG: Optional[str]
    OMNI_PANGU_TOOL_CALL_ENDS_THINKING: bool
    OMNI_PANGU_V2_HIGH_THROUGHOUT: bool
    OMNI_ENABLE_MAX_TOKENS_EXCLUDE_REASONING: bool
    # MoE / sampling
    OMNI_MAX_DISPATCH_COMBINE_THRESHOLD: int
    OMNI_BEST_EP: bool
    OMNI_NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT: bool
    # Diagnostics
    OMNI_DUMP_ENABLE: bool
    OMNI_DUMP_DIR: str
    OMNI_HEALTH_HANG_SEC: int
    OMNI_METRICS_KV_TRANSFER_SELFTEST: bool
    OMNI_METRICS_WORKER_MEM_EVERY: int
    OMNI_KV_DUMP_PATH: str
    OMNI_CUSTOM_MODEL_CONFIG_PATH: Optional[str]
    # Existing OMNI_ names
    OMNI_VLLM_PATCHES: str
    OMNI_VLLM_PATCHES_DIR: str
    OMNI_NPU_PENALTY_CACHE: bool
    OMNI_REUSE_PREFILLED_TOKENS: bool
    OMNI_SKIP_DECODE_TOKENIZE: bool
    OMNI_CONFIG_SUMMARY: bool
    OMNI_LMHEAD_USE_DEVICE_COMM_A2A: bool
    OMNI_PIGGYBACK_INPUT_IDS: bool
    OMNI_VALIDATE_PIGGYBACK_INPUT_IDS: bool
    OMNI_HYBRID_ALIGNED_DECODE: bool
    OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD: int
    OMNI_DP_ROUND_ROBIN: bool
    OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD: int


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true")


def _as_int(raw: str) -> int:
    return int(raw)


def _as_exact_one(raw: str) -> bool:
    """Preserve flags whose existing public contract accepts only ``"1"``."""
    return raw == "1"


def _as_bool_or_all(raw: str) -> bool:
    """Like ``_as_bool`` but also treats ``"all"`` as True (benchmark gates)."""
    return raw.strip().lower() in ("1", "true", "all")


def _as_int_or_default(raw: str, default: int, name: str) -> int:
    """Like ``_as_int`` but tolerant: invalid input falls back to ``default``
    with a warning instead of raising. Used by benchmark thresholds where a
    typo should not abort startup."""
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid value %r for environment variable %r; falling back to %d.",
            raw,
            name,
            default,
        )
        return default


def get_env_with_fallback(
    new_name: str,
    old_names: Optional[List[str]],
    default: Any,
    parser: Optional[Callable[[str], Any]] = None,
) -> Any:
    """Resolve a new name, then its legacy aliases, then the default.

    Read order:
        1. New name set -> use the new value.
        2. Else any legacy name set -> warn about deprecation and use it.
        3. Neither set -> return the default.
    """
    raw = os.environ.get(new_name)
    if raw is not None:
        return parser(raw) if parser is not None else raw

    if old_names:
        for old in old_names:
            raw = os.environ.get(old)
            if raw is not None:
                logger.warning(
                    "Environment variable %r is deprecated, use %r instead. "
                    "(value taken from %r)", old, new_name, old,
                )
                return parser(raw) if parser is not None else raw

    return default


# The begin-* and end-* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition
env_variables: Dict[str, Callable[[], Any]] = {
    # =========================================================================
    # PD disaggregation (Prefill/Decode separation)
    # =========================================================================
    #
    # Master switch for the entire PD-disaggregation code path.
    # Empty/None means hybrid mode (single instance does both prefill+decode).
    # Supported deployment values: "prefill" | "decode" | unset/None.
    # The loader does not reject other non-empty strings: only the exact value
    # "prefill" selects a prefill node; every other non-empty value is treated
    # as decode, and the role/kv_role validator skips unknown values.
    #
    # Cross-source invariant (validators.role_kv_role_consistent, REJECT):
    #   "prefill" must pair with kv_transfer_config.kv_role="kv_producer"
    #   "decode"  must pair with kv_transfer_config.kv_role="kv_consumer"
    #
    # Consumers: model_config/config_loader/loader.py:42-44 (decides
    # is_pd_disaggregation / is_prefill_node -> pd_scheme and P/D config file
    # selection); validators.py:71 (kv_role cross-check); config_summary.py:294
    # (meta.role); patch_trace.py (Role:<role>_<ip> tag, fallback "unknown_role");
    # patch_routed_experts.py and patch_prefilled_token_skip_tokenize.py
    # (fallback when kv_transfer_config is unavailable).
    "OMNI_PD_ROLE":
    lambda: get_env_with_fallback("OMNI_PD_ROLE", ["ROLE"], None),

    # Number of prefill Pods in the deployment. Used ONLY to compose the
    # pd_scheme string "{P}P{D}D" (e.g. "1P1D", "4P1D") that selects which
    # best-practice JSON gets loaded from
    # model_config/configs/<performance_mode>/best_practice_configs.json.
    # Not consumed by runtime scheduling logic. A valid integer that composes
    # an unregistered scheme selects the default tuning profile with a warning;
    # a non-integer value fails _as_int while resolving the environment variable
    # and can abort startup.
    # Consumer: loader.py:79 -> task_config.prefill_node_num -> pd_scheme.
    "OMNI_PD_PREFILL_POD_NUM":
    lambda: get_env_with_fallback("OMNI_PD_PREFILL_POD_NUM", ["PREFILL_POD_NUM"], 1, _as_int),

    # Symmetric to OMNI_PD_PREFILL_POD_NUM, fills the "{D}D" part of pd_scheme.
    # Consumer: loader.py:80 -> task_config.decode_node_num.
    "OMNI_PD_DECODE_POD_NUM":
    lambda: get_env_with_fallback("OMNI_PD_DECODE_POD_NUM", ["DECODE_POD_NUM"], 1, _as_int),

    # =========================================================================
    # Cache
    # =========================================================================
    #
    # Enables Omni Cache (KV cache reuse in PD deployments; also switches the
    # MLA head_size on kv_consumer decode nodes to indexer_head_size for
    # memory savings).
    #
    # Cross-source invariant (validators.omni_cache_consistent, REJECT):
    #   when additional_config explicitly contains "enable_omni_cache", it
    #   must equal this env var. Environment-only legacy launches are accepted.
    #
    # Effects (in priority order):
    #   1. loader.ModelOperatorOptConfig.__post_init__:
    #      forces operator_opt_config.use_omni_cache=True regardless of the
    #      JSON best-practice value (env wins over JSON).
    #   2. loader.ModelOperatorOptConfig class default: changes
    #      moe_seq_split_length (True -> 128*10=1280, False -> 10**9).
    #   3. worker.NPUModelRunner.get_kv_cache_spec: when True AND
    #      kv_role=="kv_consumer", uses indexer_head_size for the MLA KV cache
    #      spec.
    #   4. validators.omni_cache_consistent: cross-source consistency REJECT
    #      check.
    "OMNI_ENABLE_OMNI_CACHE":
    lambda: get_env_with_fallback("OMNI_ENABLE_OMNI_CACHE", ["ENABLE_OMNI_CACHE"], False, _as_bool),

    # =========================================================================
    # Mock-compatible device topology
    # =========================================================================

    # In no-NPU mock launches, derive local_size from
    # ASCEND_RT_VISIBLE_DEVICES instead of torch.npu.device_count().
    # Consumer: parallel_state_ext.py:151-153.
    "OMNI_NO_NPU_MOCK":
    lambda: get_env_with_fallback("OMNI_NO_NPU_MOCK", ["NO_NPU_MOCK"], False, _as_bool),

    # =========================================================================
    # Profiler (two INDEPENDENT mechanisms)
    #   A) worker-side NPU profiler:
    #      - manual: unset PROFILER_TOKEN_THRESHOLD, use /start_profile API
    #      - auto: set PROFILER_TOKEN_THRESHOLD + other OMNI env vars below
    #   B) omni-trace dynamic wrapper, enabled by
    #      OMNI_TRACE_OUTPUT_DIRECTORY and independent of (A).
    # =========================================================================

    # Enables the usefull_patch omni-trace path and selects the directory for
    # per-process trace logs. Unset/None preserves its disabled-by-default
    # behavior; patch_trace also treats an empty or whitespace-only value as
    # disabled. Consumers capture the value when patch_trace.py and
    # omni_trace.utils are imported, so later process-environment changes are
    # not observed by those consumers.
    "OMNI_TRACE_OUTPUT_DIRECTORY":
    lambda: get_env_with_fallback(
        "OMNI_TRACE_OUTPUT_DIRECTORY", None, None),

    # Mechanism A (auto mode only): token-count trigger threshold. Non-None
    # enables auto profiling AND disables vLLM's manual profile() API.
    # Prefill batches trigger only when prefill profiling is enabled and their
    # token count exceeds the threshold; decode batches trigger when their token
    # count reaches the threshold (npu_worker.py execute_model).
    # Note: value 0 makes decode trigger impossible (num_tokens is never 0
    # on a real decode step), so use >=1.
    # Requires --profiler-config.profiler=torch.
    "OMNI_PROFILE_TOKEN_THRESHOLD":
    lambda: get_env_with_fallback(
        "OMNI_PROFILE_TOKEN_THRESHOLD", ["PROFILER_TOKEN_THRESHOLD"], None, _as_int),

    # Mechanism A: after the profiler starts, stop it once profile_step
    # exceeds this many execute_model calls. Marks profile_finished=True so
    # the profiler never restarts in this worker.
    # Consumers: npu_worker.py:379-381,391,408.
    "OMNI_PROFILE_STOP_STEP":
    lambda: get_env_with_fallback("OMNI_PROFILE_STOP_STEP", ["PROFILER_STOP_STEP"], 5, _as_int),

    # Mechanism A: also trigger on prefill batches (default decode-only).
    # Prefill batches are long and produce huge traces; decode-only is the
    # practical default.
    # Consumers: npu_worker.py:339-343,409.
    "OMNI_ENABLE_PREFILL_PROFILER":
    lambda: get_env_with_fallback("OMNI_ENABLE_PREFILL_PROFILER", ["ENABLE_PREFILL_PROFILER"], False, _as_bool),

    # Mechanism A: skip this many *new requests* before allowing the trigger.
    # Used to skip cold-start (weight load / graph capture boundary steps).
    # Counter increments by len(scheduled_new_reqs) per step.
    # Consumers: npu_worker.py:326-338,410.
    "OMNI_PROFILE_SKIP_REQUESTS":
    lambda: get_env_with_fallback("OMNI_PROFILE_SKIP_REQUESTS", ["PROFILER_SKIP_REQUESTS"], 0, _as_int),

    # =========================================================================
    # Patch / Attention
    # =========================================================================

    # Comma-separated blacklist of attention backend names that must NOT be
    # replaced by plugin implementations, even when a plugin is registered.
    # Exact-match after strip (case-sensitive). Example: "NPUDSA,NPUMLA".
    # Empty string means no blacklist.
    # Consumers: attention/backends/utils.py:160-177 (_is_plugin_disabled).
    "OMNI_DISABLE_PLUGIN_BACKENDS":
    lambda: get_env_with_fallback("OMNI_DISABLE_PLUGIN_BACKENDS", ["DISABLE_PLUGIN_BACKENDS"], ""),

    # Override for the KV-cache group size in pangu_v2_hybrid models. The
    # default heuristic is min(num_layers_per_type), with a 1.25x bump to
    # max when types are similar in size. Setting this >0 forces the value
    # and logs a warning. Only effective when OMNI_VLLM_PATCHES_DIR loads
    # pangu_v2_hybrid. Larger group_size -> fewer groups but more padding
    # layers -> wasted KV memory; smaller -> more groups -> finer scheduling.
    # Consumers: patch_kv_cache_utils.py:58-75.
    "OMNI_HYBRID_ATTN_GROUP_SIZE":
    lambda: get_env_with_fallback("OMNI_HYBRID_ATTN_GROUP_SIZE", ["HYBRID_ATTN_GROUP_SIZE"], 0, _as_int),

    # Env-var fallback for the --repetition-detection CLI flag. Value must be
    # the same JSON string the CLI accepts (e.g.
    # '{"max_pattern_size":10,"min_pattern_size":2,"min_count":3}').
    # Priority: request body > env > CLI > disabled. A valid env value overwrites
    # the CLI value. JSON parse failure logs an error and leaves the CLI value in
    # place, or leaves the feature disabled when no CLI value was supplied (no
    # raise -- a bad env var must not take down a node that was otherwise
    # launched correctly, whereas a bad CLI value does fail the launch).
    # Consumers: usefull_patch/common/patch_repetition_detection_config.py,
    # patches/common/patch_user_repetition_detection.py:131-155 (superseded).
    "OMNI_REPETITION_DETECTION_CONFIG":
    lambda: get_env_with_fallback(
        "OMNI_REPETITION_DETECTION_CONFIG", ["REPETITION_DETECTION_CONFIG"], None),

    # Env-var fallback for the --reasoning-config CLI flag. Value must be the
    # same JSON string the CLI accepts (see v1/config/reasoning.py:
    # ReasoningConfig fields reasoning_start_str / reasoning_end_str /
    # thinking_token_budget / ban_tool_start_in_thinking /
    # ban_tool_end_in_thinking).
    # Priority: env > CLI > disabled. A malformed env value logs an error and
    # leaves the CLI value in place.
    # Consumers: patch_thinking_limit.py:78-102.
    "OMNI_REASONING_CONFIG":
    lambda: get_env_with_fallback("OMNI_REASONING_CONFIG", ["REASONING_CONFIG"], None),

    # Env-var fallback for the --structured-output-config CLI flag. Value
    # must be the same JSON string the CLI accepts (guided decoding schema).
    # Priority: env > CLI > disabled. A malformed env value logs an error and
    # leaves the CLI value in place.
    # Consumers: patch_vllm_structured_output.py:641-667.
    "OMNI_STRUCTURED_OUTPUT_CONFIG":
    lambda: get_env_with_fallback(
        "OMNI_STRUCTURED_OUTPUT_CONFIG", ["STRUCTURED_OUTPUT_CONFIG"], None),

    # Treats the Pangu tool-call-start marker ("<|tool_call_start|>" or
    # "[unused11]" as fallback) as an implicit end of the thinking block,
    # even when </think> has not yet been emitted. Default False matches the
    # pre-9c14d17e behavior (marker ignored). Only affects the
    # PanguReasoningParser.
    # Consumers: v1/parsers/pangu_reasoning_parser.py:54,79-91,113-125.
    "OMNI_PANGU_TOOL_CALL_ENDS_THINKING":
    lambda: get_env_with_fallback(
        "OMNI_PANGU_TOOL_CALL_ENDS_THINKING",
        ["PANGU_TOOL_CALL_ENDS_THINKING"],
        False,
        _as_bool,
    ),

    # Pangu V2 high-throughout attention backend.
    # Default False keeps NPUPanguSparseAttention (low latency).
    # Set to 1/true to build DeepSeek DSA (index_topk / dsa_layers) or MLA.
    # Consumers: v1/models/pangu/pangu_v2_moe.py.
    "OMNI_PANGU_V2_HIGH_THROUGHOUT":
    lambda: get_env_with_fallback(
        "OMNI_PANGU_V2_HIGH_THROUGHOUT", None, False, _as_bool),

    # When enabled, ``max_tokens`` limits only the content portion of the
    # output (not the reasoning/thinking portion). Exact ``"1"`` enables the
    # behavior to preserve the pre-OMNI contract of
    # ENABLE_MAX_TOKENS_EXCLUDE_REASONING; other values leave the default
    # (total-output) max_tokens accounting unchanged.
    # Consumers: usefull_patch/models/pangu_v2_hybrid/patch_scheduler.py.
    "OMNI_ENABLE_MAX_TOKENS_EXCLUDE_REASONING":
    lambda: get_env_with_fallback(
        "OMNI_ENABLE_MAX_TOKENS_EXCLUDE_REASONING",
        ["ENABLE_MAX_TOKENS_EXCLUDE_REASONING"],
        False,
        _as_exact_one,
    ),

    # =========================================================================
    # MoE / sampling
    # =========================================================================

    # Token-count threshold for choosing the MoE communication strategy when
    # operator_opt_config.decode_moe_dispatch_combine is enabled. The exact
    # high-token strategy depends on the device and parallel configuration;
    # values at or below the threshold select "dispatch_combine".
    # Startup bounds follow the implementation's device checks:
    #   Ascend910B (A2)                        -> threshold <= 256
    #   devices other than Ascend910B/Ascend950 -> threshold <= 512
    #   Ascend950 (A5)                         -> no upper-bound assertion here
    # Default 64 keeps the assertion satisfied when the env is unset.
    # Consumers: layers/fused_moe/prepare_permute_unpermute_finalize.py:762-838.
    "OMNI_MAX_DISPATCH_COMBINE_THRESHOLD":
    lambda: get_env_with_fallback(
        "OMNI_MAX_DISPATCH_COMBINE_THRESHOLD", ["MAX_DISPATCH_COMBINE_THRESHOLD"], 64, _as_int),

    # Switches the EPLB strategy from planner.plan() (default) to
    # planner.apply_best_load_balance() (an alternative heuristic). Only
    # effective when parallel_config.enable_eplb is True (--enable-eplb CLI);
    # otherwise this flag has no effect.
    # Consumers: layers/quantization/compressed_tensors/compressed_tensors_moe.py:65,316-328.
    "OMNI_BEST_EP":
    lambda: get_env_with_fallback("OMNI_BEST_EP", ["BEST_EP"], False, _as_bool),

    # CANN 8.5.1 compatibility path for npu_top_k_top_p_sample: normalize
    # float32 logits and cast logits/top_p to model dtype before invoking the
    # NPU operator. Exact "1" enables the workaround; other values leave the
    # normal float32 path unchanged. The unprefixed name is retained only as
    # a fallback for deployments that adopted the initial implementation.
    # Consumer: sample/rejection_sampler.py:NPURejectionSampler.__init__.
    "OMNI_NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT":
    lambda: get_env_with_fallback(
        "OMNI_NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT",
        ["NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT"],
        False,
        _as_exact_one,
    ),

    # =========================================================================
    # Diagnostics
    # =========================================================================

    # Registers OMNI-DUMP hooks for API, engine, and worker processes and
    # permits each role hook to install exit forensics. Unset defaults to
    # enabled; exact "1" enables and every other explicit value disables,
    # preserving the original gate semantics.
    # Consumers: diagnostics/dump/hooks.py and patch_dump.py.
    "OMNI_DUMP_ENABLE":
    lambda: get_env_with_fallback(
        "OMNI_DUMP_ENABLE", None, True, _as_exact_one),

    # Output root for OMNI-DUMP artifacts. Relative paths resolve from the
    # process working directory; the default is the production log location.
    # Consumer: diagnostics/dump/hooks.py:default_dump_dir.
    "OMNI_DUMP_DIR":
    lambda: get_env_with_fallback(
        "OMNI_DUMP_DIR", None, "/var/log/omni-npu/dump"),

    # AsyncLLM health watchdog threshold in seconds. It is read on every
    # health check, so runtime environment changes are observable. Invalid
    # integer text raises ValueError when the health check resolves the value.
    # Consumer: patch_health.py:_hang_config.
    "OMNI_HEALTH_HANG_SEC":
    lambda: get_env_with_fallback(
        "OMNI_HEALTH_HANG_SEC", None, 240, _as_int),

    # Smoke-test-only injection switch: exact "1" records one synthetic
    # decode-side KV-transfer failure when the metrics module is initialized.
    # Consumer: diagnostics/metrics/kv_transfer.py:maybe_selftest.
    "OMNI_METRICS_KV_TRANSFER_SELFTEST":
    lambda: get_env_with_fallback(
        "OMNI_METRICS_KV_TRANSFER_SELFTEST",
        None,
        False,
        _as_exact_one,
    ),

    # Number of worker metrics collection calls between NPU allocator samples.
    # The value is captured when worker_mem is imported, matching the existing
    # sampling cadence behavior. Invalid integer text fails module import.
    # Consumer: diagnostics/metrics/worker_mem.py:maybe_sample.
    "OMNI_METRICS_WORKER_MEM_EVERY":
    lambda: get_env_with_fallback(
        "OMNI_METRICS_WORKER_MEM_EVERY", None, 50, _as_int),

    # Enables KV-block hash dumping for cross-node consistency debugging.
    # Non-empty enables the maybe_dump_kv wrapper on NPUModelRunner.execute_model
    # and selects the output directory; a per-run subdirectory
    # "<path>/<timestamp>/rank<N>.json" is created. Prefill nodes record hashes;
    # decode nodes verify them (mismatch raises ValueError).
    # Empty string disables the feature with zero overhead.
    # Consumers: connector/kv_dump.py:22,287-319,406-421.
    "OMNI_KV_DUMP_PATH":
    lambda: get_env_with_fallback("OMNI_KV_DUMP_PATH", ["KV_DUMP_PATH"], ""),

    # Bypasses the entire best-practice matching pipeline and loads this JSON
    # directly as the model_extra_config. A relative path is resolved under
    # src/omni_npu/model_config/configs/ (e.g. "custom/my_tuning.json");
    # an absolute path is accepted as-is by os.path.join.
    # Highest priority: when set, task_config (PD role / low_latency / etc.)
    # has no effect on JSON selection.
    # CUSTOM_MODEL_CONFIG_PATH remains a compatibility alias during the
    # transition to OMNI_CUSTOM_MODEL_CONFIG_PATH.
    # Consumers: model_config/config_loader/loader.py:337-342.
    "OMNI_CUSTOM_MODEL_CONFIG_PATH":
    lambda: get_env_with_fallback(
        "OMNI_CUSTOM_MODEL_CONFIG_PATH",
        ["CUSTOM_MODEL_CONFIG_PATH"],
        None,
    ),

    # =========================================================================
    # Existing OMNI_-prefixed variables
    # =========================================================================

    # Comma-separated allowlist of patch names to apply, or "ALL" (or empty)
    # to apply every registered patch. Unknown names are logged and skipped.
    # The dynamic omni-trace wrapper is gated independently by
    # OMNI_TRACE_OUTPUT_DIRECTORY.
    # Consumer: vllm_patches/patch_manager.py:50-75.
    "OMNI_VLLM_PATCHES":
    lambda: get_env_with_fallback(
        "OMNI_VLLM_PATCHES", ["OMNI_NPU_VLLM_PATCHES"], ""),

    # Comma-separated list of model patch directories under
    # vllm_patches/usefull_patch/models/ (and the legacy
    # vllm_patches/patches/models/ mapping table). Empty skips model-specific
    # usefull_patch dirs and only loads usefull_patch/common/.
    # Example: "pangu_v2_hybrid" or "pangu_v2_hybrid,pangu_v2_moe".
    # Also consumed by layers/__init__.py:28-36 (decides whether to load
    # mhc/mome modules), worker/npu_model_runner.py:458-462 (pangu_v2_hybrid
    # KV cache spec branch), and pangu_v2_hybrid/patch_speculative.py:66.
    "OMNI_VLLM_PATCHES_DIR":
    lambda: get_env_with_fallback(
        "OMNI_VLLM_PATCHES_DIR", ["OMNI_NPU_PATCHES_DIR"], ""),

    # Enables the NPU-side penalty cache in NPUSamplerV1: penalty masks
    # (prompt_mask / output_mask / output_bin_counts) are precomputed in
    # npu_input_batch and reused across steps, instead of being rebuilt per
    # step by vLLM's default SamplerV1. False falls back to the upstream
    # vLLM sampler. Requires npu_input_batch to be attached to the sampler.
    # Consumers: sample/sampler.py:22,58-95.
    "OMNI_NPU_PENALTY_CACHE":
    lambda: get_env_with_fallback(
        "OMNI_NPU_PENALTY_CACHE", None, False, _as_bool),

    # In PD deployments, lets the decode node reuse the first token already
    # generated by the prefill node (carried via
    # kv_transfer_params["prefilled_token"] plus its logprobs/text), so the
    # decode side skips regenerating it. Fallback to normal generation when:
    # the decoded text ends in U+FFFD, a stop_reason was already hit on
    # prefill, or the prefilled token is EOS.
    # Independent of (and composable with) OMNI_SKIP_DECODE_TOKENIZE.
    # Consumers: patch_prefilled_token_skip_tokenize.py:129-170,275-323,
    # 362-376,420-514; attention/backends/mome.py:148.
    "OMNI_REUSE_PREFILLED_TOKENS":
    lambda: get_env_with_fallback("OMNI_REUSE_PREFILLED_TOKENS", None, False, _as_bool),

    # In PD deployments, the prefill node ships the full prompt_token_ids via
    # kv_transfer_params so the decode node can skip tokenization entirely
    # (builds PrefilledTextPrompt directly). It cannot be combined with
    # OMNI_PIGGYBACK_INPUT_IDS, whose path requires decode-side tokenization
    # to remain enabled.
    "OMNI_SKIP_DECODE_TOKENIZE":
    lambda: get_env_with_fallback("OMNI_SKIP_DECODE_TOKENIZE", None, False, _as_bool),

    # Enables the [OMNI-CONF:...] startup configuration summary emitted from
    # NPUWorker.init_device (after load_model_extra_config). local_rank 0
    # prints the full projected config; other ranks print only a sha256 hash
    # for cross-rank drift detection. Failure degrades to a single WARNING
    # line and never breaks serving. Default True because production
    # troubleshooting relies on this snapshot.
    # Consumers: diagnostics/config_summary.py:53-54,339-379.
    "OMNI_CONFIG_SUMMARY":
    lambda: get_env_with_fallback("OMNI_CONFIG_SUMMARY", None, True, _as_bool),

    # Selects the all-to-all implementation used when lm_head vocab is
    # sharded (ena_local_lmhead_parallel or ena_dp_lmhead_parallel):
    #   True  -> comm_group.device_communicator.all_to_all (HCCL-native,
    #            generally faster on NPU)
    #   False -> hand-written torch.distributed.all_to_all_single (portable)
    # Has no effect when lm_head is not vocab-sharded.
    # Consumers: v1/layers/logits_processor.py:50-66.
    "OMNI_LMHEAD_USE_DEVICE_COMM_A2A":
    lambda: get_env_with_fallback(
        "OMNI_LMHEAD_USE_DEVICE_COMM_A2A",
        ["OMNI_NPU_USE_DEVICE_COMM_A2A"],
        False,
        _as_bool,
    ),

    # Lets a compatible client send pre-tokenized input_ids on the
    # ChatCompletionRequest; the server then skips chat-template expansion
    # and tokenization (fast path). Fast-path requires: caller_ids present,
    # tokenizer available, ChatCompletionRequest, no truncate_prompt_tokens,
    # and no multimodal content. Hard invariant: asserts
    # OMNI_SKIP_DECODE_TOKENIZE==0 at request time (mutually exclusive).
    # Consumers: patch_input_ids_piggyback.py:96-103,152-254.
    "OMNI_PIGGYBACK_INPUT_IDS":
    lambda: get_env_with_fallback("OMNI_PIGGYBACK_INPUT_IDS", None, False, _as_bool),

    # Debug aid for OMNI_PIGGYBACK_INPUT_IDS: when True, additionally runs
    # the full vLLM tokenization and compares it against the caller-supplied
    # input_ids; mismatch raises ValueError with a token-level diff. Costs an
    # extra tokenize per request — only enable while validating a new client.
    # Consumers: patch_input_ids_piggyback.py:100,159,175-243.
    "OMNI_VALIDATE_PIGGYBACK_INPUT_IDS":
    lambda: get_env_with_fallback("OMNI_VALIDATE_PIGGYBACK_INPUT_IDS", None, False, _as_bool),

    # =========================================================================
    # Benchmark gates (not for production serving)
    # =========================================================================

    # Benchmark gate for hybrid (proxy-fronted) DP serving: holds decode on
    # EVERY DP rank until ALL ranks have at least
    # OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD decode-ready requests, then
    # releases everywhere on the same engine step so the first decode batch
    # is full and aligned across ranks. One-shot: once released it stays open
    # for the engine's lifetime. Accepts "1"/"true"/"all".
    # Requires OMNI_VLLM_PATCHES_DIR to include pangu_v2_benchmark. Mutually
    # exclusive with decode_profile_sync and pd_bench_aligned_decode (they
    # patch the same attrs). Inflates TTFT — benchmarking only.
    # Consumers: pangu_v2_benchmark/patch_hybrid_aligned_decode.py:109-219.
    "OMNI_HYBRID_ALIGNED_DECODE":
    lambda: get_env_with_fallback(
        "OMNI_HYBRID_ALIGNED_DECODE", None, False, _as_bool_or_all),

    # Per-rank decode-ready request count required before the hybrid aligned
    # decode gate releases. Invalid (non-integer) values log a warning and
    # fall back to 16 instead of raising — a benchmark typo should not abort
    # startup.
    # Consumers: pangu_v2_benchmark/patch_hybrid_aligned_decode.py:115-116,208-211.
    "OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD":
    lambda: get_env_with_fallback(
        "OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD",
        None,
        16,
        lambda raw: _as_int_or_default(
            raw, 16, "OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD"),
    ),

    # Forces the internal DP load balancer (DPLBAsyncMPClient) to route
    # requests strictly round-robin (request i -> rank i % dp_size) instead
    # of the default least-loaded greedy policy. Accepts "1"/"true"/"all".
    # Requires OMNI_VLLM_PATCHES_DIR to include pangu_v2_benchmark; internal-LB
    # online DP only (proxy-fronted deployments route upstream and never
    # reach this client). An explicit per-request rank (X-data-parallel-rank
    # header) always bypasses round-robin.
    # Pairs with OMNI_HYBRID_ALIGNED_DECODE: round-robin gives every rank
    # exactly max_num_seqs requests, which makes the aligned-decode threshold
    # release cleanly without deadlock. Use --api-server-count 1 for exactness.
    # Consumers: pangu_v2_benchmark/patch_dp_round_robin.py:77-126.
    "OMNI_DP_ROUND_ROBIN":
    lambda: get_env_with_fallback(
        "OMNI_DP_ROUND_ROBIN", None, False, _as_bool_or_all),

    # Per-rank finished-KV-reception count required before the PD-benchmark
    # aligned decode gate releases. Unlike the hybrid gate (which keys on
    # local prefill completion), this one keys on remote KV arrival and is
    # only meaningful on PD-disaggregated decode nodes. Requires
    # OMNI_VLLM_PATCHES_DIR to include pd_bench_aligned_decode (a separate
    # directory from pangu_v2_benchmark, so the two gates are independently
    # toggleable). Invalid values fall back to 20 with a warning.
    # Consumers: pd_bench_aligned_decode/patch_bench_aligned_decode.py:47-107.
    "OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD":
    lambda: get_env_with_fallback(
        "OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD",
        ["OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD"],
        20,
        lambda raw: _as_int_or_default(
            raw, 20, "OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD"),
    ),
}
# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
