# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Typed accessor for omni-npu fields inside ``vllm_config.additional_config``.

Background
----------
vLLM exposes ``--additional-config '<JSON>'`` as a generic escape hatch; the
parsed dict lives at ``vllm_config.additional_config`` and is shared with
vLLM itself and other plugins. omni-npu historically read this dict with
scattered ``additional_config.get("key", default)`` calls, so the owned schema,
defaults, and type checks were not defined in one place. A JSON ``"true"``
(string) or ``1`` (int) where a real ``bool`` is expected could flow downstream
and break in non-obvious ways.

This module replaces those raw reads with a single dataclass,
:class:`OmniAdditionalConfig`, that:

* declares every omni-npu-owned key as a typed field (unknown keys from
  other plugins are ignored — the dict is shared);
* validates field types in ``__post_init__`` (bools must be real ``bool``,
  positive ints must be ``int >= 1``, the sub-dict must be a ``dict``);
* raises ``TypeError`` at *construction* when a known key has the wrong
  type, so mistakes surface at startup instead of mid-request.

Because the input dictionary is shared, unknown keys are ignored. This also
means a misspelled omni-npu key such as ``enable_low_latncy`` cannot be
distinguished from another plugin's key and still falls back silently.

Parsing is lazy-per-call: every ``from_vllm_config`` re-reads the dict, so
mutations to ``vllm_config.additional_config`` after startup are visible
to later parses (see ``test_vllm_config_is_parsed_on_each_access``).

Data flow
---------
::

    vllm serve ... --additional-config '{"enable_low_latency":true, ...}'
        │
        ▼
    vllm_config.additional_config          (loose dict, shared)
        │
        ▼  OmniAdditionalConfig.from_vllm_config(vllm_config)
    ┌────────────────────────────────────────────────────────┐
    │ enable_pd_elastic_scaling / enable_low_latency /       │
    │ enable_omni_cache ──► loader.task_config               │
    │                       (picks best-practice JSON:       │
    │                        performance_mode dir, pd_scheme)│
    │                                                        │
    │ npugraph_ex_config ──► AclGraphConfig / GraphPassManager
    │                       (NPU graph compile options)      │
    │                                                        │
    │ combine_block ──► NPUModelRunner block-table sizing    │
    │                                                        │
    │ enable_full_async_rl ──► NPUWorker RL lifecycle        │
    └────────────────────────────────────────────────────────┘

Cross-source invariant (validators.omni_cache_consistent, REJECT):
when ``enable_omni_cache`` is explicitly present here, it MUST equal
``envs.OMNI_ENABLE_OMNI_CACHE``. Legacy omniinfer launchers that configure
Omni Cache through the environment only remain supported.
"""
from dataclasses import dataclass, field, fields
from typing import Any, Dict


# Field groups enforced by __post_init__. Keep in sync with the dataclass
# fields below; tests enumerate these via dataclasses.fields() so a missing
# entry here means the field is NOT type-checked.
_BOOL_FIELDS = (
    "enable_full_async_rl",
    "enable_pd_elastic_scaling",
    "enable_low_latency",
    "enable_omni_cache",
)
# Must be real ``int`` (not bool, not str) AND >= 1.
_POSITIVE_INT_FIELDS = ("combine_block",)


@dataclass
class OmniAdditionalConfig:
    """Validated omni-npu fields parsed from ``additional_config``.

    Field catalogue (6 fields). "Consumer" lists the actual read site in the
    current tree.
    """

    # ------------------------------------------------------------------
    # Worker runtime configuration
    # ------------------------------------------------------------------

    # Enables the fully asynchronous reinforcement-learning worker lifecycle.
    # Consumed by NPUWorker to select the sleep/wake-up behavior used by RL.
    enable_full_async_rl: bool = False

    # Alignment factor for the KV cache block-table width. Must be a positive
    # int. ACTIVELY CONSUMED: NPUModelRunner.max_num_blocks_per_req
    # (npu_model_runner.py:159-162). It rounds the ordinary block count up
    # to a multiple of combine_block and sets the second dimension of
    # graph_block_tables
    # (npu_model_runner.py:163-166). Values >1 can leave the width unchanged or
    # increase it by up to combine_block - 1; this code does not itself combine
    # consecutive blocks or reduce block-table memory.
    combine_block: int = 1

    # ------------------------------------------------------------------
    # NPU graph extensions (consumed by compilation/*)
    # ------------------------------------------------------------------

    # Sub-dict of options for the npugraph_ex / ACL-graph compile path.
    # The parent value must be a real dict (validated). Consumers read only the
    # recognized sub-keys below; unknown sub-keys are ignored rather than
    # forwarded to npugraph_ex.CompilerConfig. Nested value types are not
    # validated by OmniAdditionalConfig.
    #
    # Recognized sub-keys (see compilation/npugraph_ex.py:44-124,
    # compilation/acl_graph.py:198-204, compilation/pass_manager.py:40-53):
    #   enable (bool, default False)             — master switch; without it
    #                                              the npugraph_ex path is
    #                                              bypassed entirely.
    #   static_kernel_compile (bool)             — AOT-compile static kernels.
    #   super_kernel_optimize (bool)             — super-kernel fusion with
    #                                              dcci options; implies
    #                                              static_kernel_compile.
    #   capture_limit (int, default 64)          — static capture size cap.
    #   clone_input (bool, default True)         — clone graph inputs.
    #   clone_output (bool, default False)       — clone graph outputs.
    #   remove_noop_ops (bool, default True)     — strip no-op nodes.
    #   inplace_pass (bool, default True)        — convert to in-place ops.
    #   input_inplace_pass (bool, default True)  — restore in-place inputs.
    #   pattern_fusion_pass (bool, default True) — FX pattern fusion.
    #   frozen_parameter (bool, default False)   — freeze weights+graph.
    #   merge_dynamic_quant (bool)               — GraphPassManager pass:
    #                                              folds dynamic-quant nodes.
    #   enable_moe_multistream (bool)            — GraphPassManager pass
    #                                              (currently a no-op stub).
    npugraph_ex_config: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # PD elasticity / low latency / cache
    # (consumed by model_config/config_loader/loader.py:37-45 → task_config)
    # ------------------------------------------------------------------

    # Selects the PD elastic-scaling tuning profile. When True AND
    # is_pd_disaggregation, forces pd_scheme =
    # "pd_elastic_scaling" (loader.py:401-404), which selects a generic
    # default tuning profile instead of a topology-specific "{P}P{D}D" one.
    # This field does not resize pods; deployment orchestration performs the
    # actual scaling. It has no effect in hybrid (non-PD) mode.
    enable_pd_elastic_scaling: bool = False

    # Selects the low-latency tuning profile directory. When True,
    # loader picks configs from model_config/configs/low_latency/ instead
    # of high_throughout/ (loader.py:385). Typical usage:
    #     --additional-config '{"enable_low_latency":true}'
    enable_low_latency: bool = False

    # Enables Omni Cache (KV reuse in PD deployments). Fed into
    # task_config.enable_omni_cache (loader.py:82) — but note that
    # task_config only records it; the actual runtime switch is
    # operator_opt_config.use_omni_cache, which is FORCE-SET by the
    # OMNI_ENABLE_OMNI_CACHE env var in ModelOperatorOptConfig.__post_init__
    # (loader.py:210-211) regardless of this field.
    #
    # HARD INVARIANT (validators.omni_cache_consistent, REJECT):
    #     when this field is explicitly supplied in additional_config, it MUST
    #     equal envs.OMNI_ENABLE_OMNI_CACHE. Environment-only legacy launches
    #     are accepted.
    enable_omni_cache: bool = False

    def __post_init__(self) -> None:
        # Type-check the bool group: JSON "true" (str), 1 (int), None, []
        # are all rejected.
        for name in _BOOL_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"additional_config.{name} must be bool, "
                    f"got {type(value).__name__}"
                )

        # Type-check and range-check the positive-int group. Explicitly
        # exclude bool because it is a subclass of int.
        for name in _POSITIVE_INT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"additional_config.{name} must be int, "
                    f"got {type(value).__name__}"
                )
            if value < 1:
                raise ValueError(
                    f"additional_config.{name} must be >= 1, got {value}"
                )

        # The sub-dict must be a real dict; other mappings (e.g. list of
        # pairs) are rejected to fail fast on malformed JSON.
        if not isinstance(self.npugraph_ex_config, dict):
            raise TypeError(
                "additional_config.npugraph_ex_config must be dict, "
                f"got {type(self.npugraph_ex_config).__name__}"
            )

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "OmniAdditionalConfig":
        """Parse and validate fields from ``vllm_config.additional_config``.

        Unknown keys are ignored because the dictionary is shared with vLLM
        and other plugins. ``vllm_config=None`` or ``additional_config=None``
        both yield an all-defaults instance.
        """
        raw = (
            getattr(vllm_config, "additional_config", None)
            if vllm_config is not None
            else None
        )
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "OmniAdditionalConfig":
        """Parse and validate omni-npu fields from a shared dictionary.

        ``None`` is treated as ``{}`` (all defaults). Any non-dict, non-None
        value raises TypeError — ``additional_config=[]`` or ``=""`` is a
        deployment bug and must surface loudly. Field filtering happens
        *before* dataclass construction, so unknown shared keys (from vLLM
        or other plugins) never reach __init__ and never trigger the
        "unexpected keyword argument" TypeError.
        """
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(
                "additional_config must be dict or None, "
                f"got {type(raw).__name__}"
            )
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in field_names})
