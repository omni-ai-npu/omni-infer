# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Slim usefull_patch counterpart of
# ``patches/common/patch_vllm_structured_output.py``. Same section order:
#   1. ``StructuredOutputReasoningAdvancePatch`` — same-step JSON/regex
#      grammar advance at the think-end boundary, with the end index pinned
#      to the real ``</think>`` / ``[unused17]`` token. Upstream 0.25.1
#      defers JSON advance and can land the trim index on a ``{`` / ``[``
#      opener under MTP, which re-emits ``{"{`` / ``[[``. Scheduler trim /
#      accept_tokens is unchanged.
#   2. Server-side structured-output configuration:
#      ``--structured-output-config`` / ``OMNI_STRUCTURED_OUTPUT_CONFIG``.
#      Unlike the common patch, this never hangs a field on ``VllmConfig``
#      (``dataclasses.replace(vllm_config)`` rejects unknown fields); resolved
#      config lives in a process-local stash. Injection uses
#      ``InputProcessor._validate_params`` because 0.25.1 no longer has
#      ``_validate_structured_output``.
#   3. ``NPUGrammarBitmaskBackendPatch`` — force ``torch_native`` on NPU.
#
# Layout intentionally mirrors the common file so diffs stay easy to review.

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import xgrammar
from vllm import EngineArgs, SamplingParams
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.structured_output import StructuredOutputManager

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


# --------------------------------------------------------------------------
# 1) Reasoning-boundary grammar advance
# --------------------------------------------------------------------------

_THINK_END_STRINGS = ("</think>", "[unused17]")


def _reasoning_end_token_ids(reasoner: Any, tokenizer: Any) -> set[int]:
    """Token ids that actually terminate thinking (not the JSON opener)."""
    ids: set[int] = set()
    if reasoner is None:
        return ids
    for attr in ("end_token_id", "_reasoning_end_token_id"):
        tid = getattr(reasoner, attr, None)
        if isinstance(tid, int):
            ids.add(tid)
    engine = getattr(reasoner, "_parser_engine", None)
    if engine is not None:
        tid = getattr(engine, "_reasoning_end_token_id", None)
        if isinstance(tid, int):
            ids.add(tid)
    vocab = None
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = None
    end_str = getattr(reasoner, "reasoning_end_str", None)
    if callable(end_str):
        end_str = end_str()
    if isinstance(end_str, str) and isinstance(vocab, dict):
        tid = vocab.get(end_str)
        if isinstance(tid, int):
            ids.add(tid)
    if isinstance(vocab, dict):
        for s in _THINK_END_STRINGS:
            tid = vocab.get(s)
            if isinstance(tid, int):
                ids.add(tid)
    return ids


@register_patch("StructuredOutputReasoningAdvancePatch", StructuredOutputManager)
class StructuredOutputReasoningAdvancePatch(VLLMPatch):
    """Same-step FSM advance at think-end for every structured-output backend.

    Upstream defers JSON/regex/choice/grammar until the next step. Under MTP
    the opener ``{`` / ``[`` is already sampled in the same window; the next
    bitmask is still at S0 and re-emits it. Returning True lets the scheduler
    trim through the think-end marker and accept the opener now.

    Streaming ``is_reasoning_end`` can stay True after ``</think>``, so the
    first hit in the window may be the opener. Pin ``reasoning_end_token_index``
    to the real think-end token so trim keeps ``{`` / ``[``.
    """

    _attr_names_to_apply = [
        "should_advance",
        "_find_reasoning_end_index",
    ]

    def should_advance(self, request) -> bool:
        if not request.use_structured_output:
            return False

        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            structured_req.reasoning_ended = True
            structured_req.reasoning_end_token_index = (
                self._find_reasoning_end_index(reasoner, all_token_ids, start)
            )
            return True

        return False

    def _find_reasoning_end_index(
        self,
        reasoner: Any,
        all_token_ids: Sequence[int],
        start: int,
    ) -> int:
        prefix = list(itertools.islice(all_token_ids, start))
        detected = len(all_token_ids) - 1
        for idx in range(start, len(all_token_ids)):
            token = all_token_ids[idx]
            prefix.append(token)
            if reasoner.is_reasoning_end_streaming(prefix, [token]):
                detected = idx
                break

        end_ids = _reasoning_end_token_ids(reasoner, getattr(self, "tokenizer", None))
        if end_ids:
            for idx in range(detected, -1, -1):
                if all_token_ids[idx] in end_ids:
                    return idx
        return detected


# --------------------------------------------------------------------------
# 2) Server-side structured-output configuration
# --------------------------------------------------------------------------
# Allows structured-output constraints (choice / json / regex / grammar /
# json_object / structural_tag, or a request-style ``response_format``) to be
# configured once on the server and applied as a default to requests that do
# not carry their own structured output. A request that supplies its own
# ``structured_outputs`` / ``response_format`` always wins (see
# ``_validate_params`` injection below).

# ``EngineArgs`` attribute name carrying the resolved config (not hung on
# ``VllmConfig``; see module docstring).
STRUCTURED_OUTPUT_CONFIG_KEY = "structured_output_config"

# Constraint fields mirrored from ``StructuredOutputsParams``. Exactly one may
# be set (mutual exclusion is enforced in ``__post_init__``), matching
# ``StructuredOutputsParams.__post_init__``.
_STRUCTURED_OUTPUT_CONSTRAINT_FIELDS = (
    "json",
    "regex",
    "choice",
    "grammar",
    "json_object",
    "structural_tag",
)

# Process-local stash: prefer exact ``id(vllm_config)``; fall back to the
# latest stored config so ``replace()``-derived copies still resolve.
_CONFIG_BY_VLLM_CONFIG_ID: dict[int, "StructuredOutputRequestConfig"] = {}
_LATEST_STRUCTURED_OUTPUT_CONFIG: "StructuredOutputRequestConfig | None" = None


@dataclass
class StructuredOutputRequestConfig:
    """Server-side default for structured output constraints.

    Mirrors the per-request ``StructuredOutputsParams`` so the server can be
    configured with the same shapes a request body accepts:

    * ``choice``: e.g. ``["positive", "negative"]``
      (equivalent to ``"structured_outputs": {"choice": [...]}``).
    * ``response_format``: a request-style ``response_format`` dict, e.g.
      ``{"type": "structural_tag", "structures": [...], "triggers": [...]}``
      or ``{"type": "json_schema", "json_schema": {...}}`` — translated the
      same way ``protocol.py`` translates a request body.
    * direct constraint fields ``json`` / ``regex`` / ``grammar`` /
      ``json_object`` / ``structural_tag``.

    Only one constraint kind may be set. ``structural_tag`` accepts a dict
    (it is JSON-serialized, matching the request-body path).
    """

    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    structural_tag: str | dict | None = None

    # Convenience: paste a request-style ``response_format`` directly.
    response_format: dict | None = None

    # Options forwarded to ``StructuredOutputsParams``.
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None

    def __post_init__(self) -> None:
        count = sum(
            getattr(self, name) is not None
            for name in _STRUCTURED_OUTPUT_CONSTRAINT_FIELDS
        ) + (1 if self.response_format is not None else 0)

        if count > 1:
            raise ValueError(
                "You can only use one kind of structured output constraint "
                f"but multiple are specified: {self.__dict__}"
            )

    @property
    def has_constraint(self) -> bool:
        return any(
            getattr(self, name) is not None
            for name in _STRUCTURED_OUTPUT_CONSTRAINT_FIELDS
        ) or self.response_format is not None

    @staticmethod
    def as_argparse_dict() -> dict[str, Any]:
        """argparse kwargs mirroring ReasoningConfig.as_argparse_dict."""
        doc = (
            StructuredOutputRequestConfig.__doc__.strip()
            if StructuredOutputRequestConfig.__doc__
            else "Server-side structured output configuration."
        )
        return {
            "type": _parse_structured_output_config_cli,
            "default": None,
            "help": (
                f"{doc} Provide as a JSON string or a path to a JSON file, "
                'e.g. \'{"choice": ["positive", "negative"]}\'. When set, '
                "requests without their own structured output use this as the "
                "default; requests that carry structured output are unaffected."
            ),
        }

    def build_structured_outputs_params(self) -> StructuredOutputsParams | None:
        """Build a ``StructuredOutputsParams`` from this server config.

        Returns ``None`` when no constraint is configured. The result is a
        plain ``StructuredOutputsParams`` so vLLM's existing backend selection
        and validation apply identically to the request-body path.
        """
        if not self.has_constraint:
            return None

        kwargs: dict[str, Any] = {
            "disable_fallback": self.disable_fallback,
            "disable_any_whitespace": self.disable_any_whitespace,
            "disable_additional_properties": self.disable_additional_properties,
            "whitespace_pattern": self.whitespace_pattern,
        }

        if self.response_format is not None:
            self._apply_response_format(self.response_format, kwargs)
        elif self.choice is not None:
            kwargs["choice"] = self.choice
        elif self.json is not None:
            kwargs["json"] = self.json
        elif self.regex is not None:
            kwargs["regex"] = self.regex
        elif self.grammar is not None:
            kwargs["grammar"] = self.grammar
        elif self.json_object is not None:
            kwargs["json_object"] = self.json_object
        elif self.structural_tag is not None:
            kwargs["structural_tag"] = self._serialize_structural_tag(
                self.structural_tag
            )
        return StructuredOutputsParams(**kwargs)

    @staticmethod
    def _serialize_structural_tag(value: str | dict) -> str:
        """Accept a dict (like a request ``response_format`` structural tag) or
        an already-serialized JSON string, returning the JSON string form that
        ``StructuredOutputsParams.structural_tag`` expects (matching
        ``protocol.py`` which stores ``json.dumps(model_dump(by_alias=True))``).
        """
        if isinstance(value, str):
            return value
        return json.dumps(value)

    @staticmethod
    def _apply_response_format(
        response_format: dict, kwargs: dict[str, Any]
    ) -> None:
        """Translate a request-style ``response_format`` dict, mirroring
        ``protocol.py`` (lines ~791-815 / ~1235-1259)."""
        rf_type = response_format.get("type")
        if rf_type == "json_object":
            kwargs["json_object"] = True
        elif rf_type == "json_schema":
            json_schema = response_format.get("json_schema")
            if json_schema is None:
                # Allow ``{"type": "json_schema", "schema": {...}}`` shorthand.
                json_schema = {"json_schema": response_format.get("schema")}
            elif isinstance(json_schema, dict):
                json_schema = json_schema.get("json_schema", json_schema)
            kwargs["json"] = json_schema
        elif rf_type == "structural_tag":
            # ``response_format`` here already has the ``structures``/``triggers``
            # shape; serialize it verbatim, exactly like the request-body path.
            kwargs["structural_tag"] = json.dumps(response_format)
        else:
            raise ValueError(
                "Unsupported response_format type for server-side structured "
                f"output config: {rf_type!r}. Expected one of "
                "'json_object', 'json_schema', 'structural_tag'."
            )


def _coerce_structured_output_config(
    value: Any,
) -> StructuredOutputRequestConfig | None:
    """Coerce None / JSON string / dict / path-to-JSON-file into the config.

    Returns ``None`` for empty/None input. Raises on invalid JSON or unknown
    fields (forwarded to the dataclass constructor).
    """
    if value is None or isinstance(value, StructuredOutputRequestConfig):
        return value
    if isinstance(value, str):
        # JSON string or path to JSON file → parsed config (or None when empty).
        return _parse_structured_output_config_cli(value)
    if isinstance(value, dict):
        return StructuredOutputRequestConfig(**value)
    return None


def _parse_structured_output_config_cli(
    raw: str,
) -> StructuredOutputRequestConfig | None:
    """argparse type: accept a JSON string or a path to a JSON file.

    Empty / ``"None"`` → ``None`` (feature disabled).
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw == "None":
        return None
    # A path to a JSON file: load it. Distinguish from inline JSON by checking
    # for a leading ``{`` / ``[``; otherwise treat as a filesystem path.
    first = raw.lstrip()[:1]
    if first not in ("{", "["):
        try:
            with open(raw, "r", encoding="utf-8") as f:
                raw = f.read()
        except OSError as exc:
            raise argparse.ArgumentTypeError(
                f"--structured-output-config: cannot read JSON file {raw!r}: "
                f"{exc}"
            ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--structured-output-config: invalid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError(
            "--structured-output-config: expected a JSON object."
        )
    return StructuredOutputRequestConfig(**data)


# Environment variable mirroring ``--structured-output-config``. Used as a
# fallback when the CLI flag is absent (or when ``EngineArgs`` is constructed
# programmatically without ``from_cli_args``). The Environment variable flag always wins.
_STRUCTURED_OUTPUT_CONFIG_ENV = "OMNI_STRUCTURED_OUTPUT_CONFIG"


def _structured_output_config_from_env() -> StructuredOutputRequestConfig | None:
    """Resolve a ``StructuredOutputRequestConfig`` from the env var.

    The value must be the same JSON string accepted by ``--structured-output-config``.
    Returns ``None`` when the variable is unset/empty or fails to parse.
    """
    raw = envs.OMNI_STRUCTURED_OUTPUT_CONFIG
    if not raw:
        return None
    try:
        cfg = _coerce_structured_output_config(raw)
    except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
        logger.error(
            "Error parsing %s environment variable: %s",
            _STRUCTURED_OUTPUT_CONFIG_ENV,
            exc,
        )
        return None
    if cfg is not None:
        logger.info(
            "Loaded structured_output_config from the %s environment variable",
            _STRUCTURED_OUTPUT_CONFIG_ENV,
        )
    return cfg


def _store_config_for_vllm_config(
    vllm_config: Any, cfg: StructuredOutputRequestConfig | None
) -> None:
    global _LATEST_STRUCTURED_OUTPUT_CONFIG
    key = id(vllm_config)
    if cfg is None:
        _CONFIG_BY_VLLM_CONFIG_ID.pop(key, None)
        return
    _CONFIG_BY_VLLM_CONFIG_ID[key] = cfg
    _LATEST_STRUCTURED_OUTPUT_CONFIG = cfg


def _resolve_server_structured_output_config(
    vllm_config: Any,
) -> StructuredOutputRequestConfig | None:
    """Resolve the effective server-side config: stash by ``id(vllm_config)``,
    then latest-config fallback, then the env-var fallback."""
    cfg = _CONFIG_BY_VLLM_CONFIG_ID.get(id(vllm_config))
    if cfg is not None:
        return cfg
    if _LATEST_STRUCTURED_OUTPUT_CONFIG is not None:
        return _LATEST_STRUCTURED_OUTPUT_CONFIG
    # Programmatic / env-only path when create_engine_config was not used.
    return _structured_output_config_from_env()


@register_patch("StructuredOutputConfigEngineArgsPatch", EngineArgs)
class EngineArgsPatch(VLLMPatch):
    """Add ``--structured-output-config`` without mutating ``VllmConfig`` fields."""

    _attr_names_to_apply = [
        STRUCTURED_OUTPUT_CONFIG_KEY,
        "add_cli_args",
        "from_cli_args",
        "create_engine_config",
    ]

    structured_output_config: StructuredOutputRequestConfig | None = None

    @classmethod
    def apply(cls):
        """Capture currently-active EngineArgs methods, then overlay ours.

        Other usefull_patch modules may already wrap ``add_cli_args`` /
        ``from_cli_args`` / ``create_engine_config``. Capture-at-apply keeps
        the chain intact and avoids ``already patched`` conflicts.
        """
        cls.apply_bypass_conflict(
            "add_cli_args", "from_cli_args", "create_engine_config",
        )

    @staticmethod
    def add_cli_args(parser):
        parser = EngineArgsPatch._upstream_add_cli_args(parser)
        group = parser.add_argument_group(
            title="OmniStructuredOutputConfig",
            description=(
                "Server-side default structured output constraints "
                "(applied when a request carries none)."
            ),
        )
        group.add_argument(
            "--structured-output-config",
            **StructuredOutputRequestConfig.as_argparse_dict(),
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        instance = EngineArgsPatch._upstream_from_cli_args(cls, args)
        raw = getattr(args, STRUCTURED_OUTPUT_CONFIG_KEY, None)
        try:
            instance.structured_output_config = _coerce_structured_output_config(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error parsing structured_output_config: %s", exc)
            instance.structured_output_config = None
        return instance

    def create_engine_config(
        self,
        usage_context=None,
        headless: bool = False,
    ) -> VllmConfig:
        vllm_config = EngineArgsPatch._upstream_create_engine_config(
            self, usage_context, headless
        )
        structured_output_config = _coerce_structured_output_config(
            getattr(self, STRUCTURED_OUTPUT_CONFIG_KEY, None)
        )
        structured_output_config_env = _structured_output_config_from_env()
        if structured_output_config_env:
            structured_output_config = structured_output_config_env

        _store_config_for_vllm_config(vllm_config, structured_output_config)
        if structured_output_config is not None:
            logger.info(
                "Server-side structured output config enabled: %s",
                structured_output_config,
            )
        return vllm_config


# --------------------------------------------------------------------------
# 3) Request-time injection (request wins)
# --------------------------------------------------------------------------

def _maybe_inject_server_structured_outputs(
    vllm_config: Any, params: SamplingParams
) -> None:
    structured_outputs = getattr(params, "structured_outputs", None)
    empty = structured_outputs is None or structured_outputs.all_constraints_none()
    if empty:
        cfg = _resolve_server_structured_output_config(vllm_config)
        if cfg is not None:
            built = cfg.build_structured_outputs_params()
            if built is not None:
                params.structured_outputs = built


_original_validate_params = InputProcessor._validate_params


@register_patch("StructuredOutputInputProcessorPatch", InputProcessor)
class StructuredOutputInputProcessorPatch(VLLMPatch):
    """Inject server-side structured-output defaults before validation.

    Runs inside ``_validate_params`` (0.25.1 no longer has
    ``_validate_structured_output``). When the request body supplied its own
    ``structured_outputs`` / ``response_format`` the constraints are already
    populated and are left untouched (request wins). Otherwise the server-side
    config is materialized into ``params.structured_outputs`` so vLLM's
    existing backend selection + validation handle it like any request.
    """

    _attr_names_to_apply = ["_validate_params"]

    def _validate_params(self, params, supported_tasks):
        if isinstance(params, SamplingParams):
            _maybe_inject_server_structured_outputs(self.vllm_config, params)
        return _original_validate_params(self, params, supported_tasks)


# --------------------------------------------------------------------------
# 4) xgrammar bitmask backend
# --------------------------------------------------------------------------

_orig_apply_token_bitmask_inplace = xgrammar.apply_token_bitmask_inplace


@register_patch("NPUGrammarBitmaskBackendPatch", xgrammar)
class NPUGrammarBitmaskBackendPatch(VLLMPatch):
    """Force xgrammar's bitmask kernel onto the triton-free ``torch_native``
    backend on NPU to avoid ``import triton`` triggered by ``torch.compile``.
    """

    _attr_names_to_apply = ["apply_token_bitmask_inplace"]

    @staticmethod
    def apply_token_bitmask_inplace(logits, grammar_bitmask, *args, **kwargs):
        if logits.device.type not in ("cpu", "cuda"):
            kwargs["backend"] = "torch_native"
        else:
            kwargs["backend"] = "auto"

        return _orig_apply_token_bitmask_inplace(
            logits, grammar_bitmask, *args, **kwargs
        )
