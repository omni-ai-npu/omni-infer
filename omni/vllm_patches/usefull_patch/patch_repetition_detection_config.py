# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side default for repetition detection (vLLM 0.25.1)."""

# vLLM 0.25.1 already ships repetition detection natively (upstream PR #35451):
# ``RepetitionDetectionParams``, ``SamplingParams.repetition_detection``, the
# field on both OpenAI request bodies, the detection in ``check_stop``, and the
# ``repetition`` finish reason.  ``patches/common/patch_user_repetition_detection.py``
# backported all of that for 0.14 and is superseded here -- do not load both,
# they both patch ``EngineArgs.add_cli_args``.
#
# What upstream still lacks is an *engine-level* default, so an operator can
# enable detection once at launch instead of every client repeating it per
# request.  That is all this file adds:
#   1. ``--repetition-detection`` + ``VllmConfig.repetition_detection``
#   2. ``OMNI_REPETITION_DETECTION_CONFIG`` env override
#   3. injection in ``InputProcessor.process_inputs``
# Precedence: request body > env > CLI > off, same as the 0.14 patch. A valid
# env value overrides the CLI flag; a malformed one logs and leaves the CLI
# value in place.

# Keeps `X | None` annotations from being evaluated at import time, so this
# module also imports under the Python 3.9 that lives outside the container
# (the offline tests import it directly).
from __future__ import annotations

import argparse
import json
from typing import Any

from vllm import EngineArgs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

_KEY = "repetition_detection"
_CLI_FLAG = "--repetition-detection"
_ENV_NAME = "OMNI_REPETITION_DETECTION_CONFIG"

_HELP = (
    "JSON object configuring repetitive N-gram detection in output tokens, "
    'e.g. \'{"max_pattern_size":10,"min_pattern_size":2,"min_count":3}\'. '
    "Applies to requests that do not carry their own `repetition_detection`; "
    "a request-body value always wins. Set max_pattern_size to 0 (or omit the "
    f"flag) to disable. {_ENV_NAME} overrides this flag."
)

# Resolved once per process by create_engine_config().  VllmConfig can only carry
# the value as a plain attribute (see below), which a dataclasses.replace() or a
# reconstruction elsewhere would silently drop; this is the safety net.
_PROCESS_DEFAULT: RepetitionDetectionParams | None = None

_ENV_DEFAULT_UNRESOLVED = object()
_env_default: Any = _ENV_DEFAULT_UNRESOLVED


def _coerce(value: Any) -> RepetitionDetectionParams | None:
    """Accept a RepetitionDetectionParams, a JSON string, or a dict."""
    if value is None or isinstance(value, RepetitionDetectionParams):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        # __post_init__ enforces min_pattern_size <= max_pattern_size and
        # min_count >= 2 whenever max_pattern_size > 0.
        return RepetitionDetectionParams(**value)
    return None


def _parse_cli(value: str) -> RepetitionDetectionParams | None:
    """argparse ``type=`` for --repetition-detection.

    Raises so argparse fails the launch: a wrong server-wide default that starts
    silently is worse than one that does not start.
    """
    if value in ("", "None"):
        return None
    try:
        return _coerce(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid {_CLI_FLAG} JSON: {value!r} ({exc})") from exc


def _from_env() -> RepetitionDetectionParams | None:
    """Parse the env override once per process.

    Never raises: a bad env var must not take down a node that was otherwise
    launched correctly. It logs and returns None, which leaves the CLI value
    (or the disabled default) in place.
    """
    global _env_default
    if _env_default is not _ENV_DEFAULT_UNRESOLVED:
        return _env_default

    raw = envs.OMNI_REPETITION_DETECTION_CONFIG
    if not raw:
        _env_default = None
        return None

    try:
        _env_default = _coerce(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error parsing %s (%r): %s -- ignored", _ENV_NAME, raw, exc)
        _env_default = None
    else:
        if _env_default is not None:
            logger.info("Loaded repetition detection default from %s", _ENV_NAME)
    return _env_default


# ────────────────────────────────────────────────────────────
# Patch 1: VllmConfig — carry the engine-level default
# ────────────────────────────────────────────────────────────
# Class-level default only. It cannot be a real dataclass field: VllmConfig is a
# pydantic dataclass built with extra="forbid", so passing the key to its
# generated __init__ would raise. create_engine_config sets the instance value.
@register_patch("OmniRepetitionDetectionVllmConfigPatch", VllmConfig)
class VllmConfigRepetitionDetectionPatch(VLLMPatch):
    _attr_names_to_apply = [_KEY]

    repetition_detection: RepetitionDetectionParams | None = None


# ────────────────────────────────────────────────────────────
# Patch 2: EngineArgs — add --repetition-detection
# ────────────────────────────────────────────────────────────
# The flag registers in time because AsyncEngineArgs.add_cli_args calls
# load_general_plugins() -- which applies this patch -- before delegating to
# EngineArgs.add_cli_args.
_ORIG_ADD_CLI_ARGS = EngineArgs.add_cli_args
_ORIG_FROM_CLI_ARGS = EngineArgs.from_cli_args.__func__
_ORIG_CREATE_ENGINE_CONFIG = EngineArgs.create_engine_config


@register_patch("OmniRepetitionDetectionEngineArgsPatch", EngineArgs)
class EngineArgsRepetitionDetectionPatch(VLLMPatch):
    _attr_names_to_apply = [
        _KEY,
        "add_cli_args",
        "from_cli_args",
        "create_engine_config",
    ]

    repetition_detection: RepetitionDetectionParams | None = None

    @staticmethod
    def add_cli_args(parser):
        parser = _ORIG_ADD_CLI_ARGS(parser)
        group = parser.add_argument_group(
            title="OmniNPU",
            description="omni-npu out-of-tree engine options.",
        )
        try:
            group.add_argument(
                _CLI_FLAG,
                dest=_KEY,
                type=_parse_cli,
                default=None,
                help=_HELP,
            )
        except argparse.ArgumentError:
            # add_cli_args invoked twice on the same parser must not abort.
            logger.debug("%s already registered on this parser", _CLI_FLAG)
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Upstream rebuilds the instance from dataclasses.fields(cls), and
        # repetition_detection is a plain class attribute rather than a field,
        # so it has to be copied across by hand.
        instance = _ORIG_FROM_CLI_ARGS(cls, args)
        try:
            instance.repetition_detection = _coerce(getattr(args, _KEY, None))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error reading %s: %s -- disabled", _CLI_FLAG, exc)
            instance.repetition_detection = None
        return instance

    def create_engine_config(self, usage_context=None, headless: bool = False):
        vllm_config = _ORIG_CREATE_ENGINE_CONFIG(self, usage_context, headless)

        global _PROCESS_DEFAULT
        cfg = _coerce(getattr(self, _KEY, None))
        source = _CLI_FLAG

        # The env var wins over the CLI flag. _from_env() returns None both when
        # unset and when malformed, so a bad env value leaves the CLI value in
        # place rather than silently disabling the feature.
        env_cfg = _from_env()
        if env_cfg is not None:
            cfg = env_cfg
            source = _ENV_NAME

        vllm_config.repetition_detection = cfg
        _PROCESS_DEFAULT = cfg

        if cfg is not None:
            logger.info(
                "Repetition detection default (from %s): max_pattern_size=%d, "
                "min_pattern_size=%d, min_count=%d",
                source,
                cfg.max_pattern_size,
                cfg.min_pattern_size,
                cfg.min_count,
            )
        return vllm_config


# ────────────────────────────────────────────────────────────
# Patch 3: InputProcessor — inject the default into requests
# ────────────────────────────────────────────────────────────
# Not the two serving classes: upstream's to_sampling_params reads this one
# parameter straight off the request body, with no default_sampling_params
# fallback (unlike every neighbouring parameter), and get_diff_sampling_param()
# whitelists six keys that do not include it -- so writing into
# serving.default_sampling_params would be a no-op.  process_inputs is also the
# single point where the chat, completion and offline LLM.generate paths meet.
_ORIG_PROCESS_INPUTS = InputProcessor.process_inputs


def _resolve_default(processor: InputProcessor) -> RepetitionDetectionParams | None:
    """The default create_engine_config resolved, with two fallbacks in case a
    dataclasses.replace() elsewhere dropped the plain attribute."""
    cfg = getattr(getattr(processor, "vllm_config", None), _KEY, None)
    if cfg is not None:
        return cfg
    if _PROCESS_DEFAULT is not None:
        return _PROCESS_DEFAULT
    return _from_env()


def _find_sampling_params(args: tuple, kwargs: dict) -> SamplingParams | None:
    params = kwargs.get("params")
    if params is None:
        for arg in args:
            if isinstance(arg, SamplingParams):
                return arg
        return None
    return params if isinstance(params, SamplingParams) else None


@register_patch("OmniRepetitionDetectionInputProcessorPatch", InputProcessor)
class InputProcessorRepetitionDetectionPatch(VLLMPatch):
    _attr_names_to_apply = ["process_inputs"]

    def process_inputs(self, *args, **kwargs):
        # *args/**kwargs rather than the real signature: the value is mutated in
        # place, so this wrapper never has to reconstruct the call and stays
        # immune to signature churn.
        params = _find_sampling_params(args, kwargs)
        if params is not None and params.repetition_detection is None:
            default = _resolve_default(self)
            if default is not None:
                # Mutating rather than cloning: process_inputs clones params
                # itself a few lines down, and on the OpenAI paths this object is
                # built fresh per request (skip_clone=True).  Offline callers who
                # reuse one SamplingParams see the field appear on their object --
                # idempotent, and it is the default they asked for at launch.
                params.repetition_detection = default

        # Runs before the native _validate_params, so an injected value goes
        # through the same checks a client-supplied one would.
        return _ORIG_PROCESS_INPUTS(self, *args, **kwargs)
