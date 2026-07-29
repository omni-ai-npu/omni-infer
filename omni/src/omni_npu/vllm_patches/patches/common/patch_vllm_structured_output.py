# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This file carries two concerns:
#   1. ``StructuredOutputManagerPatch`` — engine-core bitmask/advance behaviour
#      for structured output + reasoning interplay (pre-existing).
#   2. Server-side structured-output configuration: a ``--structured-output-config``
#      CLI flag + ``vllm_config.structured_output_config`` that supplies structured
#      output defaults when a request body carries none (request wins).
#
# Patch files load in alphabetical filename order, and the engine-arg patch chain
# (``VllmConfigPatch`` / ``EngineArgsPatch``) is a relay where each later file
# re-registers the same patch *name* and relays the previous file's attrs. This
# file is named ``patch_vllm_...`` so it sorts AFTER ``patch_user_repetition_detection.py``
# and becomes the common chain end — letting it add a real CLI flag and populate
# ``vllm_config.structured_output_config`` via ``create_engine_config``.

import argparse
import itertools
import json
import multiprocessing
import os
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import xgrammar

import vllm.config as vllm_config_module
from vllm import EngineArgs, SamplingParams
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")

from vllm.v1.structured_output import StructuredOutputManager
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.common.patch_user_repetition_detection import (
    EngineArgsPatch as _UpstreamEngineArgsPatch,
    VllmConfigPatch as _UpstreamVllmConfigPatch,
    _relay_patch_attrs
)

logger = init_logger(__name__)


@register_patch("StructuredOutputManagerPatch", StructuredOutputManager)
class StructuredOutputManagerPatch(VLLMPatch):

    _attr_names_to_apply = ["grammar_bitmask", "should_fill_bitmask",
                            "should_advance", "_find_reasoning_end_offset",
                            "trim_reasoning_for_advance"]
    
    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

        if self._grammar_bitmask is None:
            if self.backend is None:
                raise ValueError(f"self.backend is None")
            
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    if structured_output_request is None:
                        raise ValueError(f"structured_output_request is None")
                    if structured_output_request.grammar is None:
                        raise ValueError(f"structured_output_request.grammar is None")
                grammar = structured_output_request.grammar

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    if structured_output_request is None:
                        raise ValueError(f"structured_output_request is None")
                    if structured_output_request.grammar is None:
                        raise ValueError(f"structured_output_request.grammar is None")
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)

                reasoner = self.reasoner
                detect_reasoning_end = (
                    not apply_bitmask
                    and reasoner is not None
                    and not self.enable_in_reasoning
                )
                history_prefix: list[int] | None = None

                state_advancements = 0
                post_reasoning_end_in_window = False
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for i, token in enumerate(req_tokens):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    advance_grammar = apply_bitmask
                    if token == -1:
                        apply_bitmask = False
                        advance_grammar = False
                    elif detect_reasoning_end and not apply_bitmask:
                        if history_prefix is None:
                            history_prefix = list(request.all_token_ids)
                        simulated = history_prefix + list(req_tokens[: i + 1])
                        if reasoner.is_reasoning_end_streaming(simulated, [token]):
                            # Reasoning ended mid-window. Constrain the rest
                            # of the window via bitmask. Skip grammar advance
                            # through the marker (it is reasoning content);
                            # try to advance through subsequent drafts so the
                            # next bitmask row reflects the post-advance state,
                            # but tolerate rejection since those drafts predate
                            # the bitmask and are not guaranteed valid.
                            apply_bitmask = True
                            advance_grammar = False
                            post_reasoning_end_in_window = True
                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )
                    cumulative_index += 1
                
                is_diffusion = getattr(self.vllm_config.model_config, 'is_diffusion', False)
                if not (is_diffusion and req_tokens):
                    # Diffusion LLMs don't sample a bonus token after the
                    # scheduled positions, so skip its bitmask in that case.
                    bonus_apply = self.should_fill_bitmask(request) or apply_bitmask
                    self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()
    
    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.

        if self.reasoner is not None:
            if self.enable_in_reasoning:
                return True
            if request.structured_output_request is None:
                raise ValueError(f"request.structured_output_request is None")
            
            if request.structured_output_request.reasoning_ended is None:
                request.structured_output_request.reasoning_ended = (
                    self.reasoner.is_reasoning_end(request.prompt_token_ids)
                )
            return request.structured_output_request.reasoning_ended
        return True
    
    def should_advance(
        self,
        request: "Request",
        new_token_ids: list[int] | None = None,
    ) -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            if structured_output_request is None:
                raise ValueError(f"structured_output_request is None")
            if structured_output_request.grammar is None:
                raise ValueError(f"structured_output_request.grammar is None")
            
        # by default, we should always advance
        # for cases that don't use thinking mode.
        if self.reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step.
        # When the caller passes new_token_ids (the tokens that were just
        # appended this step), use it directly as the delta window. The
        # placeholder-derived fallback assumes num_output_placeholders ==
        # len(new_token_ids), which breaks under async scheduling + spec
        # decode when some drafts are rejected (#43388): the placeholder
        # count remains > 0 after the step and the computed delta window
        # starts past the reasoning-end marker.
        all_token_ids = request.all_token_ids
        if new_token_ids:
            # The tokens were already appended this step, so the step window
            # starts exactly len(new_token_ids) from the end.
            start = len(all_token_ids) - len(new_token_ids)
            delta_ids: Iterable[int] = new_token_ids
        else:
            delta_from = (
                request.num_computed_tokens - request.num_output_placeholders
            )
            start = (
                delta_from
                if delta_from >= 0
                else max(len(all_token_ids) + delta_from, 0)
            )
            delta_ids = itertools.islice(all_token_ids, start, None)
            
        if self.reasoner.is_reasoning_end_streaming(all_token_ids, delta_ids):
            structured_req.reasoning_ended = True

            # Locate the reasoning-end marker within the step once; both
            # branches below rely on it. Everything up to and including the
            # marker is reasoning content and must never reach the grammar.
            step_tokens = (
                new_token_ids
                if new_token_ids
                else list(itertools.islice(all_token_ids, start, None))
            )
            end_offset = self._find_reasoning_end_offset(
                self.reasoner, itertools.islice(all_token_ids, start), step_tokens
            )
            # When the marker cannot be pinned to a single token (e.g. a
            # multi-token marker only recognized on the full delta),
            # conservatively treat the whole step as reasoning content.
            structured_req.reasoning_end_token_index = (
                start + end_offset
                if end_offset is not None
                else len(all_token_ids) - 1
            )
            # Reasoning just ended this step. Defer FSM advance until the next
            # pass (see reasoning_ended check above) for JSON/regex/choice/grammar:
            # advancing on the closing boundary token can accept tokens that still
            # belong to the reasoning stream. Structural tags are the only safe
            # same-step exception: they model phased output (e.g. thinking tag ->
            # answer tag), and speculative decoding must run grammar.validate_tokens
            # on draft tokens produced immediately after that transition.
            if (
                self.vllm_config.speculative_config is not None
                and structured_req.structured_output_key[0]
                == StructuredOutputOptions.STRUCTURAL_TAG
            ):
                return True

            # Deferred backends still need the post-marker tail of this step's
            # new_token_ids to be drained into the FSM, otherwise the next
            # step's bitmask preparation sees grammar at its initial state and
            # the model can emit a duplicate opening token (e.g. "{{") when
            # reasoning ended inside a spec-decode window.
            if new_token_ids and end_offset is not None:
                post_marker = list(new_token_ids[end_offset + 1:])
                grammar = structured_req.grammar
                if post_marker and grammar is not None:
                    if not grammar.accept_tokens(request.request_id, post_marker):
                        # These tokens were sampled before the grammar became
                        # active and are not guaranteed valid; tolerate the
                        # rejection like grammar_bitmask does post-marker.
                        
                        logger.warning(
                            "Grammar rejected post-reasoning tokens %s for "
                            "request %s; continuing without the advance.",
                            post_marker,
                            request.request_id,
                        )
        return False

    @staticmethod
    def _find_reasoning_end_offset(
        reasoner: "ReasoningParser",
        prefix_tokens: Iterable[int],
        step_tokens: list[int],
    ) -> int | None:
        """Locates the last reasoning token within ``step_tokens``.

        Args:
            reasoner: The request's reasoning parser.
            prefix_tokens: Tokens that precede ``step_tokens`` in the
                request, used as streaming context for the parser.
            step_tokens: The tokens produced by the current step.

        Returns:
            The offset within ``step_tokens`` of the token at which
            ``is_reasoning_end_streaming`` first fires, or None when no
            single token triggers the detection.
        """
        prefix = list(prefix_tokens)
        for offset, token in enumerate(step_tokens):
            prefix.append(token)
            if reasoner.is_reasoning_end_streaming(prefix, [token]):
                return offset
        return None

    def trim_reasoning_for_advance(
        self, request: "Request", new_token_ids: list[int]
    ) -> list[int]:
        """Drops reasoning content from tokens about to advance the grammar.

        When reasoning ends mid-step (see should_advance), the step's output
        still contains reasoning tokens up to and including the end marker.
        Those are not grammar content: feeding them to accept_tokens makes
        the grammar reject the marker and kills the request (#44006).

        Returns:
            The suffix of ``new_token_ids`` that follows the reasoning-end
            marker. Steps fully after the boundary are returned unchanged.
        """
        structured_req = request.structured_output_request
        if structured_req is None:
            return new_token_ids
        
        end_idx = getattr(structured_req, "reasoning_end_token_index", None)
        if end_idx is None:
            return new_token_ids
        
        first_idx = len(request.all_token_ids) - len(new_token_ids)
        num_reasoning = end_idx + 1 - first_idx
        if num_reasoning <= 0:
            return new_token_ids
        
        return new_token_ids[num_reasoning:]


def _register_attribute_to_class(model_cls: type, attribute: str) -> None:
    model_cls.__annotations__[attribute] = (int | None)

_register_attribute_to_class(StructuredOutputRequest, "reasoning_end_token_index")


# ===========================================================================
# Server-side structured-output configuration
# ===========================================================================
# Allows structured-output constraints (choice / json / regex / grammar /
# json_object / structural_tag, or a request-style ``response_format``) to be
# configured once on the server and applied as a default to requests that do
# not carry their own structured output. A request that supplies its own
# ``structured_outputs`` / ``response_format`` always wins (see
# ``_validate_structured_output`` below).

# ``vllm_config`` / ``EngineArgs`` attribute name carrying the resolved config.
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
_STRUCTURED_OUTPUT_CONFIG_ENV = "STRUCTURED_OUTPUT_CONFIG"


def _structured_output_config_from_env() -> StructuredOutputRequestConfig | None:
    """Resolve a ``StructuredOutputRequestConfig`` from the env var.

    The value must be the same JSON string accepted by ``--structured-output-config``.
    Returns ``None`` when the variable is unset/empty or fails to parse.
    """
    raw = os.environ.get(_STRUCTURED_OUTPUT_CONFIG_ENV)
    if not raw:
        return None
    try:
        cfg = _coerce_structured_output_config(raw)
    except argparse.ArgumentTypeError as exc:
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


def _resolve_server_structured_output_config(
    vllm_config: Any,
) -> StructuredOutputRequestConfig | None:
    """Resolve the effective server-side config: vllm_config value first,
    then the env-var fallback. Used by the request-time injection."""
    cfg = getattr(vllm_config, STRUCTURED_OUTPUT_CONFIG_KEY, None)
    cfg = _coerce_structured_output_config(cfg)
    if cfg is not None:
        return cfg
    return _structured_output_config_from_env()


@register_patch("StructuredOutputRequestConfigModulePatch", vllm_config_module)
class StructuredOutputRequestConfigModulePatch(VLLMPatch):
    """Expose the out-of-tree structured-output config through ``vllm.config``."""

    _attr_names_to_apply = ["StructuredOutputRequestConfig"]

    StructuredOutputRequestConfig = StructuredOutputRequestConfig


@register_patch("VllmConfigPatch", VllmConfig)
class VllmConfigPatch(VLLMPatch):
    """Relay ``patch_user_repetition_detection.VllmConfigPatch`` and add structured-output config."""

    _attr_names_to_apply = list(_UpstreamVllmConfigPatch._attr_names_to_apply) + [
        STRUCTURED_OUTPUT_CONFIG_KEY,
    ]

    structured_output_config: StructuredOutputRequestConfig | None = None


_relay_patch_attrs(VllmConfigPatch, _UpstreamVllmConfigPatch)


_orig_ea_add_cli_args = _UpstreamEngineArgsPatch.add_cli_args
_orig_ea_from_cli_args = _UpstreamEngineArgsPatch.from_cli_args.__func__
_orig_ea_create_engine_config = _UpstreamEngineArgsPatch.create_engine_config


@register_patch("EngineArgsPatch", EngineArgs)
class EngineArgsPatch(VLLMPatch):
    """Relay the upstream EngineArgs chain and add ``--structured-output-config``."""

    _attr_names_to_apply = list(_UpstreamEngineArgsPatch._attr_names_to_apply) + [
        STRUCTURED_OUTPUT_CONFIG_KEY,
    ]

    structured_output_config: StructuredOutputRequestConfig | None = None

    @staticmethod
    def add_cli_args(parser):
        parser = _orig_ea_add_cli_args(parser)
        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        vllm_group.add_argument(
            "--structured-output-config",
            **StructuredOutputRequestConfig.as_argparse_dict(),
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        instance = _orig_ea_from_cli_args(cls, args)
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
        vllm_config = _orig_ea_create_engine_config(self, usage_context, headless)
        structured_output_config = _coerce_structured_output_config(
            getattr(self, STRUCTURED_OUTPUT_CONFIG_KEY, None)
        )
        structured_output_config_env = _structured_output_config_from_env()
        if structured_output_config_env:
            structured_output_config = structured_output_config_env
        
        vllm_config.structured_output_config = structured_output_config
        if structured_output_config is not None:
            logger.info(
                "Server-side structured output config enabled: %s",
                structured_output_config,
            )
        return vllm_config


_relay_patch_attrs(
    EngineArgsPatch,
    _UpstreamEngineArgsPatch,
    exclude=("add_cli_args", "from_cli_args", "create_engine_config"),
)


# ---------------------------------------------------------------------------
# Request-time injection: apply the server default when the request carries
# no structured output (request wins).
# ---------------------------------------------------------------------------

# Imported lazily to avoid a circular import at module load time; the
# InputProcessor imports structured-output backends.
def _get_input_processor_cls() -> type:
    from vllm.v1.engine.input_processor import InputProcessor as _InputProcessor
    return _InputProcessor


_InputProcessor = _get_input_processor_cls()
_original_validate_structured_output = _InputProcessor._validate_structured_output


@register_patch("StructuredOutputInputProcessorPatch", _InputProcessor)
class StructuredOutputInputProcessorPatch(VLLMPatch):
    """Inject server-side structured-output defaults before validation.

    Runs inside ``_validate_structured_output`` (called from
    ``process_inputs`` → ``_validate_params`` → ``_validate_sampling_params``),
    i.e. before ``params.clone()``. When the request body supplied its own
    ``structured_outputs`` / ``response_format`` the constraints are already
    populated and are left untouched (request wins). Otherwise the server-side
    config is materialized into ``params.structured_outputs`` so vLLM's
    existing backend selection + validation handle it like any request.
    """

    _attr_names_to_apply = ["_validate_structured_output"]

    def _validate_structured_output(self, params: SamplingParams) -> None:
        structured_outputs = getattr(params, "structured_outputs", None)
        empty = structured_outputs is None or structured_outputs.all_constraints_none()
        
        if empty:
            cfg = _resolve_server_structured_output_config(self.vllm_config)
            if cfg is not None:
                built = cfg.build_structured_outputs_params()
                if built is not None:
                    params.structured_outputs = built
        return _original_validate_structured_output(self, params)
    

_orig_apply_token_bitmask_inplace = xgrammar.apply_token_bitmask_inplace


@register_patch("NPUGrammarBitmaskBackendPatch", xgrammar)
class NPUGrammarBitmaskBackendPatch(VLLMPatch):
    """Force xgrammar's bitmask kernel onto the triton-free ``torch_native``
    backend on NPU to avoid ``import triton`` triggered by ``torch.compile``.
    """

    _attr_names_to_apply = ["apply_token_bitmask_inplace"]

    @staticmethod
    def apply_token_bitmask_inplace(logits: torch.Tensor, grammar_bitmask, *args, **kwargs) -> None:
        if logits.device.type not in ("cpu", "cuda"):
            kwargs["backend"] = "torch_native"
        else:
            kwargs["backend"] = "auto"
        
        return _orig_apply_token_bitmask_inplace(logits, grammar_bitmask, *args, **kwargs)