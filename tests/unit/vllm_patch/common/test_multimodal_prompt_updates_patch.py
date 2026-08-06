# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptReplacement,
    apply_text_matches,
    apply_token_matches,
    find_mm_placeholders,
)
from vllm.tokenizers import TokenizerLike

from omni.vllm_patches.patches.common import patch_multimodal_prompt_updates as patch_mpu
from omni.vllm_patches.patches.common.patch_multimodal_prompt_updates import (
    MultimodalPromptUpdatesPatch,
    _apply_token_matches_with_placeholders,
)

_TOKEN_START = 1
_TOKEN_PREFIX_A = 9833
_TOKEN_PREFIX_B = 28747
_TOKEN_MM_PLACEHOLDER = 32000
_TOKEN_END = 918
_TOKEN_WRAP = 1550

_PATTERN_KEY_1 = "pattern_1"
_PATTERN_KEY_2 = "pattern_2"
_PATTERN_KEY_3 = "pattern_3"

_REPL_INDEX_START = -1
_REPL_INDEX_PREFIX = -2
_REPL_INDEX_END = -3

# Slow-path fallback test fixtures.
_MODALITY_IMAGE = "image"
_SLOW_PATH_PROMPT = [11, 22, 33]
_SLOW_PATH_TARGET_TOKEN = 99
_SLOW_PATH_REPLACEMENT_TOKENS = [7, 8]
_SLOW_PATH_FALLBACK_TOKEN_IDS = [11, 7, 8, 33]
_SLOW_PATH_PLACEHOLDER_START_IDX = 1


class _VllmProcessorTestDouble(BaseMultiModalProcessor):
    def __init__(
        self,
        tokenizer: TokenizerLike | None = None,
        *,
        apply_text_matches_fn: Callable[
            ..., tuple[str, Mapping[str, list[int | None]]]
        ] = apply_text_matches,
    ) -> None:
        def get_tokenizer() -> TokenizerLike | None:
            return tokenizer

        self.info = SimpleNamespace(get_tokenizer=get_tokenizer)
        self._tokenizer = tokenizer
        self._apply_text_matches_fn = apply_text_matches_fn

    def apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        return self._apply_prompt_updates(token_ids, mm_prompt_updates)

    def _apply_token_matches(
        self,
        prompt: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], Mapping[str, list[int | None]]]:
        return apply_token_matches(prompt, mm_prompt_updates, self._tokenizer)

    def _apply_text_matches(
        self,
        prompt: str,
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[str, Mapping[str, list[int | None]]]:
        return self._apply_text_matches_fn(prompt, mm_prompt_updates, self._tokenizer)

    def _find_mm_placeholders(
        self,
        new_token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        return find_mm_placeholders(
            new_token_ids,
            mm_prompt_updates,
            self._tokenizer,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, Any]:
        return {}

    def _get_prompt_updates(
        self,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: Any,
    ) -> list[Any]:
        return []


class _PatchedProcessorTestDouble(MultimodalPromptUpdatesPatch, _VllmProcessorTestDouble):
    def apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
        *,
        apply_text_matches_fn: Callable[
            ..., tuple[str, Mapping[str, list[int | None]]]
        ] | None = None,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        if apply_text_matches_fn is not None:
            tokenizer = self._tokenizer

            def bound_apply_text_matches(
                prompt: str,
                updates: MultiModalPromptUpdates,
                _tokenizer: TokenizerLike | None = None,
            ) -> tuple[str, Mapping[str, list[int | None]]]:
                return apply_text_matches_fn(prompt, updates, tokenizer)

            self._apply_text_matches_fn = bound_apply_text_matches
        return self._apply_prompt_updates(token_ids, mm_prompt_updates)


def _dynamic_replacement(idx: int) -> list[int]:
    return [-(idx + 1)]


def _build_mm_prompt_updates(
    target_by_key: dict[str, Any],
    repl_by_key: dict[str, Any],
    update_type: type[PromptInsertion] | type[PromptReplacement],
    mm_count: int,
) -> MultiModalPromptUpdates:
    return {
        key: [
            [update_type(key, target, repl_by_key[key]).resolve(i)]
            for i in range(mm_count)
        ]
        for key, target in target_by_key.items()
    }


def _vllm_origin_apply_prompt_updates(
    token_ids: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None = None,
) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
    processor = _VllmProcessorTestDouble(tokenizer)
    return processor.apply_prompt_updates(token_ids, mm_prompt_updates)


def _call_patched_apply_prompt_updates(
    token_ids: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None = None,
    *,
    apply_text_matches_fn: Callable[..., tuple[str, Mapping[str, list[int | None]]]] = (
        apply_text_matches
    ),
) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
    processor = _PatchedProcessorTestDouble(
        tokenizer,
        apply_text_matches_fn=apply_text_matches_fn,
    )
    return processor.apply_prompt_updates(token_ids, mm_prompt_updates)


def _call_patched_apply_prompt_updates_with_text_fn(
    token_ids: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None,
    apply_text_matches_fn: Callable[..., tuple[str, Mapping[str, list[int | None]]]],
) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
    processor = _PatchedProcessorTestDouble(tokenizer)
    return processor.apply_prompt_updates(
        token_ids,
        mm_prompt_updates,
        apply_text_matches_fn=apply_text_matches_fn,
    )


# Minimal case: single modality, non-empty replacement (from vLLM token vocab).
_SIMPLE_TOKEN_CASE = {
    "prompt": [
        _TOKEN_START,
        _TOKEN_PREFIX_A,
        _TOKEN_PREFIX_B,
        _TOKEN_MM_PLACEHOLDER,
        _TOKEN_MM_PLACEHOLDER,
        _TOKEN_END,
    ],
    "target_by_key": {_PATTERN_KEY_1: [_TOKEN_MM_PLACEHOLDER]},
    "repl_by_key": {
        _PATTERN_KEY_1: [
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_MM_PLACEHOLDER,
        ],
    },
}


@pytest.mark.unit
@pytest.mark.parametrize("update_type", [PromptInsertion, PromptReplacement])
def test_apply_token_matches_with_placeholders_matches_vllm_token_output(
    update_type,
):
    case = _SIMPLE_TOKEN_CASE
    mm_prompt_updates = _build_mm_prompt_updates(
        case["target_by_key"],
        case["repl_by_key"],
        update_type,
        mm_count=1,
    )

    new_token_ids, _, _ = _apply_token_matches_with_placeholders(
        case["prompt"],
        mm_prompt_updates,
        tokenizer=None,
    )
    vllm_new_token_ids, _ = apply_token_matches(
        case["prompt"],
        mm_prompt_updates,
        tokenizer=None,
    )

    assert new_token_ids == vllm_new_token_ids


@pytest.mark.unit
def test_patched_apply_prompt_updates_matches_vllm_origin_for_replace():
    case = _SIMPLE_TOKEN_CASE
    mm_prompt_updates = _build_mm_prompt_updates(
        case["target_by_key"],
        case["repl_by_key"],
        PromptReplacement,
        mm_count=1,
    )

    expected_token_ids, expected_placeholders = _vllm_origin_apply_prompt_updates(
        case["prompt"],
        mm_prompt_updates,
        tokenizer=None,
    )
    patched_token_ids, patched_placeholders = _call_patched_apply_prompt_updates(
        case["prompt"],
        mm_prompt_updates,
        tokenizer=None,
    )

    assert patched_token_ids == expected_token_ids
    assert patched_placeholders == expected_placeholders


# ---- Cases from vLLM test_find_update_tokens ----
_FIND_UPDATE_TOKEN_CASES = [
    {
        "id": "full_multi_modality",
        "prompt": [
            _TOKEN_START,
            _TOKEN_PREFIX_A,
            _TOKEN_PREFIX_B,
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_PREFIX_A,
            _TOKEN_PREFIX_B,
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_END,
        ],
        "target_by_key": {
            _PATTERN_KEY_1: [_TOKEN_MM_PLACEHOLDER],
            _PATTERN_KEY_2: [_TOKEN_PREFIX_A, _TOKEN_PREFIX_B],
            _PATTERN_KEY_3: [_TOKEN_END],
        },
        "repl_by_key": {
            _PATTERN_KEY_1: [_TOKEN_MM_PLACEHOLDER, _TOKEN_MM_PLACEHOLDER],
            _PATTERN_KEY_2: [],
            _PATTERN_KEY_3: [_TOKEN_WRAP, _TOKEN_END, _TOKEN_WRAP],
        },
        "expected_by_update_type_mm_count": {
            PromptInsertion: {
                0: [
                    _TOKEN_START,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_END,
                ],
                1: [
                    _TOKEN_START,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_END,
                    _TOKEN_WRAP,
                    _TOKEN_END,
                    _TOKEN_WRAP,
                ],
                2: [
                    _TOKEN_START,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_END,
                    _TOKEN_WRAP,
                    _TOKEN_END,
                    _TOKEN_WRAP,
                    _TOKEN_WRAP,
                    _TOKEN_END,
                    _TOKEN_WRAP,
                ],
            },
            PromptReplacement: {
                0: [
                    _TOKEN_START,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_END,
                ],
                1: [
                    _TOKEN_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_PREFIX_A,
                    _TOKEN_PREFIX_B,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_WRAP,
                    _TOKEN_END,
                    _TOKEN_WRAP,
                ],
                2: [
                    _TOKEN_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_WRAP,
                    _TOKEN_END,
                    _TOKEN_WRAP,
                ],
            },
        },
    },
    {
        "id": "empty_index_targets",
        "prompt": [],
        "target_by_key": {
            _PATTERN_KEY_1: PromptIndexTargets.start(),
            _PATTERN_KEY_2: PromptIndexTargets.prefix([_TOKEN_MM_PLACEHOLDER]),
            _PATTERN_KEY_3: PromptIndexTargets.end(),
        },
        "repl_by_key": {
            _PATTERN_KEY_1: [_REPL_INDEX_START],
            _PATTERN_KEY_2: [_REPL_INDEX_PREFIX],
            _PATTERN_KEY_3: [_REPL_INDEX_END],
        },
        "expected_by_update_type_mm_count": {
            PromptInsertion: {
                0: [],
                1: [_REPL_INDEX_START, _REPL_INDEX_END],
                2: [
                    _REPL_INDEX_START,
                    _REPL_INDEX_START,
                    _REPL_INDEX_END,
                    _REPL_INDEX_END,
                ],
            },
            PromptReplacement: {
                0: [],
                1: [_REPL_INDEX_START, _REPL_INDEX_END],
                2: [
                    _REPL_INDEX_START,
                    _REPL_INDEX_START,
                    _REPL_INDEX_END,
                    _REPL_INDEX_END,
                ],
            },
        },
    },
    {
        "id": "single_token_index_targets",
        "prompt": [_TOKEN_MM_PLACEHOLDER],
        "target_by_key": {
            _PATTERN_KEY_1: PromptIndexTargets.start(),
            _PATTERN_KEY_2: PromptIndexTargets.prefix([_TOKEN_MM_PLACEHOLDER]),
            _PATTERN_KEY_3: PromptIndexTargets.end(),
        },
        "repl_by_key": {
            _PATTERN_KEY_1: [_REPL_INDEX_START],
            _PATTERN_KEY_2: [_REPL_INDEX_PREFIX],
            _PATTERN_KEY_3: [_REPL_INDEX_END],
        },
        "expected_by_update_type_mm_count": {
            PromptInsertion: {
                0: [_TOKEN_MM_PLACEHOLDER],
                1: [
                    _REPL_INDEX_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_PREFIX,
                    _REPL_INDEX_END,
                ],
                2: [
                    _REPL_INDEX_START,
                    _REPL_INDEX_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_PREFIX,
                    _REPL_INDEX_PREFIX,
                    _REPL_INDEX_END,
                    _REPL_INDEX_END,
                ],
            },
            PromptReplacement: {
                0: [_TOKEN_MM_PLACEHOLDER],
                1: [
                    _REPL_INDEX_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_PREFIX,
                    _REPL_INDEX_END,
                ],
                2: [
                    _REPL_INDEX_START,
                    _REPL_INDEX_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_PREFIX,
                    _REPL_INDEX_PREFIX,
                    _REPL_INDEX_END,
                    _REPL_INDEX_END,
                ],
            },
        },
    },
    {
        "id": "dynamic_replacement",
        "prompt": [
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_MM_PLACEHOLDER,
        ],
        "target_by_key": {
            _PATTERN_KEY_1: [_TOKEN_MM_PLACEHOLDER],
        },
        "repl_by_key": {
            _PATTERN_KEY_1: _dynamic_replacement,
        },
        "expected_by_update_type_mm_count": {
            PromptInsertion: {
                0: [
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
                1: [
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
                2: [
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_START,
                    _REPL_INDEX_PREFIX,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
            },
            PromptReplacement: {
                0: [
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
                1: [
                    _REPL_INDEX_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
                2: [
                    _REPL_INDEX_START,
                    _REPL_INDEX_PREFIX,
                    _TOKEN_MM_PLACEHOLDER,
                ],
            },
        },
    },
    {
        "id": "prefix_index_dynamic_replacement",
        "prompt": [
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_MM_PLACEHOLDER,
            _TOKEN_MM_PLACEHOLDER,
        ],
        "target_by_key": {
            _PATTERN_KEY_1: PromptIndexTargets.prefix([_TOKEN_MM_PLACEHOLDER]),
        },
        "repl_by_key": {
            _PATTERN_KEY_1: _dynamic_replacement,
        },
        "expected_by_update_type_mm_count": {
            PromptInsertion: {
                0: [
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
                1: [
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
                2: [
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_START,
                    _REPL_INDEX_PREFIX,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
            },
            PromptReplacement: {
                0: [
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
                1: [
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_START,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
                2: [
                    _TOKEN_MM_PLACEHOLDER,
                    _REPL_INDEX_START,
                    _REPL_INDEX_PREFIX,
                    _TOKEN_MM_PLACEHOLDER,
                    _TOKEN_MM_PLACEHOLDER,
                ],
            },
        },
    },
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    _FIND_UPDATE_TOKEN_CASES,
    ids=[case["id"] for case in _FIND_UPDATE_TOKEN_CASES],
)
@pytest.mark.parametrize("update_type", [PromptInsertion, PromptReplacement])
@pytest.mark.parametrize("mm_count", [0, 1, 2])
def test_apply_token_matches_with_placeholders_find_update_tokens(
    case,
    update_type,
    mm_count,
):
    mm_prompt_updates = _build_mm_prompt_updates(
        case["target_by_key"],
        case["repl_by_key"],
        update_type,
        mm_count,
    )
    expected = case["expected_by_update_type_mm_count"][update_type][mm_count]

    new_token_ids, _, _ = _apply_token_matches_with_placeholders(
        case["prompt"],
        mm_prompt_updates,
        tokenizer=None,
    )
    vllm_new_token_ids, _ = apply_token_matches(
        case["prompt"],
        mm_prompt_updates,
        tokenizer=None,
    )

    assert new_token_ids == expected
    assert new_token_ids == vllm_new_token_ids


# vLLM test_find_update_tokens covers token output for all cases below.
# vLLM does not test _apply_prompt_updates end-to-end. Patch only keeps
# fast-path REPLACE combos where token matching fully succeeds without a
# real tokenizer (same scope as patch production usage).
_END_TO_END_REPLACE_PARAMS = [
    ("full_multi_modality", 0),
    ("full_multi_modality", 1),
    ("empty_index_targets", 0),
    ("single_token_index_targets", 0),
    ("single_token_index_targets", 1),
    ("single_token_index_targets", 2),
    ("dynamic_replacement", 0),
    ("dynamic_replacement", 1),
    ("dynamic_replacement", 2),
    ("prefix_index_dynamic_replacement", 0),
    ("prefix_index_dynamic_replacement", 1),
    ("prefix_index_dynamic_replacement", 2),
]
_FIND_UPDATE_TOKEN_CASES_BY_ID = {
    case["id"]: case for case in _FIND_UPDATE_TOKEN_CASES
}


@pytest.mark.unit
@pytest.mark.parametrize("case_id,mm_count", _END_TO_END_REPLACE_PARAMS)
def test_patched_apply_prompt_updates_find_update_tokens_replace(case_id, mm_count):
    case = _FIND_UPDATE_TOKEN_CASES_BY_ID[case_id]
    mm_prompt_updates = _build_mm_prompt_updates(
        case["target_by_key"],
        case["repl_by_key"],
        PromptReplacement,
        mm_count,
    )

    expected_token_ids, expected_placeholders = _vllm_origin_apply_prompt_updates(
        case["prompt"],
        mm_prompt_updates,
        tokenizer=None,
    )
    patched_token_ids, patched_placeholders = _call_patched_apply_prompt_updates(
        case["prompt"],
        mm_prompt_updates,
        tokenizer=None,
    )

    assert patched_token_ids == expected_token_ids
    assert patched_placeholders == expected_placeholders


@pytest.mark.unit
def test_patched_apply_prompt_updates_falls_back_to_text_matches(monkeypatch):
    prompt = _SLOW_PATH_PROMPT
    mm_prompt_updates = {
        _MODALITY_IMAGE: [
            [
                PromptReplacement(
                    _MODALITY_IMAGE,
                    [_SLOW_PATH_TARGET_TOKEN],
                    _SLOW_PATH_REPLACEMENT_TOKENS,
                ).resolve(0)
            ]
        ]
    }
    fallback_token_ids = _SLOW_PATH_FALLBACK_TOKEN_IDS
    fallback_match_result = {_MODALITY_IMAGE: [0]}

    def fake_apply_token_matches_with_placeholders(*_args, **_kwargs):
        return (
            list(prompt),
            {_MODALITY_IMAGE: [None]},
            {_MODALITY_IMAGE: []},
        )

    def fake_seq2text(*_args, **_kwargs):
        return "ignored"

    def fake_seq2tokens(_tokenizer, text_or_tokens, use_cache=False):
        if isinstance(text_or_tokens, str):
            return list(fallback_token_ids)
        return list(text_or_tokens)

    monkeypatch.setattr(
        patch_mpu,
        "_apply_token_matches_with_placeholders",
        fake_apply_token_matches_with_placeholders,
    )
    monkeypatch.setattr(
        patch_mpu,
        "_seq2text",
        fake_seq2text,
    )

    monkeypatch.setattr(patch_mpu, "_seq2tokens", fake_seq2tokens)

    def fake_apply_text_matches(_prompt, _updates, _tokenizer=None):
        return "ignored", fallback_match_result

    patched_token_ids, patched_placeholders = (
        _call_patched_apply_prompt_updates_with_text_fn(
            prompt,
            mm_prompt_updates,
            tokenizer=None,
            apply_text_matches_fn=fake_apply_text_matches,
        )
    )

    assert patched_token_ids == fallback_token_ids
    assert (
        patched_placeholders[_MODALITY_IMAGE][0].start_idx
        == _SLOW_PATH_PLACEHOLDER_START_IDX
    )
    assert (
        patched_placeholders[_MODALITY_IMAGE][0].tokens
        == _SLOW_PATH_REPLACEMENT_TOKENS
    )
