# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Token/offset contracts; the upstream matcher is a scripted interface double.

These tests cover our update assembly, not vLLM's match-selection algorithm.
"""

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class Mode(Enum):
    INSERT = "insert"
    REPLACE = "replace"


@dataclass
class Placeholder:
    modality: str
    item_idx: int
    start_idx: int
    tokens: list
    is_embed: object


def update(tokens, mask=None):
    return SimpleNamespace(content=SimpleNamespace(full=tokens, is_embed=mask))


def match(modality, item, start, end, alternative=0):
    return ((modality, item), (SimpleNamespace(start_idx=start, end_idx=end), alternative))


@pytest.fixture
def prompt(patch_env):
    tokenizer = object()
    target = type("BaseMultiModalProcessor", (), {})
    api = patch_env.stub(
        "vllm.multimodal.processing.processor", BaseMultiModalProcessor=target,
        MultiModalPromptUpdates=dict, MultiModalPromptUpdatesApplyResult=dict,
        PlaceholderFeaturesInfo=Placeholder, ResolvedPromptUpdate=object,
        UpdateMode=Mode, _find_matches=Mock(return_value=(None, [])),
        _all_items_found=lambda expected, found: expected == found,
        _seq2text=Mock(return_value="original prompt"),
        _seq2tokens=Mock(side_effect=lambda tok, seq, **kw: list(seq)),
    )
    patch_env.stub("vllm.tokenizers", TokenizerLike=object)
    mod = patch_env.load("patch_multimodal_prompt_updates")
    mod.MultimodalPromptUpdatesPatch.apply()
    processor = target()
    processor.info = SimpleNamespace(get_tokenizer=lambda: tokenizer)
    processor._apply_text_matches = Mock()
    processor._find_mm_placeholders = Mock()
    return SimpleNamespace(mod=mod, api=api, processor=processor, tokenizer=tokenizer)


def test_empty_updates_return_an_independent_token_list(prompt):
    original = [1, 2, 3]
    ids, placeholders = prompt.processor._apply_prompt_updates(original, {"image": []})
    assert ids == original and ids is not original
    assert placeholders == {}
    prompt.api._find_matches.assert_not_called()
    prompt.processor._apply_text_matches.assert_not_called()


@pytest.mark.parametrize("mode,expected,start", [
    (Mode.INSERT, [10, 20, 91, 92, 30], 2),
    (Mode.REPLACE, [10, 91, 92, 30], 1),
])
def test_insert_and_replace_keep_exact_placeholder_offsets(prompt, mode, expected, start):
    mask = [True, False]
    mask_fn = Mock(return_value=mask)
    replacement = update([91, 92], mask_fn)
    prompt.api._find_matches.return_value = (mode, [match("image", 0, 1, 2)])
    ids, placeholders = prompt.processor._apply_prompt_updates(
        [10, 20, 30], {"image": [[replacement]], "video": []})
    assert ids == expected
    assert placeholders == {"image": [Placeholder("image", 0, start, [91, 92], mask)]}
    assert placeholders["image"][0].is_embed is mask
    mask_fn.assert_called_once_with(prompt.tokenizer, replacement.content.full)
    prompt.processor._find_mm_placeholders.assert_not_called()


def test_adjacent_mixed_modal_updates_track_growth_and_shrinkage(prompt):
    updates = {"image": [[update([90, 91, 92])], [update([99]), update([])]],
               "video": [[update([80, 81])], [update([82])]]}
    # Matcher indices refer to the original input, never the expanding output.
    prompt.api._find_matches.side_effect = [
        (Mode.REPLACE, [match("image", 0, 1, 2)]),
        (Mode.REPLACE, [match("image", 1, 2, 3, 1)]),
        (Mode.INSERT, [match("video", 0, 3, 4)]),
        (Mode.REPLACE, [match("video", 1, 4, 5)]),
    ]
    ids, results, placeholders = prompt.mod._apply_token_matches_with_placeholders(
        [10, 11, 12, 13, 14, 15], updates, prompt.tokenizer)
    assert ids == [10, 90, 91, 92, 13, 80, 81, 82, 15]
    assert results == {"image": [0, 1], "video": [0, 0]}
    assert placeholders == {
        "image": [Placeholder("image", 0, 1, [90, 91, 92], None)],
        "video": [Placeholder("video", 0, 5, [80, 81], None),
                  Placeholder("video", 1, 7, [82], None)],
    }
    assert [c.kwargs["prev_end_idx"] for c in prompt.api._find_matches.call_args_list] == [0, 2, 3, 4]


def test_empty_replacement_is_matched_but_has_no_placeholder(prompt):
    prompt.api._find_matches.return_value = (Mode.REPLACE, [match("image", 0, 0, 1)])
    assert prompt.processor._apply_prompt_updates([1, 2], {"image": [[update([])]]}) == ([2], {})
    prompt.processor._apply_text_matches.assert_not_called()


@pytest.mark.parametrize("partial", [False, True])
def test_fallback_restarts_from_original_ids_and_uses_selected_alternatives(prompt, partial):
    def seq_to_tokens(tok, seq, **kw):
        return [90, 81] if seq == "rendered" else list(seq)

    original = [1, 2, 3, 4]
    updates = {"image": [[update([90])]], "video": [[update([80]), update([81])]]}
    prompt.api._find_matches.side_effect = (
        [(Mode.REPLACE, [match("image", 0, 0, 1)]), (None, [])]
        if partial else [(None, [])]
    )
    prompt.processor._apply_text_matches.return_value = ("rendered", {"image": [0], "video": [1]})
    prompt.api._seq2tokens.side_effect = seq_to_tokens
    expected = {"image": [Placeholder("image", 0, 0, [90], None)],
                "video": [Placeholder("video", 0, 1, [81], None)]}
    prompt.processor._find_mm_placeholders.return_value = expected
    assert prompt.processor._apply_prompt_updates(original, updates) == ([90, 81], expected)
    prompt.api._seq2text.assert_called_once_with(prompt.tokenizer, original, use_cache=False)
    prompt.processor._apply_text_matches.assert_called_once_with("original prompt", updates)
    prompt.processor._find_mm_placeholders.assert_called_once_with(
        [90, 81], {"image": [[updates["image"][0][0]]], "video": [[updates["video"][0][1]]]})
    assert prompt.api._seq2tokens.call_args.kwargs == {"use_cache": False}
    assert original == [1, 2, 3, 4]


def test_unmatched_item_after_text_fallback_is_an_error(prompt):
    prompt.processor._apply_text_matches.return_value = ([1], {"image": [None]})
    with pytest.raises(RuntimeError, match="mm_items\\['image'\\]\\[0\\]"):
        prompt.processor._apply_prompt_updates([1], {"image": [[update([90])]]})
    prompt.processor._find_mm_placeholders.assert_not_called()


def test_unsupported_update_mode_preserves_existing_error(prompt):
    prompt.api._find_matches.return_value = ("unsupported", [match("image", 0, 0, 1)])
    with pytest.raises(RuntimeError, match="Unsupported prompt update mode"):
        prompt.processor._apply_prompt_updates([1], {"image": [[update([90])]]})
