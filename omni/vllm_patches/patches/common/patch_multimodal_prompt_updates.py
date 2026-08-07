# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections import defaultdict
from collections.abc import Mapping, Sequence

from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    MultiModalPromptUpdates,
    MultiModalPromptUpdatesApplyResult,
    PlaceholderFeaturesInfo,
    ResolvedPromptUpdate,
    UpdateMode,
    _all_items_found,
    _find_matches,
    _seq2text,
    _seq2tokens,
)
from vllm.tokenizers import TokenizerLike

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def _apply_token_matches_with_placeholders(
    token_ids: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None,
) -> tuple[
    list[int],
    MultiModalPromptUpdatesApplyResult,
    Mapping[str, list[PlaceholderFeaturesInfo]],
]:
    """Apply token matches and record replacement offsets while building output."""
    mm_item_counts = {
        modality: len(items) for modality, items in mm_prompt_updates.items()
    }
    match_result: MultiModalPromptUpdatesApplyResult = {
        modality: [None] * len(items)
        for modality, items in mm_prompt_updates.items()
    }
    placeholders: dict[str, list[PlaceholderFeaturesInfo]] = {
        modality: []
        for modality in mm_prompt_updates
    }

    mm_found_counts = {
        modality: sum(result is not None for result in results)
        for modality, results in match_result.items()
    }
    if _all_items_found(mm_item_counts, mm_found_counts):
        return list(token_ids), match_result, placeholders

    new_token_ids: list[int] = []
    prev_end_idx = 0
    while True:
        mode, matches_to_apply = _find_matches(
            token_ids,
            mm_prompt_updates,
            tokenizer,
            prev_end_idx=prev_end_idx,
            current_result=match_result,
        )

        if mode is None:
            break

        for (modality, item_idx), (match, update_idx) in matches_to_apply:
            update = mm_prompt_updates[modality][item_idx][update_idx]

            if mode == UpdateMode.INSERT:
                end_idx_to_insert = match.end_idx
            elif mode == UpdateMode.REPLACE:
                end_idx_to_insert = match.start_idx
            else:
                raise RuntimeError(f"Unsupported prompt update mode: {mode}")

            new_token_ids.extend(token_ids[prev_end_idx:end_idx_to_insert])
            offset = len(new_token_ids)

            tokens = _seq2tokens(tokenizer, update.content.full)
            if tokens:
                content_is_embed = update.content.is_embed
                if content_is_embed is not None:
                    content_is_embed = content_is_embed(tokenizer, update.content.full)

                placeholders[modality].append(
                    PlaceholderFeaturesInfo(
                        modality=modality,
                        item_idx=item_idx,
                        start_idx=offset,
                        tokens=tokens,
                        is_embed=content_is_embed,
                    )
                )
                new_token_ids.extend(tokens)
            match_result[modality][item_idx] = update_idx

            # Exclude overlapping matches, matching vLLM's original behavior.
            prev_end_idx = match.end_idx

        mm_found_counts = {
            modality: sum(result is not None for result in results)
            for modality, results in match_result.items()
        }
        if _all_items_found(mm_item_counts, mm_found_counts):
            break

    new_token_ids.extend(token_ids[prev_end_idx:])
    return new_token_ids, match_result, placeholders


def _strip_empty_placeholders(
    placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
    return {modality: ph_list for modality, ph_list in placeholders.items() if ph_list}


@register_patch("MultimodalPromptUpdatesPatch", BaseMultiModalProcessor)
class MultimodalPromptUpdatesPatch(VLLMPatch):
    """
    Avoid the redundant _find_mm_placeholders scan after prompt updates.

    The original flow first applies prompt replacements, then scans the updated
    token list again with large slice comparisons to rebuild
    PlaceholderFeaturesInfo. This patch records replacement offsets while
    applying token matches. If token matching falls back to text matching, it
    uses the original _find_mm_placeholders for correctness.
    """

    _attr_names_to_apply = ["_apply_prompt_updates"]

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        tokenizer = self.info.get_tokenizer()

        #### patch begin
        new_token_ids, match_result, placeholders = (
            _apply_token_matches_with_placeholders(
                token_ids,
                mm_prompt_updates,
                tokenizer,
            )
        )

        if all(
            all(update_idx is not None for update_idx in update_idxs)
            for update_idxs in match_result.values()
        ):
            return new_token_ids, _strip_empty_placeholders(placeholders)

        new_text, match_result = self._apply_text_matches(
            _seq2text(tokenizer, token_ids, use_cache=False),
            mm_prompt_updates,
        )
        #### patch end

        new_token_ids = _seq2tokens(tokenizer, new_text, use_cache=False)

        matched_updates = defaultdict[str, list[Sequence[ResolvedPromptUpdate]]](list)
        for modality, update_idxs in match_result.items():
            for item_idx, update_idx in enumerate(update_idxs):
                if update_idx is None:
                    raise RuntimeError(
                        "Failed to apply prompt replacement for "
                        f"mm_items[{modality!r}][{item_idx}]"
                    )

                matched_updates[modality].append(
                    [mm_prompt_updates[modality][item_idx][update_idx]]
                )

        placeholders = self._find_mm_placeholders(
            new_token_ids,
            dict(matched_updates),
        )

        return new_token_ids, placeholders
