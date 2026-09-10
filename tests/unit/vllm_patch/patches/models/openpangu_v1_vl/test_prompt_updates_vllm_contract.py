# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cloud contract tests using the real vLLM 0.25.1 matcher and placeholder scan.

Run separately from the CPU interface-double suite. No model weights needed.
"""

import pytest

pytest.importorskip("vllm")

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode,ids,targets,contents", [
    ("replace", [1, 2, 3], [[2]], [[90, 91]]),
    ("insert", [1, 2, 3], [[2]], [[90, 91]]),
    ("replace", [1, 2, 3, 4], [[2], [3]], [[90, 91], [92]]),
    ("insert", [1, 2, 3], [[2], [2]], [[90], [91]]),
    ("replace", [1, 2, 3, 4], [[1, 2], [2, 3]], [[90], [91]]),
    ("replace", [1, 2, 3], [[2]], [[]]),
])
def test_token_assembly_agrees_with_real_vllm(mode, ids, targets, contents):
    from vllm.multimodal.processing import processor as api
    from omni_npu.vllm_patches.patches.models.openpangu_v1_vl import (
        patch_multimodal_prompt_updates as patch,
    )

    updates = {"image": [], "video": []}
    for i, (target, content) in enumerate(zip(targets, contents)):
        modality = "image" if i % 2 == 0 else "video"
        modality_updates = updates.get(modality)
        assert modality_updates is not None
        modality_updates.append([api.ResolvedPromptUpdate(
            modality=modality, item_idx=0, mode=api.UpdateMode(mode),
            target=target, content=api.PromptUpdateDetails(full=content),
        )])
    expected_ids, expected_matches = api.apply_token_matches(ids, updates, None)
    actual_ids, actual_matches, actual_ph = patch._apply_token_matches_with_placeholders(ids, updates, None)
    assert actual_ids == expected_ids
    assert actual_matches == expected_matches
    if all(index is not None for indices in expected_matches.values() for index in indices):
        selected = {modality: [[items[i][index]] for i, index in enumerate(expected_matches[modality])]
                    for modality, items in updates.items()}
        expected_ph = api.find_mm_placeholders(expected_ids, selected, None)
        assert {key: value for key, value in actual_ph.items() if value} == expected_ph


def test_embed_mask_agrees_with_real_placeholder_scan():
    import torch
    from vllm.multimodal.processing import processor as api
    from omni_npu.vllm_patches.patches.models.openpangu_v1_vl import (
        patch_multimodal_prompt_updates as patch,
    )

    update = api.PromptReplacement(
        modality="image", target=[2],
        replacement=api.PromptUpdateDetails.select_token_id([90, 91, 90], 90),
    ).resolve(0)
    updates = {"image": [[update]]}
    expected_ids, _ = api.apply_token_matches([1, 2, 3], updates, None)
    expected = api.find_mm_placeholders(expected_ids, updates, None)["image"][0]
    actual_ids, _, placeholders = patch._apply_token_matches_with_placeholders([1, 2, 3], updates, None)
    actual = placeholders["image"][0]
    assert actual_ids == expected_ids
    assert (actual.modality, actual.item_idx, actual.start_idx, actual.tokens) == (
        expected.modality, expected.item_idx, expected.start_idx, expected.tokens)
    torch.testing.assert_close(actual.is_embed, expected.is_embed, rtol=0, atol=0)
