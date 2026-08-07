# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest

pytest.importorskip("vllm")

from omni_npu.vllm_patches.patches.common.patch_user_repetition_detection import (
    RepetitionDetectionParams,
    _coerce_repetition_detection,
    _parse_repetition_detection_cli,
    _resolve_repetition_detection,
    check_sequence_repetition,
)


def test_parse_repetition_detection_cli_accepts_json_object():
    params = _parse_repetition_detection_cli(
        '{"max_pattern_size": 3, "min_pattern_size": 2, "min_count": 4}'
    )

    assert params == RepetitionDetectionParams(
        max_pattern_size=3,
        min_pattern_size=2,
        min_count=4,
    )


def test_parse_repetition_detection_cli_accepts_empty_as_disabled():
    assert _parse_repetition_detection_cli("") is None
    assert _parse_repetition_detection_cli("None") is None


def test_coerce_repetition_detection_validates_invalid_values():
    with pytest.raises(ValueError, match="min_count must be >= 2"):
        _coerce_repetition_detection(
            {
                "max_pattern_size": 4,
                "min_pattern_size": 1,
                "min_count": 1,
            }
        )


def test_resolve_repetition_detection_prefers_request_value():
    request_params = RepetitionDetectionParams(
        max_pattern_size=2,
        min_pattern_size=1,
        min_count=2,
    )
    default_params = {
        "repetition_detection": {
            "max_pattern_size": 4,
            "min_pattern_size": 1,
            "min_count": 3,
        }
    }

    assert _resolve_repetition_detection(request_params, default_params) is request_params


def test_resolve_repetition_detection_uses_default_sampling_params():
    params = _resolve_repetition_detection(
        None,
        {
            "repetition_detection": {
                "max_pattern_size": 4,
                "min_pattern_size": 2,
                "min_count": 3,
            }
        },
    )

    assert params == RepetitionDetectionParams(
        max_pattern_size=4,
        min_pattern_size=2,
        min_count=3,
    )


def test_check_sequence_repetition_detects_repeated_ngram_suffix():
    params = RepetitionDetectionParams(
        max_pattern_size=2,
        min_pattern_size=2,
        min_count=3,
    )

    assert check_sequence_repetition([1, 2, 1, 2, 1, 2], params)


def test_check_sequence_repetition_ignores_non_repeating_suffix():
    params = RepetitionDetectionParams(
        max_pattern_size=2,
        min_pattern_size=1,
        min_count=3,
    )

    assert not check_sequence_repetition([1, 2, 1, 2, 1, 3], params)
