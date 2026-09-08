# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for patch_detokenizer bounded DecodeStream suffix priming.

The patch replaces the v0.25.1 full-prompt ``DecodeStream`` prefill with a
bounded, UTF-8-safe trailing suffix.  The patch's own logic is fully testable
without a real tokenizer:

* ``_find_safe_prompt_suffix_ids`` only calls ``tokenizer.decode()`` and checks
  the returned string for U+FFFD, so a stub that controls which suffix lengths
  decode "unsafely" exercises the boundary-probing exactly.
* ``FastIncrementalDetokenizerSuffixPatch.__init__`` differs from stock
  v0.25.1 only in that ``DecodeStream`` is primed with ``prime_ids`` instead of
  the full ``prompt_token_ids``; ``DecodeStream`` is mocked and its ``ids``
  argument captured.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tokenizers

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

from omni_npu.vllm_patches.patches.common.patch_detokenizer import (
    FastIncrementalDetokenizerSuffixPatch,
    _DETOK_SUFFIX_MAX,
    _DETOK_SUFFIX_MIN,
    _find_safe_prompt_suffix_ids,
)


class _StubDecoder:
    """decode() stub: U+FFFD for chosen suffix lengths, clean otherwise.

    Models the byte-level-BPE behavior the patch guards against: a tail cut in
    the middle of a multi-byte character decodes to U+FFFD, and the probe must
    step forward to the first clean boundary.
    """

    def __init__(self, unsafe_lengths: set[int]) -> None:
        self._unsafe_lengths = set(unsafe_lengths)
        self.calls: list[int] = []

    def decode(self, ids: list[int]) -> str:
        self.calls.append(len(ids))
        return "\ufffd" if len(ids) in self._unsafe_lengths else "ok"


def _make_request(prompt_token_ids: list[int]) -> EngineCoreRequest:
    params = SamplingParams(
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )
    return EngineCoreRequest(
        request_id="ut-detok",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


# ---------------------------------------------------------------------------
# _find_safe_prompt_suffix_ids
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_find_safe_suffix_none_and_empty():
    assert _find_safe_prompt_suffix_ids(None, None) is None
    assert _find_safe_prompt_suffix_ids([], None) is None


@pytest.mark.unit
def test_find_safe_suffix_short_prompt_returns_full():
    ids = [1, 2, 3, 4]
    assert _find_safe_prompt_suffix_ids(ids, None) == ids


@pytest.mark.unit
def test_find_safe_suffix_probes_to_first_safe_window():
    decoder = _StubDecoder({_DETOK_SUFFIX_MIN, 5, 6})
    ids = list(range(40))

    assert _find_safe_prompt_suffix_ids(ids, decoder) == ids[-7:]
    # Probed 4, 5, 6 (unsafe), then 7 (safe) and stopped.
    assert decoder.calls == [4, 5, 6, 7]


@pytest.mark.unit
def test_find_safe_suffix_falls_back_to_full_prompt():
    # Every window in [suffix_min, suffix_max] decodes to U+FFFD: fall back to
    # the full prompt, matching v0.25.1 default behavior.
    decoder = _StubDecoder(set(range(_DETOK_SUFFIX_MIN, _DETOK_SUFFIX_MAX + 1)))
    ids = list(range(40))

    assert _find_safe_prompt_suffix_ids(ids, decoder) == ids
    assert decoder.calls == list(range(_DETOK_SUFFIX_MIN, _DETOK_SUFFIX_MAX + 1))


# ---------------------------------------------------------------------------
# FastIncrementalDetokenizerSuffixPatch.__init__
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_patch_init_primes_with_bounded_suffix(monkeypatch):
    captured: dict = {}

    def fake_decode_stream(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", fake_decode_stream)

    prompt_ids = list(range(40))
    decoder = _StubDecoder({_DETOK_SUFFIX_MIN, 5, 6})  # first safe at length 7
    tokenizer = SimpleNamespace(_tokenizer=decoder)

    detok = FastIncrementalDetokenizer.__new__(FastIncrementalDetokenizer)
    FastIncrementalDetokenizerSuffixPatch.__init__(
        detok, tokenizer, _make_request(prompt_ids)
    )

    # DecodeStream was primed with the bounded safe suffix, NOT the full prompt.
    assert captured["ids"] == prompt_ids[-7:]
    assert captured["skip_special_tokens"] is True
