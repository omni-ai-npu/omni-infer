# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded DecodeStream prefill for FastIncrementalDetokenizer.

v0.25.1 passes the full prompt to ``DecodeStream``. On long prompts, the first
decode ``step()`` does O(prompt_len) work and slows down TTFT.

Community PR https://github.com/vllm-project/vllm/pull/51281 proposes
``prompt[-32:]`` tail priming; reviewer asked for UTF-8 safe boundary probing
(v0.14-style) and the author has not responded. This patch adds that check:
probe suffix 4–32 tokens, pick the shortest decode without U+FFFD (``"�"``),
then ``DecodeStream(ids=...)``. Remove after upstream merges an equivalent.
"""

from __future__ import annotations

import tokenizers
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (
    BaseIncrementalDetokenizer,
    FastIncrementalDetokenizer,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_DETOK_SUFFIX_MIN = 4
_DETOK_SUFFIX_MAX = 32


def _find_safe_prompt_suffix_ids(
    prompt_token_ids: list[int] | None,
    tokenizer: Tokenizer,
    *,
    suffix_min: int = _DETOK_SUFFIX_MIN,
    suffix_max: int = _DETOK_SUFFIX_MAX,
) -> list[int] | None:
    if not prompt_token_ids:
        return None

    prompt_suffix = prompt_token_ids
    prompt_len = len(prompt_suffix)
    if prompt_len > suffix_min:
        upper = min(prompt_len + 1, suffix_max + 1)
        for i in range(suffix_min, upper):
            suffix = prompt_token_ids[-i:]
            if "\ufffd" not in tokenizer.decode(suffix):
                prompt_suffix = suffix
                break
    return prompt_suffix


@register_patch("FastIncrementalDetokenizerSuffixPatch", FastIncrementalDetokenizer)
class FastIncrementalDetokenizerSuffixPatch(VLLMPatch):
    """Replace full-prompt DecodeStream prefill with a bounded safe suffix."""

    _attr_names_to_apply = ["__init__"]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        request: EngineCoreRequest,
    ) -> None:
        BaseIncrementalDetokenizer.__init__(self, request)

        sampling_params = request.sampling_params
        if sampling_params is None:
            raise ValueError(
                "sampling_params must not be None for FastIncrementalDetokenizer"
            )

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # patch start
        # Remove this patch after https://github.com/vllm-project/vllm/pull/51281 is merged.
        # Prime with a bounded trailing suffix instead of the full prompt.
        # Look up DecodeStream on the module so backend patches (e.g. the
        # fastokens shim that replaces ``tokenizers.decoders.DecodeStream``)
        # are honored regardless of import order.
        prime_ids = _find_safe_prompt_suffix_ids(
            request.prompt_token_ids,
            self.tokenizer,
        )
        self.stream = tokenizers.decoders.DecodeStream(
            ids=prime_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
        # patch end

        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens
        )

        if not self.spaces_between_special_tokens:
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)
            if added_token_ids is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                self.spaces_between_special_tokens = True
