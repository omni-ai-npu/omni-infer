# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -*- coding: utf-8 -*-
from typing import Optional

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer
from vllm.v1.engine.detokenizer import IncrementalDetokenizer

logger = init_logger(__name__)

COT_THINK_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"
SLIDING_WINDOW_SIZE = -16

origin_from_new_request = IncrementalDetokenizer.from_new_request
origin_init = BaseIncrementalDetokenizer.__init__


def __init__(self, request: EngineCoreRequest):
    origin_init(self, request)
    # Adapt: the symbol of COT end
    self._cot_end_symbol = False


@classmethod
def from_new_request(
    cls,
    tokenizer: Optional[AnyTokenizer],
    request: EngineCoreRequest
) -> "IncrementalDetokenizer":
    incremental_detokenizer: IncrementalDetokenizer = origin_from_new_request(tokenizer, request)

    # Adapt:Init COT token
    if tokenizer is not None:
        incremental_detokenizer.set_cot_token_ids(tokenizer)
    return incremental_detokenizer


def set_cot_token_ids(self, tokenizer: Optional[AnyTokenizer]):
    vocab = tokenizer.get_vocab()
    self.cot_think_end_token_id = vocab.get(COT_THINK_END_TOKEN)


def is_cot_end(self) -> bool:
    """
    Determine whether the current output has completed the output of the chain of thought.
    """
    if self._cot_end_symbol:
        return True

    if len(self.token_ids) <= 1:
        return False

    cot_end = False

    # but we currently cannot obtain the vllm_config within the detokenizer process.
    check_token_ids = self.token_ids[SLIDING_WINDOW_SIZE:]

    # COT ends with "</think>"
    if self.cot_think_end_token_id in check_token_ids:
        cot_end = True
    if cot_end:
        self._cot_end_symbol = True

    return self._cot_end_symbol


def get_next_output_text(self, finished: bool, delta: bool) -> str:
    """If delta is True, only new text since the last call to
    this method is returned"""
    # We return the full output text if the sequence is finished.
    thinking = not self.is_cot_end()
    is_thinking_end_chunk = COT_THINK_END_TOKEN in self.output_text[self._last_output_text_offset:]
    is_DSML_chunk = DSML_TOKEN in self.output_text[self._last_output_text_offset:]
    buffer_length = 0 if finished or thinking or is_thinking_end_chunk or is_DSML_chunk else self.stop_buffer_length
    if not delta:
        return self.output_text[:-buffer_length] if buffer_length else (
            self.output_text)
    length = len(self.output_text) - buffer_length
    last_offset = self._last_output_text_offset
    if last_offset < length:
        self._last_output_text_offset = length
        return self.output_text[last_offset:length]
    return ""


def patch_thinking_bug_fix():
    from vllm.v1.engine import detokenizer
    detokenizer.IncrementalDetokenizer.from_new_request = from_new_request
    detokenizer.BaseIncrementalDetokenizer.__init__ = __init__
    detokenizer.BaseIncrementalDetokenizer.get_next_output_text = get_next_output_text
    detokenizer.BaseIncrementalDetokenizer.is_cot_end = is_cot_end
    detokenizer.BaseIncrementalDetokenizer.set_cot_token_ids = set_cot_token_ids
