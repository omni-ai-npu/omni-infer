# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -*- coding: utf-8 -*-

"""Sequence and its related classes."""
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer

logger = init_logger(__name__)

QWEN_3_THINK_START_TOKEN_ID = 151667  # <think>
QWEN_3_THINK_END_TOKEN_ID = 151668  # </think>
QWEN_3_AFTER_THINK_END_TOKEN_ID1 = 271  # \n\n
QWEN_3_AFTER_THINK_END_TOKEN_ID2 = 9612  # \n

DEEPSEEK_R1_THINK_START_TOKEN_ID = 128798 # <think>
DEEPSEEK_R1_THINK_END_TOKEN_ID = 128799 # </think>
DEEPSEEK_R1_AFTER_THINK_END_TOKEN_ID1 = 201 # \n
DEEPSEEK_R1_AFTER_THINK_END_TOKEN_ID2 = 271 # \n\n

SLIDING_WINDOW_SIZE = -16

BaseIncrementalDetokenizer._cot_end_symbol = False
BaseIncrementalDetokenizer._check_len = None


def check_len(self):
    vllm_config = get_current_vllm_config()
    schedulerConfig = vllm_config.scheduler_config
    if not self._check_len:
        self._check_len = schedulerConfig.num_scheduler_steps * (
                    vllm_config.speculative_config.num_speculative_tokens + 1) + 1
    return self._check_len


def is_cot_end(self) -> bool:
    # Return True: do not change max_tokens,
    # Return False: change max_tokens

    if self._cot_end_symbol:
        return True

    if len(self.token_ids) <= 1:
        return False

    cot_end = False

    # but we currently cannot obtain the vllm_config within the detokenizer process.
    check_token_ids = self.token_ids[SLIDING_WINDOW_SIZE:]

    # Qwen 3 output ends with "</think>\n\n" or "</think>\n"
    if QWEN_3_THINK_END_TOKEN_ID in check_token_ids:
        check_index = check_token_ids.index(QWEN_3_THINK_END_TOKEN_ID) + 1
        if check_index < len(check_token_ids) and check_token_ids[check_index] in [
            QWEN_3_AFTER_THINK_END_TOKEN_ID1, QWEN_3_AFTER_THINK_END_TOKEN_ID2
        ]:
            cot_end = True

    # Deepseek output ends with "</think>\n" or "</think>\n\n"
    if DEEPSEEK_R1_THINK_END_TOKEN_ID in check_token_ids:
        check_index = check_token_ids.index(DEEPSEEK_R1_THINK_END_TOKEN_ID) + 1
        if check_index < len(check_token_ids) and check_token_ids[check_index] in [
            DEEPSEEK_R1_AFTER_THINK_END_TOKEN_ID1, DEEPSEEK_R1_AFTER_THINK_END_TOKEN_ID2
        ]:
            cot_end = True


    if cot_end:
        self._cot_end_symbol = True

    return self._cot_end_symbol


def get_next_output_text(self, finished: bool, delta: bool) -> str:
    """If delta is True, only new text since the last call to
    this method is returned"""
    # We return the full output text if the sequence is finished.
    thinking = not self.is_cot_end()
    is_thinking_end_chunk = "</think>" in self.output_text[self._last_output_text_offset:]
    buffer_length = 0 if finished or thinking or is_thinking_end_chunk else self.stop_buffer_length
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

    detokenizer.BaseIncrementalDetokenizer.get_next_output_text = get_next_output_text
    detokenizer.BaseIncrementalDetokenizer.is_cot_end = is_cot_end
    detokenizer.BaseIncrementalDetokenizer.check_len = check_len
