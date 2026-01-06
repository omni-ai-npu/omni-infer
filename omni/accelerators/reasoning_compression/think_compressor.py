import torch
import torch.nn as nn
from vllm.sequence import SequenceData
from vllm.logger import init_logger
from omni.accelerators.reasoning_compression.config import ThinkCompressDict

# Configure logging
logger = init_logger(__name__)


class ThinkCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_parameters()

    def init_parameters(self) -> None:
        self.enable_early_stop = ThinkCompressDict.reasoner_early_think_stopping_enabled
        self.thinking_token_budget = ThinkCompressDict.thinking_token_budget
        self.think_start_token_ids = ThinkCompressDict.think_start_token_ids
        self.think_end_token_ids = ThinkCompressDict.think_end_token_ids

        # the end_token_ids is also the ids to check if token sequence end.
        self.think_end_token_ids_tags = tuple(ThinkCompressDict.think_end_token_ids)

        # if think_start_token_ids is not empty, the think len calculate should start with think_start_token_ids
        self.enforce_early_stopping = True
        if not self.think_start_token_ids is None and len(self.think_start_token_ids) > 0:
            self.enforce_early_stopping = False

        logger.info(f"enable early stop: {self.enable_early_stop}, start_token_ids: {self.think_start_token_ids}, end_token_ids: {self.think_end_token_ids}, global budget: {self.thinking_token_budget}")

    def get_think_stop_step(self):
        return self.thinking_token_budget

    def update_sampled_token_ids(self, token_ids: torch.Tensor, seq_datas) -> torch.Tensor:
        """
        For the already sampled token ids, directly update the sampling token ids based on guided decoding info
        """
        num_tokens = token_ids.shape[-1]
        if token_ids.ndim == 1:
            for req_idx, _seq_data in enumerate(seq_datas):
                target_token_id = self._get_guided_decoding_token_id(_seq_data, 0)
                if target_token_id is not None:
                    token_ids[req_idx] = target_token_id
        else:
            for req_idx, _seq_data in enumerate(seq_datas):
                # handle end of guided_decoding_token_ids for speculative decoding
                has_valid_guided_tokens = False
                # record the thinking stop position and the current possible token positions to
                # reduce overhead for speculative decoding, as multiple tokens need to be checked
                think_stop_step = self.get_think_stop_step()
                sampled_output_token_length = len(_seq_data.output_token_ids)
                max_cur_sampled_output_token_length = sampled_output_token_length + num_tokens
                for offset in range(num_tokens):
                    target_token_id = self._get_guided_decoding_token_id(
                        _seq_data, offset, think_stop_step=think_stop_step)

                    if target_token_id is not None:
                        token_ids[req_idx, offset] = target_token_id
                        # label that guided decoding for SD is enabled
                        has_valid_guided_tokens = True
                    
                    """
                    for requests with both SD and guided decoding enabled,
                    if over-length tokens over the defined guided_decoding_token_ids, 
                    set them as invalid tokens to avoid chaos in output semantics
                    """
                    if target_token_id is None:
                        if has_valid_guided_tokens:
                            token_ids[req_idx, offset] = -1
                        elif sampled_output_token_length > think_stop_step or \
                            max_cur_sampled_output_token_length < think_stop_step:
                            # if guided decoding is not enabled for all possible tokens, exit now to reduce overhead
                            break

        return token_ids

    def _get_guided_decoding_token_id(self, seq_data, offset=0, think_stop_step=None):
        if seq_data.is_think_finished:
            return None

        think_end_token_ids_in_use = self.think_end_token_ids
        output_token_ids = seq_data.output_token_ids

        if not seq_data.is_think_started:
            if self.enforce_early_stopping:
                # For backward compatibility, some old scripts do not have think start tokens
                seq_data.is_think_started = True
            else:
                if output_token_ids[-len(self.think_start_token_ids) :] == self.think_start_token_ids:
                    # When think start token is detected
                    seq_data.is_think_started = True
                    seq_data.is_think_finished = False

        if not seq_data.is_think_started:
            # When think is not started, skip
            return None

        if output_token_ids[-len(self.think_end_token_ids_tags) :] == self.think_end_token_ids_tags:
            seq_data.is_think_finished = True
            return None

        current_len = len(output_token_ids) + offset

        # Get think stop step
        if think_stop_step is None:
            think_stop_step = self.get_think_stop_step()

        if current_len < think_stop_step:
            # Reduce resource use
            return None

        # Catch over-length requests
        if current_len >= think_stop_step and current_len - think_stop_step < len(think_end_token_ids_in_use):
            tag_index = current_len - think_stop_step
            tag_idx = think_end_token_ids_in_use[tag_index]
            return tag_idx
        else:
            return None