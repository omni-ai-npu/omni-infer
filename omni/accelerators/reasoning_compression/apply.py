import os

from omni.accelerators.reasoning_compression.config import ThinkCompressDict
from vllm.logger import init_logger

# Configure logging
logger = init_logger(__name__)
def init_reasoner_compression_configs(vllm_config):
    ThinkCompressDict.reasoner_early_think_stopping_enabled = False
    if vllm_config is None or vllm_config.reasoning_config is None:
        logger.warning(f"reason config does not exist, disable reasoning compression")
        return

    reasoning_config = vllm_config.reasoning_config
    ThinkCompressDict.thinking_token_budget = reasoning_config.thinking_token_budget
    ThinkCompressDict.think_start_token_ids = reasoning_config.think_start_token_ids
    ThinkCompressDict.think_end_token_ids = reasoning_config.think_end_token_ids
    ThinkCompressDict.think_start_str = reasoning_config.think_start_str
    ThinkCompressDict.think_end_str = reasoning_config.think_end_str

    tokenizer_path = vllm_config.model_config.tokenizer
    if tokenizer_path is None or not os.path.exists(tokenizer_path):
        logger.warning(f"Tokenizer path does not exist, disable reasoning compression")
        return

    tokenizer = get_tokenizer(tokenizer_path)
    if tokenizer is None:
        logger.warning(f"tokenizer does not exist, disable reasoning compression")
        return

    # tokenize str to token ids
    if not ThinkCompressDict.think_start_str is None and ThinkCompressDict.think_start_str != "":
        ThinkCompressDict.think_start_token_ids = tuple(tokenize_without_special_tokens(tokenizer, ThinkCompressDict.think_start_str))
    elif not ThinkCompressDict.think_start_token_ids is None:
        start_token_ids = ThinkCompressDict.think_start_token_ids
        ThinkCompressDict.think_start_token_ids = start_token_ids if isinstance(start_token_ids, list) else eval(start_token_ids)
        ThinkCompressDict.think_start_token_ids = tuple(ThinkCompressDict.think_start_token_ids)

    if not ThinkCompressDict.think_end_str is None and ThinkCompressDict.think_end_str != "":
        ThinkCompressDict.think_end_token_ids = tokenize_without_special_tokens(tokenizer, ThinkCompressDict.think_end_str)
    elif not ThinkCompressDict.think_end_token_ids is None:
        end_token_ids = ThinkCompressDict.think_end_token_ids
        ThinkCompressDict.think_end_token_ids = end_token_ids if isinstance(end_token_ids, list) else eval(end_token_ids)

    ThinkCompressDict.thinking_token_budget = int(ThinkCompressDict.thinking_token_budget)
    if ThinkCompressDict.thinking_token_budget is None or ThinkCompressDict.thinking_token_budget <= 0:
        raise RuntimeError("thinking_token_budget is required or greater than 0.")

    if ThinkCompressDict.think_end_token_ids is None or len(ThinkCompressDict.think_end_token_ids) == 0:
        raise RuntimeError("think_end_token_ids is required. please add think_end_str or think_end_token_ids config.")

    ThinkCompressDict.reasoner_early_think_stopping_enabled = True

def get_tokenizer(tokenizer_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

def tokenize_without_special_tokens(tokenizer, text):
    """Use tokenizer to get tokenized ids (without special tokens)"""
    return tokenizer(text, add_special_tokens=False).input_ids
