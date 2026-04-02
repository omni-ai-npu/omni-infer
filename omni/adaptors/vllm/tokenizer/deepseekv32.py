# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from transformers import BatchEncoding
from typing import Optional
from vllm.transformers_utils.tokenizer_base import TokenizerBase
from vllm.logger import init_logger

from .deepseek_v32_encoding import encode_messages
from .hf import HfTokenizer

logger = init_logger(__name__)


class DeepseekV32Tokenizer(HfTokenizer):
    def __init__(self, tokenizer: TokenizerBase):
        self.tokenizer = tokenizer
        self.name_or_path = (tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "")
        self._added_vocab = self.tokenizer.get_added_vocab()
        self._added_vocab_size = len(self._added_vocab)

    @classmethod
    def validate_messages(cls, messages):
        try:
            assistant_without_prefix = len(messages) > 0 and messages[-1].get("role") == "assistant" and not (isinstance( messages[-1].get("prefix"), bool) and messages[-1].get("prefix"))
            if assistant_without_prefix:
                return False, f"Invalid consecutive assistant message at message index {len(messages) - 1}"
            for index in range(len(messages)):
                msg = messages[index]
                role = msg.get("role")
                content = msg.get("content")

                if role not in ["system", "developer", "user", "tool","assistant"]:
                    return False, f"Unkown role: {role}. Role just can be [system, developer, user, tool,assistant]"

            return True, ""
        except Exception as e:
            return True, ""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TokenizerBase":
        tokenizer = super().from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )
        return DeepseekV32Tokenizer(tokenizer)

    def apply_chat_template(self, messages, tools=None, **kwargs):
        thinking = kwargs.get("thinking", False)
        thinking_mode = "thinking"
        if not thinking:
            thinking_mode = "chat"

        conversation = kwargs.get("conversation", messages)

        try:
            if isinstance(conversation, (list, tuple)) and (
                isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
            ):
                conversation = conversation
        except Exception as e:
            # Log and report any library-related exceptions for further investigation.
            logger.exception(
                "An error occurred in `transformers` while applying chat template")
            raise ValueError from e

        messages = conversation.copy()
        drop_thinking = True
        if tools is not None and len(tools) > 0:
            messages.insert(0, {"role": "system"})
            messages[0]["tools"] = tools
            drop_thinking = False

        encode_config = dict(thinking_mode=thinking_mode, drop_thinking=drop_thinking)
        prompt_str = encode_messages(messages, **encode_config)  # type: ignore
        if len(messages) > 0 and messages[-1].get("role") == "system":
            if thinking_mode == "thinking":
                prompt_str += "<｜Assistant｜> <think>"
            else:
                prompt_str += "<｜Assistant｜>"
        return prompt_str

    def num_special_tokens_to_add(self) -> int:
        return len(self.encode(""))

    @property
    def all_special_tokens(self) -> list[str]:
        return self.tokenizer.all_special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self.tokenizer.all_special_ids

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def is_fast(self) -> bool:
        return self.tokenizer.is_fast

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def max_token_id(self) -> int:
        raise NotImplementedError()

    @property
    def truncation_side(self) -> str:
        return self.tokenizer.truncation_side

    @property
    def all_special_tokens_extended(self) -> list[str]:
        return []

    @property
    def pad_token(self) -> str:
        raise NotImplementedError()

    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        # </think> is an added token in DeepseekV32 tokenizer
        return self.vocab_size + self._added_vocab_size

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> "BatchEncoding":
        return self.tokenizer(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_added_vocab(self) -> dict[str, int]:
        return self._added_vocab.copy()

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self.tokenizer.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            ids, skip_special_tokens=skip_special_tokens
        )

    def encode_one(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> list[int]:
        input_ids = self.encode(text)
        if truncation:
            input_ids = input_ids[:max_length]
        return input_ids
