# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from openai_harmony import (Author, ChannelConfig, Conversation,
                            DeveloperContent, HarmonyEncodingName, Message,
                            ReasoningEffort, Role, StreamableParser,
                            SystemContent, TextContent, ToolDescription,
                            load_harmony_encoding)
_harmony_encoding = None
logger = init_logger(__name__)

def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding

def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)

def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        parser.process(token_id)
    return parser

def parse_chat_output(
        token_ids: Sequence[int]) -> tuple[Optional[str], Optional[str], bool]:
    parser = parse_output_into_messages(token_ids)
    output_msgs = parser.messages
    is_tool_call = False  # TODO: update this when tool call is supported
    if len(output_msgs) == 0:
        # The generation has stopped during reasoning.
        reasoning_content = parser.current_content
        final_content = None
    elif len(output_msgs) == 1:
        # The generation has stopped during final message.
        reasoning_content = output_msgs[0].content[0].text
        final_content = parser.current_content
    else:
        reasoning_msg = output_msgs[:-1]
        final_msg = output_msgs[-1]
        reasoning_content = "\n".join(
            [msg.content[0].text for msg in reasoning_msg])
        final_content = final_msg.content[0].text
    return reasoning_content, final_content, is_tool_call

@ReasoningParserManager.register_module("openai_gptoss")
class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GptOss model.

    The GptOss model uses harmony to extract reasoning content and this parser
    is only used for detecting the end of the reasoning content.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.reasoning_end_token_ids = self.model_tokenizer.encode(
            "<|start|>assistant<|channel|>final<|message|>")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        end_token_ids = self.reasoning_end_token_ids
        assert len(end_token_ids) > 0, "reasoning_end_token_ids is empty"
        # Check if the end sequence is present in the input_ids.
        # We search from the end of input_ids to find the last match.
        for i in range(len(input_ids) - len(end_token_ids), -1, -1):
            if input_ids[i:i + len(end_token_ids)] == end_token_ids:
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        _, content, _ = parse_chat_output(input_ids)
        if content is None:
            return []
        return self.model_tokenizer.encode(content)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        prev_reasoning, prev_content, _ = parse_chat_output(
            list(previous_token_ids))
        cur_reasoning, cur_content, _ = parse_chat_output(
            list(current_token_ids))
        reasoning_delta = None
        content_delta = None
        if cur_reasoning is not None:
            prev_r = prev_reasoning or ""
            if cur_reasoning.startswith(prev_r):
                reasoning_delta = cur_reasoning[len(prev_r):] or None
            else:
                reasoning_delta = cur_reasoning
        if cur_content is not None:
            prev_c = prev_content or ""
            if cur_content.startswith(prev_c):
                content_delta = cur_content[len(prev_c):] or None
            else:
                content_delta = cur_content
        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning_content=reasoning_delta,
                            content=content_delta)

    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], Optional[str]]:
        raise NotImplementedError(
            "gpt-oss has a special branch for parsing reasoning in non-streaming mode. This method shouldn't be used."  # noqa: E501
        )