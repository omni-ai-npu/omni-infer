# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.


import os
import unittest
from unittest.mock import MagicMock, patch
from omni_npu.v1.parsers import PanguReasoningParser
from omni_npu.v1.parsers._streaming_relay import (
    reattach_reasoning_to,
    reset_for_tests,
    stash_reasoning_from,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage


class TestPanguReasoningParserExtractReasoning(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {
            "<think>": 100,
            "</think>": 101,
            "[unused16]": 102,
            "[unused17]": 103,
        }
        self.mock_tokenizer.tokenizer = self.mock_tokenizer

    def test_extract_reasoning_with_think_tags(self):
        """case: special token is <think> and </think>"""
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "<think>正在思考如何写代码...</think>这是你的代码：print('hello')"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "正在思考如何写代码...")
        self.assertEqual(content, "这是你的代码：print('hello')")

    def test_extract_reasoning_with_unused_tags(self):
        """case: special token is [unused16] and [unused17]"""
        self.mock_tokenizer.get_vocab.return_value = {
            "[unused16]": 102,
            "[unused17]": 103
        }

        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "[unused16]分析用户需求中...[unused17]完成提取。"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "分析用户需求中...")
        self.assertEqual(content, "完成提取。")

    def test_extract_reasoning_with_only_reasoning(self):
        """case: only reasoning"""
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "<think>思考到一半"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "思考到一半")
        self.assertIsNone(content)

    def test_extract_reasoning_with_empty_content(self):
        """case: content is empty"""
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "<think>思考完了</think>"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "思考完了")
        self.assertIsNone(content)
        
    def test_extract_content_with_no_think(self):
        """case: extract content when think is False"""
        self.mock_tokenizer.get_vocab.return_value = {
            "<think>": 100,
            "</think>": 101,
        }
        chat_template_kwargs={"think": False}
        parser = PanguReasoningParser(self.mock_tokenizer,
                                      chat_template_kwargs=chat_template_kwargs
                                      )
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "PanguV2快思考不会生成/think，快思考完成提取。"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(content, model_output)

    def test_extract_reasoning_implicit_tool_call_end_enabled(self):
        """case: tool-call-start marker implicitly ends reasoning."""
        self.mock_tokenizer.get_vocab.return_value = {
            "<think>": 100,
            "</think>": 101,
            "<|tool_call_start|>": 104,
        }
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)
            request = MagicMock(spec=ChatCompletionRequest)
            model_output = (
                "<think>需要调用工具"
                "<|tool_call_start|>[{\"name\":\"get_weather\",\"arguments\":{}}]"
            )
            reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "需要调用工具")
        self.assertEqual(
            content,
            "<|tool_call_start|>[{\"name\":\"get_weather\",\"arguments\":{}}]",
        )

    def test_tool_call_start_token_uses_unused11_when_enabled(self):
        """case: implicit-end marker falls back to [unused11]."""
        self.mock_tokenizer.get_vocab.return_value = {
            "<think>": 100,
            "</think>": 101,
            "[unused11]": 104,
        }
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)
            self.assertEqual(parser.tool_call_start_token, "[unused11]")
            self.assertEqual(parser.tool_call_start_token_id, 104)

    def test_tool_call_start_token_returns_none_without_marker(self):
        """case: env enabled but tokenizer has no supported marker."""
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)

        self.assertIsNone(parser.tool_call_start_token)
        self.assertIsNone(parser.tool_call_start_token_id)


class TestPanguReasoningParserExtractReasoningStreaming(unittest.TestCase):
    def setUp(self):
        reset_for_tests()
        self.mock_tokenizer = MagicMock()
        self.vocab = {
            "<think>": 10,
            "</think>": 11,
            "<|tool_call_start|>": 12,
            "Hello": 20,
            "World": 21
        }
        self.mock_tokenizer.get_vocab.return_value = self.vocab
        self.mock_tokenizer.tokenizer = self.mock_tokenizer
        chat_template_kwargs={"think": True}
        self.parser = PanguReasoningParser(self.mock_tokenizer, 
                                           chat_template_kwargs=chat_template_kwargs
                                           )

    def test_is_reasoning_end(self):
        """测试推理结束标记的计数逻辑"""
        self.parser.delta_token_ids = [11]  # 模拟当前 delta 包含结束符

        # 第一次检测到结束符，应该返回 True
        input_ids = [10, 20, 11]
        self.assertTrue(self.parser.is_reasoning_end(input_ids))
        self.assertEqual(self.parser.is_reasoning_end_count, 1)

        # 第二次调用（模拟重复触发或其他逻辑），计数器变为 2，应返回 False
        self.assertFalse(self.parser.is_reasoning_end(input_ids))
        self.assertEqual(self.parser.is_reasoning_end_count, 2)
        
        self.parser.delta_token_ids = [12]
        input_ids = [10, 20, 11]
        self.assertTrue(self.parser.is_reasoning_end(input_ids))

    def test_is_reasoning_end_by_tool_start_token(self):
        """case: tool-call-start token is treated as implicit end when enabled."""
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)
        parser.delta_token_ids = [12]

        self.assertTrue(parser.is_reasoning_end([10, 20, 12]))
        self.assertFalse(parser.is_reasoning_end([10, 20, 12]))

    def test_is_reasoning_end_by_last_tool_start_token(self):
        """case: last token can end reasoning even when it is not in delta."""
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)
        parser.delta_token_ids = [20]

        self.assertTrue(parser.is_reasoning_end([10, 20, 12]))

    def test_should_implicit_end_ignores_delta_with_explicit_end(self):
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)

        result = parser._should_implicit_end_in_delta(
            tool_start_id=12,
            delta_token_ids=[11, 12],
            previous_token_ids=[],
        )

        self.assertFalse(result)

    def test_should_implicit_end_ignores_delta_without_tool_start(self):
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)

        result = parser._should_implicit_end_in_delta(
            tool_start_id=12,
            delta_token_ids=[20],
            previous_token_ids=[],
        )

        self.assertFalse(result)

    def test_should_implicit_end_ignores_when_explicit_end_seen_before(self):
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)

        result = parser._should_implicit_end_in_delta(
            tool_start_id=12,
            delta_token_ids=[12],
            previous_token_ids=[10, 11],
        )

        self.assertFalse(result)

    def test_is_reasoning_end_returns_false_without_end_markers(self):
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)
        parser.delta_token_ids = [20]

        self.assertFalse(parser.is_reasoning_end([10, 20]))

    def test_is_reasoning_end_empty_input_ids_returns_false(self):
        """Empty input_ids must not IndexError on input_ids[-1]."""
        self.parser.delta_token_ids = []
        self.assertFalse(self.parser.is_reasoning_end([]))

    def test_extract_reasoning_streaming_with_multi_token(self):
        """测试起始标签和推理文本在同一个 chunk 中的场景: '<think>Hello'"""
        # 模拟输入参数
        # previous: 空
        # delta: '<think>Hello' (假设对应 token IDs [10, 20])
        previous_text = ""
        current_text = "<think>Hello"
        delta_text = "<think>Hello"
        previous_token_ids = []
        current_token_ids = [10, 20]
        delta_token_ids = [10, 20]

        result = self.parser.extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids
        )

        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning, "Hello")
        self.assertIsNone(result.content)

    def test_extract_reasoning_streaming_with_normal_reasoning(self):
        """测试正常的推理过程流（标签已在之前出现过）"""
        # 模拟之前已经有了 <think>
        previous_text = "<think>"
        current_text = "<think>Thinking..."
        delta_text = "Thinking..."
        previous_token_ids = [10]
        current_token_ids = [10, 25, 26]  # 假设 25, 26 是 Thinking...
        delta_token_ids = [25, 26]

        result = self.parser.extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids
        )

        self.assertEqual(result.reasoning, "Thinking...")

    def test_extract_reasoning_streaming_whit_end(self):
        """测试推理结束的流场景"""
        previous_text = "<think>Done"
        current_text = "<think>Done</think>Answer"
        delta_text = "</think>Answer"
        previous_token_ids = [10, 30]
        current_token_ids = [10, 30, 11, 40]
        delta_token_ids = [11, 40]

        result = self.parser.extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids
        )

        # 此时应分别提取出最后一段推理和起始正文
        self.assertEqual(result.reasoning, "")  # </think> 前面没有新推理
        self.assertEqual(result.content, "Answer")

    def test_extract_reasoning_streaming_implicit_tool_call_end_same_delta(self):
        """case: reasoning and tool-call-start marker arrive in the same delta."""
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)
            result = parser.extract_reasoning_streaming(
                previous_text="",
                current_text="<think>Need tool<|tool_call_start|>",
                delta_text="<think>Need tool<|tool_call_start|>",
                previous_token_ids=[],
                current_token_ids=[10, 20, 12],
                delta_token_ids=[10, 20, 12],
            )

        self.assertEqual(result.reasoning, "Need tool")
        self.assertEqual(result.content, "<|tool_call_start|>")

        relayed = reattach_reasoning_to(None)
        self.assertEqual(relayed.reasoning, "Need tool")
        self.assertEqual(relayed.reasoning_content, "Need tool")

    def test_extract_reasoning_streaming_after_implicit_tool_call_end(self):
        """case: after implicit end, later chunks are emitted as content."""
        with patch.dict(os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}):
            parser = PanguReasoningParser(self.mock_tokenizer)
            result = parser.extract_reasoning_streaming(
                previous_text="<think>Need tool<|tool_call_start|>",
                current_text="<think>Need tool<|tool_call_start|>[",
                delta_text="[",
                previous_token_ids=[10, 20, 12],
                current_token_ids=[10, 20, 12, 30],
                delta_token_ids=[30],
            )

        self.assertIsNone(result.reasoning)
        self.assertEqual(result.content, "[")
    
    def test_extract_content_with_no_think(self):
        """测试推理结束的流场景"""
        chat_template_kwargs={"think": False}
        self.parser = PanguReasoningParser(self.mock_tokenizer, 
                                           chat_template_kwargs=chat_template_kwargs,
                                           )
        
        model_deltas = [
            "PanguV2",
            "快思考",
            "不生成",
            "/think"
        ]
        
        previous_text = ""
        previous_token_ids = []
        delta_token_ids = []
        current_token_ids = []
        for delta_text in model_deltas:
            current_text = previous_text + delta_text
            
            delta_message = self.parser.extract_reasoning_streaming(
                previous_text, current_text, delta_text,
                previous_token_ids, current_token_ids, delta_token_ids
            )

            self.assertEqual(delta_message.reasoning, None) 
            self.assertEqual(delta_message.content, delta_text)


class TestStreamingReasoningRelay(unittest.TestCase):
    def setUp(self):
        reset_for_tests()

    def test_stash_ignores_none_and_empty_reasoning(self):
        stash_reasoning_from(None)
        self.assertIsNone(reattach_reasoning_to(None))

        stash_reasoning_from(DeltaMessage(reasoning=""))
        self.assertIsNone(reattach_reasoning_to(None))

    def test_reattach_creates_delta_when_tool_parser_returned_none(self):
        stash_reasoning_from(DeltaMessage(reasoning="Thinking..."))

        result = reattach_reasoning_to(None)

        self.assertEqual(result.reasoning, "Thinking...")
        self.assertEqual(result.reasoning_content, "Thinking...")
        self.assertIsNone(reattach_reasoning_to(None))

    def test_reattach_sets_reasoning_content_on_existing_delta(self):
        stash_reasoning_from(DeltaMessage(reasoning="Thinking..."))
        delta = DeltaMessage(content="tool")

        result = reattach_reasoning_to(delta)

        self.assertIs(result, delta)
        self.assertEqual(result.reasoning, "Thinking...")
        self.assertEqual(result.reasoning_content, "Thinking...")

    def test_reattach_preserves_existing_reasoning(self):
        stash_reasoning_from(DeltaMessage(reasoning="Pending"))
        delta = DeltaMessage(reasoning="Existing")

        result = reattach_reasoning_to(delta)

        self.assertIs(result, delta)
        self.assertEqual(result.reasoning, "Existing")

if __name__ == '__main__':
    unittest.main()
