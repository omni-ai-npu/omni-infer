# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.


import os
import unittest
from unittest.mock import MagicMock, patch
from omni_npu.v1.parsers import PanguReasoningParser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.protocol import DeltaMessage

# Enable the implicit-thinking-end behavior introduced in 9c14d17e for the
# tests below that exercise that path. Default (env unset) matches pre-9c14d17e.
_TOOL_CALL_ENDS_THINKING_ON = patch.dict(
    os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}
)


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

    @_TOOL_CALL_ENDS_THINKING_ON
    def test_extract_reasoning_tool_call_start_ends_reasoning(self):
        """case: <|tool_call_start|> without </think> implicitly ends reasoning."""
        self.mock_tokenizer.get_vocab.return_value = {
            "<think>": 100,
            "</think>": 101,
            "<|tool_call_start|>": 104,
        }
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "<think>I should call a tool.<|tool_call_start|>[{...}]"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "I should call a tool.")
        # Marker is kept at the head of content for the tool parser.
        self.assertEqual(content, "<|tool_call_start|>[{...}]")

    @_TOOL_CALL_ENDS_THINKING_ON
    def test_extract_reasoning_explicit_end_wins_over_tool_call_start(self):
        """case: </think> takes priority over <|tool_call_start|>."""
        self.mock_tokenizer.get_vocab.return_value = {
            "<think>": 100,
            "</think>": 101,
            "<|tool_call_start|>": 104,
        }
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "<think>reason</think>plain answer<|tool_call_start|>x"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "reason")
        self.assertEqual(content, "plain answer<|tool_call_start|>x")

    @_TOOL_CALL_ENDS_THINKING_ON
    def test_extract_reasoning_tool_call_start_unused11_fallback(self):
        """case: [unused11] fallback used when <|tool_call_start|> is absent."""
        self.mock_tokenizer.get_vocab.return_value = {
            "[unused16]": 102,
            "[unused17]": 103,
            "[unused11]": 110,
        }
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "[unused16]thinking[unused11]tool payload"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "thinking")
        self.assertEqual(content, "[unused11]tool payload")


class TestPanguReasoningParserExtractReasoningStreaming(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.vocab = {
            "<think>": 10,
            "</think>": 11,
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


class TestPanguReasoningParserStreamingImplicitEnd(unittest.TestCase):
    """Streaming behavior when <|tool_call_start|> implicitly ends reasoning.
    Opted in via PANGU_TOOL_CALL_ENDS_THINKING=1."""

    def setUp(self):
        # Env var must be set BEFORE parser construction so the cached
        # tool_call_start_token_id picks up the marker. A class-level
        # @patch.dict decorator would only wrap test methods, not setUp.
        env_patcher = patch.dict(
            os.environ, {"PANGU_TOOL_CALL_ENDS_THINKING": "1"}
        )
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

        self.mock_tokenizer = MagicMock()
        self.vocab = {
            "<think>": 10,
            "</think>": 11,
            "<|tool_call_start|>": 12,
            "Hello": 20,
        }
        self.mock_tokenizer.get_vocab.return_value = self.vocab
        self.mock_tokenizer.tokenizer = self.mock_tokenizer
        self.parser = PanguReasoningParser(
            self.mock_tokenizer,
            chat_template_kwargs={"think": True},
        )

    def test_tool_call_start_alone_in_delta(self):
        """A single <|tool_call_start|> token after some reasoning."""
        # previous_token_ids already contains <think> + a reasoning token
        result = self.parser.extract_reasoning_streaming(
            previous_text="<think>some reasoning",
            current_text="<think>some reasoning<|tool_call_start|>",
            delta_text="<|tool_call_start|>",
            previous_token_ids=[10, 20],
            current_token_ids=[10, 20, 12],
            delta_token_ids=[12],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning, "")
        self.assertEqual(result.content, "<|tool_call_start|>")

    def test_tool_call_start_with_leading_reasoning_text_in_delta(self):
        """Multi-token delta: trailing reasoning then <|tool_call_start|>."""
        result = self.parser.extract_reasoning_streaming(
            previous_text="<think>",
            current_text="<think>last bit<|tool_call_start|>",
            delta_text="last bit<|tool_call_start|>",
            previous_token_ids=[10],
            current_token_ids=[10, 20, 12],
            delta_token_ids=[20, 12],
        )
        self.assertEqual(result.reasoning, "last bit")
        self.assertEqual(result.content, "<|tool_call_start|>")

    def test_tool_call_start_after_already_implicit_end(self):
        """Once the marker has fired, subsequent deltas are pure content."""
        result = self.parser.extract_reasoning_streaming(
            previous_text="<think>r<|tool_call_start|>",
            current_text="<think>r<|tool_call_start|>[{",
            delta_text="[{",
            previous_token_ids=[10, 20, 12],
            current_token_ids=[10, 20, 12, 30],
            delta_token_ids=[30],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.reasoning)
        self.assertEqual(result.content, "[{")

    def test_explicit_end_wins_when_both_in_delta(self):
        """If </think> is present in delta, the existing path handles it."""
        result = self.parser.extract_reasoning_streaming(
            previous_text="<think>r",
            current_text="<think>r</think>ans<|tool_call_start|>",
            delta_text="</think>ans<|tool_call_start|>",
            previous_token_ids=[10, 20],
            current_token_ids=[10, 20, 11, 21, 12],
            delta_token_ids=[11, 21, 12],
        )
        # Falls through to the </think> branch; tool-call-start logic skipped.
        self.assertEqual(result.reasoning, "")
        self.assertEqual(result.content, "ans<|tool_call_start|>")

    def test_is_reasoning_end_via_tool_call_start(self):
        """is_reasoning_end fires once when the tool-call-start marker is seen."""
        self.parser.delta_token_ids = [12]
        input_ids = [10, 20, 12]
        self.assertTrue(self.parser.is_reasoning_end(input_ids))
        self.assertEqual(self.parser.is_reasoning_end_count, 1)
        # Subsequent call should not re-fire.
        self.assertFalse(self.parser.is_reasoning_end(input_ids))


class TestPanguReasoningParserToolCallEndsThinkingEnvVar(unittest.TestCase):
    """Env var PANGU_TOOL_CALL_ENDS_THINKING controls the implicit-end
    behavior. Default (unset) matches pre-9c14d17e — marker is NOT used."""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {
            "<think>": 10,
            "</think>": 11,
            "<|tool_call_start|>": 12,
        }
        self.mock_tokenizer.tokenizer = self.mock_tokenizer
        # Ensure no inherited env var leaks into these tests.
        os.environ.pop("PANGU_TOOL_CALL_ENDS_THINKING", None)

    def test_default_property_and_id_are_none(self):
        parser = PanguReasoningParser(self.mock_tokenizer)
        self.assertIsNone(parser.tool_call_start_token)
        self.assertIsNone(parser.tool_call_start_token_id)

    def test_default_is_reasoning_end_ignores_marker(self):
        parser = PanguReasoningParser(self.mock_tokenizer)
        parser.delta_token_ids = [12]
        self.assertFalse(parser.is_reasoning_end([10, 20, 12]))
        self.assertEqual(parser.is_reasoning_end_count, 0)

    def test_default_extract_reasoning_does_not_split_on_marker(self):
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)
        reasoning, content = parser.extract_reasoning(
            "<think>reason<|tool_call_start|>[{...}]", request
        )
        self.assertEqual(reasoning, "reason<|tool_call_start|>[{...}]")
        self.assertIsNone(content)

    @_TOOL_CALL_ENDS_THINKING_ON
    def test_opt_in_enables_marker(self):
        parser = PanguReasoningParser(self.mock_tokenizer)
        self.assertEqual(parser.tool_call_start_token, "<|tool_call_start|>")
        self.assertEqual(parser.tool_call_start_token_id, 12)


if __name__ == '__main__':
    unittest.main()
