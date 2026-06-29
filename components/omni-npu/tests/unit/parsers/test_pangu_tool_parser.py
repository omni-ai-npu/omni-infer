# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import unittest
from unittest.mock import MagicMock

from omni_npu.v1.parsers import PanguToolParser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest


class TestPanguToolParserExtractToolCalls(unittest.TestCase):
    """非流式提取测试"""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {
            "<|tool_call_start|>": 104,
            "<|tool_call_end|>": 105
        }
        self.mock_tokenizer.tokenizer = self.mock_tokenizer
        self.parser = PanguToolParser(self.mock_tokenizer)
        self.request = MagicMock(spec=ChatCompletionRequest)

    def test_extract_tool_calls_with_no_tool_call(self):
        """测试没有工具调用的普通文本"""
        model_output = "Hello, I am an AI assistant."
        res = self.parser.extract_tool_calls(model_output, self.request)
        self.assertFalse(res.tools_called)
        self.assertEqual(res.content, model_output)

    def test_extract_tool_calls_with_standard_tool_call(self):
        """测试标准单工具调用"""
        model_output = (
            "Thought: I need to check weather."
            "<|tool_call_start|>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}]<|tool_call_end|>"
        )
        res = self.parser.extract_tool_calls(model_output, self.request)

        self.assertTrue(res.tools_called)
        self.assertEqual(res.content, "Thought: I need to check weather.")
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0].function.name, "get_weather")
        self.assertEqual(json.loads(res.tool_calls[0].function.arguments), {"city": "Beijing"})

    def test_extract_tool_calls_with_single_object_tool_call(self):
        """测试非流式单对象 JSON 工具调用降级处理"""
        model_output = (
            "Thought: I need to check weather."
            "<|tool_call_start|>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}<|tool_call_end|>"
        )
        res = self.parser.extract_tool_calls(model_output, self.request)

        self.assertFalse(res.tools_called)
        self.assertEqual(res.tool_calls, [])
        self.assertEqual(model_output, res.content)

    def test_extract_tool_calls_with_missing_closing_brace(self):
        """测试非流式工具调用 JSON 尾部少右花括号时降级处理"""
        model_output = (
            "<|tool_call_start|>"
            "[{\"name\": \"send_email\", \"arguments\": {\"userInput\": \"ALAN，你好！\"}]"
            "<|tool_call_end|>"
        )
        res = self.parser.extract_tool_calls(model_output, self.request)

        self.assertFalse(res.tools_called)
        self.assertEqual(res.tool_calls, [])
        self.assertEqual(model_output, res.content)

    def test_extract_tool_calls_whit_exception(self):
        """测试 JSON 格式错误时的降级处理"""
        model_output = "<|tool_call_start|>[{\"name\":: \"error\"}]<|tool_call_end|>"
        res = self.parser.extract_tool_calls(model_output, self.request)
        print(f"res:{res}")
        self.assertFalse(res.tools_called)
        self.assertEqual(model_output, res.content)


class TestPanguToolParserExtractToolCallsStreaming(unittest.TestCase):
    """流式提取测试"""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {
            "<|tool_call_start|>": 104,
            "<|tool_call_end|>": 105
        }
        self.mock_tokenizer.tokenizer = self.mock_tokenizer
        self.parser = PanguToolParser(self.mock_tokenizer)
        self.request = MagicMock(spec=ChatCompletionRequest)

    def test_extract_tool_calls_streaming_whit_basic_returns(self):
        # case1:只有结束 Token ID
        res1 = self.parser.extract_tool_calls_streaming(
            "", "", "", [], [], [105], self.request)
        self.assertIsNone(res1)

        # case2:文本中还没有开始 Token
        res3 = self.parser.extract_tool_calls_streaming(
            "", "Hello", "Hello", [], [], [1], self.request)
        self.assertEqual(res3.content, "Hello")

        # case3:只有开始 Token ID
        res5 = self.parser.extract_tool_calls_streaming(
            "", "<|tool_call_start|>", "<|tool_call_start|>", [], [], [104], self.request)
        self.assertIsNone(res5)

    def test_extract_tool_calls_streaming_whit_text_before_start_token(self):
        """case:包含普通文本和开始 Token"""
        delta = "Thought: I should use a tool.<|tool_call_start|>"
        res = self.parser.extract_tool_calls_streaming(
            "", delta, delta, [], [], [1, 104], self.request)

        self.assertEqual(res.content, "Thought: I should use a tool.")

    def test_extract_tool_calls_streaming_whit_tool_arguments(self):
        """case: 假设name已发送，现在发送city片段"""
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.streamed_args_for_tool = [""]
        self.parser.prev_tool_call_arr = [{"name": "get_weather"}]

        curr = '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
        res = self.parser.extract_tool_calls_streaming(
            "", curr, '{"city": "Beijing"}}', [], [], [], self.request)

        self.assertIn("Beijing", res.tool_calls[0].function.arguments)
        self.assertIn("Beijing", self.parser.streamed_args_for_tool[0])

    def test_extract_tool_calls_streaming_with_end_token(self):
        """case:tool end"""
        curr = "<|tool_call_start|>[...]<|tool_call_end|> End text"
        delta = " End text"
        res = self.parser.extract_tool_calls_streaming(
            "", curr, delta, [], [], [], self.request)
        print(f"res: {res}")
        self.assertIsNone(res)

    def test_extract_tool_calls_streaming_whit_exception(self):
        """case:发送了部分 JSON"""
        # [{"name": "get_weather"
        curr = "<|tool_call_start|>[{\"name\": \"get_weather\""
        res = self.parser.extract_tool_calls_streaming(
            "<|tool_call_start|>", curr, "[{\"name\": \"get_weather\"",
            [104], [104, 2], [2], self.request)

        self.assertIsNone(res)

    def test_extract_tool_calls_streaming_with_new_tool_registration(self):
        """
        case：流式场景下新工具的识别与名称发送逻辑。
        逻辑链条：
        1. 初始状态为 -1，当解析到第一个工具时，状态应切换到 0。
        2. 状态切换后，第二次调用应识别出工具名并发送 DeltaMessage。
        """
        # 0. 初始化 parser 状态，模拟刚发现开始标签后的状态
        self.parser.current_tool_id = -1
        self.parser.current_tool_name_sent = False
        self.parser.streamed_args_for_tool = []

        # 1. 构造一个能被 partial_json_parser 解析的片段
        curr = '<|tool_call_start|>[{"name": "get_weather"}'

        # 2. 调用流式提取
        # current_tool_id 从 -1 变为 0
        res = self.parser.extract_tool_calls_streaming(
            previous_text="<|tool_call_start|>",
            current_text=curr,
            delta_text='[{"name": "get_weather"}',
            previous_token_ids=[104],
            current_token_ids=[104, 1],
            delta_token_ids=[1],
            request=self.request
        )

        self.assertEqual(self.parser.current_tool_id, 0)

        # 3. 第二次调用，模拟状态已更新，发送名称
        res_name = self.parser.extract_tool_calls_streaming(
            previous_text=curr,
            current_text=curr,  # 文本没变，但状态变了
            delta_text="",
            previous_token_ids=[104, 1],
            current_token_ids=[104, 1],
            delta_token_ids=[],
            request=self.request
        )

        self.assertTrue(self.parser.current_tool_name_sent)
        self.assertIsNotNone(res_name)
        self.assertEqual(res_name.tool_calls[0].function.name, "get_weather")
        self.assertEqual(res_name.tool_calls[0].index, 0)

    def test_extract_tool_calls_streaming_whit_multiple_tools_transition(self):
        """case:多工具切换"""
        prev_args = {"city": "BJ"}
        full_args_json = json.dumps(prev_args, ensure_ascii=False)

        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = True
        self.parser.streamed_args_for_tool = [full_args_json]
        self.parser.prev_tool_call_arr = [{"name": "t1", "arguments": prev_args}]

        # 包含两个工具
        curr = '<|tool_call_start|>[{"name": "t1", "arguments": {"city": "BJ"}}, {"name": "t2"}]'

        res = self.parser.extract_tool_calls_streaming(
            "", curr, ', {"name": "t2"}]', [], [], [], self.request)

        # 验证状态切换
        self.assertEqual(self.parser.current_tool_id, 1)
        self.assertFalse(self.parser.current_tool_name_sent)
        # res DeltaToolCall id 应该为 None
        self.assertIsNone(res.tool_calls[0].id)

        res_next = self.parser.extract_tool_calls_streaming(
            curr, curr, "", [], [], [], self.request)

        self.assertIsNotNone(res_next)
        self.assertEqual(res_next.tool_calls[0].function.name, "t2")
        self.assertTrue(self.parser.current_tool_name_sent)

    def test_extract_tool_calls_streaming_does_not_emit_empty_arguments(self):
        """case: partial JSON 补出的空 arguments 不应提前发送"""
        self.parser.current_tool_id = -1
        self.parser.current_tool_name_sent = False
        self.parser.streamed_args_for_tool = []

        partial = '<|tool_call_start|>[{"name": "get_nums", "arguments": {}}]'

        res = self.parser.extract_tool_calls_streaming(
            previous_text="<|tool_call_start|>",
            current_text=partial,
             delta_text='[{"name": "get_nums", "arguments": {}',
            previous_token_ids=[104],
            current_token_ids=[104, 1],
            delta_token_ids=[1],
            request=self.request
        )
       
        self.assertIsNone(res)
        self.assertEqual(self.parser.current_tool_id, 0)
        self.assertEqual(self.parser.streamed_args_for_tool, [""])

        res_name = self.parser.extract_tool_calls_streaming(
            previous_text=partial,
            current_text=partial,
            delta_text="",
            previous_token_ids=[104, 1],
            current_token_ids=[104, 1],
            delta_token_ids=[],
            request=self.request
        )

        self.assertIsNotNone(res_name)
        self.assertEqual(res_name.tool_calls[0].function.name, "get_nums")
        self.assertEqual(self.parser.streamed_args_for_tool[0], "{}")

    def test_extract_tool_calls_streaming_func_name_with_empety_dict(self):
        
        delta_tokens= [
            '<|tool_call_start|>[',
            '{"name": ',
            '"get_',
            'weather", "arguments": {}} ',
            ']<|tool_call_end|>', 
        ]
        
        all_token_ids = [
            [104, 228],
            [45920, 13],
            [89108, 89216, 24440, 84063],
            [93166, 88870, 89216, 6731],
            [ 45932, 105],
        ]
        previous_texts = [""]
        all_previous_token_ids = [[]]
        for idx, delta_text in enumerate(delta_tokens):
            delta_token_ids = all_token_ids[idx]
            previous_text = previous_texts[0]
            previous_token_ids = all_previous_token_ids[0]
            current_text = previous_text + delta_text
            current_token_ids = previous_token_ids + list(delta_token_ids)
            
            delta_message =  self.parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                self.request,
            )
            previous_texts[0] = current_text
            all_previous_token_ids[0] = current_token_ids
            
            if delta_message and delta_message.tool_calls:
                
                name = delta_message.tool_calls[0].function.name
                args = delta_message.tool_calls[0].function.arguments
                if name and args:
                    self.assertEqual(name, "get_weather")
                    self.assertEqual(json.loads(args), {})
            
    def test_extract_tool_calls_streaming_empety_dict(self):
        delta_tokens= [ 
            '<|tool_call_start|>',
           '[',
            '{"',
            'name": ',
            '"get_',
            'weather", ',
            '"arguments": '
            '{' ,
           '}',
            '} ',
            ']',
            '<|tool_call_end|>', 
        ]

        all_token_ids = [
            [104],
            *[[i]for i in range(len(delta_tokens)-2)],
            [105],
        ]
       
        previous_texts = [""]
        all_previous_token_ids = [[]]
        for idx, delta_text in enumerate(delta_tokens):
            delta_token_ids = all_token_ids[idx]
            previous_text = previous_texts[0]
            previous_token_ids = all_previous_token_ids[0]
            current_text = previous_text + delta_text
            current_token_ids = previous_token_ids + list(delta_token_ids)
            
            delta_message = self.parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                self.request,
            )
            previous_texts[0] = current_text
            all_previous_token_ids[0] = current_token_ids
            
            if delta_message and delta_message.tool_calls:
                
                name = delta_message.tool_calls[0].function.name
                args = delta_message.tool_calls[0].function.arguments
                if name:
                    self.assertEqual(name, "get_weather")
                if args:
                    self.assertEqual(json.loads(args), {})

if __name__ == '__main__':
    unittest.main()
