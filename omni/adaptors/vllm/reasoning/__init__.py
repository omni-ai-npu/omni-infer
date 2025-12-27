from vllm import reasoning
from .pangu_reasoning_parser import PanguReasoningParser
from .kimi2_thinking_reasoning_parser import Kimi2ThinkingReasoningParser
from .gptoss_reasoning_parser import GptOssReasoningParser
from .glm4_moe_reasoning_parser import Glm4MoeModelReasoningParser

def register_reasoning():
    reasoning.__all__.append("PanguReasoningParser")
    reasoning.__all__.append("Kimi2ThinkingReasoningParser")
    reasoning.__all__.append("GptOssReasoningParser")
    reasoning.__all__.append("Glm4MoeModelReasoningParser")