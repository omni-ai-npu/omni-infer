from vllm import reasoning
from .pangu_reasoning_parser import PanguReasoningParser
from .kimi2_thinking_reasoning_parser import Kimi2ThinkingReasoningParser
from .gptoss_reasoning_parser import GptOssReasoningParser
from .glm4_moe_reasoning_parser import Glm4MoeModelReasoningParser


def _append_once(exports: list[str], name: str) -> None:
    if name not in exports:
        exports.append(name)


def register_reasoning():
    _append_once(reasoning.__all__, "PanguReasoningParser")
    _append_once(reasoning.__all__, "Kimi2ThinkingReasoningParser")
    _append_once(reasoning.__all__, "GptOssReasoningParser")
    _append_once(reasoning.__all__, "Glm4MoeModelReasoningParser")
