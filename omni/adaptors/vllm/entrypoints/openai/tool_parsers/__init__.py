from vllm.entrypoints.openai import tool_parsers
from .pangu_tool_parser import PanguToolParser
from .openai_tool_parser import OpenAIToolParser
from .glm4_moe_tool_parser import Glm4MoeModelToolParser


def _append_once(exports: list[str], name: str) -> None:
    if name not in exports:
        exports.append(name)


def register_tool():
    _append_once(tool_parsers.__all__, "PanguToolParser")
    _append_once(tool_parsers.__all__, "OpenAIToolParser")
    _append_once(tool_parsers.__all__, "Glm4MoeModelToolParser")
