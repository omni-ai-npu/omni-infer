# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Pytest fixtures shared by ``tests/unit/parsers/`` tests.

Enforces per-test isolation of the streaming reasoning→tool-parser relay
(``omni_npu/v1/parsers/_streaming_relay.py``). The relay is a
``contextvars.ContextVar`` whose default semantics rely on each request
running in its own asyncio task. ``unittest.TestCase`` tests run
synchronously in a single Context, so a test that drives
``PanguReasoningParser.extract_reasoning_streaming`` leaves a stashed
reasoning that the next test invoking
``PanguToolParser.extract_tool_calls_streaming`` would otherwise inherit
— surfacing as ``DeltaMessage(reasoning="…")`` where the test expected
``None``. This fixture resets the relay before and after every test.
"""

from __future__ import annotations

import pytest

from omni_npu.v1.parsers._streaming_relay import reset_for_tests


@pytest.fixture(autouse=True)
def _reset_streaming_relay() -> None:
    """Clear the streaming-relay ContextVar around every test in this dir."""
    reset_for_tests()
    yield
    reset_for_tests()
