# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for omni.diagnostics.watchdog.stat_logger.

Covers the vLLM StatLogger plugin that drives the watchdog heartbeat.
Pure CPU; no NPU needed.
"""

from unittest.mock import MagicMock

import pytest

from omni.diagnostics.watchdog import heartbeat
from omni.diagnostics.watchdog.stat_logger import OmniNpuStatLogger


@pytest.fixture(autouse=True)
def _reset_heartbeat():
    heartbeat.reset()
    yield
    heartbeat.reset()


@pytest.fixture
def vllm_config():
    config = MagicMock()
    config.model_config.served_model_name = "test_model"
    return config


# ----- construction -----


def test_init_default_engine_indexes(vllm_config):
    logger = OmniNpuStatLogger(vllm_config)
    assert logger.engine_indexes == [0]
    assert logger.vllm_config is vllm_config


def test_init_custom_engine_indexes(vllm_config):
    logger = OmniNpuStatLogger(vllm_config, engine_indexes=[0, 1, 2])
    assert logger.engine_indexes == [0, 1, 2]


def test_init_logs_model_and_engines(vllm_config, caplog):
    with caplog.at_level("INFO"):
        OmniNpuStatLogger(vllm_config, engine_indexes=[0, 1])
    assert "OmniNpuStatLogger loaded" in caplog.text
    assert "test_model" in caplog.text
    assert "[0, 1]" in caplog.text


# ----- record / progress -----


def test_record_marks_progress_for_default_engine(vllm_config):
    logger = OmniNpuStatLogger(vllm_config)
    logger.record(None, None)
    assert 0 in heartbeat.snapshot()


def test_record_marks_progress_for_explicit_engine(vllm_config):
    logger = OmniNpuStatLogger(vllm_config, engine_indexes=[0, 1])
    logger.record(None, None, engine_idx=1)
    assert 1 in heartbeat.snapshot()
    assert 0 not in heartbeat.snapshot()


# ----- sleep state -----


def test_record_sleep_state_awake(vllm_config):
    logger = OmniNpuStatLogger(vllm_config, engine_indexes=[0])
    logger.record_sleep_state(0, 0)  # vLLM convention: arg 0 = wake
    assert heartbeat.is_sleeping(0) is False


def test_record_sleep_state_sleeping(vllm_config):
    logger = OmniNpuStatLogger(vllm_config, engine_indexes=[0])
    logger.record_sleep_state(1, 0)  # vLLM convention: arg 1 = sleep
    assert heartbeat.is_sleeping(0) is True


def test_record_sleep_state_applies_to_all_engines(vllm_config):
    logger = OmniNpuStatLogger(vllm_config, engine_indexes=[0, 1])
    logger.record_sleep_state(1, 0)  # sleep applies to every engine owned
    assert heartbeat.is_sleeping(0) is True
    assert heartbeat.is_sleeping(1) is True


# ----- initialization -----


def test_log_engine_initialized_marks_progress(vllm_config):
    logger = OmniNpuStatLogger(vllm_config)
    logger.log_engine_initialized()
    assert 0 in heartbeat.snapshot()


def test_log_engine_initialized_applies_to_all_engines(vllm_config):
    logger = OmniNpuStatLogger(vllm_config, engine_indexes=[0, 1])
    logger.log_engine_initialized()
    assert 0 in heartbeat.snapshot()
    assert 1 in heartbeat.snapshot()


# ----- periodic flush hook -----


def test_log_is_noop(vllm_config):
    logger = OmniNpuStatLogger(vllm_config)
    logger.log()  # should not raise or change state
