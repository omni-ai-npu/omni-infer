# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared fixtures for the diagnostics unit tests.

omni_npu configures a non-propagating handler on the ``omni_npu`` logger, so
warnings emitted by its submodules never reach caplog's root handler. Every
diagnostics test that asserts on a degrade-to-warning path needs to bind caplog
to the emitting logger directly; ``capture_logger`` is that shared helper (it
replaces the per-file ``capture_on`` copies and mirrors
test_config_summary.caplog_config_summary).
"""
import logging
from contextlib import contextmanager

import pytest


@pytest.fixture
def capture_logger(caplog):
    """Return a context manager that binds caplog to a specific logger.

    Usage::

        with capture_logger(forensic.logger):
            forensic.cleanup_stale(...)
        assert any("cleanup of" in r.message for r in caplog.records)
    """
    @contextmanager
    def _capture(logger, level=logging.WARNING):
        added = caplog.handler not in logger.handlers
        if added:
            logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(level, logger=logger.name):
                yield caplog
        finally:
            if added:
                logger.removeHandler(caplog.handler)

    return _capture
