# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict

D = defaultdict(float)


def pytest_addoption(parser):
    parser.addoption("--durations-out", action="store", default="", help="write per-test durations json")


def pytest_runtest_logreport(report):
    # accumulate setup/call/teardown
    D[report.nodeid] += float(getattr(report, "duration", 0.0))


def pytest_sessionfinish(session, exitstatus):
    out = session.config.getoption("durations_out")
    if not out:
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(D), f, indent=4, sort_keys=True)
