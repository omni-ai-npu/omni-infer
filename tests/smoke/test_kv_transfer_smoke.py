# SPDX-License-Identifier: MIT
"""端到端 smoke：验证 kv_transfer 方案 B 全链路（worker→KVConnectorStats通道→前端counter→/metrics）。

需要一个**活着的 D 实例**，故默认跳过；给了 OMNI_SMOKE_METRICS_URL 才跑（e2e/smoke 阶段用），
普通单测 CI（无引擎）自动 skip，不受影响。

前提（实例侧）：
  1. D 起服务前设 OMNI_METRICS_KV_TRANSFER_SELFTEST=1（DecodeWorker 启动注入 kind=selftest 合成失败）。
  2. 起服务后发过至少一条请求（触发 model step，把 selftest 从 worker 累加器 drain 到前端）。

运行：
  OMNI_SMOKE_METRICS_URL=http://localhost:9100/metrics \
    python -m pytest tests/smoke/test_kv_transfer_smoke.py -q
"""

import os
import re
import urllib.request

import pytest

_URL = os.getenv("OMNI_SMOKE_METRICS_URL")

pytestmark = pytest.mark.skipif(
    not _URL,
    reason="设 OMNI_SMOKE_METRICS_URL=http://<host>:<port>/metrics（指向活实例）才运行",
)

_SELFTEST_PAT = re.compile(
    r'^omni_npu:kv_transfer_failures_total\{[^}]*kind="selftest"[^}]*\}\s+([0-9.eE+]+)',
    re.M,
)


def _fetch_metrics() -> str:
    with urllib.request.urlopen(_URL, timeout=10) as resp:
        return resp.read().decode()


def test_kv_transfer_selftest_counter_visible():
    """selftest 合成失败应经方案 B 全链路现于前端 /metrics，值 > 0。"""
    data = _fetch_metrics()
    values = [float(v) for v in _SELFTEST_PAT.findall(data)]
    if not (values and any(v > 0 for v in values)):
        seen = [
            line
            for line in data.splitlines()
            if "kv_transfer" in line and not line.startswith("#")
        ]
        pytest.fail(
            'omni_npu:kv_transfer_failures{kind="selftest"} 未出现或为 0。'
            "排查：①实例是否以 OMNI_METRICS_KV_TRANSFER_SELFTEST=1 启动 "
            "②是否发过请求触发 model step ③前端是否有 registered 日志。"
            f" 当前 kv_transfer 行：{seen or '无'}"
        )
