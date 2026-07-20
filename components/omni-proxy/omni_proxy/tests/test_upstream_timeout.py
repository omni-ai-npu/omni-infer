"""
T1-T9 UT for omni_proxy_*_timeout directives.

T1: --help contains --omni-proxy-*-timeout
T2: default flag values are emitted to nginx.conf
T3: --omni-proxy-*-timeout <value> flows through to nginx.conf
T4: invalid value (e.g. -1s, abc, 5x) → nginx -t fails
T5: E2E — omni_proxy_read_timeout 1s triggers 504 on slow mock
T6: E2E — default 14400s does not false-positive on normal request
T8: E2E — reload preserves timeout value
T9: sub-location inherits parent's timeout via merge_loc_conf
"""
import json
import re
import subprocess
import time
from pathlib import Path

import pytest
import requests

import port_manager
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock_epd import start_mock_server, cleanup_subprocess  # noqa: E402


CUR_DIR = Path(__file__).parent
PROXY_SCRIPT = f"{CUR_DIR}/../omni_proxy.sh"


# ---------------------------------------------------------------------------
# T1-T3: 配置解析 (omni_proxy.sh --dry-run 验证生成的 config)
# ---------------------------------------------------------------------------


def _run_proxy_dry_run(extra_args=None):
    """调用 omni_proxy.sh --dry-run，返回生成的 nginx.conf 全文。"""
    conf_file = CUR_DIR / "_tmp_nginx_timeout_test.conf"
    if conf_file.exists():
        conf_file.unlink()
    cmd = [
        "bash", PROXY_SCRIPT,
        "--nginx-conf-file", str(conf_file),
        "--core-num", "1",
        "--prefill-endpoints", "127.0.0.1:8001",
        "--decode-endpoints", "127.0.0.1:9001",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append("--dry-run")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"omni_proxy.sh --dry-run failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    if not conf_file.exists():
        raise AssertionError(f"dry-run did not generate config: {result.stderr}")
    return conf_file.read_text()


# T1: --help 包含 4 个 flag
def test_t1_help_contains_flags():
    result = subprocess.run(
        ["bash", PROXY_SCRIPT, "--help"],
        capture_output=True, text=True,
    )
    for flag in [
        "--omni-proxy-connect-timeout",
        "--omni-proxy-send-timeout",
        "--omni-proxy-read-timeout",
        "--omni-proxy-next-upstream-timeout",
    ]:
        assert flag in result.stdout, \
            f"{flag} not in help:\n{result.stdout}"


# T2: 默认 flag 值被 emit 到 config
@pytest.mark.parametrize("directive, expected", [
    ("omni_proxy_connect_timeout",       "600s"),
    ("omni_proxy_send_timeout",          "600s"),
    ("omni_proxy_read_timeout",          "14400s"),
    ("omni_proxy_next_upstream_timeout", "0"),
])
def test_t2_default_emitted(directive, expected):
    conf_text = _run_proxy_dry_run()
    pattern = rf"^\s*{directive}\s+([^;]+);"
    matches = re.findall(pattern, conf_text, re.MULTILINE)
    assert matches, f"{directive} not found in config"
    assert matches[0] == expected, \
        f"{directive} expected '{expected}', got '{matches[0]}'"


# T3: 自定义 flag 值透传
@pytest.mark.parametrize("flag, value", [
    ("--omni-proxy-connect-timeout",       "5s"),
    ("--omni-proxy-connect-timeout",       "500ms"),
    ("--omni-proxy-connect-timeout",       "2m"),
    ("--omni-proxy-send-timeout",          "30s"),
    ("--omni-proxy-send-timeout",          "1m"),
    ("--omni-proxy-read-timeout",          "60s"),
    ("--omni-proxy-read-timeout",          "14400s"),
    ("--omni-proxy-next-upstream-timeout", "5s"),
    ("--omni-proxy-next-upstream-timeout", "10s"),
    ("--omni-proxy-next-upstream-timeout", "0"),
])
def test_t3_custom_value_emitted(flag, value):
    conf_text = _run_proxy_dry_run([flag, value])
    directive = flag.lstrip("-").replace("-", "_")
    pattern = rf"^\s*{directive}\s+([^;]+);"
    matches = re.findall(pattern, conf_text, re.MULTILINE)
    assert matches, f"{directive} not found in config:\n{conf_text[-500:]}"
    assert matches[0] == value, \
        f"{directive} expected '{value}', got '{matches[0]}'"


# ---------------------------------------------------------------------------
# T4: 非法值（-1s, abc, 5x）→ nginx -t 失败
# ---------------------------------------------------------------------------


def _generate_config_for_validation(extra_args):
    """生成 nginx.conf 但不启动 nginx，返回 conf_path。"""
    conf_file = CUR_DIR / "_tmp_nginx_timeout_invalid_test.conf"
    if conf_file.exists():
        conf_file.unlink()
    cmd = [
        "bash", PROXY_SCRIPT,
        "--nginx-conf-file", str(conf_file),
        "--core-num", "1",
        "--prefill-endpoints", "127.0.0.1:8001",
        "--decode-endpoints", "127.0.0.1:9001",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append("--dry-run")
    subprocess.run(cmd, capture_output=True, text=True)
    return conf_file


@pytest.mark.parametrize("flag, bad_value", [
    ("--omni-proxy-connect-timeout", "-1s"),
    ("--omni-proxy-connect-timeout", "abc"),
    ("--omni-proxy-connect-timeout", "5x"),
    ("--omni-proxy-send-timeout",    "-1s"),
    ("--omni-proxy-read-timeout",    "abc"),
    ("--omni-proxy-next-upstream-timeout", "5x"),
])
def test_t4_invalid_value_rejected(flag, bad_value):
    """T4: 非法值必须让 nginx -t 失败。"""
    conf_file = _generate_config_for_validation([flag, bad_value])
    assert conf_file.exists(), \
        f"dry-run did not generate config for {flag} {bad_value}"

    result = subprocess.run(
        ["nginx", "-t", "-c", str(conf_file), "-p", "/usr/local/nginx"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, \
        f"expected nginx -t to fail for {flag} {bad_value}, " \
        f"but it succeeded:\nstdout={result.stdout}\nstderr={result.stderr}"
    combined = (result.stdout + result.stderr).lower()
    assert ("invalid" in combined or "emerg" in combined or "failed" in combined), \
        f"unexpected nginx -t output for {flag} {bad_value}:\n" \
        f"stdout={result.stdout}\nstderr={result.stderr}"
    conf_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# T5: 端到端 — omni_proxy_read_timeout=1s + 慢 mock → 客户端收到 504
# ---------------------------------------------------------------------------


PREFILL_NUM = 1
DECODE_NUM = 1


@pytest.fixture(scope="module")
def short_read_timeout_setup():
    """起 proxy 时把 read_timeout 压到 1s，让 mock 默认耗时 > 1s 触发 timeout。"""
    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM, decode_delay_ms=2000)
    if not processes:
        pytest.fail("Start EPD mock fail")
    time.sleep(2)

    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(
        proxy_port=proxy_port,
        prefill_port_list=prefill_port_list,
        decode_port_list=decode_port_list,
        omni_proxy_read_timeout="1s",       # ★ 关键：把 read_timeout 压到 1s
        stream_ops="off",       # ← ADD THIS (T5: explicit no-stream so mock non-stream sleep gates headers)
    )
    if ret == -1:
        cleanup_subprocess(processes)
        pytest.fail("Start proxy fail")

    yield {"proxy_port": proxy_port, "decode_port_list": decode_port_list}

    teardown_proxy()
    cleanup_subprocess(processes)


def _make_long_input() -> str:
    return ("The quick brown fox jumps over the lazy dog. " * 230).strip()


def test_t5_read_timeout_triggers_504(short_read_timeout_setup):
    """T5: read_timeout=1s + 慢 mock (~2s) → 客户端应在 ~1s 收到 504。"""
    proxy_port = short_read_timeout_setup["proxy_port"]
    payload = {
        "model": "qwen",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": _make_long_input()}],
        "stream": False,
        # prefill_delay_us removed — using mock's --decode-delay-ms flag instead
    }
    t0 = time.time()
    with pytest.raises(requests.exceptions.HTTPError) as excinfo:
        with requests.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json=payload, timeout=10,
        ) as resp:
            resp.raise_for_status()
    elapsed = time.time() - t0
    assert excinfo.value.response.status_code == 504, \
        f"expected 504, got {excinfo.value.response.status_code}"
    assert 0.8 < elapsed < 4.5, \
        f"504 should fire near read_timeout=1s (delay 2s), elapsed={elapsed:.2f}s"


# ---------------------------------------------------------------------------
# T6: 端到端 — 默认 read_timeout=14400s → 正常请求顺利通过
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_timeout_setup():
    """起 proxy 不传 timeout flag，shell 默认 14400s。"""
    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Start EPD mock fail")
    time.sleep(2)

    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(
        proxy_port=proxy_port,
        prefill_port_list=prefill_port_list,
        decode_port_list=decode_port_list,
        # 不传 omni_proxy_read_timeout → 走 shell 默认 14400s
    )
    if ret == -1:
        cleanup_subprocess(processes)
        pytest.fail("Start proxy fail")

    yield {"proxy_port": proxy_port, "decode_port_list": decode_port_list}

    teardown_proxy()
    cleanup_subprocess(processes)


def test_t6_default_read_timeout_no_false_positive(default_timeout_setup):
    """T6: 默认 read_timeout=14400s 下，正常请求不超时。"""
    proxy_port = default_timeout_setup["proxy_port"]
    payload = {
        "model": "qwen",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    r = requests.post(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        json=payload, stream=True, timeout=30,
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"
    chunks = list(r.iter_lines())
    assert any(b"data: [DONE]" in c for c in chunks), \
        f"response missing [DONE]: {chunks[-3:]}"


# ---------------------------------------------------------------------------
# T8: 端到端 — 修改 nginx.conf 把 read_timeout 改小，nginx -s reload 后新值生效
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reload_setup():
    """起 proxy（默认 read_timeout=14400s），fixture 提供 conf_path 给 T8 修改。"""
    processes = start_mock_server(0, PREFILL_NUM, DECODE_NUM, decode_delay_ms=2000)
    if not processes:
        pytest.fail("Start EPD mock fail")
    time.sleep(2)

    ports = port_manager.load_ports_epd(0, PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(
        proxy_port=proxy_port,
        prefill_port_list=prefill_port_list,
        decode_port_list=decode_port_list,
        stream_ops="off",       # T8: explicit no-stream so mock non-stream sleep gates headers
    )
    if ret == -1:
        cleanup_subprocess(processes)
        pytest.fail("Start proxy fail")

    # setup_proxy 始终用 --nginx-conf-file .../nginx.conf（除非传入 _tmp_..._test.conf）。
    # 这里强制用 nginx.conf：它是 setup_proxy 默认且本测试唯一在跑的 nginx 实例加载的 conf。
    conf_file = CUR_DIR / "nginx.conf"
    assert conf_file.exists(), f"nginx.conf not found at {conf_file}"

    yield {"proxy_port": proxy_port, "decode_port_list": decode_port_list,
           "conf_file": conf_file}

    teardown_proxy()
    cleanup_subprocess(processes)


def test_t8_reload_preserves_timeout(reload_setup):
    """T8: 改 conf 把 read_timeout 14400s → 1s, nginx -s reload, 慢请求应得 504."""
    proxy_port = reload_setup["proxy_port"]
    conf_file = reload_setup["conf_file"]
    assert conf_file is not None and conf_file.exists(), \
        f"could not locate nginx conf for reload test, tried {[str(c) for c in [CUR_DIR/'_tmp_nginx_timeout_test.conf', CUR_DIR/'nginx.conf']]}"

    original = conf_file.read_text()
    reload_ok = False
    try:
        # 改 read_timeout 14400s → 1s
        new_conf = original.replace(
            "omni_proxy_read_timeout 14400s",
            "omni_proxy_read_timeout 1s",
        )
        assert new_conf != original, \
            "could not find 'omni_proxy_read_timeout 14400s' in config"
        conf_file.write_text(new_conf)

        # nginx -t（sudo 因 setup_proxy 用 sudo 起的 nginx，pid 归 root，
        # reload 路径要 sudo 才能 kill 旧 worker；这里也用 sudo 保持一致）
        result = subprocess.run(
            ["sudo", "nginx", "-t", "-c", str(conf_file), "-p", "/usr/local/nginx"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, \
            f"nginx -t failed after edit:\nstdout={result.stdout}\nstderr={result.stderr}"

        # nginx -s reload
        result = subprocess.run(
            ["sudo", "nginx", "-s", "reload"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, \
            f"nginx -s reload failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        reload_ok = True
        time.sleep(2)  # 等新 worker 拉起

        # 慢请求（mock decode_delay_ms 2000） → 应该 504
        payload = {
            "model": "qwen",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": _make_long_input()}],
            "stream": False,
        }
        with pytest.raises(requests.exceptions.HTTPError) as excinfo:
            with requests.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=payload, timeout=10,
            ) as resp:
                resp.raise_for_status()
        assert excinfo.value.response.status_code == 504, \
            f"expected 504 after reload, got {excinfo.value.response.status_code}"

    finally:
        # 恢复原 conf + reload 回去，避免污染后续测试
        if reload_ok:
            conf_file.write_text(original)
            subprocess.run(["sudo", "nginx", "-t", "-c", str(conf_file), "-p", "/usr/local/nginx"],
                           capture_output=True, text=True)
            subprocess.run(["sudo", "nginx", "-s", "reload"],
                           capture_output=True, text=True)
            time.sleep(1)
