# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise the real async/executor paths without network or model weights."""

import asyncio
import atexit
import base64
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlparse

import pytest


@pytest.fixture
def media(patch_env):
    class MediaIO:
        def __class_getitem__(cls, item):
            return cls

    connector_cls = type("MediaConnector", (), {})
    env = patch_env.stub("vllm.envs", VLLM_MEDIA_LOADING_THREAD_COUNT=2,
                         VLLM_MEDIA_URL_ALLOW_REDIRECTS=False)
    patch_env.stub("vllm.multimodal.media", MediaConnector=connector_cls,
                   MediaIO=MediaIO)
    mod = patch_env.load("patch_media_utils")
    mod.MediaConnectorPatch.apply()
    connector = connector_cls()
    connector.connection = SimpleNamespace(async_get_bytes=AsyncMock(return_value=b"image"))
    connector._assert_url_in_allowed_media_domains = Mock()
    connector._load_file_url = Mock(return_value=b"file-image")
    io = SimpleNamespace(load_bytes=Mock(return_value="decoded"),
                         load_base64=Mock(return_value="base64-decoded"))
    yield SimpleNamespace(mod=mod, env=env, connector=connector, io=io)
    atexit.unregister(mod.global_thread_pool.shutdown)
    mod.global_thread_pool.shutdown(wait=True)


def run(media, url, **kwargs):
    return asyncio.run(media.connector.load_from_url_async(url, media.io, **kwargs))


def test_data_url_is_decoded_in_worker_without_urlparse(media, monkeypatch):
    main_thread = threading.get_ident()

    def decode(mime, data):
        assert threading.get_ident() != main_thread
        assert mime == "image/png"
        return base64.b64decode(data)

    media.io.load_base64.side_effect = decode
    parse = Mock(side_effect=AssertionError("data URL must use the fast path"))
    monkeypatch.setattr(media.mod, "urlparse", parse)
    assert run(media, "data:image/png;base64,aGVsbG8=") == b"hello"
    parse.assert_not_called()
    media.connector.connection.async_get_bytes.assert_not_awaited()


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("redirects,timeout", [(False, None), (True, 12)])
def test_http_validates_domain_before_download_and_forwards_options(media, scheme, redirects, timeout):
    media.env.VLLM_MEDIA_URL_ALLOW_REDIRECTS = redirects
    url = f"{scheme}://media.example/video.mp4?x=1"

    async def download(*args, **kwargs):
        media.connector._assert_url_in_allowed_media_domains.assert_called_once_with(urlparse(url))
        return b"downloaded"

    media.connector.connection.async_get_bytes.side_effect = download
    assert run(media, url, fetch_timeout=timeout) == "decoded"
    media.connector.connection.async_get_bytes.assert_awaited_once_with(
        url, timeout=timeout, allow_redirects=redirects)
    media.io.load_bytes.assert_called_once_with(b"downloaded")


def test_file_delegates_access_policy_and_io(media):
    url = "file:///data/a%20b.mp4"
    assert run(media, url) == b"file-image"
    media.connector._load_file_url.assert_called_once_with(urlparse(url), media.io)
    media.connector.connection.async_get_bytes.assert_not_awaited()


@pytest.mark.parametrize("url,error", [
    ("data:image/png;utf8,hello", NotImplementedError),
    ("data:image/png;base64", ValueError),
    ("data:image/png,hello", ValueError),
    ("ftp://example/image.png", ValueError),
])
def test_existing_url_format_errors_propagate(media, url, error):
    with pytest.raises(error):
        run(media, url)
    media.connector.connection.async_get_bytes.assert_not_awaited()


@pytest.mark.parametrize("stage,url,error", [
    ("domain", "https://denied.example/a", PermissionError("denied")),
    ("download", "https://media.example/a", TimeoutError("download timeout")),
    ("bytes", "https://media.example/a", ValueError("bad image")),
    ("base64", "data:image/png;base64,xxxx", ValueError("bad base64")),
    ("file", "file:///data/missing.mp4", FileNotFoundError("missing")),
])
def test_io_errors_reach_the_request_without_wrapping(media, stage, url, error):
    target = {
        "domain": media.connector._assert_url_in_allowed_media_domains,
        "download": media.connector.connection.async_get_bytes,
        "bytes": media.io.load_bytes,
        "base64": media.io.load_base64,
        "file": media.connector._load_file_url,
    }[stage]
    target.side_effect = error
    with pytest.raises(type(error)) as caught:
        run(media, url)
    assert caught.value is error
    if stage == "domain":
        media.connector.connection.async_get_bytes.assert_not_awaited()
