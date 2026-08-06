# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import atexit
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import vllm.envs as envs
from vllm.multimodal.utils import MediaConnector
from vllm.multimodal.base import MediaIO

global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT
)
atexit.register(global_thread_pool.shutdown)

_M = TypeVar("_M")

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("MediaConnectorPatch", MediaConnector)
class MediaConnectorPatch(VLLMPatch):
    _attr_names_to_apply = ['load_from_url_async']

    #####patch start: for pangu92B-VL
    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:
        loop = asyncio.get_running_loop()
        # Fast-path for data: URLs to avoid urlparse scanning the
        # entire base64 payload which can be very large.
        if url.startswith("data:"):
            data_spec, data = url[5:].split(",", 1)
            media_type, data_type = data_spec.split(";", 1)

            if data_type != "base64":
                msg = "Only base64 data URLs are supported for now."
                raise NotImplementedError(msg)
            
            future = loop.run_in_executor(
                global_thread_pool, media_io.load_base64, media_type, data
            )
            return await future

        url_spec = urlparse(url)

        if url_spec.scheme.startswith("http"):
            self._assert_url_in_allowed_media_domains(url_spec)

            connection = self.connection
            data = await connection.async_get_bytes(
                url,
                timeout=fetch_timeout,
                allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS,
            )
            future = loop.run_in_executor(global_thread_pool, media_io.load_bytes, data)
            return await future

        if url_spec.scheme == "file":
            future = loop.run_in_executor(
                global_thread_pool, self._load_file_url, url_spec, media_io
            )
            return await future
        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)
    #####patch end