# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import filelock
import vllm.transformers_utils.config as vllm_config
from transformers import PretrainedConfig

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_ORIGINAL_GET_CONFIG_PARSER = vllm_config.get_config_parser


def _get_config_parse_lock(
    model: str | Path,
    revision: str | None,
    code_revision: str | None,
    config_format: str,
) -> filelock.BaseFileLock:
    lock_dir = os.path.join(tempfile.gettempdir(), "vllm")
    os.makedirs(lock_dir, exist_ok=True)
    lock_key = f"{model}|{revision}|{code_revision}|{config_format}"
    lock_hash = hashlib.sha256(lock_key.encode()).hexdigest()
    lock_path = os.path.join(lock_dir, f"config-{lock_hash}.lock")
    return filelock.FileLock(lock_path, mode=0o666, timeout=10)


@register_patch("NPUConfigParserFileLockPatch", vllm_config)
class ConfigParserFileLockPatch(VLLMPatch):
    _attr_names_to_apply = ["get_config_parser"]

    @staticmethod
    def get_config_parser(config_format: str):
        parser = _ORIGINAL_GET_CONFIG_PARSER(config_format)
        original_parse = parser.parse

        def parse_with_file_lock(
            model: str | Path,
            trust_remote_code: bool,
            revision: str | None = None,
            code_revision: str | None = None,
            **kwargs: Any,
        ) -> tuple[dict, PretrainedConfig]:
            config_lock = _get_config_parse_lock(
                model=model,
                revision=revision,
                code_revision=code_revision,
                config_format=config_format,
            )
            try:
                with config_lock:
                    return original_parse(
                        model,
                        trust_remote_code=trust_remote_code,
                        revision=revision,
                        code_revision=code_revision,
                        **kwargs,
                    )
            except filelock.Timeout as e:
                raise RuntimeError(
                    "Timed out waiting for config parse lock (10s). "
                    "Another process may be reading the same model config."
                ) from e

        parser.parse = parse_with_file_lock
        return parser
