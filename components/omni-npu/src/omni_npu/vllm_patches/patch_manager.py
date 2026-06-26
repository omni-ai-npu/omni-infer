# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html


import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

class PatchManager:
    registered_patches: Dict[str, type] = {}

    def __init__(self):
        self.applied_patches: List[str] = []

    @classmethod
    def register(cls, patch_name: str, patch_class: type):
        cls.registered_patches[patch_name] = patch_class
        logger.info(f"patch class {patch_class.__name__} registered as {patch_name}")

    def apply_patch(self, patch_name: str):
        if patch_name not in self.registered_patches:
            logger.error(f"patch {patch_name} not registered")
            return

        if patch_name in self.applied_patches:
            logger.warning(f"patch {patch_name} already applied")
            return

        try:
            self.registered_patches[patch_name].apply()
            self.applied_patches.append(patch_name)
        except Exception as e:
            logger.error(f"failed to apply {patch_name}: {e}")

    def apply_all_patches(self):
        if not self.registered_patches:
            logger.info("no patches registered")
            return

        logger.info(f"applying patches: {list(self.registered_patches.keys())}")

        for patch_name in self.registered_patches:
            self.apply_patch(patch_name)

    def apply_patches_from_env(self):
        """
        apply patches in OMNI_NPU_VLLM_PATCHES environment variable.

        example: OMNI_NPU_VLLM_PATCHES="PatchA,PatchB"
        """
        patches_from_env = os.environ.get('OMNI_NPU_VLLM_PATCHES', '').strip()

        if not patches_from_env:
            logger.info("no patches specified in env OMNI_NPU_VLLM_PATCHES")
            return

        patch_list = [p.strip() for p in patches_from_env.split(',') if p.strip()]
        logger.info(f"applying patches: {patch_list}")

        for patch_name in patch_list:
            self.apply_patch(patch_name)

    def apply_patches(self):
        apply_all_env = os.environ.get('OMNI_NPU_VLLM_PATCHES', '').strip()

        # New logic: apply all patches when unset, empty, or 'ALL'
        if not apply_all_env or apply_all_env == 'ALL':
            self.apply_all_patches()
        else:
            self.apply_patches_from_env()

        logger.info(f"successfully applied patches: {self.applied_patches}")
