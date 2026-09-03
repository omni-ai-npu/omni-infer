# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from omni_npu.layers.mhc.mhc_rl import NPUmHCRL
from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_moe import (
    patch_process_weights_after_loading as patch_mod,
)


def test_process_weights_patch_handles_mhc_rl():
    model = nn.Module()
    mhc = NPUmHCRL.__new__(NPUmHCRL)
    nn.Module.__init__(mhc)
    mhc.process_weights_after_loading = MagicMock()
    model.add_module("mhc", mhc)
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    target_device = torch.device("cpu")

    with patch.object(patch_mod, "_ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING") as orig:
        patch_mod._patched_process_weights_after_loading(
            model,
            model_config,
            target_device,
        )

    orig.assert_called_once_with(model, model_config, target_device)
    mhc.process_weights_after_loading.assert_called_once_with()
    assert model.process_weights_after_loading_already_called is True
    assert (
        patch_mod.PanguV2MoeProcessWeightsUtilsPatch.process_weights_after_loading
        is patch_mod._patched_process_weights_after_loading
    )
