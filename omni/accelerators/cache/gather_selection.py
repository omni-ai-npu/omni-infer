# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import importlib.util
import os

import torch_npu


def register_gather_selection_ops() -> None:
    if int(os.getenv("DISABLE_GATHER_SELECTION", "0")):
        return

    if not int(os.getenv("ENABLE_HOST_MAPPING", "0")):
        raise RuntimeError(
            "Gather Selection requires Host Mapping. Set ENABLE_HOST_MAPPING=1 "
            "or set DISABLE_GATHER_SELECTION=1."
        )

    if hasattr(torch_npu, "npu_gather_selection_kv_cache"):
        return

    module_name = "custom_ops"
    if importlib.util.find_spec(module_name) is not None:
        importlib.import_module(module_name)
        if hasattr(torch_npu, "npu_gather_selection_kv_cache"):
            return

    raise RuntimeError(
        "Gather Selection is enabled, but its Python operator package is not "
        "installed. Install a compatible Gather Selection OPP and operator "
        "extension, or set DISABLE_GATHER_SELECTION=1."
    )
