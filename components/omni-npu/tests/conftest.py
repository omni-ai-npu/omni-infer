# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# Disable all omni-* plugins (e.g. omni-cache) in UT: these out-of-tree plugins
# register hooks via entry points under "omni.*" groups. When they are present
# in the test environment they can mutate omni-npu behavior or mock return
# values in unexpected ways, so UT for omni-npu runs without them enabled.
import importlib.metadata as _metadata

_orig_entry_points = _metadata.entry_points


def _filtered_entry_points(*args, **kwargs):
    group = kwargs.get("group") or (args[0] if args else None)
    if isinstance(group, str) and group.startswith("omni."):
        return []
    return _orig_entry_points(*args, **kwargs)


_metadata.entry_points = _filtered_entry_points

import pytest


@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield
