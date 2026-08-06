# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

from omni_npu.worker.npu_model_runner import NPUModelRunner


def test_iter_kv_cache_attn_groups_delegates():
    groups = [object(), object()]
    runner = object.__new__(NPUModelRunner)
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)

    assert list(runner.iter_kv_cache_attn_groups()) == groups
