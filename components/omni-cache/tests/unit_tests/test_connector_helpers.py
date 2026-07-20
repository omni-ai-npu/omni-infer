# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from omni_cache.connector.utils.helpers import (
    align_remote_block_ids,
    should_round_prompt_tokens,
)


def test_should_round_prompt_tokens_matrix():
    assert should_round_prompt_tokens("", False, False) is True
    assert should_round_prompt_tokens("SchedulerSWAPatch", False, False) is False
    assert should_round_prompt_tokens("SchedulerSWAPatch", True, True) is True


def test_align_remote_block_ids_remote_shorter():
    local, remote = align_remote_block_ids([1, 2, 3], [8, 9])
    assert local == [1, 2]
    assert remote == [8, 9]


def test_align_remote_block_ids_remote_longer():
    local, remote = align_remote_block_ids([1, 2], [7, 8, 9, 10])
    assert local == [1, 2]
    assert remote == [9, 10]
