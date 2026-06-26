# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import types

from omni_cache.connector.utils.helpers import (
    PrefillEndpoint,
    align_remote_block_ids,
    resolve_prefill_endpoint,
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


def test_resolve_prefill_endpoint_uses_dp_rank_offset():
    cfg = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(),
        parallel_config=types.SimpleNamespace(data_parallel_rank=3),
    )

    endpoint = resolve_prefill_endpoint(
        vllm_config=cfg,
        node_ip_specs=["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
        cluster_size=2,
        p_node_list="10.0.0.1,10.0.0.2;10.0.0.3,10.0.0.4",
        get_local_ip=lambda: "10.0.0.2",
        get_config_value=lambda *_args, **_kwargs: 5568,
    )

    assert isinstance(endpoint, PrefillEndpoint)
    assert endpoint.cluster_id_start == 0
    assert endpoint.host_ip == "10.0.0.1"
    assert endpoint.host_port == 5571
