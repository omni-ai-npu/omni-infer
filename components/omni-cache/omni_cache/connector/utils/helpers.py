# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from dataclasses import dataclass
from typing import Callable, List, Literal, Sequence


@dataclass(frozen=True)
class PrefillEndpoint:
    cluster_id_start: int
    host_ip: str
    host_port: int


def resolve_prefill_endpoint(
    vllm_config,
    node_ip_specs: Sequence[str],
    cluster_size: int,
    p_node_list: str,
    get_local_ip: Callable[[], str],
    get_config_value: Callable[..., int],
) -> PrefillEndpoint:
    target_ip = get_local_ip()
    if target_ip not in node_ip_specs:
        raise ValueError(f"Local IP {target_ip} not found in P_NODE_LIST {p_node_list}")

    node_idx = node_ip_specs.index(target_ip)
    cluster_id_start = node_idx // cluster_size

    kv_rank = getattr(vllm_config.kv_transfer_config, "kv_rank", None)
    if kv_rank is not None:
        cluster_id_start = kv_rank

    idx = cluster_id_start * cluster_size
    host_ip = node_ip_specs[idx] if idx < len(node_ip_specs) else target_ip
    host_port = get_config_value(
        vllm_config.kv_transfer_config,
        "kv_port",
        "VLLM_LLMDATADIST_ZMQ_PORT",
        "5568",
        int,
    )
    host_port += vllm_config.parallel_config.data_parallel_rank
    return PrefillEndpoint(
        cluster_id_start=cluster_id_start,
        host_ip=host_ip,
        host_port=host_port,
    )


def should_round_prompt_tokens(
    patches: str,
    omni_cache_enabled: bool,
    host_mapping_enabled: bool,
) -> bool:
    has_scheduler_swa_patch = "SchedulerSWAPatch" in patches
    return not has_scheduler_swa_patch or (omni_cache_enabled and host_mapping_enabled)


def align_remote_block_ids(
    local_ids: List[int],
    remote_ids: List[int],
    remote_overflow: Literal["head", "tail"] = "head",
    remote_tail_skip: int = 0,
) -> tuple[List[int], List[int]]:
    if remote_overflow not in ("head", "tail"):
        raise ValueError(f"Unsupported remote_overflow policy: {remote_overflow}")
    if remote_tail_skip < 0:
        raise ValueError(f"remote_tail_skip must be non-negative: {remote_tail_skip}")

    if len(remote_ids) < len(local_ids):
        return local_ids[: len(remote_ids)], remote_ids
    if len(remote_ids) > len(local_ids):
        if remote_overflow == "tail":
            end = len(remote_ids) - remote_tail_skip
            start = max(0, end - len(local_ids))
            aligned_remote_ids = remote_ids[start:end]
            return local_ids[: len(aligned_remote_ids)], aligned_remote_ids
        return local_ids, remote_ids[: len(local_ids)]
    return local_ids, remote_ids


def get_config_from_dict_or_env(config, config_var_name, env_var_name, default_value, value_type):
    env_value = os.environ.get(env_var_name)
    args_value = config.get(config_var_name) if isinstance(config, dict) else getattr(config, config_var_name, None)
    if env_value is not None:
        return value_type(env_value)
    if args_value is not None:
        return value_type(args_value)
    if default_value is None:
        raise ValueError(f"ENV {env_var_name} or args {config_var_name} should not be None.")
    return value_type(default_value)


__all__ = [
    "PrefillEndpoint",
    "resolve_prefill_endpoint",
    "should_round_prompt_tokens",
    "align_remote_block_ids",
    "get_config_from_dict_or_env",
]
