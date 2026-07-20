# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from typing import List, Literal


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
    "should_round_prompt_tokens",
    "align_remote_block_ids",
    "get_config_from_dict_or_env",
]
