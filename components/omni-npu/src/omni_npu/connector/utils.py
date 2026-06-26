# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import math
import numpy as np
from typing import List, Tuple

import torch
from vllm.distributed.parallel_state import get_tp_group


class TP_Convertor:

    def __init__(self, remote_tp_size: int):
        tp_group = get_tp_group()
        self.tp_comm = tp_group.device_group
        self.tp_size = tp_group.world_size
        self.tp_rank = tp_group.rank_in_group
        self.transfer_done = False

        self.remote_tp_size = remote_tp_size
        assert self.tp_size % remote_tp_size == 0
        self.stride = self.tp_size // remote_tp_size
        self.offset = self.tp_rank % self.stride

    def scheme_reorg(self,
        token_num: int,
        tail_blk: int,
        kv_group: List[List[torch.Tensor]],
        local_block_ids: List[int],
        remote_block_ids: List[int],
    ) -> List[int]: # adjusted remote_blk_ids
        if self.stride == 1:
            return remote_block_ids

        tails = [TP_Convertor.tail_blk_num( # tok_num of tail block
            token_num, i, self.tp_size, self.remote_tp_size,
        ) for i in range(self.tp_size)]     # of all ranks
        before, after = tails[self.tp_rank] # tok_num before/after a2a
        end = max(before, after)

        self.tail_blk = tail_blk
        self.kv_group = kv_group
        self.send_domain = (after, end)
        self.recv_domain = (before, end)
        self.recv_num = end - before

        a2a_map = TP_Convertor.a2a_mapper([a - b for a, b in tails])
        self.send_split = a2a_map[self.tp_rank].tolist()
        self.recv_split = a2a_map[:, self.tp_rank].tolist()

        TP_Convertor.scheduled_list().append(self)

        n_remote = len(remote_block_ids)
        return [
            remote_block_ids[
                (self.offset + i * self.stride) % n_remote
            ] for i, _ in enumerate(local_block_ids)
        ]

    # call after scheme_reorg()
    def token_reorg(self):
        if self.stride == 1:
            return
        for kvs in self.kv_group:
            send = TP_Convertor.extract_kv(kvs, self.tail_blk, self.send_domain)
            recv = send.new_empty(self.recv_num, *send.shape[1:])
            torch.distributed.all_to_all_single(
                recv, send,
                self.recv_split,
                self.send_split,
                self.tp_comm,
            )
            TP_Convertor.store_kv(recv, kvs, self.tail_blk, self.recv_domain)

    # ======================= utils =======================

    @classmethod
    def scheduled_list(cls):
        rank = get_tp_group().rank_in_group
        attr_name = f"scheduled_reorg_{rank}"
        if not hasattr(cls, attr_name):
            setattr(cls, attr_name, [])
        return getattr(cls, attr_name)

    @classmethod
    def do_scheduled_kv_reorg(cls):
        # called by MLA builder
        # len(sched) always be the same in all ranks
        sched:List[TP_Convertor] = cls.scheduled_list()
        if len(sched) == 0:
            return
        fin_num = 0
        for it in sched:
            if not it.transfer_done:
                break
            fin_num += 1

        dev = sched[0].kv_group[0][0].device
        cfg = {"dtype": torch.int32, "device":dev}
        local_num = torch.tensor([fin_num], **cfg)
        global_num = get_tp_group().all_gather(local_num)
        all_fin_num = min(global_num.tolist())
        for i in range(all_fin_num):
            sched.pop(0).token_reorg()

    @staticmethod
    def tail_blk_num(num, d_rank, d_size, p_size, pg=128):
        assert d_size >= p_size
        stride = d_size // p_size
        p_num = num // p_size + int(d_rank // stride < num % p_size)
        d_num = num // d_size + int(d_rank < num % d_size)
        before = p_num % (pg * stride) - pg * (d_rank % stride)
        after = d_num % pg
        return min(pg, max(before, 0)), after

    @staticmethod
    def link_to_remote(tp_rank, tp_size, remote_tp_size):
        assert tp_size % remote_tp_size == 0, f"{tp_size=} % {remote_tp_size=} != 0"
        return tp_rank // (tp_size // remote_tp_size)

    @staticmethod
    def a2a_mapper(movement: List[int]):
        give, take = [], []
        for i, d in enumerate(movement):
            if d > 0:
                give.append([d, i])
            if d < 0:
                take.append([-d, i])

        num = len(movement)
        map = np.zeros((num, num), np.int32)
        while len(give) > 0 and len(take) > 0:
            a, b = give[-1], take[-1]
            d = min(a[0], b[0])
            map[a[1]][b[1]] = d
            if a[0] <= d:
                give.pop()
            else:
                a[0] -= d
            if b[0] <= d:
                take.pop()
            else:
                b[0] -= d

        assert len(give) + len(take) == 0
        return map # np.int32[num, num]

    @staticmethod
    def extract_kv(
        kvs: List[torch.Tensor],
        blk_i: int,
        domain: Tuple[int, int],
    ):
        def frag_of(kv):
            assert type(kv) is torch.Tensor
            assert kv.dim() == 3 # [blk, pg, D]
            return kv[blk_i][domain[0] : domain[1]]
        return torch.stack(
            [frag_of(it) for it in kvs]
        ).transpose(0, 1).contiguous() # [T, N, D]

    @staticmethod
    def store_kv(
        x: torch.Tensor,
        kvs: List[torch.Tensor],
        blk_i: int,
        domain: Tuple[int, int],
    ):
        assert x.dim() == 3 # [T, N, D]
        assert x.size(0) == max(0, domain[1] - domain[0])
        assert x.size(1) == len(kvs)
        x = x.transpose(0, 1) # [T, N, D] -> [N, T, D]
        for dst, src in zip(kvs, x):
            assert type(dst) is torch.Tensor
            assert dst.dim() == 3           # [blk, pg, D]
            assert dst.size(2) == x.size(2) # same D size
            dst[blk_i][domain[0] : domain[1]] = src


def get_p_start_rank(p_tp_size, p_dp_size, d_tp_size, d_dp_size, d_node_num, cur_d_node, cur_d_rank):
    # only support full tp in prefill.
    if p_dp_size != 1:
        raise ValueError('p_dp_size must be 1')

    # Parameter validation
    if p_tp_size <= 0 or d_tp_size <= 0 or d_dp_size <= 0 or d_node_num <= 0:
        raise ValueError('p_tp_size, d_tp_size, d_dp_size, d_node_num must be positive')

    if cur_d_node < 0 or cur_d_node >= d_node_num:
        raise ValueError('cur_d_node < 0 or cur_d_node >= d_node_num')

    if cur_d_rank < 0:
        raise ValueError('cur_d_rank < 0')

    # Calculate device information
    devices_per_node = d_dp_size * d_tp_size
    if cur_d_rank >= devices_per_node:
        raise ValueError('cur_d_rank >= devices_per_node')

    if p_tp_size <= d_tp_size: # for DCP in MLA
        return TP_Convertor.link_to_remote(
            tp_rank=cur_d_rank % d_tp_size,
            tp_size=d_tp_size,
            remote_tp_size=p_tp_size,
        )

    # Calculate KV group information
    kv_group_size = d_tp_size
    if p_tp_size % kv_group_size != 0:
        raise ValueError('p_tp_size % kv_group_size != 0')

    # Calculate current device's position in DP group
    cur_d_dp = cur_d_rank // d_tp_size
    cur_d_tp = cur_d_rank % d_tp_size

    # as same as get_p_start_rank_v1 in deepseek.
    global_dp_group = cur_d_node + cur_d_dp * d_node_num
    total_dp_groups = d_node_num * d_dp_size

    # Calculate prefill information
    p_replica_groups = p_tp_size // kv_group_size
    total_kv_groups = p_dp_size * p_replica_groups

    # Calculate connection step size (ensure KV group are evenly distrubuted)
    link_group_step = max(1, math.ceil(total_kv_groups / total_dp_groups))
    kv_group_index = (global_dp_group * link_group_step) % total_kv_groups

    # Calculate starting prefill rank for KV group
    p_dp_index = kv_group_index // p_replica_groups
    replica_index = kv_group_index % p_replica_groups

    # Calculate the prefill rank for current decode device
    stride = p_tp_size // kv_group_size
    offset = replica_index + cur_d_tp * stride
    return p_dp_index * p_tp_size + offset

def get_config_from_dict_or_env(config, config_var_name, env_var_name, default_value, value_type):
    env_value = os.environ.get(env_var_name, None)
    if isinstance(config, dict):
        args_value = config.get(config_var_name, None)
    else:
        args_value = getattr(config, config_var_name, None)
    if env_value is None and args_value is None:
        if default_value is None:
            raise ValueError(f"ENV {env_var_name} or args {config_var_name} should not be None.")
        else:
            value = default_value
    # ENV first
    elif env_value is not None:
        value = env_value
    else:
        value = args_value
    return value_type(value)
