# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.


def generate_round_swap_schedule(num_nodes: int) -> list[list[list[int]]]:
    """
    Generate perfect matching coverings for round-robin swap using the circle method.

    Args:
        num_nodes: Even number of nodes.

    Returns:
        One list per round, each containing num_nodes/2 pairs [i, j] that partition all nodes.
        Total rounds = num_nodes - 1, each node appears exactly once per round.
        Each pair appears exactly once across all rounds.
    """
    if num_nodes % 2 != 0 or num_nodes < 2:
        raise ValueError(f"num_nodes must be even and >= 2, got {num_nodes}")
    m = num_nodes // 2
    rest_nodes = list(range(1, num_nodes))
    schedule = []
    for _ in range(num_nodes - 1):
        round_pairs = [[0, rest_nodes[0]]]
        for j in range(1, m):
            sorted_pair = sorted([rest_nodes[j], rest_nodes[num_nodes - 1 - j]])
            round_pairs.append(sorted_pair)
        schedule.append(round_pairs)
        rest_nodes = rest_nodes[1:] + [rest_nodes[0]]
    return schedule


def get_round_swap_perm(node_rank: int, num_nodes: int):
    if node_rank == 0:
        return list(range(num_nodes))
    perm = [node_rank]
    for i in range(num_nodes - 1):
        swap_idx = (num_nodes + 1 + 2 * i - node_rank - 1) % (num_nodes - 1) + 1
        if swap_idx == node_rank:
            swap_idx = 0
        perm.append(swap_idx)
    return perm
