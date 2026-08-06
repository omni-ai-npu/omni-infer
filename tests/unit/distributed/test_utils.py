# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for omni.v1.distributed.utils.
"""

import pytest

from omni.v1.distributed.utils import (
    generate_round_swap_schedule,
    get_round_swap_perm,
)


class TestGenerateRoundSwapSchedule:

    def test_num_nodes_3_raises_value_error(self):
        with pytest.raises(ValueError, match="even"):
            generate_round_swap_schedule(3)

    def test_num_nodes_4_invariants(self):
        schedule = generate_round_swap_schedule(4)

        # rounds = num_nodes - 1, pairs per round = num_nodes // 2
        assert len(schedule) == 3
        for rd in schedule:
            assert len(rd) == 2

        seen_pairs = set()
        for round_idx, rd in enumerate(schedule):
            nodes_in_round = set()
            for pair in rd:
                # each pair is 2-element
                assert len(pair) == 2
                nodes_in_round.update(pair)
                # each pair unique globally
                key = tuple(sorted(pair))
                assert key not in seen_pairs, f"duplicate pair {pair}"
                seen_pairs.add(key)
            # each node appears exactly once per round
            assert nodes_in_round == {0, 1, 2, 3}, f"round {round_idx} nodes={nodes_in_round}"

        # C(4,2) = 6 unique pairs
        assert len(seen_pairs) == 6

    def test_num_nodes_8_invariants(self):
        schedule = generate_round_swap_schedule(8)

        # rounds = num_nodes - 1, pairs per round = num_nodes // 2
        assert len(schedule) == 7
        for rd in schedule:
            assert len(rd) == 4

        seen_pairs = set()
        for round_idx, rd in enumerate(schedule):
            nodes_in_round = set()
            for pair in rd:
                assert len(pair) == 2
                nodes_in_round.update(pair)
                key = tuple(sorted(pair))
                assert key not in seen_pairs, f"duplicate pair {pair}"
                seen_pairs.add(key)
            assert nodes_in_round == set(range(8)), f"round {round_idx} nodes={nodes_in_round}"

        # C(8,2) = 28 unique pairs
        assert len(seen_pairs) == 28


class TestGetRoundSwapPerm:
    """Verify get_round_swap_perm matches the pairings from generate_round_swap_schedule."""

    def test_perm_matches_schedule_for_all_ranks(self):
        num_nodes = 4
        schedule = generate_round_swap_schedule(num_nodes)

        for node_rank in range(num_nodes):
            perm = get_round_swap_perm(node_rank, num_nodes)

            # perm[0] is the node itself
            assert perm[0] == node_rank

            # perm[i+1] should be the partner of node_rank in round i
            for round_idx, rd in enumerate(schedule):
                partner = None
                for pair in rd:
                    if node_rank in pair:
                        partner = pair[0] if pair[1] == node_rank else pair[1]
                        break
                assert perm[round_idx + 1] == partner, (
                    f"rank={node_rank} round={round_idx}: "
                    f"perm[{round_idx+1}]={perm[round_idx+1]} != partner={partner}"
                )
