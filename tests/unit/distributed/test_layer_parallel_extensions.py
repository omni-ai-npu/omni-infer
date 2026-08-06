# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for layer parallel helper utilities.
"""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch

from omni.v1.distributed import (
    communication_op_ext,
    parallel_state_ext,
)


class TestCommunicationOpExtensions(unittest.TestCase):
    """Verify communication helpers guard against missing groups and delegate correctly."""

    @patch(
        "omni.v1.distributed.communication_op_ext.get_layer_parallel_group",
    )
    def test_get_group_returns_none_when_world_size_le_one(self, mock_get_layer_group):
        group = Mock()
        group.world_size = 1
        mock_get_layer_group.return_value = group
        self.assertIsNone(communication_op_ext._get_group("layer"))

    @patch("omni.v1.distributed.communication_op_ext._get_group", return_value=None)
    def test_all_reduce_returns_input_without_group(self, _):
        tensor = torch.ones(2, 2)
        result = communication_op_ext.layer_parallel_all_reduce(tensor, "layer")
        self.assertIs(result, tensor)

    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_all_reduce_uses_group_all_reduce(self, mock_get_group):
        tensor = torch.randn(2, 2)
        group = Mock()
        group.all_reduce.return_value = torch.full((2, 2), 7.0)
        mock_get_group.return_value = group

        result = communication_op_ext.layer_parallel_all_reduce(tensor, "layer")

        group.all_reduce.assert_called_once_with(tensor)
        torch.testing.assert_close(result, torch.full((2, 2), 7.0))

    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_all_gather_delegates_dim(self, mock_get_group):
        tensor = torch.randn(2, 2)
        group = Mock()
        group.all_gather.return_value = "out"
        mock_get_group.return_value = group

        result = communication_op_ext.layer_parallel_all_gather(
            tensor, "layer", dim=1
        )

        group.all_gather.assert_called_once_with(tensor, dim=1)
        self.assertEqual(result, "out")

    @patch("omni.v1.distributed.communication_op_ext._get_group", return_value=None)
    def test_all_gather_returns_input_without_group(self, _):
        tensor = torch.randn(2, 2)
        result = communication_op_ext.layer_parallel_all_gather(tensor, "layer")
        self.assertIs(result, tensor)

    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_reduce_scatter_delegates_dim(self, mock_get_group):
        tensor = torch.randn(2, 2)
        group = Mock()
        group.reduce_scatter.return_value = "out"
        mock_get_group.return_value = group

        result = communication_op_ext.layer_parallel_reduce_scatter(
            tensor, "layer", dim=0
        )

        group.reduce_scatter.assert_called_once_with(tensor, dim=0)
        self.assertEqual(result, "out")

    @patch("omni.v1.distributed.communication_op_ext._get_group", return_value=None)
    def test_reduce_scatter_returns_input_without_group(self, _):
        tensor = torch.randn(2, 2)
        result = communication_op_ext.layer_parallel_reduce_scatter(tensor, "layer")
        self.assertIs(result, tensor)

    @patch("omni.v1.distributed.communication_op_ext.get_local_world_group")
    def test_all_gather_local_returns_input_when_world_size_one(self, mock_get_local_group):
        group = Mock()
        group.world_size = 1
        mock_get_local_group.return_value = group
        tensor = torch.randn(2, 3)

        result = communication_op_ext.all_gather_local(tensor, dim=1)
        self.assertIs(result, tensor)

    @patch("omni.v1.distributed.communication_op_ext.get_local_world_group")
    def test_all_gather_local_delegates_to_group(self, mock_get_local_group):
        group = Mock()
        group.world_size = 2
        group.all_gather.return_value = "gathered"
        mock_get_local_group.return_value = group
        tensor = torch.randn(2, 3)

        result = communication_op_ext.all_gather_local(tensor, dim=0)
        group.all_gather.assert_called_once_with(tensor, 0)
        self.assertEqual(result, "gathered")

    @patch("omni.v1.distributed.communication_op_ext.get_local_world_group")
    def test_reduce_scatter_local_delegates_to_group(self, mock_get_local_group):
        group = Mock()
        group.world_size = 2
        group.reduce_scatter.return_value = "scattered"
        mock_get_local_group.return_value = group
        tensor = torch.randn(2, 3)

        result = communication_op_ext.reduce_scatter_local(tensor)
        group.reduce_scatter.assert_called_once_with(tensor)
        self.assertEqual(result, "scattered")

    @patch("omni.v1.distributed.communication_op_ext.get_local_world_group")
    def test_all_to_all_local_requires_device_communicator(self, mock_get_local_group):
        group = Mock()
        group.world_size = 2
        group.device_communicator = None
        mock_get_local_group.return_value = group

        with self.assertRaises(RuntimeError) as context:
            communication_op_ext.all_to_all_local(torch.randn(2, 4))
        self.assertIn("no device communicator", str(context.exception))

    @patch("omni.v1.distributed.communication_op_ext.get_local_world_group")
    def test_all_to_all_local_validates_dims_and_divisibility(self, mock_get_local_group):
        communicator = Mock()
        group = Mock()
        group.world_size = 2
        group.device_communicator = communicator
        mock_get_local_group.return_value = group

        with self.assertRaises(ValueError):
            communication_op_ext.all_to_all_local(torch.randn(2, 4), scatter_dim=3, gather_dim=0)
        with self.assertRaises(ValueError):
            communication_op_ext.all_to_all_local(torch.randn(2, 4), scatter_dim=0, gather_dim=3)
        with self.assertRaises(ValueError):
            communication_op_ext.all_to_all_local(torch.randn(3, 4), scatter_dim=0, gather_dim=1)

    @patch("omni.v1.distributed.communication_op_ext.get_local_world_group")
    def test_all_to_all_local_uses_normalized_dims(self, mock_get_local_group):
        communicator = Mock()
        communicator.all_to_all.return_value = "ok"
        group = Mock()
        group.world_size = 2
        group.device_communicator = communicator
        mock_get_local_group.return_value = group
        tensor = torch.randn(2, 4)

        result = communication_op_ext.all_to_all_local(tensor, scatter_dim=-1, gather_dim=0)
        communicator.all_to_all.assert_called_once_with(tensor, scatter_dim=1, gather_dim=0)
        self.assertEqual(result, "ok")

    @patch("omni.v1.distributed.communication_op_ext._get_group", return_value=None)
    def test_layer_parallel_all2all_single_returns_input_without_group(self, _):
        tensor = torch.randn(2, 2)
        self.assertIs(
            communication_op_ext.layer_parallel_all2all_single(tensor, "layer"),
            tensor,
        )

    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_layer_parallel_all2all_single_validates_type_and_group(self, mock_get_group):
        group = Mock()
        group.world_size = 2
        group.device_group = object()
        mock_get_group.return_value = group
        tensor = torch.randn(2, 4)

        with self.assertRaises(TypeError):
            communication_op_ext.layer_parallel_all2all_single(tensor, "layer", dim="0")

        group.device_group = None
        with self.assertRaises(RuntimeError):
            communication_op_ext.layer_parallel_all2all_single(tensor, "layer", dim=0)

    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_layer_parallel_all2all_single_validates_dim_and_split_size(self, mock_get_group):
        group = Mock()
        group.world_size = 2
        group.device_group = object()
        mock_get_group.return_value = group

        with self.assertRaises(ValueError):
            communication_op_ext.layer_parallel_all2all_single(torch.randn(2, 4), "layer", dim=3)
        with self.assertRaises(ValueError):
            communication_op_ext.layer_parallel_all2all_single(torch.randn(3, 4), "layer", dim=0)

    @patch("omni.v1.distributed.communication_op_ext.dist.get_world_size", return_value=2)
    @patch("omni.v1.distributed.communication_op_ext.dist.all_to_all_single")
    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_layer_parallel_all2all_single_happy_path_nonzero_dim(
        self, mock_get_group, mock_all_to_all_single, _
    ):
        group = Mock()
        group.world_size = None  # cover fallback to dist.get_world_size
        group.device_group = object()
        mock_get_group.return_value = group

        def _copy_with_bias(out_buf, in_buf, group=None):  # noqa: ARG001
            out_buf.copy_(in_buf + 1)

        mock_all_to_all_single.side_effect = _copy_with_bias
        tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)

        result = communication_op_ext.layer_parallel_all2all_single(
            tensor, "layer", dim=1
        )
        torch.testing.assert_close(result, tensor + 1)
        mock_all_to_all_single.assert_called_once()

    @patch("omni.v1.distributed.communication_op_ext.dist.all_to_all_single")
    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_layer_parallel_all2all_single_happy_path_dim0(
        self, mock_get_group, mock_all_to_all_single
    ):
        group = Mock()
        group.world_size = 2
        group.device_group = object()
        mock_get_group.return_value = group

        def _copy_with_bias(out_buf, in_buf, group=None):  # noqa: ARG001
            out_buf.copy_(in_buf + 1)

        mock_all_to_all_single.side_effect = _copy_with_bias
        tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)

        result = communication_op_ext.layer_parallel_all2all_single(
            tensor, "layer", dim=0
        )
        torch.testing.assert_close(result, tensor + 1)
        mock_all_to_all_single.assert_called_once()

    @patch("omni.v1.distributed.communication_op_ext.dist.all_to_all_single")
    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_layer_parallel_dp2tp_all2all_exchanges_feature_shards(
        self, mock_get_group, mock_all_to_all_single
    ):
        group = Mock()
        group.world_size = 2
        group.device_group = object()
        mock_get_group.return_value = group
        tensor = torch.tensor([[3.0, 30.0, 4.0, 40.0]])

        def _receive_rank_one_shards(out_buf, _in_buf, group=None):  # noqa: ARG001
            out_buf.copy_(torch.tensor([[[2.0, 20.0]], [[4.0, 40.0]]]))

        mock_all_to_all_single.side_effect = _receive_rank_one_shards
        result = communication_op_ext.layer_parallel_dp2tp_all2all(
            tensor, "layer", dim=-1
        )

        torch.testing.assert_close(result, torch.tensor([[2.0, 20.0], [4.0, 40.0]]))
        mock_all_to_all_single.assert_called_once()

    @patch("omni.v1.distributed.communication_op_ext._get_group", return_value=None)
    def test_layer_parallel_dp2tp_all2all_returns_input_without_group(self, _):
        tensor = torch.randn(2, 4)

        result = communication_op_ext.layer_parallel_dp2tp_all2all(tensor, "layer")

        self.assertIs(result, tensor)

    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_layer_parallel_dp2tp_all2all_validates_group_dim_and_feature_size(
        self, mock_get_group
    ):
        group = Mock()
        group.world_size = 2
        group.device_group = object()
        mock_get_group.return_value = group
        tensor = torch.randn(2, 4)

        with self.assertRaises(TypeError):
            communication_op_ext.layer_parallel_dp2tp_all2all(
                tensor, "layer", dim="1"
            )
        group.device_group = None
        with self.assertRaises(RuntimeError):
            communication_op_ext.layer_parallel_dp2tp_all2all(
                tensor, "layer", dim=1
            )

        group.device_group = object()
        with self.assertRaises(ValueError):
            communication_op_ext.layer_parallel_dp2tp_all2all(
                tensor, "layer", dim=0
            )
        with self.assertRaises(ValueError):
            communication_op_ext.layer_parallel_dp2tp_all2all(
                tensor, "layer", dim=2
            )
        with self.assertRaises(ValueError):
            communication_op_ext.layer_parallel_dp2tp_all2all(
                torch.randn(2, 3), "layer", dim=1
            )

    @patch("omni.v1.distributed.communication_op_ext.dist.get_world_size", return_value=2)
    @patch("omni.v1.distributed.communication_op_ext.dist.all_to_all_single")
    @patch("omni.v1.distributed.communication_op_ext._get_group")
    def test_layer_parallel_dp2tp_all2all_uses_fallback_world_size_and_layout(
        self, mock_get_group, mock_all_to_all_single, mock_get_world_size
    ):
        group = Mock()
        group.world_size = None
        group.device_group = object()
        mock_get_group.return_value = group
        tensor = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

        def _copy_input(out_buf, in_buf, group=None):  # noqa: ARG001
            out_buf.copy_(in_buf)

        mock_all_to_all_single.side_effect = _copy_input
        result = communication_op_ext.layer_parallel_dp2tp_all2all(
            tensor, "layer", dim=1
        )

        expected = tensor.reshape(2, 2, 2, 3).movedim(1, 0).reshape(4, 2, 3)
        torch.testing.assert_close(result, expected)
        mock_get_world_size.assert_called_once_with(group=group.device_group)
        mock_all_to_all_single.assert_called_once()


class TestParallelStateExtensions(unittest.TestCase):
    """Unit tests for parallel_state helper functions."""

    def setUp(self):
        self._orig_layer_comm_dict = parallel_state_ext._LAYER_COMM_DICT
        self._orig_local_world = parallel_state_ext._LOCAL_WORLD
        self._orig_group_cache = dict(parallel_state_ext._TP_SIZE_OR_RANKS_GROUP_CACHE)
        parallel_state_ext._LAYER_COMM_DICT = None
        parallel_state_ext._LOCAL_WORLD = None
        parallel_state_ext._clear_tp_size_or_ranks_group_cache()

    def tearDown(self):
        parallel_state_ext._LAYER_COMM_DICT = self._orig_layer_comm_dict
        parallel_state_ext._LOCAL_WORLD = self._orig_local_world
        parallel_state_ext._TP_SIZE_OR_RANKS_GROUP_CACHE.clear()
        parallel_state_ext._TP_SIZE_OR_RANKS_GROUP_CACHE.update(self._orig_group_cache)

    def test_normalize_comm_op_type_aliases_and_canonical(self):
        mapping = {
            None: "NoOp",
            "NoOp": "NoOp",
            "noop": "NoOp",
            "no-op": "NoOp",
            "ALL2ALL": "ALL2ALL",
            "all_to_all": "ALL2ALL",
            "AllReduce": "AllReduce",
            "ALL-REDUCE": "AllReduce",
            "AllGather": "AllGather",
            "reduce_scatter": "ReduceScatter",
            "dp2tp_all2all": "DP2TPAll2All",
            "unexpected": "NoOp",
            "": "NoOp",
        }

        for raw, expected in mapping.items():
            self.assertEqual(
                parallel_state_ext._normalize_comm_op_type(raw),
                expected,
                msg=f"{raw} -> {expected}",
            )

    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=4)
    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_enforces_full_coverage(
        self, mock_is_initialized, mock_get_world
    ):
        spec = [[0, 1], [2, 3]]
        result = parallel_state_ext._tp_size_or_ranks_to_group_ranks(spec, "layer")
        self.assertEqual(result, spec)
        mock_is_initialized.assert_called_once()

    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=3)
    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_requires_all_ranks(self, *_):
        spec = [[0, 1]]
        with self.assertRaises(RuntimeError) as context:
            parallel_state_ext._tp_size_or_ranks_to_group_ranks(spec, "layer")
        self.assertIn("must cover all ranks", str(context.exception))

    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=32)
    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_detects_duplicates(self, *_):
        spec = [[0, 1], [1, 2]]
        with self.assertRaises(RuntimeError) as context:
            parallel_state_ext._tp_size_or_ranks_to_group_ranks(spec, "layer")
        self.assertIn("duplicate ranks across groups", str(context.exception))

    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=8)
    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_int_is_tp_group_size(self, *_):
        # int spec means tp_size (same semantics as vLLM's tensor_parallel_size).
        spec = 4
        result = parallel_state_ext._tp_size_or_ranks_to_group_ranks(spec, "layer")
        self.assertEqual(result, [[0, 1, 2, 3], [4, 5, 6, 7]])

    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=6)
    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_int_requires_divisible_world(self, *_):
        with self.assertRaises(RuntimeError) as context:
            parallel_state_ext._tp_size_or_ranks_to_group_ranks(4, "layer")
        self.assertIn("must be divisible", str(context.exception))

    def test_calculate_effective_local_size(self):
        self.assertEqual(parallel_state_ext.calculate_effective_local_size(8, 4), 4)
        with self.assertRaises(AssertionError):
            parallel_state_ext.calculate_effective_local_size(3, 5)

    @patch(
        "omni.v1.distributed.parallel_state_ext.torch.distributed.is_initialized",
        return_value=False,
    )
    def test_initialize_local_world_group_requires_dist_initialized(self, _):
        with self.assertRaises(RuntimeError):
            parallel_state_ext.initialize_local_world_group()

    @patch(
        "omni.v1.distributed.parallel_state_ext.torch.distributed.is_initialized",
        return_value=True,
    )
    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=32)
    def test_initialize_local_world_group_noop_if_already_initialized(
        self, mock_get_world_size, _
    ):
        parallel_state_ext._LOCAL_WORLD = Mock()
        parallel_state_ext.initialize_local_world_group()
        mock_get_world_size.assert_not_called()

    @patch(
        "omni.v1.distributed.parallel_state_ext.torch.distributed.is_initialized",
        return_value=True,
    )
    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=4)
    @patch(
        "omni.v1.distributed.parallel_state_ext.calculate_effective_local_size",
        return_value=2,
    )
    @patch("omni.v1.distributed.parallel_state_ext.init_model_parallel_group")
    @patch("omni.v1.distributed.parallel_state_ext.get_world_group")
    @patch(
        "omni.v1.distributed.parallel_state_ext.torch.distributed.get_backend",
        return_value="hccl",
    )
    def test_initialize_local_world_group_uses_visible_devices_when_mock_enabled(
        self,
        _,
        mock_get_world_group,
        mock_init_group,
        __,
        ___,
        ____,
    ):
        mock_world_group = Mock()
        mock_world_group.device_group = object()
        mock_world_group.local_rank = 0
        mock_get_world_group.return_value = mock_world_group
        local_world_group = Mock()
        mock_init_group.return_value = local_world_group

        with patch.dict(
            "os.environ",
            {"NO_NPU_MOCK": "1", "ASCEND_RT_VISIBLE_DEVICES": "0,1"},
            clear=False,
        ):
            parallel_state_ext.initialize_local_world_group()

        self.assertIs(parallel_state_ext._LOCAL_WORLD, local_world_group)
        mock_init_group.assert_called_once_with(
            [[0, 1], [2, 3]],
            0,
            "hccl",
            use_message_queue_broadcaster=True,
            group_name="world_local",
        )

    @patch(
        "omni.v1.distributed.parallel_state_ext.torch.distributed.is_initialized",
        return_value=True,
    )
    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=32)
    @patch(
        "omni.v1.distributed.parallel_state_ext.calculate_effective_local_size",
        side_effect=lambda local_size, world_size: local_size,
    )
    @patch("omni.v1.distributed.parallel_state_ext.init_model_parallel_group")
    @patch("omni.v1.distributed.parallel_state_ext.get_world_group")
    @patch(
        "omni.v1.distributed.parallel_state_ext.torch.distributed.get_backend",
        return_value="hccl",
    )
    def test_initialize_local_world_group_uses_num_servers_for_decode(
        self,
        _,
        mock_get_world_group,
        mock_init_group,
        __,
        ___,
        ____,
    ):
        mock_world_group = Mock()
        mock_world_group.device_group = object()
        mock_world_group.local_rank = 0
        mock_get_world_group.return_value = mock_world_group
        local_world_group = Mock()
        mock_init_group.return_value = local_world_group

        with patch.dict(
            "os.environ",
            {"NUM_SERVERS": "16", "NO_NPU_MOCK": "0"},
            clear=False,
        ):
            parallel_state_ext.initialize_local_world_group()

        self.assertIs(parallel_state_ext._LOCAL_WORLD, local_world_group)
        mock_init_group.assert_called_once_with(
            [list(range(i * 16, (i + 1) * 16)) for i in range(2)],
            0,
            "hccl",
            use_message_queue_broadcaster=True,
            group_name="world_local",
        )

    def test_get_local_world_group_raises_when_uninitialized(self):
        parallel_state_ext._LOCAL_WORLD = None
        with self.assertRaises(RuntimeError):
            parallel_state_ext.get_local_world_group()

    def test_get_moe_dispatch_ep_group_raises_when_uninitialized(self):
        with patch.object(
            parallel_state_ext.parallel_state,
            "_MOE_DISPATCH_EP",
            None,
            create=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "_MOE_DISPATCH_EP is None"):
                parallel_state_ext.get_moe_dispatch_ep_group()

    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=False)
    def test_ensure_layer_parallel_initialized_noop_when_dist_not_ready(self, _):
        parallel_state_ext.ensure_layer_parallel_initialized()
        self.assertEqual(parallel_state_ext._LAYER_COMM_DICT, {})

    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    @patch(
        "omni.v1.distributed.parallel_state_ext.initialize_local_world_group",
        return_value=None,
    )
    @patch(
        "omni.v1.distributed.parallel_state_ext._load_layer_parallel_config_from_model_extra_config",
        return_value=None,
    )
    def test_ensure_layer_parallel_initialized_with_no_model_config(self, *_):
        parallel_state_ext.ensure_layer_parallel_initialized()
        self.assertEqual(parallel_state_ext._LAYER_COMM_DICT, {})

    @patch("omni.v1.distributed.parallel_state_ext.get_tp_group")
    def test_get_layer_parallel_group_fallback_order(self, mock_get_tp_group):
        tp_group = Mock(name="tp")
        layer_group = Mock(name="layer")
        x_group = Mock(name="x")
        mock_get_tp_group.return_value = tp_group
        parallel_state_ext._LAYER_COMM_DICT = {
            "layer": {
                "parallel_group": layer_group,
                "x_transform": {"parallel_group": x_group},
            }
        }

        self.assertIs(parallel_state_ext.get_layer_parallel_group("layer", "x"), x_group)
        self.assertIs(parallel_state_ext.get_layer_parallel_group("layer", "y"), layer_group)
        self.assertIs(parallel_state_ext.get_layer_parallel_group("other"), tp_group)

    @patch("omni.v1.distributed.parallel_state_ext.get_tp_group", side_effect=AssertionError)
    def test_get_layer_parallel_group_handles_uninitialized_tp_group(self, _):
        parallel_state_ext._LAYER_COMM_DICT = None
        self.assertIsNone(parallel_state_ext.get_layer_parallel_group("layer"))

    def test_get_layer_transform_type_and_dim_default_and_configured(self):
        self.assertEqual(parallel_state_ext.get_layer_transform_type("layer", "x"), "NoOp")
        self.assertEqual(parallel_state_ext.get_layer_dim("layer", "x"), 0)

        parallel_state_ext._LAYER_COMM_DICT = {
            "layer": {"x_transform": {"type": "AllGather", "dim": 2}}
        }
        self.assertEqual(parallel_state_ext.get_layer_transform_type("layer", "x"), "AllGather")
        self.assertEqual(parallel_state_ext.get_layer_dim("layer", "x"), 2)

    def test_maybe_pad_and_slice_invalid_dim(self):
        with self.assertRaises(ValueError):
            parallel_state_ext.maybe_pad_and_slice(torch.randn(2), dim=2, layer_name_inside_block="x")

    @patch(
        "omni.v1.distributed.parallel_state_ext.is_layer_parallel_input_split_enabled",
        return_value=False,
    )
    def test_maybe_pad_and_slice_returns_input_when_split_disabled(self, _):
        tensor = torch.randn(2, 3)
        out, original = parallel_state_ext.maybe_pad_and_slice(
            tensor, dim=1, layer_name_inside_block="layer"
        )
        self.assertIs(out, tensor)
        self.assertEqual(original, tensor.shape[1])

    def test_maybe_pad_and_slice_normalizes_negative_dim(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        with patch(
            "omni.v1.distributed.parallel_state_ext.is_layer_parallel_input_split_enabled",
            return_value=True,
        ), patch(
            "omni.v1.distributed.parallel_state_ext.get_layer_parallel_world_size",
            return_value=1,
        ):
            out, original = parallel_state_ext.maybe_pad_and_slice(
                tensor, dim=-1, layer_name_inside_block="layer"
            )
        self.assertIs(out, tensor)
        self.assertEqual(original, 3)

    def test_maybe_unpad_and_all_gather_invalid_dim(self):
        with self.assertRaises(ValueError):
            parallel_state_ext.maybe_unpad_and_all_gather(
                torch.randn(2), actual_length=2, dim=3, layer_name_inside_block="x"
            )

    @patch(
        "omni.v1.distributed.parallel_state_ext.is_layer_parallel_input_split_enabled",
        return_value=True,
    )
    @patch("omni.v1.distributed.parallel_state_ext.get_layer_parallel_group", return_value=None)
    def test_maybe_unpad_and_all_gather_returns_input_without_group(self, *_):
        tensor = torch.randn(2, 3)
        out = parallel_state_ext.maybe_unpad_and_all_gather(
            tensor, actual_length=3, dim=1, layer_name_inside_block="layer"
        )
        self.assertIs(out, tensor)

    def test_maybe_unpad_and_all_gather_normalizes_negative_dim_without_split(self):
        tensor = torch.randn(2, 3)
        with patch(
            "omni.v1.distributed.parallel_state_ext.is_layer_parallel_input_split_enabled",
            return_value=False,
        ):
            out = parallel_state_ext.maybe_unpad_and_all_gather(
                tensor, actual_length=3, dim=-1, layer_name_inside_block="layer"
            )
        self.assertIs(out, tensor)

    @patch(
        "omni.v1.distributed.parallel_state_ext.is_layer_parallel_input_split_enabled",
        return_value=True,
    )
    def test_maybe_unpad_and_all_gather_returns_input_without_layer_name(self, _):
        tensor = torch.randn(2, 3)
        out = parallel_state_ext.maybe_unpad_and_all_gather(
            tensor, actual_length=2, dim=0, layer_name_inside_block=None
        )
        self.assertIs(out, tensor)

    def test_parse_tensor_transform_cfg(self):
        self.assertIsNone(
            parallel_state_ext._parse_tensor_transform_cfg(
                transform_cfg=None,
                local_rank=0,
                backend="hccl",
                group_name="group",
            )
        )

        with patch(
            "omni.v1.distributed.parallel_state_ext._create_group_from_tp_size_or_ranks",
            return_value="group_obj",
        ):
            result = parallel_state_ext._parse_tensor_transform_cfg(
                transform_cfg={"type": "all_to_all", "dim": "2", "tp_size_or_ranks": [[0]]},
                local_rank=0,
                backend="hccl",
                group_name="group",
            )
        self.assertEqual(
            result,
            {"type": "ALL2ALL", "dim": 2, "parallel_group": "group_obj"},
        )

    def test_create_group_from_tp_size_or_ranks(self):
        with patch(
            "omni.v1.distributed.parallel_state_ext._tp_size_or_ranks_to_group_ranks",
            return_value=None,
        ), patch("omni.v1.distributed.parallel_state_ext.init_model_parallel_group") as mock_init:
            self.assertIsNone(
                parallel_state_ext._create_group_from_tp_size_or_ranks(
                    tp_size_or_ranks=None,
                    local_rank=0,
                    backend="hccl",
                    group_name="g",
                )
            )
            mock_init.assert_not_called()

        with patch(
            "omni.v1.distributed.parallel_state_ext._tp_size_or_ranks_to_group_ranks",
            return_value=[[0]],
        ), patch(
            "omni.v1.distributed.parallel_state_ext.init_model_parallel_group",
            return_value="created_group",
        ) as mock_init:
            result = parallel_state_ext._create_group_from_tp_size_or_ranks(
                tp_size_or_ranks=[[0]],
                local_rank=0,
                backend="hccl",
                group_name="g",
            )
            self.assertEqual(result, "created_group")
            mock_init.assert_called_once()

    def test_create_group_from_tp_size_or_ranks_reuses_same_group_ranks(self):
        group_ranks = [[0, 1], [2, 3]]
        with patch(
            "omni.v1.distributed.parallel_state_ext._tp_size_or_ranks_to_group_ranks",
            return_value=group_ranks,
        ), patch(
            "omni.v1.distributed.parallel_state_ext.init_model_parallel_group",
            return_value="created_group",
        ) as mock_init:
            first = parallel_state_ext._create_group_from_tp_size_or_ranks(
                tp_size_or_ranks=4,
                local_rank=0,
                backend="hccl",
                group_name="layer_mlp.gate_up_proj",
            )
            second = parallel_state_ext._create_group_from_tp_size_or_ranks(
                tp_size_or_ranks=4,
                local_rank=0,
                backend="hccl",
                group_name="layer_mlp.down_proj",
            )
            self.assertIs(first, second)
            mock_init.assert_called_once()

    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=False)
    def test_tp_size_or_ranks_list_requires_initialized_dist(self, _):
        with self.assertRaises(RuntimeError):
            parallel_state_ext._tp_size_or_ranks_to_group_ranks([0, 1], "layer")

    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=32)
    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_rejects_invalid_nested_types(self, *_):
        with self.assertRaises(RuntimeError):
            parallel_state_ext._tp_size_or_ranks_to_group_ranks([0, [1]], "layer")

    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=32)
    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_rejects_duplicate_within_group(self, *_):
        with self.assertRaises(RuntimeError):
            parallel_state_ext._tp_size_or_ranks_to_group_ranks([[0, 0], [1, 2]], "layer")

    @patch("omni.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=32)
    @patch("omni.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_rejects_rank_out_of_range(self, *_):
        with self.assertRaises(RuntimeError):
            parallel_state_ext._tp_size_or_ranks_to_group_ranks([[0, 4], [1, 2]], "layer")

    def test_tp_size_or_ranks_unsupported_type_returns_none(self):
        self.assertIsNone(parallel_state_ext._tp_size_or_ranks_to_group_ranks("invalid", "layer"))

    def test_ensure_layer_parallel_initialized_uses_passed_backend(self):
        # Force re-init within this test.
        parallel_state_ext._LAYER_COMM_DICT = None

        vllm_config = Mock()
        vllm_config.parallel_config = Mock(local_rank=0)
        layer_parallel_config = {
            "self_attn.q_proj": {"tp_size_or_ranks": [[0]]},
        }

        with patch(
            "omni.v1.distributed.parallel_state_ext._load_layer_parallel_config_from_model_extra_config",
            return_value={"layer_parallel_config": layer_parallel_config},
        ), patch(
            "omni.v1.distributed.parallel_state_ext.get_current_vllm_config",
            return_value=vllm_config,
        ), patch(
            "omni.v1.distributed.parallel_state_ext.dist.is_initialized",
            return_value=True,
        ), patch(
            "omni.v1.distributed.parallel_state_ext.initialize_local_world_group",
            return_value=None,
        ), patch(
            "omni.v1.distributed.parallel_state_ext._create_group_from_tp_size_or_ranks",
            return_value=Mock(),
        ) as mock_create_group:
            parallel_state_ext.ensure_layer_parallel_initialized(backend="hccl")
            mock_create_group.assert_called()
            self.assertEqual(mock_create_group.call_args.args[2], "hccl")

    def test_openpangu_o_proj_dp2tp_all2all_config_initializes_transforms(self):
        config_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "omni_npu"
            / "model_config"
            / "configs"
            / "low_latency"
            / "openpangu_v2"
            / "pangu_v2_moe_bf16_a3_xp1d_d.json"
        )
        layer_parallel_config = json.loads(
            config_path.read_text(encoding="utf-8")
        )["model_parallel_config"]["layer_parallel_config"]
        vllm_config = Mock()
        vllm_config.parallel_config = Mock(local_rank=0)
        layer_group, x_group, y_group = Mock(), Mock(), Mock()
        groups = {
            "layer_self_attn.o_proj": layer_group,
            "layer_self_attn.o_proj_x_transform": x_group,
            "layer_self_attn.o_proj_y_transform": y_group,
        }

        with patch(
            "omni.v1.distributed.parallel_state_ext._load_layer_parallel_config_from_model_extra_config",
            return_value={"layer_parallel_config": layer_parallel_config},
        ), patch(
            "omni.v1.distributed.parallel_state_ext.get_current_vllm_config",
            return_value=vllm_config,
        ), patch(
            "omni.v1.distributed.parallel_state_ext.dist.is_initialized",
            return_value=True,
        ), patch(
            "omni.v1.distributed.parallel_state_ext.initialize_local_world_group",
        ), patch(
            "omni.v1.distributed.parallel_state_ext._create_group_from_tp_size_or_ranks",
            side_effect=lambda _spec, _rank, _backend, name: groups[name],
        ):
            parallel_state_ext.ensure_layer_parallel_initialized(backend="hccl")

        self.assertEqual(
            parallel_state_ext.get_layer_transform_type("self_attn.o_proj", "x"),
            "DP2TPAll2All",
        )
        self.assertEqual(
            parallel_state_ext.get_layer_dim("self_attn.o_proj", "x"), -1
        )
        self.assertEqual(
            parallel_state_ext.get_layer_transform_type("self_attn.o_proj", "y"),
            "ReduceScatter",
        )
        self.assertIs(
            parallel_state_ext.get_layer_parallel_group("self_attn.o_proj", "x"),
            x_group,
        )
        self.assertIs(
            parallel_state_ext.get_layer_parallel_group("self_attn.o_proj", "y"),
            y_group,
        )

    @patch(
        "omni.v1.distributed.parallel_state_ext.is_layer_parallel_input_split_enabled",
        return_value=True,
    )
    @patch(
        "omni.v1.distributed.parallel_state_ext.get_layer_parallel_world_size",
        return_value=2,
    )
    @patch(
        "omni.v1.distributed.parallel_state_ext.get_layer_parallel_rank",
        return_value=1,
    )
    def test_maybe_pad_and_slice_pads_and_slices(
        self, mock_get_rank, mock_get_world_size, mock_input_split
    ):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result, original = parallel_state_ext.maybe_pad_and_slice(
            tensor, dim=0, layer_name_inside_block="self_attn.q_proj"
        )

        self.assertEqual(original, 3)
        torch.testing.assert_close(result, torch.tensor([3.0, 0.0]))
        mock_input_split.assert_called_once()
        mock_get_world_size.assert_called_once()
        mock_get_rank.assert_called_once()

    @patch(
        "omni.v1.distributed.parallel_state_ext.is_layer_parallel_input_split_enabled",
        return_value=True,
    )
    @patch("omni.v1.distributed.parallel_state_ext.get_layer_parallel_group")
    def test_maybe_unpad_and_all_gather_trims_padding(
        self, mock_get_group, mock_input_split
    ):
        group = Mock()
        group.world_size = 2
        group.all_gather.return_value = torch.tensor([1.0, 2.0, 3.0, 0.0])
        mock_get_group.return_value = group

        local = torch.tensor([1.0, 2.0])
        result = parallel_state_ext.maybe_unpad_and_all_gather(
            local, actual_length=3, dim=0, layer_name_inside_block="self_attn.q_proj"
        )

        group.all_gather.assert_called_once_with(local, dim=0)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))
        mock_input_split.assert_called_once()

