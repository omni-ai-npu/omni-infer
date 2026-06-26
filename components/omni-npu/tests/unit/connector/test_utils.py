# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import pytest
import torch
from unittest.mock import patch, MagicMock
from omni_npu.connector.utils import get_p_start_rank, get_config_from_dict_or_env, TP_Convertor


class TestGetPStartRank:
    """Test get_p_start_rank function with various scenarios"""

    def test_valid_parameters(self):
        """Test with valid parameters that should succeed"""
        result = get_p_start_rank(p_tp_size=4,
                                  p_dp_size=1,
                                  d_tp_size=2,
                                  d_dp_size=2,
                                  d_node_num=2,
                                  cur_d_node=0,
                                  cur_d_rank=0)
        assert isinstance(result, int)
        assert result >= 0

        # Test another valid combination
        result2 = get_p_start_rank(p_tp_size=8,
                                   p_dp_size=1,
                                   d_tp_size=4,
                                   d_dp_size=1,
                                   d_node_num=1,
                                   cur_d_node=0,
                                   cur_d_rank=0)
        assert isinstance(result2, int)
        assert result2 >= 0

    def test_p_dp_size_not_1(self):
        """Test that p_dp_size must be 1"""
        with pytest.raises(ValueError, match="p_dp_size must be 1"):
            get_p_start_rank(p_tp_size=4,
                             p_dp_size=2,
                             d_tp_size=2,
                             d_dp_size=2,
                             d_node_num=2,
                             cur_d_node=0,
                             cur_d_rank=0)

    def test_negative_p_tp_size(self):
        """Test negative p_tp_size"""
        with pytest.raises(
                ValueError,
                match=
                "p_tp_size, d_tp_size, d_dp_size, d_node_num must be positive"
        ):
            get_p_start_rank(p_tp_size=-1,
                             p_dp_size=1,
                             d_tp_size=2,
                             d_dp_size=2,
                             d_node_num=2,
                             cur_d_node=0,
                             cur_d_rank=0)

    def test_zero_d_tp_size(self):
        """Test zero d_tp_size"""
        with pytest.raises(
                ValueError,
                match=
                "p_tp_size, d_tp_size, d_dp_size, d_node_num must be positive"
        ):
            get_p_start_rank(p_tp_size=4,
                             p_dp_size=1,
                             d_tp_size=0,
                             d_dp_size=2,
                             d_node_num=2,
                             cur_d_node=0,
                             cur_d_rank=0)

    def test_negative_cur_d_node(self):
        """Test negative current decode node"""
        with pytest.raises(ValueError,
                           match="cur_d_node < 0 or cur_d_node >= d_node_num"):
            get_p_start_rank(p_tp_size=4,
                             p_dp_size=1,
                             d_tp_size=2,
                             d_dp_size=2,
                             d_node_num=2,
                             cur_d_node=-1,
                             cur_d_rank=0)

    def test_cur_d_node_out_of_range(self):
        """Test current decode node out of range"""
        with pytest.raises(ValueError,
                           match="cur_d_node < 0 or cur_d_node >= d_node_num"):
            get_p_start_rank(p_tp_size=4,
                             p_dp_size=1,
                             d_tp_size=2,
                             d_dp_size=2,
                             d_node_num=2,
                             cur_d_node=2,
                             cur_d_rank=0)

    def test_negative_cur_d_rank(self):
        """Test negative current decode rank"""
        with pytest.raises(ValueError, match="cur_d_rank < 0"):
            get_p_start_rank(p_tp_size=4,
                             p_dp_size=1,
                             d_tp_size=2,
                             d_dp_size=2,
                             d_node_num=2,
                             cur_d_node=0,
                             cur_d_rank=-1)

    def test_cur_d_rank_out_of_range(self):
        """Test current decode rank out of range"""
        with pytest.raises(ValueError, match="cur_d_rank >= devices_per_node"):
            get_p_start_rank(p_tp_size=4,
                             p_dp_size=1,
                             d_tp_size=2,
                             d_dp_size=2,
                             d_node_num=2,
                             cur_d_node=0,
                             cur_d_rank=5)

    def test_p_tp_size_not_divisible_by_kv_group_size(self):
        """Test when p_tp_size is not divisible by kv_group_size"""
        with pytest.raises(ValueError, match="p_tp_size % kv_group_size != 0"):
            get_p_start_rank(p_tp_size=5,
                             p_dp_size=1,
                             d_tp_size=2,
                             d_dp_size=2,
                             d_node_num=2,
                             cur_d_node=0,
                             cur_d_rank=0)

    def test_edge_case_single_device(self):
        """Test edge case with single device"""
        result = get_p_start_rank(p_tp_size=1,
                                  p_dp_size=1,
                                  d_tp_size=1,
                                  d_dp_size=1,
                                  d_node_num=1,
                                  cur_d_node=0,
                                  cur_d_rank=0)
        assert result == 0

    def test_complex_scenario(self):
        """Test complex scenario with multiple nodes and devices"""
        result = get_p_start_rank(p_tp_size=16,
                                  p_dp_size=1,
                                  d_tp_size=4,
                                  d_dp_size=2,
                                  d_node_num=4,
                                  cur_d_node=1,
                                  cur_d_rank=3)
        assert isinstance(result, int)
        assert 0 <= result < 16


class TestGetConfigFromDictOrEnv:
    """Test get_config_from_dict_or_env function with various scenarios"""

    def test_env_variable_priority(self):
        """Test that environment variable has highest priority"""
        with patch.dict(os.environ, {'TEST_VAR': 'env_value'}):
            config = {'test_var': 'dict_value'}
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 'default_value',
                                                 str)
            assert result == 'env_value'

    def test_dict_config_when_no_env(self):
        """Test dictionary config when no environment variable"""
        with patch.dict(os.environ, {}, clear=True):
            config = {'test_var': 'dict_value'}
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 'default_value',
                                                 str)
            assert result == 'dict_value'

    def test_object_config_when_no_env(self):
        """Test object config when no environment variable"""
        with patch.dict(os.environ, {}, clear=True):

            class ConfigObject:

                def __init__(self):
                    self.test_var = 'object_value'

            config = ConfigObject()
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 'default_value',
                                                 str)
            assert result == 'object_value'

    def test_default_value_when_no_env_or_config(self):
        """Test default value when no environment variable or config"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 'default_value',
                                                 str)
            assert result == 'default_value'

    def test_no_default_value_raises_error(self):
        """Test that error is raised when no value found and no default"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            with pytest.raises(
                    ValueError,
                    match="ENV TEST_VAR or args test_var should not be None"):
                get_config_from_dict_or_env(config, 'test_var', 'TEST_VAR',
                                            None, str)

    def test_type_conversion(self):
        """Test that value is converted to specified type"""
        with patch.dict(os.environ, {'TEST_VAR': '42'}):
            config = {}
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 0, int)
            assert result == 42
            assert isinstance(result, int)

    def test_type_conversion_with_default(self):
        """Test type conversion with default value"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', '100', int)
            assert result == 100
            assert isinstance(result, int)

    def test_empty_dict_config(self):
        """Test with empty dictionary config"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 'default', str)
            assert result == 'default'

    def test_none_config_object(self):
        """Test with None config object"""
        with patch.dict(os.environ, {}, clear=True):
            config = None
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 'default', str)
            assert result == 'default'

    def test_config_object_without_attribute(self):
        """Test config object without the requested attribute"""
        with patch.dict(os.environ, {}, clear=True):

            class ConfigObject:

                def __init__(self):
                    self.other_var = 'other_value'

            config = ConfigObject()
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 'default_value',
                                                 str)
            assert result == 'default_value'


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_get_p_start_rank_large_numbers(self):
        """Test get_p_start_rank with large numbers"""
        result = get_p_start_rank(p_tp_size=64,
                                  p_dp_size=1,
                                  d_tp_size=8,
                                  d_dp_size=4,
                                  d_node_num=8,
                                  cur_d_node=4,
                                  cur_d_rank=15)
        assert isinstance(result, int)
        assert 0 <= result < 64

    def test_get_config_from_dict_or_env_special_chars(self):
        """Test get_config_from_dict_or_env with special characters"""
        with patch.dict(os.environ, {'TEST_VAR': 'special@value#123'}):
            config = {}
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', 'default', str)
            assert result == 'special@value#123'

    def test_get_config_from_dict_or_env_boolean_type(self):
        """Test get_config_from_dict_or_env with boolean type conversion"""
        with patch.dict(os.environ, {'TEST_VAR': 'true'}):
            config = {}
            result = get_config_from_dict_or_env(config, 'test_var',
                                                 'TEST_VAR', False,
                                                 lambda x: x.lower() == 'true')
            assert result is True


class TestTPConvertor:
    """Test TP_Convertor class with various scenarios"""

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_init(self, mock_get_tp_group):
        """Test TP_Convertor initialization"""
        # Mock TP group
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 8
        mock_tp_group.rank_in_group = 2
        mock_get_tp_group.return_value = mock_tp_group

        # Test valid initialization
        convertor = TP_Convertor(remote_tp_size=2)
        assert convertor.tp_comm == "mock_comm"
        assert convertor.tp_size == 8
        assert convertor.tp_rank == 2
        assert convertor.remote_tp_size == 2
        assert convertor.stride == 4  # 8 / 2
        assert convertor.offset == 2 % 4
        assert convertor.transfer_done is False

        # Test assertion when tp_size % remote_tp_size != 0
        mock_tp_group.world_size = 7
        with pytest.raises(AssertionError):
            TP_Convertor(remote_tp_size=2)

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_scheme_reorg_stride_1(self, mock_get_tp_group):
        """Test scheme_reorg when stride is 1"""
        # Mock TP group with stride 1
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 2
        mock_tp_group.rank_in_group = 0
        mock_get_tp_group.return_value = mock_tp_group

        convertor = TP_Convertor(remote_tp_size=2)
        token_num = 100
        tail_blk = 5
        kv_group = []
        local_block_ids = [1, 2, 3]
        remote_block_ids = [4, 5, 6]

        # Should return remote_block_ids unchanged when stride == 1
        result = convertor.scheme_reorg(token_num, tail_blk, kv_group,
                                        local_block_ids, remote_block_ids)
        assert result == remote_block_ids

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_scheme_reorg_with_stride(self, mock_get_tp_group):
        """Test scheme_reorg when stride > 1"""
        # Mock TP group with stride > 1
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 8
        mock_tp_group.rank_in_group = 2
        mock_get_tp_group.return_value = mock_tp_group

        convertor = TP_Convertor(remote_tp_size=2)
        token_num = 100
        tail_blk = 5

        # Create mock KV group
        kv_group = [[torch.randn(10, 128, 64) for _ in range(2)]
                    for _ in range(4)]
        local_block_ids = [1, 2, 3, 4]
        remote_block_ids = [10, 20, 30, 40]

        # Test scheme_reorg with stride > 1
        result = convertor.scheme_reorg(token_num, tail_blk, kv_group,
                                        local_block_ids, remote_block_ids)

        # Verify result has same length as local_block_ids
        assert len(result) == len(local_block_ids)

        # Verify attributes are set correctly
        assert hasattr(convertor, 'tail_blk')
        assert hasattr(convertor, 'kv_group')
        assert hasattr(convertor, 'send_domain')
        assert hasattr(convertor, 'recv_domain')
        assert hasattr(convertor, 'recv_num')
        assert hasattr(convertor, 'send_split')
        assert hasattr(convertor, 'recv_split')

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_token_reorg_stride_1(self, mock_get_tp_group):
        """Test token_reorg when stride is 1"""
        # Mock TP group with stride 1
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 2
        mock_tp_group.rank_in_group = 0
        mock_get_tp_group.return_value = mock_tp_group

        convertor = TP_Convertor(remote_tp_size=2)
        convertor.token_reorg()  # Should do nothing when stride == 1

    @patch('torch.distributed.all_to_all_single')
    @patch('omni_npu.connector.utils.get_tp_group')
    def test_token_reorg_with_stride(self, mock_get_tp_group,
                                     mock_all_to_all_single):
        """Test token_reorg when stride > 1"""
        # Mock TP group with stride > 1
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 8
        mock_tp_group.rank_in_group = 2
        mock_get_tp_group.return_value = mock_tp_group

        convertor = TP_Convertor(remote_tp_size=2)

        # Set up attributes needed for token_reorg
        token_num = 100
        tail_blk = 5
        kv_group = [[torch.randn(10, 128, 64) for _ in range(2)]]
        local_block_ids = [1, 2, 3, 4]
        remote_block_ids = [10, 20, 30, 40]

        convertor.scheme_reorg(token_num, tail_blk, kv_group, local_block_ids,
                               remote_block_ids)

        # Mock recv tensor with correct dimensions
        mock_recv = MagicMock()
        mock_recv.dim.return_value = 3
        mock_recv.size.return_value = (10, 2, 64)
        mock_recv.transpose.return_value = mock_recv

        with patch('torch.Tensor.new_empty', return_value=mock_recv):
            with patch.object(TP_Convertor, 'store_kv') as mock_store_kv:
                convertor.token_reorg()

                # Verify all_to_all_single was called
                mock_all_to_all_single.assert_called_once()
                # Verify store_kv was called
                mock_store_kv.assert_called_once()

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_scheduled_list(self, mock_get_tp_group):
        """Test scheduled_list class method"""
        # Mock TP group for different ranks
        mock_tp_group1 = MagicMock()
        mock_tp_group1.rank_in_group = 0
        mock_tp_group2 = MagicMock()
        mock_tp_group2.rank_in_group = 1

        mock_get_tp_group.side_effect = [mock_tp_group1, mock_tp_group2]

        # Test that different ranks have different scheduled lists
        list_rank0 = TP_Convertor.scheduled_list()
        list_rank1 = TP_Convertor.scheduled_list()

        assert list_rank0 is not list_rank1
        assert isinstance(list_rank0, list)
        assert isinstance(list_rank1, list)

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_tail_blk_num(self, mock_get_tp_group):
        """Test tail_blk_num static method"""
        # Test various scenarios
        num = 100
        d_rank = 2
        d_size = 8
        p_size = 2
        pg = 128

        before, after = TP_Convertor.tail_blk_num(num, d_rank, d_size, p_size,
                                                  pg)
        assert isinstance(before, int)
        assert isinstance(after, int)
        assert before >= 0
        assert after >= 0

        # Test edge case with zero
        before_zero, after_zero = TP_Convertor.tail_blk_num(0, 0, 8, 2, 128)
        assert before_zero == 0
        assert after_zero == 0

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_link_to_remote(self, mock_get_tp_group):
        """Test link_to_remote static method"""
        # Test valid cases
        assert TP_Convertor.link_to_remote(0, 8, 2) == 0
        assert TP_Convertor.link_to_remote(3, 8, 2) == 0
        assert TP_Convertor.link_to_remote(4, 8, 2) == 1
        assert TP_Convertor.link_to_remote(7, 8, 2) == 1

        # Test assertion when tp_size % remote_tp_size != 0
        with pytest.raises(AssertionError):
            TP_Convertor.link_to_remote(0, 7, 2)

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_a2a_mapper(self, mock_get_tp_group):
        """Test a2a_mapper static method"""
        # Test balanced movement
        movement = [2, -1, -1]
        result = TP_Convertor.a2a_mapper(movement)
        assert result.shape == (3, 3)
        assert result.sum() == 2

        # Test complex movement
        movement = [3, -2, 1, -2]
        result = TP_Convertor.a2a_mapper(movement)
        assert result.shape == (4, 4)
        assert result.sum() == 4

        # Test zero movement
        movement = [0, 0, 0]
        result = TP_Convertor.a2a_mapper(movement)
        assert result.shape == (3, 3)
        assert result.sum() == 0

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_extract_kv(self, mock_get_tp_group):
        """Test extract_kv static method"""
        # Create mock KV tensors
        k = torch.randn(10, 128, 64)  # [blk, pg, D]
        v = torch.randn(10, 128, 64)
        kvs = [k, v]

        blk_i = 5
        domain = (10, 20)

        result = TP_Convertor.extract_kv(kvs, blk_i, domain)

        # Check shape: [T, N, D] where T=20-10=10, N=2, D=64
        assert result.shape == (10, 2, 64)

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_store_kv(self, mock_get_tp_group):
        """Test store_kv static method"""
        # Create mock destination KV tensors
        k = torch.zeros(10, 128, 64)  # [blk, pg, D]
        v = torch.zeros(10, 128, 64)
        kvs = [k, v]

        # Create source tensor
        T = 10
        N = 2
        D = 64
        x = torch.randn(T, N, D)  # [T, N, D]

        blk_i = 5
        domain = (10, 20)

        TP_Convertor.store_kv(x, kvs, blk_i, domain)

        # Check that destination tensors were updated
        assert not torch.allclose(k[blk_i][domain[0]:domain[1]],
                                  torch.zeros(T, D))
        assert not torch.allclose(v[blk_i][domain[0]:domain[1]],
                                  torch.zeros(T, D))

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_do_scheduled_kv_reorg_empty(self, mock_get_tp_group):
        """Test do_scheduled_kv_reorg with empty list"""
        # Mock TP group
        mock_tp_group = MagicMock()
        mock_tp_group.rank_in_group = 0
        mock_tp_group.all_gather.return_value = torch.tensor([0])
        mock_get_tp_group.return_value = mock_tp_group

        # Should not raise error when list is empty
        TP_Convertor.do_scheduled_kv_reorg()

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_do_scheduled_kv_reorg_with_items(self, mock_get_tp_group):
        """Test do_scheduled_kv_reorg with items in list"""
        # Mock TP group
        mock_tp_group = MagicMock()
        mock_tp_group.rank_in_group = 0
        mock_tp_group.all_gather.return_value = torch.tensor([2])
        mock_get_tp_group.return_value = mock_tp_group

        # Add items to scheduled list
        scheduled_list = TP_Convertor.scheduled_list()

        # Create mock convertors with actual token_reorg method mocked
        mock_convertor1 = MagicMock(spec=TP_Convertor)
        mock_convertor1.transfer_done = True
        mock_convertor1.kv_group = [[torch.randn(1, 1, 1)]]

        mock_convertor2 = MagicMock(spec=TP_Convertor)
        mock_convertor2.transfer_done = True
        mock_convertor2.kv_group = [[torch.randn(1, 1, 1)]]

        mock_convertor3 = MagicMock(spec=TP_Convertor)
        mock_convertor3.transfer_done = False
        mock_convertor3.kv_group = [[torch.randn(1, 1, 1)]]

        scheduled_list.extend(
            [mock_convertor1, mock_convertor2, mock_convertor3])

        TP_Convertor.do_scheduled_kv_reorg()

        # Verify token_reorg was called on completed items
        mock_convertor1.token_reorg.assert_called_once()
        mock_convertor2.token_reorg.assert_called_once()
        mock_convertor3.token_reorg.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
