import pytest
import torch
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from omni.layers.attention.backend import mla as mla_mod

@pytest.fixture
def metadata_builder():
    runner = MagicMock()
    runner.block_size = 128
    runner.max_num_reqs = 32
    runner.device = "npu:0"
    runner.dcp_kv_cache_interleave_size = 16
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=4096
        )
    )
    
    runner.model = MagicMock()
    runner.model.config = SimpleNamespace(
        num_experts_per_tok=2,
        n_routed_experts=8
    )
    
    with patch.object(mla_mod, 'get_tensor_model_parallel_world_size', return_value=2), \
         patch.object(mla_mod, 'get_tensor_model_parallel_rank', return_value=0), \
         patch.object(mla_mod, 'get_world_group') as mock_world_group:
        
        mock_world_group.return_value = SimpleNamespace(
            world_size=2,
            rank_in_group=0
        )
        
        builder = mla_mod.AscendMLAMetadataBuilder(
            runner=runner,
            kv_cache_spec=None,
            block_table=None,
            metadata_cls=None
        )
        
        yield builder

def test_get_context_kv_index_slices_basic(metadata_builder):
    builder = metadata_builder
    
    query_lens = [10, 15]  
    seq_lens = [50, 80]    
    chunksize = 32
    dcp_kv_cache_interleave_size = 16
    
    block_tables = [
        torch.tensor([1, 2, 3], dtype=torch.int32),  
        torch.tensor([4, 5, 6, 7], dtype=torch.int32) 
    ]
    
    kv_index_list, max_kv_index_list, kv_allgather_restore_index_list = \
        builder.get_context_kv_index_slices(
            query_lens, seq_lens, block_tables, chunksize, dcp_kv_cache_interleave_size
        )
    
    assert len(kv_index_list) == 2 
    assert len(max_kv_index_list) == 2
    assert len(kv_allgather_restore_index_list) == 2
    
    assert len(kv_index_list[0]) > 0
    assert len(max_kv_index_list[0]) > 0
    assert len(kv_allgather_restore_index_list[0]) > 0
    
    assert kv_index_list[0][0].device.type == 'npu'


def test_build_chunk_restore_index(metadata_builder):
    builder = metadata_builder
    
    chunk_size = 64
    tokens_per_rank = 32
    dcp_kv_cache_interleave_size = 16
    
    chunk_kv_index = torch.randint(0, 100, (32,), dtype=torch.long, device="npu:0")
    
    restore_indices = builder._build_chunk_restore_index(
        chunk_size, tokens_per_rank, chunk_kv_index, dcp_kv_cache_interleave_size
    )
    
    assert restore_indices is not None
    assert restore_indices.shape == (chunk_size,)
    assert restore_indices.dtype == torch.long
    assert restore_indices.device.type == 'npu'
    
    assert torch.all(restore_indices >= 0)
    assert torch.all(restore_indices < tokens_per_rank * 2)  # 2个rank
    
    with patch.object(builder.runner.vllm_config.parallel_config, 'decode_context_parallel_size', 4):
        restore_indices_4dcp = builder._build_chunk_restore_index(
            chunk_size, tokens_per_rank, chunk_kv_index, dcp_kv_cache_interleave_size
        )
        
        assert restore_indices_4dcp.shape == (chunk_size,)


def test_get_context_chunk_seq_lens(metadata_builder):
    builder = metadata_builder
    
    query_lens = [10, 20, 30]
    seq_lens = [50, 80, 120]  
    chunksize = 32
    
    chunk_seq_lens = builder.get_context_chunk_seq_lens(query_lens, seq_lens, chunksize)
    
    assert len(chunk_seq_lens) == 3  
    
    context_seq_len_0 = seq_lens[0] - query_lens[0] 
    expected_chunks_0 = [32, 8, 0]  
    assert chunk_seq_lens[0] == expected_chunks_0
    
    context_seq_len_1 = seq_lens[1] - query_lens[1]  
    expected_chunks_1 = [32, 28, 0] 
    assert chunk_seq_lens[1] == expected_chunks_1
    
    context_seq_len_2 = seq_lens[2] - query_lens[2] 
    expected_chunks_2 = [32, 32, 26]  
    assert chunk_seq_lens[2] == expected_chunks_2
    
    for i, chunks in enumerate(chunk_seq_lens):
        context_len = seq_lens[i] - query_lens[i]
        non_zero_chunks = [chunk for chunk in chunks if chunk > 0]
        assert sum(non_zero_chunks) == context_len, f"请求{i}的非零chunk总和应等于上下文长度"

def test_get_kv_index(metadata_builder):
    builder = metadata_builder
    
    seq_lens = [50, 80, 120]
    block_tables = [
        torch.tensor([1, 2, 3], dtype=torch.int32),
        torch.tensor([4, 5, 6, 7], dtype=torch.int32),
        torch.tensor([8, 9, 10, 11, 12], dtype=torch.int32)
    ]
    
    kv_index = builder.get_kv_index(seq_lens, block_tables)
    
    assert kv_index is not None
    assert kv_index.device.type == 'npu'
    assert kv_index.dtype == torch.long
    
    total_seq_len = sum(seq_lens)
    assert kv_index.shape[0] == total_seq_len
    
    block_size = 128
    max_block_num = max([max(table) if len(table) > 0 else 0 for table in block_tables])
    max_expected_index = (max_block_num + 1) * block_size - 1
    assert torch.all(kv_index >= 0)
    assert torch.all(kv_index <= max_expected_index)

def test_determine_has_context():
    seq_lens_group = [[10, 20, 30], [5, 15, 25], [8, 18, 28]]
    seq_qlen_group = [[10, 18, 30], [5, 10, 25], [8, 18, 15]]
    
    result = mla_mod.determine_has_context(seq_lens_group, seq_qlen_group)
    expected = [True, False, True] 
    
    assert result == expected


def test_column_cumsum_numpy():
    seq_lens_3d = [
        [[1, 2, 3], [4, 5, 6]],
        [[10, 20], [30, 40]]   
    ]
    
    result = mla_mod.column_cumsum_numpy(seq_lens_3d)
    
    expected = [
        [[1, 5], [2, 7], [3, 9]], 
        [[10, 40], [20, 60]]      
    ]
    
    assert result == expected

def test_calculate_seq_lens_for_dcp():
    test_cases = [
        # (world_size, rank, input_positions, expected_output)
        (1, 0, torch.tensor([0, 127, 128, 255, 256]), 
         torch.tensor([1, 128, 129, 256, 257])), 
        
        (2, 0, torch.tensor([0, 127, 128, 255, 256]), 
         torch.tensor([1, 128, 128, 128, 129])),
        
        (2, 1, torch.tensor([0, 127, 128, 255, 256]), 
         torch.tensor([0, 0, 1, 128, 128])),  #
        
        (4, 0, torch.tensor([0, 127, 128, 255, 511, 512]), 
         torch.tensor([1, 128, 128, 128, 128, 129])), 
        
        (4, 2, torch.tensor([0, 127, 128, 255, 511, 512]), 
         torch.tensor([0, 0, 0, 0, 128, 128])), 
    ]
    
    for world_size, rank, input_positions, expected in test_cases:
        with patch.object(mla_mod, 'get_tensor_model_parallel_world_size') as mock_world_size, \
             patch.object(mla_mod, 'get_tensor_model_parallel_rank') as mock_rank:
            
            mock_world_size.return_value = world_size
            mock_rank.return_value = rank
            
            result = mla_mod.calculate_seq_lens_for_dcp(input_positions)
            
            assert torch.equal(result, expected), \
                f"world_size={world_size}, rank={rank} 测试失败"
            
            assert result.shape == input_positions.shape