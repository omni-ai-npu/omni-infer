# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import pytest
import torch
import torch_npu
from typing import Callable, Any, List, Tuple, Dict

from unittest.mock import MagicMock, Mock, patch

from .distributed_test_common import parse_ascend_devices, distributed_worker_pool
TEST_SEED = 0
FIRST_DIE_NO, VISIBLE_DIE_LIST = parse_ascend_devices()

# --- Logic Functions ---

def _logic_column_parallel_flash_comm_linear(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import ColumnParallelFlashCommLinear

    device = torch.device(f"npu:{device}")
    input_size = 8
    output_size = 10
    batch_size = 2

    layer = ColumnParallelFlashCommLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        skip_bias_add=True,
        params_dtype=dtype,
        quant_config=None,
        output_sizes=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
        disable_tp=False
    ).to(device)
    full_weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    full_bias = torch.randn(output_size, dtype=dtype, device=device)
    shard_size = output_size // world_size
    start = local_rank * shard_size
    end = start + shard_size
    layer.weight_loader(layer.weight, full_weight.clone())
    layer.quant_method.process_weights_after_loading(layer)
    with torch.no_grad():
        layer.bias.data.copy_(full_bias[start:end])

    input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    out, out_bias = layer(input_tensor)

    expected = torch.matmul(input_tensor, full_weight[start:end].T)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out[:,start:end], expected, atol=atol, rtol=rtol)
    assert torch.allclose(out_bias, full_bias[start:end], atol=atol, rtol=rtol)

def _logic_column_parallel_flash_comm_linear_quant(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import ColumnParallelFlashCommLinear
    from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import NPUCompressedTensorsConfig
    from omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear import W8A8Int8FCLinearMethod

    quant_config = NPUCompressedTensorsConfig(None, None, None, None, None)

    device = torch.device(f"npu:{device}")
    input_size = 8
    output_size = 10
    batch_size = 2

    with patch.dict(os.environ, {"VLLM_PLUGINS": "omni_custom_models"}):
        with patch(
            "omni_npu.layers.quantization.compressed_tensors.compressed_tensors.NPUCompressedTensorsConfig.get_fc_method",
            return_value=W8A8Int8FCLinearMethod(quant_config)
        ):
            layer = ColumnParallelFlashCommLinear(
                input_size=input_size,
                output_size=output_size,
                bias=True,
                skip_bias_add=False,
                params_dtype=dtype,
                quant_config=quant_config,
                output_sizes=None,
                prefix="model.layers.4.self_attn.o_proj",
                return_bias=True,
                disable_tp=False
            ).to(device)
    full_weight = torch.randn(output_size, input_size, dtype=torch.int8, device=device)
    full_bias = torch.randn(output_size, dtype=dtype, device=device)
    shard_size = output_size // world_size
    batch_shard_size = batch_size // world_size
    start = local_rank * shard_size
    end = start + shard_size
    layer.weight_loader(layer.weight, full_weight.clone())
    layer.quant_method.process_weights_after_loading(layer)
    with torch.no_grad():
        layer.bias.data.copy_(full_bias[start:end])

    full_input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    input_tensor = full_input_tensor[batch_shard_size*local_rank:batch_shard_size*(local_rank+1),:]
    
    out, out_bias = layer(input_tensor)
    
    x_list = []
    x_scale_list = []
    for rank in range(world_size):
        x, x_scale = torch_npu.npu_dynamic_quant(full_input_tensor[batch_shard_size*rank:batch_shard_size*(rank+1),:], smooth_scales=None)
        x_list.append(x)
        x_scale_list.append(x_scale)
    x = torch.cat(x_list, dim=0)
    x_scale = torch.cat(x_scale_list, dim=0)
    expected = torch_npu.npu_quant_matmul(
        x1=x,
        x2=full_weight[start:end].T,
        scale=layer.weight_scale,
        pertoken_scale=x_scale,
        bias=full_bias[start:end],
        output_dtype=dtype
    )
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out, expected, atol=atol, rtol=rtol, equal_nan=True)

def _logic_merged_column_parallel_flash_comm_linear(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import MergedColumnParallelFlashCommLinear
    from omni_npu.v1.distributed.communication_op_ext import layer_parallel_all_gather

    device = torch.device(f"npu:{device}")
    input_size = 6
    output_sizes = [6, 8]
    batch_size = 2

    layer = MergedColumnParallelFlashCommLinear(
        input_size=input_size,
        output_sizes=output_sizes,
        bias=True,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
        disable_tp=False
    ).to(device)

    weights = [
        torch.randn(output_sizes[0], input_size, dtype=dtype, device=device),
        torch.randn(output_sizes[1], input_size, dtype=dtype, device=device),
    ]
    biases = [
        torch.randn(output_sizes[0], dtype=dtype, device=device),
        torch.randn(output_sizes[1], dtype=dtype, device=device),
    ]
    layer.weight_loader(layer.weight, weights[0].clone(), loaded_shard_id=0)
    layer.weight_loader(layer.weight, weights[1].clone(), loaded_shard_id=1)
    layer.quant_method.process_weights_after_loading(layer)

    shard_sizes = [size // world_size for size in output_sizes]
    shard0 = weights[0][local_rank * shard_sizes[0]:(local_rank + 1) * shard_sizes[0]]
    shard1 = weights[1][local_rank * shard_sizes[1]:(local_rank + 1) * shard_sizes[1]]
    weight_shard = torch.cat([shard0, shard1], dim=0)
    bias_shard = torch.cat([
        biases[0][local_rank * shard_sizes[0]:(local_rank + 1) * shard_sizes[0]],
        biases[1][local_rank * shard_sizes[1]:(local_rank + 1) * shard_sizes[1]],
    ], dim=0)
    with torch.no_grad():
        layer.bias.data.copy_(bias_shard)

    input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    out, out_bias = layer(input_tensor)

    expected_local = layer.quant_method.apply(layer, input_tensor, layer.bias, layer.x_transform, layer.x_dim)
    expected = layer_parallel_all_gather(expected_local, layer.layer_name_inside_block, "y", dim=1)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out, expected, atol=atol, rtol=rtol)
    assert out_bias is None

def _logic_row_parallel_flash_comm_linear(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import RowParallelFlashCommLinear

    device = torch.device(f"npu:{device}")
    input_size = 8
    output_size = 6
    batch_size = 3

    full_weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    part_size = input_size // world_size
    start = local_rank * part_size
    end = start + part_size
    shard_weight = full_weight[:, start:end].contiguous()

    full_input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    shard_input = full_input[..., start:end].contiguous()

    partial_outputs = []
    for rank in range(world_size):
        start_r = rank * part_size
        end_r = start_r + part_size
        shard_in_r = full_input[..., start_r:end_r]
        shard_w_r = full_weight[:, start_r:end_r]
        partial_outputs.append(torch.matmul(shard_in_r, shard_w_r.T))
    final_output = torch.stack(partial_outputs, dim=0).sum(dim=0)

    layer_ar = RowParallelFlashCommLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
        disable_tp=False
    ).to(device)
    layer_ar.weight_loader(layer_ar.weight, full_weight.clone())
    layer_ar.quant_method.process_weights_after_loading(layer_ar)

    out_ar, out_bias_ar = layer_ar(shard_input)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out_ar, final_output, atol=atol, rtol=rtol)
    assert out_bias_ar is None

def _logic_row_parallel_flash_comm_linear_rs(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import RowParallelFlashCommLinear

    device = torch.device(f"npu:{device}")
    input_size = 8
    output_size = 6
    batch_size = 4

    full_weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    part_size = input_size // world_size
    start = local_rank * part_size
    end = start + part_size
    shard_weight = full_weight[:, start:end].contiguous()

    full_input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    shard_input = full_input[..., start:end].contiguous()

    partial_outputs = []
    for rank in range(world_size):
        start_r = rank * part_size
        end_r = start_r + part_size
        shard_in_r = full_input[..., start_r:end_r]
        shard_w_r = full_weight[:, start_r:end_r]
        partial_outputs.append(torch.matmul(shard_in_r, shard_w_r.T))
    full_output = torch.stack(partial_outputs, dim=0).sum(dim=0)
    final_output = full_output[local_rank*2:(local_rank+1)*2,:]

    layer_rs = RowParallelFlashCommLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
        disable_tp=False
    ).to(device)
    layer_rs.weight_loader(layer_rs.weight, full_weight.clone())
    layer_rs.quant_method.process_weights_after_loading(layer_rs)

    out_ar, out_bias_ar = layer_rs(shard_input)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out_ar, final_output, atol=atol, rtol=rtol)
    assert out_bias_ar is None


def _logic_qkv_parallel_flash_comm_linear(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import QKVParallelFlashCommLinear
    from omni_npu.v1.distributed.communication_op_ext import layer_parallel_all_gather

    device = torch.device(f"npu:{device}")
    hidden_size = 8
    head_size = 2
    total_num_heads = 4
    total_num_kv_heads = 2
    batch_size = 2

    layer = QKVParallelFlashCommLinear(
        hidden_size=hidden_size,
        head_size=head_size,
        total_num_heads=total_num_heads,
        total_num_kv_heads=total_num_kv_heads,
        bias=True,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
        disable_tp=False
    ).to(device)

    output_size = layer.output_size
    full_weight = torch.randn(output_size, hidden_size, dtype=dtype, device=device)
    full_bias = torch.randn(output_size, dtype=dtype, device=device)
    layer.weight_loader(layer.weight, full_weight.clone())
    layer.quant_method.process_weights_after_loading(layer)

    num_heads = layer.num_heads
    num_kv_heads = layer.num_kv_heads
    q_rows = num_heads * head_size
    kv_rows = num_kv_heads * head_size
    q_slice = slice(local_rank * q_rows, (local_rank + 1) * q_rows)
    k_start = total_num_heads * head_size + local_rank * kv_rows
    k_slice = slice(k_start, k_start + kv_rows)
    v_start = (total_num_heads + total_num_kv_heads) * head_size + local_rank * kv_rows
    v_slice = slice(v_start, v_start + kv_rows)

    weight_shard = torch.cat([
        full_weight[q_slice],
        full_weight[k_slice],
        full_weight[v_slice],
    ], dim=0)
    bias_shard = torch.cat([
        full_bias[q_slice],
        full_bias[k_slice],
        full_bias[v_slice],
    ], dim=0)
    with torch.no_grad():
        layer.bias.data.copy_(bias_shard)

    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    out, out_bias = layer(input_tensor)

    expected_local = layer.quant_method.apply(layer, input_tensor, layer.bias, layer.x_transform, layer.x_dim)
    expected = layer_parallel_all_gather(expected_local, layer.layer_name_inside_block, "y", dim=1)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out, expected, atol=atol, rtol=rtol)
    assert out_bias is None

def _logic_qkv_parallel_flash_comm_linear_a2a_4q_4kv(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import QKVParallelFlashCommLinear
    from omni_npu.v1.distributed.communication_op_ext import layer_parallel_all2all_single
    device = torch.device(f"npu:{device}")
    hidden_size = 8
    head_size = 2
    total_num_heads = 4
    total_num_kv_heads = 4
    batch_size = 2

    layer = QKVParallelFlashCommLinear(
        hidden_size=hidden_size,
        head_size=head_size,
        total_num_heads=total_num_heads,
        total_num_kv_heads=total_num_kv_heads,
        bias=True,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
        disable_tp=False
    ).to(device)

    output_size = layer.output_size
    full_weight = torch.empty(output_size, hidden_size, dtype=dtype, device=device)
    full_bias = torch.empty(output_size, dtype=dtype, device=device)
    for i in range(output_size):
        full_weight[i:i+1]=i
        full_bias[i:i+1]=i

    layer.weight_loader(layer.weight, full_weight.clone())
    layer.quant_method.process_weights_after_loading(layer)

    q_slice = slice(0, 4)
    k_slice = slice(8, 12)
    v_slice = slice(16, 20)
    q_slice2 = slice(4, 8)
    k_slice2 = slice(12, 16)
    v_slice2 = slice(20, 24)

    weight_shard = torch.cat([
        full_weight[q_slice],
        full_weight[k_slice],
        full_weight[v_slice],
        full_weight[q_slice2],
        full_weight[k_slice2],
        full_weight[v_slice2],
    ], dim=0)
    bias_shard = torch.cat([
        full_bias[q_slice],
        full_bias[k_slice],
        full_bias[v_slice],
        full_bias[q_slice2],
        full_bias[k_slice2],
        full_bias[v_slice2],
    ], dim=0)
    with torch.no_grad():
        layer.bias.data.copy_(bias_shard)

    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    out, out_bias = layer(input_tensor)

    expected_local = layer.quant_method.apply(layer, input_tensor, layer.bias, layer.x_transform, layer.x_dim)
    expected = layer_parallel_all2all_single(expected_local, layer.layer_name_inside_block, "y", dim=1)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out, expected, atol=atol, rtol=rtol)
    assert out_bias is None

def _logic_qkv_parallel_flash_comm_linear_a2a_8q_4kv(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import QKVParallelFlashCommLinear
    from omni_npu.v1.distributed.communication_op_ext import layer_parallel_all2all_single

    device = torch.device(f"npu:{device}")
    hidden_size = 8
    head_size = 2
    total_num_heads = 8
    total_num_kv_heads = 4
    batch_size = 2

    layer = QKVParallelFlashCommLinear(
        hidden_size=hidden_size,
        head_size=head_size,
        total_num_heads=total_num_heads,
        total_num_kv_heads=total_num_kv_heads,
        bias=True,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
        disable_tp=False
    ).to(device)

    output_size = layer.output_size
    full_weight = torch.empty(output_size, hidden_size, dtype=dtype, device=device)
    full_bias = torch.empty(output_size, dtype=dtype, device=device)
    for i in range(output_size):
        full_weight[i:i+1]=i
        full_bias[i:i+1]=i

    layer.weight_loader(layer.weight, full_weight.clone())
    layer.quant_method.process_weights_after_loading(layer)

    q_slice = slice(0, 8)
    k_slice = slice(16, 20)
    v_slice = slice(24, 28)
    q_slice2 = slice(8, 16)
    k_slice2 = slice(20, 24)
    v_slice2 = slice(28, 32)

    weight_shard = torch.cat([
        full_weight[q_slice],
        full_weight[k_slice],
        full_weight[v_slice],
        full_weight[q_slice2],
        full_weight[k_slice2],
        full_weight[v_slice2],
    ], dim=0)
    bias_shard = torch.cat([
        full_bias[q_slice],
        full_bias[k_slice],
        full_bias[v_slice],
        full_bias[q_slice2],
        full_bias[k_slice2],
        full_bias[v_slice2],
    ], dim=0)
    with torch.no_grad():
        layer.bias.data.copy_(bias_shard)

    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    out, out_bias = layer(input_tensor)

    expected_local = layer.quant_method.apply(layer, input_tensor, layer.bias, layer.x_transform, layer.x_dim)
    expected = layer_parallel_all2all_single(expected_local, layer.layer_name_inside_block, "y", dim=1)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out, expected, atol=atol, rtol=rtol)
    assert out_bias is None

def get_test_configs_yag() -> Tuple[Mock, Dict]:
    layer_parallel_config = {
        "input_split": False,
        "self_attn.o_proj": {
            "tp_size_or_ranks": [[0,1]],
            "x_transform": {"type": "None"},
            "y_transform": {"type": "AllGather", "dim": 1, "tp_size_or_ranks": [[0,1]]}
        },
    }
    return (layer_parallel_config)

def get_test_configs_xag() -> Tuple[Mock, Dict]:
    layer_parallel_config = {
        "input_split": False,
        "self_attn.o_proj": {
            "tp_size_or_ranks": [[0,1]],
            "x_transform": {"type": "AllGather", "dim": 0, "tp_size_or_ranks": [[0,1]]},
            "y_transform": {"type": "None"}
        },
    }
    return (layer_parallel_config)

def get_test_configs_yar() -> Tuple[Mock, Dict]:
    layer_parallel_config = {
        "input_split": False,
        "self_attn.o_proj": {
            "tp_size_or_ranks": [[0,1]],
            "x_transform": {"type": "None"},
            "y_transform": {"type": "AllReduce", "tp_size_or_ranks": [[0,1]]}
        },
    }
    return (layer_parallel_config)

def get_test_configs_yrs() -> Tuple[Mock, Dict]:
    layer_parallel_config = {
        "input_split": False,
        "self_attn.o_proj": {
            "tp_size_or_ranks": [[0,1]],
            "x_transform": {"type": "None"},
            "y_transform": {"type": "ReduceScatter", "tp_size_or_ranks": [[0,1]]}
        },
    }
    return (layer_parallel_config)

def get_test_configs_ya2a() -> Tuple[Mock, Dict]:
    layer_parallel_config = {
        "input_split": False,
        "self_attn.o_proj": {
            "tp_size_or_ranks": 1,
            "x_transform": {"type": "None"},
            "y_transform": {"type": "ALL2ALL", "tp_size_or_ranks": [[0,1]], "dim": 1}
        },
    }
    return (layer_parallel_config)

def get_test_configs_none() -> Tuple[Mock, Dict]:
    layer_parallel_config = {
        "input_split": False,
    }
    return (layer_parallel_config)


def _logic_sharded_linear(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import ShardedLinear
    from omni_npu.v1.distributed.parallel_state_ext import get_layer_parallel_group

    device = torch.device(f"npu:{device}")
    input_size = 8
    output_size = 6
    batch_size = 4

    weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    x = torch.randn(batch_size, input_size, dtype=dtype, device=device)

    golden = torch.matmul(x, weight.T)

    layer = ShardedLinear(
        input_size,
        output_size,
        bias=False,
        shard_group=get_layer_parallel_group("self_attn.o_proj"),
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
    ).to(device)
    layer.weight.weight_loader(layer.weight, weight.clone())

    layer.prefetch()
    y, bias = layer(x)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(golden, y, atol=atol, rtol=rtol)
    assert bias is None

def _logic_sharded_linear_w8a8(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import ShardedLinear
    from omni_npu.v1.distributed.parallel_state_ext import get_layer_parallel_group
    from omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear import W8A8Int8ShardedLinearMethod

    device = torch.device(f"npu:{device}")
    input_size = 8
    output_size = 6
    batch_size = 4

    weight_scale = torch.randn(output_size, dtype=dtype, device=device)
    weight = torch.randint(-128, 127, (output_size, input_size), dtype=torch.int8, device=device)
    x = torch.randn(batch_size, input_size, dtype=torch.bfloat16, device=device)

    x_int8, x_scale = torch_npu.npu_dynamic_quant(x)
    golden = torch_npu.npu_quant_matmul(
        x1=x_int8,
        x2=weight.T,
        scale=weight_scale,
        pertoken_scale=x_scale,
        output_dtype=torch.bfloat16,
    )

    mock_quant_config = MagicMock(
        get_quant_method=lambda *_, **kw: W8A8Int8ShardedLinearMethod()
    )

    layer = ShardedLinear(
        input_size,
        output_size,
        bias=False,
        shard_group=get_layer_parallel_group("self_attn.o_proj"),
        params_dtype=dtype,
        quant_config=mock_quant_config,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
    ).to(device)
    layer.weight.weight_loader(layer.weight, weight.clone())
    layer.weight_scale.weight_loader(layer.weight_scale, weight_scale.clone())

    layer.prefetch()
    y, bias = layer(x)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(golden, y, atol=atol, rtol=rtol)
    assert bias is None

def _logic_sharded_flash_comm_linear_xag(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import ShardedFlashCommLinear
    from omni_npu.v1.distributed.parallel_state_ext import get_layer_parallel_group

    device = torch.device(f"npu:{device}")
    input_size = 8
    output_size = 6
    batch_size = 4

    group = get_layer_parallel_group("self_attn.o_proj")
    weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    local_x = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    x = group.all_gather(local_x, dim=0)
    golden = torch.matmul(x, weight.T)

    layer = ShardedFlashCommLinear(
        input_size,
        output_size,
        bias=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
    ).to(device)
    layer.weight.weight_loader(layer.weight, weight.clone())

    layer.prefetch()
    y, bias = layer(local_x)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(golden, y, atol=atol, rtol=rtol)
    assert bias is None

def _logic_sharded_flash_comm_linear_yag(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.linear import ShardedFlashCommLinear
    from omni_npu.v1.distributed.parallel_state_ext import get_layer_parallel_group

    device = torch.device(f"npu:{device}")
    input_size = 8
    output_size = 6
    batch_size = 4

    group = get_layer_parallel_group("self_attn.o_proj")
    weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    x = torch.randn(batch_size, input_size, dtype=dtype, device=device)

    golden = group.all_gather(torch.matmul(x, weight.T), dim=1)

    layer = ShardedFlashCommLinear(
        input_size,
        output_size,
        bias=False,
        params_dtype=dtype,
        quant_config=None,
        prefix="model.layers.4.self_attn.o_proj",
        return_bias=True,
    ).to(device)
    layer.weight.weight_loader(layer.weight, weight.clone())

    layer.prefetch()
    y, bias = layer(x)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(golden, y, atol=atol, rtol=rtol)
    assert bias is None



# --- Distributed Tests ---
@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_configs_yag())])
def test_column_parallel_flash_comm_linear(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_column_parallel_flash_comm_linear,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.bfloat16, get_test_configs_xag())])
def test_column_parallel_flash_comm_linear_quant(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_column_parallel_flash_comm_linear_quant,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.bfloat16, get_test_configs_yag())])
def test_merged_column_parallel_flash_comm_linear(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_merged_column_parallel_flash_comm_linear,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_configs_yar())])
def test_row_parallel_flash_comm_linear_ar(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_row_parallel_flash_comm_linear,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_configs_yrs())])
def test_row_parallel_flash_comm_linear_rs(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_row_parallel_flash_comm_linear_rs,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.bfloat16, get_test_configs_yag())])
def test_qkv_parallel_flash_comm_linear(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_qkv_parallel_flash_comm_linear,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.bfloat16, get_test_configs_ya2a())])
def test_qkv_parallel_flash_comm_linear_a2a_4q_4kv(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_qkv_parallel_flash_comm_linear_a2a_4q_4kv,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.bfloat16, get_test_configs_ya2a())])
def test_qkv_parallel_flash_comm_linear_a2a_8q_4kv(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_qkv_parallel_flash_comm_linear_a2a_8q_4kv,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_configs_none())])
def test_sharded_linear(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_sharded_linear,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_configs_none())])
def test_sharded_linear_w8a8(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_sharded_linear_w8a8,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_configs_xag())])
def test_sharded_flash_comm_linear_xag(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_sharded_flash_comm_linear_xag,
        dtype,
        config=config
    )

@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_configs_yag())])
def test_sharded_flash_comm_linear_yag(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_sharded_flash_comm_linear_yag,
        dtype,
        config=config
    )
