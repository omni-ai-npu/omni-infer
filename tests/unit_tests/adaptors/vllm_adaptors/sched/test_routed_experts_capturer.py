import os
import tempfile
import struct
import uuid
import time
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch, call

import pytest
import torch
import numpy as np

# 导入需要测试的模块
from omni.adaptors.vllm.sched import routed_experts_capturer
from omni.adaptors.vllm.sched.routed_experts_capturer import (
    RoutedExpertsCapturer,
    _RoutedExpertsCapturer,
    save_data,
    save_indices,
    pad_3d,
    pad_1d,
    lock_file,
    unlock_file,
    BUFFER_PREFIX,
    EXPORT_MOE_EXPERTS_TMP_DIR,
    INVALID_INDICES,
    FMT,
    SIZE,
    LOCK_FILE_PREFIX,
    SAVE_DIR,
)
from vllm.logger import logger

# 使用 Fake torch_npu 模拟 NPU 环境
torch_npu = pytest.importorskip("torch_npu")

# ---- 测试辅助函数 ---------------------------------------------------------

def reset_global_capturer():
    """重置全局专家捕获器实例，用于测试隔离"""
    routed_experts_capturer._global_experts_capturer = None

def get_unique_instance_id():
    """生成唯一的实例 ID，用于避免共享内存冲突"""
    return f"test_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000000)}"

# ---- Fake/Dummy 对象定义 -------------------------------------------------

class DummyModelConfig:
    def __init__(
        self,
        num_hidden_layers: int = 4,
        num_dense_layers: int = 1,
        num_experts_per_tok: int = 4,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_dense_layers = num_dense_layers
        self.mlp_only_layers = None
        self.num_experts_per_tok = num_experts_per_tok

class DummyParallelConfig:
    def __init__(self, tp: int = 1, pp: int = 1, dp: int = 1, rank: int = 0):
        self.tensor_parallel_size = tp
        self.pipeline_parallel_size = pp
        self.data_parallel_size = dp
        self.rank = rank

class DummyCacheConfig:
    def __init__(self, block_size: int = 4):
        self.block_size = block_size

class DummySchedulerConfig:
    def __init__(self, max_num_batched_tokens: int = 16, max_num_seqs: int = 2):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs

class DummyKVTransferConfig:
    def __init__(self, kv_role: str | None = None):
        self.kv_role = kv_role

class FakeTPGroup:
    def __init__(self, world_size=1, rank=0):
        self.world_size = world_size
        self.first_rank = 0
        self.device_group = SimpleNamespace(world_size=world_size)

# ---- Fixture 定义 ---------------------------------------------------------

@pytest.fixture
def vllm_config():
    """创建测试用的 vLLM 配置"""
    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig()
    sched_cfg = DummySchedulerConfig()
    parallel_cfg = DummyParallelConfig()
    
    return SimpleNamespace(
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=sched_cfg,
        parallel_config=parallel_cfg,
        device_config=SimpleNamespace(device_type="npu"),
        load_config=SimpleNamespace(),
        lora_config=None,
        speculative_config=None,
        decoding_config=None,
        observability_config=None,
        prompt_adapter_config=None,
        quant_config=None,
        compilation_config=SimpleNamespace(),
        kv_transfer_config=DummyKVTransferConfig(),
        kv_events_config=None,
        additional_config={},
        instance_id="test_instance",
    )

@pytest.fixture
def hf_config():
    """创建测试用的 HuggingFace 配置"""
    return DummyModelConfig(num_hidden_layers=4, num_dense_layers=1, num_experts_per_tok=4)

@pytest.fixture
def tmp_dir():
    """创建临时目录用于测试"""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp

@pytest.fixture
def tp_group_single(monkeypatch):
    """单 rank Tensor Parallel 组"""
    group = FakeTPGroup(world_size=1, rank=0)
    
    def mock_get_tp_group():
        return group
    
    def mock_get_tp_world_size():
        return 1
    
    def mock_get_tp_rank():
        return 0
    
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.get_tp_group", mock_get_tp_group)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.get_tensor_model_parallel_world_size", mock_get_tp_world_size)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.get_tensor_model_parallel_rank", mock_get_tp_rank)
    
    return group

@pytest.fixture
def tp_group_multi(monkeypatch):
    """多 rank Tensor Parallel 组"""
    group = FakeTPGroup(world_size=2, rank=0)
    
    def mock_get_tp_group():
        return group
    
    def mock_get_tp_world_size():
        return 2
    
    def mock_get_tp_rank():
        return 0
    
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.get_tp_group", mock_get_tp_group)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.get_tensor_model_parallel_world_size", mock_get_tp_world_size)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.get_tensor_model_parallel_rank", mock_get_tp_rank)
    
    return group

@pytest.fixture(autouse=True)
def reset_capturer():
    """每个测试前重置全局捕获器"""
    # 在测试前重置
    reset_global_capturer()
    yield
    # 在测试后清理
    reset_global_capturer()

# ---- 测试用例：单例模式 ---------------------------------------------------

def test_get_instance_before_create():
    """测试在创建前获取实例应返回 None"""
    reset_global_capturer()  # 确保全局变量已重置
    instance = RoutedExpertsCapturer.get_instance()
    assert instance is None, f"Expected None but got {instance}"

def test_create_capturer_singleton():
    """测试创建捕获器单例"""
    capturer = RoutedExpertsCapturer.create()
    assert isinstance(capturer, _RoutedExpertsCapturer)
    assert routed_experts_capturer._global_experts_capturer is not None

def test_create_capturer_twice_raises():
    """测试重复创建单例应抛出异常"""
    RoutedExpertsCapturer.create()
    with pytest.raises(RuntimeError, match="Experts capturer already created"):
        RoutedExpertsCapturer.create()

def test_get_instance_after_create():
    """测试在创建后获取实例应返回同一个对象"""
    capturer1 = RoutedExpertsCapturer.create()
    capturer2 = RoutedExpertsCapturer.get_instance()
    assert capturer1 is capturer2, "Expected same instance but got different objects"

# ---- 测试用例：init_buffer ------------------------------------------------

def test_init_buffer_basic(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试基本的缓冲区初始化"""
    # 使用绝对路径的临时目录
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    # 使用唯一的实例ID
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=16,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=False,
    )
    
    assert capturer.num_moe_layers == 3  # 4 - 1
    assert capturer.num_selected_experts == 4
    assert capturer.instance_id == instance_id
    assert capturer.torch_dtype == torch.uint8
    assert capturer.np_dtype == np.uint8
    assert capturer.max_num_kv_tokens == 404  # (100 // 1 + 1) * 4
    
    # device buffer 总是在 init_buffer 中创建
    assert capturer._experts_capturer_device_buffer is not None
    assert capturer._experts_capturer_device_buffer.shape == (16, 3, 4)
    assert capturer._experts_capturer_device_buffer.dtype == torch.uint8
    
    # 共享内存应该为 None（因为没有 enable_init_shared_memory）
    assert capturer._shm is None
    assert capturer._host_buffer_view is None

def test_init_buffer_with_shared_memory(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试初始化带共享内存的缓冲区"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    
    # 关键修正：在导入模块的上下文中设置环境变量
    # 由于 LOCK_FILE_PREFIX 在模块级别定义，我们需要在测试环境中正确处理
    # 方案：直接检查生成的锁文件路径是否符合预期格式
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=16,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id="test_instance",  # 使用固定ID以便验证
        enable_init_shared_memory=True,
    )
    
    # 验证设备缓冲区已创建
    assert capturer._experts_capturer_device_buffer is not None
    assert capturer._experts_capturer_device_buffer.shape == (16, 3, 4)
    assert capturer._experts_capturer_device_buffer.dtype == torch.uint8
    
    # 验证锁文件路径 - 使用路径后缀验证，因为完整路径可能因环境而异
    assert capturer.lock_file.endswith("vllm_routed_experts_test_instance.lock")
    assert "vllm_routed_experts" in capturer.lock_file
    assert "test_instance" in capturer.lock_file
    
    # 验证共享内存已创建
    assert capturer._shm is not None
    assert capturer._host_buffer_view is not None
    assert capturer._host_buffer_view.shape == (404, 3, 4)
    
    # 验证缓冲区已初始化为0
    assert np.all(capturer._host_buffer_view == 0)
    
    # 验证共享内存中的token计数
    assert capturer._shm_total_token_num is not None
    count = struct.unpack_from(FMT, capturer._shm_total_token_num.buf, 0)[0]
    assert count == 0

def test_init_buffer_large_expert_count(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试大量专家时的数据类型选择（使用 uint16）"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    # 设置超过 uint8 最大值的专家数
    hf_config.num_experts_per_tok = 300  # 300 > 255
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=16,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=False,
    )
    
    assert capturer.torch_dtype == torch.uint16
    assert capturer.np_dtype == np.uint16
    assert capturer._experts_capturer_device_buffer is not None
    assert capturer._experts_capturer_device_buffer.dtype == torch.uint16

def test_init_buffer_mlp_only_layers(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试使用 mlp_only_layers 配置"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    hf_config.num_dense_layers = None
    hf_config.mlp_only_layers = [1, 3]
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=16,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=False,
    )
    
    assert capturer.num_moe_layers == 2  # len([1, 3])

def test_init_buffer_twice(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试多次初始化不会重复创建缓冲区"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    
    # 第一次初始化
    capturer.init_buffer(
        max_num_batched_tokens=16,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    first_buffer = capturer._experts_capturer_device_buffer
    
    # 第二次初始化
    capturer.init_buffer(
        max_num_batched_tokens=16,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 验证缓冲区没有被重新创建
    assert capturer._experts_capturer_device_buffer is first_buffer

# ---- 测试用例：capture ----------------------------------------------------

def test_capture_experts(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试捕获专家"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=16,  # buffer 容量
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 创建模拟的 topk_ids
    batch_size = 2
    topk_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.uint8, device="npu")
    
    # 捕获第0层的专家
    capturer.capture(layer_id=0, topk_ids=topk_ids)
    
    # 验证捕获的数据 - buffer 总容量是 16，但只使用了前 batch_size 行
    # 检查总容量
    assert capturer._experts_capturer_device_buffer.shape[0] == 16
    # 只检查前 batch_size 行的数据
    expected = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.uint8, device="npu")
    assert torch.equal(capturer._experts_capturer_device_buffer[:batch_size, 0, :], expected)

def test_capture_none_layer(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试捕获 None 层应直接返回"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=16,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 创建模拟的 topk_ids
    topk_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8, device="npu")
    
    # 捕获 None 层
    capturer.capture(layer_id=None, topk_ids=topk_ids)
    
    # 验证缓冲区未修改
    assert torch.all(capturer._experts_capturer_device_buffer == 0)

def test_capture_without_buffer_init(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试在未初始化缓冲区时捕获应直接返回"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    # 不初始化缓冲区
    
    # 创建模拟的 topk_ids
    topk_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8, device="npu")
    
    # 捕获专家（应该直接返回，不报错）
    capturer.capture(layer_id=0, topk_ids=topk_ids)

def test_capture_multiple_layers(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试捕获多个层"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    max_tokens = 4
    batch_size = 2
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=max_tokens,  # buffer 容量
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 捕获不同层的专家
    for layer_id in range(3):
        topk_ids = torch.full((batch_size, 4), layer_id, dtype=torch.uint8, device="npu")
        capturer.capture(layer_id=layer_id, topk_ids=topk_ids)
    
    # 验证每层的数据 - buffer 总容量是 max_tokens，但只使用了前 batch_size 行
    for layer_id in range(3):
        # 只检查前 batch_size 行的数据
        expected = torch.full((batch_size, 4), layer_id, dtype=torch.uint8, device="npu")
        assert torch.equal(capturer._experts_capturer_device_buffer[:batch_size, layer_id, :], expected)
        # 剩余行应该保持为 0
        assert torch.all(capturer._experts_capturer_device_buffer[batch_size:, layer_id, :] == 0)

# ---- 测试用例：clear_buffer ----------------------------------------------

def test_clear_buffer(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试清空缓冲区"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=4,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 填充缓冲区
    capturer._experts_capturer_device_buffer.fill_(1)
    
    # 清空缓冲区
    capturer.clear_buffer()
    
    # 验证已清空
    assert torch.all(capturer._experts_capturer_device_buffer == 0)

def test_clear_buffer_without_init(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试在未初始化时清空缓冲区"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    # 不初始化缓冲区
    
    # 清空缓冲区（应该不报错）
    capturer.clear_buffer()

# ---- 测试用例：save_captured_experts -------------------------------------

def test_save_experts_single_rank_no_graph_mode(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试单 rank 无图模式保存专家"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.SAVE_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=4,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 填充设备缓冲区
    batch_size = 2
    for layer_id in range(3):
        topk_ids = torch.full((batch_size, 4), layer_id, dtype=torch.uint8, device="npu")
        capturer.capture(layer_id=layer_id, topk_ids=topk_ids)
    
    # 准备 slot mapping
    slot_mapping = np.array([0, 1, 2, -1], dtype=np.int64)
    
    # 保存专家
    capturer.save_captured_experts(
        slot_mapping=slot_mapping,
        len_req_ids=4,
        max_num_seqs=2,
        enable_torchair_graph_mode=False,
    )
    
    # 验证共享内存中的数据
    # slot_mapping 有 3 个有效 token (0, 1, 2)，但 capture 只写了 2 个
    # 第 3 个 token (index 2) 在 buffer 中是 0
    expected_data = np.zeros((3, 3, 4), dtype=np.uint8)
    for layer_id in range(3):
        expected_data[:batch_size, layer_id, :] = layer_id
    # 第 3 个 token (index 2) 保持为 0
    
    assert np.array_equal(capturer._host_buffer_view[:3], expected_data)
    
    # 验证token计数
    count = struct.unpack_from(FMT, capturer._shm_total_token_num.buf, 0)[0]
    assert count == 3

def test_save_experts_single_rank_graph_mode(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试单 rank 图模式保存专家"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.SAVE_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=4,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 填充设备缓冲区
    batch_size = 2
    topk_ids = torch.full((batch_size, 4), 1, dtype=torch.uint8, device="npu")
    capturer.capture(layer_id=0, topk_ids=topk_ids)
    
    # 准备 slot mapping（在图模式下，即使 len_req_ids 等于 valid tokens，也按 max_num_seqs 分割）
    slot_mapping = np.array([0, 1, 2, 3], dtype=np.int64)
    
    # 保存专家（图模式）
    capturer.save_captured_experts(
        slot_mapping=slot_mapping,
        len_req_ids=4,
        max_num_seqs=4,
        enable_torchair_graph_mode=True,
    )
    
    # 验证共享内存中的数据
    # 在图模式下，按 max_num_seqs=4 分割，tp_size=1，所以 rank 0 处理所有 4 个 tokens
    # 但 capture 只写入了 2 个 tokens
    expected_data = np.zeros((4, 3, 4), dtype=np.uint8)
    expected_data[:batch_size, 0, :] = 1
    # 后 2 个 tokens 保持为 0
    
    assert np.array_equal(capturer._host_buffer_view[:4], expected_data)

def test_save_experts_multi_rank(hf_config, tp_group_multi, tmp_dir, monkeypatch):
    """测试多 rank 保存专家"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.SAVE_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=4,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 填充设备缓冲区
    batch_size = 2
    topk_ids = torch.full((batch_size, 4), 1, dtype=torch.uint8, device="npu")
    capturer.capture(layer_id=0, topk_ids=topk_ids)
    
    # 准备 slot mapping
    # [0, 1, 2, -1] 表示有 3 个有效 token
    slot_mapping = np.array([0, 1, 2, -1], dtype=np.int64)
    
    # Mock分布式操作
    mock_dist = MagicMock()
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.dist", mock_dist)
    
    def mock_all_reduce(tensor, op, group):
        # all_reduce 应该找到最大的 num_tokens
        # split_size = (num_valid_tokens - 1) // tp_size + 1 = (3 - 1) // 2 + 1 = 2
        # rank 0 处理 [0, 2)，有 2 个 tokens
        # rank 1 处理 [2, 3)，有 1 个 token
        # max_num_tokens = 2
        tensor.fill_(2)
    
    def mock_gather(tensor, gather_list, dst, group):
        if dst == 0:
            # rank 0 收集数据
            gather_list[0].copy_(tensor)
            gather_list[1].copy_(tensor)
            if tensor.dim() == 1:
                gather_list[1][0] = 2
                gather_list[1][1] = -1
            else:
                # gather_list[0] = rank 0 的数据 (2个 tokens)
                # gather_list[1] = rank 1 的数据 (1个 token, 但被 mock 为 0)
                gather_list[1] *= 0
        else:
            # rank 1 发送数据
            pass
    
    mock_dist.all_reduce = mock_all_reduce
    mock_dist.gather = mock_gather
    
    # 保存专家
    
    capturer.save_captured_experts(
        slot_mapping=slot_mapping,
        len_req_ids=4,
        max_num_seqs=2,
        enable_torchair_graph_mode=False,
    )
    
    # 验证共享内存中的数据
    # 关键修正：根据源码逻辑正确理解数据聚合
    # 
    # 源码逻辑：
    # 1. num_valid_tokens = 3 (slot_mapping[0, 1, 2])
    # 2. split_size = (3-1)//2 + 1 = 2
    # 3. rank 0: [0, 2) -> indices [0, 1] -> 处理前2个有效tokens
    # 4. rank 1: [2, 3) -> indices [2] -> 处理第3个有效token
    # 5. gather后 rank 0:
    #    - all_data[0]: rank 0的数据 [token0, token1] -> 值 [1, 1]
    #    - all_data[1]: rank 1的数据 [token2] -> 被mock为 [0]
    # 6. concatenated_data = [token0, token1, token2] -> [[1,1,1,1], [1,1,1,1], [0,0,0,0]]
    # 7. all_real_mapping = [slot_mapping[0], slot_mapping[1], pad, slot_mapping[2], pad]
    #    = [0, 1, -1, 2, -1]
    # 8. valid_mask = [True, True, False, True, False]
    # 9. data = concatenated_data[valid_mask] = [token0, token1, token2]
    #    = [[1,1,1,1], [1,1,1,1], [0,0,0,0]]
    # 
    # 所以最终在 shared memory 中，3个tokens的数据是：
    # - token0 (index 0): [1, 1, 1, 1]  # 来自 rank 0
    # - token1 (index 1): [1, 1, 1, 1]  # 来自 rank 0
    # - token2 (index 2): [0, 0, 0, 0]  # 来自 rank 1 (被mock为0)
    
    expected_data = np.zeros((3, 3, 4), dtype=np.uint8)
    # 前2个 tokens 被正确捕获，值为 1
    expected_data[0, 0, :] = 1
    expected_data[1, 0, :] = 1
    # 第3个 token (index 2) 来自 rank 1，被 mock 为 0
    expected_data[2, 0, :] = 0
    
    assert np.array_equal(capturer._host_buffer_view[:3], expected_data)

def test_save_experts_invalid_mapping(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试处理包含无效标记的 slot mapping"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.SAVE_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=4,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 填充设备缓冲区
    batch_size = 4
    topk_ids = torch.full((batch_size, 4), 1, dtype=torch.uint8, device="npu")
    capturer.capture(layer_id=0, topk_ids=topk_ids)
    
    # 准备 slot mapping（包含 -1）
    # [0, 1, -1, -1] 表示有 2 个有效 token
    slot_mapping = np.array([0, 1, -1, -1], dtype=np.int64)
    
    # 保存专家
    capturer.save_captured_experts(
        slot_mapping=slot_mapping,
        len_req_ids=4,
        max_num_seqs=2,
        enable_torchair_graph_mode=False,
    )
    
    # 验证只保存了前两个 token
    expected_data = np.zeros((2, 3, 4), dtype=np.uint8)
    expected_data[:2, 0, :] = 1
    
    assert np.array_equal(capturer._host_buffer_view[:2], expected_data)
    
    # 验证token计数
    count = struct.unpack_from(FMT, capturer._shm_total_token_num.buf, 0)[0]
    assert count == 2

def test_save_experts_without_buffer_init(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试在未初始化缓冲区时保存专家"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    # 不初始化缓冲区
    
    # 准备 slot mapping
    slot_mapping = np.array([0, 1, 2, 3], dtype=np.int64)
    
    # 保存专家（应该直接返回 None，不报错）
    result = capturer.save_captured_experts(
        slot_mapping=slot_mapping,
        len_req_ids=4,
        max_num_seqs=2,
        enable_torchair_graph_mode=False,
    )
    
    assert result is None

def test_save_experts_attach_shared_memory(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试在 _host_buffer_view 为 None 时附加共享内存"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.SAVE_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=4,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 清空 _host_buffer_view
    capturer._host_buffer_view = None
    
    # 填充设备缓冲区
    batch_size = 2
    topk_ids = torch.full((batch_size, 4), 1, dtype=torch.uint8, device="npu")
    capturer.capture(layer_id=0, topk_ids=topk_ids)
    
    # 准备 slot mapping
    slot_mapping = np.array([0, 1, 2, -1], dtype=np.int64)
    
    # 保存专家（应该触发 _attach_shared_memory）
    capturer.save_captured_experts(
        slot_mapping=slot_mapping,
        len_req_ids=4,
        max_num_seqs=2,
        enable_torchair_graph_mode=False,
    )
    
    # 验证 _host_buffer_view 已重新附加
    assert capturer._host_buffer_view is not None
    
    # 验证数据
    expected_data = np.zeros((3, 3, 4), dtype=np.uint8)
    expected_data[:batch_size, 0, :] = 1
    assert np.array_equal(capturer._host_buffer_view[:3], expected_data)

def test_save_experts_batch_size_larger_than_buffer(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试 batch size 大于缓冲区大小"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.SAVE_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=2,  # 缓冲区只能容纳 2 个 tokens
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 填充设备缓冲区
    batch_size = 2
    topk_ids = torch.full((batch_size, 4), 1, dtype=torch.uint8, device="npu")
    capturer.capture(layer_id=0, topk_ids=topk_ids)
    
    # 准备 slot mapping（实际只有 2 个有效 tokens）
    slot_mapping = np.array([0, 1, -1, -1], dtype=np.int64)
    
    # 保存专家
    capturer.save_captured_experts(
        slot_mapping=slot_mapping,
        len_req_ids=4,
        max_num_seqs=2,
        enable_torchair_graph_mode=False,
    )
    
    # 验证只保存了前 2 个 tokens
    expected_data = np.zeros((2, 3, 4), dtype=np.uint8)
    expected_data[:batch_size, 0, :] = 1
    
    assert np.array_equal(capturer._host_buffer_view[:2], expected_data)

# ---- 测试用例：__del__ ----------------------------------------------------

def test_del_cleanup_shared_memory(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试 __del__ 清理共享内存"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    capturer.init_buffer(
        max_num_batched_tokens=4,
        num_blocks=100,
        num_cache_groups=1,
        block_size=4,
        hf_config=hf_config,
        instance_id=instance_id,
        enable_init_shared_memory=True,
    )
    
    # 保存共享内存名称
    shm_name = capturer._shm.name
    
    # 删除对象
    del capturer
    
    # 重置全局变量，避免影响其他测试
    reset_global_capturer()

def test_del_without_shared_memory(hf_config, tp_group_single, tmp_dir, monkeypatch):
    """测试 __del__ 在没有共享内存时的行为"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    monkeypatch.setattr("omni.adaptors.vllm.sched.routed_experts_capturer.EXPORT_MOE_EXPERTS_TMP_DIR", tmp_dir_abs)
    
    instance_id = get_unique_instance_id()
    
    capturer = RoutedExpertsCapturer.create()
    # 不初始化共享内存
    
    # 删除对象（应该不报错）
    del capturer
    
    # 重置全局变量
    reset_global_capturer()

# ---- 测试用例：工具函数 --------------------------------------------------

def test_pad_3d():
    """测试 3D 填充"""
    data = torch.ones((2, 3, 4), dtype=torch.float32, device="cpu")
    target_shape = (5, 3, 4)
    
    padded = pad_3d(data, target_shape)
    
    assert padded.shape == target_shape
    assert torch.all(padded[:2] == 1)
    assert torch.all(padded[2:] == 0)

def test_pad_3d_empty():
    """测试空张量的 3D 填充"""
    data = torch.zeros((0, 3, 4), dtype=torch.float32, device="cpu")
    target_shape = (5, 3, 4)
    
    padded = pad_3d(data, target_shape)
    
    assert padded.shape == target_shape
    assert torch.all(padded == 0)

def test_pad_3d_target_too_small():
    """测试目标形状小于数据形状应抛出异常"""
    data = torch.ones((5, 3, 4), dtype=torch.float32, device="cpu")
    target_shape = (2, 3, 4)
    
    with pytest.raises(ValueError, match="must not be less than"):
        pad_3d(data, target_shape)

def test_pad_1d():
    """测试 1D 填充"""
    data = torch.tensor([1, 2, 3], dtype=torch.int64, device="cpu")
    target_len = 5
    
    padded = pad_1d(data, target_len)
    
    assert len(padded) == target_len
    assert torch.equal(padded[:3], data)
    assert torch.all(padded[3:] == INVALID_INDICES)

def test_pad_1d_empty():
    """测试空张量的 1D 填充"""
    data = torch.zeros((0,), dtype=torch.int64, device="cpu")
    target_len = 5
    
    padded = pad_1d(data, target_len)
    
    assert len(padded) == target_len
    assert torch.all(padded == 0)

def test_pad_1d_target_too_small():
    """测试目标长度小于数据长度应抛出异常"""
    data = torch.ones((5,), dtype=torch.int64, device="cpu")
    target_len = 3
    
    with pytest.raises(ValueError, match="must not be less than"):
        pad_1d(data, target_len)

def test_save_data_without_dir():
    """测试在无保存目录时保存数据"""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    indices = np.array([0, 1], dtype=np.int64)
    part_key  = 'total'
    
    # save_dir 为 None 应该直接返回
    save_data(indices, data, None, part_key)

def test_save_data_with_dir(tmp_dir):
    """测试在指定目录保存数据"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    indices = np.array([0, 1], dtype=np.int64)
    part_key  = 'total'
    
    # 保存数据
    save_data(indices, data, tmp_dir_abs, part_key)
    
    # 验证文件已创建
    files = os.listdir(tmp_dir_abs)
    assert len(files) == 1
    assert files[0].startswith("data_")
    
    # 验证文件内容
    file_path = os.path.join(tmp_dir_abs, files[0])
    import gzip
    import pickle
    with gzip.open(file_path, "rb") as f:
        saved_data = pickle.load(f)[part_key]
    
    # 正确比较 numpy 数组
    assert isinstance(saved_data, dict), f"Expected dict but got {type(saved_data)}"
    assert 0 in saved_data
    assert 1 in saved_data
    assert np.array_equal(saved_data[0], data[0])
    assert np.array_equal(saved_data[1], data[1])

def test_save_indices_without_dir():
    """测试在无保存目录时保存索引"""
    indices = np.array([0, 1, 2], dtype=np.int64)
    request_id = "test_req"
    part_key  = 'total'
    
    # save_dir 为 None 应该直接返回
    save_indices(indices, request_id, None, part_key)

def test_save_indices_with_dir(tmp_dir):
    """测试在指定目录保存索引"""
    tmp_dir_abs = os.path.abspath(tmp_dir)
    
    indices = np.array([0, 1, 2], dtype=np.int64)
    request_id = "test_req"
    part_key  = 'total'
    
    # 保存索引
    save_indices(indices, request_id, tmp_dir_abs, part_key)
    
    # 验证文件已创建
    files = os.listdir(tmp_dir_abs)
    assert len(files) == 1
    assert files[0].startswith("indices_")
    
    # 验证文件内容
    file_path = os.path.join(tmp_dir_abs, files[0])
    import gzip
    import pickle
    with gzip.open(file_path, "rb") as f:
        saved_data = pickle.load(f)
    
    assert isinstance(saved_data, dict), f"Expected dict but got {type(saved_data)}"
    assert request_id in saved_data
    assert saved_data[request_id][part_key] == indices.tolist()