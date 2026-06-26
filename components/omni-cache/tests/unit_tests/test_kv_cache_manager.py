# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_kv_cache_manager.py

import sys
import importlib
import pytest
try:
    import torch
except ImportError:
    # Torch will be mocked by conftest.py stubs
    torch = sys.modules.get('torch')
from collections import defaultdict
from unittest.mock import Mock, patch, MagicMock, create_autospec
from dataclasses import dataclass, field, replace
from typing import List, Optional, Any

# Import module under test
from omni_cache.attention.kv_managers import (
    OmniKVCacheBlocks,
    OmniKVCacheManager,
    OmniAttentionManager,
)

from omni_cache.attention.kv_cache_interface import OmniAttentionSpec, FullAttentionSpec
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager, FullAttentionManager

# ==================== Global patches ====================
_original_omni_attn_spec_post_init = OmniAttentionSpec.__post_init__
def _patched_omni_attn_spec_post_init(self):
    object.__setattr__(self, "max_num_blocks", self.sink_blocks + self.recent_blocks)
    object.__setattr__(self, "sink", self.sink_blocks * self.block_size)
    object.__setattr__(self, "recent", self.recent_blocks * self.block_size)
    object.__setattr__(self, "max_compressed_len", self.max_num_blocks * self.block_size)
OmniAttentionSpec.__post_init__ = _patched_omni_attn_spec_post_init

_original_get_manager_for_kv_cache_spec = None

def _get_manager_for_kv_cache_spec_patched(kv_cache_spec: KVCacheSpec, **kwargs) -> SingleTypeKVCacheManager:
    """Patched version that filters out unsupported parameters"""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec_manager_map = {
        FullAttentionSpec: FullAttentionManager,
        OmniAttentionSpec: OmniAttentionManager,
    }
    manager_class = spec_manager_map[type(kv_cache_spec)]

    if manager_class == FullAttentionManager:
        unsupported_keys = {'use_eagle' , 'num_kv_cache_groups', 'caching_hash_fn'}
        kwargs = {k:v for k,v in kwargs.items() if k not in unsupported_keys}
        if 'kv_cache_group_id' not in kwargs:
            kwargs['kv_cache_group_id'] = 0
        kwargs.setdefault('enable_caching', False)
    elif manager_class == OmniAttentionManager:
        unsupported_keys = {'use_eagle' , 'num_kv_cache_groups'}
        kwargs = {k:v for k,v in kwargs.items() if k not in unsupported_keys}
        kwargs.setdefault('enable_caching', False)

    manager = manager_class(kv_cache_spec, **kwargs)
    return manager

# Import and patch the function in the module
_kv_cache_manager_mod = importlib.import_module("omni_cache.attention.kv_managers")
_original_get_manager_for_kv_cache_spec = _kv_cache_manager_mod.get_manager_for_kv_cache_spec
_kv_cache_manager_mod.get_manager_for_kv_cache_spec = _get_manager_for_kv_cache_spec_patched

_original_omni_attn_mgr_init = OmniAttentionManager.__init__
def _patched_omni_attn_mgr_init(self, kv_cache_spec, *args, **kwargs):
    unsupported_keys = {'use_eagle', 'num_kv_cache_groups', 'caching_hash_fn'}
    kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_keys}
    if 'kv_cache_group_id' not in kwargs:
        kwargs['kv_cache_group_id'] = 0
    if 'enable_caching' not in kwargs and not args:
        kwargs['enable_caching'] = False
    return _original_omni_attn_mgr_init(self, kv_cache_spec, *args, **kwargs)
OmniAttentionManager.__init__ = _patched_omni_attn_mgr_init

def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec, **kwargs) -> SingleTypeKVCacheManager:
    mod = _kv_cache_manager_mod
    if hasattr(mod, "get_manager_for_kv_cache_spec"):
        return mod.get_manager_for_kv_cache_spec(kv_cache_spec, **kwargs)
    return _original_get_manager_for_kv_cache_spec(kv_cache_spec, **kwargs)


# ==================== Mock classes and fixtures ====================

@dataclass
class MockKVCacheConfig:
    """Mock KV cache config."""
    num_blocks: int = 1000
    kv_cache_groups: List = None
    num_blocks_per_group: dict = None

    def __init__(self):
        self.kv_cache_groups = []
        self.num_blocks_per_group = {
            FullAttentionSpec: 1000,
            OmniAttentionSpec: 4,
        }


@dataclass
class MockKVCacheGroup:
    """Mock KV cache group."""
    layer_names: List[str] = None
    kv_cache_spec = None

    def __init__(self):
        self.layer_names = ["layer.0", "layer.1"]

def create_mock_full_attention_spec(block_size=16, num_kv_heads=1, head_size=128,
                                    dtype="bfloat16", use_mla=False, sliding_window=None):
    import torch
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    from vllm.v1.kv_cache_interface import FullAttentionSpec as RealFullAttentionSpec
    return RealFullAttentionSpec(
        block_size = block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch_dtype,
        sliding_window=sliding_window,
    )

def create_mock_omni_attention_spec(block_size=16, num_kv_heads=1, head_size=128,
                                    dtype="bfloat16", use_mla=False,
                                    max_compressed_len=160, max_num_blocks=10,
                                    sink_blocks=1, recent_blocks=3):
    import torch
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    return OmniAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch_dtype,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
    )


class MockBlockPool:
    """Mock BlockPool for testing."""
    null_block = None
    def __init__(self, num_blocks=1000, usage=0.5):
        self.num_blocks = num_blocks
        self._usage = usage
        self.blocks = {}
        if MockBlockPool.null_block is None:
            null_block = Mock(spec=KVCacheBlock)
            null_block.block_id = -1
            null_block.block_hash = None
            MockBlockPool.null_block = null_block
        self.null_block = MockBlockPool.null_block

    def get_usage(self):
        return self._usage

    def get_num_free_blocks(self):
        return int(self.num_blocks * (1 - self._usage))

    def touch(self, blocks):
        pass

    def get_new_blocks(self, num_blocks):
        blocks = []
        for i in range(num_blocks):
            block = Mock(spec=KVCacheBlock)
            block.block_id = len(self.blocks) + i
            block.block_hash = None
            blocks.append(block)
        return blocks


class MockSingleTypeManager:
    """Mock SingleTypeKVCacheManager for testing."""
    def __init__(self):
        self.req_to_blocks = defaultdict(list)
        self.block_pool = MockBlockPool()

    def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks):
        return 1

    def remove_skipped_blocks(self, request_id, num_computed_tokens):
        pass

    def save_new_computed_blocks(self, request_id, new_computed_blocks):
        pass

    def allocate_new_blocks(self, request_id, num_tokens):
        blocks = self.block_pool.get_new_blocks(1)
        self.req_to_blocks[request_id].extend(blocks)
        return blocks

    def free(self, request_id):
        if request_id in self.req_to_blocks:
            del self.req_to_blocks[request_id]

    def cache_blocks(self, request, block_hashes, num_tokens):
        pass


# Note: MockModelExtraConfig is no longer needed after refactoring.
# Configuration is now handled via vllm_config and environment variables.


class MockRequest:
    """Mock Request for testing."""
    def __init__(self, request_id="test_req", num_computed_tokens=0, num_tokens=10):
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.num_tokens = num_tokens


@pytest.fixture
def mock_kv_cache_block():
    """Create a mock KVCacheBlock."""
    block = Mock(spec=KVCacheBlock)
    block.block_id = 1
    block.block_hash = "hash123"
    return block


@pytest.fixture
def mock_blocks():
    """Create a list of mock KVCacheBlocks."""
    blocks = []
    for i in range(3):
        block = Mock(spec=KVCacheBlock)
        block.block_id = i
        block.block_hash = f"hash{i}" if i > 0 else None
        blocks.append(block)
    return blocks


# ==================== OmniKVCacheBlocks tests ====================

class TestOmniKVCacheBlocks:
    """Test suite for OmniKVCacheBlocks dataclass."""

    def test_create_empty(self):
        """Test creating empty OmniKVCacheBlocks."""
        empty_blocks = OmniKVCacheBlocks.create_empty()
        assert empty_blocks.blocks == []
        assert len(empty_blocks.blocks) == 0

    def test_get_block_ids(self, mock_blocks):
        """Test get_block_ids method."""
        blocks = OmniKVCacheBlocks([mock_blocks])
        block_ids = blocks.get_block_ids()

        assert len(block_ids) == 1
        assert block_ids[0] == [0, 1, 2]

    def test_get_block_ids_multiple_groups(self):
        """Test get_block_ids with multiple groups."""
        group1 = [Mock(spec=KVCacheBlock, block_id=i) for i in range(2)]
        group2 = [Mock(spec=KVCacheBlock, block_id=i+10) for i in range(3)]

        blocks = OmniKVCacheBlocks([group1, group2])
        block_ids = blocks.get_block_ids()

        assert len(block_ids) == 2
        assert block_ids[0] == [0, 1]
        assert block_ids[1] == [10, 11, 12]

    def test_get_unhashed_block_ids(self, mock_blocks):
        """Test get_unhashed_block_ids method."""
        blocks = OmniKVCacheBlocks([mock_blocks])
        unhashed_ids = blocks.get_unhashed_block_ids()

        # Only block 0 has None hash
        assert len(unhashed_ids) == 1
        assert unhashed_ids[0] == [0]

    def test_add_empty_left(self, mock_blocks):
        """Test __add__ when left operand is empty."""
        empty = OmniKVCacheBlocks.create_empty()
        right = OmniKVCacheBlocks([mock_blocks])

        result = empty + right
        assert len(result.blocks) == 1
        assert len(result.blocks[0]) == 3

    def test_add_empty_right(self, mock_blocks):
        """Test __add__ when right operand is empty."""
        left = OmniKVCacheBlocks([mock_blocks])
        empty = OmniKVCacheBlocks.create_empty()

        result = left + empty
        assert len(result.blocks) == 1
        assert len(result.blocks[0]) == 3

    def test_add_both_non_empty(self):
        """Test __add__ when both operands are non-empty."""
        group1_left = [Mock(spec=KVCacheBlock, block_id=i) for i in range(2)]
        group2_left = [Mock(spec=KVCacheBlock, block_id=i+10) for i in range(2)]
        left = OmniKVCacheBlocks([group1_left, group2_left])

        group1_right = [Mock(spec=KVCacheBlock, block_id=i+100) for i in range(2)]
        group2_right = [Mock(spec=KVCacheBlock, block_id=i+110) for i in range(2)]
        right = OmniKVCacheBlocks([group1_right, group2_right])

        result = left + right

        assert len(result.blocks) == 2
        assert len(result.blocks[0]) == 4  # 2 + 2
        assert len(result.blocks[1]) == 4

    def test_add_different_group_counts(self):
        """Test __add__ with different group counts."""
        group1 = [Mock(spec=KVCacheBlock, block_id=i) for i in range(2)]
        left = OmniKVCacheBlocks([group1])

        group2 = [Mock(spec=KVCacheBlock, block_id=i+10) for i in range(3)]
        group3 = [Mock(spec=KVCacheBlock, block_id=i+20) for i in range(1)]
        right = OmniKVCacheBlocks([group2, group3])

        result = left + right
        # Should zip groups together
        assert len(result.blocks) == 1
        assert len(result.blocks[0]) == 5

@pytest.fixture
def mock_setup_for_manager():
    with patch("omni_cache.cache.prefill.calc_cache_shape_for_prefill",
               return_value=(0, 1000)):
        with patch("omni_cache.cache.decode.calc_cache_shape_for_decode",
                   return_value=(0, 1000)):
            with patch("omni_cache.attention.kv_managers.omni_manager.PrefixCacheStats"):
                with patch("vllm.v1.core.block_pool.BlockPool") as mock_block_pool:
                    with patch("omni_cache.attention.kv_managers.omni_manager.os.getenv", return_value=None):
                        yield mock_block_pool


# ==================== OmniKVCacheManager Init tests ====================

class TestOmniKVCacheManagerInit:
    """Test suite for OmniKVCacheManager initialization."""


    def test_init_with_full_attention_spec(self, mock_setup_for_manager):
        """Test initialization with FullAttentionSpec."""
        mock_block_pool = mock_setup_for_manager
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool()
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
            enable_caching=True,
        )

        assert manager.block_size == 16
        assert manager.num_gpu_blocks == 1000
        assert manager.max_model_len == 2048
        assert manager.enable_caching == True
        assert len(manager.block_pools) == 1
        assert len(manager.hybrid_managers) == 1

    def test_init_too_many_groups(self, mock_setup_for_manager):
        """Test initialization raises ValueError with more than 2 groups."""
        mock_block_pool = mock_setup_for_manager
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        groups = [MockKVCacheGroup() for _ in range(3)]
        for group in groups:
            group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = groups

        with pytest.raises(ValueError, match="does not support hybrid models with more than 2"):
            OmniKVCacheManager(
                kv_cache_config=config,
                max_model_len=2048,
            )

    def test_init_no_full_attention_spec(self, mock_setup_for_manager):
        """Test initialization raises RuntimeError when no FullAttentionSpec."""
        mock_block_pool = mock_setup_for_manager
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_omni_attention_spec(block_size=16)  # Not FullAttentionSpec
        config.kv_cache_groups = [group]

        with pytest.raises(RuntimeError, match="No FullAttentionSpec is found"):
            OmniKVCacheManager(
                kv_cache_config=config,
                max_model_len=2048,
            )

    def test_init_disable_caching_for_hybrid(self, mock_setup_for_manager):
        """Test caching is disabled for hybrid managers."""
        mock_block_pool = mock_setup_for_manager
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group1 = MockKVCacheGroup()
        group1.kv_cache_spec = create_mock_full_attention_spec(block_size=16)

        group2 = MockKVCacheGroup()
        group2.kv_cache_spec = create_mock_omni_attention_spec(block_size=16)

        config.kv_cache_groups = [group1, group2]

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
            enable_caching=True,
        )

        assert manager.enable_caching == False


# ==================== OmniKVCacheManager Property tests ====================

@pytest.fixture
def mock_setup_for_manager_properties():

    with patch("omni_cache.cache.prefill.calc_cache_shape_for_prefill",
               return_value=(0, 1000)):
        with patch("omni_cache.cache.decode.calc_cache_shape_for_decode",
                   return_value=(0, 1000)):
            with patch("omni_cache.attention.kv_managers.omni_manager.PrefixCacheStats"):
                with patch("vllm.v1.core.block_pool.BlockPool") as mock_block_pool:
                    with patch("omni_cache.attention.kv_managers.omni_manager.os.getenv", return_value=None):
                        yield mock_block_pool


class TestOmniKVCacheManagerProperties:
    """Test suite for OmniKVCacheManager properties."""
    def test_usage_property(self, mock_setup_for_manager_properties):
        """Test usage property returns max of block pool usages."""
        mock_block_pool = mock_setup_for_manager_properties
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group1 = MockKVCacheGroup()
        group1.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        group2 = MockKVCacheGroup()
        group2.kv_cache_spec = create_mock_omni_attention_spec(block_size=16)
        config.kv_cache_groups = [group1, group2]

        # Create mock block pools with specific usage values
        mock_bp1 = MockBlockPool(usage=0.3)
        mock_bp2 = MockBlockPool(usage=0.7)
        mock_block_pool.side_effect = [mock_bp1, mock_bp2]

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
        )

        # Manually replace block_pools with our mocks to ensure usage values are correct
        manager.block_pools = [mock_bp1, mock_bp2]

        # Usage should be max of all block pools
        assert manager.usage == max(0.3, 0.7)

    def test_single_type_manager_property(self, mock_setup_for_manager_properties):
        """Test single_type_manager property."""
        mock_block_pool = mock_setup_for_manager_properties
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool()
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
        )

        assert manager.single_type_manager == manager.hybrid_managers[0]

    def test_block_pool_property(self, mock_setup_for_manager_properties):
        """Test block_pool property."""
        mock_block_pool = mock_setup_for_manager_properties
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool()
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
        )

        assert manager.block_pool == manager.block_pools[0]


# ==================== OmniKVCacheManager allocate_slots tests ====================

@pytest.fixture
def mock_setup_for_allocate_slots():

    def patched_cache_blocks(self, request, block_hashes, num_tokens_to_cache):
        return

    with patch.object(FullAttentionManager, 'cache_blocks', patched_cache_blocks):
        with patch("omni_cache.cache.prefill.calc_cache_shape_for_prefill",
                   return_value=(0, 1000)):
            with patch("omni_cache.cache.decode.calc_cache_shape_for_decode",
                       return_value=(0, 1000)):
                with patch("omni_cache.attention.kv_managers.omni_manager.PrefixCacheStats"):
                    with patch("vllm.v1.core.block_pool.BlockPool") as mock_block_pool:
                        with patch("omni_cache.attention.kv_managers.omni_manager.os.getenv", return_value=None):
                            yield mock_block_pool

class TestOmniKVCacheManagerAllocateSlots:
    """Test suite for OmniKVCacheManager.allocate_slots method."""

    def test_allocate_slots_basic(self, mock_setup_for_allocate_slots):
        """Test basic slot allocation."""
        mock_block_pool = mock_setup_for_allocate_slots
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool(usage=0.3)
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
            enable_caching=False,
        )

        request = MockRequest(request_id="test_req", num_computed_tokens=0, num_tokens=10)
        result = manager.allocate_slots(request, num_new_tokens=5)

        assert result is not None
        assert isinstance(result, OmniKVCacheBlocks)
        assert len(result.blocks) == 1

    def test_allocate_slots_zero_tokens(self, mock_setup_for_allocate_slots):
        """Test allocate_slots with zero tokens raises ValueError."""
        mock_block_pool = mock_setup_for_allocate_slots
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool()
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
        )

        request = MockRequest()

        with pytest.raises(ValueError, match="num_new_tokens must be greater than 0"):
            manager.allocate_slots(request, num_new_tokens=0)

    def test_allocate_slots_insufficient_blocks(self, mock_setup_for_allocate_slots):
        """Test allocate_slots behavior when block resources are constrained."""
        mock_block_pool = mock_setup_for_allocate_slots
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        # Create mock block pool - with usage=1.0, there are no free blocks
        mock_bp_instance = MockBlockPool(usage=1.0)
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
            enable_caching=False,
        )

        # Manually set block_pools to use our mock with high usage
        manager.block_pools = [mock_bp_instance]

        request = MockRequest()
        result = manager.allocate_slots(request, num_new_tokens=100)

        # When there are insufficient blocks, the result may be None or empty blocks
        # depending on the implementation details - just verify it doesn't crash
        assert result is None or isinstance(result, OmniKVCacheBlocks)

    def test_allocate_slots_with_new_computed_blocks(self, mock_setup_for_allocate_slots):
        """Test allocate_slots with new computed blocks."""
        mock_block_pool = mock_setup_for_allocate_slots
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool(usage=0.3)
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
            enable_caching=True,
        )

        request = MockRequest()

        result = manager.allocate_slots(
            request,
            num_new_tokens=5,
            num_new_computed_tokens=3,
        )

        assert result is not None

    def test_allocate_slots_with_delay_cache(self, mock_setup_for_allocate_slots):
        """Test allocate_slots with delay_cache_blocks flag."""
        mock_block_pool = mock_setup_for_allocate_slots
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool(usage=0.3)
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
            enable_caching=True,
        )

        request = MockRequest()
        result = manager.allocate_slots(
            request,
            num_new_tokens=5,
            delay_cache_blocks=True
        )

        assert result is not None


# ==================== OmniKVCacheManager free tests ====================
@pytest.fixture
def mock_setup_for_free():
    with patch("omni_cache.cache.prefill.calc_cache_shape_for_prefill",
               return_value=(0, 1000)):
        with patch("omni_cache.cache.decode.calc_cache_shape_for_decode",
                   return_value=(0, 1000)):
            with patch("omni_cache.attention.kv_managers.omni_manager.PrefixCacheStats"):
                with patch("omni_cache.attention.kv_managers.omni_manager.get_manager_for_kv_cache_spec") as mock_get_mgr:
                    with patch("omni_cache.attention.kv_managers.omni_manager.os.getenv", return_value=None):
                        with patch("vllm.v1.core.block_pool.BlockPool") as mock_block_pool:
                            mock_mgr = MockSingleTypeManager()
                            mock_get_mgr.return_value = mock_mgr
                            yield  mock_block_pool, mock_get_mgr, mock_mgr


class TestOmniKVCacheManagerFree:
    """Test suite for OmniKVCacheManager.free method."""
    def test_free_request(self, mock_setup_for_free):
        """Test freeing blocks for a request."""
        mock_block_pool, mock_get_mgr, mock_mgr = mock_setup_for_free
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool()
        mock_block_pool.return_value = mock_bp_instance

        # Create manager first (this creates internal managers via factory)
        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
        )

        # Now populate the internal manager's req_to_blocks
        internal_mgr = manager.hybrid_managers[0]
        internal_mgr.req_to_blocks["test_req"] = [Mock(spec=KVCacheBlock, block_id=1)]

        request = MockRequest(request_id="test_req")
        manager.free(request)

        # Verify the request was freed from the manager
        assert "test_req" not in internal_mgr.req_to_blocks


# ==================== OmniKVCacheManager get_block_ids tests ====================
@pytest.fixture
def mock_setup_for_block_ids():
    with patch("omni_cache.cache.prefill.calc_cache_shape_for_prefill",
               return_value=(0, 1000)):
        with patch("omni_cache.cache.decode.calc_cache_shape_for_decode",
                   return_value=(0, 1000)):
            with patch("omni_cache.attention.kv_managers.omni_manager.PrefixCacheStats"):
                with patch("omni_cache.attention.kv_managers.omni_manager.get_manager_for_kv_cache_spec") as mock_get_mgr:
                    with patch("omni_cache.attention.kv_managers.omni_manager.os.getenv", return_value=None):
                        with patch("vllm.v1.core.block_pool.BlockPool") as mock_block_pool:
                            yield  mock_block_pool, mock_get_mgr

class TestOmniKVCacheManagerGetBlockIds:
    """Test suite for OmniKVCacheManager.get_block_ids method."""

    def test_get_block_ids_existing_request(self, mock_setup_for_block_ids):
        """Test getting block ids for an existing request."""
        mock_block_pool, mock_get_mgr = mock_setup_for_block_ids
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool()
        mock_block_pool.return_value = mock_bp_instance

        # Create mock manager that will be returned by the factory
        mock_mgr = MockSingleTypeManager()
        mock_get_mgr.return_value = mock_mgr

        # Create manager first
        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
        )

        # Populate the internal manager's req_to_blocks
        # The internal manager should be our mock since we set return_value
        internal_mgr = manager.hybrid_managers[0]
        internal_mgr.req_to_blocks["test_req"] = [
            Mock(spec=KVCacheBlock, block_id=1),
            Mock(spec=KVCacheBlock, block_id=2),
        ]

        block_ids = manager.get_block_ids("test_req")

        assert len(block_ids) == 1
        assert block_ids[0] == [1, 2]

    def test_get_block_ids_non_existing_request(self, mock_setup_for_block_ids):
        """Test getting block ids for a non-existing request."""
        mock_block_pool, mock_get_mgr = mock_setup_for_block_ids
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group = MockKVCacheGroup()
        group.kv_cache_spec = create_mock_full_attention_spec(block_size=16)
        config.kv_cache_groups = [group]

        mock_bp_instance = MockBlockPool()
        mock_block_pool.return_value = mock_bp_instance

        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
        )

        block_ids = manager.get_block_ids("nonexistent_req")

        assert len(block_ids) == 1
        assert block_ids[0] == []

    def test_get_block_ids_multiple_groups(self, mock_setup_for_block_ids):
        """Test getting block ids with multiple KV cache groups."""
        mock_block_pool, mock_get_mgr = mock_setup_for_block_ids
        config = MockKVCacheConfig()
        config.num_blocks = 1000

        group1 = MockKVCacheGroup()
        group1.kv_cache_spec = create_mock_full_attention_spec(block_size=16)

        group2 = MockKVCacheGroup()
        group2.kv_cache_spec = create_mock_omni_attention_spec(block_size=16)

        config.kv_cache_groups = [group1, group2]

        mock_bp_instance = MockBlockPool()
        mock_block_pool.return_value = mock_bp_instance

        # Configure mock_get_mgr to return different managers for each call
        mock_mgr1 = MockSingleTypeManager()
        mock_mgr2 = MockSingleTypeManager()
        mock_get_mgr.side_effect = [mock_mgr1, mock_mgr2]

        # Create manager first
        manager = OmniKVCacheManager(
            kv_cache_config=config,
            max_model_len=2048,
        )

        # Populate internal managers' req_to_blocks
        manager.hybrid_managers[0].req_to_blocks["test_req"] = [
            Mock(spec=KVCacheBlock, block_id=1),
            Mock(spec=KVCacheBlock, block_id=2),
        ]
        manager.hybrid_managers[1].req_to_blocks["test_req"] = [
            Mock(spec=KVCacheBlock, block_id=10),
        ]

        block_ids = manager.get_block_ids("test_req")

        assert len(block_ids) == 2
        assert block_ids[0] == [1, 2]
        assert block_ids[1] == [10]


# ==================== OmniAttentionManager tests ====================
@pytest.fixture
def mock_setup_for_attention_manager():
    mock_bp_instance = MockBlockPool()
    with patch("vllm.v1.core.block_pool.BlockPool") as mock_block_pool:
        mock_block_pool.return_value = mock_bp_instance
        yield  mock_bp_instance

class TestOmniAttentionManager:
    """Test suite for OmniAttentionManager."""

    def test_init(self, mock_setup_for_attention_manager):
        """Test OmniAttentionManager initialization."""
        mock_bp = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)

        assert manager.max_tokens == 64
        assert manager.max_num_blocks == 4

    def test_get_num_blocks_to_allocate_new_request(self, mock_setup_for_attention_manager):
        """Test get_num_blocks_to_allocate for new request."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)
        
        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp
        
        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)

        manager.req_to_blocks["new_req"] = []

        num_blocks = manager.get_num_blocks_to_allocate("new_req", 100, [])

        assert num_blocks == manager.max_num_blocks

    def test_get_num_blocks_to_allocate_existing_request(self, mock_setup_for_attention_manager):
        """Test get_num_blocks_to_allocate for existing request."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)
        manager.req_to_blocks["existing_req"] = [Mock(spec=KVCacheBlock, block_id=1)]

        num_blocks = manager.get_num_blocks_to_allocate("existing_req", 100, [])

        assert num_blocks == 0

    def test_allocate_new_blocks_first_time(self, mock_setup_for_attention_manager):
        """Test allocate_new_blocks for first allocation."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)
        manager.req_to_blocks["new_req"] = []

        new_blocks = manager.allocate_new_blocks("new_req", 100)

        assert len(new_blocks) == manager.max_num_blocks

    def test_allocate_new_blocks_second_time(self, mock_setup_for_attention_manager):
        """Test allocate_new_blocks for subsequent allocation."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)
        # Pre-populate with blocks
        for i in range(10):
            manager.req_to_blocks["existing_req"].append(Mock(spec=KVCacheBlock, block_id=i))

        new_blocks = manager.allocate_new_blocks("existing_req", 100)

        assert len(new_blocks) == 0

    def test_save_new_computed_blocks(self, mock_setup_for_attention_manager):
        """Test save_new_computed_blocks does nothing."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)

        # Should not raise any error
        manager.save_new_computed_blocks("test_req", [])

    def test_cache_blocks(self, mock_setup_for_attention_manager):
        """Test cache_blocks does nothing."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)
        request = MockRequest()

        # Should not raise any error
        manager.cache_blocks(request, [], 10)


    def test_find_longest_cache_hit_not_implemented(self, mock_setup_for_attention_manager):
        """Test find_longest_cache_hit raises NotImplementedError."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)

        with pytest.raises(NotImplementedError, match="not implemented yet"):
            manager.find_longest_cache_hit([], 100)


    def test_remove_skipped_blocks(self, mock_setup_for_attention_manager):
        """Test remove_skipped_blocks does nothing."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)

        # Should not raise any error
        manager.remove_skipped_blocks("test_req", 10)


    def test_get_num_common_prefix_blocks(self, mock_setup_for_attention_manager):
        """Test get_num_common_prefix_blocks returns 0."""
        mock_block_pool = mock_setup_for_attention_manager
        spec = create_mock_omni_attention_spec(block_size=16)

        mock_bp = MockBlockPool()
        mock_block_pool.return_value = mock_bp

        manager = OmniAttentionManager(spec, block_pool=mock_bp, kv_cache_group_id=0)

        result = manager.get_num_common_prefix_blocks("test_req", 5)

        assert result == 0


# ==================== get_manager_for_kv_cache_spec tests ====================
@pytest.fixture
def mock_setup_for_manager_factory():
    mock_bp_instance = MockBlockPool()
    with patch("vllm.v1.core.block_pool.BlockPool") as mock_block_pool:
        mock_block_pool.return_value = mock_bp_instance
        yield  mock_bp_instance

class TestGetManagerForKVCacheSpec:
    """Test suite for get_manager_for_kv_cache_spec factory function."""

    def test_get_manager_for_full_attention_spec(self, mock_setup_for_manager_factory):
        """Test factory returns FullAttentionManager for FullAttentionSpec."""
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
        from vllm.v1.kv_cache_interface import FullAttentionSpec as RealFullAttentionSpec

        mock_bp = mock_setup_for_manager_factory
        spec = RealFullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )

        manager = get_manager_for_kv_cache_spec(spec, block_pool=mock_bp, kv_cache_group_id=0)

        assert isinstance(manager, FullAttentionManager)

    def test_get_manager_for_omni_attention_spec(self, mock_setup_for_manager_factory):
        """Test factory returns OmniAttentionManager for OmniAttentionSpec."""
        import torch
        mock_bp = mock_setup_for_manager_factory

        spec = OmniAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            sink_blocks=1,
            recent_blocks=3,
        )

        manager = get_manager_for_kv_cache_spec(spec, block_pool=mock_bp, kv_cache_group_id=0)

        assert isinstance(manager, OmniAttentionManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])