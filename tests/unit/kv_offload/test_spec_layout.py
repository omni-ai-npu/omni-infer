# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for replicated-layout inference and capacity sizing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omni_npu.v1.kv_offload.cpu.spec import (
    NPUCPUOffloadingSpec,
    _iter_group_specs,
    _spec_is_replicated,
    infer_replicated_layout,
)


class MLAAttentionSpec:
    pass


class FullAttentionSpec:
    pass


class UniformTypeKVCacheSpecs:
    def __init__(self, specs):
        self.kv_cache_specs = specs


def _pc(**overrides):
    values = dict(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        world_size=8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_iter_group_specs_uniform_and_plain():
    plain = SimpleNamespace(kv_cache_spec=MLAAttentionSpec())
    uniform = SimpleNamespace(
        kv_cache_spec=UniformTypeKVCacheSpecs(
            {"a": MLAAttentionSpec(), "b": MLAAttentionSpec()}
        )
    )
    specs = list(_iter_group_specs(SimpleNamespace(kv_cache_groups=[plain, uniform])))
    assert len(specs) == 3


def test_spec_is_replicated_names_and_uniform():
    assert _spec_is_replicated(MLAAttentionSpec()) is True
    assert _spec_is_replicated(FullAttentionSpec()) is False
    uniform = UniformTypeKVCacheSpecs({"a": MLAAttentionSpec(), "b": MLAAttentionSpec()})
    assert _spec_is_replicated(uniform) is True
    mixed = UniformTypeKVCacheSpecs({"a": MLAAttentionSpec(), "b": FullAttentionSpec()})
    assert _spec_is_replicated(mixed) is False


def test_infer_replicated_layout_true_and_false_paths():
    vllm_cfg = SimpleNamespace(parallel_config=_pc())
    kv_cfg = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=MLAAttentionSpec())]
    )
    assert infer_replicated_layout(vllm_cfg, kv_cfg) is True

    assert infer_replicated_layout(
        SimpleNamespace(parallel_config=_pc(tensor_parallel_size=1, world_size=1)),
        kv_cfg,
    ) is False
    assert infer_replicated_layout(
        SimpleNamespace(parallel_config=_pc(pipeline_parallel_size=2)),
        kv_cfg,
    ) is False
    assert infer_replicated_layout(
        SimpleNamespace(parallel_config=_pc(decode_context_parallel_size=2)),
        kv_cfg,
    ) is False
    assert infer_replicated_layout(
        SimpleNamespace(parallel_config=_pc(prefill_context_parallel_size=2)),
        kv_cfg,
    ) is False
    assert infer_replicated_layout(
        SimpleNamespace(parallel_config=_pc(world_size=16)),
        kv_cfg,
    ) is False
    assert infer_replicated_layout(
        vllm_cfg, SimpleNamespace(kv_cache_groups=[])
    ) is False


def test_apply_single_copy_capacity_packed_and_unpacked():
    spec = NPUCPUOffloadingSpec.__new__(NPUCPUOffloadingSpec)
    spec.extra_config = {"cpu_bytes_to_use": 8192}
    spec.block_size_factor = 1
    spec.BLOCK_SIZE_ALIGNMENT = 4096

    unpacked = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=[
            SimpleNamespace(block_stride=0, size=4096),
            SimpleNamespace(block_stride=0, size=4096),
        ],
    )
    spec._apply_single_copy_capacity(unpacked)
    assert spec.kv_bytes_per_offloaded_block == 4096
    assert spec.cpu_page_size_per_worker == 4096
    assert spec.num_blocks == 2

    packed_bad = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=[
            SimpleNamespace(block_stride=8, size=4096),
            SimpleNamespace(block_stride=0, size=4096),
        ],
    )
    with pytest.raises(ValueError, match="block_stride"):
        spec._apply_single_copy_capacity(packed_bad)

    packed = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=[
            SimpleNamespace(block_stride=8, size=8192),
            SimpleNamespace(block_stride=8, size=1),
        ],
    )
    spec.extra_config = {"cpu_bytes_to_use": 16384}
    spec._apply_single_copy_capacity(packed)
    assert spec.kv_bytes_per_offloaded_block == 4096

    spec.extra_config = {}
    before = spec.num_blocks
    spec._apply_single_copy_capacity(SimpleNamespace(num_blocks=0, kv_cache_tensors=[]))
    assert spec.num_blocks == before


def test_get_worker_raises_when_create_returns_none():
    spec = NPUCPUOffloadingSpec.__new__(NPUCPUOffloadingSpec)
    spec._worker = None

    def _create(_caches):
        return None

    spec.create_worker = _create
    with pytest.raises(RuntimeError, match="Failed to create"):
        spec.get_worker(object())


def test_get_worker_reuses_existing():
    spec = NPUCPUOffloadingSpec.__new__(NPUCPUOffloadingSpec)
    sentinel = object()
    spec._worker = sentinel
    assert spec.get_worker(object()) is sentinel


def test_create_mmap_region_rank_selection():
    spec = NPUCPUOffloadingSpec.__new__(NPUCPUOffloadingSpec)
    spec.replicated_layout = True
    spec.num_blocks = 2
    spec.kv_bytes_per_offloaded_block = 4096
    spec.cpu_page_size_per_worker = 4096
    spec.vllm_config = SimpleNamespace(instance_id="i1")
    with patch(
        "omni_npu.v1.kv_offload.cpu.spec.NPUSharedOffloadRegion"
    ) as region_cls, patch(
        "omni_npu.v1.kv_offload.cpu.spec.get_tensor_model_parallel_rank",
        return_value=3,
    ):
        region_cls.return_value = MagicMock()
        spec._create_mmap_region()
        assert region_cls.call_args.kwargs["rank"] == 0

    spec.replicated_layout = False
    with patch(
        "omni_npu.v1.kv_offload.cpu.spec.NPUSharedOffloadRegion"
    ) as region_cls, patch(
        "omni_npu.v1.kv_offload.cpu.spec.get_tensor_model_parallel_rank",
        return_value=3,
    ):
        region_cls.return_value = MagicMock()
        spec._create_mmap_region()
        assert region_cls.call_args.kwargs["rank"] == 3


def test_create_worker_wires_rotation_and_rank():
    spec = NPUCPUOffloadingSpec.__new__(NPUCPUOffloadingSpec)
    spec.replicated_layout = True
    spec.block_size_factor = 2
    spec.num_blocks = 4
    spec.vllm_config = SimpleNamespace(
        instance_id="i1",
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )
    mmap_region = MagicMock()
    worker = MagicMock()
    with patch.object(spec, "_create_mmap_region", return_value=mmap_region), patch(
        "omni_npu.v1.kv_offload.cpu.spec.NPUCPUOffloadingWorker",
        return_value=worker,
    ) as worker_cls, patch(
        "omni_npu.v1.kv_offload.cpu.spec.get_tensor_model_parallel_rank",
        return_value=1,
    ):
        out = spec.create_worker(MagicMock())
        assert out is worker
        assert worker_cls.call_args.kwargs["rotate_store_writers"] is True
        assert worker_cls.call_args.kwargs["tp_rank"] == 1
        assert worker_cls.call_args.kwargs["mmap_region"] is mmap_region


def test_init_sets_replicated_layout_from_extra_and_inferred():
    kv_cfg = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=MLAAttentionSpec())],
        num_blocks=1,
        kv_cache_tensors=[],
    )
    vllm_cfg = SimpleNamespace(
        parallel_config=_pc(),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"canonical_layout": True, "replicated_layout": False}
        ),
        instance_id="x",
    )

    def fake_super_init(self, vllm_config, kv_cache_config):
        self.vllm_config = vllm_config
        self.extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.num_blocks = 1
        self.kv_bytes_per_offloaded_block = 4096
        self.cpu_page_size_per_worker = 4096
        self.block_size_factor = 1
        self._worker = None

    with patch(
        "omni_npu.v1.kv_offload.cpu.spec.CPUOffloadingSpec.__init__",
        fake_super_init,
    ), patch.object(
        NPUCPUOffloadingSpec, "_apply_single_copy_capacity"
    ) as apply_cap:
        spec = NPUCPUOffloadingSpec(vllm_cfg, kv_cfg)
        assert spec.replicated_layout is False
        apply_cap.assert_not_called()

    vllm_cfg2 = SimpleNamespace(
        parallel_config=_pc(),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        instance_id="x",
    )
    with patch(
        "omni_npu.v1.kv_offload.cpu.spec.CPUOffloadingSpec.__init__",
        fake_super_init,
    ), patch.object(
        NPUCPUOffloadingSpec, "_apply_single_copy_capacity"
    ) as apply_cap:
        spec = NPUCPUOffloadingSpec(vllm_cfg2, kv_cfg)
        assert spec.replicated_layout is True
        apply_cap.assert_called_once()
