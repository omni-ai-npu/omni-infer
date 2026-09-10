# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for core.checkpoint — atomic save + resume detection."""
import torch

from jointfix.core.checkpoint import (
    atomic_save,
    load_checkpoint,
    save_checkpoint,
    verify_shard,
)


def test_atomic_save_and_verify(tmp_path):
    shard = tmp_path / "model-00001.safetensors"
    atomic_save({"model.layers.0.x.weight": torch.randn(4, 8)}, shard)
    assert shard.exists()
    assert verify_shard(shard) is True
    # no leftover tmp file
    assert not (tmp_path / "model-00001.safetensors.tmp").exists()


def test_verify_shard_false_on_garbage(tmp_path):
    bad = tmp_path / "broken.safetensors"
    bad.write_bytes(b"not a safetensors file")
    assert verify_shard(bad) is False


def test_load_checkpoint_none(tmp_path):
    # no checkpoint file -> start from scratch
    assert load_checkpoint(tmp_path, weight_map={}) == (0, set())


def test_save_then_load_resume(tmp_path):
    shard = "model-00001.safetensors"
    atomic_save({"model.layers.0.x.weight": torch.randn(2, 2)}, tmp_path / shard)
    weight_map = {"model.layers.0.x.weight": shard}
    save_checkpoint(tmp_path, last_layer=0, written_shards={shard})

    resume_layer, verified = load_checkpoint(tmp_path, weight_map)
    assert resume_layer == 1               # resume from last_completed + 1
    assert verified == {shard}


def test_load_checkpoint_missing_shard_rewinds(tmp_path):
    # checkpoint records a shard that was never written -> corrupted path.
    # weight_map maps it to layer 0, so resume must rewind to layer 0.
    shard = "model-00001.safetensors"
    weight_map = {"model.layers.0.x.weight": shard}
    save_checkpoint(tmp_path, last_layer=3, written_shards={shard})

    resume_layer, verified = load_checkpoint(tmp_path, weight_map)
    assert resume_layer == 0
    assert shard not in verified
