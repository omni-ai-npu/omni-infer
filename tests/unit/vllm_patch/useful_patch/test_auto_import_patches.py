# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from pathlib import Path

from omni import vllm_patches


def _capture_loaded(monkeypatch):
    loaded = []

    def fake_import(root, base_pkg):
        loaded.append((Path(root).name, base_pkg))

    monkeypatch.setattr(vllm_patches, "import_patches_from_dir", fake_import)
    monkeypatch.delenv("OMNI_NPU_PATCHES_DIR", raising=False)
    monkeypatch.delenv("OMNI_VLLM_PATCHES_DIR", raising=False)
    return loaded


def test_auto_import_skips_models_when_env_unset(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.delenv("OMNI_VLLM_PATCHES_DIR", raising=False)

    vllm_patches.auto_import_patches()

    assert [name for name, _ in loaded] == ["common"]
    assert loaded[0][1] == "omni_npu.vllm_patches.usefull_patch.common"


def test_auto_import_high_throughout_also_loads_pangu_v2_base(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv("OMNI_VLLM_PATCHES_DIR", "high_throughout")

    vllm_patches.auto_import_patches()

    assert [name for name, _ in loaded] == [
        "common",
        "pangu_v2_base",
        "high_throughout",
    ]
    assert loaded[1][1] == (
        "omni_npu.vllm_patches.usefull_patch.models.pangu_v2_base"
    )
    assert loaded[2][1] == (
        "omni_npu.vllm_patches.usefull_patch.models.high_throughout"
    )


def test_auto_import_low_latency_also_loads_pangu_v2_base(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv("OMNI_VLLM_PATCHES_DIR", "low_latency")

    vllm_patches.auto_import_patches()

    assert [name for name, _ in loaded] == [
        "common",
        "pangu_v2_base",
        "low_latency",
    ]
    assert loaded[1][1] == (
        "omni_npu.vllm_patches.usefull_patch.models.pangu_v2_base"
    )
    assert loaded[2][1] == (
        "omni_npu.vllm_patches.usefull_patch.models.low_latency"
    )


def test_auto_import_legacy_pangu_v2_hybrid(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv("OMNI_VLLM_PATCHES_DIR", "pangu_v2_hybrid")

    vllm_patches.auto_import_patches()

    assert [name for name, _ in loaded] == [
        "common",
        "pangu_v2_base",
        "high_throughout",
    ]


def test_auto_import_legacy_pangu_v2_moe(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv("OMNI_VLLM_PATCHES_DIR", "pangu_v2_moe")

    vllm_patches.auto_import_patches()

    assert [name for name, _ in loaded] == [
        "common",
        "pangu_v2_base",
        "low_latency",
    ]


def test_auto_import_legacy_hybrid_and_moe_comma_separated(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv("OMNI_VLLM_PATCHES_DIR", "pangu_v2_hybrid, pangu_v2_moe")

    vllm_patches.auto_import_patches()

    assert [name for name, _ in loaded] == [
        "common",
        "pangu_v2_base",
        "high_throughout",
        "low_latency",
    ]


def test_auto_import_unknown_model_dir_keeps_common(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv("OMNI_VLLM_PATCHES_DIR", "does_not_exist")

    vllm_patches.auto_import_patches()

    assert [name for name, _ in loaded] == ["common"]
