# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from pathlib import Path

import pytest

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
    assert loaded[0][1] == "omni_npu.vllm_patches.patches.common"


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
        "omni_npu.vllm_patches.patches.models.pangu_v2_base"
    )
    assert loaded[2][1] == (
        "omni_npu.vllm_patches.patches.models.high_throughout"
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
        "omni_npu.vllm_patches.patches.models.pangu_v2_base"
    )
    assert loaded[2][1] == (
        "omni_npu.vllm_patches.patches.models.low_latency"
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


def test_legacy_vl_directory_is_explicit_and_deduplicated(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv(
        "OMNI_NPU_PATCHES_DIR",
        "pangu_v2_hybrid,pangu_v2_moe,openpangu_v1_vl,openpangu_v1_vl",
    )
    vllm_patches.auto_import_patches()
    assert [name for name, _ in loaded] == [
        "common", "pangu_v2_base", "high_throughout", "low_latency",
        "openpangu_v1_vl",
    ]
    assert loaded[-1][1] == "omni_npu.vllm_patches.patches.models.openpangu_v1_vl"


def test_removed_multimodal_selector_is_not_an_implicit_alias(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv("OMNI_VLLM_PATCHES_DIR", "multimodal")
    vllm_patches.auto_import_patches()
    assert [name for name, _ in loaded] == ["common"]


def test_nonempty_new_selector_takes_precedence_over_old_selector(monkeypatch):
    loaded = _capture_loaded(monkeypatch)
    monkeypatch.setenv("OMNI_VLLM_PATCHES_DIR", "low_latency")
    monkeypatch.setenv("OMNI_NPU_PATCHES_DIR", "openpangu_v1_vl")
    vllm_patches.auto_import_patches()
    assert [name for name, _ in loaded] == ["common", "pangu_v2_base", "low_latency"]


def test_legacy_vl_group_contains_six_multimodal_patches():
    models_root = Path(vllm_patches.__file__).parent / "patches/models"
    selected = vllm_patches._find_patch_dir_exact("openpangu_v1_vl", models_root)
    assert selected == [models_root / "openpangu_v1_vl"]
    assert not (models_root / "multimodal").exists()
    assert not (selected[0] / "common").exists()
    # Match the unmodified importer's filename order and ensure no nested
    # duplicate implementations remain after combining the two groups.
    paths = sorted(selected[0].rglob("*.py"), key=lambda path: path.name)
    assert [path.relative_to(selected[0]).as_posix() for path in paths] == [
        "patch_m_rotary_embedding.py",
        "patch_media_utils.py",
        "patch_mm_feature_transfer_args.py",
        "patch_multimodal_embeddings.py",
        "patch_multimodal_prompt_updates.py",
        "patch_video.py",
    ]
