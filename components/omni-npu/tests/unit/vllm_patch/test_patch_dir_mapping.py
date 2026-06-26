# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from pathlib import Path

from omni_npu import vllm_patches


def test_get_patch_dir_names_for_openpangu_v2():
    assert vllm_patches._get_patch_dir_names("openpangu_v2") == [
        "pangu_v2_base",
        "pangu_sink_swa_mla",
    ]


def test_get_patch_dir_names_for_manual_legacy_alias():
    assert vllm_patches._get_patch_dir_names("pangu_sink_swa_mla") == [
        "pangu_v2_base",
        "pangu_sink_swa_mla",
    ]


def test_get_patch_dir_names_for_minimax_m2():
    assert vllm_patches._get_patch_dir_names("minimax_m2") == ["minimax"]


def test_find_patch_dir_exact_supports_multiple_manual_dirs():
    models_root = Path(vllm_patches.patches.__file__).parent / "models"

    patch_dirs = vllm_patches._find_patch_dir_exact("pangu_sink_swa_mla", models_root)

    assert [path.name for path in patch_dirs] == [
        "pangu_v2_base",
        "pangu_sink_swa_mla",
    ]


def test_find_patch_dir_fuzzy_supports_multiple_auto_dirs():
    models_root = Path(vllm_patches.patches.__file__).parent / "models"

    patch_dirs = vllm_patches._find_patch_dir_fuzzy("openpangu_ultra_omni", models_root)

    assert [path.name for path in patch_dirs] == [
        "pangu_v2_base",
        "pangu_sink_swa_mla",
        "openpangu_ultra_omni",
    ]
