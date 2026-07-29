# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
import importlib.util
import json
import logging
import sys
import os
from pathlib import Path

from .patch_manager import PatchManager

logger = logging.getLogger(__name__)


def get_model_type_from_args():
    try:
        idx = sys.argv.index('--model') + 1 if '--model' in sys.argv else 2
        model_type = sys.argv[idx]
        if not model_type:
            raise ValueError("Command-line argument sys.argv[2] cannot be empty")
        return model_type
    except IndexError:
        raise ValueError("Model type not provided. Please ensure sys.argv[2] is passed")


def import_patches_from_dir(root: Path, base_pkg: str):
    """
    Imports all.py files in the specified directory and sorts them by file name.
    """
    py_files = sorted(root.rglob("*.py"), key=lambda p: p.name)

    for py_file in py_files:
        if py_file.name == "__init__.py":
            continue

        rel_path = py_file.relative_to(root).with_suffix("")
        module_name = ".".join((base_pkg, *rel_path.parts))

        if module_name in sys.modules:
            continue

        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def get_model_type_from_config(model_path: Path) -> str:
    """
    read model_type from config.json
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_type = config.get("model_type")
    if not model_type:
        raise ValueError("model_type field is missing in config.json")

    return model_type


def _find_patch_dir_exact(model_type: str, models_root: Path) -> list[Path]:
    """
    Exact matching: Strictly match the lowercase model type with lowercase subdirectory name.
    Applicable: User manually sets OMNI_NPU_PATCHES_DIR environment variable.

    Supports comma-separated multiple directories (e.g., "pangu_v2_base,pangu_sink_swa_mla").

    Returns:
        List of Path objects to patch directories (may be empty)
    """
    patch_dirs = []

    patch_dir_names = _get_patch_dir_names(model_type)
    expanded_model_type = ",".join(patch_dir_names) if patch_dir_names else model_type

    for dir_name in expanded_model_type.split(","):
        dir_name = dir_name.strip()
        if not dir_name:
            continue

        model_type_lower = dir_name.lower()
        for subdir in models_root.iterdir():
            if not subdir.is_dir():
                continue

            subdir_name_lower = subdir.name.lower()
            if subdir_name_lower == model_type_lower:
                logger.info(f"Exact match succeeded:'{dir_name}'->'{subdir.name}'")
                patch_dirs.append(subdir)
                break

    if not patch_dirs:
        logger.warning(f"Exact match failed: No directory for '{model_type}' in {models_root}")

    return patch_dirs


def _split_patch_dir_names(patch_dir_names: str) -> list[str]:
    return [dir_name.strip() for dir_name in patch_dir_names.split(",") if dir_name.strip()]


def _get_patch_dir_names(model_type: str) -> list[str]:
    patch_dir_map = {
        "deepseek_v3": "deepseek",
        "deepseek_v32": "deepseek",
        "qwen3": "qwen",
        "qwen3_5": "qwen3_5",
        "pangupromoe": "openpangu_v1_vl,openpangu_v1",
        "openpangu_v2": "pangu_v2_base,pangu_sink_swa_mla",
        "openpangu_ultra_omni": "pangu_v2_base,pangu_sink_swa_mla,openpangu_v1_vl",
        "pangu_sink_swa_mla": "pangu_v2_base,pangu_sink_swa_mla",
        "pangu_v2_hybrid": "pangu_v2_base,pangu_sink_swa_mla,pangu_v2_hybrid",
        "pangu_v2_hybrid_vl": "pangu_v2_base,pangu_sink_swa_mla,pangu_v2_hybrid,openpangu_v1_vl",
        "kimi_k25": "kimi",
        "minimax_m2": "minimax",
        "openpangu_vl": "openpangu_v1_vl,openpangu_v1",
    }
    return _split_patch_dir_names(patch_dir_map.get(model_type.lower(), ""))


def _find_patch_dir_fuzzy(model_type: str, models_root: Path) -> list[Path]:
    """
    Map to the specific directory under patches/models based on model_type.
    Supports:
    - Mapping table (can map to multiple directories, comma-separated)
    - Prefix matching
    - containment match

    Returns:
        List of Path objects to patch directories (may be empty)
    """
    patch_dirs = []

    patch_dir_names = _get_patch_dir_names(model_type)
    if patch_dir_names:
        for dir_name in patch_dir_names:
            candidate = models_root / dir_name
            if candidate.exists():
                patch_dirs.append(candidate)

    if not patch_dirs:
        model_type_lower = model_type.lower()
        for subdir in models_root.iterdir():
            if not subdir.is_dir():
                continue

            subdir_name_lower = subdir.name.lower()
            logger.info(f"model_type is: {model_type}, subdir name is: {subdir_name_lower}")
            if (model_type_lower.startswith(subdir_name_lower)
                    or subdir_name_lower in model_type_lower):
                patch_dirs.append(subdir)

    if not patch_dirs:
        logger.warning(
            f"No patch directory found for model_type '{model_type}' in {models_root}."
        )

    return patch_dirs


def auto_import_patches():
    """
    Load patches only from the curated usefull_patch directory.
    The files within the directory are sorted by filename.
    """
    vllm_patches_root = Path(__file__).parent
    usefull_patch_dir = vllm_patches_root / "usefull_patch"
    base_pkg = "omni_npu.vllm_patches.usefull_patch"

    if not usefull_patch_dir.exists():
        logger.warning(
            "usefull_patch directory not found: %s", usefull_patch_dir
        )
        return

    import_patches_from_dir(usefull_patch_dir, base_pkg)
    logger.info("loaded patches from %s", usefull_patch_dir)


manager = PatchManager()


def apply_patches():
    # auto import and register patches from usefull_patch only
    auto_import_patches()

    manager.apply_patches()
