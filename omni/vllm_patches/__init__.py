# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
import importlib.util
import logging
import sys
import os
from pathlib import Path

from .patch_manager import PatchManager

logger = logging.getLogger(__name__)


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


def _find_patch_dir_exact(model_type: str, models_root: Path) -> list[Path]:
    """
    Exact matching: Strictly match the lowercase model type with lowercase subdirectory name.
    Applicable: User manually sets OMNI_VLLM_PATCHES_DIR environment variable.

    Supports comma-separated multiple directories (e.g. "pangu_v2_base,high_throughout").
    ``high_throughout`` / ``low_latency`` also pull in ``pangu_v2_base``.
    Legacy names ``pangu_v2_hybrid`` and ``pangu_v2_moe`` map to those paths.

    Returns:
        List of Path objects to patch directories (may be empty)
    """
    patch_dirs = []

    expanded_names = _get_patch_dir_names(model_type) or _split_patch_dir_names(
        model_type
    )

    for dir_name in expanded_names:
        model_type_lower = dir_name.lower()
        for subdir in models_root.iterdir():
            if not subdir.is_dir():
                continue

            if subdir.name.lower() == model_type_lower:
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
        "high_throughout": "pangu_v2_base,high_throughout",
        "low_latency": "pangu_v2_base,low_latency",
        # Legacy OMNI_VLLM_PATCHES_DIR values used by older playbooks.
        "pangu_v2_hybrid": "pangu_v2_base,high_throughout",
        "pangu_v2_moe": "pangu_v2_base,low_latency",
    }
    names: list[str] = []
    seen: set[str] = set()
    for token in _split_patch_dir_names(model_type):
        expanded = _split_patch_dir_names(patch_dir_map.get(token.lower(), token))
        for name in expanded:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _get_manual_patches_dir_env() -> str:
    """Return the user-set patches-dir env value, or empty if unset."""
    for name in ("OMNI_VLLM_PATCHES_DIR", "OMNI_NPU_PATCHES_DIR"):
        value = os.getenv(name)
        if value:
            return value
    return ""


def auto_import_patches():
    """
    Load the curated usefull_patch tree:
        1. common/ is always imported
        2. models/<dir> is imported only when named in OMNI_VLLM_PATCHES_DIR
           (comma-separated). ``high_throughout`` / ``low_latency`` also
           pull in ``pangu_v2_base``. Legacy ``pangu_v2_hybrid`` /
           ``pangu_v2_moe`` map to those same paths.
    Files within each directory are sorted by filename.
    """
    vllm_patches_root = Path(__file__).parent
    usefull_patch_dir = vllm_patches_root / "usefull_patch"
    base_pkg = "omni_npu.vllm_patches.usefull_patch"
    models_root = usefull_patch_dir / "models"

    if not usefull_patch_dir.exists():
        logger.warning(
            "usefull_patch directory not found: %s", usefull_patch_dir
        )
        return

    common_dir = usefull_patch_dir / "common"
    if common_dir.exists():
        import_patches_from_dir(common_dir, f"{base_pkg}.common")
        logger.info("loaded patches from %s", common_dir)
    else:
        logger.warning("usefull_patch common directory not found: %s", common_dir)

    model_type = _get_manual_patches_dir_env()
    if not model_type:
        logger.info(
            "OMNI_VLLM_PATCHES_DIR is unset; skip usefull_patch/models/"
        )
        return

    if not models_root.exists():
        logger.warning(
            "usefull_patch models directory not found: %s", models_root
        )
        return

    for model_dir in _find_patch_dir_exact(model_type, models_root):
        import_patches_from_dir(
            model_dir, f"{base_pkg}.models.{model_dir.name}"
        )
        logger.info("loaded patches from %s", model_dir)


manager = PatchManager()


def apply_patches():
    # auto import and register patches from usefull_patch only
    auto_import_patches()

    manager.apply_patches()

    # Run dynamic trace wrapping after normal patches are applied, so namelist
    # targets wrap the final patched methods instead of earlier implementations.
    from omni_npu.vllm_patches.usefull_patch.common.patch_trace import ProfilerDynamicPatch
    ProfilerDynamicPatch()
