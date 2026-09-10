# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Packaging checks: the prestop script must ship with the wheel."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[4]
SCRIPT = ROOT / "omni" / "script" / "omni_npu_prestop.sh"


class TestPrestopShipping:
    def test_script_exists_with_shebang(self):
        assert SCRIPT.exists()
        first_line = SCRIPT.read_text().splitlines()[0]
        assert first_line.startswith("#!"), "prestop script must start with a shebang"

    def test_pyproject_declares_package_data(self):
        pyproject = (ROOT / "pyproject.toml").read_text()
        assert "[tool.setuptools.package-data]" in pyproject
        assert '"omni_npu.script" = ["*.sh"]' in pyproject

    def test_script_dir_is_a_package(self):
        assert (SCRIPT.parent / "__init__.py").exists(), (
            "package-data only ships with a real package"
        )
