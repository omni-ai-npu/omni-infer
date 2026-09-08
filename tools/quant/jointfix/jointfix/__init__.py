# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
jointfix — multi-model, pluggable-method INT8 quantization toolkit.

Two orthogonal axes:
  backends/  — MODEL axis  (pangu, hf, …)
  methods/   — METHOD axis (jointfix, … future quarot/spinquant/awq)
glued by a model- and method-agnostic core/ (primitives, stats, runner).

Importing the package registers the built-in backends and methods.
"""
from __future__ import annotations

__version__ = "0.0.1"

# Trigger registry population (decorators run on import).
from jointfix import backends as backends  # noqa: F401,E402
from jointfix import methods as methods    # noqa: F401,E402
