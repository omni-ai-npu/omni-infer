# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_utils.py

import pytest
try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None
import os
from unittest.mock import Mock, patch, MagicMock
from typing import NamedTuple, Optional, Any

# ==================== Global patches ====================
