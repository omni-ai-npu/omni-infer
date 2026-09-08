# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model backends. Importing registers the built-ins."""
from jointfix.backends.base import LayerSpec, ModelBackend  # noqa: F401
from jointfix.backends import hf as hf        # noqa: F401  (registers "hf")
from jointfix.backends import pangu as pangu  # noqa: F401  (registers "pangu")
