# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Quantization methods. Importing registers the built-ins."""
from jointfix.methods.base import QuantMethod  # noqa: F401
from jointfix.methods import jointfix as jointfix  # noqa: F401  (registers "jointfix")
from jointfix.methods import jointfix_mdmixq as jointfix_mdmixq  # noqa: F401
