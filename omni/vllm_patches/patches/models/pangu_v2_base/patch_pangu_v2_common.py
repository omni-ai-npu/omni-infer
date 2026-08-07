# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Pangu V2模型族通用补丁说明。

本文件用于说明`patches/models/pangu_v2_base/`目录的职责和使用范围。

适用模型：
- `openpangu_v2`
- `openpangu_ultra_omni`

加载顺序：
- `openpangu_v2` -> `pangu_v2_base` -> `pangu_sink_swa_mla`
- `openpangu_ultra_omni` -> `pangu_v2_base` -> `openpangu_ultra_omni`

放在本目录的补丁应满足：
- 同时适用于上述两个模型
- 目标符号和行为保持一致
- 抽取后可以减少重复代码和重复维护

不要放在本目录的补丁：
- 只对其中一个模型生效的补丁
- 两个模型实现差异明显的补丁

后续新增通用补丁时，直接在本目录下新增`patch_*.py`文件即可。
文件会随目录一起按文件名顺序导入。
"""

import logging
logger = logging.getLogger(__name__)
logger.info("Loading Pangu V2 common patch guide for openpangu_v2 and openpangu_ultra_omni")
