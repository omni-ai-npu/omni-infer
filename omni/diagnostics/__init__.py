# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Runtime diagnostics, split into two independent submodules:

* ``dump`` (OMNI-DUMP): capture call stacks, runtime stats and hardware
  state at process exit.
* ``config_summary`` (OMNI-CONF): emit a startup configuration snapshot as
  atomic single-line log records.
"""
