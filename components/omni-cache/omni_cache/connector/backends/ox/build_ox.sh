#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
if [ ! -e omni_cache/connector/backends/ox/ox ]; then
  cd omni_cache/connector/backends/ox
  bash build_deps.sh
  make
fi
