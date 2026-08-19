#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# Shared UT container configuration for multi-docker test scripts.
# Source this file from scripts under tests/.

if [[ -n "${UT_CONFIG_LOADED:-}" ]]; then
  return 0
fi
UT_CONFIG_LOADED=1

UT_CONTAINER_NAMES=(
  "DT_1"
  "DT_2"
  "DT_3"
  "DT_4"
)

# Container name -> Ascend device list
declare -A UT_CONTAINER_ASCEND_DEVICES=(
  ["DT_1"]="0,1,2,3"
  ["DT_2"]="4,5,6,7"
  ["DT_3"]="8,9,10,11"
  ["DT_4"]="12,13,14,15"
)

UT_CONTAINER_OMNI_ROOT_DEFAULT="/workspace/omniinfer"

UT_SPLITS=4
UT_DURATIONS_PATH="${UT_CONTAINER_OMNI_ROOT_DEFAULT}/tests/test_durations_v1.json"
UT_SPLIT_COMMON_ARGS="--splits ${UT_SPLITS} --splitting-algorithm least_duration --durations-path ${UT_DURATIONS_PATH}"

# Container name -> test args for tests/run_tests.sh
declare -A UT_CONTAINER_TEST_ARGS=(
  [DT_1]="all -- ${UT_SPLIT_COMMON_ARGS} --group 1"
  [DT_2]="all -- ${UT_SPLIT_COMMON_ARGS} --group 2"
  [DT_3]="all -- ${UT_SPLIT_COMMON_ARGS} --group 3"
  [DT_4]="all -- ${UT_SPLIT_COMMON_ARGS} --group 4"
)
