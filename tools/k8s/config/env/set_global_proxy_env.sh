#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${CONFIG_DIR:=$( dirname "${CUR_DIR}" )}"
: "${ROOT_DIR:=$( dirname "${CONFIG_DIR}" )}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/env/env_tools.sh

# 启动脚本使用的环境变量
set_env "CURRENT_STARTUP_ROLE" "global-proxy"

# 启动参数使用的环境变量
set_env_from_arg_or_default "CORE_NUM" "--core-num" "16" "$@"
set_env_from_arg_or_default "DECODE_LB_SDK" "--decode-lb-sdk" "least_conn" "$@"
set_env_from_arg_or_default "PREFILL_LB_SDK" "--prefill-lb-sdk" "least_conn" "$@"
set_env_from_arg_or_default "START_CORE_INDEX" "--start-core-index" "0" "$@"