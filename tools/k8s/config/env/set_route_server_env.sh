#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${CONFIG_DIR:=$( dirname "${CUR_DIR}" )}"
: "${ROOT_DIR:=$( dirname "${CONFIG_DIR}" )}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/env/env_tools.sh

# 启动脚本使用的环境变量
set_env "CURRENT_STARTUP_ROLE" "route_server"

set_env_from_arg_or_default "ROUTE_SERVER_DEPLOY_DIR" "--route-server-deploy-dir" "/opt/Deploy" "$@"
set_env_from_arg_or_default "ROUTE_SERVER_NAME" "--route-server-name" "route_server" "$@"