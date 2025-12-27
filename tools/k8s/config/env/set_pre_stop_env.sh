#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${CONFIG_DIR:=$( dirname "${CUR_DIR}" )}"
: "${ROOT_DIR:=$( dirname "${CONFIG_DIR}" )}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/env/env_tools.sh

set_env_from_arg_or_default "BUSINESS_CHECK_RETRY_INTERVAL" "--business-check-retry-interval" 5 "$@"
set_env_from_arg_or_default "BUSINESS_CHECK_SUCCESS_THRESHOLD" "--business-check-success-threshold" 3 "$@"
set_env_from_arg_or_default "BUSINESS_CHECK_TIMEOUT" "--business-check-timeout" 3600 "$@"