#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${ROOT_DIR:=${CUR_DIR}}"
: "${CONFIG_DIR:=${ROOT_DIR}/config}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh
source "${TOOL_DIR}"/grace/business_checker.sh

# set env
source "$CONFIG_DIR"/env/set_pre_stop_env.sh "$@"

# check required envs
required_envs=("DECODE_SERVERS" "PREFILL_SERVERS" "PRE_STOP_LOG_PATH")
for env_name in "${required_envs[@]}"; do
    env_value="${!env_name}"
    if [[ -z "${env_value}" ]]; then
        echo_with_time "ENV ${env_name} does NOT exist, no need to check business"
        exit 0
    fi
done

# check business finished
wait_all_requests_finished &>> "${PRE_STOP_LOG_PATH}"

exit 0
