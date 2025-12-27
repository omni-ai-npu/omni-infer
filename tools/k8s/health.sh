#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${ROOT_DIR:=${CUR_DIR}}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh
source "${TOOL_DIR}"/health/health_checker.sh

if [ -z "$1" ]; then
  PROBE_TYPE=liveness
else
  PROBE_TYPE=$1
fi

CURRENT_PD_ROLE="{{CURRENT_PD_ROLE}}"
ENDPOINTS="{{ENDPOINTS}}"
IS_HEAD="{{IS_HEAD}}"
PROBE_LOG_PATH="{{PROBE_LOG_PATH}}"
TRANSFER_TORCHAIR_CACHE="{{TRANSFER_TORCHAIR_CACHE}}"

if [ "${PROBE_TYPE:-}" = "startup" ]; then
    if all_servers_are_health "${ENDPOINTS}" 0 "${PROBE_LOG_PATH}"; then
        echo_with_time "${CURRENT_PD_ROLE} has already started up." >> "${PROBE_LOG_PATH}" 2>&1
        echo "${CURRENT_PD_ROLE} has already started up."
        if [ "${CURRENT_PD_ROLE:-}" = "decode" ] && [ "${TRANSFER_TORCHAIR_CACHE:-}" = "1" ]; then
            TORCHAIR_CACHE_SRC_PATH="{{TORCHAIR_CACHE_SRC_PATH}}"
            TORCHAIR_CACHE_DEST_PATH="{{TORCHAIR_CACHE_DEST_PATH}}"
            move_dir "torchair cache" ${TORCHAIR_CACHE_SRC_PATH} ${TORCHAIR_CACHE_DEST_PATH} &>> "${PROBE_LOG_PATH}"
        fi
    else
        echo "${CURRENT_PD_ROLE} has NOT started up."
        exit 1
    fi
elif [ "${IS_HEAD}" -eq 1 ]; then
    if all_servers_are_health "${ENDPOINTS}" 1 "${PROBE_LOG_PATH}"; then
        echo "${CURRENT_PD_ROLE} is alive."
    else
        echo_with_time "${CURRENT_PD_ROLE} is not alive" >> "${PROBE_LOG_PATH}" 2>&1
        echo "${CURRENT_PD_ROLE} is not alive"
        exit 1
    fi
else
    echo "workers of ${CURRENT_PD_ROLE} currently do not need health check."
fi
