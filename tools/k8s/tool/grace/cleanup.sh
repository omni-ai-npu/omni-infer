#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${TOOL_DIR:=$( dirname "${CUR_DIR}" )}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh
source "${TOOL_DIR}"/health/npu_checker.sh

function stop_all_processes_and_check_npus() {
    device_size="${LOCAL_DEVICE_SIZE:-0}"

    echo_with_time "Received SIGTERM, shutting down..."

    ray stop
    ps aux | grep "python" | grep -v "grep" | awk '{print $2}' | xargs kill -9
    ps aux | grep "nginx" | grep -v "grep" | awk '{print $2}' | xargs kill -9
    ps aux | grep "ray" | grep -v "grep" | awk '{print $2}' | xargs kill -9

    sleep 5

    wait

    echo_with_time "Check if all NPUs are free (count: ${device_size})"
    if all_npu_free "${device_size}"; then
        echo "NPUs are free and processes are cleaned up completely"
    else
        echo "NPUs are NOT free"
    fi
    exit 0
}
