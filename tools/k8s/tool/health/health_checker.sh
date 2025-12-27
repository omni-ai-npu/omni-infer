#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${TOOL_DIR:=$( dirname "${CUR_DIR}" )}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh

function all_servers_are_health() {
    local endpoints="$1"
    local enable_log="$2"
    local log_path="$3"
    local unhealthy_instances=()

    IFS=',' read -ra endpoint_array <<< "${endpoints}"

    for endpoint in "${endpoint_array[@]}"; do
        local health_url="http://${endpoint}/health"
        if ! curl -sL --fail "${health_url}" -o /dev/null; then
            unhealthy_instances+=("${endpoint}")
        fi
    done

    if [ ${#unhealthy_instances[@]} -eq 0 ]; then
        return 0
    else
        if [ "${enable_log}" -eq 1 ]; then
            echo_with_time "Unhealthy instances: ${unhealthy_instances[*]}" >> "${log_path}"
        fi
        return 1
    fi
}
