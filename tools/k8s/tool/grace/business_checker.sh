#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${TOOL_DIR:=$( dirname "${CUR_DIR}" )}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh

function wait_all_requests_finished() {
    local retry_interval="${BUSINESS_CHECK_RETRY_INTERVAL}"
    local start_time=$SECONDS
    local success_count=0
    local success_threshold="${BUSINESS_CHECK_SUCCESS_THRESHOLD}"
    local timeout="${BUSINESS_CHECK_TIMEOUT}"

    echo_with_time "Begin to wait for business over"

    while true; do
        local elapsed=$((SECONDS - start_time))
        if (( elapsed >= timeout )); then
            echo_with_time "waiting for business over timeout, begin to force kill processes"
            break
        fi

        all_requests_finished=true
        IFS=',' read -ra server_array <<< "${PREFILL_SERVERS},${DECODE_SERVERS}"
        for ((i=0; i<${#server_array[@]}; i++)); do
            local server=${server_array[i]}
            local url="${server}"/metrics
            local num_requests_running
            local num_requests_waiting
            if curl -sL --fail "${url}" -o /dev/null; then
                num_requests_running=$(curl -s "${url}" | grep "vllm:num_requests_running{engine=" | awk '{print $2}')
                num_requests_waiting=$(curl -s "${url}" | grep "vllm:num_requests_waiting{engine=" | awk '{print $2}')
            else
                num_requests_running="0.0"
                num_requests_waiting="0.0"
            fi
            if [[ -z "${num_requests_running}" ]] || [[ ! "${num_requests_running}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "${num_requests_running}" == "0" ]]; then
                num_requests_running="0.0"
            fi
            if [[ -z "${num_requests_waiting}" ]] || [[ ! "${num_requests_waiting}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "${num_requests_waiting}" == "0" ]]; then
                num_requests_waiting="0.0"
            fi
            if [ "${num_requests_running}" != "0.0" ] || [ "${num_requests_waiting}" != "0.0" ]; then
                echo_with_time "${server} does NOT finish, running count: ${num_requests_running}, waiting count: ${num_requests_waiting}"
                all_requests_finished=false
                success_count=0
            fi
        done

        if $all_requests_finished; then
            success_count=$((success_count + 1))
            echo_with_time "check all requests finished PASS for ${success_count}/${success_threshold} times"
        fi

        if (( success_count >= success_threshold )); then
            echo_with_time "All requests finished, begin to kill processes"
            break
        fi

        sleep "${retry_interval}"
    done
}
