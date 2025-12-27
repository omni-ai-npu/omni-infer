#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${TOOL_DIR:=$( dirname "${CUR_DIR}" )}"

: "${NPU_CHECK_MAX_RETRIES:=60}"
: "${NPU_CHECK_RETRY_INTERVAL:=2}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh
source "${TOOL_DIR}"/npu/npu_smi_tools.sh

function all_npu_ready() {
    local expected_count="$1"

    if [ "${expected_count}" -eq 0 ]; then
        echo_with_time "INFO: ${expected_count} NPU is needed, no need to check!"
        return 0
    fi

    local max_retries="${NPU_CHECK_MAX_RETRIES}"
    local retry_interval="${NPU_CHECK_RETRY_INTERVAL}"
    local retry=0

    local npu_smi_check_result=false
    local chip_count_check_result=false
    local chip_health_check_result=false

    while [ "${retry}" -lt "${max_retries}" ]; do
        if ! ${npu_smi_check_result} && ! npu_smi_works_well; then
            sleep "$retry_interval"
            retry=$((retry+1))
            continue
        else
            npu_smi_check_result=true
        fi

        if ! ${chip_count_check_result} && ! chip_count_meets_requirement "$expected_count"; then
            sleep "$retry_interval"
            retry=$((retry+1))
            continue
        else
            chip_count_check_result=true
        fi

        if ! ${chip_health_check_result} && ! all_chips_healthy; then
            sleep "$retry_interval"
            retry=$((retry+1))
            continue
        else
            chip_health_check_result=true
        fi

        if ! all_cards_free; then
            sleep "${retry_interval}"
            retry=$((retry+1))
            continue
        fi

        return 0
    done

    echo_with_time "FATAL: NPUs NOT ready after $max_retries attempts"
    return 1
}

function all_npu_free() {
    local expected_count="$1"

    if [ "${expected_count}" -eq 0 ]; then
        echo_with_time "INFO: ${expected_count} NPU was needed, no need to check!"
        return 0
    fi

    local max_retries="${NPU_CHECK_MAX_RETRIES}"
    local retry_interval="${NPU_CHECK_RETRY_INTERVAL}"
    local retry=0

    local npu_smi_check_result=false

    while [ "${retry}" -lt "${max_retries}" ]; do
        if ! ${npu_smi_check_result} && ! npu_smi_works_well; then
            sleep "$retry_interval"
            retry=$((retry+1))
            continue
        else
            npu_smi_check_result=true
        fi

        if ! all_cards_free; then
            sleep "${retry_interval}"
            retry=$((retry+1))
            continue
        fi

        return 0
    done

    echo_with_time "FATAL: NPUs NOT free after $max_retries attempts"
    return 1
}
