#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${TOOL_DIR:=$( dirname "${CUR_DIR}" )}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh

function npu_smi_works_well() {
    if ! npu-smi info; then
        echo_with_time "npu-smi info does NOT work well"
        return 1
    else
        echo_with_time "npu-smi info works well"
        return 0
    fi
}

function all_cards_free() {
    local card_count
    local free_card_count

    card_count=$(npu-smi info | awk '/Process id/,0' | head -n -1 | grep -c "+==")
    free_card_count=$(npu-smi info | grep -c "No running processes")

    echo_with_time "free cards: ${free_card_count}/${card_count}"

    if [ "${free_card_count}" -eq "${card_count}" ]; then
        return 0
    else
        return 1
    fi
}

function all_chips_healthy() {
    local chip_count
    local health_chip_count

    chip_count=$(npu-smi info | awk '/Process id/{exit} /Health/,0' | head -n -2 | grep -c "+")
    health_chip_count=$(npu-smi info | awk '/Process id/{exit} /Health/,0' | head -n -2 | grep -c "OK")

    echo_with_time "healthy chips: ${health_chip_count}/${chip_count}"

    if [ "${health_chip_count}" -eq "${chip_count}" ]; then
        return 0
    else
        return 1
    fi
}

function chip_count_meets_requirement() {
    local expected_count="$1"
    local actual_count

    actual_count=$(npu-smi info | awk '/Process id/{exit} /Health/,0' | head -n -2 | grep -c "+")

    echo_with_time "actual chips: ${actual_count}/${expected_count}"

    if [ "${actual_count}" -ge "${expected_count}" ]; then
        return 0
    else
        return 1
    fi
}
