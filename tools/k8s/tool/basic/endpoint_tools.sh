#!/bin/bash

function cross_join_ips_and_ports() {
    local ips="$1"
    local ports="$2"

    IFS=',' read -ra ip_list <<< "${ips}"
    IFS=',' read -ra port_list <<< "${ports}"

    local combinations=()
    for ip in "${ip_list[@]}"; do
        for port in "${port_list[@]}"; do
            combinations+=("$ip:$port")
        done
    done

    IFS=,; echo "${combinations[*]}"
}
