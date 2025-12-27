#!/bin/bash

function echo_with_time() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

function move_dir() {
    local dir_name="$1"
    local src_path="$2"
    local dest_path="$3"

    echo ""
    echo_with_time "begin to check src $dir_name."
    find "$src_path" -type f -exec ls -l {} \;
    echo_with_time "finish to check src $dir_name."
    echo ""
    echo_with_time "begin to move $dir_name from $src_path to $dest_path."
    cp -Rf "${src_path}" "${dest_path}"
    echo_with_time "succeed to move $dir_name."
    echo ""
    echo_with_time "begin to check dest $dir_name."
    find "$dest_path" -type f -exec ls -l {} \;
    echo_with_time "finish to check dest $dir_name."
}
