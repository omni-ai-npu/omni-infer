#!/bin/bash

ray_log_path=$1
if [ -z "$ray_log_path" ]; then
    ray_log_path="/tmp/ray/session_*/logs"
fi

log_file="${ray_log_path}/log_rotate.log"
current_date_time=$(date +%Y-%m-%d-%H-%M-%S)
echo "==========start do logrotate at $current_date_time by crontab task=========="
/usr/sbin/logrotate -v /etc/logrotate.d/ray/* 2>> "$log_file"