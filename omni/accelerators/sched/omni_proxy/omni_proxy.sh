#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

set -e

nginx_conf_file="/usr/local/nginx/conf/nginx.conf"
listen_port="7150"
core_num="4"
start_core_index="0"
prefill_endpoints=""
decode_endpoints=""
log_file="/tmp/nginx_error.log"
log_level="notice"
access_log_file="/tmp/nginx_access.log"
omni_proxy_pd_policy="sequential"
omni_proxy_model_path=""
omni_proxy_max_batch_num_token="32000"
omni_proxy_prefill_max_num_seqs="32"
omni_proxy_decode_max_num_seqs="32"
omni_proxy_prefill_starvation_timeout="400"
omni_proxy_schedule_algo="default"
stream_ops="off"

dry_run=false
stop=false
reload=false
rollback=false

print_help() {
    echo "Usage:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --nginx-conf-file <path>        Path to nginx config file (default: /usr/local/nginx/conf/nginx.conf)"
    echo "  --listen-port <PORT>            Listening port (default: 7150)"
    echo "  --core-num <N>                  Number of CPU cores to use (default: 4)"
    echo "  --start-core-index <N>          Starting CPU core index (default: 0)"
    echo "  --prefill-endpoints <list>      Comma-separated backend servers for prefill"
    echo "  --decode-endpoints <list>       Comma-separated backend servers for decode"
    echo "  --log-file <path>               Log file path (default: /tmp/nginx_error.log)"
    echo "  --log-level <LEVEL>             Log level (default: notice)"
    echo "  --access-log-file <path>        Access log file path (default: /tmp/nginx_access.log)"
    echo "  --omni-proxy-pd-policy <policy> sequential or parallel (default: sequential)"
    echo "  --omni-proxy-model-path <path>  Path to model directory (default: unset)"
    echo "  --omni-proxy-max-batch-num-token <N>      max_batch_num_token (default: 32000)"
    echo "  --stream-ops <add|set_opt|off>             Set stream_ops directive (default: off), set add to enforce turning on streaming and stream_options, set set_opt to only turn on stream_options when streaming is on"
    echo "  --omni-proxy-schedule-algo <algo>       default or earliest_batch (default: default)"
    echo "  --omni-proxy-prefill-max-num-seqs <N>      prefill_max_num_seqs (default: 32)"
    echo "  --omni-proxy-decode-max-num-seqs <N>       decode_max_num_seqs (default: 32)"
    echo "  --omni-proxy-prefill-starvation-timeout <N> prefill_starvation_timeout (default: 400)"
    echo "  --dry-run                       Only generate nginx config, do not start nginx"
    echo "  --stop                          Stop nginx"
    echo "  --reload                        Reload nginx config without restarting"
    echo "  --rollback                      Rollback nginx config if backup exists (must be used with --stop)"
    echo "  --help                          Show this help message"
    echo ""
    echo "EXAMPLE:"
    echo "  bash $0 \\"
    echo "      --nginx-conf-file /usr/local/nginx/conf/nginx.conf \\"
    echo "      --core-num 4 \\"
    echo "      --start-core-index 0 \\"
    echo "      --listen-port 7000 \\"
    echo "      --prefill-endpoints 127.0.0.1:9000,127.0.0.2:9001 \\"
    echo "      --decode-endpoints 127.0.0.3:9100,127.0.0.3:9101 \\"
    echo "      --omni-proxy-pd-policy  sequential \\"
    echo "      --omni-proxy-model-path /data/models/deepseek"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nginx-conf-file)
            nginx_conf_file="$2"
            shift 2
            ;;
        --listen-port)
            listen_port="$2"
            shift 2
            ;;
        --core-num)
            core_num="$2"
            shift 2
            ;;
        --start-core-index)
            start_core_index="$2"
            shift 2
            ;;
        --prefill-endpoints)
            prefill_endpoints="$2"
            shift 2
            ;;
        --decode-endpoints)
            decode_endpoints="$2"
            shift 2
            ;;
        --log-file)
            log_file="$2"
            shift 2
            ;;
        --log-level)
            log_level="$2"
            shift 2
            ;;
        --access-log-file)
            access_log_file="$2"
            shift 2
            ;;
        --omni-proxy-pd-policy)
            if [[ "$2" != "sequential" && "$2" != "parallel" ]]; then
                echo "Error: --omni-proxy-pd-policy must be 'sequential' or 'parallel'"
                exit 1
            fi
            omni_proxy_pd_policy="$2"
            shift 2
            ;;
        --omni-proxy-model-path)
            omni_proxy_model_path="$2"
            shift 2
            ;;
        --omni-proxy-max-batch-num-token)
            omni_proxy_max_batch_num_token="$2"
            shift 2
            ;;
        --omni-proxy-prefill-max-num-seqs)
            omni_proxy_prefill_max_num_seqs="$2"
            shift 2
            ;;
        --omni-proxy-decode-max-num-seqs)
            omni_proxy_decode_max_num_seqs="$2"
            shift 2
            ;;
        --omni-proxy-prefill-starvation-timeout)
            omni_proxy_prefill_starvation_timeout="$2"
            shift 2
            ;;
        --stream-ops)
            case "$2" in
                add|set_opt|off)
                    stream_ops="$2"
                    ;;
                *)
                    echo "Error: --stream-ops must be 'add', 'set_opt', or 'off'"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --omni-proxy-schedule-algo)
            if [[ "$2" != "default" && "$2" != "earliest_batch" ]]; then
                echo "Error: --omni-proxy-schedule-algo must be 'default' or 'earliest_batch'"
                exit 1
            fi
            omni_proxy_schedule_algo="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift 1
            ;;
        --stop)
            stop=true
            shift 1
            ;;
        --reload)
            reload=true
            shift 1
            ;;
        --rollback)
            rollback=true
            shift 1
            ;;
        --help|-h)
            print_help
            ;;
        *)
            echo "Unknown argument: $1"
            print_help
            ;;
    esac
done

function stop_nginx() {
    while pgrep nginx > /dev/null; do
        echo "Stopping existing nginx ..."
        pgrep nginx | xargs kill -15
        sleep 1
    done
    echo "Nginx stopped."
}

function start_nginx() {
    local nginx_conf_file="$1"
    # nginx -t -c "$nginx_conf_file"
    if [ $? -ne 0 ]; then
        echo "Error: nginx config $nginx_conf_file is invalid. Exiting."
        exit 1
    fi
    echo "Starting nginx with config $nginx_conf_file..."
    nginx -c "$nginx_conf_file"
}

function do_reload() {
    echo "Generating new nginx config for reload..."
    generate_nginx_conf

    echo "Testing new nginx config..."
    if ! nginx -t -c "$nginx_conf_file"; then
        echo "Error: New nginx config is invalid. Aborting reload."
        rollback_nginx_conf "$nginx_conf_file"
        exit 1
    fi

    echo "Reloading nginx with new config..."
    if ! nginx -s reload; then
        echo "Error: Failed to reload nginx. Check nginx error logs."
        exit 1
    fi
    echo "Nginx reloaded successfully."
}

function rollback_nginx_conf() {
    local nginx_conf_file="$1"
    local backup_file="${nginx_conf_file}_bak"
    if [[ -f "$backup_file" ]]; then
        \cp "$backup_file" "$nginx_conf_file"
        echo "Rolled back nginx config to $backup_file"
    else
        echo "No backup config found to rollback."
    fi
}

function gen_affinity_masks() {
    local start_core_index=$1
    local core_num=$2
    local masks=()
    for ((i=0; i<core_num; i++)); do
        local mask=""
        local total_bits=$((start_core_index + core_num))
        for ((j=0; j<total_bits; j++)); do
            if (( j == start_core_index + i )); then
                mask="1$mask"
            else
                mask="0$mask"
            fi
        done
        mask=$(echo "$mask" | sed 's/^0*//')
        masks+=("$mask")
    done
    echo "${masks[@]}"
}

function gen_upstream_block() {
    local name="$1"
    local endpoints="$2"
    local block="    upstream $name {\n"
    block+="        keepalive 2048;\n"
    block+="        keepalive_timeout 110s;\n"
    block+="        keepalive_requests 20000;\n"
    IFS=',' read -ra list <<< "$endpoints"
    for addr in "${list[@]}"; do
        block+="        server $addr max_fails=3 fail_timeout=10s;\n"
    done
    block+="    }"
    echo -e "$block"
}

function gen_access_log() {
    echo "log_format json_combined escape=json
    '{'
        'time:\"\$msec\",'
        'request_id:\"\$http_x_request_id\",'
        'request_time:\"\$request_time\",'
        'upstream_addr:\"\$upstream_addr\",'
        'connection_info:\"\$connection:\$connection_time\",'
        'time_iso8601:\"\$time_iso8601\",'
        'status:\"\$status\",'
        'upstream_connect_time:\"\$upstream_connect_time\",'
        'upstream_header_time:\"\$upstream_header_time\",'
        'upstream_response_time:\"\$upstream_response_time\",'
        'upstream_bytes_received:\"\$upstream_bytes_received\",'
        'body_bytes_sent:\"\$body_bytes_sent\",'
        'promt_tks:\"\$promt_tks\",'
        'decoded_tks:\"\$decoded_tks\",'
        'max_tks:\"\$max_tks\",'
        'max_match:\"\$max_match\",'
        'prefill_idx:\"\$prefill_idx\",'
        'decode_idx:\"\$decode_idx\",'
        'rcved:\"\$rcved\",'
        'cont_rcved:\"\$cont_rcved\",'
        'tknized:\"\$tknized\",'
        'apc:\"\$apc\",'
        'wait_p:\"\$wait_p\",'
        'p_sched:\"\$p_sched\",'
        'to_p:\"\$to_p\",'
        'p_ed:\"\$p_ed\",'
        'wait_d:\"\$wait_d\",'
        'd_sched:\"\$d_sched\",'
        'to_d:\"\$to_d\",'
        '1st_tk:\"\$1st_tk\",'
        'tpot:\"\$tpot\",'
        'ttft:\"\$ttft\",'
        'latency:\"\$latency\"'
    '}';
    access_log ${access_log_file} json_combined;
    "
}

function generate_nginx_conf() {
    affinity_masks=$(gen_affinity_masks "$start_core_index" "$core_num")
    # backup config if exists
    if [[ -f "$nginx_conf_file" ]]; then
        \cp -n "$nginx_conf_file" "${nginx_conf_file}_bak"
    fi

    # Ensure log_file directory exists
    if [[ -n "$log_file" ]]; then
        log_dir=$(dirname "$log_file")
        if [[ ! -d "$log_dir" ]]; then
            mkdir -p "$log_dir"
        fi
    fi

    local omni_proxy_schedule_algo_directive=""
    if [[ -n "$omni_proxy_schedule_algo" ]]; then
        omni_proxy_schedule_algo_directive="            omni_proxy_schedule_algo $omni_proxy_schedule_algo;"
    fi

    cat > "$nginx_conf_file" <<EOF
load_module /usr/local/nginx/modules/ngx_http_omni_proxy_module.so;
load_module /usr/local/nginx/modules/ngx_http_set_request_id_module.so;

env PYTHONHASHSEED;
env TORCH_DEVICE_BACKEND_AUTOLOAD;
env VLLM_PLUGINS;
env LD_LIBRARY_PATH;
user root;

worker_processes $core_num;
worker_rlimit_nofile 102400;
worker_cpu_affinity $affinity_masks;

error_log  $log_file  $log_level;

events {
    use epoll;
    accept_mutex off;
    multi_accept on;
    worker_connections 102400;
}

http {
    $(gen_access_log)
    proxy_http_version 1.1;
    tcp_nodelay on;
    client_max_body_size 10M;
    client_body_buffer_size 1M;

    proxy_read_timeout 14400s;
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;

$(gen_upstream_block "prefill_endpoints" "$prefill_endpoints")
    
$(gen_upstream_block "decode_endpoints" "$decode_endpoints")

    server {
        listen $listen_port reuseport;
        server_name localhost;

        location /v1 {
            set_request_id on;
            omni_proxy decode_endpoints;
            stream_ops $stream_ops;
            omni_proxy_pd_policy $omni_proxy_pd_policy;
            omni_proxy_max_batch_num_token $omni_proxy_max_batch_num_token;
            omni_proxy_prefill_max_num_seqs $omni_proxy_prefill_max_num_seqs;
            omni_proxy_decode_max_num_seqs $omni_proxy_decode_max_num_seqs;
            omni_proxy_prefill_starvation_timeout $omni_proxy_prefill_starvation_timeout;
${omni_proxy_schedule_algo_directive}
EOF

    if [[ -n "$omni_proxy_model_path" ]]; then
        cat >> "$nginx_conf_file" <<EOF
            omni_proxy_model_path $omni_proxy_model_path;
            omni_proxy_vllm_kv_port_offset 100;
EOF
    fi

    cat >> "$nginx_conf_file" <<EOF
            chunked_transfer_encoding off;
            proxy_buffering off;
            send_timeout 1h;
            postpone_output 0;
        }

        location = /omni_proxy/metrics {
            omni_proxy_metrics on;
            default_type text/plain;
        }

        location = /omni_proxy/health {
            omni_proxy_health_status on;
            default_type application/json; 
        }

        location = /internal/metrics {
            rewrite ^ /omni_proxy/metrics last;
        }
        
        location ~ ^/prefill_sub(?<orig>/.*)\$ {
            internal;
            proxy_pass http://prefill_endpoints\$orig\$is_args\$args;
            subrequest_output_buffer_size 1M;
        }
    }
}
EOF

    echo "nginx.conf generated at $nginx_conf_file"
}

function do_start() {
    if [[ -z "$prefill_endpoints" || -z "$decode_endpoints" ]]; then
        echo "Error: --prefill-endpoints and --decode-endpoints are required"
        exit 1
    fi

    generate_nginx_conf

    if [ "$dry_run" = true ]; then
        echo "Dry run complete. Configuration generated at $nginx_conf_file."
        exit 0
    fi

    stop_nginx
    start_nginx "$nginx_conf_file"
}

function do_stop() {
    stop_nginx
    if [ "$rollback" = true ]; then
        rollback_nginx_conf "$nginx_conf_file"
    fi
}

function main() {
    if [ "$reload" = true ]; then
        do_reload
    elif [ "$stop" = false ]; then
        do_start
    else
        do_stop
    fi
}

main