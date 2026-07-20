#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

set -e

nginx_conf_file="/usr/local/nginx/conf/nginx.conf"
listen_port="7150"
core_num="4"
start_core_index="0"
encode_endpoints=""
prefill_endpoints=""
decode_endpoints=""
log_file="/tmp/nginx_error.log"
log_level="notice"
access_log_file="/tmp/nginx_access.log"
omni_proxy_pd_policy="sequential"
omni_proxy_model_path=""
omni_proxy_tokenize_chunk_bytes="16384"
omni_proxy_max_batch_num_token="32000"
omni_proxy_prefill_max_num_seqs="32"
omni_proxy_decode_max_num_seqs="32"
omni_proxy_prefill_starvation_timeout="400"
omni_proxy_schedule_algo="default"
prefill_pod_size="1"
decode_pod_size="1"
stream_ops="off"
omni_proxy_max_tokens_weight=""
omni_proxy_prefill_groups=""
omni_proxy_decode_groups=""
omni_proxy_connect_timeout="600s"
omni_proxy_send_timeout="600s"
omni_proxy_read_timeout="14400s"
omni_proxy_next_upstream_timeout="0"
client_max_body_size="10M"
client_body_buffer_size="1M"
subrequest_output_buffer_size="1M"
omni_proxy_max_request_slots="16384"
no_reuseport=false
dry_run=false
stop=false
reload=false
rollback=false
keepalive_nginx=false
set_trace_headers_force="off"

print_help() {
    echo "Usage:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --nginx-conf-file <path>        Path to nginx config file (default: /usr/local/nginx/conf/nginx.conf)"
    echo "  --listen-port <PORT>            Listening port (default: 7150)"
    echo "  --core-num <N>                  Number of CPU cores to use (default: 4)"
    echo "  --start-core-index <N>          Starting CPU core index (default: 0)"
    echo "  --encode-endpoints <list>       Comma-separated backend servers for encoder"
    echo "  --prefill-endpoints <list>      Comma-separated backend servers for prefill"
    echo "  --decode-endpoints <list>       Comma-separated backend servers for decode"
    echo "  --log-file <path>               Log file path (default: /tmp/nginx_error.log)"
    echo "  --log-level <LEVEL>             Log level (default: notice)"
    echo "  --access-log-file <path>        Access log file path (default: /tmp/nginx_access.log)"
    echo "  --omni-proxy-pd-policy <policy> sequential or parallel or aggregation (default: sequential)"
    echo "  --omni-proxy-model-path <path>  Path to model directory (default: unset)"
    echo "  --omni-proxy-tokenize-chunk-bytes <N>     parallel-tokenizer segment target bytes (default: 16384)"
    echo "  --omni-proxy-max-batch-num-token <N>      max_batch_num_token (default: 32000)"
    echo "  --omni-proxy-max-tokens-weight <N>        max_tokens_weight coefficient (default: 0)"
    echo "  --omni-proxy-max-request-slots <N>      max_request_slots (default: 16384)"
    echo "  --stream-ops <add|set_opt|off>             Set stream_ops directive (default: off), set add to enforce turning on streaming and stream_options, set set_opt to only turn on stream_options when streaming is on"
    echo "  --omni-proxy-schedule-algo <algo>       default or earliest_batch (default: default)"
    echo "  --omni-proxy-prefill-max-num-seqs <N>      prefill_max_num_seqs (default: 32)"
    echo "  --omni-proxy-decode-max-num-seqs <N>       decode_max_num_seqs (default: 32)"
    echo "  --omni-proxy-prefill-starvation-timeout <N> prefill_starvation_timeout (default: 400)"
    echo "  --prefill-pod-size <N>          Number of machines per prefill instance (default: 1)"
    echo "  --decode-pod-size <N>           Number of machines per decode instance (default: 1)"
    echo "  --omni-proxy-prefill-groups <list>      Group list for prefill upstreams (e.g. \"0:1,1:1\")"
    echo "  --omni-proxy-decode-groups <list>       Group list for decode upstreams (e.g. \"0:1,1:1\")"
    echo "  --client-max-body-size <size>            client_max_body_size (default: 10M)"
    echo "  --client-body-buffer-size <size>         client_body_buffer_size (default: 1M)"
    echo "  --subrequest-output-buffer-size <size>   subrequest_output_buffer_size (default: 1M)"
    echo "  --omni-proxy-connect-timeout <T>        omni_proxy_connect_timeout (default: 600s, aligned with http proxy_connect_timeout)"
    echo "  --omni-proxy-send-timeout <T>           omni_proxy_send_timeout (default: 600s, aligned with http proxy_send_timeout)"
    echo "  --omni-proxy-read-timeout <T>           omni_proxy_read_timeout (default: 14400s, aligned with http proxy_read_timeout)"
    echo "  --omni-proxy-next-upstream-timeout <T>  omni_proxy_next_upstream_timeout (default: 0, disabled)"
    echo "  --dry-run                       Only generate nginx config, do not start nginx"
    echo "  --stop                          Stop nginx"
    echo "  --reload                        Reload nginx config without restarting"
    echo "  --rollback                      Rollback nginx config if backup exists (must be used with --stop)"
    echo "  --keepalive-nginx               Start new nginx without killing existing nginx"
    echo "  --set_trace_headers_force       Enable trace headers"
    echo "  --help                          Show this help message"
    echo ""
    echo "EXAMPLE:"
    echo "  bash $0 \\"
    echo "      --nginx-conf-file /usr/local/nginx/conf/nginx.conf \\"
    echo "      --core-num 4 \\"
    echo "      --start-core-index 0 \\"
    echo "      --listen-port 7000 \\"
    echo "      --encode-endpoints 127.0.0.1:9200,127.0.0.2:9201 \\"
    echo "      --prefill-endpoints 127.0.0.1:9000,127.0.0.2:9001 \\"
    echo "      --decode-endpoints 127.0.0.3:9100,127.0.0.3:9101 \\"
    echo "      --omni-proxy-pd-policy  sequential \\"
    echo "      --omni-proxy-model-path /data/models/deepseek \\"
    echo "      --omni-proxy-prefill-groups \"0:1,1:1\" \\"
    echo "      --omni-proxy-decode-groups \"0:1,1:1\""
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
        --encode-endpoints)
            encode_endpoints="$2"
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
            if [[ "$2" != "sequential" && "$2" != "parallel" && "$2" != "aggregation" ]]; then
                echo "Error: --omni-proxy-pd-policy must be 'sequential', 'parallel', or 'aggregation'"
                exit 1
            fi
            omni_proxy_pd_policy="$2"
            shift 2
            ;;
        --omni-proxy-model-path)
            omni_proxy_model_path="$2"
            shift 2
            ;;
        --omni-proxy-tokenize-chunk-bytes)
            omni_proxy_tokenize_chunk_bytes="$2"
            shift 2
            ;;
        --omni-proxy-max-batch-num-token)
            omni_proxy_max_batch_num_token="$2"
            shift 2
            ;;
        --omni-proxy-max-tokens-weight)
            omni_proxy_max_tokens_weight="$2"
            shift 2
            ;;
        --omni-proxy-max-request-slots)
            omni_proxy_max_request_slots="$2"
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
        --omni-proxy-prefill-groups)
            omni_proxy_prefill_groups="$2"
            shift 2
            ;;
        --omni-proxy-decode-groups)
            omni_proxy_decode_groups="$2"
            shift 2
            ;;
        --client-max-body-size)
            client_max_body_size="$2"
            shift 2
            ;;
        --client-body-buffer-size)
            client_body_buffer_size="$2"
            shift 2
            ;;
        --subrequest-output-buffer-size)
            subrequest_output_buffer_size="$2"
            shift 2
            ;;
        --omni-proxy-connect-timeout)
            omni_proxy_connect_timeout="$2"
            shift 2
            ;;
        --omni-proxy-send-timeout)
            omni_proxy_send_timeout="$2"
            shift 2
            ;;
        --omni-proxy-read-timeout)
            omni_proxy_read_timeout="$2"
            shift 2
            ;;
        --omni-proxy-next-upstream-timeout)
            omni_proxy_next_upstream_timeout="$2"
            shift 2
            ;;
        --prefill-pod-size)
            prefill_pod_size="$2"
            shift 2
            ;;
        --decode-pod-size)
            decode_pod_size="$2"
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
        --no-reuseport)
            no_reuseport=true
            shift 1
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
        --keepalive-nginx)
            keepalive_nginx=true
            shift 1
            ;;
        --set_trace_headers_force)
            set_trace_headers_force="on"
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

function set_nginx_env_defaults() {
    if [[ -z "${TORCH_DEVICE_BACKEND_AUTOLOAD:-}" ]]; then
        export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    fi

    if [[ -z "${PYTHONHASHSEED:-}" ]]; then
        export PYTHONHASHSEED=123
    fi
}

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
    set_nginx_env_defaults

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
        '\"time\":\"\$msec\",'
        '\"request_id\":\"\$http_x_request_id\",'
        '\"request_time\":\"\$request_time\",'
        '\"upstream_addr\":\"\$upstream_addr\",'
        '\"connection_info\":\"\$connection:\$connection_time\",'
        '\"time_iso8601\":\"\$time_iso8601\",'
        '\"status\":\"\$status\",'
        '\"upstream_connect_time\":\"\$upstream_connect_time\",'
        '\"upstream_header_time\":\"\$upstream_header_time\",'
        '\"upstream_response_time\":\"\$upstream_response_time\",'
        '\"upstream_bytes_received\":\"\$upstream_bytes_received\",'
        '\"body_bytes_sent\":\"\$body_bytes_sent\",'
        '\"promt_tks\":\"\$promt_tks\",'
        '\"decoded_tks\":\"\$decoded_tks\",'
        '\"max_tks\":\"\$max_tks\",'
        '\"prefill_max_match\":\"\$prefill_max_match\",'
        '\"decode_max_match\":\"\$decode_max_match\",'
        '\"encode_idx\":\"\$encode_idx\",'
        '\"prefill_idx\":\"\$prefill_idx\",'
        '\"decode_idx\":\"\$decode_idx\",'
        '\"rcved\":\"\$rcved\",'
        '\"cont_rcved\":\"\$cont_rcved\",'
        '\"e_start\":\"\$e_start\",'
        '\"e_sched\":\"\$e_sched\",'
        '\"e_end\":\"\$e_end\",'
        '\"tknized\":\"\$tknized\",'
        '\"apc\":\"\$apc\",'
        '\"wait_p\":\"\$wait_p\",'
        '\"p_sched\":\"\$p_sched\",'
        '\"to_p\":\"\$to_p\",'
        '\"p_ed\":\"\$p_ed\",'
        '\"wait_d\":\"\$wait_d\",'
        '\"d_sched\":\"\$d_sched\",'
        '\"to_d\":\"\$to_d\",'
        '\"1st_tk\":\"\$1st_tk\",'
        '\"tpot\":\"\$tpot\",'
        '\"ttft\":\"\$ttft\",'
        '\"latency\":\"\$latency\"'
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

    local omni_proxy_max_tokens_weight_directive=""
    if [[ -n "$omni_proxy_max_tokens_weight" ]]; then
        omni_proxy_max_tokens_weight_directive="            omni_proxy_max_tokens_weight $omni_proxy_max_tokens_weight;"
    fi

    local omni_proxy_prefill_groups_directive=""
    if [[ -n "$omni_proxy_prefill_groups" ]]; then
        local normalized_prefill_groups="${omni_proxy_prefill_groups//,/ }"
        omni_proxy_prefill_groups_directive="            omni_proxy_prefill_groups ${normalized_prefill_groups};"
    fi

    local omni_proxy_decode_groups_directive=""
    if [[ -n "$omni_proxy_decode_groups" ]]; then
        local normalized_decode_groups="${omni_proxy_decode_groups//,/ }"
        omni_proxy_decode_groups_directive="            omni_proxy_decode_groups ${normalized_decode_groups};"
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
    omni_proxy_max_request_slots $omni_proxy_max_request_slots;

    $(gen_access_log)
    proxy_http_version 1.1;
    tcp_nodelay on;
    client_max_body_size $client_max_body_size;
    client_body_buffer_size $client_body_buffer_size;

    proxy_read_timeout 14400s;
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
$(if [[ -n "$encode_endpoints" ]]; then
    gen_upstream_block "encode_endpoints" "$encode_endpoints"
fi)
$(if [[ "$omni_proxy_pd_policy" != "aggregation" ]]; then
    gen_upstream_block "prefill_endpoints" "$prefill_endpoints"
fi)

$(gen_upstream_block "decode_endpoints" "$decode_endpoints")

    server {
EOF
    if [ "$no_reuseport" = true ]; then
        cat >> "$nginx_conf_file" <<EOF
        listen $listen_port;
EOF
    else
        cat >> "$nginx_conf_file" <<EOF
        listen $listen_port reuseport;
EOF
    fi

    cat >> "$nginx_conf_file" <<EOF
        server_name localhost;

        location ~ ^/v1(/chat)?/completions$ {
            set_request_id on;
            set_trace_headers_force $set_trace_headers_force;
            omni_proxy decode_endpoints;
            stream_ops $stream_ops;
            omni_proxy_pd_policy $omni_proxy_pd_policy;
            omni_proxy_tokenize_chunk_bytes $omni_proxy_tokenize_chunk_bytes;
            omni_proxy_max_batch_num_token $omni_proxy_max_batch_num_token;
            omni_proxy_prefill_max_num_seqs $omni_proxy_prefill_max_num_seqs;
            omni_proxy_decode_max_num_seqs $omni_proxy_decode_max_num_seqs;
            omni_proxy_prefill_starvation_timeout $omni_proxy_prefill_starvation_timeout;
            prefill_pod_size $prefill_pod_size;
            decode_pod_size $decode_pod_size;
${omni_proxy_schedule_algo_directive}
${omni_proxy_max_tokens_weight_directive}
${omni_proxy_prefill_groups_directive}
${omni_proxy_decode_groups_directive}
            omni_proxy_connect_timeout $omni_proxy_connect_timeout;
            omni_proxy_send_timeout $omni_proxy_send_timeout;
            omni_proxy_read_timeout $omni_proxy_read_timeout;
            omni_proxy_next_upstream_timeout $omni_proxy_next_upstream_timeout;
EOF

    if [[ -n "$omni_proxy_model_path" ]]; then
        cat >> "$nginx_conf_file" <<EOF
            omni_proxy_model_path "$omni_proxy_model_path";
            omni_proxy_vllm_kv_port_offset 100;
EOF
    fi

    cat >> "$nginx_conf_file" <<EOF
            chunked_transfer_encoding off;
            proxy_buffering off;
            send_timeout 1h;
            postpone_output 0;
        }

        location /v1 {
            proxy_pass http://decode_endpoints;
            proxy_http_version 1.1;
            proxy_set_header Connection Keep-Alive;
        }

        location = /omni_proxy/metrics {
            omni_proxy_metrics on;
            default_type text/plain;
        }

        location = /omni_proxy/health {
            omni_proxy_health_status on;
            default_type application/json; 
        }

        location = /health {
            omni_proxy_broadcast on;
            default_type application/json; 
        }

        location = /metrics {
            omni_proxy_broadcast on;
            default_type text/plain;
        }

        location = /sleep {
            omni_proxy_broadcast on;
            default_type application/json;
        }

        location = /wake_up {
            omni_proxy_broadcast on;
            default_type application/json;
        }

        location = /pause {
            omni_proxy_broadcast on;
            default_type application/json;
        }

        location = /resume {
            omni_proxy_broadcast on;
            default_type application/json;
        }

        location = /internal/metrics {
            rewrite ^ /omni_proxy/metrics last;
        }
        
        location ~ ^/prefill_sub(?<orig>/.*)\$ {
            internal;
            proxy_pass http://prefill_endpoints\$orig\$is_args\$args;
            subrequest_output_buffer_size $subrequest_output_buffer_size;
        }

        $(if [[ -n "$encode_endpoints" ]]; then
cat <<ENCODE_EOF
location ~ ^/encode_sub(?<orig>/.*)$ {
            internal;
            proxy_pass http://encode_endpoints\$orig\$is_args\$args;
            subrequest_output_buffer_size $subrequest_output_buffer_size;
        }
ENCODE_EOF
fi)

        location = /omni_proxy_broadcast_sub {
            internal;
            proxy_pass http://\$arg_target;
            subrequest_output_buffer_size $subrequest_output_buffer_size;
        }

        location ~ ^/proxy/(?<backend>\d+\.\d+\.\d+\.\d+:\d+)(?<rest>/.*)$ {
            proxy_pass http://\$backend\$rest\$is_args\$args;
        }

    }
}
EOF

    echo "nginx.conf generated at $nginx_conf_file"
}

function do_start() {
    set_nginx_env_defaults

    if [[ -z "$decode_endpoints" ]]; then
        echo "Error: --decode-endpoints is required"
        exit 1
    fi

    if [[ "$omni_proxy_pd_policy" != "aggregation" && -z "$prefill_endpoints" ]]; then
        echo "Error: --prefill-endpoints is required when pd policy is not aggregation"
        exit 1
    fi

    generate_nginx_conf

    if [ "$dry_run" = true ]; then
        echo "Dry run complete. Configuration generated at $nginx_conf_file."
        exit 0
    fi

    if [ "$keepalive_nginx" = false ]; then
        stop_nginx
    fi

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
