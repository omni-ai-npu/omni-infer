#!/bin/bash
if [[ -f ~/.bashrc ]]; then
    source ~/.bashrc
fi

USER_EXTRA_ARGS=("$@")

export PYTHONHASHSEED=${PYTHONHASHSEED:-123}
inventory_hostname="${INVENTORY_HOSTNAME:-${HOSTNAME:-proxy}}"
LOG_PATH_PROXY=${LOG_PATH_PROXY:-"/home/ma-user/scripts/proxy/log"}
log_dir="${LOG_PATH_PROXY%/}/${inventory_hostname}"
rm -rf "$log_dir"
mkdir -p "$log_dir"
PREFILL_API_SERVER_LIST=''
DECODE_API_SERVER_LIST=''
base_api_port=${base_api_port:-9000}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! IP_LIST=$(python3 "$SCRIPT_DIR/get_node_ip_list.py" --mode all); then
    echo "ERROR: failed to get node IP list" >&2
    exit 1
fi
if [[ -z "$IP_LIST" ]]; then
    echo "ERROR: empty node IP list" >&2
    exit 1
fi
npu=${npu:-16}
npu=$((npu - 1))

IFS=';' read -ra head_ip_list <<< "$IP_LIST"
instance_num=${#head_ip_list[@]}
if (( instance_num < 2 )); then
    echo "ERROR: expected at least one prefill group and one decode group, got: $IP_LIST" >&2
    exit 1
fi
DECODE_INSTANCE_NUM=${DECODE_INSTANCE_NUM:-1}
for index in "${!head_ip_list[@]}"; do
    instance_ip_str="${head_ip_list[$index]}"
    if [[ -z "$instance_ip_str" ]]; then
        echo "ERROR: empty IP group in node IP list: $IP_LIST" >&2
        exit 1
    fi
    IFS=',' read -ra instance_ip_list <<< "$instance_ip_str"
    if [[ $index -lt $((instance_num - DECODE_INSTANCE_NUM)) ]]; then
        PREFILL_API_SERVER_LIST+="${instance_ip_list[0]}:${base_api_port},"
    else
        base_api_port=$((base_api_port + 100))
        for j in "${!instance_ip_list[@]}"; do
            DECODE_API_SERVER_LIST+="${instance_ip_list[$j]}:${base_api_port}@${npu},"
        done
    fi
done

if [[ "$PREFILL_API_SERVER_LIST" == *, ]]; then
    PREFILL_API_SERVER_LIST="${PREFILL_API_SERVER_LIST%,}"
fi
if [[ "$DECODE_API_SERVER_LIST" == *, ]]; then
    DECODE_API_SERVER_LIST="${DECODE_API_SERVER_LIST%,}"
fi

decode_result=""
IFS=',' read -ra decode_array <<< "$DECODE_API_SERVER_LIST"
for var in "${decode_array[@]}"; do
    [[ -n "$var" ]] || continue
    address=${var%@*}
    ip=${address%:*}
    port=${address##*:}
    num=${var#*@}
    if [[ ! "$port" =~ ^[0-9]+$ || ! "$num" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid decode endpoint spec: $var" >&2
        exit 1
    fi
    for ((i=0; i<=$num; i++)); do
        if [[ -z ${decode_result} ]]; then
            decode_result="$ip:$port"
        else
            decode_result="${decode_result},$ip:$port"
        fi
        ((port++))
    done
done

cd /workspace/omniinfer/components/omni-proxy/omni_proxy/ || exit 1
listen_port="${PROXY_NODE_PORT:-${PROXY_PORT:-}}"
if [[ -z "$listen_port" ]]; then
    echo "ERROR: PROXY_NODE_PORT or PROXY_PORT is required" >&2
    exit 1
fi


if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "ERROR: MODEL_PATH is required" >&2
    exit 1
fi

bash omni_proxy.sh \
    --listen-port "$listen_port" \
    --prefill-endpoints "$PREFILL_API_SERVER_LIST" \
    --decode-endpoints "$decode_result" \
    --log-file "$log_dir/nginx_error.log" \
    --log-level ${log_level_proxy:-notice} \
    --access-log-file "$log_dir/nginx_access.log" \
    --core-num ${core_num:-4} \
    --start-core-index ${start_core_index:-16} \
    --omni-proxy-pd-policy ${pd_policy:-sequential} \
    --omni-proxy-model-path "$MODEL_PATH" \
    --omni-proxy-max-batch-num-token ${max_num_batched_tokens:-10000} \
    --omni-proxy-prefill-max-num-seqs ${max_num_seqs:-32} \
    --omni-proxy-decode-max-num-seqs ${max_num_seqs:-6} \
    "${USER_EXTRA_ARGS[@]}"
