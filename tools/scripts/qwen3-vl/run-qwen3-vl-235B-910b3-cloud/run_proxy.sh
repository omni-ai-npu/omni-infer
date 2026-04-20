#!/bin/bash

log_file="${LOG_PATH}/proxy_run.log"
rm -rf $log_file
exec 3>&1
exec 1>>"$log_file" 2>&1

run_proxy() {
    
    cd /workspace/omniinfer/components/omni-proxy/omni_proxy/
    sudo PYTHONHASHSEED=123 bash -l omni_proxy.sh \
        --nginx-conf-file /usr/local/nginx/conf/nginx.conf \
        --start-core-index 0 \
        --core-num $DP \
        --omni-proxy-decode-max-num-seqs 512 \
        --listen-port $PROXY_PORT \
        --decode-endpoints $PROXY_API_LIST \
        --omni-proxy-pd-policy $PROXY_TYPE \
        --omni-proxy-model-path $MODEL_PATH
}

sleep 10
run_proxy