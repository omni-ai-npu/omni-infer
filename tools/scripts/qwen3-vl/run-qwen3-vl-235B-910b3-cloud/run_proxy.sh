#!/bin/bash

log_file="${LOG_PATH}/proxy_run.log"
exec 3>&1
exec 1>>"$log_file" 2>&1

run_proxy() {
    sudo pkill nginx
    sleep 1
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

HEALTH_URL="http://127.0.0.1:$PROXY_PORT/v1/chat/completions"
HEALTH_INTERVAL=60
HEALTH_TIMEOUT=5

health_check() {
    local http_code
    http_code=$(curl -s -m $HEALTH_TIMEOUT \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"model":"fake","messages":[{"role":"user","content":"health"}],"max_tokens":1}' \
        -o /dev/null \
        -w "%{http_code}" \
        "$HEALTH_URL" 2>&1)
    
    if [ -z "$http_code" ] || [ "$http_code" = "000" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Health FAILED: timeout/no response (curl: curl -X POST -H 'Content-Type: application/json' -d '{\"model\":\"fake\",\"messages\":[{\"role\":\"user\",\"content\":\"health\"}],\"max_tokens\":1}' -m $HEALTH_TIMEOUT $HEALTH_URL)"
        return 1
    elif [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Health WARNING: HTTP $http_code (ignored)"
        return 0
    fi
}

while true; do
    sleep $HEALTH_INTERVAL
    if ! health_check; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting proxy..."
        run_proxy
        sleep 10
    fi
done