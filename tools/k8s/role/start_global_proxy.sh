#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${ROOT_DIR:=$( dirname "${CUR_DIR}" )}"
: "${CONFIG_DIR:=${ROOT_DIR}/config}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh

# set specific env
echo_with_time "------------> set environment variable for ${CURRENT_STARTUP_ROLE}"
source "${CONFIG_DIR}"/env/set_global_proxy_env.sh "$@"
echo_with_time "------------> set environment variable for ${CURRENT_STARTUP_ROLE} finished"

echo_with_time "------------> set variable of health.sh for ${CURRENT_STARTUP_ROLE}"
sed -i "s|{{CURRENT_PD_ROLE}}|${CURRENT_STARTUP_ROLE}|g" "${ROOT_DIR}"/health.sh
sed -i "s|{{ENDPOINTS}}|${PREFILL_SERVERS},${DECODE_SERVERS}|g" "${ROOT_DIR}"/health.sh
sed -i "s|{{IS_HEAD}}|1|g" "${ROOT_DIR}"/health.sh
echo_with_time "------------> set variable of health.sh for ${CURRENT_STARTUP_ROLE} finished"

echo_with_time "------------> check all environment variable for ${CURRENT_STARTUP_ROLE}"
env
echo_with_time "------------> check all environment variable for ${CURRENT_STARTUP_ROLE} finished"

echo_with_time "------------> start global proxy service"
# cd "${SCRIPT_DIR}" || exit
# # bash global_proxy.sh --listen-port "${PROXY_PORT}" \
# #     --prefill-servers-list "${PREFILL_SERVERS}" \
# #     --decode-servers-list "${DECODE_SERVERS}" \
# #     --prefill-lb-sdk "${PREFILL_LB_SDK}" \
# #     --decode-lb-sdk "${DECODE_LB_SDK}" \
# #     --core-num "${CORE_NUM}" \
# #     --start-core-index "${START_CORE_INDEX}" \
# #     --enable-internal-metrics 1m \
# #     --stream-ops set_opt \
# #     --log-file "${BASE_LOG_DIR}"/proxy.log \
# #     --log-level notice &

export PYTHONHASHSEED=123
cd /workspace/omniinfer/components/omni-proxy/omni_proxy/
bash omni_proxy.sh \
  --nginx-conf-file /usr/local/nginx/conf/nginx.conf \
  --start-core-index 0 \
  --core-num 4 \
  --listen-port "${PROXY_PORT}" \
  --prefill-endpoints "${PREFILL_SERVERS}" \
  --decode-endpoints "${DECODE_SERVERS}" \
  --omni-proxy-pd-policy sequential \
  --omni-proxy-model-path "${MODEL_PATH}"