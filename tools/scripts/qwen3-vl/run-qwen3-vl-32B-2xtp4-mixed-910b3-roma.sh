#!/bin/bash
# One 8-card A2 node -> 2 x TP4 vLLM -> 1 proxy port

# PATH
MOUNT_PATH="${MOUNT_PATH:-/data/w84156052/qwen}"
BUCKET_PATH="${BUCKET_PATH:-master-0210}"
BASE_LOG_PATH="${BASE_LOG_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/logs/qwen-32b}"
MODEL_PATH="${MODEL_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/models/qwen-32b/Qwen3-VL-32B-w8a8c16-visualblock-bf16}"

# Only expose PORT outside.
PORT="${PORT:-8000}"

BASE_API_PORT="${BASE_API_PORT:-8100}"
PORT_OFFSET="${PORT_OFFSET:-1}"
PROXY_PATH="${PROXY_PATH:-/workspace/omniinfer/components/omni-proxy/omni_proxy}"
PROXY_BACKEND_HOST="${PROXY_BACKEND_HOST:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1800}"  # wait for vLLM to start in 30 minutes

MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
COMPILATION_CONFIG='{"level": 3, "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64,156,256,512], "backend":"eager", "compile_sizes":[64,156,256,512]}'

source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GLOO_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_BUFFSIZE=256
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_DISABLE_COMPILE_CACHE=1
export OMNI_NPU_VLLM_PATCHES="ALL"
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_LOGGING_LEVEL=INFO
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "${BASE_LOG_PATH}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${BASE_LOG_PATH}/${POD_IP:-$(hostname)}_${RUN_ID}"
mkdir -p "${LOG_DIR}"
RUN_SCRIPT_LOG="${LOG_DIR}/run_script.log"
exec > >(tee -a "${RUN_SCRIPT_LOG}") 2>&1

DEVICE_GROUPS=("0,1,2,3" "4,5,6,7")
INSTANCE_COUNT="${#DEVICE_GROUPS[@]}"
PIDS=()
VLLM_PIDS=()
ENDPOINTS=()

cleanup() {
  trap - INT TERM
  kill "${PIDS[@]}" >/dev/null 2>&1 || true
  wait >/dev/null 2>&1 || true
}
trap cleanup INT TERM

wait_health() {
  local name="$1"
  local host="$2"
  local port="$3"
  local timeout="$4"

  for ((t=0; t<timeout; t++)); do
    if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
      echo "[ok] ${name}: ${host}:${port}"
      return 0
    fi
    sleep 1
  done

  echo "[failed] ${name}: ${host}:${port}"
  return 1
}

for i in "${!DEVICE_GROUPS[@]}"; do
  api_port=$((BASE_API_PORT + i * PORT_OFFSET))
  ENDPOINTS+=("${PROXY_BACKEND_HOST}:${api_port}")

  (
    export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_GROUPS[$i]}"
    exec env VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models" \
      vllm serve "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${api_port}" \
        --dtype bfloat16 \
        --max-model-len 163840 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --distributed-executor-backend mp \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --data-parallel-size 1 \
        --swap-space 64 \
        --allowed-local-media-path "${MOUNT_PATH}/${BUCKET_PATH}/" \
        --media-io-kwargs '{"video":{"fps":2,"num_frames":-1}}' \
        --compilation-config "${COMPILATION_CONFIG}"
  ) >> "${LOG_DIR}/vllm_${i}.log" 2>&1 &

  PIDS+=("$!")
  VLLM_PIDS+=("$!")
done

IFS=,
DECODE_ENDPOINTS="${ENDPOINTS[*]}"
unset IFS

echo "=======================Service Starting======================="
echo "omni proxy port: ${PORT}"
echo "vllm instances endpoints: ${DECODE_ENDPOINTS}"
echo "log dir: ${LOG_DIR}"
echo "vllm logs:"
for i in "${!DEVICE_GROUPS[@]}"; do
  echo "  vllm instance ${i}: ${LOG_DIR}/vllm_${i}.log"
done
echo "  proxy: ${LOG_DIR}/proxy.log"

echo "waiting for vllm health..."
for i in "${!DEVICE_GROUPS[@]}"; do
  api_port=$((BASE_API_PORT + i * PORT_OFFSET))
  wait_health "vllm instance ${i}" "${PROXY_BACKEND_HOST}" "${api_port}" "${STARTUP_TIMEOUT_SECONDS}"
done

(
  cd "${PROXY_PATH}"
  exec sudo env PYTHONHASHSEED=123 bash -l omni_proxy.sh \
    --nginx-conf-file /usr/local/nginx/conf/nginx.conf \
    --start-core-index 0 \
    --core-num "${INSTANCE_COUNT}" \
    --omni-proxy-decode-max-num-seqs "${MAX_NUM_SEQS}" \
    --listen-port "${PORT}" \
    --decode-endpoints "${DECODE_ENDPOINTS}" \
    --omni-proxy-pd-policy aggregation \
    --omni-proxy-model-path "${MODEL_PATH}"
) >> "${LOG_DIR}/proxy.log" 2>&1 &

PROXY_PID="$!"
PIDS+=("${PROXY_PID}")

echo "all services are ready"

# Monitor vLLM worker processes; comment out this block if you do not need in-script process watching.
WARNED=()
while true; do
  alive_count=0

  for i in "${!VLLM_PIDS[@]}"; do
    if kill -0 "${VLLM_PIDS[$i]}" >/dev/null 2>&1; then
      alive_count=$((alive_count + 1))
    elif [[ -z "${WARNED[$i]:-}" ]]; then
      echo "[warning] vllm instance ${i} exited, check ${LOG_DIR}/vllm_${i}.log"
      WARNED[$i]=1
    fi
  done

  if [[ "${alive_count}" -eq 0 ]]; then
    echo "[fatal] all vllm instances exited, stopping proxy"
    cleanup
    exit 1
  fi

  sleep 30
done
