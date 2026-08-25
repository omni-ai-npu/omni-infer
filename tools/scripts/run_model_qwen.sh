# Runs QwQ-32B model in default combined deployment mode 
# Uses standard execution (no graph optimization) with full NPU utilization
python run_model_qwen.py \
    --model-path <YOUR_MODEL_PATH> \
    --deploy-mode default \
    --graph-true 'false' \
    --model-name qwen \
    --server-list 0,1,2,3,4,5,6,7 \
    --network-interface <YOUR_NETWORK_INTERFACE> \
    -host-ip <YOUR_HOST_IP> \
    --https-port 8001

# Runs QwQ-32B model in default deployment with graph optimization
# Maximizes throughput using all NPUs (0-7) with accelerated execution
# python run_model_qwen.py \
#     --model-path <YOUR_MODEL_PATH> \
#     --deploy-mode default \
#     --graph-true 'true' \
#     --model-name qwen \
#     --server-list 0,1,2,3,4,5,6,7 \
#     --network-interface <YOUR_NETWORK_INTERFACE> \
#     -host-ip <YOUR_HOST_IP> \
#     --https-port 8001

# Runs QwQ-32B in split deployment mode (Prefill and Decoder on separate devices)
# Uses standard execution and custom service port
# python run_model_qwen.py \
#     --model-path <YOUR_MODEL_PATH> \
#     --deploy-mode pd_separate \
#     --graph-true 'false' \
#     --model-name qwen \
#     --network-interface <YOUR_NETWORK_INTERFACE> \
#     --prefill-server-list 0,1,2,3,4,5,6,7 \
#     --decode-server-list 8,9,10,11,12,13,14,15 \
#     -host-ip <YOUR_HOST_IP> \
#     --https-port 8001 \
#     --service-port 6660


# Runs QwQ-32B in split deployment mode (Prefill and Decoder on separate devices)
# Uses standard execution and custom service port
# python run_model_qwen.py \
#     --model-path <YOUR_MODEL_PATH> \
#     --deploy-mode pd_separate \
#     --graph-true 'true' \
#     --model-name qwen \
#     --network-interface <YOUR_NETWORK_INTERFACE> \
#     --prefill-server-list 0,1,2,3,4,5,6,7 \
#     --decode-server-list 8,9,10,11,12,13,14,15 \
#     -host-ip <YOUR_HOST_IP> \
#     --https-port 8001 \
#     --service-port 6660

