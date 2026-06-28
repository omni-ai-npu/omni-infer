# Deployment Environment

Taking Ascend910C (A3) as an example, both the BF16 and INT8 weight versions of openPangu-2.0-Flash can be served with a **1P1D** configuration. You can use one A3 machine to form a P node and one A3 machine to form a D node. This **1P1D** setup uses a total of two A3 machines.

## Pull Image

Pull the corresponding image for your machine.

```bash
docker pull image_name:image_tag
```

Example

```bash
docker pull registry-cbu.huawei.com/omniai_omniinfer/ai-infra-infer-1.2.1-arm-openeuler-py311-a3-cann.9.1.t2.b020-pta_v2.9.0-op-1.2.1-omni_release_1.2.1-vllm-202606051040:0.0.1
```

## Configure SSH

SSH must be configured between P and D nodes that are paired for the first time. If the machines have already been used for PD disaggregation, no re-configuration is needed. Run the following commands on the P node:

```bash
# -t specifies the encryption algorithm (ed25519 recommended for better security and speed; or use rsa)
# -N "" means no passphrase for the private key (required for passwordless login)
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
# Replace with the actual IP of each Decode node
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@xxx.xxx.xx.xx
# Replace with the actual IP of each Prefill node
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@xxx.xxx.xx.xx
```

## Inference Code Dependencies

Packages and versions required by the inference code:

* `omni-npu`, Version `0.2.0`
* `omni-models`, Version `0.1.0`
* `vllm`, Version `0.14.0+empty`
* `tiktoken`, Version `0.13.0`
* `tokenizers`, Version `0.22.2`
* `torch`, Version `2.9.0`
* `torch-npu`, Version `2.9.0.post3.dev20260522`
* `transformers`, Version `4.57.6`
* `Python`, Version `3.11.12`

# Launching Inference Service on Multiple Machines with ansible-playbook

## Modify Scripts

The scripts for launching the PD disaggregation service are located at ``omniinfer/tools/ansible/template``.

* In the **omni_infer_inventory_used_for_xPyD.yml** file, fill in the IP addresses of the **P node** and **D node** machines. Set the **proxy node** to the P node IP. The default ports can be used. Note that both `ansible_host` and `host_ip` must be changed to the deployment IP addresses.

```
P:
      children:
        P0:
          hosts:
            p0:
              ansible_host: "127.0.0.1" # Replace with the local deployment IP address
              node_rank: 0
              kv_rank: 0
              node_port: "{{ global_port_base + port_offset.P + kv_rank * 10 }}"
              api_port: "{{ base_api_port + port_offset.P + kv_rank *10 + node_rank }}"
              host_ip: "127.0.0.1" # Replace with the local deployment IP address
              ascend_rt_visible_devices: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
```

* The **omni_infer_server_template_performancexPyD_92B_xxx.yml** file contains the service launch scripts and configurations for P, D, and C nodes.
  Before launching the service for the first time, modify the `environment` section and replace all path-related fields and container names with your own information.

```
environment:
    # Global Configuration
    LOG_PATH: "/path/to/server/log/" 	# Log storage path, must be set and persist throughout the entire process, otherwise the service cannot be tracked
    MODEL_PATH: "/path/to/model/weights/" 	# Local model weights path, must be consistent across all P and D nodes
    MODEL_LEN_MAX_PREFILL: "524288"
    MODEL_LEN_MAX_DECODE: "524288"
    LOG_PATH_IN_EXECUTOR: "/path/to/server/log_path_in_executor"
    CODE_PATH: "/path/to/model/codes/" # Local code path
    KV_CONNECTOR: "LLMDataDistConnector"

    # Configuration for containers
    DOCKER_IMAGE_ID: "image_name:image_tag" 	# Image used by PD disaggregation docker, must match the image pulled on each machine above
    DOCKER_NAME_P: "docker_name_p" 	# Container name created by PD disaggregation on the P node, must be set in advance
    DOCKER_NAME_D: "docker_name_d" 	# Container name created by PD disaggregation on the D node, must be set in advance
    DOCKER_NAME_C: "docker_name_c" 	# Container name created by PD disaggregation on the proxy node, must be set in advance
    SCRIPTS_PATH: "/tmp/scripts_path"

    # Configuration for lb_sdk in global proxy
    PREFILL_LB_SDK: "pd_score_balance"
    DECODE_LB_SDK: "pd_score_balance"
    USE_OMNI_PROXY: "1"

    # Tensor Parallel Size
    DECODE_TENSOR_PARALLEL_SIZE: "1" # The current script defaults to prefill TP deployment and decode DP deployment
```

Additionally, the **P node** configuration is under `run_vllm_server_prefill_cmd:`, and the **D node** configuration is under `run_vllm_server_decode_cmd:`. You can use the default configuration or enable/disable features as needed.

## Start Image

Run the following command on the P node to start the image and create a docker on each configured server. Replace the file name with the corresponding one on your machine.

```bash
ansible-playbook -i omni_infer_inventory_used_for_xPyD.yml omni_infer_server_template_performancexPyD_92B_xxx.yml --tags run_docker
```

Note: Once the docker environment is configured, it can be reused. Do not run this command again, as re-running it will overwrite the docker with the same name.

## Inference Code Adaptation

You can directly use the code bundled with the image. Run the following commands to check the installation paths of `omni-npu` and `omni-models` inside the docker. Once the docker is created, you can launch the inference service directly via `ansible-playbook`.

```bash
pip list | grep omni-npu
pip list | grep omni-models
```

Alternatively, you can pull the latest code from the repository to replace the built-in `omni-npu` and `omni-models` in the image.

### Replace Built-in Code in Image

You can enter the created docker with the following command.

```bash
docker exec -itu root docker_name /bin/bash
```

Run the following commands inside the docker to install the local inference code.

```
# Install omni-npu
cd /path/to/your/local/omni-npu
pip install -e . -v

# Install omni-models
cd /path/to/your/local/omni-models
pip install -e . -v
```

If the above commands fail to install the local code, try the following command in the code directory.

```
pip install -e . -v --index-url https://mirrors.tools.huawei.com/pypi/simple --trusted-host mirrors.tools.huawei.com --no-build-isolation
```

### INT8 Quantization

The openPangu-2.0 model supports generating W8A8 INT8 quantized weights using the Joint SmoothQuant method. This method jointly searches for the optimal $(a,b)$ smoothing parameters for each linear layer, employing Hessian channel weighting with K=2 coordinate descent iteration, and a mixed quantization strategy of write-side GPTQ and read-side RTN, compressing the model size by approximately 1.9× while maintaining inference accuracy. The quantization tool is located at `tools/quantize_joint_smooth_int8_iterative.py`.

#### Quantization Command (16-Card Ascend NPU)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export LOG=logs/joint_smooth_w8a8_$(date +%m%d_%H%M%S).log

time python3 -u tools/quantize_joint_smooth_int8_iterative.py \
    --model {path_to_float_weights} \
    --output {path_to_W8A8_quantized_weights} \
    --calib-data lm_eval/tasks/wikitext_local/train-00000-of-00001.parquet \
    --n-samples 32 \
    --seq-len 1024 \
    --num-iterations 2 \
    --iter-ab-tol 0.05 \
    --num-devices 16 \
    --device npu \
    --objective output-recon \
    --write-quant gptq \
    --skip-shared-experts \
    --build-meta \
    2>&1 | tee ${LOG}
```

**Key Parameter Descriptions:**

| Parameter | Recommended Value | Description |
|-----------|-------------------|-------------|
| `--n-samples` | 32 | Number of calibration samples |
| `--seq-len` | 1024 | Calibration sequence length |
| `--num-iterations` | 2 | Number of coordinate descent iterations (K=2), alternating optimization between gate/up ↔ down smoothing parameters |
| `--iter-ab-tol` | 0.05 | Iteration convergence threshold; stops early when the change in (a,b) between consecutive iterations is below 5% |
| `--objective` | output-recon | Search objective: minimize layer output reconstruction error (more accurate than weight error proxy) |
| `--write-quant` | gptq | Residual stream direction (o_proj / down_proj) weights use GPTQ quantization to reduce error |
| `--skip-shared-experts` | — | Keep shared expert as BF16; this path is traversed by every token, and quantization error accumulates globally |
| `--build-meta` | — | Use meta device to initialize module skeletons, avoiding BF16 full weights occupying NPU memory during quantization |

#### config.json After Quantization

After quantization is complete, the script automatically writes a `quantization_config` field into the `config.json` in the output directory:

```json
"quantization_config": {
    "quant_method": "compressed-tensors",
    "quantize": "w8a8_dynamic",
    "format": "int-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "type": "int", "num_bits": 8, "symmetric": true,
                "strategy": "channel", "dynamic": false
            },
            "input_activations": {
                "type": "int", "num_bits": 8, "symmetric": true,
                "strategy": "token", "dynamic": true
            }
        }
    },
    "ignore": ["<list of layers skipped by quantization, auto-generated by the script>"]
}
```

Weights use per-output-channel static quantization, and activations use per-token dynamic quantization. vLLM automatically recognizes the compressed-tensors format from `quantization_config`, so no additional `--quantization` parameter is needed at inference time.

### Using c8 Quantization

If using c8 quantization, i.e., configuring `--kv-cache-dtype int8_ds_mla` or `--kv-cache-dtype li_int8_ds_mla` in the `EXTRA_ARGS` of `run_vllm_server_prefill_cmd:` or `run_vllm_server_decode_cmd:`, you also need to modify the built-in vLLM in the image.
Run `vim /opt/vllm/vllm/config/cache.py` inside the docker, and add `int8_ds_mla` and `li_int8_ds_mla` to ``CacheDType``.

```python
CacheDType = Literal[
    "auto",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
	"int8_ds_mla",
	"li_int8_ds_mla",
]
```

Additionally, run `vim /opt/vllm/vllm/utils/torch_utils.py` inside the docker, and add `"int8_ds_mla":torch.int8` and `"li_int8_ds_mla":torch.bfloat16` to ``STR_DTYPE_TO_TORCH_DTYPE``.

```python
STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "fp8_inc": torch.float8_e4m3fn,
    "fp8_ds_mla": torch.uint8,
	"int8_ds_mla":torch.int8,
	"li_int8_ds_mla":torch.bfloat16,
}
```

## Launch Inference Service

Once dockers are created on each deployed A3 machine, launch the inference service with the following command in bash.

```bash
ansible-playbook -i omni_infer_inventory_used_for_xPyD.yml omni_infer_server_template_performancexPyD_92B_xxx.yml --tags run_server,run_proxy
```

The C node will start nginx+proxy inside the container, and start nginx on the master node to distribute concurrent requests across nodes. You can track the service launch progress through logs on the deployed machine.

```bash
tail -f /path/to/server/log/server_0.log
```

## Send Test Request

After the service is started, send a test request to the master node's proxy port (default is 7000) from the master node or any other node:

```bash
curl -X POST http://${MASTER_NODE_IP}:7000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openPangu-2.0-Flash",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "vllm_xargs": {"top_n_sigma": 0.05},
		"stream": false
    }'
```