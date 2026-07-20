# Deployment Environment

Taking Ascend910C (A3) as an example, the INT8 weight version of openPangu-2.0-Flash can be served with a **1P1D** configuration. You can use one A3 machine to form a P node and one A3 machine to form a D node. This **1P1D** setup uses a total of two A3 machines.

Multi-machine deployment is launched uniformly from the executor machine via ansible-playbook, so ansible must be installed on the executor machine (e.g. `yum install ansible`).

## Pull Image

Pull the corresponding image for your machine.

```bash
A3: docker pull swr.cn-east-4.myhuaweicloud.com/omni-ci/omniinfer-a3-arm:release_1.2.1.post1-202606292354-vllm
A2: docker pull swr.cn-east-4.myhuaweicloud.com/omni-ci/omniinfer-a2-arm:release_1.2.1.post1-202607010954-vllm
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

Packages and versions required by the inference code (pre-installed in the image):

* `omni-npu`, Version `0.2.0`
* `vllm`, Version `0.14.0+empty`
* `tiktoken`, Version `0.13.0`
* `tokenizers`, Version `0.22.2`
* `torch`, Version `2.9.0`
* `torch-npu`, Version `2.9.0.post3.dev20260522`
* `transformers`, Version `4.57.6`
* `Python`, Version `3.11.12`

# Launching Inference Service on Multiple Machines with ansible-playbook

## Modify Scripts

The scripts for launching the PD disaggregation service are located at `tools/ansible/template` in the repository. For **1P1D**, the corresponding files are:

* `omni_infer_inventory_used_for_1P1D.yml` — node inventory
* `omni_infer_server_template_performance1P1D_92B_w8a8_open.yml` — INT8 weight service template

* In **omni_infer_inventory_used_for_1P1D.yml**, fill in the IP addresses of the **P node**, **D node**, and **C (proxy) node** machines. Set the **proxy node** to the P node IP. Note that both `ansible_host` and `host_ip` must be changed to the deployment IP addresses.

```yaml
  children:
    P:
      hosts:
        p0:
          ansible_host: "127.0.0.1"
          node_rank: 0
          kv_rank: 0
          node_port: "{{ global_port_base + port_offset.P + kv_rank }}"
          api_port: "{{ base_api_port + port_offset.P + kv_rank }}"
          host_ip: "127.0.0.1"
          ascend_rt_visible_devices: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
```

> The example above shows the single-P-node layout for **1P1D**. For multiple P nodes (e.g. **4P1D**), the `P` group uses a grouped layout (`P0`/`P1`/… each containing one host), with `kv_rank`, `host_ip`, etc. set per node. Refer to `omni_infer_inventory_used_for_4P1D.yml` in the same directory.

* The **omni_infer_server_template_performance1P1D_92B_w8a8_open.yml** file contains the service launch scripts and configurations for P, D, and C nodes.
  Before launching the service for the first time, modify the `environment` section and replace all path-related fields and container names with your own information.

```
environment:
    # Global Configuration
    LOG_PATH: "/path/to/server/log/" 	# Required: log storage path, must persist throughout the entire process, otherwise the service cannot be tracked
    MODEL_PATH: "/path/to/model/weights/" 	# Required: local model weights path, must be consistent across all P and D nodes
    MODEL_LEN_MAX_PREFILL: "524288"
    MODEL_LEN_MAX_DECODE: "524288"
    LOG_PATH_IN_EXECUTOR: "/path/to/server/log_path_in_executor" # Optional: used when aggregating logs, pulls logs from the executor to the control machine
    KV_CONNECTOR: "LLMDataDistConnector"

    # Configuration for containers
    DOCKER_IMAGE_ID: "image_name:image_tag" 	# Image used by PD disaggregation docker, must match the image pulled on each machine above
    DOCKER_NAME_P: "docker_name_p" 	# Container name created by PD disaggregation on the P node, must be set in advance
    DOCKER_NAME_D: "docker_name_d" 	# Container name created by PD disaggregation on the D node, must be set in advance
    DOCKER_NAME_C: "docker_name_c" 	# Container name created by PD disaggregation on the proxy node, must be set in advance
    SCRIPTS_PATH: "/tmp/scripts_path"

    # Tensor Parallel Size
    DECODE_TENSOR_PARALLEL_SIZE: "1" # The current script defaults to prefill TP deployment and decode DP deployment
```

Additionally, the **P node** configuration is under `run_vllm_server_prefill_cmd:`, and the **D node** configuration is under `run_vllm_server_decode_cmd:`. You can use the default configuration or enable/disable features as needed.

## Start Image

Run the following command on the P node to start the image and create a docker on each configured server. Replace the file name with the corresponding one on your machine. Taking **1P1D** as an example:

```bash
ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_w8a8_open.yml --tags run_docker
```

Once the docker is created, you can jump to the [Launch Inference Service](#launch-inference-service) section to launch the inference service.

> **Note**: If the image is unchanged, reuse the existing docker; there is no need to run this command again (re-running it overwrites the container with the same name).

## Inference Code Adaptation

If you need to modify the inference code, run the following command to check the installation path of `omni-npu` and other components inside the docker, then enter the corresponding docker to make changes.

```bash
# Check omni-npu
pip list | grep omni-npu
```

## INT8 Quantization

For the installation and deployment of the quantization methods, see:[jointfix README](https://gitcode.com/ascend-tribe/openPangu-2.0-Infer/blob/pangu-v2-test/tools/quant/jointfix/README.md)。

## Launch Inference Service

Once dockers are created on each deployed A3 machine, launch the inference service with the following command in bash, taking **1P1D** as an example:

```bash
ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_w8a8_open.yml --tags run_server,run_proxy
```

The C node will start nginx+proxy inside the container, and start nginx on the master node to distribute concurrent requests across nodes. You can track the service launch progress through logs on the deployed machine.

```bash
# This path corresponds to the LOG_PATH configured in environment
tail -f /path/to/server/log/server_0.log
```

## Send Test Request

After the service is started, send a test request to the proxy node port (default is 7000):

> **Note**: The `model` field in the request body is `openPangu-2.0-Flash`.

```bash
# Replace ${MASTER_NODE_IP} with the ansible_host of the C node in the inventory; the port corresponds to proxy_port (default 7000)
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
        "temperature": 1,
        "top_p": 1.0,
        "top_k": -1,
		"stream": false
    }'
```

## Enable omni-cache Feature

Simply replace the corresponding server yml in the Playbook:

92B: omni_infer_server_template_performance4P1D_92B_open_omni_cache.yml, recommended inventory configuration: 4P1D

```bash
A3: ansible-playbook -i omni_infer_inventory_used_for_4P1D.yml omni_infer_server_template_performance4P1D_92B_open_omni_cache.yml --tags run_docker,run_server,run_proxy
A2: ansible-playbook -i omni_infer_inventory_used_for_1P1D_A2.yml omni_infer_server_template_performance1P1D_92B_A2_w8a8_open.yml --tags run_server,run_proxy
```

### Steps Before Switching from OmniCache Service to Other Configurations

> **Note:** If the current container has previously run an OmniCache version of the service and you need to use the same container to run a different configuration, you must first release the huge page memory occupied and reserved by OmniCache. Otherwise, subsequent services may have insufficient available memory or fail to start.

Perform the following operations on all relevant containers that have run OmniCache services:

1. Restart the container to release the huge page memory occupied by the OmniCache service.
2. After the container restarts, run the following command in the code root directory to restore the huge page memory allocation limit to its default value and release excess reserved memory:

```bash
bash omni-cache/tools/setup/set_hugepage_limit.sh --target-pages 262144