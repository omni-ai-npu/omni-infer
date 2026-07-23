# 部署环境说明

以Ascend910C (A3) 为例，openPangu-2.0-Flash的BF16权重版本可通过**1P1D**的配置拉起服务。可以使用一机A3组一个P节点，一机A3组一个D节点，这样的**1P1D**共使用两机A3。

多机部署通过 ansible-playbook 在执行机上统一拉起，执行机需安装 ansible（如 `yum install ansible`）。

## 拉取镜像

拉取机器对应镜像

```bash
docker pull swr.cn-east-4.myhuaweicloud.com/omni-ci/omniinfer-a3-arm:release_1.2.1.post1-202606292354-vllm
```

## 配置ssh

首次配对的P和D节点需要配置ssh，若使用的机器拉过PD分离则无需重新配置。在P节点执行下述命令

```bash
# -t 指定加密算法（推荐 ed25519，更安全且速度快；或使用 rsa）
# -N "" 表示不设置私钥密码（免密登录的关键）
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
# 依次替换为 Decode 节点的实际 IP
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@xxx.xxx.xx.xx
# 依次替换为 Prefill 节点的实际 IP
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@xxx.xxx.xx.xx
```

## 推理代码依赖的packages

适配推理代码的部分 packages 及版本（镜像内已预装）：

* `omni-npu`, Version `0.2.0`
* `vllm`, Version `0.14.0+empty`
* `tiktoken`, Version `0.13.0`
* `tokenizers`, Version `0.22.2`
* `torch`, Version `2.9.0`
* `torch-npu`, Version `2.9.0.post3.dev20260522`
* `transformers`, Version `4.57.6`
* `Python`, Version `3.11.12`

# ansible-playbook在多机上拉起推理服务

## 修改脚本

拉起 PD 分离服务的脚本在代码仓的 `tools/ansible/92B`和`tools/ansible/505B` 路径下。以 **1P1D** 为例，对应文件为：

* `omni_infer_inventory_used_for_1P1D.yml` — 节点 inventory
* `omni_infer_server_template_performance1P1D_92B_bf16_open.yml` — BF16 权重服务模板
* 在 **omni_infer_inventory_used_for_1P1D.yml** 中填写 **P 节点**、**D 节点** 和 **C（proxy）节点** 的机器 IP。**proxy 节点** 设为 P 节点 IP。注意 `ansible_host` 和 `host_ip` 都要修改为部署的 IP 地址。

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

> 上例为 **1P1D** 的单 P 节点写法。多 P 节点（如 **4P1D**）的 `P` 组采用分组结构（`P0`/`P1`/… 各含一个 host），并需相应设置各节点的 `kv_rank`、`host_ip` 等，可参考同目录下的 `omni_infer_inventory_used_for_4P1D.yml`。

* 在 **omni_infer_server_template_performance1P1D_92B_bf16_open.yml** 文件中写有P、D和C的服务拉起脚本和配置。
  首次拉起服务前需改动`environment`部分，将所有路径相关和容器名都换成自己的信息。

```
environment:
    # Global Configuration
    LOG_PATH: "/path/to/server/log/" 	# 必选：日志存储路径，全流程必须完整存在，不然无法跟踪服务
    MODEL_PATH: "/path/to/model/weights/" 	# 必选：本机权重路径，P和D所有节点需保持一致
    MODEL_LEN_MAX_PREFILL: "524288"
    MODEL_LEN_MAX_DECODE: "524288"
    LOG_PATH_IN_EXECUTOR: "/path/to/server/log_path_in_executor" # 可选：汇总日志时使用，将执行机的日志拉取到控制机
    KV_CONNECTOR: "LLMDataDistConnector"
    SERVED_MODEL_NAME: "openPangu-2.0-Flash" 	# 必选：对外服务名，发请求时 model 字段需与此一致

    # Configuration for containers
    DOCKER_IMAGE_ID: "image_name:image_tag" 	# PD分离docker使用的镜像，跟上文拉取到各个机器上的镜像保持一致
    DOCKER_NAME_P: "docker_p" 	# PD分离在P节点创建的容器名，需提前设置
    DOCKER_NAME_D: "docker_d" 	# PD分离在D节点创建的容器名，需提前设置
    DOCKER_NAME_C: "docker_c" 	# PD分离在proxy节点创建的容器名，需提前设置
    SCRIPTS_PATH: "/tmp/scripts_path"

    # Tensor Parallel Size
    DECODE_TENSOR_PARALLEL_SIZE: "1" # 当前脚本默认prefill TP部署，decode DP部署
```

其次**P节点**配置在`run_vllm_server_prefill_cmd:`，**D节点**配置在`run_vllm_server_decode_cmd:`，可使用默认配置，也可根据需求开关特性。

## 启动镜像

在P节点运行下述命令可启动镜像，在设置的每台服务器上创建docker。注意替换成本机上的对应文件名，以 **1P1D** 为例：

```bash
#92B
ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_bf16_open.yml --tags run_docker

#505B
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template_performance2P1D_505B_bf16_open.yml --tags run_docker
```

docker创建好后可跳转到 [推理服务拉起](#推理服务拉起) 章节拉起推理服务。

> **注意**：镜像没有变化时可复用已有 docker，无需重复运行此命令（重复运行会覆盖同名容器）。

## 推理代码适配

若需要修改推理代码，可通过下述命令查看`omni-npu`等组件在docker内的安装路径，并进入对应的docker内进行修改

```bash
# 查看omni-npu
pip list | grep omni-npu
```

## 推理服务拉起

docker在各个部署的A3机器上创建好后，在bash通过下述命令拉取推理服务，以 **1P1D** 为例：

```bash
#92B
ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_bf16_open.yml --tags run_server,run_proxy

#505B
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template_performance2P1D_505B_bf16_open.yml --tags run_server,run_proxy
```

C节点会在容器内启动nginx+proxy，在master node上启动nginx将并发的请求分配到各个节点上。可在部署的机器上通过日志追踪服务拉起的进程。

```bash
# 此处路径对应 environment 中配置的 LOG_PATH
tail -f /path/to/server/log/server_0.log
```

## 发请求测试

服务启动后，向proxy节点端口（脚本默认为7000）发送测试请求：

> **注意**：请求 body 中的 `model` 为 `openPangu-2.0-Flash`（与模板 `SERVED_MODEL_NAME` 一致，92B默认为 `openPangu-2.0-Flash`，505B默认为`openPangu-2.0-Pro`）。

```bash
# ${MASTER_NODE_IP} 替换为 inventory 中 C 节点的 ansible_host，端口对应 proxy_port（默认 7000）
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
