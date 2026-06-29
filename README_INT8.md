# 部署环境说明

以Ascend910C (A3) 为例，openPangu-2.0-Flash的INT8权重版本可通过**1P1D**的配置拉起服务。可以使用一机A3组一个P节点，一机A3组一个D节点，这样的**1P1D**共使用两机A3。

## 拉取镜像

拉取机器对应镜像

```bash
docker pull image_name:image_tag
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

拉起 PD 分离服务的脚本在 `omniinfer/tools/ansible/template` 路径下。以 **1P1D** 为例，对应文件为：

* `omni_infer_inventory_used_for_1P1D.yml` — 节点 inventory
* `omni_infer_server_template_performance1P1D_92B_open.yml` — INT8 权重服务模板

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

* 在 **omni_infer_server_template_performance1P1D_92B_open.yml** 文件中写有P、D和C的服务拉起脚本和配置。
  首次拉起服务前需改动`environment`部分，将所有路径相关和容器名都换成自己的信息。

```
environment:
    # Global Configuration
    LOG_PATH: "/path/to/server/log/" 	# 必选：日志存储路径，全流程必须完整存在，不然无法跟踪服务
    MODEL_PATH: "/path/to/model/weights/" 	# 必选：本机权重路径，P和D所有节点需保持一致
    MODEL_LEN_MAX_PREFILL: "524288"
    MODEL_LEN_MAX_DECODE: "524288"
    LOG_PATH_IN_EXECUTOR: "/path/to/server/log_path_in_executor" # 可选：汇总日志时使用，将执行机的日志拉取到控制机
    CODE_PATH: "/path/dir/" # 可选：同步代码时使用，其下需有 omniinfer 子目录（如设为 /data/pangu-v2，则代码位于 /data/pangu-v2/omniinfer）
    KV_CONNECTOR: "LLMDataDistConnector"

    # Configuration for containers
    DOCKER_IMAGE_ID: "image_name:image_tag" 	# PD分离docker使用的镜像，跟上文拉取到各个机器上的镜像保持一致
    DOCKER_NAME_P: "docker_name_p" 	# PD分离在P节点创建的容器名，需提前设置
    DOCKER_NAME_D: "docker_name_d" 	# PD分离在D节点创建的容器名，需提前设置
    DOCKER_NAME_C: "docker_name_c" 	# PD分离在proxy节点创建的容器名，需提前设置
    SCRIPTS_PATH: "/tmp/scripts_path"

    # Tensor Parallel Size
    DECODE_TENSOR_PARALLEL_SIZE: "1" # 当前脚本默认prefill TP部署，decode DP部署
```

其次**P节点**配置在`run_vllm_server_prefill_cmd:`，**D节点**配置在`run_vllm_server_decode_cmd:`，可使用默认配置，也可根据需求开关特性。

## 启动镜像

在P节点运行下述命令可启动镜像，在设置的每台服务器上创建docker。注意替换成本机上的对应文件名，以 **1P1D** 为例：

```bash
ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_open.yml --tags run_docker
```

docker创建好后可跳转到 [推理服务拉起](#推理服务拉起) 章节拉起推理服务。

> **注意**：docker内环境配置好了可以复用docker，不要再运行此命令，因为再次运行会把同名docker覆盖掉。

## 推理代码适配

若需要修改推理代码，有以下两种方式：
### 1. 容器内直接修改
通过下述命令查看`omni-npu`等组件在docker内的安装路径，并进入对应的docker内进行修改

```bash
# 查看omni-npu
pip list | grep omni-npu
```


### 2. 同步本地代码到容器（sync_code）

如果想用本地修改过的代码覆盖镜像自带代码，可使用 `sync_code` 一键同步，无需手动进容器安装。

1. 在 `environment` 中设置 `CODE_PATH`，其下需有 `omniinfer` 子目录（如 `CODE_PATH=/data/pangu-v2`，则源码位于 `/data/pangu-v2/omniinfer`）。
2. 在执行机执行，以 **1P1D** 为例：

```bash
ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_open.yml --tags sync_code
```

该命令会先把执行机 `$CODE_PATH/omniinfer` 同步到各 P/D/C 机器，再叠加拷贝进容器的 `/workspace/omniinfer`。

> 说明：拷贝为叠加覆盖（只覆盖同名文件，不删除容器中本地没有的文件），因此会保留容器内已编译的产物（如 `omni-eplb`、`omni-cache` 的 `.so` 等）。如果你在本地删除或重命名了文件，容器内的旧文件不会被自动清除。

## INT8量化

量化方法的安装和部署见：[jointfix README](https://gitcode.com/ascend-tribe/openPangu-2.0-Infer/blob/pangu-v2-test/tools/quant/jointfix/README.md)。

## 推理服务拉起

docker在各个部署的A3机器上创建好后，在bash通过下述命令拉取推理服务，以 **1P1D** 为例：

```bash
ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_open.yml --tags run_server,run_proxy
```

C节点会在容器内启动nginx+proxy，在master node上启动nginx将并发的请求分配到各个节点上。可在部署的机器上通过日志追踪服务拉起的进程。

```bash
# 此处路径对应 environment 中配置的 LOG_PATH
tail -f /path/to/server/log/server_0.log
```

## 发请求测试

服务启动后，向proxy节点端口（脚本默认为7000）发送测试请求：

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
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "vllm_xargs": {"top_n_sigma": 0.05},
		"stream": false
    }'
```

## 开启omni-cache 特性

在 Playbook 中将对应server yml 替换即可

92B: omni_infer_server_template_performance4P1D_92B_w8a8_open_omni_cache.yml 推荐inventory 形态: 4P1D

```bash
ansible-playbook -i omni_infer_inventory_used_for_4P1D.yml omni_infer_server_template_performance4P1D_92B_w8a8_open_omni_cache.yml --tags run_docker,run_server,run_proxy
```

### 从OmniCache服务切换到其他配置前的处理

> **注意：** 如果当前容器运行过OmniCache版本的服务，之后需要使用同一容器运行其他配置的服务，必须先释放OmniCache占用和预留的大页内存，否则可能导致后续服务可用内存不足或启动失败。

请在所有运行过OmniCache服务的相关容器上依次执行以下操作：

1. 重启容器，释放OmniCache服务占用的大页内存。
2. 容器重启后，在代码根目录执行以下命令，将大页内存分配上限恢复为默认值并释放多余的预留内存：

```bash
bash omni-cache/tools/setup/set_hugepage_limit.sh --target-pages 262144
```