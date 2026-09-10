# PD分离快速部署

本文档介绍如何快速拉起PD分离部署推理，支持3机1P1D、4机2P1D、8机4P1D、16机8P1D和EP144 36机18P1D。

## 硬件要求

**硬件：** CloudMatrix384推理卡

**操作系统：** Linux

**镜像版本：** swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26

[**驱动检查**](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#ascend-npu%E5%9B%BA%E4%BB%B6%E5%92%8C%E9%A9%B1%E5%8A%A8%E6%A3%80%E6%9F%A5): `npu-smi info` 检查Ascend NPU固件和驱动是否正确安装。

**网络联通：** 使用[ssh命令](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E7%BD%91%E7%BB%9C%E8%BF%9E%E9%80%9A%E6%80%A7%E6%A3%80%E6%9F%A5)确认机器互连。

## 模型准备

准备与本示例匹配的 openPangu-2.0-Pro（PanguV2 505B BF16）模型权重。将完整
权重放置在所有 Prefill 和 Decode 节点均可访问的相同绝对路径；使用共享存储时，
需确认各节点均已正确挂载该目录。随后将 Playbook 中的 `vars.model_path` 修改为
该权重目录。

## 部署

### 镜像及源码准备

```bash
docker pull swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26
git clone https://gitee.com/omniai/omniinfer.git
git clone https://github.com/vllm-project/vllm.git omniinfer/infer_engines/vllm
```

### 使用ansible一键部署

详见 [Ansible 部署文档](../tools/deploy/ansible/README.md)。以下为快速部署示例。

#### 环境准备

安装 Ansible，参考
[安装指南](./omni_infer_installation_guide.md#环境准备-1)。

#### 修改配置文件

部署配置由 Inventory 和模型 Playbook 两部分组成。Inventory 模板位于
`omniinfer/tools/deploy/ansible/inventories/`，当前提供 1P1D、2P1D 和 4P1D
三种拓扑。模型 Playbook 位于 `omniinfer/tools/deploy/ansible/playbooks/`。
本节以 2P1D 部署 PanguV2 505B BF16 为例。

1. **omni_infer_inventory_used_for_2P1D.yml**

   建议先将
   `inventories/omni_infer_inventory_used_for_2P1D.yml` 复制到仓库外，再填写真实
   地址和凭据。将 `p0/p1/d0/d1/c0` 下的 `ansible_host` 与 `host_ip`
   改为对应 IP。<span style="color:red; font-weight:bold">对于多节点 D 场景，
   所有 D 节点的 `host_ip` 均为主节点 d0 的 IP。</span>


   ```YAML
   children:
     P:
       hosts:
         p0:
           ansible_host: "127.0.0.1"  # P0节点的IP
           ...
           host_ip: "127.0.0.1"  # P0节点的IP
           ...

         p1:
           ansible_host: "127.0.0.2"  # P1节点的IP
           ...
           host_ip: "127.0.0.2"  # P1节点的IP
           ...

     D:
       hosts:
         d0:
           ansible_host: "127.0.0.3"  # D0 节点的IP
           ...
           host_ip: "127.0.0.3"       # D0 节点的IP
           ...

         d1:
           ansible_host: "127.0.0.4"  # D1 节点的IP
           ...
           host_ip: "127.0.0.3"       # D0 节点的IP, 即 D 节点的主节点 IP
           ...

     C:
       hosts:
         c0:
           ansible_host: "127.0.0.1"  # C0 节点的 IP，即 Omni Proxy 节点
           ...

   ```

   生成私钥文件，参考
   [Ansible 部署文档](../tools/deploy/ansible/README.md#密钥文件的准备)。将
   `ansible_ssh_private_key_file` 修改为私钥路径：

   ```YAML
    all:
      vars:
        ...
        ansible_ssh_private_key_file: /path/to/key.pem  # 私钥文件路径
        ...
   ```

2. **omni_infer_server_template_performance2P1D_505B_bf16_open.yml**

   修改
   `playbooks/omni_infer_server_template_performance2P1D_505B_bf16_open.yml`
   中的环境路径、镜像、容器名称和模型路径：

   ```yaml
   environment:
     LOG_PATH: /data/log_path
     LOG_PATH_IN_EXECUTOR: /data/log_path_in_executor
     CODE_PATH: /data/local_code_path
     DOCKER_IMAGE_ID: "REPOSITORY:TAG"
     DOCKER_NAME_P: "you_name_omni_infer_prefill"
     DOCKER_NAME_D: "you_name_omni_infer_decode"
     DOCKER_NAME_C: "you_name_omni_infer_proxy"
     SCRIPTS_PATH: /tmp/scripts_path

   vars:
     model: pangu-v2-505b
     model_path: /data/models/Pangu-V2-505B
   ```

   同时检查 `run_docker_profile.extra_mounts` 是否覆盖模型和源码使用的宿主机
   路径。服务参数分别由 `run_server_prefill_profile`、
   `run_server_decode_profile` 和 `run_proxy_profile` 配置。详细字段说明见
   [Ansible 开发与部署指南](../tools/deploy/ansible/DEVELOPMENT_GUIDE.md)。

#### 执行命令

```bash
cd omniinfer/tools/deploy/ansible

INVENTORY=inventories/omni_infer_inventory_used_for_2P1D.yml
PLAYBOOK=playbooks/omni_infer_server_template_performance2P1D_505B_bf16_open.yml

ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --syntax-check
ansible-playbook -i "$INVENTORY" "$PLAYBOOK"
```

#### curl 测试

拉起成功后，可以通过curl命令进行测试：

```bash
curl -X POST http://127.0.0.1:7000/v1/completions -H "Content-Type:application/json" -d '{"model": "pangu_v2_moe","temperature":0,"max_tokens":50,"prompt": "how are you?", "stream":true,"stream_options": {"include_usage": true,"continuous_usage_stats": true}}'
```

该 Playbook 未覆盖 `--served-model-name`，因此使用启动脚本的默认服务名
`pangu_v2_moe`。如果在 Playbook 中显式修改该参数，curl 请求中的 `model` 也要
保持一致。

#### 注意事项

- `sync_code` 会将 `CODE_PATH` 下的源码同步到目标机并复制进容器。执行前请确认
  `CODE_PATH` 指向本次需要部署的工作区，且目标路径中没有需要单独保留的同名文件。

## 更高性能

可以使能以下特性来获得更高的性能：

**1. 使用图缓存**

首次启动服务时，模型会从头编译。建议首次成功启动后，重新执行以下命令以启用图缓存，提升性能：

```bash
cd omniinfer/tools/deploy/ansible
ansible-playbook \
  -i inventories/omni_infer_inventory_used_for_2P1D.yml \
  playbooks/omni_infer_server_template_performance2P1D_505B_bf16_open.yml \
  --tags run_server
```


**2. 调整 proxy batch size**

在 PanguV2 Playbook 的 `run_proxy_profile` 中调整
`--omni-proxy-prefill-max-num-seqs` 和
`--omni-proxy-decode-max-num-seqs`。修改后使用 `--tags run_proxy`
重新生成配置并拉起 Proxy；不要直接修改容器内的 nginx 配置，否则后续部署会覆盖
这些修改。



**3. 增加 batch size**

在 PanguV2 Playbook 的 Prefill/Decode profile 中调整 `--max-num-seqs`
（batch size）。如果同时将 `--num-speculative-tokens` 调整为大于 0 以启用 MTP，
还需要同步修改 OmniInfer 源码中对应模型配置的 `decode_gear_list`：

```JSON
"decode_gear_list": [batch_size * (1+num_speculative_tokens)]
```

以 MTP 1 为例，`--max-num-seqs` 设置为 32 时，
`decode_gear_list` 应包含 64。具体配置文件由所用模型决定。
