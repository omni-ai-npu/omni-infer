# 环境要求

**软件系统**
- ​**​操作系统​​：** Linux
- ​**​镜像版本​​：** swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26

**关键检查项**
- **​​驱动与固件​：** 执行 npu-smi info确认昇腾 NPU 固件及驱动已正确安装（参考[官方文档](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#ascend-npu%E5%9B%BA%E4%BB%B6%E5%92%8C%E9%A9%B1%E5%8A%A8%E6%A3%80%E6%9F%A5)）
- **​​网络连通性​​：** 多机部署时需通过 SSH 免密互通检查确保节点间通信正常（单机部署可忽略）


# 单机混部快速部署（Hugging Face 模型）
## 软件准备
​**​模型名称**​​：指定 Hugging Face 格式的模型（如 Qwen/Qwen2.5-0.5B），容器启动时自动在线下载  
**代理配置​​（可选）**：若模型库需代理访问，设置 HUGGING_FACE_PROXY
```bash
MODEL_NAME="Qwen/Qwen2.5-0.5B"  # 替换为您的模型
HUGGING_FACE_PROXY="http://username:passward@hostIP:port/"  # 按需填写
```
可通过 `curl -v https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B` 判断是否需要设置网络代理
- 如果返回类似 JSON 的模型信息，则说明网络是通的，无需设置代理
- 如果提示连接失败、超时、或者无法解析域名，就是网络层的问题，需要设置代理

## 启动容器
拉取镜像并运行容器，自动启动 API 服务（默认端口 8301，可通过 -e PORT修改）
```bash
IMAGE_NAME="swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26"

# 拉取镜像
docker pull "$IMAGE_NAME"

# 启动容器（关键参数说明见下方注释）
docker run --rm -itd --shm-size=500g \
    --net=host --privileged=true \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -e PORT=8301 \                            # API 服务端口
    -e ASCEND_RT_VISIBLE_DEVICES=1 \          # 昇腾设备绑定
    -e HTTP_PROXY="${HUGGING_FACE_PROXY}" \   # 代理（若需）
    ${IMAGE_NAME} \
    --model "${MODEL_NAME}"                   # 指定模型
```

## 测试调用
通过 curl 发送请求测试模型服务（替换 8301为实际端口）
```bash
curl -X POST http://127.0.0.1:8301/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "temperature": 0,
        "max_tokens": 50,
        "prompt": "how are you?"
      }'
```

# PD 分离部署（支持多机多卡推理）

本节介绍如何快速拉起PD分离部署推理，支持3机1P1D、4机2P1D、8机4P1D、16机8P1D和EP144 36机18P1D。

## 模型准备

基于开源模型 ​​DeepSeek-V3/R1​​，需提前完成[权重转换](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2)。当前提供已转换好的权重文件，可直接下载：[权重下载链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?v2token=KBL+tPW8sihb1DQcY03GYZZdWrdKg8E2xUI8XrCsr7jo72H22pg6bY8V89ZgmD4Zq4VEcQa802+q2nR4Bydrzm9jjAO1ohYLIIDMeRtttFZR+EpCA2PWmZaVxazPlkJ6qtADFZaESGpEHUxChlLlFQ2xeLp6sXP5qVsj6JEPRh7MA6SmfqK8mLdgi/rmBjY6A0CRJFEe1K5JrgONubynmJaescenf5t0h36szT23dHV46pjw0BCjCFtxJyXqgGWc4T7pv3tugR09oHNLFaVoPi4ZlElNciul9a90kZ6ZOoNJ3ufoRyHA9bTdwqeJGg8jsBFzRr+d+tU1GXd8UaswFHUo805A3MoPlqSRiYThAz+3aPorLkveex99xiEwCq+pajn6S9GzSeY8FLjEMlopGMKfHJ6Z1B5aoGpIBY8UsjG878ixsE/YiZmetkXDO/FPYr/r9sbHqg5pLVXCmaH7uHqVvDabx6Mx0a8GRITN+yjwg08LjS3C76gwCfEqD7FegGitodr7RLSDsuewjGqjgK/7ST2J320CoBBBw5vtAAsDYiDC6LJOEZCN2ht/eAZUHvy7ZrCeBMN1AmagvqsXVJrsn2tof/CL5LpRm7z5eVoFAhLjpyKIOgWjUksthT0MLmqIZCiMoslj9BfCKv780nEeDQZqO/eerh7zT4qojk8Xaxuj9Xvv1fKtJnId608QPtLXDireSzh6aa4tF1b5W747AhNAPzLoNdOzcLheYyw=_Vsd2i2lmljqrcbVJHDC8TZw7tQFHpoZ6ZS0O3b864QM=_DuHhemY++UqCJXeQyjgwdA==)（提取码为12345678）。

## 部署流程

### 镜像及源码准备

```bash
docker pull swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26
git clone https://gitee.com/omniai/omniinfer.git
git clone https://github.com/vllm-project/vllm.git omniinfer/infer_engines/vllm
```

### 一键部署（Ansible 自动化）

详见 [Ansible 部署文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/README.md)，以下为快速示例（以 2P1D 架构为例）。

#### 环境准备

安装 Ansible 并配置 SSH 免密登录（参考 [Ansible 环境文档](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87-1) )。

#### 修改配置文件

需调整以下两个配置文件（位于 omniinfer/tools/ansible/template/目录）:

1. **omni_infer_inventory_used_for_2P1D.yml**  
   修改 P（预填充节点）、D（解码节点）、C（全局代理节点）下的 ansible_host和 host_ip为实际 IP 地址  
   **注意​​：** 多机组 D 节点（解码节点）的 host_ip需统一设置为主节点 d0的 IP（非各自 IP）  

   示例片段：
   ```YAML
   children:
     P:
       hosts:
         p0:
           ansible_host: "127.0.0.1"  # P0 节点实际 IP
           host_ip: "127.0.0.1"       # P0 节点实际 IP
         p1:
           ansible_host: "127.0.0.2"  # P1 节点实际 IP
           host_ip: "127.0.0.2"       # P1 节点实际 IP
     D:
       hosts:
         d0:
           ansible_host: "127.0.0.3"  # D0 节点实际 IP
           host_ip: "127.0.0.3"       # D0 节点实际 IP（主节点）
         d1:
           ansible_host: "127.0.0.4"  # D1 节点实际 IP
           host_ip: "127.0.0.3"       # D1 节点 host_ip 需与 d0 一致（主节点 IP）
     C:
       hosts:
         c0:
           ansible_host: "127.0.0.1"  # 全局代理节点（C0）实际 IP

   ```
   配置 SSH 私钥路径（用于 Ansible 远程连接），参考 [文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/README.md#%E5%AF%86%E9%92%A5%E6%96%87%E4%BB%B6%E7%9A%84%E5%87%86%E5%A4%87)

   ```YAML
   all:
     vars:
       ansible_ssh_private_key_file: /path/to/your/private_key.pem  # 替换为实际私钥文件路径
   ```

2. **omni_infer_server_template.yml**

    修改以下环境变量以适配实际路径与镜像：
    ```yaml
    environment:
      # 全局配置
      LOG_PATH: "/data/log_path"                          # 日志存储路径
      MODEL_PATH: "/data/models/DeepSeek-R1-w8a8"         # 模型文件存储路径（需提前放置转换后的权重）
      LOG_PATH_IN_EXECUTOR: "/data/log_path_in_executor"  # 执行器日志路径
      CODE_PATH: "/data/local_code_path"                  # omniinfer 本地代码路径
      HTTP_PROXY: ""                                      # 下载代理（无代理则留空）

      # 容器配置
      DOCKER_IMAGE_ID: "REPOSITORY:TAG"                   # 镜像与标签
      DOCKER_NAME_P: "your_prefill_container_name"        # P容器名称（自定义）
      DOCKER_NAME_D: "your_decode_container_name"         # D容器名称（自定义）
      DOCKER_NAME_C: "your_proxy_container_name"          # Proxy容器名称（自定义）
    ```

    配置说明：详细参数解释参考 [文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/template/README.md#%E7%9B%B8%E5%85%B3%E6%96%87%E4%BB%B6%E8%A7%A3%E9%87%8A%E8%AF%B4%E6%98%8E)。

#### 执行部署命令
进入模板目录并运行 Ansible Playbook：
```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml
```

#### curl 测试

部署成功后，通过 curl测试服务接口（默认端口 7000，适配多机架构）：

```bash
curl -X POST http://127.0.0.1:7000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 50,
        "prompt": "how are you?"
      }'
```

#### 注意事项

- 代码版本控制​​：Ansible 每次执行时会强制重置 vllm 代码库（checkout -f），若需修改代码，请提前提交或暂存变更。

## 性能优化建议

**1. 使用图缓存**  
首次启动服务时，模型会从头编译。建议首次成功启动后，重新执行以下命令以启用图缓存，提升性能：
```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_server
```

**2. 调整 Proxy 批处理大小**  
默认批处理大小为 25，建议根据每个 Die 的平均并发量 n调整为 n+1。操作步骤：  
进入proxy容器，修改 `/usr/local/nginx/conf/nginx.conf`中的对应参数后，执行 `nginx -c /usr/local/nginx/conf/nginx.conf -s reload` 生效。


**3. 增大推理 Batch Size**  
Ansible 默认开启 MTP（多 token 预测）优化。若需调整 `--max-num-seqs`（批处理大小），需同步修改本地代码路径下的配置文件 `omniinfer/tests/test_config/test_config_decode.json` 中的 decode_gear_list值：
```JSON
"decode_gear_list": [batch_size * (1+num_speculative_tokens)]
```

示例：当 --max-num-seqs设为 32 时，对应配置为 "decode_gear_list": [64]（假设 num_speculative_tokens=1）。