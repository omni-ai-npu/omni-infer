# 部署环境说明

以Ascend910C (A3) 为例，openPangu-2.0-Flash的BF16和INT8权重版本均可通过**1P1D**的配置拉起服务。可以使用一机A3组一个P节点，一机A3组一个D节点，这样的**1P1D**共使用两机A3。

## 拉取镜像

拉取机器对应镜像

```bash
docker pull image_name:image_tag
```

示例

```bash
docker pull registry-cbu.huawei.com/omniai_omniinfer/ai-infra-infer-1.2.1-arm-openeuler-py311-a3-cann.9.1.t2.b020-pta_v2.9.0-op-1.2.1-omni_release_1.2.1-vllm-202606051040:0.0.1
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

适配推理代码的部分packages及版本：

* `omni-npu`, Version `0.2.0`
* `omni-models`, Version `0.1.0`
* `vllm`, Version `0.14.0+empty`
* `tiktoken`, Version `0.13.0`
* `tokenizers`, Version `0.22.2`
* `torch`, Version `2.9.0`
* `torch-npu`, Version `2.9.0.post3.dev20260522`
* `transformers`, Version `4.57.6`
* `Python`, Version `3.11.12`

# ansible-playbook在多机上拉起推理服务

## 修改脚本

拉起PD分离服务的脚本在``omniinfer/tools/ansible/template``路径下。

* 在**omni_infer_inventory_used_for_xPyD.yml**文件中依次填写**P节点**和**D节点**的机器ip地址。**proxy节点**设为P节点ip。端口可用当前默认。注意ansible_host和host_ip都要修改为部署的ip地址。

```
P:
      children:
        P0:
          hosts:
            p0:
              ansible_host: "127.0.0.1" # 替换成本地部署的ip地址
              node_rank: 0
              kv_rank: 0
              node_port: "{{ global_port_base + port_offset.P + kv_rank * 10 }}"
              api_port: "{{ base_api_port + port_offset.P + kv_rank *10 + node_rank }}"
              host_ip: "127.0.0.1" # 替换成本地部署的ip地址
              ascend_rt_visible_devices: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
```

* 在**omni_infer_server_template_performancexPyD_92B_xxx.yml**文件中写有P、D和C的服务拉起脚本和配置。
  首次拉起服务前需改动`environment`部分，将所有路径相关和容器名都换成自己的信息。

```
environment:
    # Global Configuration
    LOG_PATH: "/path/to/server/log/" 	# 日志存储路径，必须设置，且全流程必须完整存在，不然无法跟踪服务
    MODEL_PATH: "/path/to/model/weights/" 	# 本机权重路径，P和D所有节点需保持一致
    MODEL_LEN_MAX_PREFILL: "524288"
    MODEL_LEN_MAX_DECODE: "524288"
    LOG_PATH_IN_EXECUTOR: "/path/to/server/log_path_in_executor"
    CODE_PATH: "/path/to/model/codes/" # 本机代码路径
    KV_CONNECTOR: "LLMDataDistConnector"

    # Configuration for containers
    DOCKER_IMAGE_ID: "image_name:image_tag" 	# PD分离docker使用的镜像，跟上文拉取到各个机器上的镜像保持一致
    DOCKER_NAME_P: "docker_name_p" 	# PD分离在P节点创建的容器名，需提前设置
    DOCKER_NAME_D: "docker_name_d" 	# PD分离在D节点创建的容器名，需提前设置
    DOCKER_NAME_C: "docker_name_c" 	# PD分离在proxy节点创建的容器名，需提前设置
    SCRIPTS_PATH: "/tmp/scripts_path"

    # Configuration for lb_sdk in global proxy
    PREFILL_LB_SDK: "pd_score_balance"
    DECODE_LB_SDK: "pd_score_balance"
    USE_OMNI_PROXY: "1"

    # Tensor Parallel Size
    DECODE_TENSOR_PARALLEL_SIZE: "1" # 当前脚本默认prefill TP部署，decode DP部署
```

其次**P节点**配置在`run_vllm_server_prefill_cmd:`，**D节点**配置在`run_vllm_server_decode_cmd:`，可使用默认配置，也可根据需求开关特性。

## 启动镜像

在P节点运行下述命令可启动镜像，在设置的每台服务器上创建docker。注意替换成本机上的对应文件名。

```bash
ansible-playbook -i omni_infer_inventory_used_for_xPyD.yml omni_infer_server_template_performancexPyD_92B_xxx.yml --tags run_docker
```

注意docker内环境配置好了可以复用docker，不要再运行此命令，因为再次运行会把同名docker覆盖掉。

## 推理代码适配

可直接使用镜像自带的代码，通过下述命令查看`omni-npu`和`omni-models`在docker内的安装路径。docker创建好后即可通过`ansible-book`直接拉起推理服务。

```bash
pip list | grep omni-npu
pip list | grep omni-models
```

也可以从代码仓中拉取最新代码替换镜像内置的`omni-npu`和`omni-models`。

### 替换镜像内置代码

可以通过下述命令进入创建好的docker。

```bash
docker exec -itu root docker_name /bin/bash
```

在docker内运行下述命令可安装本地推理代码。

```
# 安装omni-npu
cd /path/to/your/local/omni-npu
pip install -e . -v 

# 安装omni-models
cd /path/to/your/local/omni-models
pip install -e . -v
```

若上述命令无法安装本地代码，可以在代码路径内尝试下述命令。

```
pip install -e . -v --index-url https://mirrors.tools.huawei.com/pypi/simple --trusted-host mirrors.tools.huawei.com --no-build-isolation
```

### INT8量化

openPangu-2.0 模型支持使用 Joint SmoothQuant 方法生成 W8A8 INT8 量化权重。该方法对每个线性层联合搜索最优的 $(a,b)$ 平滑参数，采用 Hessian 通道加权与 K=2 坐标下降迭代，以及写端 GPTQ、读端 RTN 的混合量化策略，在保持推理精度的同时将模型体积压缩约 1.9×。量化工具位于 `tools/quantize_joint_smooth_int8_iterative.py`。

#### 量化命令（16卡昇腾 NPU）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export LOG=logs/joint_smooth_w8a8_$(date +%m%d_%H%M%S).log

time python3 -u tools/quantize_joint_smooth_int8_iterative.py \
    --model {浮点权重路径} \
    --output {W8A8量化权重路径} \
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

**关键参数说明：**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--n-samples` | 32 | 校准样本数 |
| `--seq-len` | 1024 | 校准序列长度 |
| `--num-iterations` | 2 | 坐标下降迭代轮数（K=2），在 gate/up ↔ down 的平滑参数之间交替优化 |
| `--iter-ab-tol` | 0.05 | 迭代收敛阈值，相邻两轮 (a,b) 变化低于 5% 时提前停止 |
| `--objective` | output-recon | 搜索目标：最小化层输出重建误差（比权重误差代理函数更准确） |
| `--write-quant` | gptq | 残差流方向（o_proj / down_proj）权重使用 GPTQ 量化以降低误差 |
| `--skip-shared-experts` | — | 保留 shared expert 为 BF16；该路径每 token 必经，量化误差全局累积 |
| `--build-meta` | — | 使用 meta device 初始化模块骨架，避免 BF16 全量权重在量化阶段占用 NPU 显存 |

#### 量化后 config.json

量化完成后，脚本自动在输出目录的 `config.json` 中写入 `quantization_config` 字段：

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
    "ignore": ["<跳过量化的层列表，由脚本自动生成>"]
}
```

权重采用 per-output-channel 静态量化，激活采用 per-token 动态量化。vLLM 会从 `quantization_config` 自动识别 compressed-tensors 格式，推理时无需额外指定 `--quantization` 参数。

### 使用c8量化

如果使用c8量化，即在`run_vllm_server_prefill_cmd:`或`run_vllm_server_decode_cmd:`的`EXTRA_ARGS`里配置`--kv-cache-dtype int8_ds_mla`或`--kv-cache-dtype li_int8_ds_mla`，则还需要修改镜像内置的vLLM。
在docker内运行`vim /opt/vllm/vllm/config/cache.py`，在``CacheDType``新增`int8_ds_mla`和`li_int8_ds_mla`。

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

此外，在docker内运行`vim /opt/vllm/vllm/utils/torch_utils.py`，在``STR_DTYPE_TO_TORCH_DTYPE``新增`"int8_ds_mla":torch.int8`和`"li_int8_ds_mla":torch.bfloat16`。

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

## 推理服务拉起

docker在各个部署的A3机器上创建好后，在bash通过下述命令拉取推理服务。

```bash
ansible-playbook -i omni_infer_inventory_used_for_xPyD.yml omni_infer_server_template_performancexPyD_92B_xxx.yml --tags run_server,run_proxy
```

C节点会在容器内启动nginx+proxy，在master node上启动nginx将并发的请求分配到各个节点上。可在部署的机器上通过日志追踪服务拉起的进程。

```bash
tail -f /path/to/server/log/server_0.log
```

## 发请求测试

服务启动后，在主节点或者其它节点向主节点proxy端口（脚本默认为7000）发送测试请求：

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


