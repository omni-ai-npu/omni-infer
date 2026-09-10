## openPangu-2.0 在 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 部署指导文档（jointfix 量化）

### Int8 量化

#### JointFix W8A8 量化

openPangu-2.0 模型支持使用 **jointfix** 以 JointFix 方法生成 W8A8 INT8 量化权重。该方法对每个线性层联合搜索最优的 $(a,b)$ 平滑参数，采用 Hessian 通道加权、K=2 坐标下降迭代、以及输出侧 GPTQ / 输入侧 RTN 的混合量化策略，在保持推理精度的同时将模型体积压缩约 1.9×。

jointfix 是一个模型无关、方法可插拔的独立量化工具箱。

##### 安装

```bash
cd jointfix
pip install -e .        # 安装 jointfix + 依赖(torch / safetensors / transformers / pandas / pyarrow)
```

> 装好后即有 `jointfix` 命令行；下文用 `python -m jointfix.cli` 等价。

##### 量化流程（16 卡昇腾 NPU）

jointfix 把量化分两步：**quantize**（逐层搜索 + 量化，产出每层 INT8 权重）+ **finalize**（组装成可部署的 compressed-tensors 模型）。

**Step 1 — 量化：**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 先 source CANN 环境
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export LOG=logs/omni_w8a8_$(date +%m%d_%H%M%S).log

time PYTHONPATH=.:$PYTHONPATH python -m jointfix.cli quantize \
    --backend pangu --method jointfix \
    --model {浮点权重路径} \
    --output {量化中间目录} \
    --calib-data examples/data/wikitext_train.parquet \
    --n-samples 32 --seq-len 1024 \
    --num-iterations 2 --iter-ab-tol 0.05 \
    --num-devices 16 --device npu \
    --objective output-recon --write-quant gptq \
    --skip-shared-experts \
    2>&1 | tee ${LOG}
```

> **NPU 启动注意**：必须先 `source set_env.sh`，且用 `PYTHONPATH=.:$PYTHONPATH`（**追加**，不要用 `PYTHONPATH=.` 覆盖）。否则昇腾 GE 算子编译器经 `PYTHONPATH` 找不到 `tbe`，首个被编译的算子（MoME 的 `F.conv1d`）会报 `GEInitialize failed / No module named 'tbe'`。或 `pip install -e .` 后直接用 `jointfix` 命令、完全不设 `PYTHONPATH`。

**Step 2 — 组装可部署模型：**

```bash
PYTHONPATH=.:$PYTHONPATH python -m jointfix.cli finalize \
    --model {浮点权重路径} \
    --quantized {量化中间目录} \
    --output {W8A8量化权重路径}
```

`finalize` 复用原模型的分片布局，把每层 INT8 权重 + scale 写回分片，未校准的可量化权重走 RTN，其余 passthrough，并写出 `model.safetensors.index.json`、带 `quantization_config` 的 `config.json`，以及拷贝分词器/建模代码等辅助文件。产出即为可被 vLLM 直接加载的 compressed-tensors 模型。

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
| `--num-devices` | 16 | 并行卡数：N 卡并行前向 + 分布式 gate+up 搜索 + 256 专家并行搜索（per-expert (a,b)） |
| `--device` | npu | 计算后端（npu / cuda / cpu） |

> 说明：jointfix **始终使用 meta-device 建层**，无需额外参数；在 NPU 上自动设置 `jit_compile=False`（aclnn 算子）。

##### 量化后 config.json

`finalize` 自动在输出目录的 `config.json` 中写入 `quantization_config` 字段：

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
    "ignore": ["<跳过量化的层列表，自动生成>"]
}
```

权重采用 per-output-channel 静态量化，激活采用 per-token 动态量化。vLLM 会从 `quantization_config` 自动识别 compressed-tensors 格式，推理时无需额外指定 `--quantization` 参数。

##### 自定义校准集

仓库自带 `examples/data/wikitext_train.parquet`（WikiText-2 通用文本）。GPTQ 对校准分布最敏感——若部署任务的分布与通用文本差异较大（agent 轨迹、CoT 推理），换成同分布的校准集（含 `text` 列的 parquet，通过 `--calib-data` 传入）可获得更好的量化精度。
