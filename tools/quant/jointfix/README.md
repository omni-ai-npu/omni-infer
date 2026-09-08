# jointfix

> OpenPangu VL / OMNI 的 `jointfix-mdmixq` 统一 W8A8、多模态联合校准、BF16 MTP、
> 部署和验收流程见 [README_jointfix_mdmixq_zh.md](README_jointfix_mdmixq_zh.md)。

**多模型、方法可插拔的 INT8（W8A8）训练后量化（PTQ）工具箱**，面向昇腾 NPU 上的大规模 MoE 模型（openPangu-2.0 等）。

输入一个 BF16 模型，输出一个 **vLLM 可直接加载**的 W8A8 compressed-tensors 模型，体积压缩约 **1.9×**，推理精度基本无损。

```
BF16 模型 ──jointfix quantize──▶ W8A8 可部署模型 ──▶ vLLM-Ascend
```

核心方法 **JointFix**：对每个线性层**联合搜索最优的 $(a,b)$ 平滑参数**，配合 Hessian 通道加权、K=2 坐标下降迭代，以及「输出侧 GPTQ + 输入侧 RTN」的混合量化,在保持精度的同时完成压缩。权重做 per-output-channel 静态量化，激活做 per-token 动态量化。

---

## 目录

- [架构](#架构) — 工具怎么组织的，怎么扩展
- [安装](#安装) — 依赖与命令行入口
- [快速开始](#快速开始) — **前置条件（逐步执行）** + 跑通第一条命令
- [完整生产流程](#完整生产流程多卡昇腾-npu) — quantize 出可部署模型,直接喂 vLLM
- [注意事项](#注意事项) — PYTHONPATH / CANN / 多卡等跑前须知
- [校准集](#校准集) · [许可证](#许可证)

---

## 架构

两条**正交的轴**，由一个与「模型/方法」均无关的 `core` 粘合。加新模型只动 `backends/`，加新量化方法只动 `methods/`，互不影响：

```
jointfix/
├── core/        # 模型/方法均无关:量化原语(RTN/GPTQ/fake-quant)、激活统计、
│                #   校准-量化主循环(runner)、断点(checkpoint)、部署组装(deploy)
├── backends/    # 「模型」轴: pangu(生产) · hf(实验性)
├── methods/     # 「方法」轴: jointfix(联合 (a,b) 搜索 + output-recon + GPTQ)
├── registry.py  # 把 --backend / --method 的名字解析成类
└── cli.py       # 命令行入口: jointfix quantize / finalize
examples/        # 自带默认校准集 + 单层 smoke 测试 + profiling 分析器
tests/           # pytest(纯 CPU,不需要真权重)
```

> **怎么读这张图**：一次量化 = `cli` 解析参数 → `registry` 选出 backend + method → `runner` 逐层做「BF16 前向收统计 → method 搜索+量化 → 存盘 → 量化后重新前向传误差」。所有架构相关的细节都封在 backend 里，所有量化算法都封在 method 里。

---

## 安装

需要 **Python ≥ 3.9**。

**CPU / CUDA**:

```bash
cd jointfix             # 注意不要进入 jointfix 子文件夹
pip install -e .        # 安装 jointfix + 依赖(torch>=2.1 / safetensors>=0.4 / transformers>=4.40 / numpy / pandas / pyarrow)
```

**NPU(昇腾)**:`torch` / `torch_npu` 由 CANN 提供,**绝不能让 pip 碰 torch**。`pyproject.toml` 把 `torch>=2.1` 列为硬依赖,直接 `pip install -e .` 一旦 CANN 的 torch 版本号不满足 `>=2.1`,pip 就会从 PyPI 拉 CPU 版 torch 盖掉 `torch_npu`、废掉整台机器的 NPU 后端。所以 NPU 上用 `--no-deps` 装本体、再单独补非 torch 依赖:

```bash
cd jointfix
pip install -e . --no-deps
pip install "safetensors>=0.4" "transformers>=4.40" numpy pandas pyarrow
```

装好后即有 `jointfix` 命令行；下文所有 `jointfix ...` 都等价于 `python -m jointfix.cli ...`。

---

## 快速开始

### 第 0 步:前置条件（逐步执行）

| # | 前置条件 | 执行动作 |
|---|----------|----------|
| 1 | 装好工具 | CPU/CUDA:`cd jointfix && pip install -e .`<br>**(仅 NPU)** 改用 `pip install -e . --no-deps` + 单独补非 torch 依赖,别让 pip 盖掉 CANN 的 torch_npu,见 [安装](#安装) |
| 2 | **（仅 NPU）** 加载 CANN 环境 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 3 | **（仅 NPU）** 选择可见 die | `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`（A3 每卡 2 die） |
| 4 | 准备校准集 | 已自带 `examples/data/wikitext_train.parquet`，**无需动作**；自定义见 [校准集](#校准集) |
| 5 | **（建议）** 先验证 backend 能在真权重上跑通 | `python examples/smoke_pangu_layer.py --model {浮点权重路径} --layer 0 --device npu`<br>尾部打印 `SMOKE PASS` 即可放心跑长任务 |


### 第 1 步:跑一次量化

```bash
jointfix quantize \
    --backend pangu --method jointfix \
    --model {浮点权重路径} --output {全新的输出目录} \
    --calib-data examples/data/wikitext_train.parquet \
    --n-samples 32 --seq-len 1024 \
    --num-devices 8 --device npu \
    --objective output-recon --write-quant gptq --skip-shared-experts
```

跑完 `--output` 就是 **vLLM 可直接加载的 W8A8 模型**。完整参数:`jointfix quantize --help`。

---

## 完整生产流程（多卡昇腾 NPU）

`quantize` 逐层搜索并量化,直接产出 vLLM 可加载的 W8A8 compressed-tensors 模型,写到 `--output`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh           # 先 source CANN
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15   # 16 个 die(A3:8 卡 × 2 die)
export LOG=logs/jointfix_w8a8_$(date +%m%d_%H%M%S).log

time PYTHONPATH=.:$PYTHONPATH python -m jointfix.cli quantize \
    --backend pangu --method jointfix \
    --model {浮点权重路径} \
    --output {W8A8量化权重路径} \
    --calib-data examples/data/wikitext_train.parquet \
    --n-samples 32 --seq-len 1024 \
    --num-iterations 2 --iter-ab-tol 0.05 \
    --num-devices 16 --device npu \
    --objective output-recon --write-quant gptq \
    --skip-shared-experts \
    2>&1 | tee ${LOG}
```

运行时每层打印进度 + ETA(`done i/N | elapsed | avg | ETA`);全部跑完自动组装,日志末尾打印 `deployment model written to: {W8A8量化权重路径}`。

> `--output` 除最终分片外还有逐层中间产物 `layer_*.safetensors`。**确认部署模型可用后**想省磁盘,可 `rm {W8A8量化权重路径}/layer_*.safetensors`,vLLM 只读 `index.json` 引用的分片,删掉不影响加载。

> **NPU 启动注意**:必须先 `source set_env.sh`,且 `PYTHONPATH=.:$PYTHONPATH`(**追加**,不要用 `PYTHONPATH=.` 覆盖)。详见 [⚠️ 注意事项](#注意事项)。

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--n-samples` | 32 | 校准样本数 |
| `--seq-len` | 1024 | 校准序列长度 |
| `--num-iterations` | 2 | 坐标下降迭代轮数(K=2),在 gate/up ↔ down 的平滑参数间交替优化 |
| `--iter-ab-tol` | 0.05 | 迭代收敛阈值,相邻两轮 (a,b) 变化 < 5% 时提前停止 |
| `--objective` | output-recon | 搜索目标:最小化层输出重建误差(比权重误差代理更准) |
| `--write-quant` | gptq | 残差流方向(o_proj / down_proj)权重用 GPTQ 量化以降误差 |
| `--skip-shared-experts` | 建议开 | 保留 shared expert 为 BF16,该路径每 token 必经,量化误差会全局累积 |
| `--num-devices` | 16 | 并行 **device(die)** 数:N 路并行前向 + 分布式 gate+up 搜索 + 逐专家 (a,b) 并行搜索。**A3 每卡 2 die**,8 卡 = 16 die 故填 16;其它机型按容器内 `torch.npu.device_count()` 填 |
| `--device` | npu | 计算后端(npu / cuda / cpu) |


### 部署（vLLM-Ascend）与 config.json

finalize(一步流程里自动执行)会在输出目录的 `config.json` 写入 `quantization_config`:

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
    "ignore": ["<跳过量化的层列表,自动生成>"]
}
```

权重 per-output-channel 静态量化、激活 per-token 动态量化。vLLM 从 `quantization_config` 自动识别 compressed-tensors,推理时**无需**额外指定 `--quantization`。部署到 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 时直接把 `{W8A8量化权重路径}` 当普通模型加载即可。

### OpenPangu Omni ViT + Audio Tower（无 RTN）

ViT/Audio 需要真实图像和音频激活，不能通过普通 finalize 的
`rtn_uncalibrated` 路径量化。准备两个 manifest（格式见
[`examples/data/omni_multimodal_manifest.example.md`](examples/data/omni_multimodal_manifest.example.md)），然后运行：

完整的 decoder、ViT、Audio、finalize 和 merge 命令见
[JointFix-MDMixQ 复现指南](README_jointfix_mdmixq_zh.md)。推荐使用参数化入口：

```bash
MODEL=/path/to/bf16_omni \
CALIB_MANIFEST=/path/to/decoder_calib.jsonl \
VISION_MANIFEST=/path/to/vision_calib.jsonl \
AUDIO_MANIFEST=/path/to/audio_calib.jsonl \
LAYER_DIR=/path/to/output/decoder_layers \
TEXT_BASE=/path/to/output/text_base \
MM_ARTIFACTS=/path/to/output/mm_layers \
FINAL_MODEL=/path/to/output/final_model \
bash scripts/run_jointfix_mdmixq_w8a8.sh omni all
```

该路径逐层传播已量化误差，JointFix 搜索使用真实 tower 激活，最终写权重使用
strict block-GPTQ。ViT 主干共 96 个、Audio Tower 主干共 96 个 INT8 Linear；任何权重
采样不足或 Cholesky 失败都会停止，**不会回退到 RTN**。校准产物支持按层续跑。
最终 merge 只替换明确存在的校准产物，也不会量化任何 uncovered weight。

---

## 注意事项


1. **`PYTHONPATH` 只能追加,不能覆盖(NPU 上最常见的坑)。**
   昇腾 GE 算子编译器经 `PYTHONPATH` 寻找 `tbe`;`PYTHONPATH=. python ...` 会丢掉 CANN 路径,首个被编译的算子报 `GEInitialize failed / No module named 'tbe'`。
   ✅ 用 `PYTHONPATH=.:$PYTHONPATH`,或 `pip install -e .` 后直接用 `jointfix` 命令、完全不设 `PYTHONPATH`。

2. **跑前必须 `source set_env.sh`。** 否则同样在第一个算子编译时 `GEInitialize failed`。

3. **`--output` 用全新目录(暂不支持断点续跑)。**
   复用已有部分结果的目录会触发 resume,而 resume 尚未实现,会直接 `NotImplementedError`。每次跑用一个新目录(可带时间戳);`--start-layer > 0` 同理。

4. **多卡和单卡产出的权重不同。**
   `--num-devices > 1` 时,每个路由专家**独立**搜自己的 $(a,b)$(per-expert);单卡时退化为所有专家取**中位数** $(a,b)$ 统一应用。两者数值不同,**复现 / 生产请固定用多卡**。

5. **校准集分布会影响精度(尤其 GPTQ)。** 见 [校准集](#校准集)。

---

## 校准集

仓库自带 `examples/data/wikitext_train.parquet`(WikiText-2 通用文本,含 `text` 列,~5.9 MB),开箱即用。

GPTQ 对校准分布最敏感。若部署任务的分布与通用文本差异较大(agent 轨迹、CoT 推理),换成**同分布**的校准集(含 `text` 列的 parquet,经 `--calib-data` 传入)能拿到更好的量化精度。

OpenPangu Omni 的 decoder 也可以使用真实图像/音频校准。JSONL 格式见
[`examples/data/omni_llm_manifest.example.md`](examples/data/omni_llm_manifest.example.md)，
运行时增加 `--calib-format omni`。该入口先以原始 BF16 ViT/Audio Tower 和
projector 生成模态 embedding，再替换 decoder placeholder；后续 JointFix 算法和
最终 W8A8 格式不变。多模态请求保持真实变长序列，不会把 padding token 纳入统计。

---

## 许可证

[MIT](LICENSE) © 2026 华为技术有限公司
