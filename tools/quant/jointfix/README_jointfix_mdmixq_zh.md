# JointFix-MDMixQ：VL / OMNI 统一 W8A8 量化指南

本文说明如何使用 `jointfix-mdmixq` 对 OpenPangu VL 和 OMNI 模型进行统一
W8A8 后训练量化。该方法借用 MDMixQ 的模态解耦和专家显著性思想，但**不做混合
位宽**：所有进入 compressed-tensors 的 Linear 权重均为 W8，输入激活动态 A8。

## 1. 方法范围

`jointfix-mdmixq` 在原 JointFix 上增加三项校准策略：

1. 使用真实图片/音频，经原始 BF16 tower、merger/projector 得到 decoder 输入；
2. 沿 decoder 保留逐 token 的文本/非文本标签，并按 Pangu 的真实
   `sigmoid(logit)+correction_bias` Top-8 路由采集专家显著性；
3. MoE 校准样本默认 80% 分配给文本 token。共享 gate/up 平滑由文本显著专家
   主导，并额外保护每层非文本 Top-8；down_proj 仍逐专家搜索。

Attention 继续使用原 JointFix。最终精度策略如下：

| 模块 | 精度 |
|---|---|
| Decoder Attention / Dense MLP / Routed Expert | W8A8 |
| Router | BF16 |
| Shared Expert（默认保护） | BF16 |
| MTP / Next-N layers | **BF16，强制不变量** |
| ViT / Audio backbone | 独立 JointFix strict-GPTQ W8A8 |
| merger / projector | BF16 |

## 2. 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/omni-models/tools/quant/jointfix
# 保留运行环境预装且与 torch_npu 匹配的 PyTorch，不让 pip 替换它。
python -m pip install -e . --no-deps
python -m pip install 'pillow>=9' 'soundfile>=0.12' 'librosa>=0.10'
export OMNI_MODELS_ROOT=/path/to/omni-models
export PYTHONPATH=.:"$OMNI_MODELS_ROOT":${PYTHONPATH:-}
export VLLM_PLUGINS=omni_pangu_models
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

`torch_npu`、CANN 和模型插件由 OpenPangu/Ascend 运行环境提供，不通过本包安装。
CPU 环境可以运行大部分单元测试，但不能执行真实 NPU 量化。

先确认设备没有被服务占用：

```bash
npu-smi info
docker ps --format '{{.Names}}\t{{.Status}}'
```

若确认某个服务容器占用了目标 NPU，再显式停止它：

```bash
docker stop <producer/server-container> <multimodal-sidecar-container>
```

不要停止承载量化环境和当前终端的容器。

## 3. Decoder 多模态校准集

JSONL 每行是一条请求，路径可以是绝对路径或相对 JSONL 所在目录：

```jsonl
{"text":"一段有代表性的长文本，建议接近 --seq-len 的 token 长度。"}
{"image":"images/chart.jpg","text":"请分析图表并说明理由。"}
{"audio":"audio/question.wav","text":"请理解音频并回答问题。"}
{"image":"images/page.png","audio":"audio/instruction.wav","text":"结合两种信息回答。"}
```

VL 推荐混合纯文本与 `image`；OMNI 推荐混合纯文本、`image` 和 `audio`。当前版本每条请求最多一张
图和一段音频。`--seq-len` 是媒体 placeholder 展开后的长度上限，不会把请求补齐，
因此 padding 不进入激活和路由统计。

校准建议：

- 至少32条有代表性的请求，推荐 16 条长纯文本 + 8 条图像 + 8 条音频；
- 保持图片、OCR、图表、数学和真实场景的覆盖；
- 纯文本样本尽量接近 `--seq-len`，用于保持语言主干与 BF16 MTP 的分布一致性；
- prompt 不宜只有几个字，应包含真实任务中的描述、推理和回答指令；
- `MAX_PIXELS=401408` 通常可在1024 token上限内覆盖常见比例，超长时降低分辨率。

可以用 `examples/build_mixed_omni_manifest.py` 从 WikiText 和已有媒体
manifest 确定性生成上述联合校准集。

下面的命令从文本 parquet 和已有媒体 manifest 确定性生成32条联合校准请求：

```bash
export MODEL=/path/to/bf16_omni

python examples/build_mixed_omni_manifest.py \
  --model "$MODEL" \
  --wikitext /path/to/text_calibration.parquet \
  --media-manifest /path/to/media_manifest.jsonl \
  --output /path/to/decoder_calibration.jsonl \
  --text-samples 16 --image-samples 8 --audio-samples 8 \
  --text-tokens 896
```

生成结果为32条请求：16条纯文本、8条图像、8条音频。纯文本在套入对话模板后约
915 token；多模态请求保留真实可变长度，不做padding。纯文本超过 `--seq-len` 时会
安全截断；媒体请求若展开后超长仍会报错，避免截断placeholder与tower embedding。

## 4. 推荐：通用脚本分步运行

脚本：[scripts/run_jointfix_mdmixq_w8a8.sh](scripts/run_jointfix_mdmixq_w8a8.sh)

### 4.1 OMNI模型

```bash
export MODEL=/path/to/bf16_omni
export CALIB_MANIFEST=/path/to/omni_decoder_calib.jsonl
export VISION_MANIFEST=/path/to/vision_manifest.jsonl
export AUDIO_MANIFEST=/path/to/audio_manifest.jsonl
export LAYER_DIR=/path/to/output/decoder_layers
export TEXT_BASE=/path/to/output/decoder_w8a8_mtp_bf16
export MM_ARTIFACTS=/path/to/output/mm_layers
export FINAL_MODEL=/path/to/output/final_omni_w8a8_mtp_bf16

bash scripts/run_jointfix_mdmixq_w8a8.sh omni decoder
bash scripts/run_jointfix_mdmixq_w8a8.sh omni finalize
bash scripts/run_jointfix_mdmixq_w8a8.sh omni vision
bash scripts/run_jointfix_mdmixq_w8a8.sh omni audio
bash scripts/run_jointfix_mdmixq_w8a8.sh omni merge
```

确认每一步输出后再运行下一步。也可以对新目录执行：

```bash
bash scripts/run_jointfix_mdmixq_w8a8.sh omni all
```

### 4.2 VL模型

VL模型不运行Audio Tower：

```bash
export MODEL=/path/to/bf16_vl
export CALIB_MANIFEST=/path/to/vl_decoder_calib.jsonl
export VISION_MANIFEST=/path/to/vision_manifest.jsonl
export LAYER_DIR=/path/to/output/decoder_layers
export TEXT_BASE=/path/to/output/decoder_w8a8_mtp_bf16
export MM_ARTIFACTS=/path/to/output/vision_layers
export FINAL_MODEL=/path/to/output/final_vl_w8a8_mtp_bf16

bash scripts/run_jointfix_mdmixq_w8a8.sh vl decoder
bash scripts/run_jointfix_mdmixq_w8a8.sh vl finalize
bash scripts/run_jointfix_mdmixq_w8a8.sh vl vision
bash scripts/run_jointfix_mdmixq_w8a8.sh vl merge
```

各子命令只要求本步骤需要的环境变量。例如单独迁移tower量化时，`vision` 只需要
`MODEL`、`MM_ARTIFACTS`、`VISION_MANIFEST`，不要求预先填写decoder输出路径。

脚本按 `config.json` 获取 decoder 层数和 MTP 范围。ViT/Audio 的Torch校准实现当前针对
OpenPangu相同模块命名；在其他模型上测试前先核对 `visual.blocks.*`、
`audio_tower.layers.*`、merger/projector名称。不同depth模型应先运行一条样本smoke。

## 5. Decoder 命令详解

脚本中的核心命令为：

```bash
python -m jointfix.cli quantize \
  --backend pangu --method jointfix-mdmixq \
  --model "$MODEL" --output "$LAYER_DIR" \
  --calib-data "$CALIB_MANIFEST" --calib-format omni \
  --n-samples 32 --seq-len 1024 \
  --mm-min-pixels 50176 --mm-max-pixels 401408 \
  --num-iterations 2 --iter-ab-tol 0.05 \
  --num-devices 16 --device npu \
  --objective output-recon --write-quant gptq \
  --skip-shared-experts --no-finalize
```

MDMixQ相关参数及默认值：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mdmixq-text-sample-ratio` | 0.8 | MoE样本缓存中文本token占比 |
| `--mdmixq-text-expert-fraction` | 0.25 | 每层参与共享scale目标的文本显著专家上限 |
| `--mdmixq-nontext-topk` | 8 | 每层额外保护的非文本专家数 |
| `--mdmixq-candidate-multiplier` | 4 | 路由置信候选缓存倍数 |

不要把 `jointfix-mdmixq` 与纯文本 parquet 组合；CLI会直接拒绝。

这里的含义是不要直接把 parquet 传给 `--calib-data`。应先用上面的构建器把文本窗口
和媒体请求合成 Omni JSONL，再以 `--calib-format omni` 运行。

### 5.1 长时间任务与监控

从远程终端启动时建议将标准输出写入持久日志。主机环境可使用 `nohup`；容器环境可用
等价的后台任务管理方式：

```bash
mkdir -p /path/to/logs
nohup env MODEL="$MODEL" CALIB_MANIFEST="$CALIB_MANIFEST" \
  LAYER_DIR="$LAYER_DIR" N_SAMPLES=32 SEQ_LEN=1024 \
  NUM_ITERATIONS=2 NUM_DEVICES=16 \
  bash scripts/run_jointfix_mdmixq_w8a8.sh omni decoder \
  > /path/to/logs/mdmixq_decoder.log 2>&1 &
```

随时可用以下只读命令查看进度，不依赖原SSH会话：

```bash
LAYER_DIR=/path/to/decoder_layers
find "$LAYER_DIR" -maxdepth 1 -name 'layer_*.safetensors' | wc -l
cat "$LAYER_DIR/.quantize_checkpoint.json"
ps -eo pid,etimes,cmd | grep 'jointfix.cli quantize' | grep -v grep
npu-smi info
```

注意：逐层文件和checkpoint是原子写入的，但当前runner尚未实现从非零decoder层
fast-forward校准状态。因此运行中不要停止量化容器或杀进程；若进程意外退出，保留
目录用于排查，但现版本不能直接从下一层续算，需使用新目录从第0层重跑。

## 6. 产物与强制校验

Decoder逐层目录应包含：

```text
layer_0000.safetensors ... layer_NNNN.safetensors
joint_search_traces.json
.quantize_checkpoint.json
```

trace中包含：

- 每层文本/非文本 token 数与阈值；
- 文本显著专家和非文本Top-K；
- JointFix搜索的 `(a,b)` 与重建目标；
- 每个INT8矩阵scale优化前/后的权重NMSE。

scale优化使用固定INT8 code下的逐输出行最小二乘，并在BF16 scale实际舍入后执行严格
no-regression gate。因此 `weight_nmse_after_scale_refine` 必须逐矩阵不大于
`weight_nmse_before_scale_refine`。

完整量化后可执行自动审计：

```bash
python examples/audit_jointfix_mdmixq.py \
  --model "$MODEL" --artifacts "$LAYER_DIR"
```

审计要求decoder逐层文件完整、每个MoE层同时观察到文本/非文本路由，且所有记录了
scale优化的矩阵均无权重NMSE回退；输出同时给出优化矩阵数和汇总相对改善比例。

若移动了tower产物、元数据中记录的原模型路径已不可访问，校验时显式指定同架构
checkpoint以读取动态depth：

```bash
python examples/quantize_omni_multimodal.py validate \
  --artifacts "$MM_ARTIFACTS" --model "$MODEL" \
  --require-vision --require-audio
```

`finalize-text` 会直接从原始checkpoint透传全部MTP层。验证器要求：

- Decoder存在INT8 Linear；
- MTP INT8数为0且BF16权重存在；
- tower在merge前保持BF16；
- 最终qconfig为compressed-tensors W8A8 dynamic。

Decoder与tower完成后的通用组装和校验命令：

```bash
export MODEL=/path/to/bf16_omni
export LAYER_DIR=/path/to/output/decoder_layers
export TEXT_BASE=/path/to/output/decoder_w8a8_mtp_bf16
export MM_ARTIFACTS=/path/to/output/mm_layers
export FINAL_MODEL=/path/to/output/final_omni_w8a8_mtp_bf16

python examples/audit_jointfix_mdmixq.py \
  --model "$MODEL" --artifacts "$LAYER_DIR"

MODEL="$MODEL" LAYER_DIR="$LAYER_DIR" TEXT_BASE="$TEXT_BASE" \
  bash scripts/run_jointfix_mdmixq_w8a8.sh omni finalize

python examples/quantize_omni_multimodal.py validate \
  --artifacts "$MM_ARTIFACTS" --model "$MODEL" \
  --require-vision --require-audio

MODEL="$MODEL" TEXT_BASE="$TEXT_BASE" MM_ARTIFACTS="$MM_ARTIFACTS" \
FINAL_MODEL="$FINAL_MODEL" \
  bash scripts/run_jointfix_mdmixq_w8a8.sh omni merge

python examples/quantize_omni_multimodal.py validate-text \
  --model "$FINAL_MODEL"
```

MTP的严格验证不是只看dtype计数。层范围由原模型 `config.json` 的
`num_hidden_layers` 和 `num_nextn_predict_layers` 动态确定；应对该范围内的原始模型与
最终模型张量做名称、shape、dtype和逐元素完全相等检查，且最终模型对应层不能存在
任何 `.weight_scale`。

## 7. 服务验收

用目标工程的启动脚本加载 `FINAL_MODEL`，必须保留Omni/VL architecture设置，并开启
MTP时明确指定BF16模型权重。服务就绪后至少发送：

1. 纯文本请求；
2. 图片请求；
3. OMNI模型再发送音频请求；
4. 一条较长输出请求，用日志核对MTP接受率。

将服务配置中的模型路径设为 `$FINAL_MODEL`，使用目标推理工程的标准启动方式。服务
至少应暴露健康检查、模型查询和生成接口，例如：

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

必须等待部署拓扑中的全部后端健康，并从 `/v1/models` 确认加载的是
`$FINAL_MODEL`，再开始功能和性能测试。

验收条件：HTTP成功、回复非空且语义正常、无权重加载异常、无NaN/Inf、MTP接受率不
出现明显坍塌。性能测试应固定输入/输出长度、并发和随机种子，再与BF16或原JointFix
基线比较TTFT、TPOT和接受率。

不要只用几条短回复判断MTP。三步MTP中，一次拒绝会使短样本比例非常敏感。推荐用
`vllm bench serve` 固定 `--random-input-len 2048 --random-output-len 1024`、
`--ignore-eos`，并在压测前后聚合8个decode后端的以下counter增量：

```text
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_drafts_total
```

计算口径：

```text
MTP接受率 = Δaccepted_tokens / Δdraft_tokens
每轮平均接受token = Δaccepted_tokens / Δdrafts
平均acceptance length = 1 + Δaccepted_tokens / Δdrafts
```

接受率必须和相同模型、服务配置、输入长度、输出长度、并发数及随机种子的 BF16 或
既有 W8A8+BF16 MTP 基线比较。若显著低于基线，即使MTP张量为BF16且逐项相等，也应
继续检查量化主干与MTP的分布失配，不能直接判定验收通过。

## 8. 常见问题

- **expanded prompt超过seq_len**：降低 `MAX_PIXELS` 或增大 `SEQ_LEN`。
- **NPU OOM**：先用 `npu-smi info` 找到真实占用者；确认后停止对应服务容器。
- **GPTQ退化成RTN**：检查日志中的 `GPTQ-RUN`，提高 `sample_rows/sample_limit`。
- **MTP接受率低**：先检查46层之后是否存在`.weight_scale`并逐项比对原始MTP；若
  MTP完全一致仍低，检查联合校准集是否包含足够的真实长纯文本，以及量化主干相对
  历史W8A8基线的尺度/路由漂移。
- **已有输出目录**：脚本不会覆盖，换新目录或人工确认后移动旧产物。
- **其他VL/OMNI架构**：先核对config字段、权重前缀、tower depth和processor；不要直接
  假设OpenPangu的24层ViT/16层Audio固定值适用于所有模型。
