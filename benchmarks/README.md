# benchmarks

本目录存放推理服务性能测试（benchmark）脚本。

## pangu_ultra_moe 性能压测脚本

`pangu_ultra_moe_acs_bench.sh` 通过 [acs-bench](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5906032.html) 对 vLLM 服务执行 `pangu_ultra_moe` 模型的长文本性能压测，覆盖以下场景：

- 输入 8192 token / 输出 2000 token，共 10155 个请求
- 并发 1024，采用限速注入（`request-rate 4`，每秒 4 个请求，`burstiness 100`）
- 开启投机推理（`--use-spec-decode`，`--num-spec-tokens 3`）
- 关闭 thinking（`--generation-config '{"chat_template_kwargs": {"thinking": false}}'`）
- warmup 2000 个请求后开始统计

### 前置条件

1. **安装 acs-bench**：环境需 `python>=3.10`，安装后可用 `pip show acs-bench` 验证。安装包位于 ModelArts 软件包 `AscendCloud-6.5.907-xxx.zip/AscendCloud-LLM-xxx` 的 `llm_tools` 目录：

   ```bash
   pip install acs_bench-*-py3-none-any.whl
   ```

2. **准备数据集**：`--input-path` 指定的 `claw_all_openai.json` 需为 OpenAI Chat 格式（`CustomOpenAIChat`）。使用 `acs-bench generate dataset` 生成：

   ```bash
   acs-bench generate dataset \
       --tokenizer ./tokenizer/pangu_ultra_moe \
       --dataset-type CustomOpenAIChat \
       --output-path ./claw_all_openai.json \
       --input-length 8192 \
       --num-requests 10155
   ```

   也可复用已有的同规格数据集文件。

3. **启动待测服务**：确保 vLLM 服务已就绪，`base_url` 指向服务的 `/v1` 地址（如 `http://<host>:<port>/v1`）。

### 使用方法

直接运行（使用脚本内默认的 `base_url` 与数据集路径）：

```bash
bash benchmarks/pangu_ultra_moe_acs_bench.sh
```

通过环境变量覆盖服务地址与数据集路径：

```bash
BASE_URL=http://10.0.0.1:7000/v1 INPUT_PATH=/data/datasets/claw_all_openai.json \
    bash benchmarks/pangu_ultra_moe_acs_bench.sh
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model-args` | `[{"model_name": "pangu_ultra_moe", "base_url": "..."}]` | 服务模型名与服务端地址，实际使用时需改为真实地址 |
| `--dataset-type` | `CustomOpenAIChat` | 数据集为 OpenAI Chat 格式 |
| `--input-path` | `claw_all_openai.json` 路径 | 数据集文件路径 |
| `--concurrency` | `1024` | 最大并发请求数 |
| `--num-requests` | `10155` | 压测请求总数 |
| `--input-length` / `--output-length` | `8192` / `2000` | 输入/输出 token 长度 |
| `--request-rate` / `--burstiness` | `4` / `100` | 请求到达速率（每秒请求数）及突发因子 |
| `--use-spec-decode` / `--num-spec-tokens` | `true` / `3` | 开启投机推理并设置 spec token 个数 |
| `--warmup` | `2000` | warmup 请求数，之后才开始统计 |
| `--timeout` | `7200` | 单请求超时时间（秒） |
| `--ignore-eos` | `False` | 不忽略 EOS，输出自然结束 |

### 结果产物

运行结束后，在 `--benchmark-save-path`（默认 `./benchmark_output`）下生成：

- `requests/requests_*.csv`：每个请求的时延等详情
- `summary_*.csv`：吞吐、时延等汇总指标
