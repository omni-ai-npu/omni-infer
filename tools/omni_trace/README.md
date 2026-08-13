## trace 功能配置
需要在 VLLM 启动脚本中新增以下配置项：

```bash
# ===================== trace 配置 =====================
# 1. 定义omniinfer的根目录
export OMNIINFER_ROOT="/workspace/omniinfer"

# 2. 将omniinfer/tools目录加入PYTHONPATH，让Python能识别omni_trace模块
export PYTHONPATH="${OMNIINFER_ROOT}/tools:${PYTHONPATH}"

# 3. 开启trace并指定日志输出目录；不设置该变量时trace关闭
export OMNI_TRACE_OUTPUT_DIRECTORY=/data/user/trace/

# 4. 指定运行节点类型（encode/prefill/decode三选一）
export OMNI_PD_ROLE="encode"
export OMNI_PD_ROLE="prefill"
export OMNI_PD_ROLE="decode"
# ==============================================================
```
