## trace 功能配置
需要在 VLLM 启动脚本中新增以下配置项：

```bash
# ===================== trace 配置 =====================
# 1. 定义omniinfer的根目录
OMNIINFER_ROOT="/workspace/omniinfer"

# 2. 将omniinfer/tools目录加入PYTHONPATH，让Python能识别omni_trace模块
export PYTHONPATH="${OMNIINFER_ROOT}/tools:${PYTHONPATH}"

# 3. 指定trace配置文件路径
unset PROFILING_NAMELIST
export PROFILING_NAMELIST="${OMNIINFER_ROOT}/tools/omni_trace/omnilogger_namelist.yml"

# 4. 指定trace日志输出目录
export TRACE_OUTPUT_DIRECTORY=/data/user/trace/

# 5. 指定运行节点类型（prefill/decode二选一）
export ROLE="prefill"
export ROLE="decode"

# 6. 需要配置patch环境变量
unset OMNI_NPU_VLLM_PATCHES
export OMNI_NPU_VLLM_PATCHES="ProfilerDynamicPatch,RequestStatusPatch,OpenAIServingChatTokenLoggerPatch"
# 或
unset OMNI_NPU_VLLM_PATCHES_ALL
export OMNI_NPU_VLLM_PATCHES_ALL=1
# ==============================================================