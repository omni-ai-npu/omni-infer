# Profiling
Currently, there are two logic options, and only one can be enabled at a time:
1. Old logic, triggered when PROFILER_TOKEN_THRESHOLD is set, consistent with version v0.
2. New logic, triggered when PROFILER_TOKEN_THRESHOLD is not set, consistent with vLLM.
In the Ray mode, the old logic cannot retrieve profiles.

The main configuration items are the following environment variables:
1. `VLLM_TORCH_PROFILER_DIR`: Set in environment, this variable specifies the location where profiling files will be stored. After setting it, you can start profiling data collection via the POST /start_profile interface and stop it via the POST /stop_profile interface. If not set, profiling will not be enabled.
This is consistent with the official API.

2. `PROFILER_STOP_STEP`: Only work on old logic(auto start),this variable indicates the number of steps for which profiling data will be collected. Set it to the number of steps you want to collect after starting. The default value is 5. Profiling will automatically stop after reaching the specified number of steps.

3. `PROFILER_TOKEN_THRESHOLD`: Do not set this variable if you want to use the new logic (manual start). If set, the old logic will be enabled by default. This variable represents the number of tokens for step scheduling and serves as the entry condition for profiling data collection. You can use this value to distinguish different stages of data collection. The default value is 1.

4. `VLLM_TORCH_PROFILER_RECORD_SHAPES`: Input shapes and input types of operators, of type bool. Values: True: enabled. False: disabled, Default value. This setting takes effect when torch_npu.profiler.ProfilerActivity.CPU is enabled.

5. `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY`: Memory usage of operators, of type bool. Values: True: enabled. False: disabled, Default value.

6. `VLLM_TORCH_PROFILER_WITH_STACK`: Operator call stack, of type bool. Includes call information at both the framework level and CPU operator level. Values: True: enabled. False: disabled, Default value. This setting takes effect when torch_npu.profiler.ProfilerActivity.CPU is enabled.

7. `VLLM_TORCH_PROFILER_WITH_FLOPS`: Floating-point operations of operators, of type bool (this parameter currently does not support parsing performance data). Values: True: enabled. False: disabled, Default value. This setting takes effect when torch_npu.profiler.ProfilerActivity.CPU is enabled.
Usage:
```bash
# old logic
# First, set the log directory like origin vllm
export VLLM_TORCH_PROFILER_DIR=./profiling
#(old logic) Then set how many step you need to run, it will stop automatically.You can set a large number to manually stop.
export PROFILER_STOP_STEP=5
# Do not set PROFILER_TOKEN_THRESHOLD if you want to use the new logic.
export PROFILER_TOKEN_THRESHOLD=1
```
Directly post requests to control the start and stop of profiles, consistent with the official VLLM interface.
```bash
# new logic
# First, set the log directory like origin vllm
export VLLM_TORCH_PROFILER_DIR=./profiling
# You can directly curl the service for which you want to enable the profile. The port is the API port of each node in the Ansible configuration file. You can also view the actual port in the log.
curl -X POST ip:port/start_profile
curl -X POST ip:port/stop_profile
```
Note that the original logic of vLLM generates a large number of profile files. Please stop it as soon as possible after startup; otherwise, it may lead to insufficient hard disk space.
To use the old logic and allow the profile to be automatically started, set the environment variable `PROFILER_TOKEN_THRESHOLD`.

# Profiling
目前有两种逻辑选项，且一次只能启用其中一种：
1. v0逻辑，当设置了 `PROFILER_TOKEN_THRESHOLD` 时触发，与 v0 版本一致。
2. vllm逻辑，当未设置 `PROFILER_TOKEN_THRESHOLD` 时触发，与 vLLM 版本一致。
在 Ray 模式下，v0逻辑无法获取性能分析数据。

主要的配置项是以下环境变量：
1. `VLLM_TORCH_PROFILER_DIR`：在环境中设置此变量，指定性能分析文件的存储位置。设置该变量后，可以通过 POST /start_profile 接口开始性能分析数据收集，并通过 POST /stop_profile 接口停止。如果未设置此变量，则不会启用性能分析。
这与官方 API 保持一致。
2. `PROFILER_STOP_STEP`：仅在v0逻辑（自动启动）下有效，此变量表示性能分析数据收集的步数。启动后，将其设置为你想要收集的步数。默认值为 5。达到指定步数后，性能分析将自动停止。
3. `PROFILER_TOKEN_THRESHOLD`：如果你想要使用vllm逻辑（手动启动），请不要设置此变量。如果设置了此变量，则默认启用v0逻辑。此变量表示步调度的 token 数量，作为性能分析数据收集的入口条件。你可以使用此值来区分不同阶段的数据收集。默认值为 1。
4. `VLLM_TORCH_PROFILER_RECORD_SHAPES`:算子的InputShapes和InputTypes，Bool类型。取值为：True：开启。False：关闭,默认值。开启torch_npu.profiler.ProfilerActivity.CPU时生效。
5. `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY`:算子的内存占用情况，Bool类型。取值为：True：开启。False：关闭,默认值。
6. `VLLM_TORCH_PROFILER_WITH_STACK`:算子调用栈，Bool类型。包括框架层及CPU算子层的调用信息。取值为：True：开启。False：关闭,默认值。开启torch_npu.profiler.ProfilerActivity.CPU时生效。
7  `VLLM_TORCH_PROFILER_WITH_FLOPS`:算子浮点操作，Bool类型（该参数暂不支持解析性能数据）。取值为：True：开启。False：关闭,默认值。开启torch_npu.profiler.ProfilerActivity.CPU时生效。

使用方法：
```bash
# v0逻辑
# 首先，设置日志目录，与原 vllm 相同
export VLLM_TORCH_PROFILER_DIR=./profiling
#（v0逻辑）然后设置需要运行多少步，它将自动停止。你可以设置一个较大的数字来手动停止。
export PROFILER_STOP_STEP=5
# 如果你想要使用vllm逻辑，请不要设置 PROFILER_TOKEN_THRESHOLD。
export PROFILER_TOKEN_THRESHOLD=1
```
通过直接发送请求来控制性能分析的启动和停止，与官方 VLLM 接口一致。
```bash
# vllm逻辑
# 首先，设置日志目录，与原 vllm 相同
export VLLM_TORCH_PROFILER_DIR=./profiling
# 直接对想要启用性能分析的服务发送 curl 请求。端口是 Ansible 配置文件中每个节点的 API 端口。你也可以在日志中查看实际端口。
curl -X POST ip:port/start_profile
curl -X POST ip:port/stop_profile
```
请注意，vLLM 的逻辑会生成大量profile文件。请在启动后尽快停止profile，否则可能导致硬盘空间不足。
如果你想要使用v0逻辑并允许性能分析自动启动，请设置环境变量 `PROFILER_TOKEN_THRESHOLD`。
