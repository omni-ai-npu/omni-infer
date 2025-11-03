## Function-Level Profiling Patches

This folder provides a **non-intrusive way to apply function-level profiling**.

### Supported Profilers

1. **Marker** - Simply add something before/after target functions.

### Enable Profiling

Set the corresponding environment variable to a YAML config file:

* `PROFILING_NAMELIST`

### Usage
* export PROFILING_NAMELIST=/path/to/namelist.yml
* Example yaml configs are in the [`assets/`](./assets) folder.

### Step to use omnilogger_namelist.yml for vllm tracing

Export the path to the namelist configuration:

```bash
export PROFILING_NAMELIST={project_root}/omniinfer/omni/adaptors/sglang/patches/profiler_patches/assets/omnilogger_namelist.yml
export OMNILOGGER_ARGS="$TARCE_LOG_PATH,P{{ kv_rank }},01"
```
`/dev/shm` is a special directory in the Linux system, which is essentially a memory-based file system (usually of the tmpfs type). In many cases, writing data to `/dev/shm` is indeed faster than writing to a regular disk partition.
It is recommended to configure `TARCE_LOG_PATH` under `/dev/shm`. 
Once logs are collected, parse them using:

Add the following code at the end of {project_root}/omniinfer/infer_engines/sglang/python/sglang/srt/managers/scheduler.py 
```python
import os
if os.getenv("PROFILING_NAMELIST", None):
  print("Profiler patch environmental variable is enable, applying profiler patches.")
  from omni.adaptors.sglang.patches.profiler_patches import apply_profiler_patches
```

```bash
python omni_logger_print_parse_for_sglang.py trace_log "2025-09-08 15:47:49.185"
```
"2025-09-08 15:47:49.185" is the ramp-up time. If this time is not available, it can be set to "2025-01-01 00:00:00.000".
The trace_log collects the directories of trace json files from various machines. The json files should be placed directly in the trace_log directory and not in any subdirectories. 
