## Function-Level Profiling Patches

This folder provides a **non-intrusive way to apply function-level profiling**.

### Supported Profilers

1. **Marker** - Simply add something before/after target functions.
2. **Timer** – Basic time measurement for target functions.
3. **VizTracer** – Execution trace visualization using VizTracer.
4. **Torch-NPU** – Profiling via `torch_npu.profiler`.

### Enable Profiling

Set the corresponding environment variable to a YAML config file:

* `PROFILING_NAMELIST`

### Usage
* export PROFILING_NAMELIST=/path/to/namelist.yml
* Example yaml configs are in the [`assets/`](./assets) folder.
* export ROLE=prefill or export ROLE=decode

### Step to use omnilogger_namelist.yml for vllm tracing

Export the path to the namelist configuration:

```bash
export PROFILING_NAMELIST={project_root}/omni_infer/omni/tools/profiler/assets/omnilogger_namelist.yml
```

By default, logs are saved to `/tmp/trace_output_directory`.
To change this location, set the `TRACE_OUTPUT_DIRECTORY` environment variable:

```bash
export TRACE_OUTPUT_DIRECTORY=/your/custom/path
```
You can collect logs from multiple nodes by specifying them in a `server_list.txt` file, then running the provided script.

`server_list.txt`

```
10.11.123.1
10.11.123.2
10.11.123.3
10.11.123.4
```

 `collect_logs.sh`

```bash
#!/bin/bash
# Usage: ./collect_logs.sh server_list.txt /tmp/trace_output_directory  nginx_log_path your_log_directory

SERVER_LIST="$1"
REMOTE_FOLDER="$2"
PROXY_FOLDER="$3"
TARGET_FOLDER="$4"

mkdir $TARGET_FOLDER

for IP in $(cat "$SERVER_LIST"); do
    echo "Collecting logs from $IP..."
    scp -i key.pem -r "root@$IP:$REMOTE_FOLDER" "./logs_$IP"
    mv "./logs_$IP" $TARGET_FOLDER

    # copy logs in proxy
    if ssh -i key.pem root@$IP "test -f '$PROXY_FOLDER/nginx_error.log'"; then
        echo "nginx_error.log found on $IP, copying..."
        scp -i key.pem "root@$IP:$PROXY_FOLDER/nginx_error.log" "$TARGET_FOLDER/nginx_${IP}.log"
    fi
done
```

Once logs are collected, parse them using:

```bash
python parse_logs.py your_log_directory
```

Convert log to JSON that can be accepted by Jaeger
```bashs
python log_to_jaeger.py your_log_directory trace.json
```