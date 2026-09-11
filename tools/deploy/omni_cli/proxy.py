# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import shlex

def register_values(inventory):
    # calculate PREFILL_API_SERVER_LIST
    prefill_api_server_list = []
    
    for host, vars in inventory['all']['children']['P']['hosts'].items():
        ansible_host_val = vars.get('ansible_host', '')
        host_ip_val = vars.get('host_ip', '')
        api_port_val = vars.get('env', '').get('API_PORT', '')
        
        if ansible_host_val and host_ip_val and ansible_host_val == host_ip_val:
            entry = f"{ansible_host_val}:{api_port_val}"
            if entry not in prefill_api_server_list:
                prefill_api_server_list.append(entry)

    prefill_api_server_list_result = ','.join(prefill_api_server_list)

    # calculate DECODE_API_SERVER_LIST
    decode_api_server_list = []
    
    for host, vars in inventory['all']['children']['D']['hosts'].items():
        ip = vars.get('ansible_host', '')
        api_port_val = vars.get('env', {}).get('API_PORT', '')

        tp_str = vars.get('args', {}).get('tp', '1')
        try:
            tp = int(tp_str)
        except (ValueError, TypeError):
            tp = 1

        devices_str = vars.get('ascend_rt_visible_devices', '')
        device_count = devices_str.count(',') + 1

        num = int(device_count / tp)
        
        if ip:
            entry = f"{ip}:{api_port_val}@{num}"
            if entry not in decode_api_server_list:
                decode_api_server_list.append(entry)

    decode_api_server_list_result = ','.join(decode_api_server_list)

    return {
        'PREFILL_API_SERVER_LIST': prefill_api_server_list_result,
        'DECODE_API_SERVER_LIST': decode_api_server_list_result
    }

def process_results(results, inventory, inv_file, cloud_mode=False):
    prefill_result = results['PREFILL_API_SERVER_LIST']
    decode_api_servers = results['DECODE_API_SERVER_LIST']

    prefill_result = ' '.join(prefill_result.split())

    decode_result = ""

    decode_array = decode_api_servers.split(',')

    for var in decode_array:
        address = var.split('@')[0]
        num = int(var.split('@')[1])

        ip = address.split(':')[0]
        port = int(address.split(':')[1])

        for i in range(num):
            if not decode_result:
                decode_result = f"{ip}:{port}"
            else:
                decode_result += f",{ip}:{port}"
            port += 1

    for host, vars in inventory['all']['children']['C']['hosts'].items():
        api_port_val = vars.get('env', '').get('API_PORT', '')
        container_name = vars.get('container_name', '')

        env: Dict[str, Any] = vars.get("env", {}) or {}
        log_path = str(env.get("LOG_PATH") or "").strip()

        args: Dict[str, Any] = vars.get("args", {}) or {}
        prefill_lb_sdk = args.get('prefill-lb-sdk', 'pd_score_balance')
        decode_lb_sdk = args.get('decode-lb-sdk', 'pd_score_balance')

        # use_omni_proxy
        use_omni_proxy = vars.get('use_omni_proxy', False)
        log_level = vars.get('env', '').get('LOG_LEVEL', 'notice')
        access_log_file = vars.get('env', '').get('ACCESS_LOG_FILE', '/tmp/nginx_access.log')
        core_num = vars.get('env', '').get('CORE_NUM', 4)
        start_core_index = vars.get('env', '').get('START_CORE_INDEX', 16)
        omni_proxy_pd_policy = vars.get('env', '').get('OMNI_PROXY_PD_POLICY', 'sequential')
        model_path = vars.get('env', '').get('MODEL_PATH', '')
        omni_proxy_max_batch_num_token = vars.get('env', '').get('OMNI_PROXY_MAX_BATCH_NUM_TOKEN', 100000)
        omni_proxy_prefill_max_num_seqs = vars.get('env', '').get('OMNI_PROXY_PREFILL_MAX_NUM_SEQS', 32)
        omni_proxy_decode_max_num_seqs = vars.get('env', '').get('OMNI_PROXY_DECODE_MAX_NUM_SEQS', 48)
        stream_ops = vars.get('env', '').get('STREAM_OPS', 'off')
        omni_proxy_schedule_algo = vars.get('env', '').get('OMNI_PROXY_SCHEDULE_ALGO', 'default')
        omni_proxy_prefill_starvation_timeout = vars.get('env', '').get('OMNI_PROXY_PREFILL_STARVATION_TIMEOUT', 400)
        export_block = _build_export_block(vars.get('env', ''))

    prefill_max_num_seqs = 16
    decode_max_num_seqs = 32
    for _, vars in inventory['all']['children']['P']['hosts'].items():
        args: Dict[str, Any] = vars.get("args", {}) or {}
        prefill_max_num_seqs = args.get('extra-args', {}).get('max-num-seqs', prefill_max_num_seqs)

    for _, vars in inventory['all']['children']['D']['hosts'].items():
        args: Dict[str, Any] = vars.get("args", {}) or {}
        decode_max_num_seqs = args.get('extra-args', {}).get('max-num-seqs', decode_max_num_seqs)

    file_name = f"./omni_start_c.sh"
    if use_omni_proxy:
        with open(file_name, "w") as tf:
            script_path = Path(tf.name)
            tf.write("#!/usr/bin/env bash\n")
            if not cloud_mode:
                tf.write(f"docker exec -i {shlex.quote(container_name)} bash -s <<'EOF'\n")
            tf.write("source ~/.bashrc\n\n")
            tf.write("# Export environment variables\n")
            tf.write(export_block + "\n\n")
            tf.write(f'echo "{export_block}\n" > {log_path}/omni_cli.log\n\n')
            tf.write(f"ps aux | grep 'nginx' | grep -v 'grep' | awk '{{print $2}}' | xargs kill -9; cd /workspace/omniinfer/components/omni-proxy/omni_proxy/; bash omni_proxy.sh \\\n\
              --listen-port {api_port_val} \\\n\
              --prefill-endpoints {prefill_result} \\\n\
              --decode-endpoints {decode_result} \\\n\
              --log-file {log_path}/nginx/nginx_error.log \\\n\
              --log-level {log_level} \\\n\
              --access-log-file {access_log_file} \\\n\
              --core-num {core_num} \\\n\
              --start-core-index {start_core_index} \\\n\
              --stream-ops {stream_ops} \\\n\
              --omni-proxy-schedule-algo {omni_proxy_schedule_algo} \\\n\
              --omni-proxy-prefill-starvation-timeout {omni_proxy_prefill_starvation_timeout} \\\n\
              --omni-proxy-pd-policy {omni_proxy_pd_policy} \\\n\
              --omni-proxy-model-path {model_path} \\\n\
              --omni-proxy-max-batch-num-token {omni_proxy_max_batch_num_token} \\\n\
              --omni-proxy-prefill-max-num-seqs {omni_proxy_prefill_max_num_seqs} \\\n\
              --omni-proxy-decode-max-num-seqs {omni_proxy_decode_max_num_seqs}\n\n")
            if not cloud_mode:
                tf.write("EOF\n")
    else:
        with open(file_name, "w") as tf:
            script_path = Path(tf.name)
            tf.write("#!/usr/bin/env bash\n")
            if not cloud_mode:
                tf.write(f"docker exec -i {shlex.quote(container_name)} bash -s <<'EOF'\n")
            tf.write(f"ps aux | grep 'nginx' | grep -v 'grep' | awk '{{print $2}}' | xargs kill -9; cd /workspace/omniinfer/components/omni-proxy/omni_proxy/; bash global_proxy.sh \\\n\
              --listen-port {api_port_val} \\\n\
              --prefill-servers-list {prefill_result} \\\n\
              --decode-servers-list {decode_result} \\\n\
              --log-file {log_path}/nginx/nginx_error.log \\\n\
              --log-level notice \\\n\
              --core-num 4 \\\n\
              --start-core-index 16 \\\n\
              --prefill-max-num-seqs {prefill_max_num_seqs} \\\n\
              --decode-max-num-seqs {decode_max_num_seqs} \\\n\
              --prefill-lb-sdk {prefill_lb_sdk} \\\n\
              --decode-lb-sdk {decode_lb_sdk}\n\n")
            if not cloud_mode:
                tf.write("EOF\n")

    os.chmod(script_path, 0o755)

    if cloud_mode:
        print(f"[INFO] Generated cloud proxy script: {script_path}")
        return

    cmd = (
        f"ansible {shlex.quote(host)} "
        f"-i {shlex.quote(str(inv_file))} "
        f"-m script "
        f"-a {shlex.quote(str(script_path))}"
    )

    try:
        os.system(cmd)
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass


def _build_export_block(env: Dict[str, Any]) -> str:
    """Build a sequence of export lines."""
    items_plain = [(k, v) for k, v in env.items() if "$" not in str(v)]
    items_refs = [(k, v) for k, v in env.items() if "$" in str(v)]

    lines = []
    for k, v in items_plain + items_refs:
        if v is None:
            v = ""
        lines.append(f'export {k}={_double_quotes(v)}')
    return "\n".join(lines)


def _double_quotes(s: str) -> str:
    """Wrap value in double quotes for a safe shell arg."""
    s = str(s)
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{s}"'


def omni_run_proxy(inventory, cloud_mode=False):
    inv_file = Path(inventory).expanduser().resolve()
    inv = None
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)
    
    result = register_values(inv)
    process_results(result, inv, inv_file, cloud_mode=cloud_mode)
