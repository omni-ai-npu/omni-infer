# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import csv
import re
import os
from collections import defaultdict
import sys
import argparse
import pandas as pd
import openpyxl
import traceback


_CHAT_REQUEST_ID_RE = re.compile(
    r"^(?P<base>chatcmpl-.+)-(?P<suffix>[0-9a-f]{8})$"
)
_COMPLETION_REQUEST_ID_RE = re.compile(
    r"^(?P<base>cmpl-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:-\d+)?(?:-[0-9a-f]{8})?$"
)


def _normalize_request_id(request_id):
    match = _COMPLETION_REQUEST_ID_RE.match(request_id)
    if match:
        return match.group("base")

    match = _CHAT_REQUEST_ID_RE.match(request_id)
    if match:
        return match.group("base")

    return request_id


def parse_trace_logs(log_dir, disable_encode=False):
    pattern = (
        r"<<<Action: (.*?); Timestamp:([\d.]+); RequestID:([a-z0-9-]+)(?:; Role:(\S+))?"
    )
    data_by_request = defaultdict(dict)
    request_role = defaultdict(dict)
    action_timestamps = {}
    prefill_engine_step_lines = []
    encode_engine_step_lines = []
    decode_engine_step_lines = []
    engine_core_info = {}

    time_analysis_path = os.path.join(log_dir, "time_analysis.xlsx")
    engine_step_path = os.path.join(log_dir, "engine_step.xlsx")
    try:
        log_files = []
        for dirpath, _, filenames in os.walk(log_dir):
            for filename in filenames:
                if filename.endswith(".log"):
                    log_files.append(os.path.join(dirpath, filename))

        for log_file_path in log_files:
            _collect_engine_core_info(engine_core_info, log_file_path)

        for log_file_path in log_files:
            dirpath, filename = os.path.split(log_file_path)
            _get_step_line(
                pattern,
                data_by_request,
                request_role,
                action_timestamps,
                prefill_engine_step_lines,
                encode_engine_step_lines,
                decode_engine_step_lines,
                engine_core_info,
                dirpath,
                filename,
                disable_encode,
            )

        # process time analysis
        if data_by_request:
            df_final = _get_final_df(
                data_by_request, request_role, disable_encode=disable_encode
            )

            with pd.ExcelWriter(time_analysis_path, engine="openpyxl") as writer:
                df_final.to_excel(writer, sheet_name="time_analysis", index=False)
                summary_data = {
                    "RequestID": list(data_by_request.keys()),
                    "ActionCount": [
                        len(actions) for actions in data_by_request.values()
                    ],
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)

            print(
                f"Successfully parsed time analysis files. Check {time_analysis_path}."
            )

        else:
            print("No valid action record found in any log files.")

        # Process engine_step_lines
        prefill_engine_step_headers = _get_prefill_engine_step_headers()
        encode_engine_step_headers = _get_encode_engine_step_headers()
        decode_engine_step_headers = _get_decode_engine_step_headers()
        with pd.ExcelWriter(engine_step_path, engine="openpyxl") as writer:
            if not disable_encode:
                _encode_engine_step_sheet(
                    encode_engine_step_lines,
                    engine_step_path,
                    writer,
                    encode_engine_step_headers,
                )

            _engine_step_sheet(
                prefill_engine_step_lines,
                engine_step_path,
                writer,
                prefill_engine_step_headers,
                "prefill_engine_step",
            )

            _decode_engine_step_sheet(
                decode_engine_step_lines,
                engine_step_path,
                writer,
                decode_engine_step_headers,
            )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.print_exc())


def _get_prefill_engine_step_headers():
    return [
        "node",
        "engine_step start",
        "engine_step end",
        "execute time(ms)",
        "running_reqs_num_after_step",
        "total_tokens",
        "waiting_reqs_num_after_step",
        "reqs_ids",
        "bs_tokens",
        "execute_model_start_time",
        "execute_model_end_time",
        "execute_model_cost_time(ms)",
        "kv_cache_usage",
        "kv_blocks_num",
        "start_free_block_num",
        "end_free_block_num",
        "cost_blocks_num",
        "engine_core_str",
    ]


def _get_encode_engine_step_headers():
    return [
        "node",
        "engine_step start",
        "engine_step end",
        "execute time(ms)",
        "running_reqs_num_after_step",
        "total_tokens",
        "waiting_reqs_num_after_step",
        "reqs_ids",
        "bs_tokens",
        "execute_model_start_time",
        "execute_model_end_time",
        "execute_model_cost_time(ms)",
        "engine_core_str",
        "main_model_start_time",
        "main_model_end_time",
        "execute_main_model_cost_time",
    ]


def _get_decode_engine_step_headers():
    return _get_prefill_engine_step_headers() + [
        "main_model_start_time",
        "main_model_end_time",
        "execute_main_model_cost_time",
        "mtp_model_start_time",
        "mtp_model_end_time",
        "execute_mtp_model_cost_time",
        "prefix",
    ]


_ENCODE_ACTION_KEYS = frozenset(
    {
        "Encoder api server get request",
        "Finish process request for encode engine",
        "Start process request in encode engine",
        "Encoder add waiting queue",
        "Encoder try to schedule in waiting queue",
        "Encoder start has_caches",
        "Encoder done has_caches",
        "Start append running sequence for encode",
        "Encoder start execute_model",
        "Encoder start _execute_mm_encoder",
        "Encoder start save_caches",
        "Encoder done save_caches",
        "Encoder done _execute_mm_encoder",
        "Encoder done execute_model",
        "Finish encode pickle and start response",
    }
)


def _get_final_df(data_by_request, request_role, disable_encode=False):
    full_map = _get_action_map()
    if disable_encode:
        action_map = {
            k: v for k, v in full_map.items() if k not in _ENCODE_ACTION_KEYS
        }
        node_cols = ["RequestID", "P_NODE", "D_NODE"]
    else:
        action_map = full_map
        node_cols = ["RequestID", "E_NODE", "P_NODE", "D_NODE"]
    fieldnames = node_cols + list(action_map.keys())
    data = []
    for request_id, actions in data_by_request.items():
        encode = request_role[request_id].get("encode")
        decode = request_role[request_id].get("decode")
        prefill = request_role[request_id].get("prefill")
        if decode is None or prefill is None:
            print(
                f'request_id: {request_role[request_id].get("request_id")} decode or prefill is None'
            )
            continue
        row = {"RequestID": request_id, "P_NODE": prefill, "D_NODE": decode}
        if not disable_encode:
            row["E_NODE"] = encode
        # Add timestamps for each action, "-" for missing actions
        for action in action_map.keys():
            row[action] = actions.get(action, "-")
        data.append(row)

    df = pd.DataFrame(data, columns=fieldnames)
    # chinese_row
    chinese_row = {col: "" for col in node_cols}
    chinese_row.update(action_map)
    df_cn = pd.DataFrame([chinese_row], columns=fieldnames)
    df_final = pd.concat([df.iloc[:0], df_cn, df.iloc[0:]], ignore_index=True)
    return df_final


def _get_step_line(
    pattern,
    data_by_request,
    request_role,
    action_timestamps,
    prefill_engine_step_lines,
    encode_engine_step_lines,
    decode_engine_step_lines,
    engine_core_info,
    dirpath,
    filename,
    disable_encode,
):
    if filename.endswith(".log"):
        log_file_path = os.path.join(dirpath, filename)
        print(f"Processing log file: {log_file_path}")
        try:
            with open(log_file_path, "r", encoding="latin1") as file:
                for line in file:
                    # for engine step
                    if "profile: " in line:
                        st_idx = line.find("profile:") + len("profile: ")
                        line = line[st_idx:]
                        node = line.split("|", 1)[0].strip()
                        if node.startswith("prefill_"):
                            prefill_engine_step_lines.append(line)
                        elif node.startswith("encode_"):
                            if not disable_encode:
                                _set_encode_info(
                                    encode_engine_step_lines, engine_core_info, line
                                )
                        elif node.startswith("decode_"):
                            _set_decode_info(
                                decode_engine_step_lines, engine_core_info, line
                            )
                        continue
                    # for time analysis
                    if "<<<Action" in line:
                        st_idx = line.find("<<<Action")
                        line = line[st_idx:]  # skip prefix if any
                        match = re.match(pattern, line.strip())
                        if match:
                            action, timestamp, request_id, role = match.groups()
                            request_id = _normalize_request_id(request_id)
                            role, ip = role.split("_")
                            action = action.strip()
                            timestamp = float(timestamp)
                            # min value
                            if (
                                action not in data_by_request[request_id]
                                or timestamp < data_by_request[request_id][action]
                            ):
                                data_by_request[request_id][action] = timestamp
                            request_role[request_id][role] = ip
                            if (
                                action not in action_timestamps
                                or timestamp < action_timestamps[action]
                            ):
                                action_timestamps[action] = timestamp
        except Exception as e:
            print(f"Error reading {log_file_path}: {str(e)}")
            print(traceback.print_exc())


def _collect_engine_core_info(engine_core_info, log_file_path):
    try:
        with open(log_file_path, "r", encoding="latin1") as file:
            for line in file:
                if "profile_mainmodel:" in line:
                    _get_main_model_info(engine_core_info, line)
                elif "profile_mtpmodel:" in line:
                    _get_mtp_model_info(engine_core_info, line)
    except Exception as e:
        print(f"Error reading {log_file_path}: {str(e)}")
        print(traceback.print_exc())


def _decode_engine_step_sheet(
    decode_engine_step_lines, engine_step_path, writer, decode_engine_step_headers
):
    if len(decode_engine_step_lines) != 0:
        decode_data = []
        decode_data_headers = decode_engine_step_headers[:-1]
        for line in decode_engine_step_lines:
            values = line.split("|")
            values[-1] = values[-1].split("=")[-1]
            row = dict(zip(decode_data_headers, values))
            decode_data.append(row)

        df_decode = pd.DataFrame(decode_data, columns=decode_engine_step_headers)
        df_decode["prefix"] = (
            df_decode["node"]
            + "_"
            + df_decode["engine_core_str"].str.extract(r"(\d+)", expand=False)
        )
        df_decode.to_excel(writer, sheet_name="decode_engine_step", index=False)

        print(
            f"Successfully parsed decode engine step logs. "
            f"Added 'decode_engine_step' sheet to {engine_step_path}."
        )

        # dump die load and die time
        _decode_die_load_sheet(engine_step_path, writer, df_decode)

    else:
        print("No valid decode engine step record found in log files.")


def _get_mtp_model_main_model_headers():
    return [
        "main_model_start_time",
        "main_model_end_time",
        "execute_main_model_cost_time",
        "mtp_model_start_time",
        "mtp_model_end_time",
        "execute_mtp_model_cost_time",
    ]


def _get_main_model_headers():
    return [
        "main_model_start_time",
        "main_model_end_time",
        "execute_main_model_cost_time",
    ]


def _engine_step_sheet(
    engine_step_lines, engine_step_path, writer, engine_step_headers, sheet_name
):
    if len(engine_step_lines) != 0:
        engine_data = []
        for line in engine_step_lines:
            values = line.split("|")
            values[-1] = values[-1].split("=")[-1]
            row = dict(zip(engine_step_headers, values))
            engine_data.append(row)

        df_engine = pd.DataFrame(engine_data, columns=engine_step_headers)
        df_engine.to_excel(writer, sheet_name=sheet_name, index=False)

        print(
            f"Successfully parsed {sheet_name} logs. Added '{sheet_name}' {engine_step_path}."
        )
    else:
        print(f"No valid {sheet_name} record found in log files.")


def _encode_engine_step_sheet(
    encode_engine_step_lines, engine_step_path, writer, engine_step_headers
):
    if len(encode_engine_step_lines) != 0:
        encode_data = []
        for line in encode_engine_step_lines:
            values = line.split("|")
            values[-1] = values[-1].split("=")[-1]
            row = dict(zip(engine_step_headers, values))
            encode_data.append(row)

        df_encode = pd.DataFrame(encode_data, columns=engine_step_headers)
        df_encode.to_excel(writer, sheet_name="encode_engine_step", index=False)

        print(
            f"Successfully parsed encode engine step logs. "
            f"Added 'encode_engine_step' sheet to {engine_step_path}."
        )
    else:
        print("No valid encode engine step record found in log files.")


def _set_encode_info(encode_engine_step_lines, engine_core_info, line):
    parts = line.strip().split("|")
    if len(parts) >= 18:
        core_str = parts[17].strip()
        info = engine_core_info.get(
            core_str,
            {
                "main_model_start_time": "",
                "main_model_end_time": "",
                "execute_main_model_cost_time": "",
            },
        )
        selected_parts = parts[:12] + [core_str]
        selected_parts.extend(
            [
                str(info.get("main_model_start_time", "")),
                str(info.get("main_model_end_time", "")),
                str(info.get("execute_main_model_cost_time", "")),
            ]
        )
        line = "|".join(selected_parts) + "\n"
    encode_engine_step_lines.append(line)
    return line


def _decode_die_load_sheet(engine_step_path, writer, df_decode):
    decode_die_load_columns = _get_decode_die_load_columns()
    grouped = df_decode.groupby("prefix")
    wide_blocks = []

    for prefix, group in grouped:
        group = group.reset_index(drop=True)
        filtered = group[decode_die_load_columns].copy()

        # Rename columns with prefix
        filtered.columns = [f"{prefix}_{col}" for col in filtered.columns]

        # Reset index for alignment and add to list
        wide_blocks.append(filtered.reset_index(drop=True))
    final_df = pd.concat(wide_blocks, axis=1)
    final_df.to_excel(writer, sheet_name="decode_die_load", index=False)
    print(
        f"Successfully parsed decode die load. "
        f"Added 'decode_die_load' sheet to {engine_step_path}."
    )


def _get_decode_die_load_columns():
    return [
        "execute_model_start_time",
        "total_tokens",
        "running_reqs_num_after_step",
        "waiting_reqs_num_after_step",
        "execute_model_cost_time(ms)",
        "start_free_block_num",
        "cost_blocks_num",
    ]


def _get_action_map() -> dict:
    return {
        "Encoder api server get request": "E：api server收到请求",
        "Finish process request for encode engine": "E：api server处理完请求并准备提交给engine",
        "Start process request in encode engine": "E：engine开始处理请求",
        "Encoder add waiting queue": "E：进入waiting队列",
        "Encoder try to schedule in waiting queue": "E：首次尝试加入running队列",
        "Encoder start has_caches": "E：has_caches开始",
        "Encoder done has_caches": "E：has_caches完成",
        "Start append running sequence for encode": "E：进入running队列",
        "Encoder start execute_model": "E：execute_model开始",
        "Encoder start _execute_mm_encoder": "E：_execute_mm_encoder开始",
        "Encoder start save_caches": "E：save_caches开始",
        "Encoder done save_caches": "E：save_caches结束",
        "Encoder done _execute_mm_encoder": "E：_execute_mm_encoder结束",
        "Encoder done execute_model": "E：execute_model结束",
        "Finish encode pickle and start response": "E：开始返回响应",
        "PD api server get request": "P：api server收到请求",
        "Get prefill request and start pickle": "P：api server开始处理请求",
        "Finish process request for prefill engine": "P：api server处理完请求并准备提交给engine",
        "Start process request in prefill engine": "P：engine开始处理请求",
        "Prefill add waiting queue": "P：进入waiting队列",
        "Prefill try to schedule in waiting queue": "P：首次尝试加入running队列",
        "Prefill start has_caches": "P：has_caches开始",
        "Prefill done has_caches": "P：has_caches结束",
        "Prefill fail to add result of kv insufficient": "P：首次kv不足加入失败",
        "Prefill get new_blocks": "P：kv分配完成",
        "Start append running sequence for prefill": "P：进入running队列",
        "Prefill start execute_model": "P：execute_model开始",
        "Prefill start execute main model": "P：main model开始",
        "Prefill start start_load_caches": "P：start_load_caches开始",
        "Prefill done start_load_caches": "P：start_load_caches结束",
        "Prefill start _execute_mm_encoder": "P：_execute_mm_encoder开始",
        "Prefill done _execute_mm_encoder": "P：_execute_mm_encoder结束",
        "Prefill start _gather_mm_embeddings": "P：_gather_mm_embeddings开始",
        "Prefill done _gather_mm_embeddings": "P：_gather_mm_embeddings结束",
        "Prefill start model forward": "P：模型前向开始",
        "Prefill done model forward": "P：模型前向结束",
        "Prefill done execute main model": "P：main model结束",
        "Prefill start execute mtp model": "P：mtp model开始",
        "Prefill done execute mtp model": "P：mtp model结束",
        "Prefill done execute_model": "P：execute_model结束",
        "Start to send output in prefill stage": "P：开始发送engine输出",
        "Client get prefill output": "P：client收到输出",
        "Pop output queues": "P：client出队处理",
        "Finish prefill pickle and start response": "P：api server收到请求并准备返回",
        "Enter decode to generate": "D：api server收到请求",
        "Finish process request for decode engine": "D：api server处理完请求并准备提交给engine",
        "Start to dispatch decode request": "D：engine开始分发请求",
        "Add need pulling sequence": "D：加入需要pulling队列",
        "Start pull kv": "D：开始pull kv",
        "Finish pull kv": "D：结束pull kv",
        "Prefill free kv blocks": "P侧释放kv（和前后列时间戳可能存在时钟误差）",
        "Start append running sequence for decode": "D：进入running队列",
        "Start to send output": "D：触发首个decode token执行",
        "First decode output token": "D：返回第一个token",
        "Second decode output token": "D：返回第二个token",
        "Third decode output token": "D：返回第三个token",
        "Finish decode pickle and start response": "D：api server收到推理结果",
    }


def _set_decode_info(decode_engine_step_lines, engine_core_info, line):
    core_match = re.search(r"(\d+-\d+\.\d+)", line)
    if core_match:
        core_str = core_match.group(1)
        info = engine_core_info.get(
            core_str,
            {
                "main_model_start_time": 0.0,
                "main_model_end_time": 0.0,
                "execute_main_model_cost_time": 0.0,
                "mtp_model_start_time": 0.0,
                "mtp_model_end_time": 0.0,
                "execute_mtp_model_cost_time": 0.0,
            },
        )
        line = (
            line.strip()
            + f"|{info.get('main_model_start_time')}|{info.get('main_model_end_time')}|{info.get('execute_main_model_cost_time')}|{info.get('mtp_model_start_time')}|{info.get('mtp_model_end_time')}|{info.get('execute_mtp_model_cost_time')}\n"
        )
    decode_engine_step_lines.append(line)
    return line


def _get_mtp_model_info(engine_core_info, line):
    parts = line.split("|")
    if len(parts) >= 5:
        core_str = parts[-1].strip()
        mtp_start = float(parts[1])
        mtp_end = float(parts[2])
        mtp_cost = float(parts[3])
        if core_str not in engine_core_info:
            engine_core_info[core_str] = {}
        engine_core_info[core_str].update(
            {
                "mtp_model_start_time": mtp_start,
                "mtp_model_end_time": mtp_end,
                "execute_mtp_model_cost_time": mtp_cost,
            }
        )


def _get_main_model_info(engine_core_info, line):
    parts = line.split("|")
    if len(parts) >= 5:
        core_str = parts[-1].strip()
        main_start = float(parts[1])
        main_end = float(parts[2])
        main_cost = float(parts[3])
        if core_str not in engine_core_info:
            engine_core_info[core_str] = {}
        engine_core_info[core_str].update(
            {
                "main_model_start_time": main_start,
                "main_model_end_time": main_end,
                "execute_main_model_cost_time": main_cost,
            }
        )


if __name__ == "__main__":
    # Usage: python parse_logs.py <trace_log_dir> [--disable-encode]
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--disable-encode", action="store_true")
    args = parser.parse_args()
    parse_trace_logs(args.log_dir, disable_encode=args.disable_encode)
