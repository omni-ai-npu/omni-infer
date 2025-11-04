import re
import os
import sys
import uuid
import json
from collections import defaultdict
import numpy as np
import argparse


ACTION_LIST = [
    "Start to schedule",                            #0
    "Time received",                                #1
    "Enter state tokenization",                     #2
    "Enter state APC matching",
    "Enter state P waiting",
    "Enter state P scheduled",
    "Enter state P running",                        #6

    "PD api server get request",                    #7
    "Start process request in prefill engine",
    "Prefill add waiting queue",
    "try to schedule in waiting queue",
    "Prefill get new_blocks",
    "success add to seq groups",
    "Prefill start execute_model",                  #13
    "Prefill done execute_model",
    "Start to send output in prefill stage",
    "Finish prefill pickle and start response",     #16

    "Finish P running",                             #17
    "Enter state D waiting",
    "Enter state D scheduled",
    "Enter state D running",                        #20

    "Enter decode to generate",                     #21
    "Start to dispatch decode request",             #22
    "Add need pulling sequence",          
    "Start pull kv",
    "Finish pull kv",                               #25
    "Start append running sequece for decode",    
    "Start to send output",
    "First decode output token",
    "Second decode output token",
    "Third decode output token",                    #30
    "Finish decode pickle and start response",      #31

    "Proxy got first token",                        #32
    "Prefill free kv blocks",
    "Proxy got second token",
    "Proxy got third token",                        #35
    "Received all tokens",                          #36
]

EXTRA_SPANS = [
    (0, 36, "Trace for Omni Proxy", "big_single"),   # root span
    (0, 32, "TTFT", "big_single"),
    (0, 16, "Prefill", "big_single"),
    (0, 1, "Proxy received request", "big_single"),
    (1, 2, "Process request", "big_single"),
    (2, 3, "Tokenizer", "big_single"),
    (3, 4, "APC matching", "big_single"),
    (4, 5, "P waiting", "big_single"),
    (5, 6, "P scheduled", "big_single"),
    (6, 16, "P running", "big_single"),
    (7, 12, "Add to running for prefill", "big"),
    (12, 14, "Prefill execute_model", "big"),
    (14, 16, "Send output in prefill stage", "big"),

    (16, 17, "Finish P running and return to proxy", "big_single"),
    (18, 19, "D waiting", "big_single"),
    (19, 20, "D scheduled", "big_single"),
    (20, 31, "D running", "big_single"),
    (21, 23, "Add to need pulling for decode", "big"),
    (23, 25, "Pull kv", "big"),
    (26, 31, "Send output in decode stage", "big"),

    (25, 33, "Prefill free kv", "special"),
    (32, 34, "TPOT1", "add"),
    (34, 35, "TPOT2", "add"),
    (35, 36, "Received all tokens", "add")
]

def normalize_reqid(reqid):
    if reqid.startswith("chatcmpl-"):
        return reqid[len("chatcmpl-"):]
    return reqid

def parse_log(log_dir):
    pattern = re.compile(
        r'<<<Action: (.*?); Timestamp:([\d.]+); RequestID:([a-z0-9-]+)(?:[;, ]|$)'
    )
    pattern_with_role = re.compile(
        r'<<<Action: (.*?); Timestamp:([\d.]+); RequestID:([a-z0-9-]+); Role:([a-zA-Z0-9]+)_(\d+\.\d+\.\d+\.\d+)'
    )
    host_pattern = re.compile(r'host: "([\d\.]+):\d+"')
    req_action_times = defaultdict(lambda: defaultdict(list))
    req_roles = defaultdict(lambda: defaultdict(tuple))
    for dirpath, _, filenames in os.walk(log_dir):
        for filename in filenames:
            if filename.endswith('.log'):
                with open(os.path.join(dirpath, filename), 'r', encoding='latin1') as f:
                    for line in f:
                        m = pattern_with_role.search(line)
                        if m:
                            action, ts, reqid, role, ip = m.groups()
                            reqid = normalize_reqid(reqid)
                            ts = float(ts)
                            action = action.strip()
                            req_action_times[reqid][action].append(ts)
                            if action not in req_roles[reqid]:
                                req_roles[reqid][action] = (role, ip)
                            continue
                        m2 = pattern.search(line)
                        if m2:
                            action, ts, reqid = m2.groups()
                            reqid = normalize_reqid(reqid)
                            ts = float(ts)
                            action = action.strip()
                            req_action_times[reqid][action].append(ts)
                            host_match = host_pattern.search(line)
                            if host_match:
                                host_ip = host_match.group(1)
                            if action not in req_roles[reqid]:
                                req_roles[reqid][action] = ("proxy", host_ip or "proxy")
    req_action_times = {k: v for k, v in req_action_times.items() if len(v) >= len(ACTION_LIST)-1}
    return req_action_times, req_roles

def build_spans(reqid, action_times, roles, print_avg_trace, TP_x, duration_dict, filter_flag):
    action_first_ts = []
    action_role_ip = []
    for i, action in enumerate(ACTION_LIST):
        if action in action_times:
            first_ts = min(action_times[action])
            role, ip = roles[action]
            action_first_ts.append(first_ts)
            action_role_ip.append((role, ip))
        else:
            action_first_ts.append(None)
            action_role_ip.append((None, None))
        # print(f"{i}: {action}, action_first_ts: {action_first_ts[i]}")


    # 1. generate all small_spans（(i,i+1), i in [6,16-1],[20,31-1]）
    small_spans = []
    for i in range(len(ACTION_LIST)):
        if 6 <= i <= 15 or 20 <= i <= 30:
            if action_first_ts[i] is not None and action_first_ts[i+1] is not None:
                small_spans.append({
                    "start_idx": i,
                    "end_idx": i+1,
                    "start_time": action_first_ts[i],
                    "end_time": action_first_ts[i+1],
                    "span_name": ACTION_LIST[i+1],
                    "role": action_role_ip[i+1][0],
                    "ip": action_role_ip[i+1][1],
                    "span_type": "small"
                })

    # 2. generate all big_spans
    big_spans = []
    for start, end, span_name, span_type in EXTRA_SPANS:
        if action_first_ts[start] is not None and action_first_ts[end] is not None:
            big_spans.append({
                "start_idx": start,
                "end_idx": end,
                "start_time": action_first_ts[start],
                "end_time": action_first_ts[end],
                "span_name": span_name,
                "role": action_role_ip[end][0] or "custom",
                "ip": action_role_ip[end][1] or "custom",
                "span_type": span_type
            })

    # 3. special parent-child relation in big_spans
    span_objs = []
    all_spans = big_spans + small_spans
    for i, s in enumerate(all_spans):
        s["span_id"] = str(uuid.uuid4())#1000 + i
        s["children"] = []
        span_objs.append(s)
    name2idx = {s["span_name"]: i for i, s in enumerate(span_objs)}
    idxmap = {(s.get("start_idx"), s.get("end_idx")): i for i, s in enumerate(span_objs)}
    idxmap_backup = idxmap

    parent_of_span = [None] * len(span_objs)
    # special big span（(21,23) --> Prefill free kv
    if "Prefill free kv" in name2idx and (21,23) in idxmap:
        parent_of_span[name2idx["Prefill free kv"]] = span_objs[idxmap[(21,23)]]["span_id"]

    # add child spans for "TTFT"
    chain_add = [(i, s) for i, s in enumerate(span_objs) if s["span_type"] == "add"]
    chain_add.sort(key=lambda x: (span_objs[x[0]]["start_time"]))
    chain_add = [x[0] for x in chain_add]
    if chain_add:
        parent_of_span[chain_add[0]] = span_objs[name2idx["TTFT"]]["span_id"]
        for idx in range(1, len(chain_add)):
            parent_of_span[chain_add[idx]] = span_objs[chain_add[idx - 1]]["span_id"]

    # 4. generate the main trace
    # fetch big_spans in big/big_single，Sort by start_time and interval span (higher priority).
    chain_spans = [(i, s) for i, s in enumerate(span_objs) if s["span_type"] in ("big", "big_single")]
    chain_spans.sort(key=lambda x: (span_objs[x[0]]["start_time"], -(span_objs[x[0]]["end_time"]-span_objs[x[0]]["start_time"])))
    chain_idxs = [x[0] for x in chain_spans]

    for idx in range(1, len(chain_idxs)):
        parent_of_span[chain_idxs[idx]] = span_objs[chain_idxs[idx-1]]["span_id"]
    # If a large span breaks, fill it with a small span.
    for idx in range(len(chain_idxs)-1):
        cur = chain_idxs[idx]
        nxt = chain_idxs[idx+1]
        # print(span_objs[cur]["span_name"],span_objs[nxt]["span_name"])
        i1, j1 = span_objs[cur]["start_idx"], span_objs[cur]["end_idx"]
        i2, j2 = span_objs[nxt]["start_idx"], span_objs[nxt]["end_idx"]
        fill = []
        if i2 > i1 and j2 <= j1:
            for k in range(i1, i2):
                key = (k, k+1)
                if key in idxmap:
                    fill.append(idxmap[key])
        elif j1 < i2:
            for k in range(j1, i2):
                if (k, k+1) in idxmap:
                    fill.append(idxmap[(k, k+1)])
        if fill:
            parent_of_span[fill[0]] = span_objs[cur]["span_id"]
            for m in range(1, len(fill)):
                parent_of_span[fill[m]] = span_objs[fill[m-1]]["span_id"]
            parent_of_span[nxt] = span_objs[fill[-1]]["span_id"]
        else:
            parent_of_span[nxt] = span_objs[cur]["span_id"]

    # 5.  Big span hangs child chain
    for i, s in enumerate(span_objs):
        if s["span_type"] in ("big"):
            i0, j0 = s["start_idx"], s["end_idx"]
            chain = [idxmap_backup[(k, k+1)] for k in range(i0, j0) if (k, k+1) in idxmap_backup]
            if chain:
                parent_of_span[chain[0]] = s["span_id"]
                for m in range(1, len(chain)):
                    parent_of_span[chain[m]] = span_objs[chain[m-1]]["span_id"]  

    # 6. output
    spans_out = []
    for i, s in enumerate(span_objs):
        if s["start_time"] is None or s["end_time"] is None:
            continue
        # print( reqid, "-->", s.get("start_idx"), s.get("end_idx"), s.get("span_name"), s.get("span_type"))
        key = (s.get("start_idx"), s.get("end_idx"), s.get("span_name"), s.get("span_type"))
        if key == EXTRA_SPANS[0]:
            operation_name = reqid
        elif key[2] == "PD api server get request":
            operation_name = "Send from Omni proxy to P nodes"
        elif key == EXTRA_SPANS[13]:
            operation_name = "Send from P node to Omni proxy"
        elif key[2] == "Enter decode to generate":
            operation_name = "Send from Omni proxy to D node"
        else:
            operation_name = " " 
        tags = [
            {"key": "RequestID", "type": "string", "value": reqid},
            {"key": "role", "type": "string", "value": s["role"]},
            {"key": "ip", "type": "string", "value": s["ip"]},
            {"key": "start_action", "type": "string", "value": ACTION_LIST[s.get("start_idx")]},
            {"key": "end_action", "type": "string", "value": ACTION_LIST[s.get("end_idx")]},
            {"key": "start_time", "type": "float", "value": s["start_time"]},
            {"key": "end_time", "type": "float", "value": s["end_time"]}
        ]
        span_json = {
            "traceID": "TBD",
            "spanID": str(s["span_id"]),
            "operationName": operation_name,
            "references": [],
            "startTime": int(s["start_time"] * 1e6),
            "duration": int((s["end_time"] - s["start_time"]) * 1e6),
            "tags": tags,
            "processID": s["span_name"],
            "logs": [
                {"timestamp": int(s["start_time"] * 1e6), "fields": []}
            ]
        }
        if parent_of_span[i] is not None:
            span_json["references"].append({
                "refType": "CHILD_OF",
                "traceID": "TBD",
                "spanID": str(parent_of_span[i])
            })
        spans_out.append(span_json)
    
    # --- summary trace added: duration collection ---
    if filter_flag:
        for i, s in enumerate(span_objs):       
            key = (s.get("start_idx"), s.get("end_idx"), s.get("span_name"), s.get("span_type"))
            if s["start_time"] is None or s["end_time"] is None:
                continue
            duration_us = int((s["end_time"] - s["start_time"]) * 1e6)
            duration_dict[key].append(duration_us)
    
    if print_avg_trace:
        spans_avg = []
        for i, s in enumerate(span_objs):
            key = (s.get("start_idx"), s.get("end_idx"), s.get("span_name"), s.get("span_type"))
            durations = duration_dict.get(key, [])
            avg_duration = sum(durations) / len(durations) if durations else 0

            if key == EXTRA_SPANS[0]:
                processID_name = "Request Average Trace"
            else:
                processID_name = s.get("span_name")
            if key[2] == "PD api server get request":
                operation_name = "Send from Omni proxy to P nodes"
            elif key == EXTRA_SPANS[13]:
                operation_name = "Send from P node to Omni proxy"
            elif key[2] == "Enter decode to generate":
                operation_name = "Send from Omni proxy to D node"
            else:
                operation_name = " " 
            avg_tags = [
                {"key": "RequestID", "type": "string", "value": "average_all_requests"},
                {"key": "role", "type": "string", "value": s["role"]},
                {"key": "start_action", "type": "string", "value": ACTION_LIST[s.get("start_idx")]},
                {"key": "end_action", "type": "string", "value": ACTION_LIST[s.get("end_idx")]},
                {"key": "average time duration", "type": "float", "value": avg_duration}
            ]
            avg_span_json = {
                "traceID": "TBD",
                "spanID": str(s["span_id"]),
                "operationName": operation_name,
                "references": [],
                "startTime": int(s["start_time"] * 1e6),
                "duration": round(avg_duration),
                "tags": avg_tags,
                "processID": processID_name,
                "logs": [
                    {"timestamp": "Average time duration", "fields": []}
                ]
            }
            if parent_of_span[i] is not None:
                avg_span_json["references"].append({
                    "refType": "CHILD_OF",
                    "traceID": "TBD",
                    "spanID": str(parent_of_span[i])
                })
            spans_avg.append(avg_span_json)

        spans_TP_x_dict = {}
        for x in TP_x:
            spans_TP_x_dict[x] =[]
            for i, s in enumerate(span_objs):
                key = (s.get("start_idx"), s.get("end_idx"), s.get("span_name"), s.get("span_type"))
                durations = duration_dict.get(key, [])
                arr = np.array(durations)
                tpx_duration = np.percentile(arr, x)

                if key == EXTRA_SPANS[0]:
                    processID_name = f"Request TP{x} Percentile Trace"
                else:
                    processID_name = s.get("span_name")
                if key[2] == "PD api server get request":
                    operation_name = "Send from Omni proxy to P nodes"
                elif key == EXTRA_SPANS[13]:
                    operation_name = "Send from P node to Omni proxy"
                elif key[2] == "Enter decode to generate":
                    operation_name = "Send from Omni proxy to D node"
                else:
                    operation_name = " "
                TPx_tags = [
                    {"key": "RequestID", "type": "string", "value": "average_all_requests"},
                    {"key": "role", "type": "string", "value": s["role"]},
                    {"key": "start_action", "type": "string", "value": ACTION_LIST[s.get("start_idx")]},
                    {"key": "end_action", "type": "string", "value": ACTION_LIST[s.get("end_idx")]},
                    {"key": "average time duration", "type": "float", "value": tpx_duration}
                ]
                TPx_span_json = {
                    "traceID": "TBD",
                    "spanID": str(s["span_id"]),
                    "operationName": operation_name,
                    "references": [],
                    "startTime": int(s["start_time"] * 1e6),
                    "duration": round(tpx_duration),
                    "tags": TPx_tags,
                    "processID": processID_name,
                    "logs": [
                        {"timestamp": f"TP{x} percentile time duration", "fields": []}
                    ]
                }
                if parent_of_span[i] is not None:
                    TPx_span_json["references"].append({
                        "refType": "CHILD_OF",
                        "traceID": "TBD",
                        "spanID": str(parent_of_span[i])
                    })
                spans_TP_x_dict[x].append(TPx_span_json)

        return spans_avg, spans_TP_x_dict

    return spans_out, None

def build_jaeger_trace(reqid, spans):
    trace_id = uuid.uuid4().hex
    for s in spans:
        s["traceID"] = trace_id
        for ref in s["references"]:
            ref["traceID"] = trace_id
    processes = {}
    for s in spans:
        svc = s["processID"]
        if svc not in processes:
            processes[svc] = {
                "serviceName": svc,
                "tags": [
                    {"key": "service", "type": "string", "value": svc}
                ]
            }
    return {
        "traceID": trace_id,
        "spans": spans,
        "processes": processes
    }

def main(log_dir, output_json, select_reqids_set, num):
    req_action_times, req_roles = parse_log(log_dir)
    traces = []
    duration_dict = defaultdict(list)
    duration_dict_none = defaultdict(list)
    TP_x = (90, 95, 99)
    skip_indx_before = num                           # Skip all indices before this value 
    ship_indx_after =len(req_action_times)-1-num     # Skip all indices after this value

    #--- filter only selected reqids---
    selected_reqids = list(req_action_times.keys())
    print(f"Total requests with indices [0, {len(req_action_times)-1}]; Shown selected request id: {select_reqids_set}")
    if select_reqids_set:
        selected_reqids = [reqid for reqid in req_action_times if reqid in select_reqids_set]
        if not selected_reqids:
            print("No matching request ids found in log.")
            return
        else:
            for reqid in selected_reqids:
                spans, _ = build_spans(reqid, req_action_times[reqid], req_roles[reqid], False, TP_x, duration_dict_none, False)
                if spans:
                    traces.append(build_jaeger_trace(reqid, spans))
    for indx, reqid in enumerate(req_action_times):
        if indx ==len(req_action_times)-2:  #use the last second request (with "Prefill free kv" span) as the req_template
            req_template = reqid
            # print("req_template id is:", reqid)
        if indx >= skip_indx_before and indx <= ship_indx_after:
            filter_flag = True              # for calculting TP_x
        else:
            filter_flag = False
        spans,_ = build_spans(reqid, req_action_times[reqid], req_roles[reqid], False, TP_x, duration_dict, filter_flag)
        
        if select_reqids_set is None:
            traces.append(build_jaeger_trace(reqid, spans))

    # print_avg_trace=TRUE ---> print statistical information of traces, reqid index in [skip_indx_before, ship_indx_after]
    print(f"Exclude the first  {num} requests and last {num} requests; Analyzing traces statistics for requests with indices [{skip_indx_before}, {ship_indx_after}]")
    (avg_spans, spans_TP_x_dict) = build_spans(req_template, req_action_times[req_template], req_roles[req_template], True, TP_x, duration_dict, False)
    traces.insert(0, build_jaeger_trace(req_template, avg_spans))
    for x in TP_x:
        traces.insert(0, build_jaeger_trace(req_template, spans_TP_x_dict[x]))

    jaeger_data = {"data": traces}
    with open(output_json, "w") as f:
        json.dump(jaeger_data, f, indent=2)
    print(f"Done. Output to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="python log_to_jaeger.py /path/to/log_dir trace.json [--skip-req_num num --select-reqids reqid1,reqid2,...]")
    parser.add_argument("log_dir")
    parser.add_argument("output_json")
    parser.add_argument("--skip-req_num", type=int, default="1")
    parser.add_argument("--select-reqids", type=str, default="")
    args = parser.parse_args()
    if args.select_reqids:
        select_reqids_set = set(args.select_reqids.split(","))
    else:
        select_reqids_set = None
    num = args.skip_req_num
    main(args.log_dir, args.output_json, select_reqids_set, num)