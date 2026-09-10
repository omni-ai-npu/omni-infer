# -*- coding: utf-8 -*-
"""
用法:
    python convert_body.py <源JSON> [输出JSON] [--fix-unit-names]
                           [--allow-unknown-model-address]
    输出默认: 与源文件同目录, 文件名后缀 "-corrected.json"
    也可用环境变量 MODELARTS_BODY_OUTPUT 指定输出路径(优先级低于命令行第2参数)

其他环境变量:
    MODELARTS_WORKSPACE_ID(写入 workspace_id), MODELARTS_MODEL_SOURCE,
    MODELARTS_MODEL_ADDRESS
"""
import copy
import json
import os
import re
import sys

WORKSPACE_ID = os.environ.get(
    "MODELARTS_WORKSPACE_ID", "1d670a16e0b34acbb27975dfec4d3790")
# OBS 挂载所需认证凭证(密钥名/类型); 快照缺省时按此补齐, 否则报 ModelArts.8162
SECRET_NAME = os.environ.get("MODELARTS_SECRET_NAME",
                             "aifm-infra-infer_MA_secret")
SECRET_TYPE = os.environ.get("MODELARTS_SECRET_TYPE", "dew")
DEFAULT_TYPE = "REAL_TIME"
DEFAULT_VERSION = "1.0.0"

# ---------------- API Explorer schema 白名单 ----------------
ROOT_ALLOWED = {
    "name", "version", "description", "type", "deploy_type", "group_configs",
    "runtime_config", "upgrade_config", "lts_strategy", "log_configs", "tags",
    "workspace_id", "schedule", "custom_metrics_path",
    "deploy_timeout_minutes", "task_type", "workload_type",
}
GROUP_ALLOWED = {
    "id", "name", "pool_id", "count", "system_log_dump_enable",
    "unit_configs", "weight", "secret_type", "secret_name", "priority",
    "high_avail_switch", "schedule_strategy", "version", "version_id",
    "description", "framework", "running_count", "deploy_type",
    "mirror_traffic_enable", "mirror_traffic_weight", "version_count",
    "workload_type", "update_at", "model", "advanced_config",
}
UNIT_ALLOWED = {
    "id", "name", "role", "custom_spec", "flavor", "flavor_display_name",
    "image", "models", "codes", "files", "dumps", "count", "cmd",
    "termination_grace", "envs", "readiness_health", "startup_health",
    "liveness_health", "port", "recovery", "npu_reset_enable", "group_count",
    "affinity", "security_config", "pool_resource_flavor",
}
MODEL_ALLOWED = {
    "source", "address", "mount_path", "host_cache", "efs_sub_path",
    "read_only", "os_warm_up", "source_name", "asset_id",
}
# 快照里会出现在顶层、但属于"组"的字段(需下沉到 group_configs[0])
GROUP_FROM_ROOT = {
    "pool_id", "count", "system_log_dump_enable", "weight", "secret_type",
    "secret_name", "priority", "high_avail_switch", "schedule_strategy",
    "framework", "mirror_traffic_enable", "mirror_traffic_weight",
    "deploy_type", "description", "workload_type", "model", "advanced_config",
    "unit_configs",
}
# 响应侧字段(绝不透传)
RESPONSE_ONLY = {
    "id", "infer_name", "status", "create_at", "update_at", "user_name",
    "version_id", "version_count", "running_count", "dispatcher_group_id",
    "failure_reason", "predict_url", "lts_state", "lts_event_state",
    "lts_status", "lts_event_status", "lts_file_status",
}


def _as_int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _envs_to_str(envs):
    """envs 必须是 string->string; 非字符串值统一转字符串。"""
    out = {}
    for k, v in (envs or {}).items():
        if v is None:
            continue
        out[str(k)] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return out


# ---------------------------------------------------------------- model 归一化
def normalize_model(model, units, service_name, report, allow_unknown=False):
    """归一化 model 块(见模块 docstring 第 2 点)。无法安全归一化则抛错。"""
    m = {k: copy.deepcopy(v) for k, v in (model or {}).items()
         if k in MODEL_ALLOWED}
    if not m:
        return None

    override_src = os.environ.get("MODELARTS_MODEL_SOURCE")
    override_addr = os.environ.get("MODELARTS_MODEL_ADDRESS")
    if override_src:
        m["source"] = override_src
    if override_addr:
        m["address"] = override_addr
        report.append("model.address 由环境变量指定: %s" % override_addr)

    addr = str(m.get("address") or "")
    if not addr.startswith(("obs://", "s3://")):
        model_path = ""
        for u in units:
            mp = (u.get("envs") or {}).get("MODEL_PATH")
            if mp:
                model_path = str(mp)
                break
        mt = re.match(r"^/mnt/(bucket-[^/]+)/(.+)$", model_path)
        if mt:
            m["source"] = "OBS"
            m["address"] = "obs://%s/%s" % (mt.group(1), mt.group(2))
            report.append(
                "model: EFS/缺省地址 → OBS 形式 address=%s (由 MODEL_PATH=%s 推导)"
                % (m["address"], model_path))
        elif allow_unknown:
            report.append(
                "警告: 无法推导 model.address(原值=%r, MODEL_PATH=%r), "
                "已按原样保留" % (addr, model_path))
        else:
            raise ValueError(
                "无法确定 model.address: 原值=%r, MODEL_PATH=%r。\n"
                "请用 MODELARTS_MODEL_ADDRESS 环境变量显式指定, 或加 "
                "--allow-unknown-model-address 保留原样。" % (addr, model_path))

    m["mount_path"] = "/mnt"
    m.setdefault("efs_sub_path", "/")
    m["read_only"] = bool(m.get("read_only", True))
    m.setdefault("host_cache", False)
    m.setdefault("os_warm_up", False)
    return m


# ---------------------------------------------------------------- 主转换
def normalize_create_body(raw, report=None, allow_unknown_model=False,
                          fix_unit_names=False):
    report = report if report is not None else []

    if "group_configs" in raw:
        body = copy.deepcopy(raw)
        body["workspace_id"] = body.get("workspace_id") or WORKSPACE_ID
        report.append("输入已是规范体(group_configs), 仅做幂等清洗")
        return body

    dropped = set()

    # ---- units ----
    units = []
    seen_names = {}
    for u in raw.get("unit_configs", []) or []:
        nu = {}
        for k, v in u.items():
            if k in RESPONSE_ONLY or k not in UNIT_ALLOWED:
                dropped.add("unit.%s" % k)
                continue
            if k == "id":
                dropped.add("unit.id")
                continue
            nu[k] = copy.deepcopy(v)
        if "envs" in nu:
            nu["envs"] = _envs_to_str(nu["envs"])
        if "count" in nu:
            nu["count"] = _as_int(nu["count"], 1)
        if "group_count" in nu:
            nu["group_count"] = _as_int(nu["group_count"], 1)
        if "npu_reset_enable" in nu:
            nu["npu_reset_enable"] = bool(nu["npu_reset_enable"])
        name = nu.get("name", "")
        if name in seen_names:
            if fix_unit_names:
                base = name
                mm = re.match(r"^(.*?)-(\d+)$", name)
                if mm:
                    base = mm.group(1)
                idx = 1
                while ("%s-%d" % (base, idx)) in seen_names:
                    idx += 1
                nu["name"] = "%s-%d" % (base, idx)
                report.append("unit 重名自动修正: %s → %s" % (name, nu["name"]))
            else:
                report.append(
                    "警告: unit 名称重复 %r —— 可能导致创建/调度异常; "
                    "可加 --fix-unit-names 自动改名" % name)
        seen_names[nu.get("name", "")] = True
        units.append(nu)

    # ---- group ----
    group = {}
    for k, v in raw.items():
        if k in GROUP_FROM_ROOT:
            group[k] = copy.deepcopy(v)
        elif k in RESPONSE_ONLY or k not in ROOT_ALLOWED:
            dropped.add(k)
    group["name"] = "deploy-%s" % raw.get("name", "service")
    group["unit_configs"] = units
    if "model" in group:
        group["model"] = normalize_model(group["model"], units,
                                         raw.get("name", ""), report,
                                         allow_unknown_model)
    if "count" in group:
        group["count"] = _as_int(group["count"], 1)
    if "weight" in group:
        group["weight"] = _as_int(group["weight"], 100)
    if "priority" in group:
        group["priority"] = _as_int(group["priority"], 3)
    if "system_log_dump_enable" in group:
        group["system_log_dump_enable"] = bool(group["system_log_dump_enable"])
    # OBS 挂载必须提供认证凭证(secret_name/secret_type), 否则报 ModelArts.8162
    if group.get("model") and group["model"].get("source") == "OBS":
        if not group.get("secret_name"):
            group["secret_name"] = SECRET_NAME
            report.append("model 为 OBS 挂载: 快照缺少 secret_name, 已补默认值 %s"
                          % SECRET_NAME)
        group.setdefault("secret_type", SECRET_TYPE)

    # ---- root ----
    body = {
        "name": raw.get("name", ""),
        "version": DEFAULT_VERSION,
        "description": raw.get("description", ""),
        "type": raw.get("type") or DEFAULT_TYPE,
        "deploy_type": raw.get("deploy_type", "MULTI"),
        "group_configs": [group],
        "workspace_id": WORKSPACE_ID,
        "workload_type": raw.get("workload_type", "LWS"),
        "runtime_config": {
            "service_invoke": {
                "port": 7000, "protocol": "HTTPS", "auth_type": "API_KEY",
                "dynamic_routing_enable": False, "ems_enable": False,
                "internet_access_enable": True,
                "intranet_approval_enable": True,
                "request_retry_enable": False,
            },
            "service_limit": {
                "rate_limit": {"num": 200, "unit": "SECONDS"},
                "request_size_limit": 20, "request_timeout": 30,
                "ip_white_list": [], "ip_black_list": [],
            },
            "service_secret": {
                "group_enable": False, "secret_enable": False,
                "secret_volumes": [],
            },
        },
    }
    for k in ("task_type", "deploy_timeout_minutes", "tags", "schedule",
              "custom_metrics_path", "lts_strategy", "log_configs",
              "upgrade_config"):
        if raw.get(k) not in (None, {}, []):
            body[k] = copy.deepcopy(raw[k])

    if dropped:
        report.append("剔除字段(响应侧/非 schema): %s"
                      % ", ".join(sorted(dropped)))
    for k in ("high_avail_switch", "secret_name"):
        if k not in group:
            report.append("提示: 组内缺少 %s(快照原本没有), 已按缺省提交" % k)
    return body


def check_body(body):
    problems = []
    for k in ("name", "type", "deploy_type", "group_configs", "workspace_id",
              "runtime_config"):
        if not body.get(k):
            problems.append("顶层缺少必填字段 %s" % k)
    for i, g in enumerate(body.get("group_configs", [])):
        for k in ("name", "pool_id", "unit_configs"):
            if not g.get(k):
                problems.append("group_configs[%d] 缺少必填字段 %s" % (i, k))
        model = g.get("model")
        if model and not str(model.get("address", "")).startswith(
                ("obs://", "s3://")):
            problems.append("group_configs[%d].model.address 非 OBS 形式"
                            % i)
        for j, u in enumerate(g.get("unit_configs", [])):
            for k in ("name", "count", "image", "cmd"):
                if not u.get(k):
                    problems.append(
                        "group_configs[%d].unit_configs[%d] 缺少 %s"
                        % (i, j, k))
    return (not problems), problems


# ---------------------------------------------------------------- 入口
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    src = args[0]
    dst = (args[1] if len(args) > 1
           else os.environ.get("MODELARTS_BODY_OUTPUT")
           or os.path.splitext(src)[0] + "-corrected.json")

    if not os.path.isfile(src):
        raise FileNotFoundError("源JSON不存在: %s" % src)
    with open(src, "r", encoding="utf-8") as f:
        raw = json.load(f)

    report = []
    body = normalize_create_body(
        raw, report,
        allow_unknown_model="--allow-unknown-model-address" in flags,
        fix_unit_names="--fix-unit-names" in flags)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)

    g = body["group_configs"][0]
    ok, problems = check_body(body)

    print("源文件  :", src)
    print("输出文件:", dst)
    print("服务名  :", body["name"])
    print("组      :", g.get("name"), "| pool:", g.get("pool_id"))
    print("model   :", json.dumps(g.get("model"), ensure_ascii=False))
    print("units   :", [(u.get("name"), u.get("count")) for u in g["unit_configs"]])
    for r in report:
        print("  -", r)
    print("自检    :", "PASS" if ok else "FAIL")
    for p in problems:
        print("    !", p)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
