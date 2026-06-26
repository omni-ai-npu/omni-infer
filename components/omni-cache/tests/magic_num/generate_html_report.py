# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Semgrep 魔鬼数字检测 - HTML 报告生成器

检测 5 类魔鬼数字：赋值语句、比较运算、加减乘除运算、数组索引/切片、函数参数
支持整数和浮点数检测

使用方法:
    # 默认模式（0, 1, -1 显示为 INFO 级别）
    python generate_html_report.py <target_path>

    # 严格模式（0, 1, -1 也作为 WARNING 级别）
    python generate_html_report.py <target_path> --strict

    # 增量检测（只检测 git diff 的文件）
    python generate_html_report.py <target_path> --diff

    # 严格模式 + 增量检测
    python generate_html_report.py <target_path> --strict --diff
"""

import subprocess
import json
import argparse
import os
import re
from pathlib import Path
from datetime import datetime


# 常见数字列表（整数和浮点数形式）
COMMON_NUMBERS = {'0', '1', '-1', '0.0', '1.0', '-1.0'}

# 具体的常见数字分类
ZERO_NUMBERS = {'0', '0.0'}
ONE_NUMBERS = {'1', '1.0'}
MINUS_ONE_NUMBERS = {'-1', '-1.0'}

# 规则名称映射
RULE_NAMES = {
    "magic-number-assign": "赋值语句",
    "magic-number-compare": "比较运算",
    "magic-number-arithmetic": "加减乘除运算",
    "magic-number-index": "数组索引/切片",
    "magic-number-func-arg": "函数参数"
}


def get_git_diff_files(target_dir: str) -> list:
    """获取当前分支未提交的变更文件（暂存区、工作目录、untracked）"""
    files = set()

    # 找到 git 仓库根目录
    try:
        result = subprocess.run(
            ["git", "-C", target_dir, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"无法找到 git 仓库根目录: {result.stderr}")
            return []
        git_root = result.stdout.strip()
    except Exception as e:
        print(f"获取 git 根目录失败: {e}")
        return []

    # 获取 target_dir 相对于 git_root 的路径
    target_abs = os.path.abspath(target_dir)
    git_root_abs = os.path.abspath(git_root)
    rel_target = os.path.relpath(target_abs, git_root_abs)

    def run_git_cmd(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and line.endswith('.py'):
                        files.add(line)
        except:
            pass

    # 1. 暂存区的文件（已 git add 但未 commit）
    run_git_cmd(["git", "-C", git_root, "diff", "--cached", "--name-only", "--diff-filter=ACMR"])

    # 2. 工作目录未暂存的修改
    run_git_cmd(["git", "-C", git_root, "diff", "--name-only", "--diff-filter=ACMR"])

    # 3. untracked 文件（未 git add 的新文件）
    run_git_cmd(["git", "-C", git_root, "ls-files", "--others", "--exclude-standard"])

    # 过滤：只返回 target_dir 下的文件
    filtered_files = []
    for f in files:
        # 检查文件是否在 target_dir 下
        if f.startswith(rel_target + '/') or f.startswith(rel_target + '\\'):
            filtered_files.append(os.path.join(git_root, f))

    return filtered_files


def run_semgrep(targets: list, config: str = "magic_rule.yaml") -> dict:
    """运行 Semgrep 检测"""
    if not targets:
        return {"results": []}

    cmd = ["semgrep", "--config", config, "--json", "--quiet"] + targets
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"results": []}


def is_common_number_issue(result: dict) -> bool:
    """判断是否为常见数字（0, 1, -1 及其浮点数形式）相关的问题"""
    metavars = result.get("extra", {}).get("metavars", {})
    num_var = metavars.get("$NUM", {})
    if num_var:
        content = num_var.get("abstract_content", "")
        if content in COMMON_NUMBERS:
            return True
    msg = result.get("extra", {}).get("message", "")
    match = re.search(r'魔鬼数字:\s*(-?[0-9]+(?:\.[0-9]+)?)', msg)
    if match and match.group(1) in COMMON_NUMBERS:
        return True
    return False


def get_common_number_type(result: dict) -> str:
    """获取常见数字的类型：'0', '1', '-1' 或 None"""
    metavars = result.get("extra", {}).get("metavars", {})
    num_var = metavars.get("$NUM", {})
    content = ""
    if num_var:
        content = num_var.get("abstract_content", "")
    else:
        msg = result.get("extra", {}).get("message", "")
        match = re.search(r'魔鬼数字:\s*(-?[0-9]+(?:\.[0-9]+)?)', msg)
        if match:
            content = match.group(1)

    if content in ZERO_NUMBERS:
        return '0'
    elif content in ONE_NUMBERS:
        return '1'
    elif content in MINUS_ONE_NUMBERS:
        return '-1'
    return None


def is_constant_definition(result: dict) -> bool:
    """判断是否在常量定义上下文中（排除算术运算规则的误报）"""
    rule = result.get("check_id", "").split(".")[-1]
    # 只对算术运算规则进行过滤
    if rule != "magic-number-arithmetic":
        return False

    filepath = result.get("path", "")
    line_no = result.get("start", {}).get("line", 0)
    if not filepath or not line_no:
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if line_no > len(lines):
            return False
        line = lines[line_no - 1].strip()
        # 匹配常量定义：UPPER_CASE = ...
        if re.match(r'^[_A-Z][A-Z0-9_]*\s*=', line):
            return True
    except:
        pass
    return False


def filter_results(results: list) -> list:
    """过滤掉常量定义上下文中的问题"""
    return [r for r in results if not is_constant_definition(r)]


def get_css():
    """返回 CSS 样式"""
    return """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
    .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
    h1 { color: #333; margin-bottom: 10px; }
    h2 { color: #444; margin: 20px 0 10px; }
    .subtitle { color: #666; margin-bottom: 20px; }

    .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-top: 15px; }
    .summary-item { text-align: center; padding: 12px; background: #f8f9fa; border-radius: 6px; }
    .summary-item .number { font-size: 28px; font-weight: bold; }
    .summary-item .label { color: #666; margin-top: 5px; font-size: 14px; }
    .warning-count .number { color: #e74c3c; }
    .info-count .number { color: #3498db; }
    .files-count .number { color: #2ecc71; }

    .stats-table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .stats-table th, .stats-table td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #eee; }
    .stats-table th { background: #34495e; color: white; font-weight: 600; }
    .stats-table tr:hover { background: #f8f9fa; }
    .stats-table a { color: #2980b9; text-decoration: none; }
    .stats-table a:hover { text-decoration: underline; }
    .badge { padding: 2px 8px; border-radius: 10px; font-size: 12px; color: white; }
    .badge-warning { background: #e74c3c; }
    .badge-info { background: #3498db; }

    .file-content { background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 20px; }
    .file-header { background: #5a6c7d; color: #fff; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; }
    .file-header a { color: #74b9ff; }
    .file-header h2 { color: #fff; margin: 0; }
    .file-header .stats { color: #fff; }
    .code-table { width: 100%; border-collapse: collapse; font-family: "SF Mono", "Monaco", "Inconsolata", "Fira Mono", "Droid Sans Mono", "Source Code Pro", monospace; font-size: 13px; line-height: 1.5; }
    .code-table td { padding: 1px 8px 1px 8px; vertical-align: top; }
    .line-number { width: 50px; min-width: 50px; text-align: right; color: #6c6c6c; background: #f7f7f7; border-right: 1px solid #e1e4e8; user-select: none; font-size: 12px; }
    .line-content { white-space: pre; padding-left: 12px; }
    .issue-line-warning { background: #fff3cd; }
    .issue-line-warning .line-number { background: #ffc107; color: #856404; }
    .issue-line-info { background: #d1ecf1; }
    .issue-line-info .line-number { background: #17a2b8; color: #fff; }
    .issue-info { padding: 5px 10px; margin-left: 60px; border-left: 3px solid; font-size: 12px; background: #fafafa; }
    .issue-info-warning { border-color: #e74c3c; color: #c0392b; }
    .issue-info-info { border-color: #3498db; color: #2980b9; }
    .magic-num { font-weight: bold; }

    .mode-badge { display: inline-block; padding: 4px 10px; border-radius: 15px; font-size: 12px; margin-left: 10px; }
    .mode-strict { background: #e74c3c; color: white; }
    .mode-default { background: #3498db; color: white; }
    .mode-diff { background: #27ae60; color: white; }

    .cmd-box { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 8px; font-family: monospace; margin-bottom: 20px; overflow-x: auto; font-size: 13px; }
    .cmd-box .comment { color: #95a5a6; }

    .breadcrumb { color: #666; margin-bottom: 15px; font-size: 14px; }
    .breadcrumb a { color: #2980b9; text-decoration: none; }
    .breadcrumb a:hover { text-decoration: underline; }

    .sort-btn { margin-left: 15px; padding: 4px 12px; border: 1px solid #ddd; border-radius: 4px; background: #fff; cursor: pointer; font-size: 12px; }
    .sort-btn:hover { background: #f0f0f0; }
    .sort-btn.active { background: #3498db; color: white; border-color: #3498db; }
    """


def generate_html_report(
    semgrep_output: dict,
    target: str,
    output_dir: str,
    strict_mode: bool = False,
    is_diff_mode: bool = False,
    diff_files: list = None
):
    """生成 HTML 报告"""

    results = semgrep_output.get("results", [])

    # 过滤掉常量定义上下文中的问题
    results = filter_results(results)

    # 分类结果：WARNING 和 INFO（仅默认模式）
    warning_results = []
    info_results = []
    info_zero_results = []
    info_one_results = []
    info_minus_one_results = []

    for r in results:
        if not strict_mode and is_common_number_issue(r):
            info_results.append(r)
            num_type = get_common_number_type(r)
            if num_type == '0':
                info_zero_results.append(r)
            elif num_type == '1':
                info_one_results.append(r)
            elif num_type == '-1':
                info_minus_one_results.append(r)
        else:
            warning_results.append(r)

    all_results = warning_results + info_results

    # 按文件分组
    file_results = {}
    for r in all_results:
        filepath = r.get("path", "")
        if filepath not in file_results:
            file_results[filepath] = []
        file_results[filepath].append(r)

    # 按规则分组
    rule_results = {}
    for r in all_results:
        rule = r.get("check_id", "").split(".")[-1]
        if rule not in rule_results:
            rule_results[rule] = []
        rule_results[rule].append(r)

    # 统计
    total_warning = len(warning_results)
    total_info = len(info_results)
    total_files = len(file_results)

    # INFO 分类统计：0, 1, -1
    info_zero = sum(1 for r in info_results if get_common_number_type(r) == '0')
    info_one = sum(1 for r in info_results if get_common_number_type(r) == '1')
    info_minus_one = sum(1 for r in info_results if get_common_number_type(r) == '-1')

    # 按规则统计
    rule_stats = {}
    for r in all_results:
        rule = r.get("check_id", "").split(".")[-1]
        if rule not in rule_stats:
            rule_stats[rule] = {"WARNING": 0, "INFO": 0}
        if r in warning_results:
            rule_stats[rule]["WARNING"] += 1
        else:
            rule_stats[rule]["INFO"] += 1

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    css = get_css()

    # ============================================
    # 1. 生成规则详情页面及对应的文件详情页面
    # ============================================
    for rule, rule_issues in rule_results.items():
        rule_name = RULE_NAMES.get(rule, rule)
        rule_safe_name = rule.replace('-', '_')

        # 按文件分组该规则的问题
        rule_file_results = {}
        for r in rule_issues:
            filepath = r.get("path", "")
            if filepath not in rule_file_results:
                rule_file_results[filepath] = []
            rule_file_results[filepath].append(r)

        # 统计该规则
        rule_warning = sum(1 for r in rule_issues if r in warning_results)
        rule_info = sum(1 for r in rule_issues if r in info_results)

        # 生成该规则下的文件详情页面（只显示该规则的问题）
        for filepath, issues in rule_file_results.items():
            short_path = filepath.split('/omni_cache/')[-1] if '/omni_cache/' in filepath else os.path.basename(filepath)
            safe_name = short_path.replace('/', '_').replace('\\', '_')

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                lines = []

            # 标记问题行
            issue_lines = {}
            for issue in issues:
                line_no = issue.get("start", {}).get("line", 0)
                severity = "WARNING" if issue in warning_results else "INFO"
                if line_no not in issue_lines:
                    issue_lines[line_no] = []
                issue_lines[line_no].append((issue, severity))

            file_warning = sum(1 for i in issues if i in warning_results)
            file_info = sum(1 for i in issues if i in info_results)

            file_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{short_path} - 规则: {rule_name}</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <div class="file-header">
        <div>
            <a href="rule_{rule_safe_name}.html">← 返回规则: {rule_name}</a>
            <h2 style="margin-top: 10px;">{short_path}</h2>
            <div style="color: #ccc; font-size: 14px; margin-top: 5px;">规则: {rule_name} ({rule})</div>
        </div>
        <div>
            <span class="badge badge-warning">{file_warning} WARNING</span>
            {f'<span class="badge badge-info" style="margin-left: 10px;">' + str(file_info) + ' INFO</span>' if file_info > 0 else ''}
        </div>
    </div>
    <div class="file-content">
    <table class="code-table">
"""

            for i, line in enumerate(lines, 1):
                line_escaped = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
                if i in issue_lines:
                    for idx, (issue, severity) in enumerate(issue_lines[i]):
                        msg = issue.get("extra", {}).get("message", "")
                        severity_lower = severity.lower()

                        metavars = issue.get("extra", {}).get("metavars", {})
                        num_var = metavars.get("$NUM", {})
                        magic_num = num_var.get("abstract_content", "") if num_var else ""

                        line_with_bold = line_escaped
                        if magic_num:
                            line_with_bold = re.sub(r'(?<![a-zA-Z0-9_])' + re.escape(magic_num) + r'(?![a-zA-Z0-9_])',
                                                    f'<b class="magic-num">{magic_num}</b>', line_escaped)

                        msg_with_bold = msg
                        if magic_num:
                            msg_with_bold = msg.replace(magic_num, f'<b>{magic_num}</b>')

                        if idx == 0:
                            file_html += f"""    <tr class="issue-line-{severity_lower}">
        <td class="line-number">{i}</td>
        <td class="line-content">{line_with_bold}</td>
    </tr>
"""
                        file_html += f"""    <tr><td></td><td class="issue-info issue-info-{severity_lower}"><span class="badge badge-{severity_lower}">{severity}</span> {msg_with_bold}</td></tr>
"""
                else:
                    file_html += f"""    <tr>
        <td class="line-number">{i}</td>
        <td class="line-content">{line_escaped}</td>
    </tr>
"""

            file_html += """    </table>
    </div>
</div>
</body>
</html>"""

            with open(os.path.join(output_dir, f"rule_{rule_safe_name}_{safe_name}.html"), 'w', encoding='utf-8') as f:
                f.write(file_html)

        # 生成规则详情页面
        rule_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{rule_name} - 魔鬼数字检测</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <div class="file-header">
        <div>
            <a href="index.html">← 返回首页</a>
            <h2 style="margin-top: 10px;">规则: {rule_name}</h2>
            <div style="color: #ccc; font-size: 14px; margin-top: 5px;">{rule}</div>
        </div>
        <div>
            <span class="badge badge-warning">{rule_warning} WARNING</span>
            {f'<span class="badge badge-info" style="margin-left: 10px;">' + str(rule_info) + ' INFO</span>' if rule_info > 0 else ''}
        </div>
    </div>

    <h2>涉及文件 ({len(rule_file_results)} 个)</h2>
    <table class="stats-table">
        <tr><th>文件</th><th>问题数</th><th>级别</th></tr>
"""

        for filepath, issues in sorted(rule_file_results.items(), key=lambda x: x[0]):
            short_path = filepath.split('/omni_cache/')[-1] if '/omni_cache/' in filepath else os.path.basename(filepath)
            safe_name = short_path.replace('/', '_').replace('\\', '_')
            file_warning_cnt = sum(1 for i in issues if i in warning_results)
            file_info_cnt = sum(1 for i in issues if i in info_results)

            levels = []
            if file_warning_cnt > 0:
                levels.append(f'<span class="badge badge-warning">{file_warning_cnt} WARNING</span>')
            if file_info_cnt > 0:
                levels.append(f'<span class="badge badge-info">{file_info_cnt} INFO</span>')

            rule_html += f"""        <tr>
            <td><a href="rule_{rule_safe_name}_{safe_name}.html">{short_path}</a></td>
            <td>{len(issues)}</td>
            <td>{' '.join(levels)}</td>
        </tr>
"""

        rule_html += """    </table>
</div>
</body>
</html>"""

        with open(os.path.join(output_dir, f"rule_{rule_safe_name}.html"), 'w', encoding='utf-8') as f:
            f.write(rule_html)

    # ============================================
    # 1.5 生成 INFO 分类详情页面及文件详情页面
    # ============================================
    info_categories = {
        '0': {'results': info_zero_results, 'label': '0 (包括 0.0)'},
        '1': {'results': info_one_results, 'label': '1 (包括 1.0)'},
        '-1': {'results': info_minus_one_results, 'label': '-1 (包括 -1.0)'}
    }

    for cat_key, cat_data in info_categories.items():
        cat_results = cat_data['results']
        cat_label = cat_data['label']
        safe_key = cat_key.replace('-', 'minus')

        # 按文件分组
        cat_file_results = {}
        for r in cat_results:
            filepath = r.get("path", "")
            if filepath not in cat_file_results:
                cat_file_results[filepath] = []
            cat_file_results[filepath].append(r)

        # 按规则分组
        cat_rule_results = {}
        for r in cat_results:
            rule = r.get("check_id", "").split(".")[-1]
            if rule not in cat_rule_results:
                cat_rule_results[rule] = []
            cat_rule_results[rule].append(r)

        # 生成该分类下的文件详情页面
        for filepath, issues in cat_file_results.items():
            short_path = filepath.split('/omni_cache/')[-1] if '/omni_cache/' in filepath else os.path.basename(filepath)
            safe_name = short_path.replace('/', '_').replace('\\', '_')

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except:
                lines = []

            # 标记问题行
            issue_lines = {}
            for issue in issues:
                line_no = issue.get("start", {}).get("line", 0)
                if line_no not in issue_lines:
                    issue_lines[line_no] = []
                issue_lines[line_no].append((issue, "INFO"))

            # 生成文件页面
            file_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{short_path} - INFO: {cat_label}</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <div class="file-header">
        <div>
            <a href="info_{safe_key}.html">← 返回 INFO: {cat_label}</a>
            <h2 style="margin-top: 10px;">{short_path}</h2>
            <div style="color: #ccc; font-size: 14px; margin-top: 5px;">INFO 分类: {cat_label}</div>
        </div>
        <div>
            <span class="badge badge-info">{len(issues)} INFO</span>
        </div>
    </div>
    <div class="file-content">
    <table class="code-table">
"""

            for i, line in enumerate(lines, 1):
                line_escaped = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
                if i in issue_lines:
                    for idx, (issue, severity) in enumerate(issue_lines[i]):
                        rule = issue.get("check_id", "").split(".")[-1]
                        msg = issue.get("extra", {}).get("message", "")

                        # 提取魔鬼数字并加粗
                        metavars = issue.get("extra", {}).get("metavars", {})
                        num_var = metavars.get("$NUM", {})
                        magic_num = num_var.get("abstract_content", "") if num_var else ""

                        # 在代码行中加粗魔鬼数字
                        line_with_bold = line_escaped
                        if magic_num:
                            line_with_bold = re.sub(r'(?<![a-zA-Z0-9_])' + re.escape(magic_num) + r'(?![a-zA-Z0-9_])',
                                                    f'<b class="magic-num">{magic_num}</b>', line_escaped)

                        # 消息中的数字加粗
                        msg_with_bold = msg
                        if magic_num:
                            msg_with_bold = msg.replace(magic_num, f'<b>{magic_num}</b>')

                        if idx == 0:
                            file_html += f"""    <tr class="issue-line-info">
        <td class="line-number">{i}</td>
        <td class="line-content">{line_with_bold}</td>
    </tr>
"""
                        file_html += f"""    <tr><td></td><td class="issue-info issue-info-info"><span class="badge badge-info">INFO</span> {msg_with_bold}</td></tr>
"""
                else:
                    file_html += f"""    <tr>
        <td class="line-number">{i}</td>
        <td class="line-content">{line_escaped}</td>
    </tr>
"""

            file_html += """    </table>
    </div>
</div>
</body>
</html>"""

            with open(os.path.join(output_dir, f"info_{safe_key}_{safe_name}.html"), 'w', encoding='utf-8') as f:
                f.write(file_html)

        # 生成该分类下的规则详情页面及对应的文件详情页面
        for rule_key, rule_issues in cat_rule_results.items():
            rule_name = RULE_NAMES.get(rule_key, rule_key)
            rule_safe = rule_key.replace('-', '_')

            # 按文件分组该规则的问题
            rule_file_results = {}
            for r in rule_issues:
                filepath = r.get("path", "")
                if filepath not in rule_file_results:
                    rule_file_results[filepath] = []
                rule_file_results[filepath].append(r)

            # 生成该规则下的文件详情页面（只显示该规则+该分类的问题）
            for filepath, issues in rule_file_results.items():
                short_path = filepath.split('/omni_cache/')[-1] if '/omni_cache/' in filepath else os.path.basename(filepath)
                safe_name = short_path.replace('/', '_').replace('\\', '_')

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except:
                    lines = []

                # 标记问题行
                issue_lines = {}
                for issue in issues:
                    line_no = issue.get("start", {}).get("line", 0)
                    if line_no not in issue_lines:
                        issue_lines[line_no] = []
                    issue_lines[line_no].append((issue, "INFO"))

                file_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{short_path} - INFO: {cat_label} / {rule_name}</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <div class="file-header">
        <div>
            <a href="info_{safe_key}_rule_{rule_safe}.html">← 返回 {cat_label} / {rule_name}</a>
            <h2 style="margin-top: 10px;">{short_path}</h2>
            <div style="color: #ccc; font-size: 14px; margin-top: 5px;">INFO: {cat_label} | 规则: {rule_name} ({rule_key})</div>
        </div>
        <div>
            <span class="badge badge-info">{len(issues)} INFO</span>
        </div>
    </div>
    <div class="file-content">
    <table class="code-table">
"""

                for i, line in enumerate(lines, 1):
                    line_escaped = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
                    if i in issue_lines:
                        for idx, (issue, severity) in enumerate(issue_lines[i]):
                            msg = issue.get("extra", {}).get("message", "")

                            metavars = issue.get("extra", {}).get("metavars", {})
                            num_var = metavars.get("$NUM", {})
                            magic_num = num_var.get("abstract_content", "") if num_var else ""

                            line_with_bold = line_escaped
                            if magic_num:
                                line_with_bold = re.sub(r'(?<![a-zA-Z0-9_])' + re.escape(magic_num) + r'(?![a-zA-Z0-9_])',
                                                        f'<b class="magic-num">{magic_num}</b>', line_escaped)

                            msg_with_bold = msg
                            if magic_num:
                                msg_with_bold = msg.replace(magic_num, f'<b>{magic_num}</b>')

                            if idx == 0:
                                file_html += f"""    <tr class="issue-line-info">
        <td class="line-number">{i}</td>
        <td class="line-content">{line_with_bold}</td>
    </tr>
"""
                            file_html += f"""    <tr><td></td><td class="issue-info issue-info-info"><span class="badge badge-info">INFO</span> {msg_with_bold}</td></tr>
"""
                    else:
                        file_html += f"""    <tr>
        <td class="line-number">{i}</td>
        <td class="line-content">{line_escaped}</td>
    </tr>
"""

                file_html += """    </table>
    </div>
</div>
</body>
</html>"""

                with open(os.path.join(output_dir, f"info_{safe_key}_rule_{rule_safe}_{safe_name}.html"), 'w', encoding='utf-8') as f:
                    f.write(file_html)

            # 生成规则详情页面
            rule_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{rule_name} - INFO: {cat_label}</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <div class="file-header">
        <div>
            <a href="info_{safe_key}.html">← 返回 INFO: {cat_label}</a>
            <h2 style="margin-top: 10px;">规则: {rule_name}</h2>
            <div style="color: #ccc; font-size: 14px; margin-top: 5px;">{rule_key}</div>
        </div>
        <div>
            <span class="badge badge-info">{len(rule_issues)} 个</span>
        </div>
    </div>

    <h2>涉及文件 ({len(rule_file_results)} 个)</h2>
    <table class="stats-table">
        <tr><th>文件</th><th>数量</th></tr>
"""

            for filepath, issues in sorted(rule_file_results.items(), key=lambda x: x[0]):
                short_path = filepath.split('/omni_cache/')[-1] if '/omni_cache/' in filepath else os.path.basename(filepath)
                safe_name = short_path.replace('/', '_').replace('\\', '_')
                rule_html += f"""        <tr>
            <td><a href="info_{safe_key}_rule_{rule_safe}_{safe_name}.html">{short_path}</a></td>
            <td><span class="badge badge-info">{len(issues)}</span></td>
        </tr>
"""

            rule_html += """    </table>
</div>
</body>
</html>"""

            with open(os.path.join(output_dir, f"info_{safe_key}_rule_{rule_safe}.html"), 'w', encoding='utf-8') as f:
                f.write(rule_html)

        # 生成分类主页
        cat_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>INFO: {cat_label} - 魔鬼数字检测</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <div class="file-header">
        <div>
            <a href="index.html">← 返回首页</a>
            <h2 style="margin-top: 10px;">INFO: {cat_label}</h2>
        </div>
        <div>
            <span class="badge badge-info">{len(cat_results)} 个</span>
        </div>
    </div>

    <h2>按规则分布（点击查看详情）</h2>
    <table class="stats-table">
        <tr><th>规则</th><th>检测场景</th><th>数量</th></tr>
"""

        for rule in ["magic-number-assign", "magic-number-compare", "magic-number-arithmetic", "magic-number-index", "magic-number-func-arg"]:
            if rule in cat_rule_results:
                count = len(cat_rule_results[rule])
                rule_safe = rule.replace('-', '_')
                cat_html += f"""        <tr>
            <td><a href="info_{safe_key}_rule_{rule_safe}.html">{rule}</a></td>
            <td>{RULE_NAMES.get(rule, "")}</td>
            <td><span class="badge badge-info">{count}</span></td>
        </tr>
"""

        cat_html += """    </table>

    <h2>涉及文件""" + f" ({len(cat_file_results)} 个)" + """</h2>
    <table class="stats-table">
        <tr><th>文件</th><th>数量</th></tr>
"""

        for filepath, issues in sorted(cat_file_results.items(), key=lambda x: x[0]):
            short_path = filepath.split('/omni_cache/')[-1] if '/omni_cache/' in filepath else os.path.basename(filepath)
            safe_name = short_path.replace('/', '_').replace('\\', '_')
            cat_html += f"""        <tr>
            <td><a href="info_{safe_key}_{safe_name}.html">{short_path}</a></td>
            <td><span class="badge badge-info">{len(issues)}</span></td>
        </tr>
"""

        cat_html += """    </table>
</div>
</body>
</html>"""

        with open(os.path.join(output_dir, f"info_{safe_key}.html"), 'w', encoding='utf-8') as f:
            f.write(cat_html)

    # ============================================
    # 2. 生成文件详情页面
    # ============================================
    for filepath, issues in file_results.items():
        short_path = filepath.split('/omni_cache/')[-1] if '/omni_cache/' in filepath else os.path.basename(filepath)
        safe_name = short_path.replace('/', '_').replace('\\', '_')

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            lines = []

        # 标记问题行
        issue_lines = {}
        for issue in issues:
            line_no = issue.get("start", {}).get("line", 0)
            severity = "WARNING" if issue in warning_results else "INFO"
            if line_no not in issue_lines:
                issue_lines[line_no] = []
            issue_lines[line_no].append((issue, severity))

        # 统计该文件
        file_warning = sum(1 for i in issues if i in warning_results)
        file_info = sum(1 for i in issues if i in info_results)

        # 生成文件页面
        file_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{short_path}</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <div class="file-header">
        <div>
            <a href="index.html">← 返回首页</a>
            <h2 style="margin-top: 10px;">{short_path}</h2>
        </div>
        <div>
            <span class="badge badge-warning">{file_warning} WARNING</span>
            {f'<span class="badge badge-info" style="margin-left: 10px;">' + str(file_info) + ' INFO</span>' if file_info > 0 else ''}
        </div>
    </div>
    <div class="file-content">
    <table class="code-table">
"""

        for i, line in enumerate(lines, 1):
            line_escaped = line.rstrip().replace('<', '&lt;').replace('>', '&gt;')
            if i in issue_lines:
                for idx, (issue, severity) in enumerate(issue_lines[i]):
                    rule = issue.get("check_id", "").split(".")[-1]
                    msg = issue.get("extra", {}).get("message", "")
                    severity_lower = severity.lower()

                    # 提取魔鬼数字并加粗
                    metavars = issue.get("extra", {}).get("metavars", {})
                    num_var = metavars.get("$NUM", {})
                    magic_num = num_var.get("abstract_content", "") if num_var else ""

                    # 在代码行中加粗魔鬼数字
                    line_with_bold = line_escaped
                    if magic_num:
                        # 对数字加粗
                        line_with_bold = re.sub(r'(?<![a-zA-Z0-9_])' + re.escape(magic_num) + r'(?![a-zA-Z0-9_])',
                                                f'<b class="magic-num">{magic_num}</b>', line_escaped)

                    # 消息中的数字加粗
                    msg_with_bold = msg
                    if magic_num:
                        msg_with_bold = msg.replace(magic_num, f'<b>{magic_num}</b>')

                    if idx == 0:
                        file_html += f"""    <tr class="issue-line-{severity_lower}">
        <td class="line-number">{i}</td>
        <td class="line-content">{line_with_bold}</td>
    </tr>
"""
                    file_html += f"""    <tr><td></td><td class="issue-info issue-info-{severity_lower}"><span class="badge badge-{severity_lower}">{severity}</span> {msg_with_bold}</td></tr>
"""
            else:
                file_html += f"""    <tr>
        <td class="line-number">{i}</td>
        <td class="line-content">{line_escaped}</td>
    </tr>
"""

        file_html += """    </table>
    </div>
</div>
</body>
</html>"""

        with open(os.path.join(output_dir, f"{safe_name}.html"), 'w', encoding='utf-8') as f:
            f.write(file_html)

    # ============================================
    # 3. 生成首页
    # ============================================
    mode_badges = []
    if strict_mode:
        mode_badges.append('<span class="mode-badge mode-strict">严格模式</span>')
    else:
        mode_badges.append('<span class="mode-badge mode-default">默认模式</span>')
    if is_diff_mode:
        mode_badges.append('<span class="mode-badge mode-diff">增量检测</span>')

    diff_info = f"检测 {len(diff_files)} 个变更文件" if diff_files else ""

    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>魔鬼数字检测报告</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <h1>魔鬼数字检测报告 {''.join(mode_badges)}</h1>
    <div class="subtitle">
        检测时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 目标: {target}
        {'| ' + diff_info if diff_info else ''}
    </div>

    <div class="cmd-box">
        <div class="comment"># 使用方法</div>
        <div>python generate_html_report.py &lt;target&gt;                    # 默认模式</div>
        <div>python generate_html_report.py &lt;target&gt; --strict            # 严格模式</div>
        <div>python generate_html_report.py &lt;target&gt; --diff              # 增量检测</div>
        <div>python generate_html_report.py &lt;target&gt; --strict --diff     # 严格模式 + 增量检测</div>
    </div>

    <div class="summary">
        <div class="summary-grid">
            <div class="summary-item warning-count">
                <div class="number">{total_warning}</div>
                <div class="label">WARNING</div>
            </div>
            <div class="summary-item info-count">
                <div class="number">{total_info}</div>
                <div class="label">INFO (0,1,-1)</div>
            </div>
            <div class="summary-item files-count">
                <div class="number">{total_files}</div>
                <div class="label">涉及文件</div>
            </div>
        </div>
    </div>

    <h2>按规则统计（点击查看详情）</h2>
    <table class="stats-table">
        <tr><th>规则</th><th>检测场景</th><th>WARNING</th><th>INFO</th></tr>
"""

    for rule in ["magic-number-assign", "magic-number-compare", "magic-number-arithmetic", "magic-number-index", "magic-number-func-arg"]:
        if rule in rule_stats:
            stats = rule_stats[rule]
            rule_safe_name = rule.replace('-', '_')
            index_html += f"""        <tr>
            <td><a href="rule_{rule_safe_name}.html">{rule}</a></td>
            <td>{RULE_NAMES.get(rule, "")}</td>
            <td><span class="badge badge-warning">{stats['WARNING']}</span></td>
            <td>{f"<span class='badge badge-info'>{stats['INFO']}</span>" if stats['INFO'] > 0 else '-'}</td>
        </tr>
"""

    index_html += """    </table>

    <h2>INFO 分类统计（点击查看详情）</h2>
    <table class="stats-table">
        <tr><th>分类</th><th>说明</th><th>数量</th></tr>
        <tr><td><a href="info_0.html">0</a></td><td>包括 0.0</td><td><span class="badge badge-info">""" + str(info_zero) + """</span></td></tr>
        <tr><td><a href="info_1.html">1</a></td><td>包括 1.0</td><td><span class="badge badge-info">""" + str(info_one) + """</span></td></tr>
        <tr><td><a href="info_minus1.html">-1</a></td><td>包括 -1.0</td><td><span class="badge badge-info">""" + str(info_minus_one) + """</span></td></tr>
    </table>

    <h2>按文件统计
        <button class="sort-btn active" onclick="sortFiles('path', this)">按路径排序</button>
        <button class="sort-btn" onclick="sortFiles('warning', this)">按WARNING数排序</button>
    </h2>
    <table class="stats-table" id="file-table">
        <thead><tr><th>文件</th><th>WARNING</th><th>INFO</th></tr></thead>
        <tbody>
"""

    for filepath, issues in sorted(file_results.items(), key=lambda x: x[0]):
        short_path = filepath.split('/omni_cache/')[-1] if '/omni_cache/' in filepath else os.path.basename(filepath)
        safe_name = short_path.replace('/', '_').replace('\\', '_')
        warning_cnt = sum(1 for i in issues if i in warning_results)
        info_cnt = sum(1 for i in issues if i in info_results)

        index_html += f"""        <tr data-path="{short_path}" data-warning="{warning_cnt}" data-info="{info_cnt}">
            <td><a href="{safe_name}.html">{short_path}</a></td>
            <td><span class="badge badge-warning">{warning_cnt}</span></td>
            <td>{f"<span class='badge badge-info'>{info_cnt}</span>" if info_cnt > 0 else '-'}</td>
        </tr>
"""

    index_html += """        </tbody>
    </table>

    <script>
    function sortFiles(sortBy, btn) {
        var tbody = document.querySelector('#file-table tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));

        // Update button states
        document.querySelectorAll('.sort-btn').forEach(function(b) {
            b.classList.remove('active');
        });
        btn.classList.add('active');

        rows.sort(function(a, b) {
            if (sortBy === 'path') {
                return a.dataset.path.localeCompare(b.dataset.path);
            } else if (sortBy === 'warning') {
                return parseInt(b.dataset.warning) - parseInt(a.dataset.warning);
            }
        });

        rows.forEach(function(row) {
            tbody.appendChild(row);
        });
    }
    </script>
</div>
</body>
</html>"""

    with open(os.path.join(output_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(index_html)

    # 输出摘要
    print(f"\n{'='*60}")
    print(f"HTML 报告已生成: {output_dir}/index.html")
    print(f"{'='*60}")
    print(f"  模式: {'严格模式' if strict_mode else '默认模式'}")
    print(f"  WARNING: {total_warning} 个")
    print(f"  INFO:    {total_info} 个 (0: {info_zero}, 1: {info_one}, -1: {info_minus_one})")
    print(f"  涉及文件: {total_files} 个")
    if is_diff_mode and diff_files:
        print(f"  变更文件: {len(diff_files)} 个")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Semgrep 魔鬼数字检测 - HTML 报告生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_html_report.py ../omni-cache/omni_cache
  python generate_html_report.py ../omni-cache/omni_cache --strict
  python generate_html_report.py ../omni-cache/omni_cache --diff
  python generate_html_report.py ../omni-cache/omni_cache --strict --diff
        """
    )
    parser.add_argument("target", help="检测目标（文件或目录路径）")
    parser.add_argument("--config", default="magic_rule.yaml", help="Semgrep 规则配置文件")
    parser.add_argument("--output", default="html_report", help="HTML 输出目录")
    parser.add_argument("--strict", action="store_true", help="严格模式：0, 1, -1 也作为 WARNING")
    parser.add_argument("--diff", action="store_true", help="增量检测：只检测当前分支未提交的变更文件")

    args = parser.parse_args()

    if not Path(args.target).exists():
        print(f"目标路径不存在: {args.target}")
        exit(1)

    if not Path(args.config).exists():
        print(f"规则文件不存在: {args.config}")
        exit(1)

    # 确定检测目标
    if args.diff:
        # 获取当前分支名
        try:
            result = subprocess.run(
                ["git", "-C", args.target, "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True
            )
            current_branch = result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            current_branch = "unknown"
        print(f"[增量检测] 当前分支: {current_branch}")
        diff_files = get_git_diff_files(args.target)
        if not diff_files:
            print("没有发现变更的 Python 文件")
            generate_html_report({"results": []}, args.target, args.output, args.strict, True, [])
            return
        print(f"发现 {len(diff_files)} 个变更文件未commit")
        targets = diff_files
    else:
        diff_files = None
        targets = [args.target]

    print(f"检测模式: {'严格模式' if args.strict else '默认模式'}")
    print(f"使用规则: {args.config}")
    print("正在检测...")

    # 运行检测
    semgrep_output = run_semgrep(targets, args.config)

    # 生成报告
    generate_html_report(
        semgrep_output,
        args.target,
        args.output,
        args.strict,
        args.diff,
        diff_files
    )


if __name__ == "__main__":
    main()