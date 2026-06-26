# Semgrep 魔鬼数字检测工具

检测 Python 代码中的"魔鬼数字"（magic numbers），生成类似 pytest-cov 的交互式 HTML 报告，提醒开发者将其定义为常量以提高代码可读性和可维护性。

## 目录结构

```
├── run_magic_num_tests.sh      # 魔鬼数字检测（推荐入口）
└── magic_num/                  # 魔鬼数字检测工具
    ├── magic_rule.yaml         # Semgrep 规则文件
    ├── generate_html_report.py # HTML 报告生成脚本
    └── README.md               # 本文档
```
### 整体流程

```
Python 源码
    │
    ▼
┌──────────────┐     magic_rule.yaml (5 条规则)
│   Semgrep    │◄────────────────────────────
│  (AST 级扫描) │
└──────┬───────┘
       │ JSON 结果 (--json)
       ▼
┌──────────────────────┐
│ generate_html_report │  ← 解析 JSON → 分类/分组 → 读源码 → 拼接 HTML
│     .py              │
└──────┬───────────────┘
       │
       ▼
  HTML 交互式报告
  (首页 / 规则页 / 文件页)
```


## 规则说明

共 5 类检测规则：

| 规则 ID | 检测场景 | 示例 |
|---------|----------|------|
| `magic-number-assign` | 赋值语句 | `x = 100`, `y = 3.14` |
| `magic-number-compare` | 比较运算 | `if x > 90:` |
| `magic-number-arithmetic` | 加减乘除运算 | `x * 100`, `x + 10`, `x / 2.0` |
| `magic-number-index` | 数组索引/切片 | `arr[5]`, `arr[:10]`, `arr[1:10:2]` |
| `magic-number-func-arg` | 函数参数 | `func(100)`, `func(size=30)` |

**支持检测：**
- 整数：`100`, `-1`, `0`
- 浮点数：`9.0`, `888.22587`, `-3.14`

### 特殊处理

- **打印信息不报**： 过滤logger.xxx, print 中的数字
- **常量定义不报**: 全大写变量名视为常量定义，不会误报
  ```python
  MAX_SIZE = 100      # 不会报（大写变量名）
  timeout = 30        # 会报（小写变量名）
  MAX_SIZE = 2 * 1024 * 1024      # 不会报（过滤）
  ```

## 检测模式

### 默认模式

- 所有数字都检测（整数和浮点数）
- 0, 1, -1（及其浮点形式 0.0, 1.0, -1.0）显示为 INFO 级别（蓝色），与其他 WARNING（黄色）区分
- 便于快速识别真正需要关注的魔鬼数字

### 严格模式

- 所有数字统一为 WARNING 级别
- 0, 1, -1 也计入 WARNING

## 安装依赖

### 系统要求

- Python 3.8+
- glibc 2.34+ (Linux)

### 安装步骤

**注意：** 最新版 semgrep (1.121+) 需要 glibc 2.35+，如果你的系统 glibc 版本较低（如 openEuler 22.03 的 glibc 2.34），需要安装旧版本。

检查 glibc 版本：
```bash
ldd --version | head -1
```

**glibc >= 2.35 的系统：**
```bash
pip install semgrep
```

**glibc < 2.35 的系统（如 openEuler 22.03）：**
```bash
# 安装兼容 glibc 2.34 的版本
pip install semgrep==1.120.0

# 安装 setuptools（新版本移除了 pkg_resources）
pip install "setuptools<80"
```

### 验证安装

```bash
semgrep --version
```

## 使用方法：（推荐Bash 脚本）

```bash
./run_magic_num_tests.sh [OPTIONS] [TARGET_PATH]
```
**Bash 脚本参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `TARGET_PATH` | 检测目标（文件或目录路径） | `omni_cache` |
| `--strict` | 严格模式：0, 1, -1 也作为 WARNING | 否 |
| `--diff` | 增量检测：只检测当前分支未提交的变更文件 | 否 |
| `--output DIR` | HTML 输出目录 | `tests/htmlcov_magic_num` |
| `--config FILE` | Semgrep 规则配置文件 | `magic_rule.yaml` |
| `--help` | 显示帮助信息 | - |

**示例：**

```bash
# 默认模式检测 omni_cache 目录
./run_magic_num_tests.sh

# 严格模式（INFO级别报警提升到Warning级别）
./run_magic_num_tests.sh --strict

# 增量检测（当前分支未提交的变更）
./run_magic_num_tests.sh --diff --output diff_report

# 严格模式 + 增量检测
./run_magic_num_tests.sh --strict --diff

# 指定检测目标
./run_magic_num_tests.sh /path/to/project
```

## HTML 报告说明

生成的 HTML 报告包含：

- **首页**: 总体统计（WARNING/INFO 分开）、按规则统计（可点击查看详情）、按文件统计
- **规则详情页**: 点击规则名进入，显示该规则涉及的所有文件
- **文件详情页**: 点击文件名进入，显示代码和问题行高亮
- **颜色区分**: 黄色 = WARNING，蓝色 = INFO（仅默认模式）

报告位置：`tests/htmlcov_magic_num/index.html`

### 查看报告

**VSCode (推荐):**
1. 安装 [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) 扩展
2. 打开 `tests/htmlcov_magic_num/index.html`
3. 右键选择 "Show Preview" 或点击右上角预览按钮
4. 在预览中点击文件名查看具体问题

**Python HTTP 服务器:**
```bash
# 在报告目录启动 HTTP 服务器
cd tests/htmlcov_magic_num && python -m http.server 8081
# 然后在浏览器打开 http://localhost:8081
```

**直接打开:**
```bash
# 复制报告到本地机器后直接在浏览器打开
scp -r server:/path/to/htmlcov_magic_num ./
# 然后打开 htmlcov_magic_num/index.html
```

## 示例输出

```bash
$ ./run_magic_num_tests.sh

============================================================
Semgrep 魔鬼数字检测
============================================================
  目标路径: /data/project/omni_cache
  规则文件: /data/project/tests/magic_num/magic_rule.yaml
  输出目录: /data/project/tests/htmlcov_magic_num
  严格模式: false
  增量检测: false
============================================================

检测模式: 默认模式
使用规则: magic_rule.yaml
正在检测...

============================================================
HTML 报告已生成: tests/htmlcov_magic_num/index.html
============================================================
  模式: 默认模式
  WARNING: 89 个
  INFO:    120 个 (0, 1, -1)
  涉及文件: 45 个
============================================================
```

```bash
$ ./run_magic_num_tests.sh --diff --strict

============================================================
Semgrep 魔鬼数字检测
============================================================
  目标路径: /data/project/omni_cache
  规则文件: /data/project/tests/magic_num/magic_rule.yaml
  输出目录: /data/project/tests/htmlcov_magic_num
  严格模式: true
  增量检测: true
============================================================

[增量检测] 当前分支: master
发现 3 个变更文件未commit
检测模式: 严格模式
使用规则: magic_rule.yaml
正在检测...

============================================================
HTML 报告已生成: tests/htmlcov_magic_num/index.html
============================================================
  模式: 严格模式
  WARNING: 8 个
  INFO:    0 个 (0, 1, -1)
  涉及文件: 2 个
  变更文件: 3 个
============================================================
```

## 参考资料

- [Semgrep 官方文档](https://semgrep.dev/docs/)
- [Semgrep 规则语法](https://semgrep.dev/docs/writing-rules/rule-syntax/)
- [Semgrep Pattern 语法](https://semgrep.dev/docs/writing-rules/pattern-syntax/)