# copy_data：镜像构建所需的本地安装包

本目录用于存放 **L1 镜像（Dockerfile.base）构建时必须从本机拷贝进去的安装包**。  
构建时 Dockerfile 会执行 `COPY copy_data /workspace/copy_data/`，因此所有 `.run` / `.whl` 必须直接放在本目录下（不要再套一层无关子目录），文件名需能被脚本按通配符匹配到。

> 自定义算子源码、编包脚本、无法 git 拉取的代码仓库，请放到 [`docker/codes/`](../codes/README.md)，**不要**放到本目录。

---

## 1. 当前推荐版本（aarch64 / Python 3.11）

以下为本仓库当前文档化的一组配套包，架构为 `aarch64`、Python 为 `3.11`（对应 `cp311`）：

| 类型 | 示例文件名 | 下载入口 |
| --- | --- | --- |
| CANN Toolkit | `Ascend-cann-toolkit_8.3.T1_linux-aarch64.run` | [昇腾 CANN 社区版下载](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| CANN Kernels（Atlas A3 / 910C） | `Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run` | 同上 |
| CANN NNAL（可选） | `Ascend-cann-nnal_*.run` | 同上 |
| torch_npu | `torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl` | [Ascend/pytorch Releases](https://gitcode.com/Ascend/pytorch/releases) |
| torchvision | `torchvision-0.24.0-cp311-cp311-linux_aarch64.whl` | [PyTorch torchvision 官方 wheel](https://download.pytorch.org/whl/torchvision/) |
| torch（仅 x86_64 需要本地提供） | `torch-2.9.0-cp311-cp311-linux_x86_64.whl` | [PyTorch 官方 wheel](https://download.pytorch.org/whl/cpu/) |

换版本时必须保证 **CANN ↔ torch_npu ↔ torch ↔ torchvision ↔ Python / 架构** 互相匹配。  
每个 torch_npu Release 页面都会写明配套的 CANN 版本，请以 [gitcode 发行说明](https://gitcode.com/Ascend/pytorch/releases) 为准。

---

## 2. 如何下载 CANN 包

下载地址：

[https://www.hiascend.com/zh/developer/download/community/result?module=cann](https://www.hiascend.com/zh/developer/download/community/result?module=cann)

按页面筛选项逐项选择后下载 `.run` 安装包：

1. **软件版本**：与 torch_npu 发行说明中的配套 CANN 版本一致（当前文档示例为 `8.3.T1`）。
2. **芯片 / 产品形态**：
   - Atlas A3 训练/推理系列（对应构建参数 `--npu-platform 910C`）→ 下载 `Atlas-A3-cann-kernels_*.run`
   - Atlas A2 系列（对应 `--npu-platform 910B`）→ 下载 `Ascend-cann-kernels-910b_*.run` 或页面上对应的 A2 kernels 包
3. **系统架构**：`linux-aarch64` 或 `linux-x86_64`，必须与 `--arch` 一致。
4. **需要下载的包**（整包安装 `whole`，这也是脚本默认模式）：
   - **必选**：`Ascend-cann-toolkit_<version>_linux-<arch>.run`
   - **必选**：对应芯片的 Kernels / ops 包（`Atlas-A3-cann-kernels_*.run` 或 `A*-cann-kernels*.run` / `A*-ops*.run`）
   - **可选**：`Ascend-cann-nnal_*.run`（加速库，有则安装）

如果使用 `--cann-install-mode split`（分包安装），需要把页面上的分包全部放到本目录，脚本会按文件名匹配安装：

| 分包文件名模式 | 说明 |
| --- | --- |
| `CANN-toolkit-*.run` | 开发套件 |
| `CANN-runtime-*.run` | 运行时 |
| `CANN-compiler-*.run` | 编译器 |
| `CANN-opp-*.run` | 算子包 |
| `CANN-hccl-*.run` | 集合通信 |
| `CANN-aoe-*.run` | AOE |
| `Ascend*-opp_kernel-*.run` | 二进制 kernel |
| `tfadapter*` | TensorFlow adapter（可选，存在则安装） |
| `Ascend-cann-nnal*.run` | NNAL（可选） |

下载完成后把 `.run` 文件直接放到本目录，保持官网原始文件名，不要重命名成无法匹配通配符的名字。

---

## 3. 如何下载 torch_npu

下载地址：

[https://gitcode.com/Ascend/pytorch/releases](https://gitcode.com/Ascend/pytorch/releases)

操作步骤：

1. 打开发行版列表，找到与目标 PyTorch 版本对应的 Release。  
   例如当前文档使用 torch 2.9.0，应选择名称中带 `pytorch2.9.0` 的发行版（如 `v26.0.0-pytorch2.9.0`），并下载其中的 `torch_npu-2.9.0.post2-...whl`。
2. 在 Release 的 **Assets / 下载** 中选择 **Python 版本 + CPU 架构** 匹配的 wheel：
   - Python 3.11 + aarch64：`torch_npu-*-cp311-*-aarch64.whl`
   - Python 3.11 + x86_64：`torch_npu-*-cp311-*-x86_64.whl`
3. 阅读该 Release 的配套版本说明，确认 CANN 版本，再回到上一节下载对应 CANN 包。
4. 将下载好的 `torch_npu-*.whl` 放到本目录。

> Dockerfile.base 通过 `find` 匹配 `torch_npu-*aarch64.whl` 或 `torch_npu-*x86_64.whl`，同一架构请只放一个 torch_npu wheel，避免匹配到错误文件。

---

## 4. 如何下载 torchvision / torch

- **torchvision**  
  官方索引：[https://download.pytorch.org/whl/torchvision/](https://download.pytorch.org/whl/torchvision/)  
  当前文档组合：`torchvision-0.24.0` 依赖 `torch-2.9.0`。  
  aarch64 示例：`torchvision-0.24.0-cp311-cp311-linux_aarch64.whl`

- **torch**
  - **aarch64**：只需提供 `torch_npu`。安装 `torch_npu` 时会自动拉取对应版本的 `torch`，不必把 torch wheel 放到本目录。
  - **x86_64**：必须自行下载与 `torch_npu` / `torchvision` 版本一致的 `torch-*-x86_64.whl`，放到本目录。推荐从 [https://download.pytorch.org/whl/cpu/](https://download.pytorch.org/whl/cpu/) 获取 CPU 版 wheel（昇腾场景使用 NPU，不使用 CUDA 版 torch）。

版本对应关系（常用）：

| torch | torchvision |
| --- | --- |
| 2.9.0 | 0.24.0 |
| 2.8.0 | 0.23.0 |
| 2.7.x | 0.22.x |

Python 版本必须与镜像内 Python 一致。脚本默认 `--python-version 3.11.12`，因此 wheel 应为 `cp311`。

---

## 5. 本目录最终应有的文件

### 5.1 aarch64 + 整包安装（默认，最常见）

```text
docker/copy_data/
├── README.md                                          # 本说明
├── Ascend-cann-toolkit_8.3.T1_linux-aarch64.run       # 必选
├── Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run     # 必选（A3/910C）
├── Ascend-cann-nnal_*.run                             # 可选
├── torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl
└── torchvision-0.24.0-cp311-cp311-linux_aarch64.whl
```

### 5.2 x86_64 + 整包安装

在 5.1 的基础上：

- 所有 `.run` / `.whl` 换成 `x86_64` 架构
- **额外**放入 `torch-2.9.0-cp311-cp311-linux_x86_64.whl`

### 5.3 架构差异小结

| 包 | aarch64 | x86_64 |
| --- | --- | --- |
| CANN toolkit / kernels / nnal | 需要，架构后缀 `aarch64` | 需要，架构后缀 `x86_64` |
| torch_npu | 需要 | 需要 |
| torchvision | 需要 | 需要 |
| torch | 不需要（随 torch_npu 自动安装） | **必须本地提供** |

---

## 6. 文件名匹配规则（请勿随意改名）

`Dockerfile.base` 按文件名通配符查找，改名会导致构建失败：

| 用途 | 匹配规则 |
| --- | --- |
| torch_npu | `torch_npu-*aarch64.whl` 或 `torch_npu-*x86_64.whl` |
| torchvision | `torchvision-*aarch64.whl` 或 `torchvision-*x86_64.whl` |
| torch（仅 x86_64） | `torch-*x86_64.whl` |
| Toolkit（whole） | `Ascend-cann-toolkit_*.run` |
| Kernels / ops（whole） | `A*-ops*.run` 或 `A*-cann-kernels*.run` |
| NNAL | `Ascend-cann-nnal*.run` |

建议：下载后保持官网原始文件名，直接拷贝到本目录即可。
