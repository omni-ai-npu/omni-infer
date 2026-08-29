## 安装前准备

构建镜像前，先按架构准备好本地安装包，并放到 `docker/copy_data/`。  
各包的下载入口、筛选项、文件名匹配规则见 **[copy_data/README.md](./copy_data/README.md)**。

当前文档示例组合（**aarch64 + Python 3.11**）：

| 包 | 示例文件 | 下载地址 |
| --- | --- | --- |
| CANN Toolkit | `Ascend-cann-toolkit_8.3.T1_linux-aarch64.run` | [昇腾 CANN 社区版](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| CANN Kernels | `Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run` | 同上 |
| torch_npu | `torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl` | [Ascend/pytorch Releases](https://gitcode.com/Ascend/pytorch/releases) |
| torchvision | `torchvision-0.24.0-cp311-cp311-linux_aarch64.whl` | [PyTorch torchvision wheel](https://download.pytorch.org/whl/torchvision/) |
| 自定义算子 / 离线源码 | 算子源码 + 编包脚本等 | 自行准备，放到 `docker/codes/`，详见 [codes/README.md](./codes/README.md) |

### CANN 包怎么下

1. 打开 [CANN 社区版下载页](https://www.hiascend.com/zh/developer/download/community/result?module=cann)。
2. 选择与 torch_npu 配套的 CANN 版本（当前示例 `8.3.T1`；以 [torch_npu 发行说明](https://gitcode.com/Ascend/pytorch/releases) 中的「配套版本」为准）。
3. 按芯片选型：A3/910C 下 `Atlas-A3-cann-kernels_*.run`；A2/910B 下对应的 A2 kernels。
4. 架构选 `linux-aarch64` 或 `linux-x86_64`，与 `--arch` 一致。
5. **整包模式（默认 `--cann-install-mode whole`）**至少需要：
   - `Ascend-cann-toolkit_<version>_linux-<arch>.run`
   - 对应芯片的 `*cann-kernels*.run` 或 `*ops*.run`
   - 可选：`Ascend-cann-nnal_*.run`
6. 将 `.run` 文件放到 `docker/copy_data/`，保持官网原始文件名。

分包安装（`--cann-install-mode split`）需要的文件列表见 [copy_data/README.md](./copy_data/README.md)。

### torch_npu 怎么下

1. 打开 [https://gitcode.com/Ascend/pytorch/releases](https://gitcode.com/Ascend/pytorch/releases)。
2. 选择与目标 PyTorch 版本对应的 Release（当前示例为 torch 2.9.0，选名称含 `pytorch2.9.0` 的发行版，下载 `torch_npu-2.9.0.post2-...whl`）。
3. Assets 中按 **Python 版本 + 架构** 选 wheel：
   - Python 3.11 + aarch64：`torch_npu-*-cp311-*-aarch64.whl`
   - Python 3.11 + x86_64：`torch_npu-*-cp311-*-x86_64.whl`
4. 同一架构只放一个 torch_npu wheel 到 `docker/copy_data/`。

### 注意

1. **aarch64** 只需提供对应版本的 `torch_npu` 和 `torchvision`，安装 `torch_npu` 时会自动下载匹配的 `torch`；**x86_64 必须自行提供**同版本的 `torch-*-x86_64.whl`。
2. `torchvision` 与 `torch` 有严格版本对应。当前示例：`torchvision-0.24.0` 依赖 `torch-2.9.0`。
3. wheel / `.run` 等本地安装包统一放在 **`docker/copy_data/`**（相对本仓库的 `docker` 目录）。
4. 自定义算子源码、编包脚本、预编译算子包、以及无法 git 拉取的代码，统一放在 **`docker/codes/`**，目录约定见 [codes/README.md](./codes/README.md)。
5. 镜像内默认 Python 为 `3.11.12`（`--python-version`），因此 wheel 一般选 `cp311`。

准备完成后，`docker/copy_data/` 在 aarch64 整包模式下至少应类似：

```text
docker/copy_data/
├── Ascend-cann-toolkit_8.3.T1_linux-aarch64.run
├── Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run
├── torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl
└── torchvision-0.24.0-cp311-cp311-linux_aarch64.whl
```

## 镜像分层

下面说明脚本中涉及的镜像分层（层次越高依赖越低层），便于理解 L1 / L2 构建的职责和包含的软件包。

- 基础镜像 (`BASE_IMAGE`)
	- 说明：系统级基础镜像，通常由用户预先提供或基于官方发行版构建（脚本默认 `test-infer-base:0.1`）。
	- 要求：已安装与 `--python-version` 匹配的 Python，并能使用 `yum` / `pip`。

- L1 镜像（由 `Dockerfile.base` 生成，脚本变量名为 `L1_IMAGE`）
	- 说明：在 `BASE_IMAGE` 上安装 CANN、`torch_npu`、`torchvision`（以及 x86_64 下的 `torch`）和内核相关组件。
	- 作用：为推理与硬件加速提供运行时环境（Ascend runtime、toolkit、kernels、`torch_npu` 等），作为可复用的中间层供 L2 依赖。
	- 输入包来源：`docker/copy_data/`。

- L2 镜像（由 `Dockerfile.omniinfer` 生成，脚本变量名为 `L2_IMAGE`）
	- 说明：应用层 / apiserver 镜像，基于 L1 构建，包含 omniinfer 服务、Python 依赖、模型服务与用户级工具。
	- 作用：打包并暴露模型推理服务（apiserver / omniinfer）、管理脚本、自定义算子以及运行时第三方 Python 包，直接用于运行容器。
	- 输入来源：`docker/codes/`、`docker/requirements/`，以及 `--branch` 指定的 omniinfer 源码。

## 镜像一键构建

`docker_build_run.sh` 支持一键构建：可单独制作 L1 / L2，支持 CANN 整包或分包安装，支持自定义算子（需提供算子包和对应 build 脚本）。

在 `docker/` 目录下执行：

```bash
cd docker
bash docker_build_run.sh [options]
```

### docker_build_run 脚本参数说明

`docker_build_run.sh` 中参数的各个字段说明如下：

| 字段 | 含义 |
| :--- | :--- |
| `--arch <arch>` | 目标构建平台架构：`aarch64` 或 `x86_64`。默认：`aarch64` |
| `--proxy <proxy>` | 构建时 HTTP 代理（拉取外部资源）。示例：`http://user:pass@host:port/` |
| `--hugging-face-proxy <proxy>` | 运行时容器下载模型用的 HTTP 代理。默认同 `--proxy` |
| `--pip-index-url <url>` | Docker 构建中 pip 索引地址。默认：`https://mirrors.huaweicloud.com/repository/pypi/simple` |
| `--pip-trusted-host <host>` | pip trusted host，避免证书或主机校验失败。默认：`mirrors.huaweicloud.com` |
| `--model-name <name>` | 运行时要下载或启动的模型名（例如 `"Qwen/Qwen2.5-0.5B"`），会传给容器。 |
| `--cann-install-mode <split\|whole>` | L1 构建时 CANN 安装方式：`whole`（整包，默认）或 `split`（分包）。 |
| `--base-image <image>` | 构建 L1 时使用的系统基础镜像 tag。默认：`test-infer-base:0.1` |
| `--L1-image <image>` | L1 构建完成后的镜像 tag。默认：`test-infer-meddle:0.1` |
| `--L2-image <image>` | L2 构建完成后的镜像 tag。默认：`test-infer-omniinfer:0.1` |
| `--branch <tag>` | 打进镜像的 omniinfer 源码分支或 tag。默认：`master` |
| `--custom-ops <ops>` | 要编进镜像的自定义算子（逗号分隔的脚本名，不含 `.sh`）。默认空。 |
| `--npu-platform <platform>` | 编自定义算子时的硬件平台：`910B` 或 `910C`。默认：`910C` |
| `--python-version <version>` | 构建时使用的 Python 版本。默认：`3.11.12`（对应 wheel 的 `cp311`） |
| `--start-server <True\|False>` | 构建结束后是否启动容器执行 `start_server.sh`。默认：`True` |
| `--build-target <L1\|L2\|both\|skip>` | 构建目标：`L1` 只构建 `Dockerfile.base`；`L2` 只构建 `Dockerfile.omniinfer`；`both` 先 L1 再 L2（默认）；`skip` 跳过构建。 |
| `--build-network <green\|blue>` | L2 构建时 Rust / Cargo 网络环境。默认：`green` |
| `--vllm-version <version>` | 安装的 vLLM 版本。默认：`v0.14.0` |
| `--install-modules <modules>` | 要安装的 omniinfer 模块，逗号分隔。默认：`omni-proxy`；`omni-npu` 已由 omniinfer 自身集成，无需再传入。 |
| `--skip-pull <True\|False>` | 是否向 omniinfer `build/build.sh` 传递 `--skip-pull`。默认：`false` |
| `--build-for-roma <True\|False>` | 是否继续构建 Roma 镜像。默认：`false` |
| `--roma-image <image>` | Roma 镜像 tag。默认：`test-infer-ROMA:0.1` |

### 命令执行

下面给出若干常见示例——把占位符替换为实际镜像 tag、模型名、自定义算子包以及构建目标。

**示例 1 — 全量构建（默认 both），并指定自定义镜像 tag**：

```bash
bash docker_build_run.sh \
	--arch aarch64 \
	--base-image new-infer-base:0.1 \
	--L1-image new-infer-meddle:0.1 \
	--L2-image new-infer-omniinfer:0.1 \
	--model-name "Qwen/Qwen2.5-0.5B" \
	--branch master
```

该示例会串行构建 L1 和 L2：先执行 `Dockerfile.base`，基础镜像为 `new-infer-base:0.1`（需自行提供），输出 L1：`new-infer-meddle:0.1`；再执行 `Dockerfile.omniinfer`，输入为该 L1，输出 L2：`new-infer-omniinfer:0.1`。镜像内 omniinfer 代码取 `master` 分支。

**示例 2 — 仅构建 L1（只构建 Dockerfile.base），并使用 split 模式安装 CANN**：

```bash
bash docker_build_run.sh --build-target L1 --cann-install-mode split \
	--pip-index-url "https://mirrors.huaweicloud.com/repository/pypi/simple" \
	--pip-trusted-host "mirrors.huaweicloud.com" \
	--L1-image new-infer-meddle:0.1
```

该示例会跳过 `Dockerfile.omniinfer`，只输出 L1（`new-infer-meddle:0.1`）。若不指定 `--L1-image`，则使用默认 tag `test-infer-meddle:0.1`。若需要指定基础镜像，请同时提供 `--base-image`，否则会使用默认的 `test-infer-base:0.1`。分包所需文件见 [copy_data/README.md](./copy_data/README.md)。

**示例 3 — 仅构建 L2（跳过 base），加入自定义算子包并指定源码版本**：

```bash
bash docker_build_run.sh --build-target L2 \
	--L1-image new-infer-meddle:0.1 \
	--L2-image test-infer-omniinfer:latest \
	--branch "dev_v1.0.0" \
	--custom-ops build_omni-ops_packages \
	--npu-platform 910C \
	--start-server False
```

该示例会跳过 `Dockerfile.base`，只输出 L2（`test-infer-omniinfer:latest`）。**必须**提供已存在的 `--L1-image`，否则会使用默认 `test-infer-meddle:0.1` 导致构建失败。镜像构建完后不启动容器。

关于 `--custom-ops`：传入 `build_omni-ops_packages` 会执行
`docker/codes/ops_build_images/build_omni-ops_packages.sh`，并传入
`--npu-platform <910B|910C>`，在当前 L2 构建容器内编译并安装 omni-ops 推理和
训练算子。源码目录结构、预编译 `.run`/`.whl` 命名、HiXL / moxing / 离线 vLLM 等可选文件，见
**[codes/README.md](./codes/README.md)**。

## 容器启动

当前镜像默认将 `ENTRYPOINT` 设置为 `start_server.sh`，因此若只想进 shell，启动容器时需要加 `--entrypoint=bash`。也可以自行覆盖 `ENTRYPOINT`。

一键脚本在 `--start-server True`（默认）时，会用类似下面的方式拉起服务：

```bash
docker run --rm -it --shm-size=500g \
	--net=host --privileged=true \
	--device=/dev/davinci_manager \
	--device=/dev/hisi_hdc \
	--device=/dev/devmm_svm \
	-e PORT=8301 \
	-e ASCEND_RT_VISIBLE_DEVICES=1 \
	-e HTTP_PROXY="${HUGGING_FACE_PROXY}" \
	-e MODEL_NAME="${MODEL_NAME}" \
	<L2镜像> \
	--model "${MODEL_NAME}"
```

只进入容器、不拉服务：

```bash
docker run --rm -it --privileged --net=host \
	--device=/dev/davinci_manager \
	--device=/dev/hisi_hdc \
	--device=/dev/devmm_svm \
	--entrypoint=bash \
	<L2镜像>
```
