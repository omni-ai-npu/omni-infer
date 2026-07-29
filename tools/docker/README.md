## 安装前准备
安装前需要提供以下本地包（aarch64环境下）：
- [torchvision-0.24.0-cp311-cp311-linux_aarch64.whl](https://download.pytorch.org/whl/torchvision/)
- [torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl](https://download.pytorch.org/whl/cu126/torch/)
- Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run
- Ascend-cann-toolkit_8.3.T1_linux-aarch64.run
- 所需的自定义算子包代码

**注意**： 

1. aarch64环境下只需要提供对应版本的torch_npu，安装时自动下载对应版本的torch包，而x86_64必须自行提供对应版本的torch包；
2. torchvision包和torch包有依赖关系，请下载对应版本的包到本地，当前torchvision-0.24.0依赖torch-2.9.0。
3. whl包等本地安装包统一放在/omniifer/tools/docker/copy_data路径下面。
4. 自定义算子包和对应的编包脚本放在/omniifer/tools/docker/codes路径下，其他无法git需要自行下载的代码也统一放到该路径下。
## 镜像分层

下面简要说明脚本中涉及的镜像分层（层次越高依赖越低层），便于理解 L1/L2 构建的职责和包含的软件包。

- 基础镜像 (BASE_IMAGE)
	- 说明：系统级基础镜像，通常由用户预先提供或基于官方发行版构建。

- L1 镜像（由 Dockerfile.base 生成，脚本中变量名为 L1_IMAGE）
	- 说明：开发/设备层镜像，通常在 BASE_IMAGE 基础上安装 CANN、torch_npu和内核驱动相关组件。
	- 作用：为推理与硬件加速提供完整的运行时环境（例如 Ascend runtime、驱动、torch_npu 等），并打包为可复用的中间层镜像供 L2 依赖。

- L2 镜像（由 Dockerfile.omniinfer 生成，脚本中变量名为 L2_IMAGE）
	- 说明：应用层镜像／apiserver 镜像，基于 L1 镜像构建，包含 omniinfer 服务、Python 依赖、模型服务与用户级工具。
	- 作用：打包并暴露模型推理服务（apiserver/omniinfer）、管理脚本、模型下载配置、自定义算子以及运行时需要的第三方 Python 包，直接用于运行容器并对外提供推理服务。


## 镜像一键构建
该脚本目前支持执行脚本一键构建，支持L1镜像和L2镜像单独制作、支持cann包的整包/分包安装、支持自定义算子包安装（需提供自定义算子包和对应的build脚本）。

### docker_build_run脚本参数说明

`docker_build_run.sh` 中参数的各个字段说明如下：
| 字段                            | 含义                                                                                                                               |
| :-------------------------------- | :-----------------------------------------------------------------------------------------------------------------------------------
| `--arch <arch>`             | 目标构建平台架构（例如: aarch64 或 x86_64）。默认值: aarch64
| `--proxy <proxy>`           | 构建时使用的 HTTP 代理（用于在构建镜像过程中访问外部资源）。示例: http://user:pass@host:port/
| `--hugging-face-proxy <proxy>` | 运行时容器用于模型下载的 HTTP 代理（传递给启动容器的环境变量）。默认同 --proxy
| `--pip-index-url <url>`     | Docker 构建中使用的 pip 索引地址（用于安装 pip 包）。示例: https://mirrors.huaweicloud.com/repository/pypi/simple
| `--pip-trusted-host <host>` | pip 的 trusted host，用于允许在受信任的索引上安装包（避免证书或主机校验失败）。
| `--model-name <name>`       | 运行时要下载或启动的模型名字（例如: "Qwen/Qwen2.5-0.5B"）。此参数会被传递到容器运行时。
| `--cann-install-mode <split\|whole>`| 在构建 Dockerfile.base 时，指定 CANN 包的安装方式：`whole`（使用完整安装包）或 `split`（分包安装）。默认: whole
| `--base-image <image>`      | 指定用于构建的基础镜像标签（传入 Dockerfile.base 或作为上层镜像引用）。
| `--L1-image <image>`        | L1 开发镜像（base/dev）构建完成后打的镜像 tag。默认: test-infer-meddle:0.1
| `--L2-image <image>`        | L2 用户/服务镜像（apiserver/omniinfer）构建完成后打的镜像 tag。默认: test-infer-omniinfer:0.1
| `--branch <tag>`  | 要包含到镜像中的 omni 源码版本或分支（例如: master 或具体 tag）。
| `--custom-ops <ops>`        | 需要加入镜像的自定义算子包（逗号分隔的包名或构建脚本路径）。默认空，即不加入额外自定义算子。
| `--npu-platform <platform> `        | 在构建自定义算子包时，指定当前机器硬件平台类型：910B 或 910C。默认910C。
| `--start-server <True\|False> `        | 指定是否启动容器执行start_server.sh，默认为True。
| `--build-target <L1\|L2\|both>` | 选择要构建的目标：`L1`（只构建 Dockerfile.base）、`L2`（只构建 Dockerfile.omniinfer）、`both`（先构建 L1 再构建 L2，默认）。

### 命令执行

下面给出若干常见的执行示例——把示例中的占位符替换为实际值：镜像tag、模型名、自定义算子包以及构建目标等。

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
在这个示例中，将会串行构建L1和L2镜像，即执行`Dockerfile.base`和`Dockerfile.omniinfer`，`Dockerfile.base`的输入的基础镜像是`new-infer-base:0.1`(需自行提供，一般使用基础镜像)，输出镜像为L1镜像：`new-infer-meddle:0.1`；`Dockerfile.omniinfer`的输入为`new-infer-meddle:0.1`，输出镜像为`new-infer-omniinfer:0.1`。镜像安装的omniinfer代码为master分支的代码

**示例 2 — 仅构建 L1（只构建 Dockerfile.base），并使用 split 模式安装 CANN**：

```bash
bash docker_build_run.sh --build-target L1 --cann-install-mode split \
	--pip-index-url "https://mirrors.huaweicloud.com/repository/pypi/simple" \
	--pip-trusted-host "mirrors.huaweicloud.com" \
    --L1-image new-infer-meddle:0.1
```

在这个示例中，会跳过`Dockerfile.omniinfer`执行输出L1镜像（`new-infer-meddle:0.1`），若不指定，则会按默认tag名输出`test-infer-meddle:0.1`镜像。注意，若需要指定基础镜像，也需要提供`--base-image`入参，否则会使用默认的`new-infer-base:0.1`镜像名。

**示例 3 — 仅构建 L2（跳过 base），加入自定义算子包并指定源码版本**：

```bash
bash docker_build_run.sh --build-target L2 \
	--L1-image new-infer-meddle:0.1 \
	--L2-image test-infer-omniinfer:latest \
	--omni-version-num "dev_v1.0.0" \
    --custom-ops build_cann_recipes_ops,build_omni_ops \
    --start-server False
```
在这个示例中，会跳过`Dockerfile.base`执行输出L2镜像（`test-infer-omniinfer:latest`）。注意，这里必须提供对应的`--L1-image`入参，否则会使用默认的`test-infer-meddle:0.1`,导致镜像构建失败。
镜像构建完之后不再启动容器执行start_server.sh。

关于`--custom-ops`的使用，自定义算子的传参为`ops1,ops2,...`，其中ops1指的是所要加入的自定义算子，命名为/omniifer/tools/docker/ops_code路径对应的sh脚本文件名。Dockerfile.omniinfer会自动执行传入脚本名对应的自定义算子构建脚本。

## 容器启动

当前的镜像制作默认将ENTRYPOINT设置为`start_server.sh`，因此起容器时需要加上`--entrypoint=bash`指令。ENTRYPOINT也支持自行覆盖。