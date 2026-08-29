# codes：L2 镜像源码、算子脚本和可选安装包

`docker/codes/` 会被 `Dockerfile.omniinfer` 复制到容器内的
`/workspace/dist/codes/`。本目录按照 omni-ops 原始编包环境划分为源码、编译脚本和
编译产物三个区域。

CANN、torch、torch_npu、torchvision 等 L1 安装包应放在
[`docker/copy_data/`](../copy_data/README.md)，不要放在本目录。

## 1. 目录结构

```text
docker/codes/                         # 工作根目录
├── README.md
├── code/                             # 自定义算子源码
│   └── omni-ops/                     # https://gitee.com/omniai/omni-ops
├── ops_build_images/                 # 自定义算子编包脚本
│   ├── build_omni-ops_packages.sh    # 统一编译入口
│   └── ops_scripts/
│       ├── build_omni-ops_inference_release.sh
│       ├── build_omni-ops_training_release.sh
│       └── scan_op.py
├── ops-packages/                     # 编译产物，构建时自动创建
├── install_ops_by_whl.sh             # 安装预编译或本次生成的算子包
├── build_info.txt                    # 可选
├── cann-hixl_*.run                   # 可选
├── moxing_framework-*.whl            # 可选
├── vllm/                             # 可选离线源码
└── omniinfer/                        # 可选离线源码
```

容器内对应路径：

```text
/workspace/dist/codes/
├── code/omni-ops/
├── ops_build_images/
└── ops-packages/
```

`code/omni-ops/` 和 `ops-packages/` 已加入 `.gitignore`，不会把上游源码或编译产物
提交到当前仓库。

## 2. 准备 omni-ops 源码

在当前仓库根目录执行：

```bash
git clone https://gitee.com/omniai/omni-ops.git docker/codes/code/omni-ops
git -C docker/codes/code/omni-ops checkout <已验证的-tag-或-commit>
```

生产构建应固定已验证的 tag 或 commit，不建议直接依赖持续变化的 `master`。

当前 release 脚本会使用 `sed` 修改 omni-ops 中的 `CMakeLists.txt` 和 `setup.py`。
如需更换版本或编译参数，建议重新准备干净源码。

## 3. 在 Dockerfile 中编译推理和训练算子

`Dockerfile.omniinfer` 会按 `--custom-ops` 指定的名称，依次在以下位置查找脚本：

```text
/workspace/dist/codes/<脚本名>.sh
/workspace/dist/codes/ops_build_images/<脚本名>.sh
```

因此仍然只需传脚本基本名称：

```bash
cd docker

bash docker_build_run.sh \
  --build-target L2 \
  --L1-image new-infer-meddle:0.1 \
  --L2-image new-infer-omniinfer:0.1 \
  --custom-ops build_omni-ops_packages \
  --npu-platform 910C \
  --start-server False
```

Dockerfile 实际执行：

```bash
/workspace/dist/codes/ops_build_images/build_omni-ops_packages.sh \
  --npu-platform 910C
```

入口脚本检测到长选项后，会在当前 L2 构建容器内直接编译，不会再次执行
`docker run`。默认同时编译推理和训练算子，然后安装生成的 `.run` 和 `.whl`。

平台映射：

- `910B` → `ascend910b`
- `910C` → `ascend910_93`

### Dockerfile 模式选项

```text
--omni-ops-path <path>       omni-ops 源码路径
                             默认 /workspace/dist/codes/code/omni-ops
--npu-platform <910B|910C>   NPU 平台，默认 910C
--build-type <type>          inference、training 或 both，默认 both
--project-version <version>  release-* 或 poc-*，默认 poc-0.8.0rc1
--cann-version <version>     CANN 版本，默认 8.5RC1
--pta-version <version>      PTA 版本，默认 2.6.0
```

手动只编译推理算子：

```bash
bash /workspace/dist/codes/ops_build_images/build_omni-ops_packages.sh \
  --omni-ops-path /workspace/dist/codes/code/omni-ops \
  --npu-platform 910C \
  --build-type inference \
  --project-version release-1.5.0 \
  --cann-version 9.1.0 \
  --pta-version 2.9.0
```

## 4. 在宿主机启动独立编译容器

入口脚本保留了原始位置参数模式。该模式负责清理同名容器、启动编译镜像、挂载工作
目录、配置内部 pip 源、安装编译依赖，并调用指定 release 脚本。

### 环境要求

- 已安装 Docker，当前用户具有 Docker 执行权限。
- 编译镜像包含 CANN Toolkit、Python、PyTorch、torch_npu 和 AscendC 编译工具。
- 执行机器能够访问 `http://mirrors.tools.huawei.com/pypi/simple/`。
- 工作根目录可读写，并同时包含源码和编包脚本。

### 用法

```bash
bash ops_build_images/build_omni-ops_packages.sh \
  <container_name> \
  <image_name> \
  <DIR_PATH> \
  <build_script_file> \
  [COMPILE_UNIT] \
  [project_version] \
  [cann_version] \
  <ops_local_path_suffix> \
  <ops_work_path> \
  [pta_version]
```

参数说明：

1. `container_name`：临时 Docker 容器名；同名旧容器会先被清理。
2. `image_name`：包含 CANN/PTA 编译环境的 L1 镜像。
3. `DIR_PATH`：容器内 omni-ops 源码绝对路径，例如
   `/data/ops_work/code/omni-ops`。
4. `build_script_file`：`ops_scripts/` 下的 release 脚本文件名。
5. `COMPILE_UNIT`：默认 `ascend910_93`；支持 `ascend910b`、
   `ascend910_93`、`ascend950`。
6. `project_version`：默认 `poc-0.8.0rc1`；必须以 `release-` 或 `poc-` 开头。
7. `cann_version`：默认 `8.5RC1`，用于产物名称和 wheel 版本。
8. `ops_local_path_suffix`：`ops-packages/` 下的产物子目录。
9. `ops_work_path`：宿主机工作根目录，以相同绝对路径挂载到容器。
10. `pta_version`：默认 `2.6.0`。

`DIR_PATH` 和 `ops_build_images/` 必须都位于 `ops_work_path` 中，否则容器无法访问。

### 宿主机工作目录示例

```text
/data/ops_work/
├── code/
│   └── omni-ops/
├── ops_build_images/
│   ├── build_omni-ops_packages.sh
│   └── ops_scripts/
└── ops-packages/
```

### 编译训练算子

```bash
cd /data/ops_work

bash ops_build_images/build_omni-ops_packages.sh \
  20260826045008_train \
  <包含CANN和PTA的L1编译镜像> \
  /data/ops_work/code/omni-ops \
  build_omni-ops_training_release.sh \
  ascend910_93 \
  poc-1.5.0rc1 \
  9.1.0 \
  develop/20260826/a3/training \
  /data/ops_work \
  2.9.0
```

### 编译推理算子

```bash
cd /data/ops_work

bash ops_build_images/build_omni-ops_packages.sh \
  20260826045008_inference \
  <包含CANN和PTA的L1编译镜像> \
  /data/ops_work/code/omni-ops \
  build_omni-ops_inference_release.sh \
  ascend910_93 \
  release-1.5.0 \
  9.1.0 \
  develop/20260826/a3/inference \
  /data/ops_work \
  2.9.0
```

## 5. 编译执行流程

宿主机模式：

1. 执行 `docker rm -f` 清理同名旧容器；容器不存在时不会报错退出。
2. 使用 `--privileged --ipc=host --shm-size=128g` 启动临时容器。
3. 将 `ops_work_path` 挂载到容器内同名绝对路径。
4. 加载 `~/.bashrc` 和 `/etc/profile`。
5. 配置内部 pip 镜像并安装编译依赖。
6. 调用 `ops_build_images/ops_scripts/` 下指定的 release 脚本。

Dockerfile 模式：

1. 根据 `--npu-platform` 得到 Ascend 编译单元。
2. 默认依次调用推理和训练 release 脚本。
3. release 脚本编译 AscendC 和 PyTorch 扩展。
4. 训练脚本额外尝试编译 Triton、PyPTO 包。
5. 产物复制到 `ops-packages/`。
6. 调用 `install_ops_by_whl.sh` 安装推理和训练算子包。

## 6. 编译产物

推理：

```text
ops-packages/<ops_local_path_suffix>/
├── cann<芯片>-omni_inference_custom_ops-*.run
└── omni_inference_ascendc_custom_ops-*.whl
```

训练：

```text
ops-packages/<ops_local_path_suffix>/
├── cann<芯片>-omni_training_custom_ops-*.run
├── omni_training_ascendc_custom_ops-*.whl
├── omni_training_triton_custom_ops-*.whl
├── omni_training_triton_custom_ops_*.tar/.tar.gz/.zip
├── omni_training_pypto_custom_ops-*.whl
└── omni_training_pypto_custom_ops_*.tar/.tar.gz/.zip
```

芯片名称由编译单元推导：

- `ascend910b` → `910B`
- `ascend910_93` → `910C`
- `ascend950` → `910D`

产物根目录的计算方式：

```text
$DIR_PATH/../../ops-packages/<ops_local_path_suffix>/
```

当 `DIR_PATH=/data/ops_work/code/omni-ops` 时，产物位于：

```text
/data/ops_work/ops-packages/<ops_local_path_suffix>/
```

在 Dockerfile 模式下，产物位于：

```text
/workspace/dist/codes/ops-packages/inference/
/workspace/dist/codes/ops-packages/training/
```

训练脚本只强制检查 AscendC `.run` 和 PTA `.whl`。Triton、PyPTO 编译和复制使用
`|| true`，主脚本成功不代表这些附加产物一定存在。

## 7. 直接安装预编译包

如果不需要从源码编译，可将匹配的 `.run` 和 `.whl` 放在 `docker/codes/` 或其一层
子目录，再由 `install_ops_by_whl.sh` 安装。

支持的文件名：

- 训练 `.run`：`*omni_training_custom_ops*.run`
- 训练 wheel：`*omni_training_ascendc_custom_ops*.whl`
- 推理 `.run`：`*omni_inference_custom_ops*.run`
- 推理 wheel：`*omni_inference_ascendc_custom_ops*.whl`
- 第三方 `.run`：`CANN-custom_ops*.run`
- 第三方 wheel：`custom_ops*.whl`

同一组 `.run` 和 `.whl` 必须同时存在，缺少任意一个都会跳过该组。

## 8. scan_op.py 算子扫描

release 脚本会执行：

```bash
python3 ops_build_images/ops_scripts/scan_op.py \
  /data/ops_work/code/omni-ops \
  inference \
  ascend910_93
```

扫描路径：

```text
<omni-ops>/<inference或training>/ascendc/src/**/op_host/*_def.cpp
```

`devices` 参数支持逗号分隔多个设备：

```bash
python3 ops_build_images/ops_scripts/scan_op.py \
  /data/ops_work/code/omni-ops \
  inference \
  "ascend910b,ascend910_93"
```

当前 release 脚本会保存扫描结果，但实际执行的是全量构建：

```bash
bash build.sh --compute-unit "${COMPILE_UNIT}"
```

如果需要按扫描结果编译，应检查结果非空后改为：

```bash
bash build.sh -n "${ops}" --compute-unit "${COMPILE_UNIT}"
```

## 9. 其它可选文件

- `cann-hixl*.run`：在 L2 OpenAI 阶段安装 CANN HiXL。
- `moxing_framework*.whl`：安装 moxing 并写入 `MOX_OBS_SIGNATURE=obs`。
- `*/obsutil*`：构建时移动到 `/workspace` 并添加执行权限。
- `build_info.txt`：追加到镜像内 `/workspace/build_info.txt`。
- `vllm/`：构建机无法访问 GitHub 时提供离线 vLLM Git 仓库。
- `omniinfer/`：构建机无法访问 Gitee 时提供离线 omniinfer Git 仓库。

## 10. 注意事项和常见错误

- release 脚本当前包含硬编码代理地址及凭据。应改用环境变量，并轮换已经提交或共享
  过的凭据。
- 宿主机模式使用内部 PyPI 镜像；非对应网络环境需要更换软件源。
- `ops_work_path` 必须使用绝对路径，并与容器内路径保持一致。
- `build_script_file` 只传文件名，不要添加 `ops_scripts/` 前缀。
- `project_version` 的 `release-` 前缀生成正式包，`poc-` 前缀生成带日期的开发包。
- `bisheng compilation tool not found`：CANN 环境未加载或镜像不含开发套件。
- `No *.run package found`：AscendC 编译失败或文件名与源码版本不匹配。
- `scan_op.py` 输出为空：检查 `.AddConfig(...)` 和 `op_host/*_def.cpp`。
- `Unsupported SocVersion`：编译单元与 CANN/源码支持范围不一致。
- 找不到 wheel：检查 `setup.py` 的包名和版本字符串是否成功替换。
- Triton/PyPTO 产物缺失但脚本成功：检查被 `|| true` 忽略的上游错误。

## 11. 产物校验

```bash
find docker/codes/ops-packages -maxdepth 3 -type f -print
```

检查 wheel：

```bash
python3 -m zipfile -t \
  docker/codes/ops-packages/inference/omni_inference_ascendc_custom_ops-*.whl

python3 -m zipfile -t \
  docker/codes/ops-packages/training/omni_training_ascendc_custom_ops-*.whl
```
