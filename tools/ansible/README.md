# OmniInfer Ansible 部署

本目录保存当前维护的 OmniInfer 模型部署脚本。部署框架通过数据化 playbook
描述模型、容器和服务参数，通过 `common` role 复用通用任务；只有任务序列本身
存在差异时，才新增专用 role，做法可参考
[`elastic_server`](roles/elastic_server/README.md)。

开发和扩展前请先阅读：

- [OmniInfer Ansible 开发与部署指南](DEVELOPMENT_GUIDE.md)：目录结构、Inventory、
  profiles、阶段派生变量、旧版变量迁移、tags、执行顺序和扩展规范。

当前维护的生产 Playbook 只有：

- [PanguV2](playbooks/omni_infer_server_template_panguv2.yml)
- [DSV32](playbooks/omni_infer_server_template_dsv32.yml)

新增场景可以复制
[通用 Playbook 示例](examples/omni_infer_server_template_example.yml)。
该文件不是生产部署入口。

`template/` 保存重构前的历史部署脚本，仅用于追溯原部署流程，后续不继续维护。
新增部署和问题修复应落在 `playbooks/`、`roles/common/` 或对应的专用 role 中。

## 框架运行链路

当前 Ansible 框架从配置解析到容器服务启动的完整文件链路如下：

```text
inventory.yml
      +
模型 Playbook
      +
Role defaults/main.yml
      │
      ▼
Role tasks/main.yml
      │
      ▼
具体 tasks/*.yml
      │
      ├── 读取 Inventory 拓扑
      ├── 合并公共默认值与 Playbook Profile
      ├── 构造 docker run / docker exec
      └── 调用 templates/*.j2
              │
              ▼
      生成 Shell / JSON / YAML
              │
              ▼
      写入远端 $SCRIPTS_PATH
              │
              ▼
      通过共享挂载进入容器
              │
              ▼
      docker exec 执行生成脚本
              │
              ▼
      Prefill / Decode / Proxy 服务
```

其中，Inventory 描述目标节点和部署拓扑，模型 Playbook 声明环境与模型差异，
Role defaults 提供公共默认值，`tasks/main.yml` 决定任务执行顺序，具体 task
负责计算变量、合并 Profile、渲染模板和执行容器命令。`tools/ansible/template/`
中的旧文件不在这条运行链路中，仅用于追溯重构前的部署实现。

# 环境准备
## 在执行机安装 ansible-playbook
```bash
# 安装ansible-playbook
yum install ansible

# 参考open euler系统的公司内部的yum源
rm  /etc/yum.repos.d/*

echo "[openEuler-everything]
name=openEuler-everything
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/
enabled=1
gpgcheck=0
gpgkey=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/everything/aarch64/RPM-GPG-KEY-openEuler
        
[openEuler-EPOL]
name=openEuler-epol
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/EPOL/main/aarch64/
enabled=1
gpgcheck=0
[openEuler-update]
name=openEuler-update
baseurl=http://mirrors.tools.huawei.com/openeuler/openEuler-22.03-LTS-SP4/update/aarch64/
enabled=1
gpgcheck=0" > /etc/yum.repos.d/openeuler.repo
```

## 在执行机安装 sshpass
执行 ansible 依赖 sshpass 链接各个目标机，做远程机器管理
```bash
yum install openssh-server
```

## 密钥文件的准备
请注意，这里的密钥文件仅仅是用于执行机通过 ansible 去登录目标机，如果你已经有登录目标机的密钥文件，放在 omni_infer_inventory.yml 的 ansible_ssh_private_key_file 指定路径即可。
1. 首先在执行机生成密钥对:
    ```bash
    ssh-keygen -t ed25519 -C "Your SSH key comment" -f ~/.ssh/my_key  # -t 指定密钥类型（推荐ed25519）， -f 指定文件名
    ```
2. 密钥文件默认存放位置为: 私钥：~/.ssh/id_ed25519 公钥：~/.ssh/id_ed25519.pub. 设置密钥文件权限:
    ```bash
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/id_ed25519   # 私钥必须设为 600
    chmod 644 ~/.ssh/id_ed25519.pub
    ```
3. 部署公钥到远程目标机:
    ```bash
    # 以下例子是通过密码去传输密钥文件到远程目标机
    ssh-copy-id -i ~/.ssh/id_ed25519.pub user@remote-host
    ```

## 配置说明

### Inventory

仓库不提交面向当前 roles 的具体 Inventory。用户应在仓库外维护自己的
`inventory.yml`，避免把环境地址和凭据提交到代码仓。原始 CI、长期测试和
`template/` 下的 Inventory 只用于追溯旧部署环境，不能直接用于当前 playbook。

使用前至少检查以下字段：

| 字段 | 说明 |
| --- | --- |
| `ansible_user` | Ansible 登录远端节点的用户。 |
| `ansible_ssh_private_key_file` / `ansible_password` | SSH 凭据，二选一；不要提交真实密码或私钥。 |
| `global_port_base` / `base_api_port` / `proxy_port` | Prefill、Decode 和 Proxy 的端口基准。 |
| `port_offset.P` / `port_offset.D` | 区分 Prefill 与 Decode 的端口范围。 |
| `ansible_host` | 当前 Inventory host 的连接地址。 |
| `host_ip` | 物理机或 Pod 主实例地址；同一多节点 Prefill Pod 使用相同值。 |
| `node_rank` | 节点序号，从 0 开始。 |
| `kv_rank` | Prefill KV rank，从 0 开始。 |
| `node_port` / `api_port` | 当前实例的 master port 和 API Server 端口。 |
| `ascend_rt_visible_devices` | 当前实例使用的 NPU 卡号，例如 `"0,1,2,3"`，不能包含空格或多余逗号。 |

Inventory 必须保留顶层 `P`、`D`、`C` 分组。详细的原始 Inventory 文件用途、
拓扑规则和全部字段见 [Inventory 约定](DEVELOPMENT_GUIDE.md#5-inventory-约定)。

### Playbook

每个模型 playbook 的 `environment` 保存任务执行环境，`vars` 保存 Ansible
直接管理的部署参数：

| 字段 | 位置 | 说明 |
| --- | --- | --- |
| `LOG_PATH` | `environment` | Prefill、Decode 和 Proxy 在远端节点上的日志根目录。 |
| `LOG_PATH_IN_EXECUTOR` | `environment` | 日志拉取到执行机后的根目录。 |
| `CODE_PATH` | `environment` | 执行机和远端宿主机上的代码同步路径。 |
| `SCRIPTS_PATH` | `environment` | 自动生成的服务启动脚本在宿主机及容器中的共享路径。 |
| `model` | `vars` | 当前模型或部署场景的名称。 |
| `model_path` | `vars` | 所有 Prefill、Decode 节点均可访问的模型目录。 |
| `DOCKER_IMAGE_ID` | `environment` | 部署使用的容器镜像。 |
| `DOCKER_NAME_P` / `DOCKER_NAME_D` / `DOCKER_NAME_C` | `environment` | 三类容器的名称前缀，实际名称会追加 Inventory host 名。 |

模型环境变量、准备命令和 CLI 参数分别放在对应 profile 中，不要写回公共 task。
Prefill/Decode 最大模型长度直接配置在各自 profile 的 `args` 中；KV Connector
通过 `run_server_common_profile.kv_connector` 配置。
全部 profile 字段见 [Playbook Profile 参考](DEVELOPMENT_GUIDE.md#7-playbook-profile-参考)。

## 使用方式

在本目录执行：

```bash
cd tools/ansible

PLAYBOOK=playbooks/omni_infer_server_template_panguv2.yml
INVENTORY=/path/to/inventory.yml

# 先验证语法和 tags
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --syntax-check
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --list-tags

# 执行完整部署
ansible-playbook -i "$INVENTORY" "$PLAYBOOK"

# 按阶段执行
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  --tags run_docker,sync_code,pip_install
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  --tags stop_server,run_server,proc_bind,run_proxy,fetch_log
```

Tags 只筛选任务，不改变入口文件定义的执行顺序，也不会自动补齐依赖阶段。
例如单独执行 `run_server` 时，容器、代码和依赖必须已经准备完成。

PanguV2 使用 `elastic_server`，还支持 `add_node`、
`add_node_with_sync_code`、`delete_node` 和 `reload_proxy`；这些生命周期必须通过
对应 tag 显式执行。扩缩容前应先更新 Inventory，并阅读
[Tags 说明](DEVELOPMENT_GUIDE.md#8-tags)。

## Playbook 与 roles 不在同一目录

仓库内 playbook 应从 `tools/ansible` 目录执行，本目录的 `ansible.cfg` 已配置：

```ini
[defaults]
roles_path = roles
```

如果用户自己的 playbook 位于仓库外，可在单次执行时指定本仓库 roles 的绝对路径：

```bash
ANSIBLE_ROLES_PATH=/absolute/path/to/omniinfer/tools/ansible/roles \
  ansible-playbook -i /path/to/inventory.yml /path/to/model.yml
```

长期使用时，建议在外部项目的 `ansible.cfg` 中配置：

```ini
[defaults]
roles_path = /absolute/path/to/omniinfer/tools/ansible/roles
```

如果还需要外部项目自己的 roles，使用冒号连接多个路径。也可以通过
`ANSIBLE_CONFIG=/absolute/path/to/omniinfer/tools/ansible/ansible.cfg` 直接复用
仓库配置。完整说明和检查方式见
[Playbook 不在仓库内时如何导入 Role](DEVELOPMENT_GUIDE.md#32-playbook-不在仓库内时如何导入-role)。
