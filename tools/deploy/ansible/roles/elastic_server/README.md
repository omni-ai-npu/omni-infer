# Elastic Server Role

`elastic_server` 是在 `common` 公共部署阶段之上增加弹性生命周期的专用 role。
它不保存 PanguV2 模型参数；当前由 PanguV2 playbook 使用，但能力本身只负责
增删节点、增量代码同步和 Proxy 刷新。

模型路径、容器镜像、环境变量、启动 runner 和 CLI 参数仍由 playbook 的 profiles
声明。容器、拓扑、服务、Proxy 和日志等同构任务继续从 `common` 导入。

## 使用入口

Playbook 通过静态导入完整流程：

```yaml
tasks:
  - ansible.builtin.import_role:
      name: elastic_server
```

静态导入会按 `tasks/main.yml` 中的声明顺序展开任务，因此原有 tags 仍可通过
`--list-tasks` 和 `--list-tags` 查看。不要在 playbook 中同时导入完整的 `common`
和 `elastic_server`，否则公共部署阶段会重复执行。

## 任务文件

| 文件 | 作用 |
| --- | --- |
| `tasks/main.yml` | 定义公共部署阶段与弹性生命周期的完整顺序。 |
| `tasks/prepare_nodes.yml` | 检查容器，仅为尚未部署的 P/D 节点创建容器和目录。 |
| `tasks/sync_node_code.yml` | 为新节点同步代码、复制代码到容器，并按需执行安装命令。 |
| `tasks/activate_nodes.yml` | 为新节点生成配置及启动脚本，启动 vLLM，并按需执行 CPU 绑核。 |
| `tasks/delete_nodes.yml` | 校验待删除 IP、刷新 Proxy，并停止和删除目标节点容器。 |
| `tasks/reload_proxy.yml` | 重新生成 Omni Proxy 脚本并执行 reload。 |
| `defaults/main.yml` | 保存 `delete_node_profile` 默认值。 |

`prepare_nodes.yml` 会将容器检查结果保存到 `existing_containers`。后续同步和激活
任务只处理容器不存在的节点，已有节点不会被当作新增节点重复部署。

## Tags

除 `common` 提供的标准 tags 外，本 role 增加以下生命周期：

| Tag | 执行内容 | 使用前提 |
| --- | --- | --- |
| `add_node` | 创建缺失容器并直接激活新节点。 | 新节点所需代码已经位于宿主机和容器可用位置。 |
| `add_node_with_sync_code` | 创建缺失容器，完成代码同步、容器复制和安装后再激活。 | `sync_code_profile` 已配置正确；安装命令可按需配置。 |
| `delete_node` | 校验 IP、按删除后的拓扑刷新 Proxy，并停止和删除目标节点上的 P/D 容器。 | 先从 P/D Inventory 分组移除目标节点，再显式传入 `delete_node_profile.ips`。 |
| `reload_proxy` | 重新生成脚本并 reload Omni Proxy。 | Proxy 容器和启动配置已经存在。 |

以上 tags 均为显式生命周期，不会在普通完整部署中额外触发。
`add_node` 和 `add_node_with_sync_code` 不会自动选择 `reload_proxy`；需要刷新静态
Proxy 端点时，应在节点激活成功后单独执行 `reload_proxy`。

`run_server_common_profile.restart_proxy=true` 只控制普通 `run_server` 生命周期是否
联动刷新 Proxy，不改变两个新增节点 tags 的行为。

## 配置

### 继承的公共 Profiles

弹性流程复用以下 `common` 配置：

- `run_docker_profile`
- `sync_code_profile`
- `pip_install_profile`
- `run_server_prefill_profile`
- `run_server_decode_profile`
- `run_server_common_profile`
- `run_proxy_profile`
- `proc_bind_profile`
- `fetch_log_profile`

字段含义见 [OmniInfer Ansible 开发与部署指南](../../DEVELOPMENT_GUIDE.md) 和
[Common role README](../common/README.md)。模型差异应保留在 playbook 中，不要
为某个模型扩展 `elastic_server/defaults/main.yml`。

### `delete_node_profile`

`delete_node_profile` 仅属于本 role：

```yaml
delete_node_profile:
  ips:
    - 192.0.2.20
```

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `ips` | `[]` | 待删除节点的 IPv4 地址列表；执行 `delete_node` 时不能为空。 |

建议通过 extra vars 临时传入生产 IP，不要将真实地址固化在 playbook：

```bash
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  --tags delete_node \
  -e '{"delete_node_profile":{"ips":["192.0.2.20"]}}'
```

执行 `delete_node` 前，应先从 Inventory 的 P/D 分组中移除目标节点，但保留 C
分组及其他活动节点。Role 会先用更新后的 Inventory 刷新 Proxy，再把
`delete_node_profile.ips` 注册为临时委托目标，并通过 `all.vars` 中的公共 SSH
配置连接目标节点、停止服务和删除 P/D 容器。若 SSH 用户或凭据只配置在已删除的
host 条目中，应先将这些连接配置移到 `all.vars`。

### 环境与 Inventory

Role 需要 playbook 或 Inventory 提供：

- `LOG_PATH`、`LOG_PATH_IN_EXECUTOR`、`CODE_PATH` 和 `SCRIPTS_PATH`。
- `DOCKER_IMAGE_ID`、`DOCKER_NAME_P`、`DOCKER_NAME_D` 和 `DOCKER_NAME_C`。
- `run_proxy_profile` 中的 Omni Proxy 配置。
- 顶层 `P`、`D`、`C` 分组及公共拓扑要求的 host 变量。

上述环境变量和配置必须由 playbook 或 Inventory 明确设置，`elastic_server`
不提供环境兜底值。

## 操作示例

从 `tools/deploy/ansible` 目录执行：

```bash
INVENTORY=/path/to/inventory.yml
PLAYBOOK=playbooks/omni_infer_server_template_panguv2.yml

# 先检查解析结果。
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --syntax-check
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --list-tasks --tags add_node

# 代码已经准备好时新增节点。
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --tags add_node

# 同步和安装代码后新增节点。
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  --tags add_node_with_sync_code

# 节点拓扑变化后按需刷新 Proxy。
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --tags reload_proxy
```

新增节点前必须先把节点加入 Inventory，并确认 SSH、Docker、模型路径、挂载目录和
端口可用。执行 `delete_node` 前应确认 IP 列表和 Proxy 拓扑；这些操作会直接启动
或删除远端容器，不能只依赖 `--syntax-check` 判断运行结果。

## 扩展边界

- 可复用阶段继续从 `common` 导入，不复制公共 task。
- 只有弹性生命周期独有的 task 才放入本 role。
- 模型环境变量、准备命令和 CLI 参数留在 playbook profiles 中。
- 新增生命周期应使用语义明确的 task 文件和 tag，不向 `main.yml` 增加模型布尔开关。
- 调整公共阶段顺序或公共 profile 契约前，需要评估 `common` 标准流程和所有维护中
  playbook 的兼容性。
