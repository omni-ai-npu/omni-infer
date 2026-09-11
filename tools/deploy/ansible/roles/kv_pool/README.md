# KV pool Role

本 role 采用 `feat/ansible-epd-develop` 中 `epd_server` 的组织方式：专用阶段
保存在自身的 `tasks/` 中，并静态导入完整的 `common` 阶段。公共任务不反向
调用 KV pool，也不拆分已有阶段。

## 入口与阶段

独立的 92B BF16 1P1D 场景位于
[`playbooks/kv_pool/`](../../playbooks/kv_pool/)，对应的 P/D/C/M 配置位于
[`inventories/kv_pool/`](../../inventories/kv_pool/)。原有普通 Playbook 不接入本 role。
新 Playbook 选择以下入口：

```yaml
tasks:
  - ansible.builtin.import_role:
      name: kv_pool
```

| 阶段 | 实现 |
| --- | --- |
| `run_docker.yml` | 导入完整 common 容器阶段，再准备独立 M 容器。 |
| common `deploy_code`、`set_topology` | 直接复用代码部署和拓扑计算。 |
| `run_mooncake_master.yml` | 调用外部 master 启动任务，位于原有拓扑计算之后。 |
| common `stop_server` | 直接复用原有停止阶段。 |
| `run_server.yml` | 在 P 侧解析 KV profile、安装模块、生成 runtime，再导入完整 common 启动阶段。 |
| common `bind_cpus`、`run_proxy`、`fetch_logs` | 按 common 的顺序直接复用绑核、Proxy 和日志阶段。 |

`main.yml` 以 common 的标准阶段序列为基准，只增加 KV pool 的专用步骤。
公共阶段内部的执行顺序、Docker 命令生成及 P/D 启动保持原样。

## 配置

`enable_kv_pool` 和 `kv_pool_repo_root` 保持为顶层配置。
角色默认关闭 KV pool，独立 KV pool Playbook 显式设置 `enable_kv_pool: true`：

```yaml
enable_kv_pool: false
kv_pool_repo_root: "{{ (role_path ~ '/../../../../../../Pangu-FusionComm') | realpath }}"
```

只覆盖 `enable_kv_pool` 时仍使用默认的外部仓库路径。
关闭时不加载外部模块。M 实际容器名只在启用
且处理 M 主机时由 `ansible_env.DOCKER_NAME_M` 生成。M 容器仅在不存在时创建，
不会随 P/D/C 重部署删除。

`kv_pool_repo_root` 指向执行机上的 Pangu-FusionComm checkout，默认定位到
OmniInfer 仓库旁的 `Pangu-FusionComm` 目录。路径由 role 所在位置解析，不随
Playbook 移入子目录而改变；部署时可覆盖为实际绝对路径。启用前须准备兼容的：

- `kv_pool/deploy/install_kv_pool.yml`
- `kv_pool/deploy/run_mooncake_master.yml`
- `kv_pool/deploy/_kv_pool_runtime.sh.j2`

本 role 不获取外部仓库；外部模块继续维护安装、master 启动和 runtime 的实现。
`run_server_prefill_profile.kv_pool` 与以下专用默认值递归合并：

```yaml
global_segment_size_gb: 32
local_buffer_size_gb: 8
hugepage_enabled: true
```

独立 Playbook 在 Prefill profile 的 `prepare_commands` 中加载生成的 runtime，
在 `args` 中传入 `--kv-transfer-config`。`pd_run.sh` 的已有 override 逻辑
优先使用这个配置；公共模板无需识别 KV pool 开关。

Tag 名称保持原样：`run_server` 准备并启动 P/D；`run_mooncake_master` 调用
master 启动任务。单独选择 `run_server` 不自动追加 master 启动，前置容器、
源码和 master 就绪状态仍由调用方保证。

`clean_up` 复用 common 的 P/D/C 容器清理，并在 KV pool 启用时停止、删除
已有 M 容器，不重新创建容器。M 清理任务仅在显式选择 `clean_up` 时执行，
普通部署及 `run_docker` 保留已有 M 容器。

本 role 不引入弹性扩缩容、Proxy reload 或 `run_server` 自动刷新 Proxy。
绑核、Proxy 启动和日志收集分别使用 common 的 `proc_bind`、`run_proxy`、
`fetch_log` tags。
本地语法和编排检查不能替代兼容外部模块与真实服务环境的部署验收。

## 独立配置的使用

从 `tools/deploy/ansible` 执行。Inventory 中的 loopback 地址是占位配置，实际
部署前复制到自己的配置目录并填写节点地址、设备及连接信息：

```bash
PLAYBOOK=playbooks/kv_pool/omni_infer_server_template_performance1P1D_92B_bf16_kv_pool.yml
INVENTORY=inventories/kv_pool/omni_infer_inventory_used_for_1P8_1D8_for_kv_pool.yml

# 以下检查不执行部署。
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --syntax-check
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --list-hosts
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --list-tags

# 实际部署使用已填写的 Inventory 与兼容的外部模块。
ansible-playbook -i /path/to/kv_pool_inventory.yml "$PLAYBOOK" \
  -e kv_pool_repo_root=/path/to/Pangu-FusionComm
```

仓库的 `ansible.cfg` 仍使用 `roles_path = roles`，嵌套 Playbook 无需更改 role
搜索配置。从其他目录运行时，应显式指定 `ANSIBLE_CONFIG` 或
`ANSIBLE_ROLES_PATH`，与普通 Playbook 的要求相同。
