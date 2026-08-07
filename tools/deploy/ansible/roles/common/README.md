# Common 公共部署阶段

该 role 保存标准部署流程和弹性部署流程共同使用的 task，包括共享的变量约定、
执行顺序、执行条件和公开 tags，同时维护这些公共阶段直接使用的默认值。
弹性生命周期专用的 task 和默认值仍属于 `elastic_server`。Inventory 拓扑计算和
标准部署的 Docker 命令生成由这里处理；弹性扩容保留一份必须同步修改的相同定义。
各 Playbook 通过 profiles 声明模型和源模板差异。

该 role 既通过 `tasks/main.yml` 提供固定的标准部署流程，也提供下列可单独通过
`tasks_from` 导入的公共阶段。`elastic_server` 负责弹性流程，并静态导入这些公共
阶段。Ansible 会在原位置展开静态导入，因此 `--list-tasks` 和 `--list-tags`
仍能看到准确的任务顺序和公开 tags。

可用阶段：

- `run_docker`
- `deploy_code`
- `set_topology`
- `stop_server`
- `manage_mooncake`
- `run_server`
- `bind_cpus`
- `run_proxy`
- `fetch_logs`

`run_docker` 在同一个阶段中保留容器名和 Docker 命令计算，以及原有的
容器清理、启动和日志目录创建顺序。
`deploy_code`、`run_server` 和 `run_proxy` 将相邻任务组织在一起，但不会合并它们的
公开 tags，也不会改变内部执行顺序。标准流程和弹性流程会在各自的 `main.yml`
导入 `stop_server` 时附加公开的 `stop_server` tag；弹性节点激活流程则在自己的
弹性生命周期 tags 下导入同一个阶段。

这些是固定的编排边界，不是扩展钩子。不要向其中加入由 profile 驱动的回调或任意
命令。

变量归属：

- `container_workspace` 是服务容器内 OmniInfer 代码的公共根目录。Inventory、
  group vars 或 extra vars 可以覆盖默认值 `/workspace`，但不会改变宿主机上的
  `CODE_PATH`。
- `docker_exec_cmd` 和默认的 `start_docker_cmd_*` 属于公共 Docker 执行阶段。
- `run_docker_profile` 属于 `run_docker`，用于配置公共 Docker 参数、环境变量和
  Playbook 特有挂载。
- `sync_code_profile` 属于 `sync_code`，用于配置执行机到宿主机的代码同步，以及
  `container_copy` 中各服务的完整命令或 `null`。命令缺失、为空或为 `null` 时
  跳过对应服务；整个 `container_copy` 也可以设置为 `null`。
- `pip_install_profile` 属于 `pip_install`，其中每个服务的值都是完整命令或
  `null`。命令缺失、为空或为 `null` 时跳过对应服务；整个 profile 也可以设置为
  `null`。
- `run_server_prefill_profile` 属于 Prefill 启动流程，包括 `runner`、`workdir`、
  vLLM `docker_envs`、加载 `.bashrc` 后执行的准备命令、多节点后端和有序 CLI
  `args`。公共 task 会固定注入通信网卡、主机地址、端口和多节点坐标等拓扑环境。
- `run_server_decode_profile` 管理对应的 Decode 启动字段。Role 只在 P 主机解析
  Prefill profile，只在 D 主机解析 Decode profile，因此一侧可以安全引用另一侧
  不存在的主机变量。
- `run_server_common_profile` 管理 Prefill 和 Decode 共用的 KV Connector、等待时间
  以及弹性流程中可选的 Proxy 刷新行为。
- `mooncake_profile` 管理 Mooncake 配置生成、服务启动和停止后的等待时间。
  Mooncake 不常用，并且独立于两个服务启动 profile。
- `run_proxy_profile` 属于 `run_proxy`，管理 Omni Proxy 的工作目录、命令和
  有序 CLI 参数。`elastic_server` 的 Proxy reload 复用同一份配置，其中
  `prepare_commands` 同时用于普通启动和 reload。模板负责规范化端点，并生成
  监听端口和 P/D endpoint 参数。
- `proc_bind_profile` 属于 `proc_bind`，用于控制可选的 CPU 绑核。
- `fetch_log_profile` 属于 `fetch_log`，用于控制执行机日志目录创建和日志收集。
- 清理脚本的内容属于固定公共行为，因此不作为 profile 输入。
- `ACTUAL_DOCKER_NAME_*` 是 `run_docker` 的输出，不是默认值。
- `node_port`、`ansible_env`、Inventory groups 和 host variables 都由调用方提供，
  因此不会在 role 中设置默认值。

不要仅仅因为两个 task 看起来相似就将其放入这里。公共阶段必须在所有使用它的流程
中保持相同的变量约定和 tag 行为。

标准 Playbook 导入完整的 common 流程：

```yaml
tasks:
  - ansible.builtin.import_role:
      name: common
```

当前维护的 PanguV2 Playbook 导入固定的 `elastic_server` 入口。入口只负责静态
编排，模型参数仍保留在 Playbook 中。动态 `include_role` 仅用于现有的、
带 `never` tag 的 Proxy 重放流程，因为该流程需要在运行时附加额外 tag。
