# OmniInfer Ansible 开发与部署指南

本文面向需要编写模型 Playbook 或新增专用 Role 的开发者，说明当前目录结构、
Playbook 写法、Inventory 约定、Profile、Tags 和 Role 扩展方式。

当前维护的生产入口只有
[`playbooks/omni_infer_server_template_panguv2.yml`](playbooks/omni_infer_server_template_panguv2.yml) 和
[`playbooks/omni_infer_server_template_dsv32.yml`](playbooks/omni_infer_server_template_dsv32.yml)。
新增场景可以从受维护的
[`examples/omni_infer_server_template_example.yml`](examples/omni_infer_server_template_example.yml)
开始。
本文中的配置片段用于说明通用写法，不绑定某一个具体模型。

## 1. 通用设计

当前部署框架由三层组成：

1. **Playbook 配置层**：只声明 Inventory 对应的环境、模型 profile 和编排入口。
2. **Common 任务层**：按 tags 提供容器、代码同步、服务、Proxy 和日志等同构阶段。
3. **专用 Role 编排层**：当某类部署需要新增任务或改变阶段顺序时，创建独立 role，
   显式复用 common task，并在准确位置插入专用 task。

新增模型时应优先只增加 playbook 配置。默认任务序列无法表达需求时，可以像
`elastic_server` 一样新增对应 role，并按需要复用已有阶段。

旧的单文件 Playbook 已移除。新模型和部署流程统一使用 `playbooks/`、
`roles/`；1P1D、2P1D 和 4P1D 的拓扑模板保存在 `inventories/`。

## 2. 目录结构

```text
tools/deploy/ansible/
├── ansible.cfg
├── DEVELOPMENT_GUIDE.md            # Ansible 开发与部署指南
├── examples/
│   ├── inventory_1p1d.yml             # 仅供本地解析/check 的安全拓扑 fixture
│   └── omni_infer_server_template_example.yml  # 可复制的通用 Playbook 示例
├── inventories/
│   ├── omni_infer_inventory_used_for_1P1D.yml
│   ├── omni_infer_inventory_used_for_2P1D.yml
│   └── omni_infer_inventory_used_for_4P1D.yml
├── playbooks/
│   ├── omni_infer_server_template_panguv2.yml
│   └── omni_infer_server_template_dsv32.yml
├── roles/
│   ├── common/
│   │   ├── defaults/main.yml       # 公共默认值
│   │   ├── tasks/                  # 按 tags 划分的公共阶段
│   │   └── templates/              # Prefill/Decode/Proxy 启动模板
│   └── elastic_server/
│       ├── README.md                # 弹性生命周期说明
│       ├── defaults/main.yml
│       └── tasks/                  # 扩缩容生命周期
```

从 `tools/deploy/ansible` 目录执行命令。`ansible.cfg` 已将 `roles_path` 设置为
`roles`，从其他目录执行时需要自行保证角色搜索路径正确。

### 2.1 框架运行链路

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
Role 提供默认值并决定任务顺序。Playbook 编写者只需要使用公开 Profile；
仓库中的所有受维护部署入口都使用这条 role-based 运行链路。

## 3. Role 导入

### 3.1 使用仓库内 Playbook

从 `tools/deploy/ansible` 目录执行时，Ansible 会自动读取本目录的
`ansible.cfg`：

```ini
[defaults]
roles_path = roles
```

因此 `playbooks/*.yml` 可以直接使用：

```yaml
- ansible.builtin.import_role:
    name: common
```

可通过下面的命令确认实际读取的配置和 role 搜索路径：

```bash
ansible --version
ansible-config dump --only-changed
```

输出中的 `config file` 应指向 `tools/deploy/ansible/ansible.cfg`，
`DEFAULT_ROLES_PATH` 应包含 `tools/deploy/ansible/roles`。

### 3.2 Playbook 不在仓库内时如何导入 Role

Playbook 可以存放在任意目录，但 Ansible 必须能找到本仓库的 `roles/`。以下三种
方式任选一种；`import_role` 内容无需改变。

**方式一：通过环境变量指定 role 路径，最适合单次执行。**

```bash
ANSIBLE_ROLES_PATH=/absolute/path/to/omniinfer/tools/deploy/ansible/roles \
  ansible-playbook -i /path/to/inventory.yml /path/to/model.yml
```

多个 role 根目录使用冒号分隔：

```bash
ANSIBLE_ROLES_PATH=/path/to/project/roles:/absolute/path/to/omniinfer/tools/deploy/ansible/roles \
  ansible-playbook -i /path/to/inventory.yml /path/to/model.yml
```

**方式二：显式复用仓库的 `ansible.cfg`，适合脚本或 CI。**

```bash
ANSIBLE_CONFIG=/absolute/path/to/omniinfer/tools/deploy/ansible/ansible.cfg \
  ansible-playbook -i /path/to/inventory.yml /path/to/model.yml
```

当前配置中的相对 `roles_path = roles` 会定位到该配置文件同级的 `roles/`。

**方式三：在外部项目维护自己的 `ansible.cfg`，适合长期使用。**

```ini
[defaults]
roles_path = /absolute/path/to/omniinfer/tools/deploy/ansible/roles
```

在该配置生效的目录执行，或通过 `ANSIBLE_CONFIG=/path/to/ansible.cfg` 显式选择。
不要只把仓库的 playbook 复制到外部目录然后依赖默认搜索规则；默认情况下，
Ansible 不会自动搜索 OmniInfer 仓库的 `roles/`。

配置完成后先检查：

```bash
ansible-config dump --only-changed
ansible-playbook -i /path/to/inventory.yml /path/to/model.yml --syntax-check
ansible-playbook -i /path/to/inventory.yml /path/to/model.yml --list-tasks
```

## 4. 编写 Playbook

模型、镜像、环境变量、准备命令或 CLI 参数不同时，只新增 Playbook；需要扩缩容时
选择 `elastic_server`；只有任务顺序或生命周期无法复用时才新增 Role。

```text
任务顺序与 common 是否一致？
├── 是
│   ├── 普通部署：Playbook 导入 common
│   └── 需要扩缩容：Playbook 导入 elastic_server
└── 否
    └── 按第 10 节新增专用 Role
```

复制现有文件作为开发起点是受支持的，不要求开发者从空白 Playbook 开始。可以优先
复制受维护的示例；已有生产 Playbook 与目标模型接近时，也可以直接复制该
Playbook。这里的约束针对最终提交结果，而不是开发起步方式：提交前应删除无关的
源模型环境变量、准备命令、CLI 参数和挂载，只保留目标场景需要的差异。

### 4.1 选择开发起点

新场景优先复制通用示例：

```bash
cp examples/omni_infer_server_template_example.yml \
  playbooks/omni_infer_server_template_<model>.yml
```

也可以复制最接近的维护中 Playbook。无论从哪个文件开始，都需要重新确认：

- `environment` 中的路径、镜像和容器名称属于目标环境。
- Prefill、Decode 和 Proxy Profile 中没有遗留源模型参数。
- Inventory 分组和 Profile 中引用的 host 变量在目标拓扑中存在。
- 任务序列与 common 一致时导入 `common`；需要既有扩缩容能力时导入
  `elastic_server`；存在新任务或新顺序时新增专用 Role。

完整字段见第 7 节。

### 4.2 配置放置位置

同一个配置只保留一个来源：

| 位置 | 应放内容 | 不应放内容 |
| --- | --- | --- |
| Inventory | SSH 连接变量、P/D/C 分组、IP、端口、rank 和卡号等环境拓扑 | 模型 CLI、容器内准备命令、提交到仓库的真实密码或私钥内容 |
| Playbook `environment` | 日志、代码、脚本路径、镜像和容器基础名称 | 最大模型长度、并行度和 KV Connector 等服务参数 |
| Playbook `vars` | `model`、`model_path` 和各阶段 Profile | Role 内部计算变量 |
| `-e` extra vars | 删除节点 IP、临时开关等单次操作参数 | 应长期版本管理的模型配置 |

- 直接使用第 7 节公开的 Profile 名称和字段，不为同一个含义增加别名。
- Ansible 普通变量使用小写 snake_case；传给容器或进程的环境变量保持大写。
- 模型特有的环境变量、准备命令和 CLI 参数留在对应 Playbook。

### 4.3 Profile 写法

Role 会自动补充公共默认值，Playbook 只覆盖当前模型不同的字段。字典递归合并，
列表和标量整体替换，不会自动追加。

```text
docker_envs
└── docker exec 创建进程时就必须存在的环境变量

prepare_commands
└── 加载容器 .bashrc 后执行的 export 和准备命令

args
└── 按顺序传给 runner 的完整 CLI 片段
```

- 仅作为 CLI 参数使用的值直接放入 `args`，不要先定义环境变量再层层传递。
- `sync_code_profile.container_copy` 和 `pip_install_profile` 的服务值是完整 Bash
  命令或 `null`；`null`、空字符串或缺失时跳过。
- Prefill 和 Decode Profile 分别引用各自主机上的 Inventory 变量，不要跨分组依赖
  不存在的变量。
- Profile 只描述配置值，不用于插入 task 或改变任务顺序。

### 4.4 变量引用

Play 级 `environment` 通过 `ansible_env.<NAME>` 读取：

```yaml
path: "{{ ansible_env.CODE_PATH }}"
```

`environment` 中的名称不会自动成为普通 Ansible 变量，因此不要直接写
`{{ CODE_PATH }}`。`vars` 中的 `model_path` 则直接使用：

```jinja2
--model-path {{ model_path | quote }}
```

`{{ value }}` 在 Ansible 渲染时展开，`${VALUE}` 或 `$VALUE` 在生成的 Shell
执行时展开。允许任意字符串的单个 Shell 参数应使用 `quote`；`args` 中每一项是
完整且有序的 CLI 片段，调用方负责片段内部引号。

为保持现有部署脚本契约，生成脚本和配置文件的 task 仍使用
`dest: "$SCRIPTS_PATH/<文件名>"`，弹性节点清理也仍由 Shell 展开
`${SCRIPTS_PATH}`。修改这些字面量时需要同时验证宿主机模块执行和容器共享挂载。

### 4.5 按阶段可引用的派生变量

`common` 的 `always` tasks 共生成 20 个变量，其中 18 个可以作为后续
Playbook、Profile 或专用 Role 的输入，另外 2 个仅用于内部拓扑计算。18 个公开
名称由 3 个容器名和 15 个拓扑变量组成；其中 2 个是兼容别名，因此实际对应
16 份不同的值。

这些变量是当前 `ansible-playbook` 调用中、当前 Inventory host 上的运行时
facts，不会跨两次命令持久化。`set_topology` 虽然使用
`delegate_to: localhost`，结果仍属于当前 Inventory host。

公共任务的求值顺序为：

```text
run_docker（容器名初始化）
→ run_docker / clean_up
→ sync_code / pip_install
→ set_topology
→ stop_server
→ manage_mooncake
→ run_server
→ proc_bind
→ run_proxy
→ fetch_log
```

`always` 只表示选择其他 Tags 时仍会执行，不会改变上述顺序：

| 求值点 | 新增的公开变量 | 后续可用位置 |
| --- | ---: | --- |
| Play 开始、Gather Facts 完成 | 不计入上述 18 个 | Inventory 变量、Play `vars`、Role defaults 和 `ansible_env.*` 可供全部阶段使用。 |
| `run_docker` 的 `always` task | 3 | `run_docker` 及之后的全部阶段。 |
| `run_docker` / `clean_up` | 0 | 只生成容器检查结果和内部 Docker 命令。 |
| `sync_code` / `pip_install` | 0 | 只生成 `resolved_*` 内部变量。 |
| `set_topology` | 15 | `stop_server`、Mooncake、`run_server`、`proc_bind`、`run_proxy`、`fetch_log` 以及排在其后的专用任务。 |
| 后续公共阶段和 Elastic 生命周期 | 0 | 只生成阶段内部变量或重新计算现有拓扑。 |

因此，拓扑变量不能用于 `run_docker_profile`、`sync_code_profile` 或
`pip_install_profile`，因为这些 Profile 在 `set_topology` 之前已经被消费。
不要把派生拓扑变量写入 Play 级 `environment`。

#### 4.5.1 可直接引用的 18 个变量

| 变量 | 作用域 | 含义 |
| --- | --- | --- |
| `ACTUAL_DOCKER_NAME_P` | P/D/C | 当前 host 对应的 Prefill 实际容器名。 |
| `ACTUAL_DOCKER_NAME_D` | P/D/C | 当前 host 对应的 Decode 实际容器名。 |
| `ACTUAL_DOCKER_NAME_C` | P/D/C | 当前 host 对应的 Proxy 实际容器名。 |
| `PREFILL_API_SERVER_LIST` | P/D/C | 每个 Prefill Pod 主节点的 `ip:api_port`，以逗号分隔。 |
| `DECODE_API_SERVER_LIST` | P/D/C | `ip:api_port@设备数减一` 列表，供 Proxy 展开 Decode API 端口。 |
| `DECODE_API_SERVER_LIST_ALL` | P/D/C | 当前与 `DECODE_API_SERVER_LIST` 相同的别名。 |
| `OMNI_PD_PREFILL_POD_NUM` | P/D/C | 按 P 组唯一 `host_ip` 计算的 Prefill Pod 数。 |
| `OMNI_PD_DECODE_POD_NUM` | P/D/C | 按 D 组唯一 `host_ip` 计算的 Decode Pod 数。 |
| `DECODE_SERVER_IP_LIST` | P/D/C | 顶层 D 组全部 `ansible_host`，当前是 `_ALL` 的别名。 |
| `DECODE_SERVER_IP_LIST_ALL` | P/D/C | 顶层 D 组全部 `ansible_host`。 |
| `DECODE_SERVER_IP_LIST_BY_GROUP` | D | 当前 D host 所属 `D<n>` 组的 `ansible_host`；未使用数字子组时回退顶层 D。 |
| `DECODE_SERVER_ALL` | P/D/C | 顶层 D 组全部 `ascend_rt_visible_devices` 的拼接结果。 |
| `DECODE_SERVER_BY_GROUP` | D | 当前 D host 所属 `D<n>` 组的卡号拼接结果。 |
| `DECODE_SERVER_OFFSET` | P/D/C | 按顶层 D 组顺序计算的全局设备偏移字典。 |
| `DECODE_SERVER_OFFSET_BY_GROUP` | D | 当前 `D<n>` 组内从零计算的设备偏移字典。 |
| `DECODE_SERVER_OFFSET_ALL` | P/D/C | 所有 `D<n>` host 的组内设备偏移字典，每个组从零开始。 |
| `NODE_IP_LIST` | P | 与当前 P host 使用相同 `host_ip` 的 Prefill 节点列表。 |
| `NNODES` | P | `NODE_IP_LIST` 中的 Prefill 节点数量。 |

Profile 中推荐直接引用这些 facts，不要在 `vars` 中重新计算。例如：

```yaml
run_server_prefill_profile:
  prepare_commands: |-
    export SERVER_IP_LIST={{
      DECODE_SERVER_IP_LIST_ALL
      | replace(' ', '')
      | trim
      | quote
    }}

run_server_decode_profile:
  prepare_commands: |-
    export DECODE_DATA_PARALLEL_SIZE={{ DECODE_SERVER_BY_GROUP | quote }}
    export SERVER_IP_LIST={{
      DECODE_SERVER_IP_LIST_BY_GROUP
      | replace(' ', '')
      | trim
      | quote
    }}
```

在自定义 task 中使用实际容器名时，可以写：

```yaml
- name: Run a model-specific command in the Decode container.
  ansible.builtin.command:
    argv:
      - docker
      - exec
      - "{{ ACTUAL_DOCKER_NAME_D }}"
      - /path/to/command
  when: "'D' in group_names"
```

`decode_inventory_groups`、`decode_inventory_scope_groups`、`default_interface`、
Mooncake 临时 facts、所有 `resolved_*`、`*_cmd*` 和 `register` 结果属于 Role
内部实现，不作为用户接口。需要公共拓扑的专用 Role 应在自己的任务序列中先导入
`common: set_topology`，或放在完整 `common` Role 之后。

### 4.6 从历史模板迁移到当前 Profile

旧模板把路径、模型参数、容器参数和运行时参数集中放在 Play
`environment`，并在 `vars` 中保存整段 Shell 命令。迁移时应按参数的实际生效
阶段拆分，而不是保留旧变量名再逐层传递。

#### 4.6.1 旧顶层变量

| 旧变量 | 当前设置位置 |
| --- | --- |
| `LOG_PATH` | 保留在 Play `environment.LOG_PATH`。 |
| `LOG_PATH_IN_EXECUTOR` | 保留在 Play `environment.LOG_PATH_IN_EXECUTOR`；默认日志拉取会使用。 |
| `CODE_PATH` | 保留在 Play `environment.CODE_PATH`；默认代码同步会使用。 |
| `SCRIPTS_PATH` | 保留在 Play `environment.SCRIPTS_PATH`。 |
| `DOCKER_IMAGE_ID` | 保留在 Play `environment.DOCKER_IMAGE_ID`。 |
| `DOCKER_NAME_P/D/C` | 保留在 Play `environment` 中对应名称。 |
| `MODEL_PATH` | 改为 `vars.model_path`；同时新增 `vars.model` 标识场景。 |
| `MODEL_LEN_MAX_PREFILL` | 写成 `run_server_prefill_profile.args` 中的 `--max-model-len <值>`。 |
| `MODEL_LEN_MAX_DECODE` | 写成 `run_server_decode_profile.args` 中的 `--max-model-len <值>`。 |
| `DECODE_TENSOR_PARALLEL_SIZE` | 写成 Decode `args` 中的 `--tp <值>`。 |
| `KV_CONNECTOR` | 改为 `run_server_common_profile.kv_connector`；使用默认 `LLMDataDistConnector` 时可省略。 |
| `USE_OMNI_PROXY` | `1` 改为 `run_proxy_profile.type: omni-proxy`，`0` 改为 `global-proxy`。 |
| `PREFILL_LB_SDK` | 仅 Global Proxy 迁移为 `global-proxy.args` 中的 `--prefill-lb-sdk <值>`。 |
| `DECODE_LB_SDK` | 仅 Global Proxy 迁移为 `global-proxy.args` 中的 `--decode-lb-sdk <值>`。 |
| `OMNI_INFER_SCRIPTS` | 通常由 `container_workspace` 和默认 `workdir` 替代；runner 目录特殊时覆盖对应 `workdir`。 |

从历史部署配置迁移且缺少 `CODE_PATH`、`LOG_PATH_IN_EXECUTOR` 时，使用默认
流程必须补齐。自定义流程只有在相应阶段及其默认值都不再引用这些路径时才能省略。

### 4.7 迁移后的 Playbook 示例

一份典型的核心迁移结果如下，完整结构见
[`examples/omni_infer_server_template_example.yml`](examples/omni_infer_server_template_example.yml)：

```yaml
vars:
  model: model-name
  model_path: /data/models/model

  run_server_prefill_profile:
    runner: pd_run.sh
    prepare_commands: |-
      export SERVER_IP_LIST={{
        DECODE_SERVER_IP_LIST_ALL
        | replace(' ', '')
        | trim
        | quote
      }}
      export HCCL_CONNECT_TIMEOUT=1800
    args:
      - --max-model-len 32000
      - --gpu-util 0.92
      - >-
        --extra-args '--max-num-seqs 16 --enable-expert-parallel'

  run_server_decode_profile:
    runner: pd_run.sh
    prepare_commands: |-
      export DECODE_DATA_PARALLEL_SIZE={{ DECODE_SERVER_BY_GROUP | quote }}
      export SERVER_IP_LIST={{
        DECODE_SERVER_IP_LIST_BY_GROUP
        | replace(' ', '')
        | trim
        | quote
      }}
    args:
      - --max-model-len 16384
      - --tp 1
      - --gpu-util 0.9

  run_server_common_profile:
    kv_connector: LLMDataDistConnector
```

## 5. Inventory 约定

### 5.1 当前 Inventory 文件

仓库不提交包含真实环境信息的生产 Inventory。用户应参考本节字段结构，从
`inventories/` 复制对应拓扑到仓库外，再填写实际地址和凭据。仓库提供的
`examples/inventory_1p1d.yml` 只包含 loopback 地址和本地连接，用于
`--syntax-check`、任务列表及受控的 check-mode 验证。

| 文件 | 用途和兼容性 |
| --- | --- |
| `examples/inventory_1p1d.yml` | 当前 roles 可解析的安全 1P1D+C fixture；仅用于语法、列表和 check-mode 验证，不能完整部署。 |
| `inventories/omni_infer_inventory_used_for_1P1D.yml` | 1P1D 拓扑模板。 |
| `inventories/omni_infer_inventory_used_for_2P1D.yml` | 2P1D 拓扑模板，也可用于验证新增 Prefill 节点后的拓扑。 |
| `inventories/omni_infer_inventory_used_for_4P1D.yml` | 4P1D 拓扑模板。 |

Inventory 中的 IP、密码和私钥路径都属于环境数据。实际部署 Inventory 应保存在
仓库外；不要把真实环境地址或凭据提交到仓库。

相关环境配置分布如下：

- Playbook 与本仓库 roles 不在同一目录时的 `ANSIBLE_ROLES_PATH` 或
  `ANSIBLE_CONFIG` 配置见第 3.2 节。
- `LOG_PATH`、`CODE_PATH`、容器名等部署环境变量见第 6 节。

一份最小的当前结构如下：

```yaml
all:
  vars:
    ansible_user: root
    ansible_ssh_private_key_file: /absolute/path/to/deploy-key
    ansible_ssh_common_args: >-
      -o StrictHostKeyChecking=no -o IdentitiesOnly=yes

    global_port_base: 8000
    base_api_port: 9000
    proxy_port: 7000
    port_offset:
      P: 0
      D: 100

  children:
    P:
      hosts:
        p0:
          ansible_host: 10.0.0.11
          host_ip: 10.0.0.11
          node_rank: 0
          kv_rank: 0
          node_port: "{{ global_port_base + port_offset.P + kv_rank }}"
          api_port: "{{ base_api_port + port_offset.P + kv_rank }}"
          ascend_rt_visible_devices: "0,1,2,3,4,5,6,7"

    D:
      hosts:
        d0:
          ansible_host: 10.0.0.21
          host_ip: 10.0.0.21
          node_rank: 0
          node_port: "{{ global_port_base + port_offset.D }}"
          api_port: "{{ base_api_port + port_offset.D + node_rank }}"
          ascend_rt_visible_devices: "0,1,2,3,4,5,6,7"

    C:
      hosts:
        c0:
          ansible_host: 10.0.0.11
          host_ip: 10.0.0.11
          node_rank: 0
          node_port: "{{ proxy_port + node_rank }}"
```

同一物理机可以同时承载 P 和 C，方法是让两个 Inventory host 使用相同的
`ansible_host`，但 host 名称仍必须唯一。不要把 P、D、C 合并为同一个 Inventory
host，因为公共 tasks 通过组成员身份决定容器和服务类型。

### 5.2 固定分组

Inventory 必须使用以下顶层组名：

- `P`：Prefill 实例。
- `D`：Decode 实例。
- `C`：Proxy 实例。

Decode 可以继续划分为 `D0`、`D1` 等数字后缀子组。当前 Decode 启动脚本会根据
当前主机所属的 `D<n>` 子组计算：

- `DECODE_SERVER_IP_LIST_BY_GROUP`
- `DECODE_SERVER_BY_GROUP`
- `DECODE_SERVER_OFFSET_BY_GROUP`

没有 `D<n>` 子组时统一使用顶层 `D`。Prefill 也可以使用 `P0`、`P1` 子组组织
Inventory。公共拓扑分别在顶层 `P`、`D` 中按唯一 `host_ip` 计算 Prefill 和
Decode Pod 数量；同一多节点 Pod 的所有 Inventory host 必须配置相同的
`host_ip`。

### 5.3 `all.vars`

| 变量 | 必需 | 说明 |
| --- | --- | --- |
| `ansible_user` | 是 | SSH 用户。 |
| `ansible_password` / `ansible_ssh_private_key_file` | 二选一 | SSH 凭据。不要把真实密码提交到仓库。 |
| `ansible_ssh_common_args` | 否 | SSH 附加参数。 |
| `global_port_base` | 是 | Prefill/Decode master port 基准。 |
| `base_api_port` | 是 | API Server 端口基准。 |
| `proxy_port` | 是 | Proxy 监听端口基准。 |
| `port_offset.P` | 是 | Prefill 端口偏移。 |
| `port_offset.D` | 是 | Decode 端口偏移。 |
| `etcd_port` | 否 | Mooncake/LMCache 使用；普通部署不用关注。 |
| `mooncake_master_port` | 否 | Mooncake/LMCache 使用；普通部署不用关注。 |
| `mooncake_metrics_port` | 否 | Mooncake/LMCache 使用；普通部署不用关注。 |

### 5.4 Host 变量

| 变量 | P | D | C | 说明 |
| --- | --- | --- | --- | --- |
| `ansible_host` | 是 | 是 | 是 | Ansible 连接地址，也是当前实例地址。 |
| `host_ip` | 是 | 是 | 建议 | 物理机或 Pod 主实例地址。同一多节点 Prefill Pod 使用相同值；Decode 中 `ansible_host == host_ip` 的实例优先排列。 |
| `node_rank` | 是 | 是 | 是 | 实例序号，用于端口和容器名计算。 |
| `kv_rank` | 是 | 否 | 否 | Prefill KV rank，从 0 开始。 |
| `node_port` | 是 | 是 | 是 | Prefill/Decode master port 或 Proxy 监听端口。 |
| `api_port` | 是 | 是 | 否 | Prefill/Decode API Server 起始端口。 |
| `ascend_rt_visible_devices` | 是 | 是 | 否 | 当前实例使用的 NPU，例如 `"0,1,2,3"`；不要包含空格或多余逗号。 |

容器实际名称由 playbook 环境中的基础名称加 Inventory host 名组成：

```text
ACTUAL_DOCKER_NAME_P = DOCKER_NAME_P + "_" + inventory_hostname
ACTUAL_DOCKER_NAME_D = DOCKER_NAME_D + "_" + inventory_hostname
ACTUAL_DOCKER_NAME_C = DOCKER_NAME_C + "_" + inventory_hostname
```

## 6. Play 级环境变量与变量

除 `model` 和 `model_path` 外，这些变量放在 play 的 `environment` 下，当前
任务通过 `ansible_env.<NAME>` 使用。`model` 和 `model_path` 放在 `vars` 下，
由 Profile、模板和公共 Docker 挂载直接引用。

| 变量 | 位置 | 说明 |
| --- | --- | --- |
| `LOG_PATH` | `environment` | 远端宿主机和容器共享的日志根目录。Role 会创建 `<LOG_PATH>/<inventory_hostname>`。 |
| `LOG_PATH_IN_EXECUTOR` | `environment` | 执行机上的日志汇总目录，`fetch_log` 会把远端日志拉到这里。 |
| `model` | `vars` | 模型或部署场景名称，用于标识当前启动配置和错误信息。 |
| `model_path` | `vars` | 所有 P/D 节点可见的模型目录，公共 Docker 阶段会原路径挂载。 |
| `CODE_PATH` | `environment` | 远端宿主机代码目录，不是容器路径。 |
| `DOCKER_IMAGE_ID` | `environment` | 部署镜像。 |
| `DOCKER_NAME_P` | `environment` | Prefill 容器基础名称。 |
| `DOCKER_NAME_D` | `environment` | Decode 容器基础名称。 |
| `DOCKER_NAME_C` | `environment` | Proxy 容器基础名称。 |
| `SCRIPTS_PATH` | `environment` | Ansible 生成脚本的共享目录，会以相同路径挂载进容器。 |

不要混淆：

- `CODE_PATH` 是远端宿主机代码位置。
- `container_workspace` 是容器内代码根目录，默认 `/workspace`。
- `SCRIPTS_PATH` 是 Ansible 生成脚本的中转目录，通常使用 `/tmp` 下的独立目录。

## 7. Playbook Profile 参考

### 7.1 `container_workspace`

容器内 OmniInfer 仓库的父目录，默认：

```yaml
container_workspace: /workspace
```

仓库实际位置为 `{{ container_workspace }}/omniinfer`。该变量同时影响代码复制、
安装命令、Prefill/Decode/Proxy 默认工作目录和绑核脚本。
如果只需要为某个服务使用特殊工作目录，覆盖对应的 `workdir`，不要修改全局变量。

### 7.2 `run_docker_profile`

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `shm_size` | `500g` | Docker `--shm-size`。 |
| `envs` | `[]` | 创建容器时传递的完整 `NAME=value` 列表。 |
| `extra_mounts` | `[]` | 额外挂载列表，每项格式为 `source:destination`。 |

公共 role 已固定网络、设备、权限、entrypoint 和基础挂载。`extra_mounts` 不会与
playbook 中的其他列表自动拼接；playbook 一旦设置该字段，就应列出它需要的全部
额外挂载。

`run_docker` 会先停止并删除同名旧容器，然后创建新容器，具有破坏性。

### 7.3 `sync_code_profile`

`host_sync` 控制执行机到远端宿主机的 rsync：

公共和弹性同步 task 保持使用 `ansible.builtin.synchronize`。

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `true` | 是否执行宿主机代码同步。 |
| `source` | `{{ ansible_env.CODE_PATH }}/` | rsync 源目录。 |
| `destination` | `{{ ansible_env.CODE_PATH }}/` | rsync 目标目录。 |
| `delete` | `true` | 删除目标端不存在于源端的文件，使用前必须确认目录正确。 |
| `recursive` | `true` | 递归同步目录。 |
| `rsync_opts` | `[]` | 额外 rsync 参数。 |
| `throttle` | `0` | 同步任务并发上限；`0` 使用 Ansible 默认行为。 |

可选的 `container_copy` 包含三个服务命令：

```yaml
sync_code_profile:
  container_copy:
    prefill: "完整 Bash 命令或 null"
    decode: "完整 Bash 命令或 null"
    proxy: "完整 Bash 命令或 null"
```

这些命令在远端宿主机执行，Role 只提供实际容器名环境变量。命令必须自行完成
清理、建目录和 `docker cp`。镜像已包含正确代码时可以省略整个
`container_copy`。

### 7.4 `pip_install_profile`

该 profile 没有公共默认值。仅在需要更新容器内安装包或编译代码时配置：

```yaml
pip_install_profile:
  prefill: "完整 Bash 命令或 null"
  decode: "完整 Bash 命令或 null"
  proxy: "完整 Bash 命令或 null"
```

命令在远端宿主机执行，直接使用 task 注入的 `$DOCKER_NAME_P`、
`$DOCKER_NAME_D` 或 `$DOCKER_NAME_C` 进入对应容器。没有安装步骤时不要在
playbook 中声明空 profile。

### 7.5 `run_server_prefill_profile`

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `runner` | `pd_run_pangu_ultra_moe.sh` | `workdir` 下的 Prefill 启动脚本文件名。 |
| `workdir` | `{{ container_workspace }}/omniinfer/tools/deploy/start_server` | 容器内启动脚本目录。 |
| `docker_envs` | `{}` | 通过 `docker exec -e` 传入启动进程的环境变量。 |
| `prepare_commands` | `""` | 加载 `.bashrc` 后、CLI 启动前执行的 Bash；环境变量使用显式 `export`。 |
| `args` | `[]` | 追加到 PD runner 的有序 CLI 参数列表。 |
| `multi_node_backend` | `ray` | 多节点执行后端；设为 `null` 时不追加该参数。单节点固定使用 `mp`。 |

### 7.6 `run_server_decode_profile`

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `runner` | `pd_run_pangu_ultra_moe.sh` | `workdir` 下的 Decode 启动脚本文件名。 |
| `workdir` | `{{ container_workspace }}/omniinfer/tools/deploy/start_server` | 容器内启动脚本目录。 |
| `docker_envs` | `{}` | 通过 `docker exec -e` 传入启动进程的环境变量。 |
| `prepare_commands` | `""` | 加载 `.bashrc` 后、CLI 启动前执行的 Bash；环境变量使用显式 `export`。 |
| `args` | `[]` | 追加到 PD runner 的有序 CLI 参数列表。 |

`docker_envs` 和 `prepare_commands` 不可互换：

- 需要在 `docker exec` 创建进程时就存在的变量放入 `docker_envs`。
- 依赖容器 `.bashrc` 或只影响 runner 的变量在 `prepare_commands` 中显式
  `export`。
- 通信网卡由 `run_server` 自动检测并通过 `SOCKET_IFNAME` 固定传入，不需要在
  `docker_envs` 中重复配置。
- 仅作为 CLI 参数使用的值应直接写入 `args`，例如
  `--max-model-len 524288`，无需先定义成环境变量再传入容器。

Profile 合并、通信网卡检测以及 P/D 的 `docker exec` 命令均直接写在
`roles/common/tasks/run_server.yml` 中，没有单独的 resolver task。

公共 task 固定传入 `SOCKET_IFNAME`、`HOST_IP`、`MASTER_PORT`、`API_PORT`、
`OMNI_PD_PREFILL_POD_NUM` 和 `OMNI_PD_DECODE_POD_NUM`；Prefill 还会固定传入
`IP`、`NNODES`、`NODE_RANK` 和 `NODE_IP_LIST`，Decode 固定传入当前
`inventory_hostname` 对应的 `HOST`。公共 J2 固定角色、KV rank、Decode
`--num-servers` 和 `--num-dp` 等拓扑参数。并行度、显存占用和模型特异参数应
通过 playbook 的 `docker_envs`、`prepare_commands` 和 `args` 提供；模型路径
通过 `model_path` 配置，KV Connector 通过
`run_server_common_profile.kv_connector` 配置。

Prefill 模板会传入 `--ascend-rt-visible-devices`，Decode 模板不直接传入该参数；
两者都不会强制追加 `--use-inventory-devices`，设备分配继续使用 runner 的默认
行为。

Prefill 和 Decode 启动模板中的 `KV_PARALLEL_SIZE` 都固定为 Prefill Pod 数加
一个 Decode rank，即 `OMNI_PD_PREFILL_POD_NUM + 1`。

Prefill 和 Decode 的 `runner` 分开配置。两边使用同一 runner 时可以都省略并使用
默认值；需要覆盖时应在两个 profile 中分别声明，避免产生隐含的跨服务耦合。

### 7.7 `run_server_common_profile`

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `kv_connector` | `LLMDataDistConnector` | Prefill 和 Decode 共用的 KV Connector；直接生成到启动脚本中。 |
| `multi_node_prefill_wait_seconds` | `20` | 多节点 Prefill 启动后、Decode 启动前的等待时间。 |
| `single_node_prefill_wait_seconds` | `0` | Decode 启动后、单节点 Prefill 启动前的等待时间。 |
| `restart_proxy` | `false` | `elastic_server` 使用 `run_server` 时是否同时刷新 Proxy。普通 common 流程不用配置。 |

这些值都不常改。全部使用默认值时，不要在 playbook 中声明空的
`run_server_common_profile`。

### 7.8 `mooncake_profile`

Mooncake 不常用，普通部署不用关注，也不建议在新 playbook 中显式配置。
仅当 `run_server_common_profile.kv_connector=LMCacheConnectorV1` 且确实需要
Mooncake/etcd 时才查看：

```yaml
run_server_common_profile:
  kv_connector: LMCacheConnectorV1

mooncake_profile:
  generate_config: true
  start_services: false
  wait_seconds: 0
```

- `generate_config` 生成 Mooncake 和 LMCache 配置文件。
- `start_services` 在 Proxy 容器内启动 etcd 和 mooncake master。
- `wait_seconds` 是停止 Mooncake 进程后的等待时间。
- 使用时 Inventory 还必须提供三个 Mooncake 端口。

### 7.9 `run_proxy_profile`

| 字段 | 说明 |
| --- | --- |
| `type` | Proxy 类型；可选 `omni-proxy` 或 `global-proxy`，默认 `omni-proxy`。 |
| `prepare_commands` | 加载 `.bashrc` 后、启动或 reload Proxy 前执行的 Bash；环境变量使用显式 `export`。 |
| `omni-proxy.workdir` | Omni Proxy 工作目录。 |
| `omni-proxy.command` | Omni Proxy 主命令。 |
| `omni-proxy.args` | Omni Proxy 有序 CLI 参数。 |
| `global-proxy.workdir` | Global Proxy 工作目录。 |
| `global-proxy.command` | Global Proxy 主命令。 |
| `global-proxy.args` | Global Proxy 有序 CLI 参数。 |

默认 Omni Proxy 使用
`{{ container_workspace }}/omniinfer/components/omni-proxy/omni_proxy/`
和 `bash omni_proxy.sh`；Global Proxy 保持源配置中的
`{{ container_workspace }}/omniinfer/tools/scripts` 和
`bash global_proxy.sh`。

模板渲染时通过 `type` 选择 Proxy 实现，不再依赖运行时环境变量。
监听端口和 P/D endpoint 参数由公共 J2 根据 `type` 固定生成，不需要写入
`args`。
`prepare_commands` 同时作用于普通启动和 reload。
`reload_proxy` 只在 `type: omni-proxy` 时执行，并在原 `args` 后追加 `--reload`。
当前只有 `elastic_server` 入口导入该任务。

### 7.10 `proc_bind_profile`

```yaml
proc_bind_profile:
  enabled: false
```

默认不执行实际绑核。`proc_bind` 阶段仍会为 P/D 容器中的
`tools/deploy/ansible/scripts/bind_cpu.sh` 执行 `chmod +x`；启用后才会等待
Decode 服务就绪，并通过 task 环境中的 `ROLE`、`SCRIPTS_PATH` 执行该脚本。
该 helper 路径由 `container_workspace` 决定，不跟随 Prefill/Decode `workdir`。

### 7.11 `fetch_log_profile`

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `create_directory` | `true` | 在执行机创建每个 host 的日志目录。 |
| `fetch` | `true` | 从远端拉取 `<LOG_PATH>/<inventory_hostname>/`。 |

一般不需要在 playbook 中配置。只想部署、不想拉日志时通过 extra vars 临时关闭。

### 7.12 `delete_node_profile`

该 profile 属于 `elastic_server`，仅用于显式 `delete_node` tag：

```yaml
delete_node_profile:
  ips:
    - 10.0.0.10
```

默认列表为空，执行删除时会校验 IPv4。Role 刷新 Proxy 后，在目标 IP 上停止
vLLM，并使用 `DOCKER_NAME_P`、`DOCKER_NAME_D` 名称过滤器停止和删除匹配容器。
建议通过 `-e` 临时传入，不要把生产 IP 提交到 playbook。执行前必须先从
Inventory 的 P/D 分组中移除目标节点，使 Proxy 按删除后的拓扑刷新；目标 IP
仍须能使用 `all.vars` 中的 SSH 配置连接。

## 8. Tags

| Tag | 行为 | 注意事项 |
| --- | --- | --- |
| `always` | 生成容器名和部署拓扑。 | 选择其他 tag 时仍会运行，除非显式 `--skip-tags always`。 |
| `run_docker` | 删除旧容器并重新创建 P/D/C 容器。 | 有破坏性；会丢失未挂载的容器内数据。 |
| `clean_up` | 只停止并删除旧容器。 | 不重新创建容器。 |
| `sync_code` | rsync 到远端，并按需复制进容器。 | 默认会删除目标端多余文件；执行前必须确认源和目标目录。 |
| `pip_install` | 执行 `pip_install_profile` 中的完整命令。 | 未配置或为 `null` 时跳过。 |
| `stop_server` | 停止 Prefill/Decode/Ray，并处理 Mooncake 停止。 | 不删除容器。 |
| `run_server` | 计算启动命令并启动 Prefill/Decode。 | 假设容器、代码和依赖已经准备好。 |
| `proc_bind` | 按需执行 CPU 绑核。 | `enabled=false` 时不会执行实际绑核。 |
| `run_proxy` | 停止旧 Proxy 并启动新 Proxy。 | 依据 `run_proxy_profile.type` 选择实现。 |
| `fetch_log` | 拉取日志到执行机。 | 默认创建目录并拉取。 |
| `reload_proxy` | 重新生成脚本并 reload Omni Proxy。 | 仅 `elastic_server` 支持；带 `never`，必须显式选择。 |
| `delete_node` | 按删除后的 Inventory 刷新 Proxy，并删除指定节点。 | 带 `never`；先从 P/D 分组移除目标节点，再显式传入 IP。 |
| `add_node` | 新节点建容器并激活服务。 | 仅 `elastic_server` 入口支持。 |
| `add_node_with_sync_code` | 新节点同步代码后激活。 | 仅 `elastic_server` 入口支持。 |

`elastic_server` 将“仅使用已有代码”与“同步代码后添加”拆成 `add_node` 和
`add_node_with_sync_code`。这两个流程只在目标 P/D 容器均不存在时创建并激活
节点；若检测到同名容器已存在，不会通过就绪标记或健康探针修复部分完成的激活。

`stop_server` 和 `run_proxy` 的旧进程清理属于 best-effort 操作：清理命令失败
不会令 Play 失败。`delete_node` 刷新 Proxy 后会停止目标节点上的 vLLM，等待固定
30 秒，再按 P/D 容器基础名称过滤并删除容器；该流程不额外等待 Proxy 健康检查。

## 9. 常用命令

从 `tools/deploy/ansible` 目录执行：

```bash
PLAYBOOK=playbooks/omni_infer_server_template_panguv2.yml
INVENTORY=/path/to/inventory.yml

# 查看语法、任务和 tags，不修改远端
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --syntax-check
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --list-tasks
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --list-tags

# 执行完整部署；never 类任务不会自动执行
ansible-playbook -i "$INVENTORY" "$PLAYBOOK"

# 分阶段执行
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --tags run_docker
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  --tags sync_code,pip_install
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  --tags stop_server,run_server,proc_bind,run_proxy
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --tags fetch_log

# 仅 elastic_server：reload Omni Proxy
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --tags reload_proxy

# 删除节点；高风险操作
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --tags delete_node \
  -e '{"delete_node_profile":{"ips":["10.0.0.10"]}}'
```

Tags 只做筛选，不自动补齐依赖。例如单独执行 `run_server` 不会创建容器或同步代码。

新增或修改自己的 Playbook 时，至少执行本节开头的 `--syntax-check`、
`--list-tasks` 和 `--list-tags`。高风险生命周期只能使用测试 Inventory 在隔离
环境验证，提交前再执行 `git diff --check`。

仓库内的安全 fixture 可用于本地静态验证：

```bash
ansible-playbook -i examples/inventory_1p1d.yml "$PLAYBOOK" --syntax-check
ansible-playbook -i examples/inventory_1p1d.yml "$PLAYBOOK" --list-tags
ansible-playbook -i examples/inventory_1p1d.yml "$PLAYBOOK" \
  --check --tags always
```

该 fixture 通过 `ansible_connection: local` 使用 loopback 地址。不要对它执行
完整部署、`run_docker` 或弹性生命周期 tags，以免操作执行机上的 Docker。

## 10. 新增专用 Role

需求包含新增 task、改变公共阶段顺序、弹性节点或特殊生命周期时，新增按能力命名的
Role，做法参考 `elastic_server`：

```text
roles/my_server/
├── README.md
├── defaults/main.yml       # 只放该流程真正需要的默认值
└── tasks/
    ├── main.yml            # 显式定义阶段顺序
    └── special_stage.yml   # 专用 task
```

`tasks/main.yml` 示例：

```yaml
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
---
- ansible.builtin.import_role:
    name: common
    tasks_from: run_docker

# Insert the role-specific stage at its required position.
- ansible.builtin.import_tasks: special_stage.yml
  tags: special_stage

- ansible.builtin.import_role:
    name: common
    tasks_from: set_topology
- ansible.builtin.import_role:
    name: common
    tasks_from: run_server
```

Playbook 只选择这个新入口：

```yaml
tasks:
  - ansible.builtin.import_role:
      name: my_server
```

如果同一 Role 有少量稳定的任务序列变体，可以增加 `tasks_from` 文件；不要通过一个
巨型 `main.yml` 增加大量布尔分支。还应遵守：

- 能复用的 task 继续从 common 导入，不复制 common task。
- 模型参数继续由 Playbook Profile 提供，Role 只负责任务编排和专用生命周期。
- 执行顺序由 `import_role` / `import_tasks` 的声明顺序明确表达。
- Tags 只筛选任务，不改变声明顺序；非默认生命周期使用 `never` 加语义 tag。
- Task 使用 FQCN 和清晰的英文 `name`，P、D、C 专用任务用 `when` 限定作用域。
- 只有脚本结构或执行顺序不同才新增模板；仅参数不同仍通过 Playbook 配置。
- 每个专用 Role 提供 README，说明入口、任务顺序、公开 Tags 和使用示例。
