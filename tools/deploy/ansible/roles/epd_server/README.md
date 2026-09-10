# EPD deployment role

`epd_server` manages a fixed Encoder / Prefill / Decode topology and generates the
EC, multimodal feature and Proxy routing configuration. It imports the existing
`common` stages for P/D/container/code operations. The model playbook keeps model
paths, plugin selection, media preprocessing and performance flags.

`tasks/main.yml` lists the deployment stages in execution order. EPD stage files
import the existing `common` tasks and add Encoder operations locally. Stages with
no Encoder additions import common directly from main:

| Stage file | Tags | Operations |
| --- | --- | --- |
| `set_topology.yml` | `always` | Validate and build Encoder topology, then import common topology. |
| `run_docker.yml` | `run_docker`, `clean_up`, common `always` | Prepare the independent E container and Encoder log directories, then import common P/D/C container preparation. |
| `deploy_code.yml` | `sync_code`, `pip_install` | Synchronize/copy and install independent E code, then import the existing common code stage under the respective tags. |
| `stop_server.yml` | `stop_server` | Stop independent E, then stop P/D and co-located E through common. |
| `run_server.yml` | `run_server` | Resolve and validate the Encoder profile, launch E in the background, then import common P/D startup. |
| common `bind_cpus` (direct import) | `proc_bind` | Reuse common CPU binding. |
| common `run_proxy` (direct import) | `run_proxy` | Reuse common Proxy startup with EPD routing. |
| `fetch_log.yml` | `fetch_log` | Collect independent E logs, then import common P/D/C log collection. |

Encoder operations precede the common import in each deployment stage. The code
stage keeps common's existing `deploy_code` boundary; common is unchanged.
Independent E hosts resolve their code profiles locally and synchronize once per
E physical host before copying code into E containers. If E and P/D/C share a
physical host, common performs its existing host synchronization afterward.
Tags select tasks within this order. Every stage is imported once, so combining
`stop_server,run_server,run_proxy` does not repeat stopping or Proxy startup.
`run_server` selects E/P/D startup. Use `stop_server,run_server` to stop old
services before launching them, and select `run_proxy` to start the Proxy.
`proc_bind` remains an independent selection. `clean_up` only selects container
removal, never creation.

Configuration generation uses native Ansible and Jinja. `set_topology.yml`
collects Encoder and MM receiver endpoints in inventory order, then builds the
current host's transport configuration. `epd_topology_config.prefill` and
`epd_topology_config.decode` contain local EC/MM settings, or an empty dictionary
on hosts outside the respective group. Encoder settings are registered directly
as local `epd_encoder_*` facts. `epd_topology_config.encode_endpoints` contains
Encoder endpoint strings in inventory order. The playbook builds the Proxy
argument array in `run_proxy_profile.prepare_commands`; `args` expands that array.
Network mode requires a nonempty E group. Each P/D profile formats the
current host's EC/MM configuration and appends the transport arguments to
`EXTRA_ARGS` in `prepare_commands`. Ansible renders these commands after topology
is available. Proxy preparation also reads the Encoder endpoints at that stage.
The role does not parse or rewrite model arguments or generate profile overrides.
Encoder profiles resolve and validate at the start of `run_server`, after
placement and common topology are available. They can reference those facts and
playbook transport arguments. Common resolves P/D profiles in `run_server` and
the Proxy profile in `run_proxy`.

The supplied playbook is
`playbooks/epd/omni_infer_server_template_performance1P1D_92B_VL_a3_low_latency.yml`.
Its P/D model and performance settings come from the
[feature/vllm_0.25.1_vl template](https://gitee.com/xingchengis/omniinfer/blob/feature/vllm_0.25.1_vl/tools/deploy/ansible/playbooks/omni_infer_server_template_performance1P1D_92B_VL_a3_low_latency.yml).
Encoder settings follow the existing 92B VL EPD template in omni-models.

## Supported topologies

| Configuration | SharedMemConnector | NetworkConnector |
| --- | --- | --- |
| Encoder placement | One process inside each P container | Independent E inventory hosts/containers |
| Encoder devices | Full P device list, shared with P | E `ascend_rt_visible_devices` |
| Encoder HTTP port | P `e_api_port` | E `api_port` |
| EC transfer | Shared memory | E `host_ip` and `ec_port` |
| MM features | E disk output to P; network output to D | E network output to P and D |
| Proxy routing | Paired E/P endpoint groups | Encode endpoints and sequential routing |

P/D topology and startup use the existing common stages. Encoder configuration
is generated per inventory entry, with SharedMem E/P pairing following the P
inventory order. The `elastic_server` expansion and contraction tags are not
supported; do not invoke the two roles together. Service endpoints are collected
from the full inventory; `--limit` selects the hosts on which tasks execute.

SharedMem does not reserve or split a dedicated Encoder card. E and P share the
entire device list and container. Their HTTP ports must differ. Network mode
requires an `E` group with `ansible_host`, `host_ip`, `api_port`, `ec_port`, and
`ascend_rt_visible_devices`. P/D MM listeners default to port 5555; set host-level
`mm_feature_transfer_port` values when P and D share an IP, for example
P 5556 and D 5555.

The compatible Network EC connector selects its local IP from the default route
(to `8.8.8.8`), rather than the Encoder `host_ip` setting. On hosts with multiple
network interfaces, ensure this route selects the advertised E address and that
P can reach it; an inventory value alone does not change the connector binding.

## Configuration

```yaml
epd_profile:
  enabled: true
  connector_type: SharedMemConnector
  mm_feature_transfer_port: 5555
  mm_feature_storage_root: /tmp/omni_epd_mm_features
  ec_cache_max_gb: 50
  load_ec_async: true

run_server_encode_profile:
  docker_envs: {}
  prepare_commands: ""
  args:
    - --served-model-name pangu_ultra_moe
    - --max-model-len 16384
```

`ec_cache_max_gb` is GiB. For the compatible shared-memory implementation it
becomes `ec_shared_memory_max_bytes`; that historical field is converted to GiB
inside the connector. Confirm this convention in the target plugin version.
`run_server_encode_profile.args` contains Bash CLI fragments, matching existing
P/D `args`. Configure the served model name and maximum model length in this
list. The Encoder template receives `model_path` and supplies the model path,
host, port, TP, DP, encoder-only, EC and MM arguments; do not repeat those flags.

Append transport arguments in each profile's `prepare_commands`, after the
existing model arguments:

```yaml
run_server_prefill_profile:
  prepare_commands: |-
    # Existing model preparation, including EXTRA_ARGS assignments.
    {% set config = epd_topology_config.prefill %}
    {% if config.get('ec_config') %}
    EXTRA_ARGS="${EXTRA_ARGS} --ec-transfer-config "{{ config.ec_config | to_json(separators=(',', ':')) | quote }}
    {% endif %}
    EXTRA_ARGS="${EXTRA_ARGS} --mm-feature-transfer-config "{{ config.mm_config | to_json(separators=(',', ':')) | quote }}
  args:
    # Existing runner options.
    - --extra-args "${EXTRA_ARGS}"

run_server_decode_profile:
  prepare_commands: |-
    # Existing model preparation, including EXTRA_ARGS assignments.
    EXTRA_ARGS="${EXTRA_ARGS} --mm-feature-transfer-config "{{ epd_topology_config.decode.mm_config | to_json(separators=(',', ':')) | quote }}
  args:
    # Existing runner options.
    - --extra-args "${EXTRA_ARGS}"
```

`quote` protects the JSON while Bash appends it to `EXTRA_ARGS`. The runner's
JSON-aware argument parser then preserves each object as one CLI value.
The double-quoted expansion passes one `--extra-args` value through the existing
common template and runner.

List fixed Proxy options directly in `args`. Build routing arguments in
`prepare_commands` and expand the Bash array as the final `args` entry:

```yaml
run_proxy_profile:
  prepare_commands: |-
    export PYTHONHASHSEED=1234
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0

    encode_endpoints={{ epd_topology_config.encode_endpoints | join(',') | quote }}
    proxy_args=()
    {% if epd_profile.enabled %}
    proxy_args+=(--encode-endpoints "$encode_endpoints")
    {% endif %}
    {% if epd_profile.enabled and epd_profile.connector_type == 'SharedMemConnector' %}
    epd_proxy_groups={{ range(epd_topology_config.encode_endpoints | length) | map('string') | map('regex_replace', '$', ':1') | join(',') | quote }}
    proxy_args+=(
      --omni-proxy-encode-ep-groups "$epd_proxy_groups"
      --omni-proxy-prefill-ep-groups "$epd_proxy_groups"
      --omni-proxy-pd-policy epd_e_p_node_share
    )
    {% else %}
    proxy_args+=(--omni-proxy-pd-policy sequential)
    {% endif %}
  args:
    - --log-level notice
    - --core-num 4
    - '"${proxy_args[@]}"'
```

When migrating older profiles, retain exactly one `--extra-args` and remove
manually supplied EC/MM transfer flags and EPD Proxy routing flags. Automatic
merging, replacement and deduplication are no longer performed. If overriding
P/D `prepare_commands` or `args`, retain the assignment and explicit reference
above. Keep the final array reference when editing Proxy `args` so generated
routes remain included. The quoted array expansion preserves argument boundaries
without `eval` or unquoted string splitting.

`epd_profile.enabled: false` runs ordinary PD through the same role: no Encoder
is launched, the role restores P-to-D multimodal feature transfer, and the
playbook selects sequential Proxy routing. Stop the existing EPD services using the original
configuration before deploying this PD configuration, so the old independent
Encoder is not left running.

## Code, namespace and shared data

Update the image, source/model paths, container names and real inventory before
running deployment. `CODE_PATH` must contain an `omniinfer/` source directory.
The sample copies it into `{{ container_workspace }}/omniinfer`, retaining
image components. It does not declare `pip_install_profile`; runtime dependencies
are provided by the image. SharedMem E reuses the P source copy. Independent E
uses `sync_code_profile.container_copy.encode`, which receives the resolved
`DOCKER_NAME_E` from the role.

The role still supports optional `pip_install_profile` commands for deployments
that need installation or updates. `pip_install_profile.encode` receives the
same `DOCKER_NAME_E` for independent E containers.

The current project distribution is `omni_infer`; its Python namespace is
`omni_npu`. File copying alone does not guarantee that Python loads this source
instead of a preinstalled wheel. The image must provide compatible
CANN/torch/vLLM and model plugins. The sample preserves the supplied VL plugin
names and selects `OMNI_VLLM_PATCHES_DIR=low_latency` with
`OMNI_VLLM_PATCHES=ALL`, matching develop's `patches/common` and
`patches/models/{pangu_v2_base,low_latency}` layout. VL model registration is
provided by the image's `omni_pangu_models` plugin. The target plugin must
recognize those settings. The Encoder command uses vLLM 0.25.1's native
`--mm-encoder-only` flag to skip the language model; the old
`--convert mm_encoder_only` syntax is not supported by this version.
Before deployment, verify EC shared-memory/network connectors, MM feature connectors and
compatible Omni Proxy EPD routing. The role does not install missing
Encoder/EC dependencies in a locally trimmed plugin tree or upgrade the
Proxy binary bundled with an older image.

E and P must use the same container working directory in SharedMem mode because
the local MM feature connector keeps its metadata database in the working directory. The default is
`{{ container_workspace }}/omniinfer/tools/deploy/start_server`. Keep the related
volume and IPC namespaces accessible to both processes. The model profiles use
separate object-storage buffer names (`ENCODE_SHM_BUFFER`, `PREFILL_SHM_BUFFER`,
`DECODE_SHM_BUFFER`); the role-generated EC configuration is a separate transfer
mechanism. MM feature files are isolated under
`epd_profile.mm_feature_storage_root/<inventory_hostname>`; co-located E/P use
the P path while D gets its own path, even when `/tmp` is host-mounted.
Run stop/cleanup before starting E so cache cleanup cannot erase
newly produced Encoder data.

Encoder output depends on container placement:

- SharedMem: `{{ LOG_PATH }}/{{ inventory_hostname }}/encode/server_0.log`,
  under the paired P inventory name.
- Network: `{{ LOG_PATH }}/{{ inventory_hostname }}/server_0.log`,
  under the independent E inventory name.

`fetch_log` includes both layouts. P/D and Proxy retain their common log layout.

## Configure and inspect the reference inventory

The supplied inventory is
`inventories/epd/omni_infer_inventory_used_for_EPD.yml`. Its active P/D/C
layout is for SharedMem; the independent E group is commented out for Network
mode. Update the addresses, connection settings, devices and ports for your
deployment, or copy the file and configure that copy.

Run these commands from `tools/deploy/ansible` on a Linux Ansible controller
with the collections used by `common` installed. Set `INVENTORY` to your
configured file. The commands inspect the inventory and playbook structure;
they do not validate model, driver, plugin or NPU readiness.

```bash
PLAYBOOK=playbooks/epd/omni_infer_server_template_performance1P1D_92B_VL_a3_low_latency.yml
INVENTORY=inventories/epd/omni_infer_inventory_used_for_EPD.yml

ansible-inventory -i "$INVENTORY" --graph
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --syntax-check
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --list-hosts
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --list-tasks
```

The playbook defaults to SharedMem. For Network, uncomment and configure the E
group in the same inventory, then select Network explicitly. Playbook vars take
precedence over inventory vars, so setting the connector only in inventory does
not override the playbook:

```bash
NETWORK_VARS='{"epd_profile":{"enabled":true,"connector_type":"NetworkConnector"}}'
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" -e "$NETWORK_VARS" --syntax-check
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" -e "$NETWORK_VARS" --list-tasks
```

## Deploy a configured inventory

Use the configured inventory and matching image/source. A full run prepares
containers and code, stops old services, launches E with `docker exec -d`,
then proceeds to P/D and the EPD Proxy without polling the Encoder `/health`
endpoint. Deployment completion confirms launch commands were issued; check
Encoder readiness and startup failures in its log. To restart existing E/P/D
services, select `stop_server,run_server` so stopping old services precedes
Encoder startup. Include `run_proxy` when the Proxy also needs to be restarted.

```bash
# INVENTORY must contain your configured deployment hosts.
ansible-playbook -i "$INVENTORY" "$PLAYBOOK"

# Repeat the same Network extra-vars here if the deployment uses Network mode.
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" \
  --tags stop_server,run_server,run_proxy
ansible-playbook -i "$INVENTORY" "$PLAYBOOK" --tags fetch_log
```

Changing between SharedMem, Network and PD can move or remove E processes and
containers. Stop services with the old configuration before starting the new
configuration. The role currently manages fixed topology lifecycle; dynamic
E/P pairing during elastic resizing is not implemented.
