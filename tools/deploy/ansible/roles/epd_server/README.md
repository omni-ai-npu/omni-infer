# EPD deployment role

`epd_server` manages a fixed Encoder / Prefill / Decode topology and generates the
EC, multimodal feature and Proxy routing configuration. It imports the existing
`common` stages for P/D/container/code operations. Model paths, plugin selection,
media preprocessing and performance flags are supplied through role variables
and profiles.

`tasks/main.yml` lists the deployment stages in execution order. EPD stage files
import the existing `common` tasks and add Encoder operations locally. Stages with
no Encoder additions import common directly from main:

| Stage file | Tags | Operations |
| --- | --- | --- |
| `set_topology.yml` | `always` | Validate and build Encoder topology, then import common topology. |
| `run_docker.yml` | `run_docker`, `clean_up`, common `always` | Prepare the independent E container and Encoder log directories, then import common P/D/C container preparation. |
| `deploy_code.yml` | `sync_code`, `pip_install` | Synchronize/copy and install independent E code, then import the existing common code stage under the respective tags. |
| `stop_server.yml` | `stop_server` | Stop independent E, then stop P/D and co-located E through common. |
| `run_server.yml` | `run_server` | Validate Encoder startup, clean service SHM/MM data, configure Network socket buffers, save the E launch script, launch E in the background, then import common P/D startup. |
| common `bind_cpus` (direct import) | `proc_bind` | Reuse common CPU binding. |
| common `run_proxy` (direct import) | `run_proxy` | Reuse common Proxy startup with EPD routing. |
| `fetch_log.yml` | `fetch_log` | Collect independent E logs, then import common P/D/C log collection. |

Encoder operations precede the common import in each deployment stage. The code
stage keeps common's existing `deploy_code` boundary. Independent E hosts resolve
their code profiles locally and synchronize once per E physical host before
copying code into E containers. If E and P/D/C share a physical host, common
performs its existing host synchronization afterward.

Tags select tasks within this order. Every stage is imported once, so combining
`stop_server,run_server,run_proxy` does not repeat stopping or Proxy startup.
`run_server` selects E/P/D startup. Use `stop_server,run_server` to stop old
services before launching them, and select `run_proxy` to start the Proxy.
`proc_bind` remains an independent selection. `clean_up` only selects container
removal, never creation.

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

## Configuration

```yaml
epd_profile:
  enabled: true
  connector_type: SharedMemConnector
  mm_feature_transfer_port: 5555
  mm_feature_storage_root: /tmp/omni_epd_mm_features
  ec_cache_max_gb: 50
  load_ec_async: true
```

These are the role defaults; `epd_profile` overrides them recursively.
`enabled: true` selects EPD, with `SharedMemConnector` or `NetworkConnector`
controlling Encoder placement and transport. `enabled: false` runs ordinary PD
without an Encoder and restores P-to-D MM feature transfer.
`mm_feature_transfer_port` is the default MM receiver port; a host-level value
overrides it. `mm_feature_storage_root` is the root for host-local MM files.
`load_ec_async` controls asynchronous EC loading for the Network consumer.

`ec_cache_max_gb` is GiB. For the compatible shared-memory implementation it
becomes `ec_shared_memory_max_bytes`; that historical field is converted to GiB
inside the connector. Confirm this convention in the target plugin version.

### Launch profiles

`run_server_encode_profile` provides `workdir`, `docker_envs`,
`prepare_commands` and `args`. Its default working directory is
`{{ container_workspace }}/omniinfer/tools/deploy/start_server`; the other fields
default to an empty mapping, string and list respectively.
`run_server_encode_profile.args` contains Bash CLI fragments, matching existing
P/D `args`. Supply model-specific options and the `--mm-encoder-only` flag in
this list. The Encoder template receives `model_path` and supplies the model
path, host, port, TP, DP, EC and MM arguments; do not repeat those flags.

P/D and Proxy use `run_server_prefill_profile`, `run_server_decode_profile` and
`run_proxy_profile` from common. Their `prepare_commands` and `args` must
explicitly consume the generated transport and routing configuration below.
Encoder profiles resolve and validate at the start of `run_server`; common
resolves P/D profiles in `run_server` and the Proxy profile in `run_proxy`.
All these profiles can reference the topology facts at that stage.

### Generated transport configuration

Configuration generation uses native Ansible and Jinja. `set_topology.yml`
collects Encoder and MM receiver endpoints in inventory order, then builds the
current host's transport configuration. `epd_topology_config.prefill` and
`epd_topology_config.decode` are empty dictionaries on hosts outside the
respective group. Encoder settings are registered as local `epd_encoder_*` facts
only on Encoder hosts.

EC configuration describes the transfer of Encoder cache data to Prefill. MM
configuration describes multimodal feature storage and transfer endpoints.
These values are configuration dictionaries, not the cached tensors or feature
files themselves.

| Fact | Contents |
| --- | --- |
| `epd_topology_config.prefill.ec_config` | P's EC consumer connector, E addresses/ports in Network mode, and connector cache/loading options. `null` when EPD is disabled. |
| `epd_topology_config.prefill.mm_config` | P's MM connector configuration: a local disk reader in SharedMem mode, a network receiver with local storage in Network mode, or a network producer in PD mode. |
| `epd_topology_config.decode.mm_config` | D's MM network receiver and local disk storage configuration. |
| `epd_topology_config.encode_endpoints` | Encoder HTTP `host:port` strings in inventory order; empty when EPD is disabled. |
| `epd_encoder_ec_config` | E's EC producer connector and cache/port settings. |
| `epd_encoder_mm_config` | E's MM producer configuration: local disk plus network output in SharedMem mode, or network output in Network mode. |

Serialize the P/D dictionaries as JSON for `--ec-transfer-config` and
`--mm-feature-transfer-config`. Omit the EC flag when `ec_config` is `null`.
Keep each JSON object as a single CLI value when passing it through the common
runner. The Encoder template handles serialization of its own EC/MM settings.
The role does not parse or rewrite model arguments or generate profile overrides.

The Proxy profile consumes `encode_endpoints` through `--encode-endpoints`.
SharedMem mode requires matching E/P groups in inventory order and the
`epd_e_p_node_share` policy. Network and ordinary PD use sequential routing;
PD omits Encoder endpoints. These routing arguments must be supplied explicitly
through the Proxy profile.

## Code and shared data

Configure the image, source/model paths, container names and inventory for the
deployment. SharedMem E reuses the P source copy and installed dependencies.
Independent E uses `sync_code_profile.container_copy.encode`, which receives the
resolved `DOCKER_NAME_E` from the role. Optional `pip_install_profile.encode`
commands receive the same container name for installation or updates.

The runtime must provide compatible Encoder/EC connectors, MM feature connectors
and Omni Proxy EPD routing. Configure source synchronization and installation
profiles for the image and plugins in use.

E and P must use the same container working directory in SharedMem mode because
the local MM feature connector keeps its metadata database in the working
directory. Keep the related volume and IPC namespaces accessible to both
processes. Configure distinct `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` values in
E/P/D `docker_envs` for their object-storage buffers; these are separate from the
role-generated EC transfer configuration. MM feature files are isolated under
`epd_profile.mm_feature_storage_root/<inventory_hostname>`; co-located E/P use
the P path while D gets its own path, even when `/tmp` is host-mounted.

During `run_server`, the role cleans the configured object-storage SHM prefixes
and the current inventory host's MM directory contents before launching E/P/D.
SharedMem E/P are cleaned together in the P container before E starts. Network
mode also sets socket send/receive buffer defaults and limits to 16 MiB in the
service containers.

Encoder output depends on container placement:

- SharedMem: `{{ LOG_PATH }}/{{ inventory_hostname }}/encode/server_0.log`,
  under the paired P inventory name.
- Network: `{{ LOG_PATH }}/{{ inventory_hostname }}/server_0.log`,
  under the independent E inventory name.

The generated Encoder launch script is also saved as `run_encode.log` in the
same log directory. `fetch_log` includes both layouts. P/D and Proxy retain their
common log layout.

## Service lifecycle

A full run prepares containers and code, stops old services, launches E with
`docker exec -d`, then proceeds to P/D and the EPD Proxy without polling the
Encoder `/health` endpoint. Deployment completion confirms launch commands were
issued; check Encoder readiness and startup failures in its log. To restart
existing E/P/D services, select `stop_server,run_server` so stopping old services
precedes Encoder startup. Include `run_proxy` when the Proxy also needs to be
restarted.

Changing between SharedMem, Network and PD can move or remove E processes and
containers. Stop services with the old configuration before starting the new
configuration. The role currently manages fixed topology lifecycle; dynamic
E/P pairing during elastic resizing is not implemented.
