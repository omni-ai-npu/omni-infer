# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ANSIBLE_ROOT = REPOSITORY_ROOT / "tools" / "deploy" / "ansible"


def _read(relative_path: str) -> str:
    return (ANSIBLE_ROOT / relative_path).read_text(encoding="utf-8")


def _maintained_framework_files():
    roots = [
        ANSIBLE_ROOT / "DEVELOPMENT_GUIDE.md",
        ANSIBLE_ROOT / "README.md",
        ANSIBLE_ROOT / "ansible.cfg",
        ANSIBLE_ROOT / "requirements.yml",
    ]
    for directory in ("examples", "inventory", "playbooks", "roles"):
        roots.extend(
            path
            for path in (ANSIBLE_ROOT / directory).rglob("*")
            if path.is_file()
        )
    return roots


def test_maintained_framework_uses_the_current_repository_layout():
    forbidden = [
        "tools/ansible/",
        "/workspace/omniinfer/tools/scripts",
        "SPDX-License-Identifier: MIT",
        "components/omni-npu",
    ]

    for path in _maintained_framework_files():
        contents = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in contents, f"{value!r} remains in {path}"


def test_only_supported_inventory_templates_are_retained():
    inventory_root = ANSIBLE_ROOT / "inventory"
    expected = {
        "omni_infer_inventory_used_for_1P1D.yml",
        "omni_infer_inventory_used_for_2P1D.yml",
        "omni_infer_inventory_used_for_4P1D.yml",
    }

    assert {path.name for path in inventory_root.iterdir()} == expected
    assert not (ANSIBLE_ROOT / "template").exists()
    for legacy_name in (
        "omni_infer_inventory_used_for_CI.yml",
        "omni_infer_inventory_user_for_long_term_test.yml",
        "omni_infer_server_used_for_CI.yml",
        "omni_infer_server_used_for_long_term_test.yml",
    ):
        assert not (ANSIBLE_ROOT / legacy_name).exists()


def test_playbooks_preserve_the_refactor_source_sync_semantics():
    defaults = _read("roles/common/defaults/main.yml")
    dsv32 = _read("playbooks/omni_infer_server_template_dsv32.yml")
    pangu = _read("playbooks/omni_infer_server_template_panguv2.yml")

    assert "delete: true" in defaults

    assert "source: '{{ ansible_env.CODE_PATH }}/omniinfer'" in dsv32
    assert "delete: false" in dsv32
    assert dsv32.count(
        "docker cp {{ ansible_env.CODE_PATH }}/omniinfer"
    ) == 3

    assert "host_sync:" not in pangu
    for role in ("P", "D", "C"):
        assert f"docker exec $DOCKER_NAME_{role} rm -rf" in pangu
        assert f"docker exec $DOCKER_NAME_{role} mkdir -p" in pangu
    assert pangu.count(
        "docker cp {{ ansible_env.CODE_PATH }}/omniinfer/omni"
    ) == 3
    assert "{{ ansible_env.CODE_PATH }}/omniinfer/." not in pangu

    for contents in (dsv32, pangu):
        assert "components/omni-npu" not in contents
        assert "python -m pip install --no-build-isolation -e" not in contents


def test_elastic_launch_renders_and_runs_its_own_source_commands():
    activate_nodes = _read("roles/elastic_server/tasks/activate_nodes.yml")
    run_server = _read("roles/common/tasks/run_server.yml")

    for template in (
        "common/templates/run_vllm_prefill.sh.j2",
        "common/templates/run_vllm_decode.sh.j2",
    ):
        assert template in activate_nodes

    assert "tasks_from: resolve_run_server" not in activate_nodes
    assert "import_tasks: resolve_run_server.yml" not in run_server
    assert not (
        ANSIBLE_ROOT / "roles" / "common" / "tasks" / "resolve_run_server.yml"
    ).exists()
    assert "Resolve the Prefill run_server profile." in run_server
    assert "Generate the resolved Decode vLLM docker start command." in run_server
    assert "{{ resolved_run_server_vllm_cmd_p }}" in activate_nodes
    assert "{{ resolved_run_server_vllm_cmd_d }}" in activate_nodes


def test_kv_parallel_size_uses_prefill_pods_plus_one_decode_rank():
    expected = "KV_PARALLEL_SIZE=$((OMNI_PD_PREFILL_POD_NUM + 1))"

    for template in (
        "roles/common/templates/run_vllm_prefill.sh.j2",
        "roles/common/templates/run_vllm_decode.sh.j2",
    ):
        assert expected in _read(template)


def test_ansible_launch_does_not_force_inventory_device_selection():
    prefill = _read("roles/common/templates/run_vllm_prefill.sh.j2")
    decode = _read("roles/common/templates/run_vllm_decode.sh.j2")

    assert "--ascend-rt-visible-devices" in prefill
    assert "--ascend-rt-visible-devices" not in decode
    assert "--use-inventory-devices" not in prefill + decode


def test_cpu_binding_preserves_the_source_task_gates_and_environment():
    contents = _read("roles/common/tasks/bind_cpus.yml")

    assert contents.count("resolved_proc_bind_profile.enabled | bool") == 3
    assert contents.count('when: "\'P\' in group_names"') == 1
    assert contents.count('when: "\'D\' in group_names"') == 1
    assert 'ROLE: "P"' in contents
    assert 'ROLE: "D"' in contents
    assert "-e ROLE=P" not in contents
    assert "-e ROLE=D" not in contents
    assert (
        "omniinfer/tools/deploy/ansible/scripts/bind_cpu.sh"
        in contents
    )


def test_elastic_sync_matches_the_source_new_container_flow():
    contents = _read("roles/elastic_server/tasks/sync_node_code.yml")

    assert "ansible.builtin.synchronize:" in contents
    assert "src: \"{{ resolved_sync_code_profile.host_sync.source }}\"" in contents
    assert "dest: \"{{ resolved_sync_code_profile.host_sync.destination }}\"" in contents
    for field in ("delete", "recursive", "rsync_opts", "throttle"):
        assert f"host_sync.{field}" not in contents
    assert "existing_containers.stdout == ''" in contents
    assert "elastic_node_requires_activation" not in contents


def test_runtime_tasks_preserve_source_fact_and_synchronize_names():
    runtime_files = [
        path
        for role_name in ("common", "elastic_server")
        for path in (ANSIBLE_ROOT / "roles" / role_name).rglob("*")
        if path.suffix in {".yml", ".j2"}
    ]
    runtime_files.extend((ANSIBLE_ROOT / "playbooks").glob("*.yml"))

    for path in runtime_files:
        contents = path.read_text(encoding="utf-8")
        assert "ansible_facts.env" not in contents

    assert "ansible.builtin.synchronize:" in _read(
        "roles/common/tasks/deploy_code.yml"
    )
    assert "ansible.builtin.synchronize:" in _read(
        "roles/common/tasks/fetch_logs.yml"
    )
    assert "ansible.builtin.synchronize:" in _read(
        "roles/elastic_server/tasks/sync_node_code.yml"
    )


def test_ansible_module_destinations_preserve_literal_script_paths():
    expected_destinations = {
        "roles/common/tasks/manage_mooncake.yml": (
            "mooncake_config.json",
            "lmcache_mooncake_config.yml",
        ),
        "roles/common/tasks/run_proxy.yml": (
            "kill_nginx_processes.sh",
            "run_proxy_server.sh",
        ),
        "roles/common/tasks/run_server.yml": (
            "vllm_run_for_p.sh",
            "vllm_run_for_d.sh",
        ),
        "roles/common/tasks/stop_server.yml": (
            "kill_python_processes.sh",
            "kill_ray_processes.sh",
        ),
        "roles/elastic_server/tasks/activate_nodes.yml": (
            "mooncake_config.json",
            "lmcache_mooncake_config.yml",
            "vllm_run_for_p.sh",
            "vllm_run_for_d.sh",
        ),
        "roles/elastic_server/tasks/reload_proxy.yml": (
            "reload_proxy_server.sh",
        ),
    }

    for relative_path, filenames in expected_destinations.items():
        contents = _read(relative_path)
        for filename in filenames:
            assert f'dest: "$SCRIPTS_PATH/{filename}"' in contents


def test_elastic_activation_uses_the_source_new_container_gate():
    prepare = _read("roles/elastic_server/tasks/prepare_nodes.yml")
    sync = _read("roles/elastic_server/tasks/sync_node_code.yml")
    activate = _read("roles/elastic_server/tasks/activate_nodes.yml")
    main = _read("roles/elastic_server/tasks/main.yml")

    assert "register: existing_containers" in prepare
    assert "docker inspect" in prepare
    assert "existing_containers.stdout == ''" in prepare
    assert "existing_containers.stdout == ''" in sync
    assert "existing_containers.stdout == ''" in activate
    assert 'rm -rf ${SCRIPTS_PATH}/*' in activate

    for task_name in (
        "tasks_from: set_topology",
        "tasks_from: stop_server",
    ):
        assert task_name in activate

    enhanced_state = (
        "elastic_node_requires_activation",
        "elastic_node_runtime_healthy",
        "com.huawei.omniinfer.elastic-node",
        ".omni_ansible_activation_complete",
        "Application startup complete",
        "elastic_api_readiness",
        "elastic_prefill_worker_readiness",
    )
    for value in enhanced_state:
        assert value not in prepare + sync + activate + main

    assert "Stop stale services before retry-time code synchronization" not in main
    assert main.index("import_tasks: prepare_nodes.yml") < main.index(
        "import_tasks: sync_node_code.yml"
    )
    assert main.index("import_tasks: sync_node_code.yml") < main.index(
        "import_tasks: activate_nodes.yml"
    )


def test_service_stop_matches_the_source_best_effort_behavior():
    stop_server = _read("roles/common/tasks/stop_server.yml")
    run_proxy = _read("roles/common/tasks/run_proxy.yml")

    assert stop_server.count("failed_when: false") == 4
    assert stop_server.count("no_log: true") == 4
    assert run_proxy.count("failed_when: false") == 1
    assert run_proxy.count("no_log: true") == 1
    assert stop_server.count("xargs kill -9") == 4
    assert run_proxy.count("xargs kill -9") == 1
    for process_name in ("VLLM", "vllm", "python", "ray"):
        assert f'grep "{process_name}"' in stop_server
    assert 'grep "nginx"' in run_proxy
    assert "pgrep" not in stop_server + run_proxy


def test_elastic_restart_proxy_matches_the_source_workflow():
    contents = _read("roles/elastic_server/tasks/main.yml")

    assert ").restart_proxy | bool" in contents
    assert "ansible_run_tags" not in contents


def test_elastic_node_deletion_matches_the_source_ip_and_name_filter_flow():
    defaults = _read("roles/elastic_server/defaults/main.yml")
    deletion = _read("roles/elastic_server/tasks/delete_nodes.yml")

    assert "ips: []" in defaults
    assert "inventory_hostnames" not in defaults + deletion
    assert (
        'delete_docker_name_p: "{{ ansible_env.DOCKER_NAME_P }}"'
        in deletion
    )
    assert (
        'delete_docker_name_d: "{{ ansible_env.DOCKER_NAME_D }}"'
        in deletion
    )
    assert "pgrep vllm || true" in deletion
    assert "kill -15 $pids" in deletion
    assert "seconds: 30" in deletion
    assert "--filter 'name={{ hostvars[item].delete_docker_name_p }}'" in deletion
    assert "--filter 'name={{ hostvars[item].delete_docker_name_d }}'" in deletion
    assert "docker stop $container_ids" in deletion
    assert "docker rm $container_ids" in deletion
    assert "delete_node_containers.yml" not in deletion
    assert not (
        ANSIBLE_ROOT
        / "roles"
        / "elastic_server"
        / "tasks"
        / "delete_node_containers.yml"
    ).exists()
    assert "refreshed_proxy_process" not in deletion
    assert "proxy_readiness" not in deletion


def test_plugin_and_proxy_profiles_preserve_the_source_values():
    defaults = _read("roles/common/defaults/main.yml")
    pangu = _read("playbooks/omni_infer_server_template_panguv2.yml")
    dsv32 = _read("playbooks/omni_infer_server_template_dsv32.yml")

    assert defaults.count(
        "{{ container_workspace }}/omniinfer/components/"
        "omni-proxy/omni_proxy/"
    ) == 1
    assert (
        'workdir: "{{ container_workspace }}/omniinfer/tools/scripts"'
        in defaults
    )
    assert "command: bash omni_proxy.sh" in defaults
    assert "command: bash global_proxy.sh" in defaults

    assert pangu.count(
        'VLLM_PLUGINS="omni-npu,omni_pangu_models,'
        'omni_npu_patches,omni_custom_models"'
    ) == 2
    assert dsv32.count(
        'VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models"'
    ) == 2


def test_safe_inventory_fixture_exposes_the_role_contract():
    contents = _read("examples/inventory_1p1d.yml")

    for group in ("P:", "D:", "D0:", "C:"):
        assert group in contents
    for host_field in (
        "ansible_host:",
        "host_ip:",
        "node_rank:",
        "node_port:",
        "api_port:",
        "ascend_rt_visible_devices:",
    ):
        assert host_field in contents
    assert "kv_rank:" in contents
    assert "ansible_connection: local" in contents


def test_required_ansible_collections_are_declared():
    contents = _read("requirements.yml")

    assert "ansible.posix" in contents
    assert "ansible.utils" in contents


def test_deployment_cli_does_not_pin_ansible_core_for_removed_templates():
    contents = (
        REPOSITORY_ROOT / "tools" / "deploy" / "pyproject.toml"
    ).read_text(encoding="utf-8")

    assert '"ansible>=8.0"' in contents
    assert "ansible-core<" not in contents


def test_removed_external_model_config_interface_does_not_remain():
    assert not (REPOSITORY_ROOT / "tests" / "test_config").exists()

    forbidden = (
        "MODEL_EXTRA_CFG_PATH",
        "--model-extra-cfg-path",
        "tests/test_config",
        "/test/test_config",
    )
    text_suffixes = {
        ".json",
        ".j2",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    for root_name in ("docs", "tools", "omni"):
        for path in (REPOSITORY_ROOT / root_name).rglob("*"):
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            if REPOSITORY_ROOT / "omni" / "tests" in path.parents:
                continue
            contents = path.read_text(encoding="utf-8")
            for value in forbidden:
                assert value not in contents, f"{value!r} remains in {path}"

    loader = (
        REPOSITORY_ROOT
        / "omni"
        / "src"
        / "omni_npu"
        / "model_config"
        / "config_loader"
        / "loader.py"
    ).read_text(encoding="utf-8")
    assert "envs.OMNI_CUSTOM_MODEL_CONFIG_PATH" in loader

    pangu_playbook = _read(
        "playbooks/omni_infer_server_template_panguv2.yml"
    )
    assert "export CUSTOM_MODEL_CONFIG_PATH=" not in pangu_playbook
    assert pangu_playbook.count(
        "export OMNI_CUSTOM_MODEL_CONFIG_PATH="
    ) == 2
    custom_paths = re.findall(
        r'export OMNI_CUSTOM_MODEL_CONFIG_PATH="([^"]+)"',
        pangu_playbook,
    )
    config_root = (
        REPOSITORY_ROOT
        / "omni"
        / "src"
        / "omni_npu"
        / "model_config"
        / "configs"
    )
    assert all((config_root / path).is_file() for path in custom_paths)


def test_user_docs_do_not_reference_removed_deployment_entrypoints():
    docs = [
        REPOSITORY_ROOT / "docs" / "omni_infer_installation_guide.md",
        REPOSITORY_ROOT / "docs" / "omni_infer_quick_start.md",
        REPOSITORY_ROOT / "docs" / "omni_cli_usage.md",
    ]
    forbidden = (
        "tools/deploy/ansible/template",
        "tools/ansible/template",
        "omni_infer_server_used_for_",
        "tools/omni_cli/",
    )

    for path in docs:
        contents = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in contents, f"{value!r} remains in {path}"
