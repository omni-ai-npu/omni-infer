#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Functionality:
1. Parse server_profiles.yml file and analyze host_ip and ansible_host matching relationships for P and D group nodes
2. Generate omni_cli commands to modify configuration based on provided node IP mapping
3. Quick query mode: return node count in format like "4p4d"
4. Generate and execute shell scripts

Usage:
    python analyze_nodes.py <yaml_file> [options]

Examples:
    # Quick query for node count (returns string like "4p4d")
    python analyze_nodes.py server_profiles.yml --quick

    # Analyze YAML file only
    python analyze_nodes.py server_profiles.yml

    # Specify node IP mapping
    python analyze_nodes.py server_profiles.yml --node-ips "p0:7.242.106.30;p1:7.242.106.31;d0:7.242.106.50"

    # Generate and execute shell script
    python analyze_nodes.py server_profiles.yml \\
        --node-ips "p0:7.242.106.30;p1:7.242.106.31;p2:7.242.106.32;p3:7.242.106.33;d0:7.242.106.50;d1:7.242.106.51;d2:7.242.106.52;d3:7.242.106.53" \\
        --model-path /data/models/GLM-5 \\
        --service-name glm5 \\
        --log-path /tmp/logs \\
        --nic-name eth0 \\
        --global-rank-table /tmp/ranktable/global_ranktable.json \\
        --rank-table-d /tmp/ranktable/d_ranktable.json \\
        --rank-table-p /tmp/ranktable/p_ranktable.json \\
        --output /tmp/omni_cli_config.sh \\
        --run

Parameter Description:
    yaml_file: YAML configuration file path
    --quick, -q: Quick query mode, returns node count like "4p4d"
    --node-ips: Node IP mapping, format like "p0:ip1;p1:ip2;d0:ip10"
    --model-path: Model path for setting MODEL_PATH
    --service-name: Service name for setting served-model-name
    --log-path: Log path for setting LOG_PATH
    --nic-name: Network interface name for setting GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME/SOCKET_IFNAME
    --global-rank-table: Global rank table file path for setting GLOBAL_RANK_TABLE_FILE_PATH
    --rank-table-d: D group rank table file path for setting D group's RANK_TABLE_FILE_PATH
    --rank-table-p: P group rank table file path for setting P group's RANK_TABLE_FILE_PATH
    --output, -o: Output shell script path
    --run, -r: Execute script immediately after generation
"""

import sys
import os
import argparse
import yaml
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NodeInfo:
    """Node information data class"""
    name: str
    group: str
    ansible_host: Optional[str] = None
    host_ip: Optional[str] = None
    p_node_list: Optional[str] = None  # P_NODE_LIST parameter


@dataclass
class GroupAnalysis:
    """Group analysis result data class"""
    group_name: str
    node_count: int = 0
    nodes: Dict[str, NodeInfo] = field(default_factory=dict)
    # node_name -> list of matched nodes (nodes where ansible_host equals this node's host_ip)
    host_ip_matches: Dict[str, List[str]] = field(default_factory=dict)


class ServerProfileAnalyzer:
    """Server Profile Analyzer"""

    def __init__(self, yaml_file: str):
        self.yaml_file = yaml_file
        self.groups: Dict[str, GroupAnalysis] = {}

    def parse_yaml(self) -> None:
        """Parse YAML file using PyYAML"""
        with open(self.yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse structure: all -> children -> [P, D, C] -> hosts -> [node_name] -> attributes
        if not data or 'all' not in data:
            print("Error: Invalid YAML file format, missing 'all' root node")
            return

        children = data['all'].get('children', {})

        for group_name, group_data in children.items():
            # Create group analysis object
            self.groups[group_name] = GroupAnalysis(group_name=group_name)

            hosts = group_data.get('hosts', {})
            for node_name, node_attrs in hosts.items():
                # Get P_NODE_LIST from env
                env = node_attrs.get('env', {})
                p_node_list = env.get('P_NODE_LIST') if env else None

                # Create node information
                node_info = NodeInfo(
                    name=node_name,
                    group=group_name,
                    ansible_host=node_attrs.get('ansible_host'),
                    host_ip=node_attrs.get('host_ip'),
                    p_node_list=p_node_list
                )
                self.groups[group_name].nodes[node_name] = node_info
                self.groups[group_name].node_count += 1

    def analyze_matches(self) -> None:
        """Analyze host_ip and ansible_host matching relationships"""
        for group_name, group_data in self.groups.items():
            # Find nodes with ansible_host matching each node's host_ip
            for node_name, node_info in group_data.nodes.items():
                if node_info.host_ip:
                    matched_nodes = []
                    for other_name, other_info in group_data.nodes.items():
                        if other_info.ansible_host == node_info.host_ip:
                            matched_nodes.append(other_name)
                    group_data.host_ip_matches[node_name] = matched_nodes

    def print_analysis(self) -> None:
        """Print analysis results"""
        print("=" * 60)
        print("           Node Analysis Results")
        print("=" * 60)

        for group_name in ['P', 'D']:
            if group_name not in self.groups:
                continue

            group_data = self.groups[group_name]
            print(f"\n[{group_name} Group Node Analysis]")
            print("-" * 50)
            print(f"{group_name} group has {group_data.node_count} nodes\n")

            # Output node details
            print("Node Details:")
            for node_name, node_info in sorted(group_data.nodes.items()):
                print(f"  - {node_name}:")
                print(f"      ansible_host: {node_info.ansible_host}")
                print(f"      host_ip: {node_info.host_ip}")

            # Output matching relationships
            print(f"\nhost_ip and ansible_host Matching Relationships:")
            for node_name, node_info in sorted(group_data.nodes.items()):
                matched = group_data.host_ip_matches.get(node_name, [])
                if matched:
                    print(f"  - {node_name}'s host_ip ({node_info.host_ip}) matches the following nodes' ansible_host:")
                    for m in matched:
                        m_info = group_data.nodes[m]
                        print(f"      * {m} (ansible_host: {m_info.ansible_host})")
                else:
                    print(f"  - {node_name}'s host_ip ({node_info.host_ip}) has no matching ansible_host")

        # Summary statistics
        print("\n" + "=" * 60)
        print("           Summary Statistics")
        print("=" * 60)

        for group_name in ['P', 'D']:
            if group_name not in self.groups:
                continue
            group_data = self.groups[group_name]
            print(f"\n[{group_name} Group host_ip Distribution Statistics]")
            ip_count = defaultdict(int)
            for node_info in group_data.nodes.values():
                if node_info.host_ip:
                    ip_count[node_info.host_ip] += 1
            for ip, count in sorted(ip_count.items()):
                print(f"  {ip}: {count} nodes")

        # P_NODE_LIST analysis
        print("\n" + "=" * 60)
        print("           P_NODE_LIST Analysis")
        print("=" * 60)
        for group_name in ['P', 'D']:
            if group_name not in self.groups:
                continue
            group_data = self.groups[group_name]
            print(f"\n[{group_name} Group P_NODE_LIST Configuration]")
            for node_name, node_info in sorted(group_data.nodes.items()):
                if node_info.p_node_list:
                    # Parse IPs in P_NODE_LIST and find corresponding nodes
                    p_ips = node_info.p_node_list.split(',')
                    matched_p_nodes = []
                    if 'P' in self.groups:
                        for ip in p_ips:
                            ip = ip.strip()
                            for p_name, p_info in self.groups['P'].nodes.items():
                                if p_info.ansible_host == ip:
                                    matched_p_nodes.append(p_name)
                                    break
                    print(f"  - {node_name}:")
                    print(f"      P_NODE_LIST: {node_info.p_node_list}")
                    print(f"      Corresponding P nodes: {', '.join(matched_p_nodes) if matched_p_nodes else 'No match'}")

    def get_result_dict(self) -> dict:
        """Get structured result dictionary"""
        result = {}
        for group_name, group_data in self.groups.items():
            result[group_name] = {
                'node_count': group_data.node_count,
                'nodes': {
                    name: {
                        'ansible_host': info.ansible_host,
                        'host_ip': info.host_ip,
                        'p_node_list': info.p_node_list
                    }
                    for name, info in sorted(group_data.nodes.items())
                },
                'host_ip_matches': group_data.host_ip_matches
            }
        return result

    def get_node_count_str(self) -> str:
        """
        Get node count string in format like "4p4d"

        Returns:
            Node count string, e.g. "4p4d"
        """
        p_count = self.groups.get('P', GroupAnalysis('P')).node_count
        d_count = self.groups.get('D', GroupAnalysis('D')).node_count
        return f"{p_count}p{d_count}d"

    def print_quick_info(self) -> None:
        """Quick print of node count information (for external program calls)"""
        print(self.get_node_count_str())

    def validate_node_coverage(self, node_ip_mapping: str) -> tuple:
        """
        Validate whether all nodes (except C group) have corresponding IPs provided

        Args:
            node_ip_mapping: Format like "p0:ip1;p1:ip2;d0:ip10;d1:ip11"

        Returns:
            (is_valid, missing_nodes): Whether valid, list of missing nodes
        """
        # Parse mapping string
        mapping = self.parse_node_ip_mapping(node_ip_mapping)

        # Collect all nodes that need to be checked (except C group)
        required_nodes = set()
        for group_name, group_data in self.groups.items():
            if group_name.upper() != 'C':  # Exclude C group
                required_nodes.update(group_data.nodes.keys())

        # Check for missing nodes
        provided_nodes = set(mapping.keys())
        missing_nodes = required_nodes - provided_nodes

        return len(missing_nodes) == 0, sorted(missing_nodes)

    def parse_node_ip_mapping(self, node_ip_mapping: str) -> Dict[str, str]:
        """
        Parse node IP mapping string

        Args:
            node_ip_mapping: Format like "p0:ip1;p1:ip2;d0:ip10"

        Returns:
            Dictionary mapping node names to IPs
        """
        mapping = {}
        for item in node_ip_mapping.split(';'):
            item = item.strip()
            if ':' in item:
                node_name, ip = item.split(':', 1)
                mapping[node_name.strip()] = ip.strip()
        return mapping

    def generate_global_parameter_commands(self, model_path: str = None,
                                          service_name: str = None,
                                          log_path: str = None,
                                          nic_name: str = None,
                                          global_rank_table: str = None) -> List[str]:
        """
        Generate global parameter commands (applied to all nodes)

        Args:
            model_path: Model path
            service_name: Service name
            log_path: Log path
            nic_name: Network interface name
            global_rank_table: Global rank table file path

        Returns:
            List of global parameter commands
        """
        commands = []

        if model_path or service_name or log_path or nic_name or global_rank_table:
            commands.append("# === Global Parameter Settings (all nodes) ===")
            if model_path:
                commands.append(f"omni_cli cfg --set all env --MODEL_PATH {model_path}")
            if service_name:
                commands.append(f"omni_cli cfg --set all args --served-model-name {service_name}")
            if log_path:
                commands.append(f"omni_cli cfg --set all env --LOG_PATH {log_path}")
            if nic_name:
                commands.append(f"omni_cli cfg --set all env --GLOO_SOCKET_IFNAME {nic_name}")
                commands.append(f"omni_cli cfg --set all env --TP_SOCKET_IFNAME {nic_name}")
                commands.append(f"omni_cli cfg --set all env --SOCKET_IFNAME {nic_name}")
            if global_rank_table:
                commands.append(f"omni_cli cfg --set all env --GLOBAL_RANK_TABLE_FILE_PATH {global_rank_table}")

        return commands

    def generate_group_parameter_commands(self, rank_table_d: str = None,
                                         rank_table_p: str = None) -> List[str]:
        """
        Generate group parameter commands

        Args:
            rank_table_d: D group rank table file path
            rank_table_p: P group rank table file path

        Returns:
            List of group parameter commands
        """
        commands = []

        if rank_table_d:
            commands.append("# === D Group Parameter Settings ===")
            commands.append(f"omni_cli cfg --set d env --RANK_TABLE_FILE_PATH {rank_table_d}")

        if rank_table_p:
            commands.append("# === P Group Parameter Settings ===")
            commands.append(f"omni_cli cfg --set p env --RANK_TABLE_FILE_PATH {rank_table_p}")

        return commands

    def collect_node_ip_lists(self, node_ip_mapping: str) -> tuple:
        """
        Collect IP lists for D and P nodes

        Args:
            node_ip_mapping: Format like "p0:ip1;p1:ip2;d0:ip10"

        Returns:
            (server_ip_list, p_node_list): D nodes' IP list, P nodes' IP list
        """
        mapping = self.parse_node_ip_mapping(node_ip_mapping)

        # Collect all D nodes' new IP list
        d_ip_list = []
        if 'D' in self.groups:
            for node_name in sorted(self.groups['D'].nodes.keys()):
                if node_name in mapping:
                    d_ip_list.append(mapping[node_name])
        server_ip_list = ','.join(d_ip_list)

        # Collect all P nodes' new IP list
        p_ip_list = []
        if 'P' in self.groups:
            for node_name in sorted(self.groups['P'].nodes.keys()):
                if node_name in mapping:
                    p_ip_list.append(mapping[node_name])
        p_node_list = ','.join(p_ip_list)

        return server_ip_list, p_node_list

    def generate_node_specific_commands(self, node_ip_mapping: str, proxy_port: str = None) -> List[str]:
        """
        Generate node-specific configuration commands

        Args:
            node_ip_mapping: Format like "p0:ip1;p1:ip2;d0:ip10"
            proxy_port: Proxy port for C node API_PORT setting

        Returns:
            List of node-specific commands
        """
        commands = []

        if not node_ip_mapping:
            return commands

        mapping = self.parse_node_ip_mapping(node_ip_mapping)
        server_ip_list, p_node_list = self.collect_node_ip_lists(node_ip_mapping)

        # Build original ansible_host -> node_name mapping (for finding nodes corresponding to host_ip)
        ansible_host_to_node = {}
        for group_data in self.groups.values():
            for node_name, node_info in group_data.nodes.items():
                if node_info.ansible_host:
                    ansible_host_to_node[node_info.ansible_host] = node_name

        # Generate commands
        for node_name, new_ansible_ip in mapping.items():
            # Find the group the node belongs to
            node_info = None
            for group_data in self.groups.values():
                if node_name in group_data.nodes:
                    node_info = group_data.nodes[node_name]
                    break

            if node_info is None:
                print(f"Warning: Node {node_name} does not exist in configuration, skipping")
                continue

            # Find the original node name corresponding to host_ip
            original_host_ip = node_info.host_ip
            target_node = ansible_host_to_node.get(original_host_ip)

            # Determine the new IP to use for host_ip and HOST_IP
            if target_node and target_node in mapping:
                new_host_ip = mapping[target_node]
            else:
                # If corresponding node not found or that node has no IP provided, use own new IP
                new_host_ip = new_ansible_ip

            # Generate commands to modify node
            commands.append(f"# === Modify Node {node_name} ===")
            commands.append(f"omni_cli cfg --set {node_name} --ansible_host {new_ansible_ip}")
            commands.append(f"omni_cli cfg --set {node_name} env --HOST_IP {new_host_ip}")
            if node_info.group == 'P':
                commands.append(f"omni_cli cfg --set {node_name} env --LOCAL_HOST_IP {new_ansible_ip}")
            commands.append(f"omni_cli cfg --set {node_name} --host_ip {new_host_ip}")
            commands.append(f"omni_cli cfg --set {node_name} env --SERVER_IP_LIST {server_ip_list}")
            commands.append(f"omni_cli cfg --set {node_name} env --P_NODE_LIST {p_node_list}")

        if proxy_port and 'C' in self.groups:
            for node_name, node_info in self.groups['C'].nodes.items():
                commands.append(f"# === Modify Node {node_name} ===")
                commands.append(f"omni_cli cfg --set {node_name} env --API_PORT {proxy_port}")

        return commands

    def generate_startup_command(self) -> List[str]:
        """
        Generate service startup command

        Returns:
            List containing startup command
        """
        return [
            "# === Start Service ===",
            "omni_cli start --skip-verify-config --run-dev --cloud-mode"
        ]

    def generate_omni_cli_commands(self, node_ip_mapping: str = None,
                                    model_path: str = None,
                                    service_name: str = None,
                                    log_path: str = None,
                                    nic_name: str = None,
                                    global_rank_table: str = None,
                                    rank_table_d: str = None,
                                    rank_table_p: str = None,
                                    proxy_port: str = None) -> List[str]:
        """
        Generate omni_cli commands based on node IP mapping

        For each input node:
        1. ansible_host is changed to the provided IP
        2. host_ip and HOST_IP are changed to the new IP corresponding to the original node's ansible_host
        3. SERVER_IP_LIST is set to the IP list of all D nodes
        4. P_NODE_LIST is set to the IP list of all P nodes

        Global parameters (applied to all nodes):
        - MODEL_PATH: Model path
        - served-model-name: Service name
        - LOG_PATH: Log path
        - GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME/SOCKET_IFNAME: Network interface name
        - GLOBAL_RANK_TABLE_FILE_PATH: Global rank table file path

        Group parameters:
        - D group RANK_TABLE_FILE_PATH: D group rank table file path
        - P group RANK_TABLE_FILE_PATH: P group rank table file path

        Args:
            node_ip_mapping: Format like "p0:ip1;p1:ip2;d0:ip10;d1:ip11"
            model_path: Model path
            service_name: Service name
            log_path: Log path
            nic_name: Network interface name
            global_rank_table: Global rank table file path
            rank_table_d: D group rank table file path
            rank_table_p: P group rank table file path
            proxy_port: Proxy port for C node API_PORT setting

        Returns:
            List of omni_cli commands
        """
        commands = []

        # Generate global parameter commands
        commands.extend(self.generate_global_parameter_commands(
            model_path, service_name, log_path, nic_name, global_rank_table
        ))

        # Generate group parameter commands
        commands.extend(self.generate_group_parameter_commands(rank_table_d, rank_table_p))

        # Generate node-specific commands
        commands.extend(self.generate_node_specific_commands(node_ip_mapping, proxy_port))

        # Add startup command
        commands.extend(self.generate_startup_command())

        return commands

    def print_command_parameters(self, node_ip_mapping: str = None,
                                 model_path: str = None,
                                 service_name: str = None,
                                 log_path: str = None,
                                 nic_name: str = None,
                                 global_rank_table: str = None,
                                 rank_table_d: str = None,
                                 rank_table_p: str = None,
                                 proxy_port: str = None) -> None:
        """Print input parameters"""
        if node_ip_mapping:
            print(f"Node IP Mapping: {node_ip_mapping}")
        if model_path:
            print(f"Model Path: {model_path}")
        if service_name:
            print(f"Service Name: {service_name}")
        if log_path:
            print(f"Log Path: {log_path}")
        if nic_name:
            print(f"Network Interface Name: {nic_name}")
        if global_rank_table:
            print(f"Global Rank Table: {global_rank_table}")
        if rank_table_d:
            print(f"D Group Rank Table: {rank_table_d}")
        if rank_table_p:
            print(f"P Group Rank Table: {rank_table_p}")
        if proxy_port:
            print(f"Proxy Port: {proxy_port}")

    def print_coverage_validation(self, node_ip_mapping: str = None) -> None:
        """Print node coverage validation results"""
        if node_ip_mapping:
            is_valid, missing_nodes = self.validate_node_coverage(node_ip_mapping)

            if not is_valid:
                print("\n" + "!" * 60)
                print("Warning: The following nodes have no IP mapping (except C group nodes):")
                for node in missing_nodes:
                    print(f"  - {node}")
                print("!" * 60)
                print("\nPlease ensure all P and D group nodes have corresponding IP mappings.")
                print("Continuing to generate commands, but configuration may be incomplete...\n")
            else:
                print("\nNode Coverage Check: Passed (All P and D group nodes have IPs provided)")

    def print_command_explanation(self) -> None:
        """Print command explanation"""
        print("\nExplanation:")
        print("  - ansible_host: Changed to provided IP")
        print("  - host_ip/HOST_IP: Changed to new IP of the originally corresponding node (ansible_host equals current host_ip)")
        print("  - SERVER_IP_LIST: IP list of all D nodes")
        print("  - P_NODE_LIST: IP list of all P nodes")
        print("  - MODEL_PATH: Model path (global)")
        print("  - served-model-name: Service name (global)")
        print("  - LOG_PATH: Log path (global)")
        print("  - GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME/SOCKET_IFNAME: Network interface name (global)")
        print("  - GLOBAL_RANK_TABLE_FILE_PATH: Global rank table file path (global)")
        print("  - RANK_TABLE_FILE_PATH: D/P group rank table file path (group level)")
        print("  - Startup command: omni_cli start --skip-verify-config --run-dev --cloud-mode")
        print()

    def print_commands(self, commands: List[str]) -> None:
        """
        Print generated commands

        Args:
            commands: List of commands to print
        """
        if commands:
            for cmd in commands:
                print(cmd)
            # Filter out comment lines to count actual commands
            actual_commands = [cmd for cmd in commands if not cmd.startswith('#')]
            print(f"\nTotal {len(actual_commands)} commands generated")
        else:
            print("No commands generated, please check input parameters")

    def generate_shell_script(self, commands: List[str]) -> str:
        """
        Generate shell script content

        Args:
            commands: List of commands

        Returns:
            Shell script content
        """
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated omni_cli configuration script",
            "# Generated by analyze_nodes.py",
            "",
            "set -e  # Exit on error",
            "",
            "echo '===== Starting omni_cli configuration ====='",
            ""
        ]

        for cmd in commands:
            if cmd.startswith('#'):
                script_lines.append(f"echo '{cmd}'")
            else:
                script_lines.append(f"echo 'Executing: {cmd}'")
                script_lines.append(cmd)
            script_lines.append("")

        script_lines.extend([
            "echo '===== omni_cli configuration completed ====='",
            ""
        ])

        return '\n'.join(script_lines)

    def save_shell_script(self, script_content: str, output_path: str) -> None:
        """
        Save shell script to file

        Args:
            script_content: Shell script content
            output_path: Output file path
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Add execute permission
        os.chmod(output_path, 0o755)

    def run_shell_script(self, script_path: str) -> int:
        """
        Execute shell script

        Args:
            script_path: Shell script file path

        Returns:
            Script execution return code
        """
        try:
            result = subprocess.run(
                ['bash', script_path],
                capture_output=False,
                text=True
            )
            if result.returncode == 0:
                print("\nScript executed successfully")
            else:
                print(f"\nScript execution failed, exit code: {result.returncode}")
            return result.returncode
        except Exception as e:
            print(f"\nError occurred while executing script: {e}")
            return 1

    def create_temp_script(self, script_content: str) -> str:
        """
        Create temporary script file

        Args:
            script_content: Shell script content

        Returns:
            Temporary script file path
        """
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            return f.name

    def process_script_generation_and_execution(self, commands: List[str],
                                                 output_script: str = None,
                                                 run_script: bool = False) -> None:
        """
        Process script generation and execution

        Args:
            commands: List of commands
            output_script: Output script path
            run_script: Whether to execute the script
        """
        if not output_script and not run_script:
            return

        script_content = self.generate_shell_script(commands)

        if output_script:
            self.save_shell_script(script_content, output_script)
            print(f"\nShell script saved to: {output_script}")

        if run_script:
            script_path = output_script if output_script else self.create_temp_script(script_content)

            print(f"\nExecuting script: {script_path}")
            self.run_shell_script(script_path)


def print_command_header() -> None:
    """Print command generation header"""
    print("\n" + "=" * 60)
    print("        Generated omni_cli Commands")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Parse server_profiles.yml file and generate omni_cli commands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick query for node count (returns string like "4p4d")
    python analyze_nodes.py server_profiles.yml --quick

    # Analyze YAML file only
    python analyze_nodes.py server_profiles.yml

    # Specify node IP mapping
    python analyze_nodes.py server_profiles.yml --node-ips "p0:7.242.106.30;p1:7.242.106.31;d0:7.242.106.50"

    # Generate shell script
    python analyze_nodes.py server_profiles.yml \\
        --node-ips "p0:ip1;p1:ip2;d0:ip3" \\
        --output /tmp/omni_cli_config.sh

    # Generate and execute shell script
    python analyze_nodes.py server_profiles.yml \\
        --node-ips "p0:ip1;p1:ip2;d0:ip3" \\
        --output /tmp/omni_cli_config.sh \\
        --run

    # Full parameter example
    python analyze_nodes.py server_profiles.yml \\
        --node-ips "p0:7.242.106.30;p1:7.242.106.31;p2:7.242.106.32;p3:7.242.106.33;d0:7.242.106.50;d1:7.242.106.51;d2:7.242.106.52;d3:7.242.106.53" \\
        --model-path /data/models/GLM-5 \\
        --service-name glm5 \\
        --log-path /tmp/logs \\
        --nic-name eth0 \\
        --global-rank-table /tmp/ranktable/global_ranktable.json \\
        --rank-table-d /tmp/ranktable/d_ranktable.json \\
        --rank-table-p /tmp/ranktable/p_ranktable.json \\
        --output /tmp/omni_cli_config.sh \\
        --run
        """
    )
    parser.add_argument('yaml_file', help='YAML configuration file path')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick query mode, returns node count like "4p4d"')
    parser.add_argument('--node-ips', '-n', help='Node IP mapping, format like "p0:ip1;p1:ip2;d0:ip10"')
    parser.add_argument('--model-path', '-m', help='Model path for setting MODEL_PATH')
    parser.add_argument('--service-name', '-s', help='Service name for setting served-model-name')
    parser.add_argument('--log-path', '-l', help='Log path for setting LOG_PATH')
    parser.add_argument('--nic-name', '-i', help='Network interface name for setting SOCKET_IFNAME related parameters')
    parser.add_argument('--global-rank-table', '-g', help='Global rank table file path for setting GLOBAL_RANK_TABLE_FILE_PATH')
    parser.add_argument('--rank-table-d', '-d', help='D group rank table file path for setting D group RANK_TABLE_FILE_PATH')
    parser.add_argument('--rank-table-p', '-p', help='P group rank table file path for setting P group RANK_TABLE_FILE_PATH')
    parser.add_argument('--proxy-port', help='Proxy port for setting C node API_PORT')
    parser.add_argument('--output', '-o', help='Output shell script path')
    parser.add_argument('--run', '-r', action='store_true',
                        help='Execute script immediately after generation')

    args = parser.parse_args()

    # Create analyzer and perform analysis
    analyzer = ServerProfileAnalyzer(args.yaml_file)
    analyzer.parse_yaml()

    # Quick query mode: only return node count
    if args.quick:
        analyzer.print_quick_info()
        return

    analyzer.analyze_matches()

    # Print analysis results
    analyzer.print_analysis()

    # Print structured data
    print("\n" + "=" * 60)
    print("        Structured Data (JSON Format)")
    print("=" * 60)
    import json
    print(json.dumps(analyzer.get_result_dict(), indent=2, ensure_ascii=False))

    # Generate omni_cli commands
    has_params = (args.node_ips or args.model_path or args.service_name or
                  args.log_path or args.nic_name or args.global_rank_table or
                  args.rank_table_d or args.rank_table_p or args.proxy_port)
    if has_params:
        print_command_header()

        # Print input parameters
        analyzer.print_command_parameters(
            node_ip_mapping=args.node_ips,
            model_path=args.model_path,
            service_name=args.service_name,
            log_path=args.log_path,
            nic_name=args.nic_name,
            global_rank_table=args.global_rank_table,
            rank_table_d=args.rank_table_d,
            rank_table_p=args.rank_table_p,
            proxy_port=args.proxy_port
        )

        # Print coverage validation
        analyzer.print_coverage_validation(args.node_ips)

        # Print command explanation
        analyzer.print_command_explanation()

        # Generate commands
        commands = analyzer.generate_omni_cli_commands(
            node_ip_mapping=args.node_ips,
            model_path=args.model_path,
            service_name=args.service_name,
            log_path=args.log_path,
            nic_name=args.nic_name,
            global_rank_table=args.global_rank_table,
            rank_table_d=args.rank_table_d,
            rank_table_p=args.rank_table_p,
            proxy_port=args.proxy_port
        )

        # Print commands
        # analyzer.print_commands(commands)

        # Process script generation and execution
        analyzer.process_script_generation_and_execution(
            commands, args.output, args.run
        )

        # Return validation result
        if args.node_ips:
            is_valid, _ = analyzer.validate_node_coverage(args.node_ips)
            if not is_valid:
                sys.exit(1)  # Incomplete node coverage, return non-zero exit code


if __name__ == '__main__':
    main()