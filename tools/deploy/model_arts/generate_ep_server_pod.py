#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Generate UB endpoint configuration files from HCCL topology data."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def find_peer_eid_from_1dmesh(
    local_id: int,
    device_eid_ports: list[str],
    topo_data: dict[str, Any],
    rootinfo: dict[str, Any],
) -> str:
    """Find the EID connected to a local device through a 1D-mesh edge."""
    p2p_edges = [
        edge
        for edge in topo_data.get("edge_list", [])
        if edge.get("topo_type") == "1DMESH"
        and edge.get("link_type") == "PEER2PEER"
        and edge.get("net_layer") == 0
    ]

    for edge in p2p_edges:
        local_a = edge["local_a"]
        local_b = edge["local_b"]
        if local_a == local_id:
            local_ports = edge.get("local_a_ports", [])
            peer_ports = edge.get("local_b_ports", [])
            peer_local_id = local_b
        elif local_b == local_id:
            local_ports = edge.get("local_b_ports", [])
            peer_ports = edge.get("local_a_ports", [])
            peer_local_id = local_a
        else:
            continue

        if not set(device_eid_ports) & set(local_ports):
            continue
        for rank in rootinfo.get("rank_list", []):
            if rank["local_id"] != peer_local_id:
                continue
            for level in rank.get("level_list", []):
                if level.get("net_layer") != 0:
                    continue
                for address in level.get("rank_addr_list", []):
                    if set(peer_ports) & set(address.get("ports", [])):
                        return address["addr"]
    return ""


def get_protocol_from_eid(
    net_layer: int,
    topo_data: dict[str, Any],
    device_eid_ports: list[str],
) -> str:
    """Determine the UB protocol for an EID from its topology layer."""
    if net_layer == 0:
        return "ub_ctp"

    for edge in topo_data.get("edge_list", []):
        if edge.get("topo_type") != "CLOS" or edge.get("net_layer") != net_layer:
            continue
        edge_ports = set(edge.get("local_a_ports", [])) | set(
            edge.get("local_b_ports", [])
        )
        if not edge_ports & set(device_eid_ports):
            continue
        protocols = edge.get("protocols", [])
        if "UB_CTP" in protocols:
            return "ub_ctp"
        if "UB_TP" in protocols:
            return "ub_tp"
    return "ub_ctp"


def generate_endpoint_list(
    local_id: int,
    device_info: dict[str, Any],
    topo_data: dict[str, Any],
    rootinfo: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate endpoint entries for one NPU device."""
    endpoint_list = []
    seen_eids = set()

    for level in device_info["level_list"]:
        net_layer = level["net_layer"]
        for address in level["rank_addr_list"]:
            eid = address["addr"]
            if eid in seen_eids:
                continue
            seen_eids.add(eid)

            ports = address.get("ports", [])
            plane_id = address.get("plane_id", "plane_0")
            protocol = get_protocol_from_eid(net_layer, topo_data, ports)
            endpoint = {
                "protocol": protocol,
                "comm_id": eid,
                "placement": "device",
            }
            if (net_layer >= 1 or protocol == "ub_tp") and plane_id:
                endpoint["plane"] = plane_id
            if net_layer == 0:
                peer_eid = find_peer_eid_from_1dmesh(
                    local_id, ports, topo_data, rootinfo
                )
                if peer_eid:
                    endpoint["dst_eid"] = peer_eid
            endpoint_list.append(endpoint)

    return endpoint_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NPU endpoint configs from HCCL topology and rootinfo."
    )
    parser.add_argument("--local", "-l", action="store_true")
    parser.add_argument("--pod", "-p", action="store_true")
    parser.add_argument("--server", "-s", action="store_true")
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--rootinfo-path")
    parser.add_argument("--topo-path")
    return parser.parse_args()


def resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve production paths or local fixture paths."""
    if not args.local:
        rootinfo_path = Path("/etc/hccl_rootinfo.json")
        with rootinfo_path.open(encoding="utf-8") as file:
            rootinfo = json.load(file)
        return rootinfo_path, Path(rootinfo["topo_file_path"])

    mode = "server" if args.server else "pod"
    rootinfo_path = Path(
        args.rootinfo_path
        or (
            "./server/hccl_rootinfo_08server.json"
            if mode == "server"
            else "./pod/hccl_rootinfo.json"
        )
    )
    topo_path = Path(
        args.topo_path
        or ("./server/atlas_850_1.json" if mode == "server" else "./pod/atlas_950_1.json")
    )
    return rootinfo_path, topo_path


def main() -> int:
    args = parse_args()
    rootinfo_path, topo_path = resolve_input_paths(args)
    print(f"Loading: {rootinfo_path}")
    with rootinfo_path.open(encoding="utf-8") as file:
        rootinfo = json.load(file)
    print(f"Loading topology: {topo_path}")
    with topo_path.open(encoding="utf-8") as file:
        topo_data = json.load(file)

    device_id_to_local_id = {
        rank["device_id"]: rank["local_id"]
        for rank in rootinfo.get("rank_list", [])
    }
    if not device_id_to_local_id:
        print(f"Error: rank_list in {rootinfo_path} is empty", file=sys.stderr)
        return 1

    output_dir = (
        Path("./hixlep")
        if args.local
        else Path(os.getenv("HIXLP_ENDPOINT_PATH", "/etc/hixlep"))
    )
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    ranks_by_device_id = {
        rank["device_id"]: rank for rank in rootinfo.get("rank_list", [])
    }
    for device_id, local_id in sorted(device_id_to_local_id.items()):
        device_info = ranks_by_device_id[device_id]
        endpoint_list = generate_endpoint_list(
            local_id, device_info, topo_data, rootinfo
        )
        if not endpoint_list:
            print(
                f"Error: no endpoint resolved for device_id {device_id} "
                f"(local_id {local_id})",
                file=sys.stderr,
            )
            return 1

        net_instance_id = next(
            (
                level.get("net_instance_id")
                for level in device_info.get("level_list", [])
                if level.get("net_layer") == 1
            ),
            None,
        )
        output = {
            "version": "1.3",
            "net_instance_id": net_instance_id,
            "endpoint_list": endpoint_list,
        }
        output_path = output_dir / f"ub_endpoint_npu_{device_id}.json"
        if args.dry_run:
            print(f"[Dry run] Would generate: {output_path}")
            continue

        temporary_path = output_path.with_name(
            f"{output_path.name}.tmp.{os.getpid()}"
        )
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)
        os.replace(temporary_path, output_path)
        print(f"Generated: {output_path}")

    action = "Would generate" if args.dry_run else "Generated"
    print(f"{action} {len(device_id_to_local_id)} endpoint configuration files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
