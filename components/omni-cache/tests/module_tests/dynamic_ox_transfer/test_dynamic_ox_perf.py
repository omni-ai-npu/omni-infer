#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""OX performance test for the dynamic-topology request interface.

The validated workload and ZMQ flow remain in ``test_ox_perf.py``. This file
only replaces the two topology-specific operations:

* D-side OX starts without ``--shard-list``;
* every request carries ``remote_ox_shard_list``.
"""

import os
import random
import sys
import time

import msgpack

if __package__:
    from . import test_ox_perf as perf
else:
    import test_ox_perf as perf


_remote_ox_shard_list = None
_remote_ox_clusters = None


def _parse_clusters(shard_list):
    clusters = [cluster.strip() for cluster in shard_list.split(";") if cluster.strip()]
    return clusters or [shard_list]


def _build_dynamic_request(request_id, src_ids, dst_ids, cluster_count):
    if _remote_ox_clusters:
        remote_ox_shard_list = random.choice(_remote_ox_clusters)
    else:
        remote_ox_shard_list = _remote_ox_shard_list
    return msgpack.packb({
        "request_id": request_id,
        "table_id": 0,
        "src_block_ids": src_ids,
        "dst_block_ids": dst_ids,
        "remote_ox_shard_list": remote_ox_shard_list,
        "src_dp_rank": 0,
    })


def _run_dynamic_d_side(args):
    global _remote_ox_shard_list, _remote_ox_clusters
    _remote_ox_shard_list = args.shard_list
    _remote_ox_clusters = _parse_clusters(args.shard_list)

    block_size = perf.compute_block_size(
        args.num_layers,
        args.tokens_per_block,
        args.dims,
        args.dtype,
    )
    tp_size = perf.parse_tp_size(args.shard_list)
    block_tp_size = block_size // tp_size

    print(
        f"[UT] dynamic topology: shards={args.shard_list}, "
        f"clusters={len(_remote_ox_clusters)}, "
        f"tp_size={tp_size}, block_size={block_size} bytes, "
        f"block_tp_size={block_tp_size} bytes"
    )

    shm_size = args.num_block_tables * args.num_blocks * block_size
    perf.create_shm_file(args.block_table_shm, shm_size)

    command = [
        str(args.ox_path),
        "--zmq-port", str(args.zmq_port),
        "--block-table-shm", args.block_table_shm,
        "--num-block-tables", str(args.num_block_tables),
        "--num-blocks", str(args.num_blocks),
        "--num-layers", str(args.num_layers),
        "--tokens-per-block", str(args.tokens_per_block),
        "--num-connections-per-req",
        str(args.num_connections_per_req),
        "--num-connections", str(args.num_connections),
        "--dims", ",".join(map(str, args.dims)),
        "--dtype", str(args.dtype),
        "--num-threads", str(args.num_threads),
    ]

    log_path = os.path.join(args.log_dir, "ox_perf_dynamic_d.log")
    ox = perf.OXProcess(command, log_path)
    print(
        f"[UT] Waiting {args.connect_wait}s for D-side OX startup ..."
    )
    time.sleep(args.connect_wait)

    try:
        if not ox.is_alive():
            print(
                "[UT] ERROR: D-side OX process exited prematurely.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        perf.run_benchmark(args, block_tp_size, tp_size)
    finally:
        ox.stop()


def main():
    perf._build_request = _build_dynamic_request
    perf.run_d_side = _run_dynamic_d_side
    perf.main()


if __name__ == "__main__":
    main()
