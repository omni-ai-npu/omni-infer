# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from functools import partial

import torch
import torch_npu

from vllm.logger import init_logger

from omni_npu.compilation.acl_graph import (
    GraphTaskEntry,
    OpDescriptor,
    get_graph_params,
    weak_ref_tensors,
)

logger = init_logger(__name__)
try:
    import omni_custom_ops
except ImportError as e:
    logger.warning(f"Failed to import omni_custom_ops: {e}")
except Exception as e:
    logger.warning(f"Error occurred while importing omni_custom_ops: {e}")


def _pad_list(lst, n, fill=None):
    """Pad or truncate a list to length n.

    When padding, ``fill`` is used for the new elements.
    If ``fill`` is None, the last element of ``lst`` is repeated.
    """
    if not lst or len(lst) == n:
        return copy.deepcopy(lst)
    if len(lst) > n:
        return lst[:n]
    if fill is None:
        fill = lst[-1]
    return lst + [fill] * (n - len(lst))


def _extract_fia_params(metadata, vllm_config, batch_descriptor):
    """Extract and pad sequence lengths from attention metadata."""

    seq_qlen, seq_kvlen = None, None
    if not hasattr(metadata, "decode") and hasattr(metadata, "query_start_loc"):
        # GQA case: use query_start_loc for qlen and seq_lens for kvlen
        seq_qlen = metadata.query_start_loc[1:]
        seq_kvlen = metadata.seq_lens
    else:
        # MLA/DSA case: use query_cumlens for qlen and seq_lens for kvlen
        seq_qlen = metadata.decode.query_cumlens
        seq_kvlen = metadata.decode.seq_lens

    if isinstance(seq_kvlen, torch.Tensor):
        return seq_qlen, seq_kvlen

    padding_lens = batch_descriptor.num_reqs
    if padding_lens is None:
        padding_lens = min(
            vllm_config.scheduler_config.max_num_seqs,
            batch_descriptor.num_tokens,
        )
    seq_qlen[-1] = batch_descriptor.num_tokens
    seq_qlen = _pad_list(seq_qlen, padding_lens)
    seq_kvlen = _pad_list(seq_kvlen, padding_lens, 0)
    return seq_qlen, seq_kvlen


def _compute_fia_dynamic_kwargs(
    forward_context, layer_name, vllm_config, *, seq_len_q_key, seq_len_kv_key,
):
    """Compute dynamic seq-len kwargs for FIA attention ops.

    Args:
        forward_context: current forward context
        layer_name: attention layer name
        vllm_config: VllmConfig for scheduler params
        seq_len_q_key: kwarg name for the query sequence lengths
        seq_len_kv_key: kwarg name for the kv sequence lengths
    Returns:
        dict of updated kwargs, or None to skip
    """
    if layer_name not in forward_context.attn_metadata:
        return None

    metadata = forward_context.attn_metadata[layer_name]
    seq_len_q, seq_len_kv = _extract_fia_params(
        metadata, vllm_config, forward_context.batch_descriptor,
    )

    return {
        seq_len_q_key: seq_len_q,
        seq_len_kv_key: seq_len_kv,
    }

_WEAK_REF_KEYS = (
    "query", "key", "value", "query_rope", "key_rope",
    "block_table", "atten_mask",
    "key_sink", "value_sink", "key_rope_sink",
)

DUMMY_OP_DESCRIPTOR = OpDescriptor(
    op_out_fn=lambda **kwargs: None,
    workspace_fn=lambda **kwargs: None,
    weak_ref_keys=set(),
    compute_dynamic_kwargs=lambda **kwargs: None,
)

try:
    OP_FIA_V1 = OpDescriptor(
        op_out_fn=torch_npu.npu_fused_infer_attention_score.out,
        workspace_fn=torch_npu._npu_fused_infer_attention_score_get_max_workspace,
        weak_ref_keys=_WEAK_REF_KEYS,
        compute_dynamic_kwargs=partial(
            _compute_fia_dynamic_kwargs,
            seq_len_q_key="actual_seq_lengths",
            seq_len_kv_key="actual_seq_lengths_kv",
        ),
    )
except Exception as e:
    OP_FIA_V1 = DUMMY_OP_DESCRIPTOR
    logger.warning(f"Failed to create OP_FIA_V1 descriptor: {e}")

try:
    OP_FIA_V2 = OpDescriptor(
        op_out_fn=torch_npu.npu_fused_infer_attention_score_v2.out,
        workspace_fn=torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace,
        weak_ref_keys=_WEAK_REF_KEYS,
        compute_dynamic_kwargs=partial(
            _compute_fia_dynamic_kwargs,
            seq_len_q_key="actual_seq_qlen",
            seq_len_kv_key="actual_seq_kvlen",
        ),
    )
except Exception as e:
    OP_FIA_V2 = DUMMY_OP_DESCRIPTOR
    logger.warning(f"Failed to create OP_FIA_V2 descriptor: {e}")

try:
    OP_FIA_SINK = OpDescriptor(
        op_out_fn=torch.ops.custom.npu_fused_infer_attention_sink.out,
        workspace_fn=torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace,
        weak_ref_keys=_WEAK_REF_KEYS,
        compute_dynamic_kwargs=partial(
            _compute_fia_dynamic_kwargs,
            seq_len_q_key="actual_seq_qlen",
            seq_len_kv_key="actual_seq_kvlen",
        ),
    )
except Exception as e:
    OP_FIA_SINK = DUMMY_OP_DESCRIPTOR
    logger.warning(f"Failed to create OP_FIA_SINK descriptor: {e}")

try:
    OP_FIA_PIONEER = OpDescriptor(
        op_out_fn=torch_npu._npu_attention_pioneer.out,
        workspace_fn=torch_npu._npu_attention_pioneer_get_max_workspace,
        weak_ref_keys=_WEAK_REF_KEYS,
        compute_dynamic_kwargs=partial(
            _compute_fia_dynamic_kwargs,
            seq_len_q_key="actual_seq_lengths",
            seq_len_kv_key="actual_seq_lengths_kv",
        ),
    )
except Exception as e:
    OP_FIA_PIONEER = DUMMY_OP_DESCRIPTOR
    logger.warning(f"Failed to create OP_FIA_PIONEER descriptor: {e}")


def _get_or_create_workspace(
    op_desc: OpDescriptor,
    op_kwargs: dict,
    num_tokens: int,
) -> torch.Tensor:
    """Return the cached workspace for (num_tokens, workspace_fn), or create one."""
    graph_params = get_graph_params()
    workspace_fn = op_desc.workspace_fn
    workspace = graph_params.workspaces.get(num_tokens, {}).get(workspace_fn)
    if workspace is None:
        workspace = workspace_fn(
            **op_kwargs,
        )
        if num_tokens not in graph_params.workspaces:
            graph_params.workspaces[num_tokens] = {}
        graph_params.workspaces[num_tokens][workspace_fn] = weak_ref_tensors(workspace)
    return workspace


def _capture_kwargs(
    op_desc: OpDescriptor,
    op_kwargs: dict,
) -> dict:
    """Snapshot kwargs, applying weak refs to keys listed in op_desc."""
    captured_kwargs = op_kwargs.copy()
    for key in op_desc.weak_ref_keys:
        if key in captured_kwargs and captured_kwargs[key] is not None:
            captured_kwargs[key] = weak_ref_tensors(captured_kwargs[key])
    return captured_kwargs


def capture_graph_task(
    op_desc: OpDescriptor,
    op_kwargs: dict,
    out_tensors: list[torch.Tensor],
    num_tokens: int,
    layer_name: str,
) -> None:
    """Generic graph task capture.

    Records an op into a task group and stores the handle, event, and
    full kwargs snapshot for later update/replay.
    """
    graph_params = get_graph_params()
    stream = torch_npu.npu.current_stream()

    # Event setup — recorded into the graph for replay synchronization
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)

    # Workspace setup — also captured into the graph,
    # but cached and shared across replays of the same op/layer and num_tokens 
    # to save memory and redundant workspace computation
    workspace = _get_or_create_workspace(op_desc, op_kwargs, num_tokens)

    # Snapshot kwargs, applying weak refs to specified keys 
    captured_kwargs = _capture_kwargs(op_desc, op_kwargs)

    # Capture the task group
    torch.npu.graph_task_group_begin(stream)
    op_desc.op_out_fn(
        **op_kwargs,
        workspace=workspace,
        out=out_tensors,
    )
    handle = torch.npu.graph_task_group_end(stream)

    # Store entry
    entry = GraphTaskEntry(
        op_desc=op_desc,
        captured_kwargs=captured_kwargs,
        out_tensors=[weak_ref_tensors(t) for t in out_tensors],
        handle=handle,
        event=event,
    )
    graph_params.task_entries[num_tokens][layer_name] = entry
