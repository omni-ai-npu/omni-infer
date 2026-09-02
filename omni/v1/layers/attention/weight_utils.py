# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from torch import nn


def _plain_layout(param: torch.Tensor) -> torch.Tensor:
    """Materialize the pre-PWAL (out, in) layout of a possibly NZ/transposed
    weight without disturbing the stored tensor (mirrors the veRL reload
    handling in the linear weight_loader)."""
    data = param.data
    if getattr(param, "is_weight_nz", False):
        import torch_npu

        data = torch_npu.npu_format_cast(data, torch_npu.Format.ND)
    if getattr(param, "is_weight_transposed", False):
        data = data.t()
    return data


def _store_post_pwal(param: torch.Tensor, plain: torch.Tensor) -> None:
    """Write a pre-PWAL-layout slice into a weight that may already have been
    transposed/NZ-cast by a previous process_weights_after_loading (second
    load / RL weight sync), re-applying the same layout afterwards."""
    is_weight_nz = getattr(param, "is_weight_nz", False)
    is_weight_transposed = getattr(param, "is_weight_transposed", False)
    if not (is_weight_nz or is_weight_transposed):
        param.data.copy_(plain)
        return
    if is_weight_nz:
        import torch_npu

        param.data = torch_npu.npu_format_cast(
            param.data, torch_npu.Format.ND
        )
    if is_weight_transposed:
        param.data = param.data.t_()
    param.data.copy_(plain)
    if is_weight_transposed:
        param.data = param.data.t_()
    if is_weight_nz:
        import torch_npu

        param.data = torch_npu.npu_format_cast(
            param.data, torch_npu.Format.FRACTAL_NZ
        )
        from omni_npu.compilation.acl_graph import set_aclgraph_recapture

        set_aclgraph_recapture(True)


def release_q_b_proj_storage(q_b_proj: nn.Module) -> None:
    """Release the redundant full q_b_proj weight storage.

    When the split q-up projections (q_b_nope_proj / q_b_pe_proj) carry the
    compute, keeping the full q_b_proj as well doubles the q-up weight HBM
    footprint. Call this once the split projections have been populated
    (after loading). The wrapped split loader re-materializes the storage on
    the next load (RL weight sync), so the release is reload-safe.

    No-op unless the wrapped split loader has recorded the plain-layout
    shape, i.e. unless the weight has actually been loaded.
    """
    weight = getattr(q_b_proj, "weight", None)
    if weight is None or getattr(weight, "_q_b_storage_released", False):
        return
    if not hasattr(weight, "_q_b_plain_shape"):
        # Weights not loaded yet (e.g. the init-time post_weight_load).
        return
    weight._q_b_storage_released = True
    weight.data = torch.empty(0, dtype=weight.dtype, device=weight.device)


def install_q_b_split_loaders(
    q_b_proj: nn.Module,
    q_b_nope_proj: nn.Module,
    q_b_pe_proj: nn.Module,
    qk_head_dim: int,
    qk_nope_head_dim: int,
) -> None:
    """Populate split q-up projections from q_b_proj checkpoint data."""

    def make_split_loader(orig_loader, nope_param, pe_param):
        def split_loader(param, loaded_weight, *args, **kwargs):
            released = getattr(param, "_q_b_storage_released", False)
            if released:
                # Storage was released after the first load to save HBM;
                # re-materialize it so the wrapped loader can populate it
                # again. Treat this as a fresh plain-layout load: PWAL does
                # not re-run for the source projection on reload.
                param.data = torch.empty(
                    param._q_b_plain_shape,
                    dtype=loaded_weight.dtype,
                    device=loaded_weight.device,
                )
                for layout_attr in ("is_weight_nz", "is_weight_transposed"):
                    if getattr(param, layout_attr, False):
                        setattr(param, layout_attr, False)
            loader_result = orig_loader(param, loaded_weight, *args, **kwargs)
            # After a reload the wrapped loader has already re-applied the
            # post-PWAL layout; slice on the restored plain layout instead.
            data = _plain_layout(param)
            # Record the plain (out, in) loader-facing shape so that a later
            # reload can re-materialize the released storage with it.
            param._q_b_plain_shape = tuple(data.shape)
            if data.dim() not in (1, 2):
                raise ValueError(
                    f"Unsupported q_b_proj layout: shape={tuple(data.shape)}"
                )
            out_dim = data.shape[0]
            if out_dim % qk_head_dim != 0:
                raise ValueError(
                    f"q_b_proj output dim {out_dim} is not divisible by "
                    f"{qk_head_dim}"
                )
            local_heads = out_dim // qk_head_dim
            trailing_shape = data.shape[1:]
            expected_nope_shape = (
                local_heads * qk_nope_head_dim,
                *trailing_shape,
            )
            expected_pe_shape = (
                local_heads * (qk_head_dim - qk_nope_head_dim),
                *trailing_shape,
            )
            nope_plain = _plain_layout(nope_param)
            pe_plain = _plain_layout(pe_param)
            if (
                tuple(nope_plain.shape) != expected_nope_shape
                or tuple(pe_plain.shape) != expected_pe_shape
            ):
                raise ValueError(
                    "Split q_b projection shape mismatch: "
                    f"nope={tuple(nope_plain.shape)}/{expected_nope_shape}, "
                    f"pe={tuple(pe_plain.shape)}/{expected_pe_shape}"
                )
            data_by_head = data.reshape(
                local_heads,
                qk_head_dim,
                *trailing_shape,
            )
            nope = (
                data_by_head[:, :qk_nope_head_dim]
                .contiguous()
                .reshape(expected_nope_shape)
            )
            pe = (
                data_by_head[:, qk_nope_head_dim:]
                .contiguous()
                .reshape(expected_pe_shape)
            )
            with torch.no_grad():
                _store_post_pwal(nope_param, nope)
                _store_post_pwal(pe_param, pe)
            if released:
                # Keep the HBM saving across reloads: the split projections
                # are the only compute consumers of the q-up weights.
                param.data = torch.empty(
                    0, dtype=loaded_weight.dtype, device=loaded_weight.device
                )
            return loader_result

        return split_loader

    for attr in ("weight", "weight_scale", "weight_offset"):
        src = getattr(q_b_proj, attr, None)
        nope_dst = getattr(q_b_nope_proj, attr, None)
        pe_dst = getattr(q_b_pe_proj, attr, None)
        if src is None or nope_dst is None or pe_dst is None:
            continue
        orig_loader = getattr(src, "weight_loader", None)
        if orig_loader is None:
            continue
        src.weight_loader = make_split_loader(orig_loader, nope_dst, pe_dst)


def mark_split_q_up_params_loaded(
    module: nn.Module,
    loaded_params: set[str],
) -> set[str]:
    """Mark split q-up parameters populated by the q_b_proj loader.

    The split projections have no standalone checkpoint entries. Their
    parameters are populated by q_b_proj's wrapped weight loader, so model
    loading must report them as initialized whenever the corresponding source
    parameter was loaded.
    """
    split_projection_names = {"q_b_nope_proj", "q_b_pe_proj"}
    derived_names: set[str] = set()
    for name, _ in module.named_parameters():
        parts = name.split(".")
        for index, part in enumerate(parts):
            if part not in split_projection_names:
                continue
            source_parts = parts.copy()
            source_parts[index] = "q_b_proj"
            if ".".join(source_parts) in loaded_params:
                derived_names.add(name)
            break
    loaded_params.update(derived_names)
    return loaded_params


def load_sharded_param_weight(
    module: nn.Module,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    """Shard and copy a checkpoint tensor into a parameter (GGUF / TP aware)."""
    output_dim = getattr(param, "output_dim", None)
    is_sharded_weight = getattr(param, "is_sharded_weight", False)
    use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
    is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit
    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        param.weight_type = loaded_weight.item()
    if is_gguf_weight and isinstance(param, nn.UninitializedParameter):
        final_shape = list(loaded_weight.shape)
        if output_dim is not None:
            tp_size = getattr(module, "tp_size", 1)
            if final_shape[output_dim] % tp_size != 0:
                raise ValueError("loaded weight cannot be sharded across tp_size")
            final_shape[output_dim] = final_shape[output_dim] // tp_size
        param.materialize(final_shape, dtype=loaded_weight.dtype)
    param_data = param.data
    if output_dim is not None and not is_sharded_weight:
        shard_size = param_data.shape[output_dim]
        tp_rank = getattr(module, "tp_rank", 0)
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)
    if param_data.shape != loaded_weight.shape:
        raise ValueError("loaded weight shape does not match parameter shape")
    param_data.copy_(loaded_weight)


def run_post_weight_load(model: nn.Module) -> None:
    """Drive DSA/MLA second-phase weight processing (absorb / sink / conv merge)."""
    for _, module in model.named_modules():
        if module is model:
            continue
        if hasattr(module, "post_weight_load"):
            module.post_weight_load()
        if hasattr(module, "absorb_kv_b_weights"):
            module.absorb_kv_b_weights()