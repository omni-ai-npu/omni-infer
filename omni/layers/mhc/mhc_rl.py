# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from omni_npu.layers.mhc.cube_side_task_ops import (
    maybe_register_mhc_task,
    resolve_mhc_h_res,
)
from omni_npu.layers.utils import SIDE_STREAM_NAME, named_stream
from omni_npu.v1.utils import on_ascend950
from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)
try:
    import omni_training_custom_ops
except ImportError as e:
    logger.warning(f"Failed to import omni_training_custom_ops: {e}")
except Exception as e:
    logger.warning(f"Error occurred while importing omni_training_custom_ops: {e}")


MHC_DIRECT_PENDING_KEY = "mhc_direct_pending"
MHC_FUSED_SPLIT_PENDING_KEY = "mhc_fused_split_pending"


def mhc_direct_launch(
    holder_layer_name: str,
    task_key: str,
    h_res: torch.Tensor,
) -> torch.Tensor:
    """Launch Sinkhorn immediately on the shared side stream.

    The Cube-side task path is triggered by quantized matmul wrappers. BF16
    linear and grouped-matmul paths have no such trigger, so their Sinkhorn
    must be submitted here instead.
    """
    fwctx = get_forward_context()
    mhc_module = fwctx.no_compile_layers[holder_layer_name]
    main_stream = torch.npu.current_stream()
    side_stream = named_stream(SIDE_STREAM_NAME)

    h_res_holder = [h_res]
    ready_event = torch.npu.Event()
    done_event = torch.npu.Event()
    ready_event.record(main_stream)
    with torch.npu.stream(side_stream):
        ready_event.wait(side_stream)
        h_res.record_stream(side_stream)
        h_res_holder[0] = mhc_module.mhc_sinkhorn(h_res)
        done_event.record()

    fwctx.additional_kwargs.setdefault(MHC_DIRECT_PENDING_KEY, {})[task_key] = (
        h_res_holder,
        ready_event,
        done_event,
    )
    return h_res


def mhc_direct_launch_fake(
    holder_layer_name: str,
    task_key: str,
    h_res: torch.Tensor,
) -> torch.Tensor:
    return h_res


def mhc_direct_fetch(
    holder_layer_name: str,
    task_key: str,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Wait for and return the Sinkhorn result launched by mhc_direct_launch."""
    fwctx = get_forward_context()
    pending = fwctx.additional_kwargs.get(MHC_DIRECT_PENDING_KEY)
    entry = pending.pop(task_key, None) if pending else None
    if entry is None:
        # Preserve correctness if launch was skipped by an eager/graph fallback.
        return fwctx.no_compile_layers[holder_layer_name].mhc_sinkhorn(fallback)

    h_res_holder, _ready_event, done_event = entry
    main_stream = torch.npu.current_stream()
    main_stream.wait_event(done_event)
    result = h_res_holder[0]
    result.record_stream(main_stream)
    return result


def mhc_direct_fetch_fake(
    holder_layer_name: str,
    task_key: str,
    fallback: torch.Tensor,
) -> torch.Tensor:
    return fallback


direct_register_custom_op(
    op_name="mhc_direct_launch",
    op_func=mhc_direct_launch,
    mutates_args=[],
    fake_impl=mhc_direct_launch_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="mhc_direct_fetch",
    op_func=mhc_direct_fetch,
    mutates_args=[],
    fake_impl=mhc_direct_fetch_fake,
    dispatch_key="PrivateUse1",
)


def mhc_fused_split_launch(
    holder_layer_name: str,
    task_key: str,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch fused MHC split + Sinkhorn outside the compiled FX graph."""
    fwctx = get_forward_context()
    mhc_module = fwctx.no_compile_layers[holder_layer_name]
    main_stream = torch.npu.current_stream()
    side_stream = named_stream(SIDE_STREAM_NAME)

    ready_event = torch.npu.Event()
    done_event = torch.npu.Event()
    ready_event.record(main_stream)
    with torch.npu.stream(side_stream):
        ready_event.wait(side_stream)
        residual.record_stream(side_stream)
        h_post, h_res = mhc_module.mhc_pre_split_post_res(residual)
        h_res = mhc_module.mhc_sinkhorn(h_res)
        done_event.record()

    fwctx.additional_kwargs.setdefault(
        MHC_FUSED_SPLIT_PENDING_KEY, {}
    )[task_key] = (ready_event, done_event)
    return h_post, h_res


def mhc_fused_split_launch_fake(
    holder_layer_name: str,
    task_key: str,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mhc_module = get_forward_context().no_compile_layers[holder_layer_name]
    residual = residual.reshape(
        -1, mhc_module.num_stream, mhc_module.hidden_size
    )
    h_post = torch.empty_like(residual[..., 0], dtype=torch.float32)
    h_res = torch.empty_like(
        residual[..., :mhc_module.num_stream], dtype=torch.float32
    )
    return h_post, h_res


def mhc_fused_split_fetch(
    holder_layer_name: str,
    task_key: str,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wait for fused side work and transfer its outputs to the main stream."""
    fwctx = get_forward_context()
    pending = fwctx.additional_kwargs.get(MHC_FUSED_SPLIT_PENDING_KEY)
    entry = pending.pop(task_key, None) if pending else None
    if entry is not None:
        _ready_event, done_event = entry
        main_stream = torch.npu.current_stream()
        main_stream.wait_event(done_event)
        h_post.record_stream(main_stream)
        h_res.record_stream(main_stream)
    return h_post, h_res


def mhc_fused_split_fetch_fake(
    holder_layer_name: str,
    task_key: str,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(h_post), torch.empty_like(h_res)


direct_register_custom_op(
    op_name="mhc_fused_split_launch",
    op_func=mhc_fused_split_launch,
    mutates_args=[],
    fake_impl=mhc_fused_split_launch_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="mhc_fused_split_fetch",
    op_func=mhc_fused_split_fetch,
    mutates_args=[],
    fake_impl=mhc_fused_split_fetch_fake,
    dispatch_key="PrivateUse1",
)


class NPUmHCRL(torch.nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        pre_only: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        self.hc_eps = 1e-6
        assert config.mhc_use_gamma
        self.pre_only = pre_only
        self.on_ascend950 = on_ascend950()

        self.prefix = prefix
        self.enable_mhc_multistream = bool(
            model_extra_config.operator_opt_config.enable_mhc_multistream
            and not self.pre_only
            and self.prefix
        )
        if self.enable_mhc_multistream:
            vllm_config = get_current_vllm_config()
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self
            # Quantized HiF8/MXFP8 kernels consume Cube-side tasks themselves.
            # Unquantized BF16 kernels do not, so submit Sinkhorn directly.
            self.use_direct_mhc_multistream = (
                getattr(vllm_config, "quant_config", None) is None
            )
        else:
            self.use_direct_mhc_multistream = False
        self.use_mhc_fusion_op = bool(
            getattr(model_extra_config.operator_opt_config, "use_mhc_fusion_op", False)
            and not self.pre_only
        )
        if not self.pre_only:
            self.branch_alpha = torch.nn.Parameter(
                torch.empty(3, dtype=torch.float32)
            )
            self.branch_beta = torch.nn.Parameter(
                torch.empty(self.num_stream * (self.num_stream + 2), dtype=torch.float32)
            ) 
        else:
            self.branch_alpha_pre = torch.nn.Parameter(
                torch.empty(1, dtype=torch.float32)
            )
            self.branch_beta_pre = torch.nn.Parameter(
                torch.empty(self.num_stream, dtype=torch.float32)
            )

        self.phi = ReplicatedLinear(
            self.hidden_size * self.num_stream,
            output_size=self.num_stream if self.pre_only \
                        else (self.num_stream + 2) * self.num_stream,
            bias=False,
            prefix=f"{prefix}.phi",
            params_dtype=torch.float32,
        )

        self.norm_gamma = torch.nn.Parameter(
            torch.empty(self.hidden_size * self.num_stream, dtype=torch.float32)
        )

    def _mhc_source_params(self) -> list[torch.nn.Parameter]:
        params = [self.phi.weight, self.norm_gamma]
        if self.pre_only:
            params.extend([self.branch_alpha_pre, self.branch_beta_pre])
        else:
            params.extend([self.branch_alpha, self.branch_beta])
        return params

    def _set_derived_buffer(self, name: str, value: torch.Tensor) -> None:
        existing = getattr(self, name, None)
        if existing is None:
            self.register_buffer(name, value, persistent=False)
        else:
            existing.copy_(value)

    def process_weights_after_loading(self) -> None:
        source_versions = tuple(p._version for p in self._mhc_source_params())
        if getattr(self, "_mhc_processed_weight_versions", None) == source_versions:
            return
        self._set_derived_buffer("phi_weight", self.phi.weight * self.norm_gamma)
        self._set_derived_buffer(
            "phi_weight_pre",
            self.phi.weight[:self.num_stream] * self.norm_gamma,
        )
        self._set_derived_buffer(
            "phi_weight_post_res",
            self.phi.weight[self.num_stream:] * self.norm_gamma,
        )

        if self.pre_only:
            self.branch_alpha_post_res = None
            self.branch_beta_post_res = None
        else:
            self._set_derived_buffer(
                "branch_alpha_pre", self.branch_alpha[0:1]
            )
            self._set_derived_buffer(
                "branch_alpha_post_res", self.branch_alpha[1:]
            )
            self._set_derived_buffer(
                "branch_beta_pre", self.branch_beta[:self.num_stream]
            )
            self._set_derived_buffer(
                "branch_beta_post_res", self.branch_beta[self.num_stream:]
            )
        self._mhc_processed_weight_versions = source_versions

    def post_weight_load(self) -> None:
        # Match the training path by rounding loaded weights to the model dtype once,
        # while keeping FP32 parameter storage for the MHC computation. For a BF16
        # checkpoint and FP16 model dtype: BF16 load -> FP32 -> FP16 round -> FP32.
        with torch.no_grad():
            for parameter in self._mhc_source_params():
                parameter.copy_(parameter.to(model_extra_config.dtype))

    def _mhc_pre_naive(self, hidden_states: torch.Tensor):
        shape, dtype = hidden_states.size(), hidden_states.dtype
        hidden_states = hidden_states.flatten(-2).float()
        rsqrt = torch.rsqrt(
            hidden_states.square().mean(-1, keepdim=True) + self.norm_eps,
        )
        mixes = torch.nn.functional.linear(
            hidden_states * rsqrt * self.norm_gamma,
            self.phi.weight,
        )
        h_pre, h_post, h_res = mixes.split(
            [self.num_stream, self.num_stream, self.num_stream**2], dim=-1,
        )

        h_pre = torch.nn.functional.sigmoid(
            h_pre * self.branch_alpha_pre + self.branch_beta_pre
        ) + self.hc_eps
        h_post = 2 * torch.nn.functional.sigmoid(
            h_post * self.branch_alpha_post + self.branch_beta_post
        )
        h_res = h_res.unflatten(-1, (self.num_stream, self.num_stream)) * self.branch_alpha_res \
                    + self.branch_beta_res.view(self.num_stream, self.num_stream)
        hidden_states = torch.sum(
            h_pre.unsqueeze(-1) * hidden_states.view(shape), dim=-2,
        ).to(dtype)
        return hidden_states, h_post, h_res

    def _mhc_sinkhorn_naive(self, h_res: torch.Tensor):
        h_res = h_res.softmax(-1) + self.hc_eps

        # === Step 1: Initial Col Norm ===
        col_sum = h_res.sum(-2, keepdim=True) + self.hc_eps
        h_res = h_res / col_sum

        # === Step 2: Loop ===
        for _ in range(self.mhc_recur_norm - 1):
            # Row Norm
            row_sum = h_res.sum(-1, keepdim=True) + self.hc_eps
            h_res = h_res / row_sum

            # Col Norm
            col_sum = h_res.sum(-2, keepdim=True) + self.hc_eps
            h_res = h_res / col_sum

        return h_res

    def _mhc_post_naive(
            self,
            hidden_states: torch.Tensor,
            h_post: torch.Tensor,
            residual: torch.Tensor,
            h_res: torch.Tensor,
    ):
        hidden_states = (
            h_post.unsqueeze(-1) * hidden_states.unsqueeze(-2) \
            + torch.sum(h_res.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
        ).to(hidden_states.dtype)
        return hidden_states

    def mhc_pre(
        self,
        hidden_states: torch.Tensor,
    ):
        if self.pre_only:
            dtype = hidden_states.dtype
            hidden_states = hidden_states.view(-1, self.hidden_size * self.num_stream)
            hidden_states = hidden_states.float()
            if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
                normalized_hidden_states, _ = torch_npu.npu_rms_norm(
                    hidden_states,
                    self.norm_gamma.view(self.hidden_size * self.num_stream),
                    self.hc_eps,
                )
                hpre_weight = self.phi(normalized_hidden_states)[0]
            else:
                rsqrt = torch.rsqrt(
                    hidden_states.square().mean(-1, keepdim=True) + self.hc_eps
                )
                hpre_weight = self.phi(
                    hidden_states
                    * rsqrt
                    * self.norm_gamma.view(1, self.hidden_size * self.num_stream),
                )[0]

            hpre_weight = torch.nn.functional.sigmoid(
                hpre_weight * self.branch_alpha_pre
                + self.branch_beta_pre.view(1, self.num_stream)
            ) + self.hc_eps
            hpre_weight = hpre_weight.view(-1, self.num_stream, 1)
            hidden_states = hidden_states.view(-1, self.num_stream, self.hidden_size)
            hidden_states = torch.sum(
                hpre_weight * hidden_states,
                dim=1
            )
            hidden_states = hidden_states.view(-1, self.hidden_size)
            hidden_states = hidden_states.to(dtype)
            h_post = None
            h_res = None
        else:
            hidden_states = hidden_states.view(-1, self.num_stream, self.hidden_size)

            if not self.on_ascend950:
                if model_extra_config.operator_opt_config.use_batch_invariant_op:
                    phi_weight = self.phi.weight
                    gamma = self.norm_gamma.view(self.num_stream, self.hidden_size)
                    out_flag = 1
                else:
                    phi_weight = self.phi.weight * self.norm_gamma
                    gamma = None
                    out_flag = 0
                hidden_states, h_post, h_res, _, _, _ = \
                    torch.ops.custom.npu_manifold_constrained_hyper_connection_pre(
                        hidden_states,
                        phi_weight,
                        self.branch_alpha,
                        self.branch_beta,
                        gamma=gamma,
                        norm_eps=self.hc_eps,
                        hc_eps=self.hc_eps,
                        out_flag=out_flag,
                )
            else:
                hidden_states, h_post, h_res, _, _, _ = torch_npu.npu_mhc_pre(
                    hidden_states,
                    self.phi.weight,
                    self.branch_alpha,
                    self.branch_beta,
                    gamma=self.norm_gamma.view(self.num_stream, -1),
                    out_flag=0,
                )
            
        return hidden_states, h_post, h_res

    def mhc_sinkhorn(
        self,
        h_res: torch.Tensor,
    ):
        if self.pre_only:
            return h_res

        if not self.on_ascend950:
            h_res, _, _ = torch.ops.custom.npu_sinkhorn(
                h_res,
                eps=self.hc_eps,
                num_iters=self.mhc_recur_norm,
                out_flag=1 if model_extra_config.operator_opt_config.use_batch_invariant_op else 0,
            )
        else:
            h_res, _, _ = torch_npu.npu_mhc_sinkhorn(
                h_res,
                eps=self.hc_eps,
                num_iters=self.mhc_recur_norm,
                out_flag=0,
            )

        return h_res

    def can_use_fusion(self, hidden_states: torch.Tensor) -> bool:
        """Whether fusion is enabled and its post-load weights are ready."""
        return (
            self.use_mhc_fusion_op
            and hasattr(self, "phi_weight_pre")
            and hidden_states.shape[0] <= 256
        )

    def _reshape_residual_for_fusion(
        self,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Adapt master's flattened MHC residual to the fusion-op layout."""
        return residual.reshape(-1, self.num_stream, self.hidden_size)

    def mhc_pre_split_post_res(
        self,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the post/residual branches after fused pre-only execution."""
        if self.pre_only:
            raise ValueError("mhc_pre_split_post_res requires a full MHC module")
        residual = self._reshape_residual_for_fusion(residual)
        return torch.ops.custom.npu_ai_infra_mhc_pre_split_post_res(
            residual,
            self.phi_weight_post_res,
            self.branch_alpha_post_res,
            self.branch_beta_post_res,
            # NPUmHCRL deliberately uses hc_eps for the MHC RMS in mhc_pre.
            # Keep the split fusion path numerically equivalent to that RL
            # baseline instead of inheriting NPUmHC's rms_norm_eps behavior.
            norm_eps=self.hc_eps,
        )

    def launch_fused_split_sinkhorn(
        self,
        residual: torch.Tensor,
        task_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch split_post_res + Sinkhorn on the shared side stream."""
        if not self.enable_mhc_multistream or not task_key:
            h_post, h_res = self.mhc_pre_split_post_res(residual)
            return h_post, self.mhc_sinkhorn(h_res)
        return torch.ops.vllm.mhc_fused_split_launch(
            self.prefix, task_key, residual
        )

    def resolve_fused_split_sinkhorn(
        self,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
        task_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for fused side work immediately before the main-stream use."""
        if not self.enable_mhc_multistream or not task_key:
            return h_post, h_res
        return torch.ops.vllm.mhc_fused_split_fetch(
            self.prefix, task_key, h_post, h_res
        )

    @staticmethod
    def _norm_weight_fp32(norm_module: nn.Module) -> torch.Tensor:
        weight_fp32 = getattr(norm_module, "weight_fp32", None)
        if weight_fp32 is not None:
            return weight_fp32
        return norm_module.weight.float()

    def mhc_sandwich_norm_post_preonly(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
        post_norm_module: nn.Module,
        pre_mhc_module: "NPUmHCRL",
        pre_norm_module: nn.Module,
        block_norm_module: nn.Module | None = None,
        return_h_in_f32: bool = False,
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
        """Fuse post norm/MHC post with the next MHC pre-only and pre norm."""
        residual = self._reshape_residual_for_fusion(residual)
        hidden_states, residual, hidden_states_fp32 = (
            torch.ops.custom.npu_ai_infra_mhc_sandwich_norm_post_preonly_v2(
                hidden_states,
                residual,
                h_post,
                h_res,
                pre_mhc_module.phi_weight_pre,
                pre_mhc_module.branch_alpha_pre,
                pre_mhc_module.branch_beta_pre,
                self._norm_weight_fp32(post_norm_module),
                self._norm_weight_fp32(pre_norm_module),
                gamma_2=(
                    self._norm_weight_fp32(block_norm_module)
                    if block_norm_module is not None
                    else None
                ),
                norm_eps=pre_mhc_module.norm_eps,
                hc_eps=pre_mhc_module.hc_eps,
                return_h_in_f32=return_h_in_f32,
            )
        )
        if return_h_in_f32:
            hidden_states = {
                "hidden_states_bf16": hidden_states,
                "hidden_states_fp32": hidden_states_fp32,
            }
        return hidden_states, residual

    def maybe_register_sinkhorn(
        self,
        h_res: torch.Tensor,
        task_key: str,
    ) -> torch.Tensor:
        """Register Sinkhorn to overlap with the matching Cube-heavy op."""
        if not self.enable_mhc_multistream or not task_key:
            return h_res
        if self.use_direct_mhc_multistream:
            return torch.ops.vllm.mhc_direct_launch(
                self.prefix, task_key, h_res,
            )
        return maybe_register_mhc_task(self.prefix, task_key, h_res)

    def resolve_sinkhorn(
        self,
        h_res: torch.Tensor,
        task_key: str,
    ) -> torch.Tensor:
        """Fetch the side-stream result, or run Sinkhorn synchronously."""
        if not self.enable_mhc_multistream or not task_key:
            return self.mhc_sinkhorn(h_res)
        if self.use_direct_mhc_multistream:
            return torch.ops.vllm.mhc_direct_fetch(
                self.prefix, task_key, h_res,
            )
        return resolve_mhc_h_res(self, task_key, h_res)

    def mhc_post(
            self,
            hidden_states: torch.Tensor,
            h_post: torch.Tensor,
            residual: torch.Tensor,
            h_res: torch.Tensor,
    ):
        if self.pre_only:
            return residual

        residual = residual.view(-1, self.num_stream, self.hidden_size)
        if not self.on_ascend950:
            hidden_states = torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
                residual,
                h_res,
                hidden_states,
                h_post,
            )
        else:
            hidden_states = torch_npu.npu_mhc_post(
                residual,
                h_res,
                hidden_states,
                h_post,
            )
        
        hidden_states = hidden_states.view(-1, self.num_stream * self.hidden_size)

        return hidden_states
