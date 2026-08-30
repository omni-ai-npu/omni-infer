# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import multiprocessing
import os
from typing import Callable, Optional, Union

import torch
import torch_npu
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
    UnquantizedFusedMoEMethod,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from omni_npu.compilation.acl_graph import set_aclgraph_recapture
from omni_npu.layers.fused_moe.fused_moe import fused_experts_tp
from omni_npu.layers.fused_moe.fused_moe_method_base import NPUFusedMoEMethodBase
from omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize import PreparePermuteOptions, PreparePermuteResult
from omni_npu.layers.prefetch import PrefetchManager
from omni_npu.layers.utils import named_stream
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.plugin_decorators import attn_decorator
from omni_npu.v1.utils import on_ascend950

torch.npu.config.allow_internal_format = True
logger = init_logger(__name__)
FULL_LOAD_WEIGHT_NDIM = 3


@UnquantizedFusedMoEMethod.register_oot
class NPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod, NPUFusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.model_prefetch = PrefetchManager()
        self.sub_stream = named_stream("sub_stream")
        self.agrs_overlap_stream = named_stream("moe_agrs_overlap_stream")
        self.enable_agrs_finalize_metadata_overlap = (
            model_extra_config.operator_opt_config.enable_agrs_finalize_metadata_overlap
        )
        self.on_ascend950 = on_ascend950()

    def apply(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        orig_num_tokens = hidden_states.shape[0]
        strategy, strategy_impl = self.select_communication_strategy(orig_num_tokens)
        is_sequence_parallel = getattr(layer, "is_sequence_parallel", False)
        is_need_slice = (
            not model_extra_config.parall_config.ena_seq_parallel and self.tp_size > 1 and not is_sequence_parallel
            and orig_num_tokens > 1
            and not model_extra_config.operator_opt_config.enable_moe_allreduce
        )
        x_slice = hidden_states
        if is_need_slice:
            padded_num_tokens = -(orig_num_tokens // -self.tp_size) * self.tp_size
            local_num_tokens = padded_num_tokens // self.tp_size
            num_pads = padded_num_tokens - orig_num_tokens

            if num_pads > 0:
                x_slice = torch.nn.functional.pad(x_slice, (0, 0, 0, num_pads), value=0)

            start = self.tp_rank * local_num_tokens
            end = (self.tp_rank + 1) * local_num_tokens
            x_slice = x_slice[start:end]

        if router_logits is not None and (
            layer.gate is None
            or getattr(layer, "use_precomputed_router_logits", False)
        ):
            if is_need_slice:
                if num_pads > 0:
                    router_logits = torch.nn.functional.pad(
                        router_logits, (0, 0, 0, num_pads), value=0
                    )
                router_logits = router_logits[start:end]
        elif layer.gate is not None:
            gate_weight = getattr(layer.gate, "weight", None)
            gate_in_fp32 = (
                model_extra_config.operator_opt_config.router_gating_in_fp32
                or (
                    gate_weight is not None
                    and gate_weight.dtype == torch.float32
                )
            )
            if gate_in_fp32:
                router_logits, _ = layer.gate(x_slice.float())
            else:
                router_logits, _ = layer.gate(x_slice)
        else:
            assert router_logits is not None, "Expected gate or router_logits must be provided."
            if is_need_slice:
                if num_pads > 0:
                    router_logits = torch.nn.functional.pad(router_logits, (0, 0, 0, num_pads), value=0)
                router_logits = router_logits[start:end]

        enable_prefetch = model_extra_config.operator_opt_config.enable_prefetch
        shared_expert_multi_stream = model_extra_config.operator_opt_config.shared_expert_multi_stream
        cur_stream = torch_npu.npu.current_stream()
        if enable_prefetch:
            self.sub_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.sub_stream):
                self.model_prefetch.prefetch("moe", router_logits, layer=layer)

        topk_weights, topk_ids = NPUFusedMoERunner.select_experts(
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        if model_extra_config.operator_opt_config.enable_precision_strong_consistency:
            # Sort each row of topk_ids in ascending order, and reorder topk_weights accordingly.
            sorted_indices = topk_ids.argsort(dim=-1)
            topk_ids = topk_ids.gather(dim=-1, index=sorted_indices).contiguous()
            topk_weights = topk_weights.gather(dim=-1, index=sorted_indices).contiguous()

        # TODO: Support TP-only mode
        if not layer.moe_parallel_config.use_ep:
            if self.on_ascend950:
                if shared_expert_multi_stream or enable_prefetch:
                    cur_stream.wait_stream(self.sub_stream)
                routed_output = fused_experts_tp(
                    layer=layer,
                    x=x_slice,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                )
                if layer.shared_experts is not None:
                    shared_output = layer.shared_experts(x_slice)
                    return shared_output, routed_output + shared_output
                return routed_output
            return fused_experts_tp(
                layer=layer,
                x=x_slice,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )

        if (
            getattr(layer, "vllm_config", None) is not None
            and layer.vllm_config.model_config is not None
            and getattr(layer.vllm_config.model_config, "enable_return_routed_experts", False)
        ):
            # In dummy/profile runs the capturer may not be initialized yet.
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)

        prepare_permute_result = self.apply_prepare_permute(
            strategy_impl, layer, x_slice, topk_ids,
            options=PreparePermuteOptions(topk_weights=topk_weights),
        )
        finalize_metadata = None
        overlap_experts = (
            strategy == "agrs" and self.enable_agrs_finalize_metadata_overlap and self.agrs_overlap_stream is not None
        )
        if overlap_experts:
            self.agrs_overlap_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.agrs_overlap_stream):
                output = self.apply_experts(
                    layer=layer,
                    prepare_permute_result=prepare_permute_result,
                    activation=activation,
                )
        if strategy == "agrs" and self.enable_agrs_finalize_metadata_overlap:
            finalize_metadata = self.prepare_finalize_metadata(
                strategy_impl, layer, topk_weights, prepare_permute_result
            )
        if not overlap_experts:
            output = self.apply_experts(
                layer=layer,
                prepare_permute_result=prepare_permute_result,
                activation=activation,
            )

        shared_output = None
        shared_experts = layer.shared_experts
        if shared_experts is not None:
            if shared_expert_multi_stream:
                self.sub_stream.wait_stream(cur_stream)
                with torch.npu.stream(self.sub_stream):
                    if shared_experts.gate_up_proj.tp_size > 1:
                        # Shared experts with TP>1 require full hidden_states;
                        # output is all-reduced later.
                        shared_output = shared_experts(hidden_states)
                    else:
                        shared_output = shared_experts(x_slice)
            else:
                if shared_experts.gate_up_proj.tp_size > 1:
                    # Shared experts with TP>1 require full hidden_states;
                    # output is all-reduced later.
                    shared_output = shared_experts(hidden_states)
                else:
                    shared_output = shared_experts(x_slice)

        if enable_prefetch:
            self.sub_stream.wait_stream(cur_stream)
            with torch.npu.stream(self.sub_stream):
                self.model_prefetch.prefetch("next_attn", shared_output, layer=layer)

        if overlap_experts:
            cur_stream.wait_stream(self.agrs_overlap_stream)
        routed_output = self.apply_unpermute_finalize(
            strategy_impl,
            layer,
            output,
            topk_ids,
            topk_weights,
            prepare_permute_result,
            finalize_metadata=finalize_metadata,
        )

        if shared_expert_multi_stream or enable_prefetch:
            cur_stream.wait_stream(self.sub_stream)

        use_custom_model_add = "omni_custom_models" in os.environ.get("VLLM_PLUGINS", "")
        share_expert_tp = (
            shared_experts.gate_up_proj.tp_size if shared_experts is not None else 1
        )
        if shared_experts is not None:
            if share_expert_tp > 1:
                if not model_extra_config.operator_opt_config.enable_moe_allreduce:
                    shared_output = tensor_model_parallel_all_reduce(shared_output)
            elif use_custom_model_add:
                routed_output = routed_output + shared_output

        if is_need_slice:
            routed_output = tensor_model_parallel_all_gather(routed_output, dim=0)[:orig_num_tokens]

        if shared_experts is not None and share_expert_tp > 1 and use_custom_model_add:
            routed_output = routed_output + shared_output
        if shared_output is not None:
            return shared_output, routed_output
        return routed_output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        if model_extra_config.operator_opt_config.gmm_nz:
            current_method = multiprocessing.get_start_method()
            multiprocessing.set_start_method("spawn", force=True)
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, torch_npu.Format.FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, torch_npu.Format.FRACTAL_NZ)
            multiprocessing.set_start_method(current_method, force=True)
            opt_raw = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")  # bytes
            allow_internal_format = opt_raw.strip().lower() == b"enable"
            if allow_internal_format:
                set_weight_attrs(layer.w13_weight, {"is_weight_nz": True})
                set_weight_attrs(layer.w2_weight, {"is_weight_nz": True})
        set_weight_attrs(layer.w13_weight, {"is_weight_transposed": True})
        set_weight_attrs(layer.w2_weight, {"is_weight_transposed": True})

    def apply_experts(
        self,
        layer: torch.nn.Module,
        prepare_permute_result: PreparePermuteResult,
        activation: str = "silu",
    ) -> torch.Tensor:
        hidden_states = prepare_permute_result.hidden_states_sorted_by_experts
        expert_tokens = prepare_permute_result.expert_tokens.to(torch.int64)
        group_list_type = int(layer.moe_parallel_config.use_ep)
        gate_up_proj = torch_npu.npu_grouped_matmul(
            [hidden_states],
            [layer.w13_weight],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type,
        )[0]
        intermediate_hidden_states = torch_npu.npu_swiglu(gate_up_proj)
        return torch_npu.npu_grouped_matmul(
            [intermediate_hidden_states],
            [layer.w2_weight],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            group_type=0,
            group_list_type=group_list_type,
        )[0]


@MoERunner.register_oot
class NPUFusedMoERunner(MoERunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.routed_experts.quant_method.make_communication_strategy_selector(self)

    @property
    def shared_experts(self):
        """The model's own shared-experts module, not vLLM's container.

        vLLM now wraps whatever the model passes as ``shared_experts`` in a
        ``SharedExperts`` object that adds DBO bookkeeping and stream overlap,
        and whose ``forward`` takes a second ``order`` argument. omni's MoE
        path calls the module with one argument and reads its projections
        (``gate_up_proj.tp_size``) to decide how to slice, so every omni caller
        wants the wrapped module -- there are around forty such reads across
        the quant methods. Unwrapping once here fixes them all, and leaves
        ``_shared_experts`` holding the container so vLLM's own machinery
        still works.

        Sunset: remove once omni's MoE path drives the shared experts through
        vLLM's ordering protocol instead of calling them directly.
        """
        shared = self._shared_experts
        try:
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
                SharedExperts,
            )
        except ImportError:
            return shared
        return shared._layer if isinstance(shared, SharedExperts) else shared

    @shared_experts.setter
    def shared_experts(self, value):
        # The parent exposes a read-only property; keep assignment working for
        # callers that set it directly, writing through to the attribute the
        # getter reads.
        self._shared_experts = value

    def __getattr__(self, name: str):
        """Answer for ``routed_experts`` anything the runner does not define.

        vLLM split the MoE layer in two: this runner owns routing, the gate and
        the shared experts, while ``routed_experts`` owns the expert weights,
        the quant method and every routing parameter. omni's MoE kernels were
        written against the single pre-split layer and reach for both halves
        through one object -- ``layer.gate`` and ``layer.shared_experts`` next
        to ``layer.w13_weight`` and ``layer.global_num_experts`` -- and the
        runner is what ``register_layer_for_moe_forward_op`` puts in the
        forward context, so it is the object they get. Delegating here keeps
        that contract without threading ``.routed_experts`` through every
        kernel, quant method and strategy implementation.

        Sunset: remove once the omni kernels take the two modules separately,
        the way vLLM's own ``RoutedExperts.forward_modular`` does.
        """
        parent_getattr = getattr(super(), "__getattr__", None)
        if parent_getattr is not None:
            try:
                return parent_getattr(name)
            except AttributeError:
                pass
        # Dunders are never delegated. torch, copy and pickle probe names like
        # __getstate__ and __deepcopy__ on every module; answering those from
        # routed_experts hands out the wrong object's protocol methods, which
        # silently wedges the distributed worker pool.
        if name.startswith("__"):
            raise AttributeError(name)
        # Reach into the instance dicts directly: going through
        # self.routed_experts would re-enter this method and recurse when that
        # is itself the missing name (during __init__, before it is assigned).
        experts = self.__dict__.get("_modules", {}).get("routed_experts")
        if experts is not None:
            try:
                return getattr(experts, name)
            except AttributeError:
                pass
        # A few of the old layer attributes did not move to routed_experts but
        # into the config object -- moe_parallel_config is the one the kernels
        # ask for by name.
        moe_config = getattr(experts, "moe_config", None) if experts is not None else None
        if moe_config is None:
            moe_config = self.__dict__.get("moe_config")
        if moe_config is not None and hasattr(moe_config, name):
            return getattr(moe_config, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        # Some NPU-packed weights are kept transposed in memory for kernels.
        # Temporarily switch to canonical layout for parent loading logic.
        if getattr(param, "is_weight_nz", False):
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.ND)
        full_load = loaded_weight.ndim == 3
        is_weight_transposed = getattr(param, "is_weight_transposed", False)

        if is_weight_transposed:
            param.data = param.data.transpose(1, 2)

        ep_rank = get_ep_group().rank_in_group
        if self.enable_eplb:
            exists_locally, local_pos = self._quant_method.planner.is_expert_on_current_rank(
                self._quant_method.moe_layer_idx,
                expert_id,
                ep_rank,
                self.local_num_experts,
            )
            if not exists_locally:
                return False if return_success else None
            expert_id = local_pos + self.local_num_experts * ep_rank

        if any(
            name in weight_name
            for name in ("bias", "int4_scale", "weight_offset")
        ):
            shard_dim = 0 if "bias" in weight_name else 1
            quant_method = getattr(param, "quant_method", None)
            expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if expert_id == -1:
                return False if return_success else None
            expert_data = param.data if full_load else param.data[expert_id]
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self.routed_experts._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            else:
                raise ValueError(
                    f"quant method must be {FusedMoeWeightScaleSupported.CHANNEL.value}, "
                    f"{weight_name=}"
                )
            result = True if return_success else None
        else:
            result = self.routed_experts.weight_loader(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=return_success,
            )

        if is_weight_transposed:
            param.data = param.data.transpose(1, 2)
        if getattr(param, "is_weight_nz", False):
            param.data = torch_npu.npu_format_cast(param.data, torch_npu.Format.FRACTAL_NZ)
            set_aclgraph_recapture(True)
        return result

    def maybe_init_modular_kernel(self) -> None:
        return None

    @attn_decorator(type="moe_ffn")
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor | None = None,
    ):
        if router_logits is None:
            router_logits = hidden_states
        if self._shared_experts is None:
            return torch.ops.vllm.npu_moe_forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
                layer_name=self.layer_name,
            )
        return torch.ops.vllm.npu_moe_forward_shared(
            hidden_states=hidden_states,
            router_logits=router_logits,
            layer_name=self.layer_name,
        )

    @staticmethod
    def select_experts(
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: Optional[torch.Tensor] = None,
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ):
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # profile run, force load balance
            ep_rank = get_ep_group().rank_in_group
            global_num_experts = router_logits.shape[1]
            num_tokens = router_logits.shape[0]
            topk_ids = (
                torch.arange(
                    ep_rank * num_tokens * top_k,
                    (ep_rank + 1) * num_tokens * top_k,
                    dtype=torch.int32,
                    device=router_logits.device,
                ).view(num_tokens, top_k)
                % global_num_experts
            )
            topk_weights = torch.rand_like(topk_ids, dtype=router_logits.dtype)
            return topk_weights, topk_ids

        if use_grouped_topk:
            if topk_group is None:
                raise ValueError("Unsupported topk_group is None")
            if num_expert_group is None:
                raise ValueError("Unsupported num_expert_group is None")
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits.float(),
                k=top_k,
                bias=e_score_correction_bias,
                k_group=topk_group,
                group_count=num_expert_group,
                group_select_mode=1,
                renorm=0,
                norm_type=1,
                routed_scaling_factor=routed_scaling_factor,
                eps=1e-20,
            )
        elif custom_routing_function is None:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits.float(),
                k=top_k,
                routed_scaling_factor=routed_scaling_factor,
                bias=e_score_correction_bias,
            )
            if renormalize:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights, topk_ids = custom_routing_function(
                gating_output=router_logits, topk=top_k, renormalize=renormalize
            )

        return topk_weights, topk_ids


class _NPUFusedMoEFactory:
    select_experts = NPUFusedMoERunner.select_experts
    make_expert_params_mapping = staticmethod(fused_moe_make_expert_params_mapping)


class NPUFusedMoE(_NPUFusedMoEFactory):
    def __new__(cls, *args, gate=None, **kwargs):
        return FusedMoE(*args, gate=gate, runner_cls=NPUFusedMoERunner, **kwargs)


class NPUSharedFusedMoE(_NPUFusedMoEFactory):
    def __new__(cls, *args, gate=None, shared_experts=None, **kwargs):
        return FusedMoE(
            *args,
            gate=gate,
            shared_experts=shared_experts,
            runner_cls=NPUFusedMoERunner,
            **kwargs,
        )


def _moe_enable_eplb(layer) -> bool:
    """Whether expert-load balancing is on for this layer.

    It used to be a layer attribute; after vLLM split the MoE layer it lives
    in the parallel config inside ``moe_config``. Read whichever the object in
    hand actually carries, so both shapes work.
    """
    value = getattr(layer, "enable_eplb", None)
    if value is not None:
        return bool(value)
    moe_config = getattr(layer, "moe_config", None)
    parallel = getattr(moe_config, "moe_parallel_config", None)
    return bool(getattr(parallel, "enable_eplb", False))


def _npu_moe_apply(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
    return self.quant_method.apply(
        layer=self,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=self.top_k,
        renormalize=self.renormalize,
        use_grouped_topk=self.use_grouped_topk,
        global_num_experts=self.global_num_experts,
        expert_map=self.expert_map if not self.rocm_aiter_fmoe_enabled else self.expert_mask,
        topk_group=self.topk_group,
        num_expert_group=self.num_expert_group,
        custom_routing_function=self.custom_routing_function,
        scoring_func=self.scoring_func,
        routed_scaling_factor=self.routed_scaling_factor,
        e_score_correction_bias=self.e_score_correction_bias,
        activation=self.activation,
        apply_router_weight_on_input=self.apply_router_weight_on_input,
        enable_eplb=_moe_enable_eplb(self),
        expert_load_view=getattr(self, "expert_load_view", None),
        logical_to_physical_map=getattr(self, "logical_to_physical_map", None),
        logical_replica_count=getattr(self, "logical_replica_count", None),
    )


def npu_moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is None
    return _npu_moe_apply(self, hidden_states, router_logits)


def npu_moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def npu_moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is not None
    return _npu_moe_apply(self, hidden_states, router_logits)


def npu_moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_out = torch.empty_like(hidden_states)
    fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="npu_moe_forward",
    op_func=npu_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=npu_moe_forward_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_moe_forward_shared",
    op_func=npu_moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=npu_moe_forward_shared_fake,
    dispatch_key="PrivateUse1",
)
