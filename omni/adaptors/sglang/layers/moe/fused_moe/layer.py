from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch
import torch_npu
import torchair as tng
import torch.distributed as dist

from sglang.srt.distributed import (
    get_moe_ep_group, 
    get_pp_group,
    get_world_group,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE as BaseFusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical


logger = logging.getLogger(__name__)


class FusedMoE(BaseFusedMoE):

    def __init__(self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_fused_shared_experts=num_fused_shared_experts,
            layer_id=layer_id,
            top_k=top_k,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            activation=activation,
            # apply_router_weight_on_input=apply_router_weight_on_input, # TODO: arg not found
            routed_scaling_factor=routed_scaling_factor,
            activation_alpha=None,
            swiglu_limit=None,
            with_bias=None,
        )

        assert self.quant_method is not None

        self.planner = kwargs.get("planner", None)
        self.moe_layer_idx = kwargs.get("moe_layer_idx", None)
        self.expert_mapping = kwargs.get("expert_mapping", None)

        self.global_rank = get_world_group().rank_in_group
        self.world_size = get_world_group().world_size

        self.tp_size = 1 # disable tp when enable deepep moe
        self.tp_rank = self.global_rank % self.tp_size

        self.all_to_all_group_size = self.world_size // self.tp_size
        self.all_to_all_rank = self.global_rank // self.tp_size
        self.moe_all_to_all_group = get_world_group().device_group
        self.moe_all_to_all_group_name = self.moe_all_to_all_group._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(self.global_rank)

        moe_rs_group = get_pp_group().device_group
        moe_rs_group_rank = get_pp_group().rank_in_group
        self.moe_rs_group_name = moe_rs_group._get_backend(torch.device("npu")).get_hccl_comm_name(moe_rs_group_rank)

        self.moe_layer_idx = None
        self.expert_mapping = None
        if global_server_args_dict["enable_omni_placement"]:
            self.moe_layer_idx = get_global_expert_location_metadata().get_moe_layer_idx(layer_id)
            self.expert_mapping = get_global_expert_location_metadata().expert_mapping_on_current_layer(self.moe_layer_idx)

        self.quant_scale = torch.nn.Parameter(
            torch.ones(
                (self.num_local_experts, self.w2_weight.size(-1)),
                dtype=torch.float,
            )
        )

    def forward(self,
        hidden_states: torch.Tensor,
        topk_ids,
        forward_batch,
        dynamic_scale=None,
        comm_group = None,
        **kwargs,
    ):
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        if forward_batch.is_extend_in_batch:
            final_hidden_states = self.quant_method.apply(
                layer=self,
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                scale=dynamic_scale,
                forward_batch=forward_batch,
                comm_group=comm_group,
            )
        else:
            raise NotImplementedError("moe forward not support decode")
        return final_hidden_states

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        routed_scaling_factor: Optional[torch.Tensor] = None,
    ):
        # DeekSeekv2 uses grouped_top_k
        # adapt: When num_expert_group=1, it degenerates to fused_topk.
        if use_grouped_topk:  # and num_expert_group != 1:
            # adapt end.
            if topk_group is None:
                raise ValueError(f"Unsupported topk_group is None")
            if num_expert_group is None:
                raise ValueError(f"Unsupported num_expert_group is None")

            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits.float(),
                k=top_k,                       # topk is currently 8
                bias=e_score_correction_bias,  # float32
                k_group=topk_group,            # fix: 4
                group_count=num_expert_group,  # fix 8
                group_select_mode=1,           # 0: maximum in group; 1: topk2.sum(fix)
                renorm=0,                      # 0: softmax->topk(fix); 1: topk->softmax
                norm_type=1,                   # 0: softmax; 1: sigmoid(fix)
                routed_scaling_factor=routed_scaling_factor,
                eps=float(1e-20))

            row_idx = torch.arange(
                topk_ids.numel(),
                device="npu",
                dtype=torch.int32
            ).view(
                -1, router_logits.shape[0]
            ).transpose(0, 1)
        elif custom_routing_function is None:
            topk_weights, topk_ids, row_idx = FusedMoE.fused_topk(
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
        else:
            topk_weights, topk_ids, row_idx = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids, row_idx

    @staticmethod
    def fused_topk(
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ):
        topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(gating_output, k=topk)

        if renormalize:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids, row_idx

    def apply_expert_load_balance(self,
        topk_ids: torch.Tensor,
        best_topk_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # omni placement
        if global_server_args_dict["enable_omni_placement"]:
            topk_ids = topk_ids_logical_to_physical(topk_ids, layer_idx_moe=self.moe_layer_idx, expert_mapping=self.expert_mapping)

        # Forced load balance
        #to do: get best_ep from model_extra_config
        import os
        best_ep=os.getenv("BEST_EP", 'False') == 'True'
        if best_ep:
            if best_topk_ids is None:
                t = (topk_ids.shape[0] * 8) // 256
                topk_ids = torch.arange(256, device='npu', dtype=torch.int32).unsqueeze(
                    0).repeat(t + 1, 1).view(-1, 8)[:topk_ids.shape[0]]
            elif global_server_args_dict["enable_torch_compile"]:
                topk_ids = tng.scope.npu_wait_tensor(best_topk_ids, topk_ids)
            else:
                topk_ids = best_topk_ids
        return topk_ids

def get_moe_impl_class():

    if global_server_args_dict["moe_a2a_backend"].is_deepep():
        return FusedMoE

    return BaseFusedMoE

def moe_infer_fusion(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    scale: torch.Tensor,
    forward_batch,
    comm_group = None,
) -> torch.Tensor:
    world_size = get_world_group().world_size
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    max_num_deployed_expert = layer.w13_weight.shape[0] * get_moe_ep_group().world_size
    expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = (
        torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids.to(torch.int),
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            scale=scale,  # None: non-quant; tensor with shape [num_rows,]: quant
            expert_num=max_num_deployed_expert,
            expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
            expert_tokens_num_flag=True,
            active_expert_range=[0, max_num_deployed_expert],
            quant_mode=1,  # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
        )
    )
    tokens_per_expert_group = tokens_per_expert.new_empty(
        tokens_per_expert.shape[0]
    )
    dist.all_to_all_single(
        tokens_per_expert_group, tokens_per_expert, group=comm_group
    )
    combine_tokens = torch.stack(
        [tokens_per_expert_group, tokens_per_expert], dim=0
    )
    # view: EP, E // EP
    combine_tokens = combine_tokens.view(2, world_size, -1).sum(2)
    all_tokens = combine_tokens[0].sum()
    combine_tokens_cpu = combine_tokens.cpu().tolist()
    input_splits = combine_tokens_cpu[1]
    output_splits = combine_tokens_cpu[0]

    gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
    dist.all_to_all_single(
        gathered_tokens,
        expanded_x,
        output_splits,
        input_splits,
        group=comm_group)

    dynamic_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
    dist.all_to_all_single(
        dynamic_scale,
        pertoken_scale,
        output_splits,
        input_splits,
        group=comm_group)

    # reroute
    (
        hidden_states,
        dynamic_scale,
        topk_ids,
        expert_tokens,
    ) = torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(world_size, -1),
        per_token_scales=dynamic_scale,
    )
    expert_tokens = expert_tokens.to(torch.int64)

    get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
        layer.layer_id,
        [],
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        num_tokens_per_expert=expert_tokens,
    )

    hidden_size = hidden_states.size(-1)

    if dynamic_scale is not None and dynamic_scale.dim() > 1:
        dynamic_scale = dynamic_scale.reshape(-1)
        hidden_states = hidden_states.view(-1, hidden_size)

    # GroupGemm-0
    gateup_output = torch_npu.npu_grouped_matmul(
        [hidden_states],
        [layer.w13_weight],
        group_list=expert_tokens,
        split_item=3,
        group_type=0,
        scale=None,
        per_token_scale=None,
        output_dtype=torch.int32,
        tuning_config=[0],
        group_list_type=1,
    )[0]
    down_input, dynamic_scale = torch_npu.npu_dequant_swiglu_quant(
        gateup_output,
        weight_scale=layer.w13_weight_scale.squeeze(-1),
        activation_scale=dynamic_scale,
        quant_scale=layer.quant_scale,
        group_index=expert_tokens,
        activate_left=True,
        quant_mode=1,
    )

    del gateup_output

    if dynamic_scale is not None and dynamic_scale.dim() > 1:
        inter_size = down_input.size(-1)
        dynamic_scale = dynamic_scale.reshape(-1)
        down_input = down_input.view(-1, inter_size)

    # GroupGemm-1
    hidden_states = torch_npu.npu_grouped_matmul(
        [down_input],
        [layer.w2_weight],
        group_list=expert_tokens,
        split_item=3,
        group_type=0,
        scale=[layer.w2_weight_scale.squeeze(-1)],
        per_token_scale=[dynamic_scale],
        output_dtype=torch.bfloat16,
        tuning_config=[0],
        group_list_type=1,
    )[0]

    new_x = torch.index_select(hidden_states, 0, topk_ids.float().argsort().int())
    gathered_tokens = new_x.new_empty(*expanded_x.shape)
    dist.all_to_all_single(
        gathered_tokens, new_x, input_splits, output_splits, group=comm_group
    )

    return hidden_states, gathered_tokens, expanded_row_idx
