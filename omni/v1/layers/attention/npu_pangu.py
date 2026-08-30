# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Callable, Optional, Union, Tuple
from itertools import accumulate

import torch
import torch_npu
from torch.nn import functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.model_executor.models.utils import extract_layer_index
from vllm.distributed import get_tp_group, split_tensor_along_last_dim
from vllm.config import VllmConfig, CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonDecodeMetadata,
    MLACommonMetadata,
    MLACommonPrefillMetadata,
)
from omni_npu.attention.backends.mome import NPUMomeAttentionMetadata
from vllm.logger import init_logger

from omni_npu.v1.utils import current_stream, on_ascend950
from omni_npu.v1.layers.utils import yarn_get_mscale
from omni_npu.v1.layers.linear import (
    ColumnParallelFlashCommLinear,
    RowParallelFlashCommLinear,
    ShardedLinear,
)
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.attention.backends.utils import SPManager, DummySPManager, conv_sp
from omni_npu.layers.mome.npu_mome import ColumnParallelMOME
from omni_npu.layers.attention.npu_sparse_attentions import (
    MLASWAAttention,
    DSAAttention,
    MomeAttention,
)

from omni_npu.compilation.utils import (
    capture_graph_task,
    OP_FIA_SINK,
    OP_FIA_PIONEER,
)
from omni_npu.plugin_decorators import attn_decorator

from omni_npu.layers.utils import named_stream

from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

try:
    import omni_custom_ops
    import omni_training_custom_ops
except ImportError as e:
    logger.warning(f"Failed to import omni_custom_ops: {e}")


class CrossLayerSharedOp:
    """A metadata-producing op whose output is shared across layers
    within a single forward step.

    Holds one persistent buffer per `caller` tag. The first layer in a
    step that calls with `recompute=True` runs the underlying op and
    copies the result into the buffer for its caller; later layers in
    the same step call with `recompute=False` and read the buffer
    directly. Buffer addresses are stable across steps so aclgraph /
    cudagraph captures them safely.

    Producer detection lives on the caller (e.g. a per-layer flag like
    `is_fa_metadata_producer`); this class only manages the buffers.
    """

    def __init__(
        self,
        op: Callable[..., torch.Tensor],
        shape: tuple[int, ...],
        dtype: torch.dtype,
        callers: tuple[str, ...],
        device: str | torch.device = "npu",
    ):
        self._op = op
        self._buffers: dict[str, torch.Tensor] = {
            caller: torch.empty(shape, dtype=dtype, device=device)
            for caller in callers
        }
        self._default_buffer: torch.Tensor = torch.empty(
            shape, dtype=dtype, device=device,
        )

    def __call__(
        self,
        op_args: dict,
        recompute: bool,
        caller: str,
    ) -> torch.Tensor:
        buffer = self._buffers.get(caller, self._default_buffer)
        if buffer is self._default_buffer or recompute:
            buffer.copy_(self._op(**op_args))
        return buffer


npu_fused_infer_attention_sink_metadata: CrossLayerSharedOp | None = None
npu_ai_infra_attention_pioneer_metadata: CrossLayerSharedOp | None = None


def _get_slot_mapping_2d(attn_metadata, layer_idx=-1):
    slot_mapping_2d = getattr(attn_metadata, "slot_mapping_2d", None)
    if slot_mapping_2d is not None:
        return slot_mapping_2d
    elif hasattr(attn_metadata, "get_slot_mapping_2d"):
        # A5 HiF8: eagerly populate !1421's own slot_mapping_2d memo on first use
        # and reuse it for the rest of the step. Pure decode / prefill skip
        # _prepare_phase_inputs (which sets this memo for mixed batches), and the
        # A5 hif8 ds_mla scatter sites call with layer_idx=-1, which !1421's
        # per-layer closure would otherwise recompute on every layer. The no-arg
        # get_slot_mapping_2d() call also matches SWA/MLA metadata. Value-identical;
        # computes the 2-D slot map once per metadata group per step.
        if on_ascend950():
            attn_metadata.slot_mapping_2d = attn_metadata.get_slot_mapping_2d()
            return attn_metadata.slot_mapping_2d
        # MLA's lambda takes no args; DSA's closure expects layer_idx.
        # Skip the positional arg when caller didn't supply one.
        if layer_idx == -1:
            return attn_metadata.get_slot_mapping_2d()
        return attn_metadata.get_slot_mapping_2d(layer_idx)
    else:
        return None


def decode_only(attn_metadata):
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    return has_decode and (not has_prefill)


class NPUPanguIndexer(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.layer_name = prefix
        self.layer_idx = extract_layer_index(prefix)
        self.block_size = cache_config.block_size
        self.block_size_c8 = 2 * self.block_size
        self.on_ascend950 = on_ascend950()
        self._init_indexer_weights()
        self.quant_cache_dtype = ["hif8_ds_mla", "fp8_ds_mla", "int8_ds_mla", "li_int8_ds_mla"]
        self.use_rope_fusion_op = model_extra_config.operator_opt_config.use_rope_fusion_op

    def _init_indexer_weights(self):
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.index_head_dim * self.index_n_heads,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.wq_b",
            return_bias=False,
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.wk",
            return_bias=False,
        )
        self.k_norm = RMSNorm(
            self.index_head_dim,
            eps=self.config.rms_norm_eps,
        )
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.index_n_heads,
            quant_config=None,
            bias=False,
            prefix=f"{self.layer_name}.weights_proj",
            return_bias=False,
        )

    def _apply_lightning_indexer(self, *args, **kwargs):
        if self.cache_config.cache_dtype in self.quant_cache_dtype:
            quant_output = self._apply_lightning_indexer_quant(*args, **kwargs)
            return quant_output
        else:
            unquant_output = self._apply_lightning_indexer_unquant(*args, **kwargs)
            return unquant_output

    def _apply_lightning_indexer_cp(self, *args, **kwargs):
        if self.cache_config.cache_dtype in self.quant_cache_dtype:
            quant_output = self._apply_lightning_indexer_cp_quant(*args, **kwargs)
            return quant_output
        else:
            unquant_output = self._apply_lightning_indexer_cp_unquant(*args, **kwargs)
            return unquant_output

    def _update_indexer_cache(self, *args, **kwargs):
        if quant_output := self._update_indexer_cache_quant(*args, **kwargs):
            return quant_output
        else:
            unquant_output = self._update_indexer_cache_unquant(*args, **kwargs)
            return unquant_output

    def _apply_lightning_indexer_unquant(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode

        return torch.ops.custom.npu_lightning_indexer_enhance(
            query=q,
            key=kv_cache[1].unsqueeze(2),
            weights=weights,
            actual_seq_lengths_query=metadata.query_cumlens,
            actual_seq_lengths_key=metadata.seq_lens,
            block_table=metadata.block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            sparse_block_size=1,
            sparse_block_mode=False,
        )[0]

    def _apply_lightning_indexer_quant(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode

        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla", "fp8_ds_mla"]:
            if self.cache_config.cache_dtype == "hif8_ds_mla":
                q_quant, q_scale = torch_npu.npu_dynamic_quant(
                    q, dst_type=torch_npu.hifloat8,
                    dst_type_max=15.0
                )
                query_dtype = torch_npu.hifloat8
                key_dtype = torch_npu.hifloat8
            else:  # fp8_ds_mla
                q_quant, q_scale = torch_npu.npu_dynamic_quant(
                    q, dst_type=torch.float8_e4m3fn
                )
                query_dtype = None
                key_dtype = None

            return torch_npu.npu_quant_lightning_indexer(
                query=q_quant,
                key=kv_cache[1].unsqueeze(-2) if len(kv_cache[1].shape) == 3 else kv_cache[1],
                weights=weights,
                query_dequant_scale=q_scale,
                key_dequant_scale=kv_cache[2],
                actual_seq_lengths_query=metadata.query_cumlens.clone().to(torch.int32),
                actual_seq_lengths_key=metadata.seq_lens.clone().to(torch.int32),
                block_table=metadata.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
                query_dtype=query_dtype,
                key_dtype=key_dtype
            )
        elif self.cache_config.cache_dtype in ["int8_ds_mla", "li_int8_ds_mla"]:
            q_int8, q_scale = torch_npu.npu_dynamic_quant(q)
            return torch_npu.torch.ops.custom.npu_ai_infra_quant_lightning_indexer(
                query=q_int8,
                key=kv_cache[1].unsqueeze(2),
                weights=weights.to(torch.float16),
                query_dequant_scale=q_scale.to(torch.float16),
                key_dequant_scale=kv_cache[2],
                actual_seq_lengths_query=metadata.query_cumlens,
                actual_seq_lengths_key=metadata.seq_lens,
                block_table=metadata.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
        else:
            raise RuntimeError(
                f"Unsupported cache_dtype '{self.cache_config.cache_dtype}' "
                f"for quant lightning indexer."
            )

    def _apply_lightning_indexer_cp_unquant(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        sp_manager: Optional[MLACommonMetadata] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        actual_seq_lengths_query, actual_seq_lengths_kv, _, block_table = sp_manager.cp_attn_meta()

        return torch.ops.custom.npu_lightning_indexer_enhance(
            query=q,
            key=kv_cache[1].unsqueeze(2),
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_kv,
            block_table=block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            sparse_block_size=1,
            sparse_block_mode=False,
        )[0]

    def _apply_lightning_indexer_cp_quant(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        sp_manager: Optional[MLACommonMetadata] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        actual_seq_lengths_query, actual_seq_lengths_kv, _, block_table = sp_manager.cp_attn_meta()

        if self.cache_config.cache_dtype in ["int8_ds_mla", "li_int8_ds_mla"]:
            q_int8, q_scale = torch_npu.npu_dynamic_quant(q)
            return torch_npu.torch.ops.custom.npu_ai_infra_quant_lightning_indexer(
                query=q_int8,
                key=kv_cache[1].unsqueeze(2),
                weights=weights.to(torch.float16),
                query_dequant_scale=q_scale.to(torch.float16),
                key_dequant_scale=kv_cache[2],
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_kv,
                block_table=block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
        else:
            raise RuntimeError(
                f"Unsupported cache_dtype '{self.cache_config.cache_dtype}' "
                f"for CP quant lightning indexer."
            )

    def _update_indexer_cache_unquant(
        self,
        k: torch.Tensor,
        attention_metadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:

        slot_mapping_2d = _get_slot_mapping_2d(attention_metadata, self.layer_idx)

        # TODO: need fix
        torch.ops.custom.npu_ai_infra_scatter_block_update_(
            kv_cache[1],
            slot_mapping_2d,
            k.view(-1, k.shape[-1]),
        )
        return True

    def _update_indexer_cache_quant(
        self,
        k: torch.Tensor,
        attention_metadata,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:

        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla"]:
            k_hif8, k_scale = torch_npu.npu_dynamic_quant(
                k, dst_type=torch_npu.hifloat8,
                dst_type_max=15.0
            )
            slot_mapping_2d = _get_slot_mapping_2d(attention_metadata)

            torch_npu.npu_scatter_nd_update_(
                kv_cache[1].view(torch.int8),
                slot_mapping_2d,
                k_hif8.view(torch.int8),
            )
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2],
                slot_mapping_2d,
                k_scale.unsqueeze(-1),
            )
            return True

        elif self.on_ascend950 and self.cache_config.cache_dtype in ["fp8_ds_mla"]:
            k_fp8, k_scale = torch_npu.npu_dynamic_quant(
                k, dst_type=torch.float8_e4m3fn
            )
            slot_mapping_2d = _get_slot_mapping_2d(attention_metadata)

            torch_npu.npu_scatter_nd_update_(
                kv_cache[1],
                slot_mapping_2d,
                k_fp8,
            )
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2],
                slot_mapping_2d,
                k_scale.unsqueeze(-1),
            )
            return True

        elif self.cache_config.cache_dtype in ["int8_ds_mla", "li_int8_ds_mla"]:
            k_int8, k_scale = torch_npu.npu_dynamic_quant(k)
            k_scale_fp16 = k_scale.to(torch.float16).view(-1, 1)

            if self.cache_config.cache_dtype == "li_int8_ds_mla":
                slot_mapping_2d = _get_slot_mapping_2d(attention_metadata, self.layer_idx)

                torch.ops.custom.npu_ai_infra_scatter_block_update_(
                    kv_cache[1],
                    slot_mapping_2d,
                    k_int8.view(-1, k_int8.shape[-1]),
                )
                torch.ops.custom.npu_ai_infra_scatter_block_update_(
                    kv_cache[2],
                    slot_mapping_2d,
                    k_scale_fp16.view(-1, k_scale_fp16.shape[-1]),
                )
            else:
                slot_mapping_2d = _get_slot_mapping_2d(attention_metadata, self.layer_idx)
                torch.ops.custom.npu_ai_infra_scatter_block_update_(
                    kv_cache[1],
                    slot_mapping_2d,
                    k_int8.view(-1, k_int8.shape[-1]),
                )
                torch.ops.custom.npu_ai_infra_scatter_block_update_(
                    kv_cache[2],
                    slot_mapping_2d,
                    k_scale_fp16.view(-1, k_scale_fp16.shape[-1]),
                )
            return True

        else:
            return False

    def _rope_split_q(self, q, cos, sin):
        # Split the indexer query, apply rope to the pe part, and reassemble.
        # Shared by _indexer_prolog and forward_cp, which differ only in the
        # cos/sin source (main vs context-parallel path).
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = torch.split(
            q,
            [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.index_n_heads, self.qk_rope_head_dim),
            cos.view(-1, 1, 1, self.qk_rope_head_dim),
            sin.view(-1, 1, 1, self.qk_rope_head_dim),
        ).squeeze(1)
        return torch.cat([q_pe, q_nope], dim=-1)

    def _indexer_prolog(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        q = self.wq_b(qr)
        k = self.wk(hidden_states)
        k = self.k_norm(k)

        if self.use_rope_fusion_op:
            q, k = torch_npu.npu_apply_rotary_pos_emb(
                q.view(-1, 1, self.index_n_heads, self.index_head_dim),
                k.view(-1, 1, 1, self.index_head_dim),
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                layout="BSND",
                rotary_mode="half",
            )
            if self.on_ascend950:
                # A5 HiF8: !1324 leaves the captured k at the 4D input-arg shape
                # (T, 1, 1, index_head_dim); a 4D k would flow into the hif8 path
                # (_update_indexer_cache_quant: npu_dynamic_quant + npu_scatter_nd_update_),
                # diverging from tested-good's 2D k. The rope values are already applied,
                # so restore the pre-!1324 2D shape. Non-A5 keeps !1324's k as-is.
                k = k.view(-1, self.index_head_dim)
            q = q.view(-1, self.index_n_heads, self.index_head_dim)
        else:
            q = self._rope_split_q(q, cos, sin)

            k_pe, k_nope = torch.split(
                k, 
                [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
                dim=-1,
            )
            k_pe = torch_npu.npu_rotary_mul(
                k_pe.view(-1, 1, 1, self.qk_rope_head_dim),
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
            ).squeeze(1).squeeze(1)

            k = torch.cat([k_pe, k_nope], dim=-1)

        weights = self.weights_proj(hidden_states)

        return q, k, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        q, k, weights = self._indexer_prolog(
            hidden_states,
            qr,
            cos,
            sin,
        )

        kv_cache_2 = kv_cache[2] if len(kv_cache) > 2 else None
        parent_name = self.layer_name.rsplit(".indexer", 1)[0]
        kv_cache = torch.ops.vllm.npu_pangu_indexer_cache_update(
            k,
            kv_cache[0], kv_cache[1], kv_cache_2,
            parent_name,
        )

        return torch.ops.vllm.npu_pangu_lightning_indexer(
            q, weights,
            kv_cache[0], kv_cache[1], kv_cache_2,
            parent_name,
        )

    def forward_cp(
        self,
        sp_x: torch.Tensor,
        q_lora: torch.Tensor,
        sp_cos: torch.Tensor,
        sp_sin: torch.Tensor,
        cp_cos: torch.Tensor,
        cp_sin: torch.Tensor,
        sp_manager: Optional[SPManager] = None,
        attn_metadata: Optional[MLACommonMetadata] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q = self.wq_b(q_lora)
        k = self.wk(sp_x)
        k = self.k_norm(k)

        if self.use_rope_fusion_op:
            torch_npu.npu_apply_rotary_pos_emb(
                k.view(-1, 1, 1, self.index_head_dim),
                k.view(-1, 1, 1, self.index_head_dim),
                sp_cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sp_sin.view(-1, 1, 1, self.qk_rope_head_dim),
                layout="BSND",
                rotary_mode="half",
            )
            k = k.view(-1, self.index_head_dim)

            q_view = q.view(-1, 1, self.index_n_heads, self.index_head_dim)
            torch_npu.npu_apply_rotary_pos_emb(
                q_view,
                q_view,
                cp_cos.view(-1, 1, 1, self.qk_rope_head_dim),
                cp_sin.view(-1, 1, 1, self.qk_rope_head_dim),
                layout="BSND",
                rotary_mode="half",
            )
            q = q.view(-1, self.index_n_heads, self.index_head_dim)
        else:
            q = self._rope_split_q(q, cp_cos, cp_sin)

            k_pe, k_nope = torch.split(
                k,
                [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim],
                dim=-1,
            )
            k_pe = torch_npu.npu_rotary_mul(
                k_pe.view(-1, 1, 1, self.qk_rope_head_dim),
                sp_cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sp_sin.view(-1, 1, 1, self.qk_rope_head_dim),
            ).squeeze(1).squeeze(1)

            k = torch.cat([k_pe, k_nope], dim=-1)

        k = sp_manager.ag_tokens(k)

        weights = self.weights_proj(sp_x)
        weights = sp_manager.sp_to_cp(weights)
        self._update_indexer_cache(
            k,
            attn_metadata,
            kv_cache,
        )

        topk_indices = self._apply_lightning_indexer_cp(
            q,
            weights,
            sp_manager,
            kv_cache,
        )

        return topk_indices


class NPUPanguSparseAttention(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        rope_theta: int,
        swa_layers: list[int],
        param_sink_number: int,
        sliding_window_list: list[int],
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix = prefix
        self.layer_idx = extract_layer_index(self.prefix)
        assert len(swa_layers) == len(sliding_window_list)
        self.swa_layers = swa_layers if swa_layers else []
        self.sliding_window_list = sliding_window_list if sliding_window_list else []
        self.aligned_window_size = max(self.sliding_window_list)
        self.skip_topk = False
        if self.layer_idx in self.swa_layers:
            # SWA layer
            pos_in_swa = self.swa_layers.index(self.layer_idx)
            self.sliding_window = self.sliding_window_list[pos_in_swa]
            self.is_dsa_layer = False
        elif self.layer_idx >= config.num_hidden_layers:
            # MTP layer
            self.sliding_window = self.sliding_window_list[-1]
            self.is_dsa_layer = False
        elif (getattr(config, "index_topk", None) or 0) > 0:
            # DSA layer
            self.sliding_window = None
            self.is_dsa_layer = True
            self.index_topk = config.index_topk
            self.index_head_dim = config.index_head_dim
            self.skip_topk = self._skip_topk(config)
            if self.skip_topk:
                logger.info(
                    "Index Share enabled: layer %s skip_topk=True (reuse prior DSA topk)",
                    self.layer_idx,
                )
        else:
            # MLA layer
            # set a very large sliding window to disable the sliding window attention and fall back to global attention
            self.sliding_window = max(1024 * 1024, self.aligned_window_size + 1)
            self.is_dsa_layer = False
        # SWA / Full MLA differ in pre_tokens; keep separate metadata buffers + producers.
        # Non-SWA is Full MLA only when DSA is not enabled (index_topk); otherwise non-SWA is DSA.
        first_swa = self.swa_layers[0] if self.swa_layers else None
        has_dsa = bool(getattr(config, "index_topk", None))
        first_mla = None
        if not has_dsa:
            first_mla = next(
                (i for i in range(config.num_hidden_layers) if i not in self.swa_layers),
                None,
            )
        self.is_fa_metadata_producer = self.layer_idx in (
            first_swa, first_mla, config.num_hidden_layers,
        )
        self._fa_meta_suffix = (
            ""
            if self.sliding_window is not None
            and self.sliding_window <= self.aligned_window_size
            else "_mla"
        )
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rope_theta = rope_theta
        self.num_heads = num_heads
        self.tp_size = get_tp_group().world_size
        assert num_heads % self.tp_size == 0
        self.ena_dsa_cp = model_extra_config.parall_config.ena_context_parallel
        self.is_cp_layer = self.is_dsa_layer and self.ena_dsa_cp
        self.ena_swa_attn_seq_parallel = model_extra_config.parall_config.ena_swa_attn_seq_parallel
        self.is_attn_sp_layer = self.ena_swa_attn_seq_parallel and not self.is_dsa_layer
        self.num_local_heads = (
            num_heads
            if self.is_cp_layer or self.is_attn_sp_layer
            else num_heads // self.tp_size
        )
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.quant_symbol = quant_config is not None
        self.rope_interleave = getattr(config, "rope_interleave", False) or getattr(config, "rope_interleaved", False)
        self.moe_comm_strategy = model_extra_config.operator_opt_config.moe_comm_strategy
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.cache_config = cache_config
        # k_nope ds_mla cache quant: direct hif8 cast with unit scale (a3-aligned);
        # the legacy per-128-tile dynamic-quant branch is retained but disabled.
        self.use_dynamic_quant_k_nope = False
        self.hf_config = config
        self.layer_name = prefix
        self.param_sink_number = param_sink_number
        self.on_ascend950 = on_ascend950()
        self.is_pd_disagg = vllm_config.kv_transfer_config is not None
        self.is_prefill_node = (self.is_pd_disagg and \
            vllm_config.kv_transfer_config.kv_role == "kv_producer")
        # o_conv cache is transferred as-is in PD disaggregation; TP-sharded
        # cache/layout would mismatch between prefill and decode nodes.
        self.disable_o_conv_tp = (
            self.ena_dsa_cp or self.is_attn_sp_layer or self.is_pd_disagg
        )
        assert model_extra_config.operator_opt_config.use_noncontiguous_kv
        self.use_aicpu_fa_tiling = model_extra_config.operator_opt_config.use_aicpu_fa_tiling
        self.enable_flashcomm2 = model_extra_config.parall_config.enable_flashcomm2
        self.sharded_o_proj = (
            model_extra_config.parall_config.sharded_o_proj
            and self.is_pd_disagg
            and self.is_prefill_node
        )
        self.use_mome = getattr(self.hf_config, "use_mome", False)
        self.use_mome_inplace_update = model_extra_config.operator_opt_config.use_mome_inplace_update
        self.first_chunk_pa = (
            (vllm_config.scheduler_config.enable_chunked_prefill or
            vllm_config.cache_config.enable_prefix_caching) and
            not model_extra_config.operator_opt_config.optimize_first_chunk
        )
        self.enable_mome_sp = model_extra_config.operator_opt_config.enable_mome_sp

        if self.is_cp_layer:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self.num_computed_for_cp = torch.zeros(
                (max_num_reqs*2, ), 
                device="npu", 
                dtype=torch.int32, 
            )
        else:
            self.num_computed_for_cp = None

        self.quant_cache_dtype = ["hif8_ds_mla", "fp8_ds_mla", "int8_ds_mla"]
        self.block_size = self.cache_config.block_size
        self.block_size_c8 = 2 * self.block_size

        self.dummy_value_cache = torch.zeros(
            (1, self.block_size_c8, 1, self.kv_lora_rank),
            device='npu',
            dtype=torch.bfloat16,
        )
        self.dummy_value_cache_hif8_fp8 = torch.zeros(
            (1, self.block_size_c8, 1, 656),
            device='npu',
            dtype=torch.uint8,
        )

        # Switch between split (q_b_nope_proj / q_b_pe_proj children) and
        # unsplit (single self.q_b_proj + torch.split) q_up in MLA multi-stream
        # paths. Must be set before _init_MLA_weights so the children are
        # created when split is enabled.
        self.split_q_up_in_multistream = (
            model_extra_config.operator_opt_config.split_q_up_in_multistream
        )

        self._init_MLA_weights()
        self._init_rotary_emb()
        self._init_param_sinks()
        self._align_pagesize()
        self._init_attention_layers()
        self._init_mome_layer()
        self._init_cross_layer_shared_ops()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Optional one-shot callback fired in _forward_decode after the attention core
        # and before _mla_epilog. Used by callers to launch side-stream work that
        # overlaps with v_up / o_proj / all_reduce inside _mla_epilog.
        # Cleared after each invocation so it never leaks across forwards.
        # Initialized unconditionally: on Ascend950 the multistream prolog that
        # would set it is bypassed on the hif8_ds_mla decode path, so the
        # _forward_decode read must still see a defined (None) attribute.
        self.pre_epilog_callback = None

        if self.on_ascend950:
            self.side_stream = named_stream("sub_stream")
        else:
            # Side stream for MLA prolog Q/KV overlap (set externally via set_side_stream)
            self.side_stream = None

    def _init_MLA_weights(self):
        replicate_attention_weights = self.is_cp_layer or self.is_attn_sp_layer
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.q_a_proj",
            return_bias=False,
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.kv_a_proj_with_mqa",
            return_bias=False,
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            eps=self.hf_config.rms_norm_eps,
        )
        self.q_b_proj = ColumnParallelFlashCommLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.q_b_proj",
            return_bias=False,
            disable_tp=replicate_attention_weights,
        )
        if self.split_q_up_in_multistream:
            # Per-half projections used by the MLA multi-stream split path.
            # They share q_b_proj's quant_config so each child's quant_method
            # handles dequant / NZ cast / scale trans correctly via its own
            # process_weights_after_loading. Weights are populated from the
            # q_b_proj checkpoint slice via a custom weight_loader installed
            # below, before vLLM calls per-module PWAL.
            self.q_b_nope_proj = ColumnParallelFlashCommLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_nope_head_dim,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{self.layer_name}.q_b_nope_proj",
                return_bias=False,
                disable_tp=replicate_attention_weights,
            )
            self.q_b_pe_proj = ColumnParallelFlashCommLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_rope_head_dim,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{self.layer_name}.q_b_pe_proj",
                return_bias=False,
                disable_tp=replicate_attention_weights,
            )
            self._install_q_b_split_loaders()
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=self.hf_config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelFlashCommLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.layer_name}.kv_b_proj",
            return_bias=False,
            disable_tp=replicate_attention_weights,
        )
        if self.sharded_o_proj:
            self.o_proj = ShardedLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                shard_group=get_tp_group(),
                quant_config=self.quant_config,
                prefix=f"{self.layer_name}.o_proj",
                return_bias=False,
            )
        else:
            self.o_proj = RowParallelFlashCommLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=self.quant_config,
                reduce_results=False,
                prefix=f"{self.layer_name}.o_proj",
                disable_tp=True if (
                    replicate_attention_weights
                    or (self.enable_flashcomm2 and not self.is_dsa_layer)
                ) else False,
            )

    def _apply_o_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        if self.sharded_o_proj:
            return self.o_proj(attn_output)
        return self.o_proj(attn_output)[0]

    def _init_rotary_emb(self):
        rope_parameters = {
            "rope_theta": self.hf_config.rope_parameters["rope_theta"],
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": self.max_position_embeddings,
            "type": "yarn",
            "rope_type": "deepseek_yarn",
        }

        config = self.hf_config
        rope_scaling = getattr(config, "rope_scaling", None)
        is_mrope = rope_scaling is not None and rope_scaling.get("mrope_section") is not None

        def _build_rope(num_hidden_layers_cache: int | None = None):
            if is_mrope:
                from vllm.model_executor.layers.rotary_embedding import get_rope_wrapper
                return get_rope_wrapper(
                    self.qk_rope_head_dim,
                    max_position=self.max_position_embeddings,
                    rotary_dim=self.qk_rope_head_dim,
                    base=config.rope_parameters["rope_theta"],
                    rope_scaling=rope_scaling,
                    num_hidden_layers_cache=num_hidden_layers_cache
                )

            return get_rope(
                self.qk_rope_head_dim,
                max_position=self.max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=(not self.rope_interleave),
            )

        def _get_num_cache_layers(default_num_layers: int):
            if getattr(config, "is_mtp_layer", False):
                return config.num_nextn_predict_layers
            return default_num_layers


        self.rotary_emb = _build_rope(_get_num_cache_layers(config.num_hidden_layers))

        if self.is_dsa_layer and not self.skip_topk:
            if is_mrope:
                assert isinstance(config.dsa_layers, list)
                num_cache_layers = _get_num_cache_layers(len(config.dsa_layers))
            else:
                num_cache_layers = None
            self.indexer_rope_emb = _build_rope(num_cache_layers)
        else:
            self.indexer_rope_emb = None

        if (
            self.hf_config.rope_parameters["rope_type"] != "default"
            and self.hf_config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = self.hf_config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = self.hf_config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

    def _init_param_sinks(self):
        self.param_sink_compressed_kv = torch.nn.Parameter(
            torch.empty(
                (self.param_sink_number, self.kv_lora_rank), 
                device='npu', 
                dtype=torch.bfloat16,
            )
        )
        self.param_sink_k_pe = torch.nn.Parameter(
            torch.empty(
                (self.param_sink_number, self.qk_rope_head_dim), 
                device='npu', 
                dtype=torch.bfloat16,
            )
        )
        self.block_size = self.cache_config.block_size
        self.sink_slot_mapping = torch.arange(
            self.param_sink_number, device='npu', dtype=torch.int32,
        )
        self.sink_slot_indices = torch.stack(
            [self.sink_slot_mapping // self.block_size, self.sink_slot_mapping % self.block_size],
            dim=1,
        )

    def _align_pagesize(self):
        self.mome_kernel_width = getattr(self.hf_config, "router_sliding_window", 0)
        if self.use_mome:
            assert self.num_heads % self.tp_size == 0, \
                "For MoME attention, num_heads should be divisible by tp_size."
            if self.disable_o_conv_tp:
                o_mome_cache_shape = (self.num_heads * self.v_head_dim,)
            else:
                o_mome_cache_shape = (self.num_heads * self.v_head_dim // self.tp_size,)
            self.mome_state_shapes = (
                (self.q_lora_rank,),
                (self.kv_lora_rank,),
                o_mome_cache_shape,
            )
            self.mome_state_dtypes = (
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
            )

        if not self.vllm_config.speculative_config:
            self.num_spec_tokens = 0
        else:
            self.num_spec_tokens = self.vllm_config.speculative_config.num_speculative_tokens

        self.cache_dtype_str = self.cache_config.cache_dtype
        self.page_size_padded = self._calculate_page_size_padded(
            cache_config=self.cache_config,
            cache_dtype_str=self.cache_dtype_str,
            config=self.hf_config,
        )

    def _init_attention_layers(self):
        if self.is_dsa_layer and not self.skip_topk:
            self.indexer = NPUPanguIndexer(
                self.vllm_config,
                self.hf_config,
                self.quant_config,
                self.cache_config,
                f"{self.layer_name}.indexer",
            )
        else:
            self.indexer = None

        attn_kwargs = {
            "num_heads": self.num_local_heads,
            "scale":  self.scaling,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "kv_b_proj": self.kv_b_proj,
            "quant_config": self.quant_config,
            "cache_config": self.cache_config,
            "prefix": f"{self.layer_name}.attn",
        }

        if self.is_dsa_layer:
            attn_kwargs.update({
                "indexer": self.indexer,
                "indexer_head_dim": self.index_head_dim,
                "cache_dtype_str": self.cache_dtype_str,
                "page_size_padded": self.page_size_padded,
            })
            if self.cache_dtype_str in ["fp8_ds_mla", "hif8_ds_mla", "int8_ds_mla"]:
                attn_kwargs.update({
                    "block_size": self.block_size_c8,
                })
            self.attn = DSAAttention(**attn_kwargs)
        else:
            attn_kwargs.update({
                "cache_dtype_str": self.cache_dtype_str,
                "page_size_padded": self.page_size_padded,
                "sliding_window": self.aligned_window_size 
                                  if self.sliding_window <= self.aligned_window_size
                                  else None,
                "num_extra_reserved_blocks": model_extra_config.operator_opt_config.num_extra_reserved_blocks,
            })
            self.attn = MLASWAAttention(**attn_kwargs)

    def _init_mome_layer(self):
        if not self.use_mome:
            return

        num_extra_token = 1 if self.is_pd_disagg else 0
        fake_num_spec_tokens = max(self.num_spec_tokens, num_extra_token)

        mome_kwargs = {
            "kernel_size": self.mome_kernel_width,
            "num_spec_tokens": fake_num_spec_tokens,
            "state_dtypes": self.mome_state_dtypes,
            "state_shapes": self.mome_state_shapes,
            "quant_config": self.quant_config,
            "cache_config": self.cache_config,
            "prefix": f"{self.layer_name}.mome",
            "page_size_padded": self.page_size_padded,
            "num_extra_reserved_blocks": model_extra_config.operator_opt_config.num_extra_reserved_blocks,
        }
        self.mome_attn = MomeAttention(**mome_kwargs)

        self.qa_conv = ColumnParallelMOME(
            dim=self.q_lora_rank,
            kernel_width=self.mome_kernel_width,
            prefix=f"{self.layer_name}.qa_conv",
            disable_tp=True,
        )
        self.compresskv_conv = ColumnParallelMOME(
            dim=self.kv_lora_rank,
            kernel_width=self.mome_kernel_width,
            prefix=f"{self.layer_name}.compresskv_conv",
            disable_tp=True,
        )
        self.o_conv = ColumnParallelMOME(
            dim=self.num_heads * self.v_head_dim,
            kernel_width=self.mome_kernel_width,
            prefix=f"{self.layer_name}.o_conv",
            disable_tp=self.disable_o_conv_tp,
        )

        # kv_cache slot per conv. Keep these as int attributes: a Module-keyed
        # dict here breaks ACLGraph capture under torch >= 2.12 Dynamo.
        self.qa_conv.mome_cache_index = 0
        self.compresskv_conv.mome_cache_index = 1
        self.o_conv.mome_cache_index = 2

    def _init_cross_layer_shared_ops(self):
        global npu_fused_infer_attention_sink_metadata, npu_ai_infra_attention_pioneer_metadata
        if npu_fused_infer_attention_sink_metadata is None:
            npu_fused_infer_attention_sink_metadata = CrossLayerSharedOp(
                op=torch.ops.custom._npu_fused_infer_attention_sink_metadata,
                shape=(1024,),
                dtype=torch.int32,
                callers=(
                    "decode", "decode_mla",
                    "prefill_absorb", "prefill_absorb_mla",
                    "prefill", "prefill_mla",
                ),
            )
        if npu_ai_infra_attention_pioneer_metadata is None and self.on_ascend950:
            npu_ai_infra_attention_pioneer_metadata = CrossLayerSharedOp(
                op=torch.ops.custom.npu_ai_infra_attention_pioneer_metadata,
                # A5 (Ascend950) pioneer FA-metadata is length 1024, not 2048.
                shape=(1024,) if self.on_ascend950 else (2048,),
                dtype=torch.int32,
                callers=(
                    "decode", "decode_mla",
                    "prefill_absorb", "prefill_absorb_mla",
                    "prefill", "prefill_mla",
                ),
            )

    def _calculate_page_size_padded(
        self,
        cache_config: CacheConfig,
        cache_dtype_str: str | None,
        config: DeepseekV2Config | DeepseekV3Config,
    ) -> int | None:
        """
        Calculate page_size_padded for alignment across different attention mechanisms.

        Alignment priority:
        1. If DSA exists: align to DSA page size
        2. Otherwise: align to max(MOME page size, MLA/SWA page size)

        Args:
            cache_config: Cache configuration
            cache_dtype_str: Quantization dtype string (e.g., "fp8_ds_mla", "hif8_ds_mla", "int8_ds_mla")
            config: Model configuration

        Returns:
            page_size_padded in bytes, or None if no padding needed
        """
        from vllm.utils.torch_utils import get_dtype_size
        from math import prod

        block_size = cache_config.block_size
        dtype = torch.bfloat16  # Default dtype
        dtype_size = get_dtype_size(dtype)

        # Calculate MLA/SWA page size
        mla_head_size = self.kv_lora_rank + self.qk_rope_head_dim
        mla_page_size = block_size * mla_head_size * dtype_size

        # Calculate DSA page size if DSA layer exists
        dsa_page_size = None
        if (getattr(config, "index_topk", None) or 0) > 0:
            index_head_dim = getattr(config, "index_head_dim", 0)
            if cache_dtype_str in ["fp8_ds_mla", "hif8_ds_mla"]:
                # Quant case: 512 fp8 + 64 bf16 + 4 fp32 + 128 int8 + 1 fp32
                # See DeepseekV3 quantized DSA format
                dsa_page_size = self.block_size_c8 * (656 + 128 + 4)
            elif cache_dtype_str == "int8_ds_mla":
                # Quant case: 512 int8 + 64 bf16 + 4 fp32 + 128 int8 + 1 bf16
                dsa_page_size = self.block_size_c8 * (656 + 128 + 2)
            elif cache_dtype_str == "li_int8_ds_mla":
                # Li-Quant-Only case: 576 bf16 + 128 int8 + 1 bf16
                dsa_page_size = block_size * (576 * 2 + 128 + 2)
            else:
                # Non-quant case: standard attention format
                dsa_page_size = block_size * (mla_head_size + index_head_dim) * dtype_size

        # Calculate MOME page size if MOME is enabled
        mome_page_size = None
        if self.use_mome:
            num_extra_token = 1 if self.is_pd_disagg else 0
            num_total_tokens = self.mome_kernel_width - 1 + \
                max(self.num_spec_tokens, num_extra_token)
            mome_page_size = sum(
                prod(shape) * get_dtype_size(dtype)
                for (shape, dtype) in zip(self.mome_state_shapes, self.mome_state_dtypes)
            ) * num_total_tokens

        required_page_sizes = [mla_page_size]
        if dsa_page_size is not None:
            required_page_sizes.append(dsa_page_size)
        if mome_page_size is not None:
            required_page_sizes.append(mome_page_size)

        target_page_size = max(required_page_sizes)

        return target_page_size

    def _install_q_b_split_loaders(self) -> None:
        """Wrap q_b_proj's per-param weight_loader so checkpoint loads also
        populate q_b_nope_proj / q_b_pe_proj with sliced PRE-PWAL data. Each
        child's own process_weights_after_loading then handles all quant-
        specific transforms (transpose, NZ cast, npu_trans_quant_param,
        dtype promotion, ...) the same way it would for q_b_proj itself.
        """
        nope_proj = self.q_b_nope_proj
        pe_proj = self.q_b_pe_proj
        qk_head_dim = self.qk_head_dim
        qk_nope_head_dim = self.qk_nope_head_dim

        def make_split_loader(orig_loader, nope_param, pe_param):
            def split_loader(param, loaded_weight):
                # First fill q_b_proj's param via its original loader (handles
                # TP sharding, dtype, etc. — leaves param.data in PRE-PWAL form).
                orig_loader(param, loaded_weight)
                data = param.data
                out_dim = data.shape[0]
                if out_dim % qk_head_dim != 0:
                    return
                local_heads = out_dim // qk_head_dim
                if data.dim() == 2:
                    d3 = data.view(local_heads, qk_head_dim, -1)
                    nope = d3[:, :qk_nope_head_dim, :].contiguous().view(-1, data.shape[-1])
                    pe = d3[:, qk_nope_head_dim:, :].contiguous().view(-1, data.shape[-1])
                elif data.dim() == 1:
                    d2 = data.view(local_heads, qk_head_dim)
                    nope = d2[:, :qk_nope_head_dim].contiguous().view(-1)
                    pe = d2[:, qk_nope_head_dim:].contiguous().view(-1)
                else:
                    return
                nope_param.data.copy_(nope)
                pe_param.data.copy_(pe)
            return split_loader

        for attr in ("weight", "weight_scale", "weight_offset"):
            src = getattr(self.q_b_proj, attr, None)
            nope_dst = getattr(nope_proj, attr, None)
            pe_dst = getattr(pe_proj, attr, None)
            if src is None or nope_dst is None or pe_dst is None:
                continue
            orig_loader = getattr(src, "weight_loader", None)
            if orig_loader is None:
                continue
            # Direct attribute set: set_weight_attrs asserts no overwrite, but
            # create_weights has already attached the original loader so we
            # need to replace it rather than add a new attribute.
            src.weight_loader = make_split_loader(orig_loader, nope_dst, pe_dst)

    def process_weights_after_loading(self) -> None:
        kv_b_proj_weight = self.kv_b_proj.weight.t().contiguous()
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank
        )
        w_uk, w_uv = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )
        self.W_UV = w_uv.transpose(1, 2).contiguous()
        self.W_UK_T = w_uk.contiguous()

        self.sink_k_nope = self.kv_a_layernorm(self.param_sink_compressed_kv).unsqueeze(1).contiguous()
        self.sink_k_pe = self.param_sink_k_pe.unsqueeze(1).contiguous()
        self.sink_kv = torch.cat([self.sink_k_nope, self.sink_k_pe], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.on_ascend950:
            # A5: dispatch through the registered custom op so the attention
            # (incl. the MLA prolog) stays opaque to Dynamo/cudagraph, matching
            # the pre-!1156 (last tested-good) behavior. !1156 un-wrapped this to
            # a plain call to enable MLA-prolog multi-stream, but that traces
            # prolog internals (slot_mapping_2d, FA metadata, pre_epilog_callback)
            # into the captured graph, which the cudagraph buffer-mutation guard
            # rejects. Non-950 keeps the plain call and the prolog multi-stream.
            return torch.ops.vllm.npu_pangu_forward(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                layer_name=self.layer_name,
            )
        return npu_pangu_forward(hidden_states, cos, sin, self.layer_name)

    def _prepare_phase_inputs(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        phase: str,
    ):
        num_decode_tokens = attn_metadata.num_decode_tokens

        if phase == "prefill":
            # first phase: backup originals
            attn_metadata.origin_slot_mapping = attn_metadata.slot_mapping.clone()
            attn_metadata.orig_num_actual_tokens = attn_metadata.num_actual_tokens
            num_actual_tokens = attn_metadata.num_actual_tokens

            sliced_hidden = hidden_states[num_decode_tokens:num_actual_tokens, ...]
            sliced_cos = cos[num_decode_tokens:num_actual_tokens, ...]
            sliced_sin = sin[num_decode_tokens:num_actual_tokens, ...]
            attn_metadata.prefill.slot_mapping = attn_metadata.origin_slot_mapping[num_decode_tokens:num_actual_tokens]
            attn_metadata.slot_mapping = attn_metadata.prefill.slot_mapping
            slot_mapping_2d = _get_slot_mapping_2d(attn_metadata)
            if slot_mapping_2d is not None:
                attn_metadata.origin_slot_mapping_2d = slot_mapping_2d
                attn_metadata.prefill.slot_mapping_2d = slot_mapping_2d[num_decode_tokens:num_actual_tokens]
                attn_metadata.slot_mapping_2d = attn_metadata.prefill.slot_mapping_2d
            attn_metadata.saved_decode = attn_metadata.decode
            attn_metadata.decode = None
            attn_metadata.num_actual_tokens = num_actual_tokens - num_decode_tokens
        else:
            saved_decode = getattr(attn_metadata, 'saved_decode', None)
            if saved_decode is not None:
                attn_metadata.decode = attn_metadata.saved_decode
            origin_slot_mapping = attn_metadata.origin_slot_mapping

            sliced_hidden = hidden_states[:num_decode_tokens, ...]
            sliced_cos = cos[:num_decode_tokens, ...]
            sliced_sin = sin[:num_decode_tokens, ...]
            attn_metadata.decode.slot_mapping = origin_slot_mapping[:num_decode_tokens]
            attn_metadata.slot_mapping = attn_metadata.decode.slot_mapping
            origin_slot_mapping_2d = getattr(attn_metadata, "origin_slot_mapping_2d", None)
            if origin_slot_mapping_2d is not None:
                attn_metadata.decode.slot_mapping_2d = origin_slot_mapping_2d[:num_decode_tokens]
                attn_metadata.slot_mapping_2d = attn_metadata.decode.slot_mapping_2d
            attn_metadata.saved_prefill = attn_metadata.prefill
            attn_metadata.prefill = None
            attn_metadata.num_actual_tokens = num_decode_tokens
        return sliced_hidden, sliced_cos, sliced_sin

    def _restore_phase_metadata(self, attn_metadata: MLACommonMetadata):
        saved_prefill = getattr(attn_metadata, 'saved_prefill', None)
        if saved_prefill is not None:
            attn_metadata.prefill = saved_prefill
        attn_metadata.slot_mapping = attn_metadata.origin_slot_mapping
        if getattr(attn_metadata, 'origin_slot_mapping_2d', None) is not None:
            attn_metadata.slot_mapping_2d = attn_metadata.origin_slot_mapping_2d
        attn_metadata.num_actual_tokens = attn_metadata.orig_num_actual_tokens

    def _forward_dummy(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        if self.tp_size > 1 and self.moe_comm_strategy != "allreduce":
            hidden_states = get_tp_group().all_gather(hidden_states, dim=0)

        attn_output = torch.zeros(
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if self.enable_flashcomm2 and not self.is_dsa_layer:
            # FlashComm2.0 dummy: simulate all_to_all + full o_proj
            x = attn_output.view(self.tp_size, -1, attn_output.shape[-1])
            output = torch.zeros_like(x)
            torch.distributed.all_to_all_single(output.flatten(), x.flatten(), group=get_tp_group().device_group)
            attn_output = output.transpose(0, 1).reshape(attn_output.shape[0] // self.tp_size, -1)

        if self.sharded_o_proj:
            self.o_proj.prefetch(torch.npu.current_stream())
        hidden_states = self._apply_o_proj(attn_output)

        if self.tp_size > 1:
            if self.enable_flashcomm2 and not self.is_dsa_layer:
                pass  # all_to_all + full o_proj already complete
            elif self.moe_comm_strategy == "allreduce":
                hidden_states = get_tp_group().all_reduce(hidden_states)
            else:
                hidden_states = get_tp_group().reduce_scatter(hidden_states, dim=0)

        return hidden_states

    def _forward_decode(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:

        # with torch.npu.npugraph_ex.scope.limit_core_num(8,8):
        q_nope, q_pe, kv_cache, topk_indices = self._mla_prolog(
            hidden_states,
            cos,
            sin,
            attn_metadata,
            mome_metadata,
        )

        if self.is_dsa_layer:
            attn_output = self._apply_DSA_attention(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_cache=kv_cache,
                topk_indices=topk_indices,
                attn_metadata=attn_metadata,
            )
        else:
            # with torch.npu.npugraph_ex.scope.limit_core_num(8,8):
            attn_output = torch.ops.vllm.npu_pangu_swa_decode(
                q_nope, q_pe, kv_cache[0], kv_cache[1], self.prefix,
            )

        # Fire one-shot pre-epilog hook so the caller can enqueue side-stream
        # work that overlaps with v_up / o_proj / all_reduce inside the epilog.
        if self.pre_epilog_callback is not None:
            cb = self.pre_epilog_callback
            self.pre_epilog_callback = None
            cb()

        # with torch.npu.npugraph_ex.scope.limit_core_num(8,8):
        res = self._mla_epilog(attn_output, attn_metadata, mome_metadata)
        return res

    @attn_decorator(type="mome")
    def _apply_MOME(
        self,
        x: torch.Tensor,
        layer: ColumnParallelMOME, 
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
        inplace: bool = False, 
        ena_sp: bool = False,
    ):
        if attn_metadata is None or mome_metadata is None:
            # warm up run
            return x

        kv_cache = self.mome_attn.kv_cache
        kv_index = layer.mome_cache_index
        if not self.on_ascend950:
            if ena_sp:
                init_idx, save_idx, meta = mome_metadata.conv_sp_meta
                return conv_sp(
                    x,
                    layer.weight,
                    kv_cache[kv_index],
                    init_idx,
                    save_idx,
                    meta,
                    inplace,
                )

            x = torch.ops.vllm.npu_pangu_mome_conv(
                x, layer.weight, kv_cache[kv_index],
                mome_metadata.query_start_loc,
                cache_indices=mome_metadata.cache_indices,
                num_accepted_tokens=mome_metadata.num_accepted_tokens,
                num_computed_tokens=mome_metadata.num_computed_tokens,
                block_idx_first_scheduled_token=mome_metadata.block_idx_first_scheduled_token,
                block_idx_last_scheduled_token=mome_metadata.block_idx_last_scheduled_token,
                initial_state_idx=mome_metadata.block_idx_last_computed_token,
                pad_slot_id=mome_metadata.pad_slot_id,
                max_query_len=mome_metadata.max_query_len,
                block_size=mome_metadata.B_size,
                mode=1,
                inplace=inplace,
            )
        else:
            # A5: the new fused mome kernel resolves prefill / decode / mixed
            # from the metadata in a single forward() call.
            x = layer.forward(x, kv_cache[kv_index], mome_metadata, inplace=inplace)

        return x

    def _apply_SWA_attention_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        num_actual_tokens = attn_metadata.decode.num_tokens
        num_tokens = q_nope.size(0)
        kwargs = {
            "query": q_nope[:num_actual_tokens],
            "key": kv_cache[0],
            "value": kv_cache[0],
            "query_rope": q_pe[:num_actual_tokens],
            "key_rope": kv_cache[1],
            "num_key_value_heads": 1,
            "input_layout": "TND_NTD",
            "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
            "sparse_mode": 4,
            "pre_tokens": self.sliding_window-1,
            "next_tokens": 0,
            "block_table": attn_metadata.decode.block_table,
            "block_size": self.block_size,
            "key_sink": self.sink_k_nope,
            "value_sink": self.sink_k_nope,
            "key_rope_sink": self.sink_k_pe,
        }

        if self.on_ascend950:
            if self.use_aicpu_fa_tiling:
                query_cumlens = attn_metadata.decode.query_cumlens.to(torch.int64)
                seq_lens = attn_metadata.decode.seq_lens.to(torch.int64)
                fia_meta_args = {
                    "num_heads_q": self.num_local_heads,
                    "num_heads_kv": 1,
                    "head_dim_qk": q_nope.shape[-1],
                    "head_dim_v": kv_cache[0].shape[-1],
                    "actual_seq_lengths": query_cumlens,
                    "actual_seq_lengths_kv": seq_lens,
                    "batch_size": attn_metadata.decode.block_table.shape[0],
                    "sparse_mode": 4,
                    "pre_tokens": self.sliding_window - 1,
                    "next_tokens": 0,
                    "input_layout": "TND",
                    "sink_number": self.param_sink_number,
                    "rope_head_dim": q_pe.shape[-1],
                    "block_size": self.block_size,
                }
                meta_data = npu_ai_infra_attention_pioneer_metadata(
                    fia_meta_args,
                    self.is_fa_metadata_producer,
                    "decode" + self._fa_meta_suffix,
                )
                kwargs.update({
                    "metaData": meta_data,
                    "num_heads": self.num_local_heads,
                    "actual_seq_lengths": query_cumlens,
                    "actual_seq_lengths_kv": seq_lens,
                    "softmax_scale": self.scaling,
                })
                forward_context = get_forward_context()
                if num_actual_tokens == num_tokens:
                    attn_output = torch.ops.custom.npu_ai_infra_attention_pioneer(**kwargs)[0]
                else:
                    attn_output_shape = [self.num_local_heads, num_tokens, self.kv_lora_rank]
                    attn_output = torch.zeros(attn_output_shape, device=q_nope.device, dtype=q_nope.dtype)
                    attn_output[:, :num_actual_tokens] = torch.ops.custom.npu_ai_infra_attention_pioneer(**kwargs)[0]
            else:
                kwargs.update({
                    "num_heads": self.num_local_heads,
                    "actual_seq_lengths": attn_metadata.decode.query_cumlens,
                    "actual_seq_lengths_kv": attn_metadata.decode.seq_lens,
                    "scale": self.scaling,
                })
                attn_output_shape = [self.num_local_heads, num_tokens, self.kv_lora_rank]
                attn_output = torch.zeros(attn_output_shape, device=q_nope.device, dtype=q_nope.dtype)
                softmax_lse = torch.zeros(
                    (num_tokens, self.num_local_heads, 1),
                    device=q_nope.device,
                    dtype=torch.float32,
                )
                forward_context = get_forward_context()
                if forward_context.capturing:
                    capture_graph_task(
                        op_desc=OP_FIA_PIONEER,
                        op_kwargs=kwargs,
                        out_tensors=[attn_output, softmax_lse],
                        num_tokens=num_tokens,
                        layer_name=self.attn.layer_name,
                    )
                else:
                    attn_output[:, :num_actual_tokens] = torch_npu._npu_attention_pioneer(**kwargs)[0]
        elif self.use_aicpu_fa_tiling:
            cur_stream = torch.npu.current_stream()
            stream_limit = torch.npu.get_stream_limit(cur_stream)
            query_cumlens = attn_metadata.decode.query_cumlens.to(torch.int64)
            seq_lens = attn_metadata.decode.seq_lens.to(torch.int64)
            meta_data_args = {
                "num_heads_q": self.num_local_heads,
                "num_heads_kv": 1,
                "head_dim_qk": q_nope.shape[-1],
                "head_dim_v": kv_cache[0].shape[-1],
                "actual_seq_lengths": query_cumlens,
                "actual_seq_lengths_kv": seq_lens,
                "sparse_mode": 4,
                "pre_tokens": self.sliding_window - 1,
                "next_tokens": 0,
                "input_layout": "TND",
                "input_layout_kv": "BnBsH",
                "rope_head_dim": q_pe.shape[-1],
                "k_sink_num": self.param_sink_number,
                "block_size": self.block_size,
                "aic_core_num": stream_limit["cube_core_num"],
                "aiv_core_num": stream_limit["vector_core_num"],
            }
            meta_data = npu_fused_infer_attention_sink_metadata(
                meta_data_args,
                self.is_fa_metadata_producer,
                "decode" + self._fa_meta_suffix,
            )
            kwargs.update({
                "num_query_heads": self.num_local_heads,
                "actual_seq_qlen": query_cumlens,
                "actual_seq_kvlen": seq_lens,
                "softmax_scale": self.scaling,
                "meta_data": meta_data,
            })
            if num_actual_tokens == num_tokens:
                attn_output = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]
            else:
                attn_output_shape = [self.num_local_heads, num_tokens, self.kv_lora_rank]
                attn_output = torch.zeros(attn_output_shape, device=q_nope.device, dtype=q_nope.dtype)
                attn_output[:, :num_actual_tokens] = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]
        else:
            kwargs.update({
                "num_query_heads": self.num_local_heads,
                "actual_seq_qlen": attn_metadata.decode.query_cumlens,
                "actual_seq_kvlen": attn_metadata.decode.seq_lens,
                "softmax_scale": self.scaling,
            })
            attn_output_shape = [self.num_local_heads, num_tokens, self.kv_lora_rank]
            attn_output = torch.zeros(attn_output_shape, device=q_nope.device, dtype=q_nope.dtype)
            softmax_lse = torch.zeros((num_tokens, self.num_local_heads, 1), device=q_nope.device, dtype=torch.float32)
            forward_context = get_forward_context()
            if forward_context.capturing:
                capture_graph_task(
                    op_desc=OP_FIA_SINK,
                    op_kwargs=kwargs,
                    out_tensors=[attn_output, softmax_lse],
                    num_tokens=num_tokens,
                    layer_name=self.attn.layer_name,
                )
            else:
                attn_output[:, :num_actual_tokens] = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]

        # Defer v_up (W_UV absorb) to _mla_epilog so that pre_epilog_callback's
        # side-stream work can overlap with it. Return latent [T, N, L].
        return attn_output.view(self.num_local_heads, -1, self.kv_lora_rank) \
                          .transpose(0, 1) \
                          .contiguous()

    @attn_decorator(type="dsa")
    def _apply_DSA_attention_cp(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor,
        sp_manager: Optional[SPManager] = None,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:
        actual_seq_lengths_query, actual_seq_lengths_kv, _, block_table = sp_manager.cp_attn_meta()

        q = torch.cat([q_nope, q_pe], dim=-1)

        if self.cache_config.cache_dtype in ["int8_ds_mla"]:
            attn_output = torch.ops.custom.npu_ai_infra_kv_quant_sparse_flash_attention(
                query=q,
                key=kv_cache[0].unsqueeze(2),
                value=kv_cache[0].unsqueeze(2),
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                sparse_block_size=1,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                key_sink=self.sink_kv,
                value_sink=self.sink_k_nope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                block_table=block_table,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                rope_head_dim=64,
            )
        else:
            attn_output = torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(
                query=q,
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                pre_tokens=(1<<63)-1,
                next_tokens=(1<<63)-1,
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                key_sink=self.sink_kv,
                value_sink=self.sink_k_nope,
            )[0]

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        attn_output = (
            torch_npu.npu_transpose_batchmatmul(attn_output, self.W_UV, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
                .reshape(-1, self.num_local_heads * self.v_head_dim)
        )

        return attn_output

    def _forward_prefill_cp(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:
        sp_manager: SPManager = (
            attn_metadata.prefill.sp_manager
            if attn_metadata is not None
            else DummySPManager(get_tp_group()))
        cos = cos[:attn_metadata.num_actual_tokens]
        sin = sin[:attn_metadata.num_actual_tokens]

        sp_cos = sp_manager.slice_tokens(cos, cached="cos")
        sp_sin = sp_manager.slice_tokens(sin, cached="sin")
        cp_cos = sp_manager.cp_slice(cos, cached="cos")
        cp_sin = sp_manager.cp_slice(sin, cached="sin")
        sp_x = hidden_states

        ### Q stream begins ###
        q_lora = self.q_a_proj(sp_x)

        if self.use_mome:
            if self.enable_mome_sp:
                self._apply_MOME(q_lora, self.qa_conv, attn_metadata, mome_metadata, True, True)
                q_lora = sp_manager.sp_to_cp(q_lora)
            else:
                q_lora = sp_manager.ag_tokens(q_lora)
                q_lora = self._apply_MOME(q_lora, self.qa_conv, attn_metadata, mome_metadata)
                q_lora = sp_manager.cp_slice(q_lora)
        else:
            q_lora = sp_manager.sp_to_cp(q_lora)

        q_lora = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_lora)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim)
        q_nope = (
            torch_npu.npu_transpose_batchmatmul(q_nope, self.W_UK_T, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
                .reshape(-1, self.num_local_heads, self.kv_lora_rank)
        )

        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.num_local_heads, self.qk_rope_head_dim),
            cp_cos.view(-1, 1, 1, self.qk_rope_head_dim),
            cp_sin.view(-1, 1, 1, self.qk_rope_head_dim),
            rotary_mode="half" if not self.rope_interleave else "interleave",
        ).squeeze(1)
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ### Q stream ends ###

        kv_cache = self.attn.kv_cache
        if self.skip_topk:
            topk_indices = self._get_topk_indices(attn_metadata)
        else:
            ### Indexer stream begins ###
            topk_indices = self.indexer.forward_cp(
                sp_x,
                q_lora,
                sp_cos,
                sp_sin,
                cp_cos,
                cp_sin,
                sp_manager,
                attn_metadata,
                kv_cache,
            )
            self._set_topk_indices(attn_metadata, topk_indices)
            ### Indexer stream ends ###

        ### KV stream begins ###
        kv = self.kv_a_proj_with_mqa(sp_x)
        if not self.enable_mome_sp:
            kv = sp_manager.ag_tokens(kv)
            if self.use_mome_inplace_update:
                self._apply_MOME(
                    kv[:, :self.kv_lora_rank],
                    self.compresskv_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                    inplace=True,
                )
            else:
                k_nope, k_pe = torch.split(
                    kv,
                    [self.kv_lora_rank, self.qk_rope_head_dim],
                    dim=-1,
                )
                k_nope = self._apply_MOME(
                    k_nope,
                    self.compresskv_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                )
                kv = torch.cat([k_nope, k_pe], dim=-1)
        else:
            self._apply_MOME(
                kv[:, :self.kv_lora_rank],
                self.compresskv_conv,
                attn_metadata=attn_metadata,
                mome_metadata=mome_metadata,
                inplace=True,
                ena_sp=True,
            )
            kv = sp_manager.ag_tokens(kv)

        kwargs = self._kv_rmsnorm_rope_cache_v2_kwargs(
            kv, cos, sin, attn_metadata,
            k_cache=None,
            ckv_cache=kv_cache[0].unsqueeze(2),
        )

        if self.cache_config.cache_dtype in ["int8_ds_mla"]:
            kwargs.update({
                "quant_mode": "pertile128",
            })

        k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)
        ### KV stream ends ###

        if self.sharded_o_proj:
            cur_stream = torch.npu.current_stream()
            prefetch_stream = named_stream("pangu_dsa_o_proj_prefetch")
            prefetch_stream.wait_stream(cur_stream)
            with torch.npu.stream(prefetch_stream):
                self.o_proj.prefetch(prefetch_stream)

        attn_output = self._apply_DSA_attention_cp(
            q_nope,
            q_pe,
            kv_cache,
            topk_indices,
            sp_manager,
            attn_metadata=attn_metadata,
        )

        if self.use_mome:
            if self.enable_mome_sp:
                attn_output = sp_manager.cp_to_sp(attn_output)
                self._apply_MOME(
                    attn_output,
                    self.o_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                    inplace=True,
                    ena_sp=True,
                )
                return self._apply_o_proj(attn_output)
            else:
                attn_output = sp_manager.cp_to_sp(attn_output)
                attn_output = sp_manager.ag_tokens(attn_output)
                attn_output = self._apply_MOME(
                    attn_output,
                    self.o_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                )
                if self.o_proj.tp_size == 1:
                    attn_output = sp_manager.slice_tokens(attn_output)
                return self._apply_o_proj(attn_output)

        hidden_states = self._apply_o_proj(attn_output)
        return sp_manager.cp_to_sp(hidden_states)

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:
        
        if self.is_dsa_layer:
            q_nope, q_pe, kv_cache, topk_indices = self._mla_prolog(
                hidden_states,
                cos,
                sin,
                attn_metadata,
                mome_metadata,
            )
            attn_output = self._apply_DSA_attention(
                q_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata=attn_metadata,
            )
        else:
            mla_output = self._mla_prolog(
                hidden_states,
                cos,
                sin,
                attn_metadata,
                mome_metadata,
            )
            if (
                self.first_chunk_pa
                or getattr(attn_metadata.prefill, "chunked_context", None)
                is not None
            ):
                q_nope, q_pe, kv_cache, _ = mla_output
                attn_output = self._apply_SWA_attention_prefill_absorb(
                    q_nope,
                    q_pe,
                    kv_cache,
                    attn_metadata=attn_metadata,
                )
            else:
                q_nope, q_pe, k_up_nope, k_pe, v_up = mla_output
                attn_output = self._apply_SWA_attention_prefill(
                    q_nope,
                    q_pe,
                    k_up_nope,
                    k_pe,
                    v_up,
                    attn_metadata=attn_metadata,
                )

        if self.sharded_o_proj:
            cur_stream = torch.npu.current_stream()
            prefetch_stream = named_stream("pangu_o_proj_prefetch")
            prefetch_stream.wait_stream(cur_stream)
            with torch.npu.stream(prefetch_stream):
                self.o_proj.prefetch(prefetch_stream)

        return self._mla_epilog(attn_output, attn_metadata, mome_metadata)

    def _forward_prefill_sp(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:
        """SWA prefill with contiguous token SP and replicated head weights."""
        assert not self.on_ascend950, "ena_swa_attn_seq_parallel is supported on A3 only"
        assert attn_metadata is not None and attn_metadata.prefill is not None
        sp_manager = attn_metadata.prefill.sp_manager
        assert sp_manager is not None, "SWA SP metadata is missing SPManager"
        assert hidden_states.size(0) == sp_manager.sp_len

        num_actual_tokens = attn_metadata.num_actual_tokens
        full_cos = cos[:num_actual_tokens]
        full_sin = sin[:num_actual_tokens]
        local_cos = sp_manager.slice_tokens(full_cos, cached="swa_cos")
        local_sin = sp_manager.slice_tokens(full_sin, cached="swa_sin")

        # Q remains token-sharded and uses all replicated heads.
        q_lora = self.q_a_proj(hidden_states)
        if self.use_mome:
            if self.enable_mome_sp:
                self._apply_MOME(q_lora, self.qa_conv, attn_metadata, mome_metadata, True, True)
            else:
                q_lora = sp_manager.ag_tokens(q_lora)
                q_lora = self._apply_MOME(q_lora, self.qa_conv, attn_metadata, mome_metadata)
                q_lora = sp_manager.slice_tokens(q_lora)
        q_lora = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_lora)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = self._q_rope(q_pe, local_cos, local_sin)
        q_nope = self._w_uk_t_absorb(q_nope)

        # KV is projected locally, then gathered so every rank writes the same
        # full-token cache with the global slot mapping.
        kv = self.kv_a_proj_with_mqa(hidden_states)
        gather_before_mome = self.use_mome and not self.enable_mome_sp
        if gather_before_mome:
            kv = sp_manager.ag_tokens(kv)
        if self.use_mome:
            if self.use_mome_inplace_update:
                self._apply_MOME(
                    kv[:, :self.kv_lora_rank],
                    self.compresskv_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                    inplace=True,
                    ena_sp=self.enable_mome_sp,
                )
            else:
                k_nope, k_pe = torch.split(
                    kv,
                    [self.kv_lora_rank, self.qk_rope_head_dim],
                    dim=-1,
                )
                k_nope = self._apply_MOME(
                    k_nope,
                    self.compresskv_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                    ena_sp=self.enable_mome_sp,
                )
                kv = torch.cat([k_nope, k_pe], dim=-1)
        if not gather_before_mome:
            kv = sp_manager.ag_tokens(kv)

        kv_cache = self.attn.kv_cache
        kv_result = self._npu_kvrmsnorm_rope_cache(
            kv,
            kv_cache,
            full_cos,
            full_sin,
            attn_metadata,
            None,
        )

        if self.sharded_o_proj:
            cur_stream = torch.npu.current_stream()
            prefetch_stream = named_stream("pangu_o_proj_prefetch")
            prefetch_stream.wait_stream(cur_stream)
            with torch.npu.stream(prefetch_stream):
                self.o_proj.prefetch(prefetch_stream)

        attn_output = self._apply_SWA_attention_prefill_absorb(
            q_nope,
            q_pe,
            kv_result[0],
            attn_metadata=attn_metadata,
            sp_manager=sp_manager,
        )

        if sp_manager.valid_token_count < sp_manager.sp_len:
            padded_output = hidden_states.new_zeros(
                sp_manager.sp_len, self.num_heads * self.v_head_dim
            )
            padded_output[:sp_manager.valid_token_count] = attn_output
            attn_output = padded_output

        if self.use_mome:
            if self.enable_mome_sp:
                self._apply_MOME(
                    attn_output,
                    self.o_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                    inplace=True,
                    ena_sp=True,
                )
            else:
                attn_output = sp_manager.ag_tokens(attn_output)
                attn_output = self._apply_MOME(
                    attn_output,
                    self.o_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                )
                attn_output = sp_manager.slice_tokens(attn_output)

        return self._apply_o_proj(attn_output)

    def _forward_prefill_FC2(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:
        """FlashComm2.0 prefill path: MLA prolog -> SWA attention -> FC2 epilog.

        FC2 optimization: hidden_states arrives ungathered (TP-local).
        We project locally, then all_gather the smaller q_lora / kv tensors
        and trim to num_actual_tokens to remove TP padding.
        cos / sin are already global and need no gathering.
        """
        enable_pa = (
            self.first_chunk_pa or
            getattr(attn_metadata.prefill, "chunked_context", None) is not None
        )

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = num_actual_tokens - num_decode_tokens

        # cos/sin are global tensors — slice to prefill range directly
        prefill_cos = cos[num_decode_tokens:num_actual_tokens]
        prefill_sin = sin[num_decode_tokens:num_actual_tokens]

        # hidden_states is TP-local (ungathered); compute total padded size
        local_tokens = hidden_states.shape[0]
        total_padded_tokens = local_tokens * self.tp_size

        # get KV cache for this layer
        kv_cache = self.attn.kv_cache

        need_all_gather = self.tp_size > 1 and self.moe_comm_strategy != "allreduce"

        ### Q stream begins ###
        # Project, apply MoME and layer norm on local tokens, then gather all tokens
        q_lora = self.q_a_proj(hidden_states)
        if not self.enable_mome_sp:
            if need_all_gather:
                q_lora = get_tp_group().all_gather(q_lora, dim=0)
            q_lora = q_lora[:num_prefill_tokens] # Trim TP padding to keep only actual prefill tokens
            if self.use_mome:
                q_lora = self._apply_MOME(
                    q_lora,
                    self.qa_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                )
            q_lora = self.q_a_layernorm(q_lora)
        else:
            self._apply_MOME(
                q_lora,
                self.qa_conv,
                attn_metadata=attn_metadata,
                mome_metadata=mome_metadata,
                inplace=True,
                ena_sp=True,
            )
            q_lora = self.q_a_layernorm(q_lora)
            if need_all_gather:
                q_lora = get_tp_group().all_gather(q_lora, dim=0)
            q_lora = q_lora[:num_prefill_tokens] # Trim padding tokens after all-gather

        q = self.q_b_proj(q_lora)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.num_local_heads, self.qk_rope_head_dim),
            prefill_cos.view(-1, 1, 1, self.qk_rope_head_dim),
            prefill_sin.view(-1, 1, 1, self.qk_rope_head_dim),
            rotary_mode="half" if not self.rope_interleave else "interleave",
        ).squeeze(1)
        if enable_pa:
            q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim)            
            q_nope = (
                torch_npu.npu_transpose_batchmatmul(q_nope, self.W_UK_T, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
                    .reshape(-1, self.num_local_heads, self.kv_lora_rank)
            )
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ### Q stream ends ###

        ### KV stream begins ###
        # Project, apply MoME and layer norm on local tokens, then gather the smaller kv tensor
        kv = self.kv_a_proj_with_mqa(hidden_states)
        if not self.enable_mome_sp:
            if need_all_gather:
                kv = get_tp_group().all_gather(kv, dim=0)
            kv = kv[:num_prefill_tokens] # Trim TP padding to keep only actual prefill tokens
            if self.use_mome:
                if self.use_mome_inplace_update:
                    self._apply_MOME(
                        kv[:, :self.kv_lora_rank],
                        self.compresskv_conv,
                        attn_metadata=attn_metadata,
                        mome_metadata=mome_metadata,
                        inplace=True, 
                    )
                else:
                    k_nope, k_pe = torch.split(
                        kv,
                        [self.kv_lora_rank, self.qk_rope_head_dim],
                        dim=-1,
                    )
                    k_nope = self._apply_MOME(
                        k_nope, 
                        self.compresskv_conv, 
                        attn_metadata=attn_metadata, 
                        mome_metadata=mome_metadata, 
                        inplace=False, 
                    )
                    kv = torch.cat([k_nope, k_pe], dim=-1)
        else:
            self._apply_MOME(
                kv[:, :self.kv_lora_rank],
                self.compresskv_conv,
                attn_metadata=attn_metadata,
                mome_metadata=mome_metadata,
                inplace=True,
                ena_sp=True,
            )
            kv = get_tp_group().all_gather(kv, dim=0)[:num_prefill_tokens]

        kv_rmsnorm_result = self._npu_kvrmsnorm_rope_cache(
            kv,
            kv_cache,
            prefill_cos,
            prefill_sin,
            attn_metadata,
            None,
        )
        ### KV stream ends ###

        # --- SWA Attention ---
        if enable_pa:
            kv_cache, _ = kv_rmsnorm_result
            attn_output = self._apply_SWA_attention_prefill_absorb(
                q_nope,
                q_pe,
                kv_cache,
                attn_metadata=attn_metadata,
            )
        else:
            k_up_nope, k_pe, v_up = kv_rmsnorm_result
            attn_output = self._apply_SWA_attention_prefill(
                q_nope,
                q_pe,
                k_up_nope,
                k_pe,
                v_up,
                attn_metadata=attn_metadata,
            )

        if self.sharded_o_proj:
            cur_stream = torch.npu.current_stream()
            prefetch_stream = named_stream("pangu_o_proj_prefetch")
            prefetch_stream.wait_stream(cur_stream)
            with torch.npu.stream(prefetch_stream):
                self.o_proj.prefetch(prefetch_stream)

        # We convert from TP to SP in the following block and,
        # we have different ways to do this depending on whether MoME+SP is turned on.
        # Note that the MoME has all the heads in its weight.
        if not self.enable_mome_sp:
            # Path 1: all-gather the heads -> MoME -> slice out the local tokens
            tp_group = get_tp_group()
            if attn_output.size(1) < self.o_conv.input_size_per_partition:
                attn_output = tp_group.all_gather(attn_output, dim=1)
            if self.use_mome:
                attn_output = self._apply_MOME(
                    attn_output,
                    self.o_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                )
            if need_all_gather:
                tp_rank = tp_group.rank_in_group
                attn_output = attn_output[tp_rank * local_tokens: min((tp_rank+1) * local_tokens, num_actual_tokens)]
                if attn_output.size(0) < local_tokens:
                    attn_output = F.pad(attn_output, (0, 0, 0, local_tokens - attn_output.size(0)))
            # NOTE: if not self.use_mome, one could in theory use all-to-all
            # in the place of (all-gather on heads) + (slice on tokens)
        else:
            # Path 2: Use all-to-all
            attn_output = F.pad(attn_output, (0, 0, num_decode_tokens, total_padded_tokens - num_actual_tokens))
            attn_output = attn_output.view(self.tp_size, -1, attn_output.shape[-1])
            output = torch.zeros_like(attn_output)
            # all_to_all: [tp_size, N_local, local_dim] -> [N_local, num_heads * v_dim]
            torch.distributed.all_to_all_single(
                output.flatten(), attn_output.flatten(),
                group=get_tp_group().device_group
            )
            attn_output = output.transpose(0, 1).reshape(local_tokens, self.num_heads * self.v_head_dim)

            # --- FC2 MLA Epilog ---
            self._apply_MOME(
                attn_output,
                self.o_conv,
                attn_metadata=attn_metadata,
                mome_metadata=mome_metadata,
                inplace=True,
                ena_sp=True,
            )

        return self._apply_o_proj(attn_output)

    @attn_decorator(type="mla")
    def _apply_SWA_attention_prefill_absorb(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Optional[MLACommonMetadata] = None,
        sp_manager: Optional[SPManager] = None,
    ) -> torch.Tensor:
        assert attn_metadata is not None and attn_metadata.prefill is not None

        if sp_manager is not None:
            assert not self.on_ascend950, "ena_swa_attn_seq_parallel is supported on A3 only"
            query_cumlens, seq_lens, block_table = sp_manager.sp_attn_meta()
            num_tokens = sp_manager.valid_token_count
            if num_tokens == 0:
                # Empty SP shard: no tokens to attend. Return early WITHOUT
                # launching kernels — but this function must still be CALLED on
                # every rank so the @attn_decorator plugin hooks (D2H offload /
                # H2D prefetch, the latter containing a TP-group AllGather) fire
                # uniformly and the TP communicator stays in lockstep.
                return q_nope.new_empty((0, self.num_heads * self.v_head_dim))
        else:
            query_cumlens = attn_metadata.prefill.query_cumlens
            seq_lens = attn_metadata.prefill.seq_lens
            block_table = attn_metadata.prefill.block_table
            num_tokens = attn_metadata.num_actual_tokens

        kwargs = {
            "query": q_nope[:num_tokens],
            "key": kv_cache[0],
            "value": kv_cache[0],
            "query_rope": q_pe[:num_tokens],
            "key_rope": kv_cache[1],
            "num_key_value_heads": 1,
            "input_layout": "TND_NTD",
            "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
            "sparse_mode": 4,
            "pre_tokens": self.sliding_window - 1,
            "next_tokens": 0,
            "block_table": block_table,
            "block_size": self.block_size,
            "key_sink": self.sink_k_nope,
            "value_sink": self.sink_k_nope,
            "key_rope_sink": self.sink_k_pe,
        }

        if self.on_ascend950:
            kwargs.update({
                "num_heads": self.num_local_heads,
                "actual_seq_lengths": query_cumlens,
                "actual_seq_lengths_kv": seq_lens,
                "scale": self.scaling,
            })
            attn_output = torch_npu._npu_attention_pioneer(**kwargs)[0]
        elif self.use_aicpu_fa_tiling:
            query_cumlens = query_cumlens.to(torch.int64)
            seq_lens = seq_lens.to(torch.int64)
            meta_data_args = {
                "num_heads_q": self.num_local_heads,
                "num_heads_kv": 1,
                "head_dim_qk": q_nope.shape[-1],
                "head_dim_v": kv_cache[0].shape[-1],
                "actual_seq_lengths": query_cumlens,
                "actual_seq_lengths_kv": seq_lens,
                "sparse_mode": 4,
                "pre_tokens": self.sliding_window - 1,
                "next_tokens": 0,
                "input_layout": "TND",
                "input_layout_kv": "BnBsH",
                "rope_head_dim": q_pe.shape[-1],
                "k_sink_num": self.param_sink_number,
                "block_size": self.block_size,
            }
            meta_data = npu_fused_infer_attention_sink_metadata(
                meta_data_args,
                self.is_fa_metadata_producer,
                "prefill_absorb" + self._fa_meta_suffix,
            )
            kwargs.update({
                "num_query_heads": self.num_local_heads,
                "actual_seq_qlen": query_cumlens,
                "actual_seq_kvlen": seq_lens,
                "softmax_scale": self.scaling,
                "meta_data": meta_data,
            })
            attn_output = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]
        else:
            kwargs.update({
                "num_query_heads": self.num_local_heads,
                "actual_seq_qlen": query_cumlens,
                "actual_seq_kvlen": seq_lens,
                "softmax_scale": self.scaling,
            })
            attn_output = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]

        attn_output = (
            torch_npu.npu_transpose_batchmatmul(
                attn_output, self.W_UV, perm_y=(1, 0, 2)
            ).reshape(-1, self.num_local_heads * self.v_head_dim)
        )

        return attn_output

    @attn_decorator(type="mla")
    def _apply_SWA_attention_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:
        sink_kv = self.kv_b_proj(
            self.kv_a_layernorm(self.param_sink_compressed_kv)
        )
        sink_k_nope, sink_v = torch.split(
            sink_kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim),
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        sink_k_pe = self.param_sink_k_pe.view(-1, 1, self.qk_rope_head_dim) \
                                        .repeat(1, self.num_local_heads, 1)

        # Note:
        # Currently, attn_metadata.prefill.seq_lens is constructed as the "true" sequence lengths.
        # We pass actual_seq_lengths_kv=query_cumlens instead, as the ops need cumulative sum.
        # Need to fix this when chunked prefill or prefix caching is enabled.

        if self.on_ascend950:
            if self.use_aicpu_fa_tiling:
                query = torch.cat([q_nope, q_pe], dim=-1)
                key = torch.cat([k_nope, k_pe], dim=-1)
                sink_key = torch.cat([sink_k_nope, sink_k_pe], dim=-1)
                query_cumlens = attn_metadata.prefill.query_cumlens.to(torch.int64)
                fia_meta_args = {
                    "num_heads_q": self.num_local_heads,
                    "num_heads_kv": self.num_local_heads,
                    "head_dim_qk": query.shape[-1],
                    "head_dim_v": v.shape[-1],
                    "actual_seq_lengths": query_cumlens,
                    "actual_seq_lengths_kv": query_cumlens,
                    "batch_size": query_cumlens.shape[0],
                    "sparse_mode": 4,
                    "pre_tokens": self.sliding_window - 1,
                    "next_tokens": 0,
                    "input_layout": "TND",
                    "sink_number": sink_k_nope.shape[0],
                    "rope_head_dim": q_pe.shape[-1],
                    "block_size": 0,
                    "soc_version": "ascend950",
                }

                meta_data = npu_ai_infra_attention_pioneer_metadata(
                    fia_meta_args,
                    self.is_fa_metadata_producer,
                    "prefill" + self._fa_meta_suffix,
                )
                kwargs = {
                    "query": query.contiguous(),
                    "key": key.contiguous(),
                    "value": v.contiguous(),
                    "metaData": meta_data,
                    "actual_seq_lengths": query_cumlens,
                    "actual_seq_lengths_kv": query_cumlens,
                    "num_heads": self.num_local_heads,
                    "num_key_value_heads": self.num_local_heads,
                    "input_layout": "TND",
                    "softmax_scale": self.scaling,
                    "sparse_mode": 4,
                    "pre_tokens": self.sliding_window-1,
                    "next_tokens": 0,
                    "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
                    "softmax_lse_flag": False,
                    "key_sink": sink_key.contiguous(),
                    "value_sink": sink_v.contiguous(),
                }
                attn_output = torch.ops.custom.npu_ai_infra_attention_pioneer(**kwargs)[0]
            else:
                query = torch.cat([q_nope, q_pe], dim=-1)
                key = torch.cat([k_nope, k_pe], dim=-1)
                sink_key = torch.cat([sink_k_nope, sink_k_pe], dim=-1)
                kwargs = {
                    "query": query,
                    "key": key,
                    "value": v,
                    "actual_seq_lengths": attn_metadata.prefill.query_cumlens,
                    "actual_seq_lengths_kv": attn_metadata.prefill.query_cumlens,
                    "num_heads": self.num_local_heads,
                    "num_key_value_heads": self.num_local_heads,
                    "input_layout": "TND",
                    "scale": self.scaling,
                    "sparse_mode": 4,
                    "pre_tokens": self.sliding_window-1,
                    "next_tokens": 0,
                    "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
                    "softmax_lse_flag": False,
                    "key_sink": sink_key,
                    "value_sink": sink_v,
                }
                attn_output = torch_npu._npu_attention_pioneer(**kwargs)[0]
        else:
            kwargs = {
                "query": q_nope,
                "key": k_nope,
                "value": v,
                "query_rope": q_pe,
                "key_rope": k_pe,
                "num_query_heads": self.num_local_heads,
                "num_key_value_heads": self.num_local_heads,
                "input_layout": "TND",
                "atten_mask": self.attn.impl.SHARE_MASK_TRIL_SPARSE,
                "sparse_mode": 4,
                "softmax_scale": self.scaling,
                "pre_tokens": self.sliding_window-1,
                "next_tokens": 0,
                "actual_seq_qlen": attn_metadata.prefill.query_cumlens,
                "actual_seq_kvlen": attn_metadata.prefill.query_cumlens,
                "key_sink": sink_k_nope,
                "value_sink": sink_v,
                "key_rope_sink": sink_k_pe,
            }
            if self.use_aicpu_fa_tiling:
                query_cumlens = attn_metadata.prefill.query_cumlens.to(torch.int64)
                meta_data_args = {
                    "num_heads_q": self.num_local_heads,
                    "num_heads_kv": self.num_local_heads,
                    "head_dim_qk": q_nope.shape[-1],
                    "head_dim_v": v.shape[-1],
                    "actual_seq_lengths": query_cumlens,
                    "actual_seq_lengths_kv": query_cumlens,
                    "sparse_mode": 4,
                    "pre_tokens": self.sliding_window - 1,
                    "next_tokens": 0,
                    "input_layout": "TND",
                    "input_layout_kv": "TND",
                    "rope_head_dim": q_pe.shape[-1],
                    "k_sink_num": sink_k_nope.shape[0],
                    "block_size": self.block_size,
                }
                meta_data = npu_fused_infer_attention_sink_metadata(
                    meta_data_args,
                    self.is_fa_metadata_producer,
                    "prefill" + self._fa_meta_suffix,
                )
                kwargs.update({
                    "actual_seq_qlen": query_cumlens,
                    "actual_seq_kvlen": query_cumlens,
                    "meta_data": meta_data,
                })

            attn_output = torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0]

        return attn_output.view(-1, self.num_local_heads * self.v_head_dim)

    @attn_decorator(type="dsa")
    def _apply_DSA_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
    ) -> torch.Tensor:

        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode

        q = torch.cat([q_nope, q_pe], dim=-1)

        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla", "fp8_ds_mla"]:
            if self.cache_config.cache_dtype == "hif8_ds_mla":
                key_dtype = torch_npu.hifloat8
                value_dtype = torch_npu.hifloat8
                key = kv_cache[0].unsqueeze(2) if len(kv_cache[0].shape) == 3 else kv_cache[0]
                value = self.dummy_value_cache_hif8_fp8
            else:  # fp8_ds_mla
                key_dtype = None
                value_dtype = None
                key = kv_cache[0].unsqueeze(2) if len(kv_cache[0].shape) == 3 else kv_cache[0]
                value = self.dummy_value_cache_hif8_fp8
                key = key.view(torch.float8_e4m3fn)
                value = value.view(torch.float8_e4m3fn)
            attn_output = torch.ops.custom.npu_ai_infra_kv_quant_sparse_flash_attention(
                query=q,
                key=key,
                value=value,
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                block_table=metadata.block_table,
                actual_seq_lengths_query=metadata.query_cumlens.clone(),
                actual_seq_lengths_kv=metadata.seq_lens.clone(),
                sparse_block_size=1,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                rope_head_dim=64,
                key_dtype=key_dtype,
                value_dtype=value_dtype,
                key_sink=self.sink_kv,
                value_sink=self.sink_k_nope
            )
        elif self.cache_config.cache_dtype in ["int8_ds_mla"]:
            attn_output = torch.ops.custom.npu_ai_infra_kv_quant_sparse_flash_attention(
                query=q,
                key=kv_cache[0].unsqueeze(2),
                value=kv_cache[0].unsqueeze(2),
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                key_quant_mode=2,
                value_quant_mode=2,
                sparse_block_size=1,
                actual_seq_lengths_query=metadata.query_cumlens,
                actual_seq_lengths_kv=metadata.seq_lens,
                key_sink=self.sink_kv,
                value_sink=self.sink_k_nope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                block_table=metadata.block_table,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                rope_head_dim=64,
            )
        else:
            attn_output = torch.ops.custom.npu_ai_infra_sparse_flash_attention_pioneer(
                query=q,
                key=kv_cache[0].unsqueeze(2),
                value=self.dummy_value_cache,
                sparse_indices=topk_indices,
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=metadata.block_table,
                actual_seq_lengths_query=metadata.query_cumlens,
                actual_seq_lengths_kv=metadata.seq_lens,
                pre_tokens=(1<<63)-1,
                next_tokens=(1<<63)-1,
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                key_sink=self.sink_kv,
                value_sink=self.sink_k_nope,
            )[0]

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        if attn_metadata.prefill is not None:
            # Prefill: apply v_up here (no pre_epilog_callback fires on this path).
            attn_output = (
                torch_npu.npu_transpose_batchmatmul(attn_output, self.W_UV, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
                    .reshape(-1, self.num_local_heads * self.v_head_dim)
            )
            return attn_output

        # Decode: defer v_up to _mla_epilog so it can overlap with side-stream
        # work launched by pre_epilog_callback. Return latent [T, N, L].
        return attn_output

    def _mla_prolog_hif8_fp8_decode_multistream(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor], # DSA/MLA/SWA absorb
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: # MLA/SWA non-absorb
        # get KV cache for this layer
        kv_cache = self.attn.kv_cache

        kv_a_done_event = torch.npu.Event()
        q_a_norm_done_event = torch.npu.Event()
        kv_ready_to_scatter_event = torch.npu.Event()
        q_b_proj_done_event = torch.npu.Event()

        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_a_done_event.record()

        with torch.npu.stream(self.side_stream):
            kv_a_done_event.wait(self.side_stream)
            q_lora = self.q_a_proj(hidden_states)
            hidden_states.record_stream(self.side_stream)
            if self.use_mome:
                q_lora = self._apply_MOME(
                    q_lora,
                    self.qa_conv,
                    attn_metadata=attn_metadata,
                    mome_metadata=mome_metadata,
                )
            q_lora = self.q_a_layernorm(q_lora)
            q_a_norm_done_event.record()

        k_nope, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.use_mome:
            k_nope = self._apply_MOME(
                k_nope,
                self.compresskv_conv,
                attn_metadata=attn_metadata,
                mome_metadata=mome_metadata,
            )

        actual_seq_kvlen = attn_metadata.slot_mapping.shape[0]
        k_nope = k_nope[:actual_seq_kvlen, ...]
        k_pe = k_pe[:actual_seq_kvlen, ...]
        cos = cos[:actual_seq_kvlen, ...]
        sin = sin[:actual_seq_kvlen, ...]

        k_pe = torch_npu.npu_rotary_mul(
            k_pe.view(-1, 1, 1, self.qk_rope_head_dim),
            cos.view(-1, 1, 1, self.qk_rope_head_dim),
            sin.view(-1, 1, 1, self.qk_rope_head_dim),
        ).squeeze(1).squeeze(1)

        k_nope = self.kv_a_layernorm(k_nope)
        if self.is_dsa_layer and self.cache_config.cache_dtype == "hif8_ds_mla":
            if self.use_dynamic_quant_k_nope:
                # Retained legacy per-128-tile dynamic-quant path (disabled:
                # use_dynamic_quant_k_nope is False; default is the direct cast).
                tile_size = 128
                k_nope_shape = k_nope.shape
                shape_with_tile = (*k_nope_shape[:-1], k_nope_shape[-1] // tile_size, tile_size)
                k_nope_quant, k_nope_scale = torch_npu.npu_dynamic_quant(
                    k_nope.view(shape_with_tile), dst_type=torch_npu.hifloat8, dst_type_max=15.0
                )
                k_nope_quant = k_nope_quant.view(k_nope_shape)
                k_nope_scale = k_nope_scale.view(k_nope_shape[:-1] + (k_nope_shape[-1] // tile_size,))
            else:
                # Default: direct hif8 cast with unit scale (a3-aligned).
                k_nope_scale = torch.ones(
                    (k_nope.shape[0], k_nope.shape[1] // 128), dtype=torch.float32, device=k_nope.device,
                )
                k_nope_quant = torch_npu.npu_dtype_cast(k_nope, torch_npu.hifloat8)
            kv = torch.cat(
                [k_nope_quant, k_pe.view(torch.uint8), k_nope_scale.view(torch.uint8)],
                dim=-1,
            )
            q_a_norm_done_event.wait(torch.npu.current_stream())
            kv_ready_to_scatter_event.record()
            slot_mapping_2d = _get_slot_mapping_2d(attn_metadata)
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(torch.int8),
                slot_mapping_2d,
                kv.view(torch.int8)
            )
        elif self.is_dsa_layer and self.cache_config.cache_dtype == "fp8_ds_mla":
            tile_size = 128
            k_nope_shape = k_nope.shape
            shape_with_tile = (*k_nope_shape[:-1], k_nope_shape[-1] // tile_size, tile_size)
            k_nope_quant, k_nope_scale = torch_npu.npu_dynamic_quant(
                k_nope.view(shape_with_tile), dst_type=torch.float8_e4m3fn
            )
            k_nope_quant = k_nope_quant.view(k_nope_shape)
            k_nope_scale = k_nope_scale.view(k_nope_shape[:-1] + (k_nope_shape[-1] // tile_size,))
            kv = torch.cat(
                [k_nope_quant.view(torch.uint8), k_pe.view(torch.uint8), k_nope_scale.view(torch.uint8)],
                dim=-1,
            )
            q_a_norm_done_event.wait(torch.npu.current_stream())
            kv_ready_to_scatter_event.record()
            slot_mapping_2d = _get_slot_mapping_2d(attn_metadata)
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(torch.int8),
                slot_mapping_2d,
                kv.view(torch.int8)
            )
        else:  # SWA is always bf16
            q_a_norm_done_event.wait(torch.npu.current_stream())
            kv_ready_to_scatter_event.record()
            slot_mapping_2d = _get_slot_mapping_2d(attn_metadata)
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0],
                slot_mapping_2d,
                k_nope.squeeze(1).squeeze(1),
            )
            torch_npu.npu_scatter_nd_update_(
                kv_cache[1],
                slot_mapping_2d,
                k_pe.squeeze(1).squeeze(1),
            )

        with torch.npu.stream(self.side_stream):
            kv_ready_to_scatter_event.wait(self.side_stream)
            q = self.q_b_proj(q_lora)
            q_b_proj_done_event.record()

        q.record_stream(torch.npu.current_stream())
        q_b_proj_done_event.wait(torch.npu.current_stream())
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim)
        q_nope = (
            torch_npu.npu_transpose_batchmatmul(q_nope, self.W_UK_T, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
                .reshape(-1, self.num_local_heads, self.kv_lora_rank)
        )

        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.num_local_heads, self.qk_rope_head_dim),
            cos.view(-1, 1, 1, self.qk_rope_head_dim),
            sin.view(-1, 1, 1, self.qk_rope_head_dim),
            rotary_mode="half" if not self.rope_interleave else "interleave",
        ).squeeze(1)
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()

        if self.is_dsa_layer:
            if self.skip_topk:
                topk_indices = self._get_topk_indices(attn_metadata)
            else:
                topk_indices = self.indexer(
                    hidden_states,
                    q_lora,
                    cos,
                    sin,
                    attn_metadata,
                    kv_cache,
                )
                self._set_topk_indices(attn_metadata, topk_indices)
            q_lora.record_stream(torch.npu.current_stream())
        else:
            topk_indices = None

        ret = (kv_cache, topk_indices)

        output = (q_nope, q_pe, *ret)
        return output

    def _mla_prolog(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor], # DSA/MLA/SWA absorb
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]: # MLA/SWA non-absorb

        # mla prolog multistream override for hif8 decode only specific case
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        if self.cache_config.cache_dtype in ["hif8_ds_mla", "fp8_ds_mla"] and decode_only(attn_metadata):
            return self._mla_prolog_hif8_fp8_decode_multistream(
                hidden_states,
                cos, sin,
                attn_metadata=attn_metadata,
                mome_metadata=mome_metadata
            )

        # get KV cache for this layer
        kv_cache = self.attn.kv_cache

        # A5 HiF8 (Ascend950): pin the MLA prolog to the original single-stream
        # path. The swa/dsa side-stream prologs (added in !1156, continually
        # retuned upstream for A3) run only where side_stream is non-None — i.e.
        # A5 alone — so A5 silently inherits A3-targeted churn. Sequential is
        # numerically identical (same ops, no stream overlap); keeps A5 last-known-good.
        if self.on_ascend950:
            return self._mla_prolog_sequential(
                hidden_states, cos, sin, kv_cache, attn_metadata, mome_metadata,
            )
        is_decode = attn_metadata.decode is not None or self.is_dsa_layer
        use_side = self.side_stream is not None and is_decode

        # Dispatch to the appropriate implementation
        # Low-latency (naive/allreduce): SWA and DSA multi-stream
        # high-throughput (allgather/alltoall) multi-stream
        if use_side and self.is_dsa_layer:
            return self._mla_prolog_dsa_multistream(
                hidden_states, cos, sin, kv_cache, attn_metadata, mome_metadata,
            )
        elif use_side:
            return self._mla_prolog_swa_multistream(
                hidden_states, cos, sin, kv_cache, attn_metadata, mome_metadata,
            )
        else:
            return self._mla_prolog_sequential(
                hidden_states, cos, sin, kv_cache, attn_metadata, mome_metadata,
            )

    def _q_rope(self, q_pe, cos, sin):
        """Q RoPE."""
        q_pe = torch_npu.npu_rotary_mul(
            q_pe.view(-1, 1, self.num_local_heads, self.qk_rope_head_dim),
            cos.view(-1, 1, 1, self.qk_rope_head_dim),
            sin.view(-1, 1, 1, self.qk_rope_head_dim),
            rotary_mode="half" if not self.rope_interleave else "interleave",
        ).squeeze(1)
        return q_pe.contiguous()

    def _w_uk_t_absorb(self, q_nope):
        """W_UK_T absorb: project q_nope into KV lora space."""
        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim)
        q_nope = (
            torch_npu.npu_transpose_batchmatmul(
                q_nope, self.W_UK_T, perm_x1=(1, 0, 2), perm_y=(1, 0, 2),
            ).reshape(-1, self.num_local_heads, self.kv_lora_rank)
        )
        return q_nope.contiguous()

    def _kv_down_mome(self, hidden_states, attn_metadata, mome_metadata):
        """KV stream: kv_a_proj → MOME on k_nope slice.

        Non-Ascend950: use the specialized inplace wrapper that takes the
        whole kv tensor (a leaf tensor, not a view) so AOT can declare
        mutates_args=["kv"] without hitting view-input-mutation assert,
        and the kvrmsnorm afterwards sees the post-MoME data — no cat.

        Ascend950: forward_decode/forward_prefill ignore inplace and return
        a fresh tensor, so cat the MoME output back with k_pe.
        """
        kv = self.kv_a_proj_with_mqa(hidden_states)
        if self.use_mome:
            if self.use_mome_inplace_update:
                conv_states = self.mome_attn.kv_cache[1]
                kv = torch.ops.vllm.npu_pangu_kv_down_mome_inplace(
                    kv,
                    self.compresskv_conv.weight,
                    conv_states,
                    mome_metadata.query_start_loc,
                    mome_metadata.cache_indices,
                    mome_metadata.num_accepted_tokens,
                    mome_metadata.num_computed_tokens,
                    mome_metadata.block_idx_first_scheduled_token,
                    mome_metadata.block_idx_last_scheduled_token,
                    mome_metadata.block_idx_last_computed_token,
                    mome_metadata.pad_slot_id,
                    mome_metadata.max_query_len,
                    mome_metadata.B_size,
                    self.kv_lora_rank,
                    attn_metadata.num_actual_tokens,
                )
            else:
                k_nope, k_pe = torch.split(
                    kv,
                    [self.kv_lora_rank, self.qk_rope_head_dim],
                    dim=-1,
                )
                k_nope = self._apply_MOME(
                    k_nope, 
                    self.compresskv_conv, 
                    attn_metadata=attn_metadata, 
                    mome_metadata=mome_metadata, 
                    inplace=False, 
                )
                kv = torch.cat([k_nope, k_pe], dim=-1)
        return kv

    def _mla_prolog_sequential(
        self, hidden_states, cos, sin, kv_cache, attn_metadata, mome_metadata,
    ):
        """Original sequential path (prefill / DSA without multi-stream / no side_stream)."""
        enable_pa = (
            self.first_chunk_pa or
            getattr(attn_metadata.prefill, "chunked_context", None) is not None
        )
        ### Q stream begins ###
        q_lora = self.q_a_proj(hidden_states)
        if self.use_mome:
            q_lora = self._apply_MOME(
                q_lora,
                self.qa_conv,
                attn_metadata=attn_metadata,
                mome_metadata=mome_metadata,
            )
        q_lora = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_lora)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        if (
            attn_metadata.decode is not None
            or self.is_dsa_layer
            or enable_pa
        ):
            q_nope = self._w_uk_t_absorb(q_nope)

        q_pe = self._q_rope(q_pe, cos, sin)
        ### Q stream ends ###

        ### Indexer stream begins ###
        topk_indices = None
        if self.is_dsa_layer:
            if self.skip_topk:
                topk_indices = self._get_topk_indices(attn_metadata)
            else:
                topk_indices = self.indexer(
                    hidden_states,
                    q_lora,
                    cos,
                    sin,
                    attn_metadata,
                    kv_cache,
                )
                self._set_topk_indices(attn_metadata, topk_indices)
        else:
            topk_indices = None
        ### Indexer stream ends ###

        ### KV stream begins ###
        kv = self._kv_down_mome(hidden_states, attn_metadata, mome_metadata)
        # Wrapper only handles absorb/DSA paths (return 2-tuple). For SWA
        # prefill non-absorb (returns 3 tensors), use direct dispatcher.
        if (
            attn_metadata.decode is not None
            or self.is_dsa_layer
            or enable_pa
        ):
            new_kv_cache = torch.ops.vllm.npu_pangu_kv_cache_update(
                kv, kv_cache[0], kv_cache[1], cos, sin, self.prefix,
            )
            ret = (new_kv_cache, topk_indices)
        else:
            ret = self._npu_kvrmsnorm_rope_cache(
                kv,
                kv_cache,
                cos,
                sin,
                attn_metadata,
                topk_indices,
            )
        ### KV stream ends ###

        output = (q_nope, q_pe, *ret)
        return output

    def _mla_prolog_swa_multistream(
        self, hidden_states, cos, sin, kv_cache, attn_metadata, mome_metadata,
    ):
        """SWA decode multi-stream: main computes Q nope path, side computes KV + Q pe path.

        split_q_up_in_multistream=True (default):
            Main: q_a_proj -> mome -> norm -> [quant?] -> q_b_nope -> W_UK_T
            Side: kv_down -> cache_update -> [wait q_lora] -> q_b_pe -> rope
        split_q_up_in_multistream=False:
            Main: q_a_proj -> mome -> norm -> q_b_proj (full forward) -> split -> W_UK_T (q_nope)
            Side: kv_down -> cache_update -> [wait q_pe] -> rope (q_pe)
        """
        main_stream = torch.npu.current_stream()
        split_q_up = self.split_q_up_in_multistream

        # Share hidden_states with side stream (read-only)
        hidden_states_event = torch.npu.Event()
        hidden_states_event.record()
        hidden_states.record_stream(self.side_stream)

        with torch.npu.npugraph_ex.scope.limit_core_num(16,24):
            # Main stream: q_lora
            q_lora = self.q_a_proj(hidden_states)
            if self.use_mome:
                q_lora = self._apply_MOME(
                    q_lora, self.qa_conv, attn_metadata=attn_metadata, mome_metadata=mome_metadata,
                )
            q_lora = self.q_a_layernorm(q_lora)

            if split_q_up:
                # Share q_lora with side stream so q_b_pe_proj can run there.
                q_lora_event = torch.npu.Event()
                q_lora_event.record()
                q_lora.record_stream(self.side_stream)

                # Main stream: q_nope via q_b_nope_proj (its quant_method.apply
                # handles dequant / NZ / scale exactly like q_b_proj does).
                q_nope = self.q_b_nope_proj(q_lora).view(
                    -1, self.num_local_heads, self.qk_nope_head_dim,
                )
            else:
                # Unsplit: main runs full q_b_proj then split; q_pe handed to side for rope.
                q = self.q_b_proj(q_lora).view(-1, self.num_local_heads, self.qk_head_dim)
                q_nope, q_pe = torch.split(
                    q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1,
                )
                q_nope = q_nope.contiguous()
                q_pe = q_pe.contiguous()

                q_pe_event = torch.npu.Event()
                q_pe_event.record()
                q_pe.record_stream(self.side_stream)

            # Main stream: q_nope absorb (overlaps with side stream KV + rope)
            q_nope = self._w_uk_t_absorb(q_nope)

        # Side stream: KV + q_pe rope (runs in parallel with main stream absorb)
        with torch.npu.stream(self.side_stream):
            hidden_states_event.wait(self.side_stream)
            with torch.npu.npugraph_ex.scope.limit_core_num(8,24):
                kv = self._kv_down_mome(hidden_states, attn_metadata, mome_metadata)
                new_kv_cache = torch.ops.vllm.npu_pangu_kv_cache_update(
                    kv, kv_cache[0], kv_cache[1], cos, sin, self.prefix,
                )

                if split_q_up:
                    q_lora_event.wait(self.side_stream)
                    q_pe = self.q_b_pe_proj(q_lora).view(
                        -1, self.num_local_heads, self.qk_rope_head_dim,
                    )
                else:
                    q_pe_event.wait(self.side_stream)
                q_pe = self._q_rope(q_pe, cos, sin)

            side_done_event = torch.npu.Event()
            side_done_event.record()

        # Sync: main stream waits for side stream outputs
        side_done_event.wait(main_stream)
        q_pe.record_stream(main_stream)
        new_kv_cache[0].record_stream(main_stream)
        new_kv_cache[1].record_stream(main_stream)

        return (q_nope, q_pe, new_kv_cache, None)

    def _mla_prolog_dsa_multistream(
        self, hidden_states, cos, sin, kv_cache, attn_metadata, mome_metadata,
    ):
        """DSA decode multi-stream: main stream does Q + indexer k/w + kvrmsnorm,
        side stream does KV down + indexer q + LI.

        Main stream: q_a -> norm -> indexer_k/w -> q_b_nope/pe -> W_UK_T -> rope -> [wait kv] -> kvrmsnorm
        Side stream: kv_down -> [signal kv] -> [wait q_lora] -> indexer_q -> [wait k/w] -> rope -> cache_update -> LI

        kv_cache_update (kvrmsnorm) writes kv_cache[0] only; indexer_cache_update / LI
        touch kv_cache[1] / kv_cache[2], so no false dependency between streams.
        """
        main_stream = torch.npu.current_stream()

        # Share hidden_states with side stream
        hidden_states_event = torch.npu.Event()
        hidden_states_event.record()
        hidden_states.record_stream(self.side_stream)

        with torch.npu.npugraph_ex.scope.limit_core_num(16,24):
            # Main stream: q_lora
            q_lora = self.q_a_proj(hidden_states)
            if self.use_mome:
                q_lora = self._apply_MOME(
                    q_lora, self.qa_conv, attn_metadata=attn_metadata, mome_metadata=mome_metadata,
                )
            q_lora = self.q_a_layernorm(q_lora)

            # Signal q_lora ready
            q_lora_event = torch.npu.Event()
            q_lora_event.record()
            q_lora.record_stream(self.side_stream)

            # Main stream: indexer k/w (parallel to q_lora on main, needed by side stream)
            if not self.skip_topk:
                indexer_k = self.indexer.wk(hidden_states)
                indexer_k = self.indexer.k_norm(indexer_k)
                indexer_weights = self.indexer.weights_proj(hidden_states)

            # Signal indexer k/w ready
            indexer_kw_event = torch.npu.Event()
            indexer_kw_event.record()
            if not self.skip_topk:
                indexer_k.record_stream(self.side_stream)
                indexer_weights.record_stream(self.side_stream)

        # Event signalled by side stream once kv (from kv_down_mome) is ready
        # for main-stream consumption by npu_pangu_kv_cache_update.
        kv_ready_event = torch.npu.Event()

        # Keep an alias to the original kv_cache buffers for the main-stream
        # kv_cache_update. The side stream rebinds `kv_cache` to the SSA-edge
        # tuple returned by npu_pangu_indexer_cache_update so LI sees the
        # dependency; if main stream consumed that rebound tuple too, FX
        # would record a false data edge from indexer_cache_update into
        # kv_cache_update and could serialise the two ops in graph mode.
        kv_cache_main = kv_cache

        # Side stream: KV down + indexer q + LI
        with torch.npu.stream(self.side_stream):
            hidden_states_event.wait(self.side_stream)

            with torch.npu.npugraph_ex.scope.limit_core_num(8,24):
                # KV down (independent of q_lora)
                kv = self._kv_down_mome(hidden_states, attn_metadata, mome_metadata)

                # Hand kv off to main stream for kvrmsnorm/cache update.
                kv.record_stream(main_stream)
                kv_ready_event.record()

                # indexer_q needs q_lora
                q_lora_event.wait(self.side_stream)
                if not self.skip_topk:
                    indexer_q = self.indexer.wq_b(q_lora)

                # indexer rope needs indexer_k/w
                indexer_kw_event.wait(self.side_stream)

                # Indexer RoPE
                if not self.skip_topk:
                    if self.indexer.use_rope_fusion_op:
                        indexer_q, indexer_k = torch_npu.npu_apply_rotary_pos_emb(
                            indexer_q.view(-1, 1, self.indexer.index_n_heads, self.indexer.index_head_dim),
                            indexer_k.view(-1, 1, 1, self.indexer.index_head_dim),
                            cos.view(-1, 1, 1, self.indexer.qk_rope_head_dim),
                            sin.view(-1, 1, 1, self.indexer.qk_rope_head_dim),
                            layout="BSND", rotary_mode="half"
                        )
                        indexer_q = indexer_q.view(-1, self.indexer.index_n_heads, self.indexer.index_head_dim)
                    else:
                        indexer_q = indexer_q.view(-1, self.indexer.index_n_heads, self.indexer.index_head_dim)
                        q_pe_i, q_nope_i = torch.split(
                            indexer_q,
                            [self.indexer.qk_rope_head_dim,
                             self.indexer.index_head_dim - self.indexer.qk_rope_head_dim],
                            dim=-1,
                        )
                        q_pe_i = torch_npu.npu_rotary_mul(
                            q_pe_i.view(-1, 1, self.indexer.index_n_heads, self.indexer.qk_rope_head_dim),
                            cos.view(-1, 1, 1, self.indexer.qk_rope_head_dim),
                            sin.view(-1, 1, 1, self.indexer.qk_rope_head_dim),
                        ).squeeze(1)
                        indexer_q = torch.cat([q_pe_i, q_nope_i], dim=-1)

                        k_pe_i, k_nope_i = torch.split(
                            indexer_k,
                            [self.indexer.qk_rope_head_dim,
                             self.indexer.index_head_dim - self.indexer.qk_rope_head_dim],
                            dim=-1,
                        )
                        k_pe_i = torch_npu.npu_rotary_mul(
                            k_pe_i.view(-1, 1, 1, self.indexer.qk_rope_head_dim),
                            cos.view(-1, 1, 1, self.indexer.qk_rope_head_dim),
                            sin.view(-1, 1, 1, self.indexer.qk_rope_head_dim),
                        ).squeeze(1).squeeze(1)
                        indexer_k = torch.cat([k_pe_i, k_nope_i], dim=-1)

                    # Lightning indexer + cache update
                    kv_cache_2 = kv_cache[2] if len(kv_cache) > 2 else None
                    kv_cache = torch.ops.vllm.npu_pangu_indexer_cache_update(
                        indexer_k,
                        kv_cache[0], kv_cache[1], kv_cache_2,
                        self.prefix,
                    )
            if self.skip_topk:
                topk_indices = self._get_topk_indices(attn_metadata)
            else:
                with torch.npu.npugraph_ex.scope.limit_core_num(20, 40):
                    topk_indices = torch.ops.vllm.npu_pangu_lightning_indexer(
                        indexer_q, indexer_weights,
                        kv_cache[0], kv_cache[1], kv_cache_2,
                        self.prefix,
                    )
                self._set_topk_indices(attn_metadata, topk_indices)

            side_done_event = torch.npu.Event()
            side_done_event.record()

        with torch.npu.npugraph_ex.scope.limit_core_num(12,24):
            # Main stream: full Q path
            if self.split_q_up_in_multistream:
                q_nope = self.q_b_nope_proj(q_lora).view(
                    -1, self.num_local_heads, self.qk_nope_head_dim,
                )
                q_pe = self.q_b_pe_proj(q_lora).view(
                    -1, self.num_local_heads, self.qk_rope_head_dim,
                )
            else:
                q = self.q_b_proj(q_lora).view(-1, self.num_local_heads, self.qk_head_dim)
                q_nope, q_pe = torch.split(
                    q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1,
                )
                q_nope = q_nope.contiguous()
                q_pe = q_pe.contiguous()
            q_nope = self._w_uk_t_absorb(q_nope)
            q_pe = self._q_rope(q_pe, cos, sin)

            # KV cache update (kvrmsnorm) — moved from side stream so it overlaps
            # the indexer q/rope/cache_update/LI on the side stream. Use the
            # original kv_cache buffers (kv_cache_main) to avoid creating a
            # false FX edge from npu_pangu_indexer_cache_update on the side.
            kv_ready_event.wait(main_stream)
            new_kv_cache = torch.ops.vllm.npu_pangu_kv_cache_update(
                kv, kv_cache_main[0], kv_cache_main[1], cos, sin, self.prefix,
            )

        # Sync: main stream waits for side stream's topk_indices
        side_done_event.wait(main_stream)
        if not self.skip_topk:
            topk_indices.record_stream(main_stream)

        return (q_nope, q_pe, new_kv_cache, topk_indices)

    def _kv_rmsnorm_rope_cache_v2_kwargs(
        self,
        kv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata],
        **extra,
    ) -> dict:
        kwargs = {
            "kv": kv.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim),
            "gamma": self.kv_a_layernorm.weight,
            "cos": cos.view(-1, 1, 1, self.qk_rope_head_dim),
            "sin": sin.view(-1, 1, 1, self.qk_rope_head_dim),
            "index": attn_metadata.slot_mapping,
            "epsilon": self.kv_a_layernorm.variance_epsilon,
            "cache_mode": "PA",
            "rotary_mode": "half" if not self.rope_interleave else "interleave-half",
            "quant_mode": "none",
            "is_output_kv": True,
        }
        kwargs.update(extra)
        return kwargs

    def _npu_kvrmsnorm_rope_cache(self, *args, **kwargs):
        if self.is_dsa_layer and self.cache_config.cache_dtype in self.quant_cache_dtype:
            return self._npu_kvrmsnorm_rope_cache_quant(*args, **kwargs)
        else:
            return self._npu_kvrmsnorm_rope_cache_unquant(*args, **kwargs)

    def _npu_kvrmsnorm_rope_cache_quant(
        self,
        kv: torch.Tensor, 
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata],
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # DSA layer c8 shape
        if self.on_ascend950 and self.cache_config.cache_dtype in ["hif8_ds_mla", "fp8_ds_mla"]:
            actual_seq_kvlen = attn_metadata.slot_mapping.shape[0]
            cos = cos[:actual_seq_kvlen, ...]
            sin = sin[:actual_seq_kvlen, ...]
            k_nope, k_pe = torch.split(
                kv[:actual_seq_kvlen, ...],
                [self.kv_lora_rank, self.qk_rope_head_dim],
                dim=-1,
            )

            k_pe = torch_npu.npu_rotary_mul(
                k_pe.view(-1, 1, 1, self.qk_rope_head_dim),
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
            ).squeeze(1).squeeze(1)

            k_nope = self.kv_a_layernorm(k_nope)
            if self.cache_config.cache_dtype == "hif8_ds_mla":
                if self.use_dynamic_quant_k_nope:
                    # Retained legacy per-128-tile dynamic-quant path (disabled:
                    # use_dynamic_quant_k_nope is False; default is the direct cast).
                    tile_size = 128
                    k_nope_shape = k_nope.shape
                    shape_with_tile = (*k_nope_shape[:-1], k_nope_shape[-1] // tile_size, tile_size)
                    k_nope_quant, k_nope_scale = torch_npu.npu_dynamic_quant(
                        k_nope.view(shape_with_tile), dst_type=torch_npu.hifloat8,
                        dst_type_max=15.0
                    )
                    k_nope_quant = k_nope_quant.view(k_nope_shape)
                    k_nope_scale = k_nope_scale.view(k_nope_shape[:-1] + (k_nope_shape[-1] // tile_size,))
                else:
                    # Default: direct hif8 cast with unit scale (a3-aligned).
                    k_nope_scale = torch.ones(
                        (k_nope.shape[0], k_nope.shape[1] // 128), dtype=torch.float32, device=k_nope.device,
                    )
                    k_nope_quant = torch_npu.npu_dtype_cast(k_nope, torch_npu.hifloat8)
            else:  # fp8_ds_mla
                tile_size = 128
                k_nope_shape = k_nope.shape
                shape_with_tile = (*k_nope_shape[:-1], k_nope_shape[-1] // tile_size, tile_size)
                k_nope_quant, k_nope_scale = torch_npu.npu_dynamic_quant(
                    k_nope.view(shape_with_tile), dst_type=torch.float8_e4m3fn
                )
                k_nope_quant = k_nope_quant.view(k_nope_shape)
                k_nope_scale = k_nope_scale.view(k_nope_shape[:-1] + (k_nope_shape[-1] // tile_size,))
            if self.cache_config.cache_dtype == "hif8_ds_mla":
                kv = torch.cat(
                    [k_nope_quant, k_pe.view(torch.uint8), k_nope_scale.view(torch.uint8)],
                    dim=-1,
                )
            else:  # fp8_ds_mla
                kv = torch.cat(
                    [k_nope_quant.view(torch.uint8), k_pe.view(torch.uint8), k_nope_scale.view(torch.uint8)],
                    dim=-1,
                )

            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(torch.int8),
                _get_slot_mapping_2d(attn_metadata),
                kv.view(torch.int8)
            )

            return kv_cache, topk_indices
        elif self.cache_config.cache_dtype in ["int8_ds_mla"]:
            actual_seq_kvlen = attn_metadata.slot_mapping.shape[0]
            cos = cos[:actual_seq_kvlen, ...]
            sin = sin[:actual_seq_kvlen, ...]

            kv = kv[:actual_seq_kvlen, ...]

            kwargs = self._kv_rmsnorm_rope_cache_v2_kwargs(
                kv, cos, sin, attn_metadata,
                quant_mode="pertile128",
                k_cache=None,
                ckv_cache=kv_cache[0].unsqueeze(2),
            )
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)
            return kv_cache, topk_indices
        else:
            raise RuntimeError(
                f"Unsupported cache_dtype '{self.cache_config.cache_dtype}' "
                f"for kv rmsnorm rope cache quant."
            )

    def _naive_kvrmsnorm_rope_cache(
        self,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: MLACommonMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        update_k_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Naive torch equivalent of npu_ai_infra_kv_rmsnorm_rope_cache_v2 for Ascend 950."""
        k_nope = k_nope.view(-1, 1, 1, self.kv_lora_rank)
        k_pe = k_pe.view(-1, 1, 1, self.qk_rope_head_dim)

        # RMSNorm on k_nope
        k_nope = torch_npu.npu_rms_norm(k_nope, self.kv_a_layernorm.weight, self.kv_a_layernorm.variance_epsilon)[0]

        # Rotary embedding on k_pe
        rotary_mode = "half" if not self.rope_interleave else "interleave"
        k_pe = torch_npu.npu_rotary_mul(k_pe, cos, sin, rotary_mode=rotary_mode)

        # Scatter update caches
        slot_indices = _get_slot_mapping_2d(attn_metadata)

        torch_npu.npu_scatter_nd_update_(
            kv_cache[0],
            slot_indices,
            k_nope.squeeze(1).squeeze(1),
        )
        if update_k_cache:
            torch_npu.npu_scatter_nd_update_(
                kv_cache[1],
                slot_indices,
                k_pe.squeeze(1).squeeze(1),
            )

        return k_pe.squeeze(1).squeeze(1), k_nope.squeeze(1).squeeze(1)

    def _npu_kvrmsnorm_rope_cache_unquant(
        self,
        kv: torch.Tensor, 
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata],
        topk_indices: Optional[torch.Tensor] = None,
    ):
        enable_pa = (
            self.first_chunk_pa or 
            getattr(attn_metadata.prefill, "chunked_context", None) is not None
        )
        # the rest cases are unquantized
        actual_seq_kvlen = attn_metadata.slot_mapping.shape[0]
        cos = cos[:actual_seq_kvlen, ...]
        sin = sin[:actual_seq_kvlen, ...]

        kv = kv[:actual_seq_kvlen, ...]
        if self.on_ascend950:
            k_nope, k_pe = torch.split(
                kv,
                [self.kv_lora_rank, self.qk_rope_head_dim],
                dim=-1,
            )
        else:
        # 950 use naive kernel for kv rmsnorm and update, accept separate nope and rope
            kwargs = self._kv_rmsnorm_rope_cache_v2_kwargs(
                kv, cos, sin, attn_metadata,
            )

        if self.is_dsa_layer:
            # DSA shape
            kwargs.update({
                "k_cache": None,
                "ckv_cache": kv_cache[0].unsqueeze(2),
            })
            k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)

            return kv_cache, topk_indices

        elif (
            attn_metadata.decode is not None
            or enable_pa
        ):
            # MLA/SWA absorb shape
            if self.on_ascend950:
                k_pe, k_nope = self._naive_kvrmsnorm_rope_cache(
                    k_nope, k_pe, kv_cache, attn_metadata,
                    cos.view(-1, 1, 1, self.qk_rope_head_dim),
                    sin.view(-1, 1, 1, self.qk_rope_head_dim),
                    update_k_cache=True,
                )
            else:
                kwargs.update({
                    "k_cache": kv_cache[1].unsqueeze(2),
                    "ckv_cache": kv_cache[0].unsqueeze(2),
                })
                k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)

            return kv_cache, topk_indices

        else:
            # MLA/SWA non-absorb shape
            if self.on_ascend950:
                k_pe, k_nope = self._naive_kvrmsnorm_rope_cache(
                    k_nope, k_pe, kv_cache, attn_metadata,
                    cos.view(-1, 1, 1, self.qk_rope_head_dim),
                    sin.view(-1, 1, 1, self.qk_rope_head_dim),
                    update_k_cache=True,
                )
            else:
                kwargs.update({
                    "k_cache": kv_cache[1].unsqueeze(2),
                    "ckv_cache": kv_cache[0].unsqueeze(2),
                })
                k_pe, k_nope = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(**kwargs)

            kv_up = self.kv_b_proj(k_nope)
            kv_up = kv_up.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_up_nope, v_up = torch.split(
                kv_up,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
            k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim) \
                       .repeat(1, self.num_local_heads, 1)

            return k_up_nope.contiguous(), k_pe.contiguous(), v_up.contiguous()

    def _mla_epilog(
        self,
        attn_output: torch.Tensor,
        attn_metadata: Optional[MLACommonMetadata] = None,
        mome_metadata: Optional[NPUMomeAttentionMetadata] = None,
    ) -> torch.Tensor:

        # v_up (W_UV absorb) for decode paths that defer it to here so
        # pre_epilog_callback's side-stream work can overlap with it.
        # Latent input is [T, N, L]; post-v_up paths pass [T, N*V] (2D).
        if attn_output.dim() == 3:
            attn_output = (
                torch_npu.npu_transpose_batchmatmul(
                    attn_output, self.W_UV, perm_x1=(1, 0, 2), perm_y=(1, 0, 2),
                ).reshape(-1, self.num_local_heads * self.v_head_dim)
            )

        if self.use_mome:
            # Gather head shards only when o_conv keeps full channels (CP or PD disagg).
            if self.disable_o_conv_tp and attn_output.size(-1) < self.o_conv.dim:
                attn_output = get_tp_group().all_gather(attn_output, dim=1)
            attn_output = self._apply_MOME(
                attn_output,
                self.o_conv,
                attn_metadata=attn_metadata,
                mome_metadata=mome_metadata,
            )
            if self.disable_o_conv_tp and self.o_proj.requires_input_partition():
                attn_output = split_tensor_along_last_dim(attn_output, num_partitions=self.o_proj.tp_size)
                attn_output = attn_output[self.o_proj.tp_rank].contiguous()

        return self._apply_o_proj(attn_output)

    def _skip_topk(self, config: DeepseekV2Config | DeepseekV3Config) -> bool:
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is None:
            return False
        return indexer_types[self.layer_idx] == "shared"

    @staticmethod
    def _get_topk_metadata(
        attn_metadata: MLACommonMetadata,
    ) -> MLACommonPrefillMetadata | MLACommonDecodeMetadata:
        if attn_metadata.prefill is not None:
            return attn_metadata.prefill
        return attn_metadata.decode

    @classmethod
    def _get_topk_indices(cls, attn_metadata: MLACommonMetadata) -> torch.Tensor:
        return cls._get_topk_metadata(attn_metadata).topk_indices_buffer

    @classmethod
    def _set_topk_indices(cls, attn_metadata: MLACommonMetadata, topk_indices: torch.Tensor) -> None:
        cls._get_topk_metadata(attn_metadata).topk_indices_buffer = topk_indices


def npu_pangu_forward(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]    
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        mome_metadata = attn_metadata.get(f"{self.prefix}.mome")
        attn_metadata = attn_metadata.get(f"{self.prefix}.attn")
    else:
        mome_metadata = None

    if attn_metadata is None:
        return self._forward_dummy(
            hidden_states,
        )
    else:
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0

        enable_cp = self.is_cp_layer and not has_decode \
            and num_actual_tokens > attn_metadata.num_prefills * self.tp_size * 2
        # Need to make sure there are at least 8 tokens (or else no tokens) on each rank
        # so that there are enough tokens to communicate
        enable_attn_sp = self.is_attn_sp_layer and not has_decode
        enable_flashcomm2 = self.enable_flashcomm2 and not self.is_dsa_layer and not has_decode

        # Only skip the global all_gather when the CP/SP/FC2 path will actually run
        # (pure prefill, no decode). Mixed batch and decode paths still need it.
        if self.tp_size > 1 and self.moe_comm_strategy != "allreduce":
            is_prefill_cp = enable_cp or enable_attn_sp or enable_flashcomm2
            if not is_prefill_cp or (has_decode and has_prefill):
                hidden_states = get_tp_group().all_gather(hidden_states, dim=0)

        if has_decode and has_prefill:
            prefill_hidden_states, prefill_cos, prefill_sin = self._prepare_phase_inputs(
                hidden_states, cos, sin, attn_metadata,
                phase="prefill",
            )
            hidden_states[num_decode_tokens:num_actual_tokens] = self._forward_prefill(
                prefill_hidden_states,
                prefill_cos,
                prefill_sin,
                attn_metadata,
                mome_metadata.prefill if mome_metadata is not None else None,
            )

            decode_hidden_states, decode_cos, decode_sin = self._prepare_phase_inputs(
                hidden_states, cos, sin, attn_metadata,
                phase="decode",
            )
            hidden_states[:num_decode_tokens] = self._forward_decode(
                decode_hidden_states,
                decode_cos,
                decode_sin,
                attn_metadata,
                mome_metadata.decode if mome_metadata is not None else None,
            )

            self._restore_phase_metadata(attn_metadata)

        elif attn_metadata.prefill is not None:
            if enable_cp:
                assert (
                    self.moe_comm_strategy != "allreduce"
                ), "Context parallel is not supported with allreduce MoE communication strategy"
                return self._forward_prefill_cp(
                    hidden_states,
                    cos,
                    sin,
                    attn_metadata,
                    mome_metadata,
                )
            elif enable_attn_sp:
                assert (
                    self.moe_comm_strategy != "allreduce"
                ), (
                    "Attention sequence parallelism is not supported with the "
                    "allreduce MoE communication strategy"
                )
                return self._forward_prefill_sp(
                    hidden_states,
                    cos,
                    sin,
                    attn_metadata,
                    mome_metadata,
                )
            elif enable_flashcomm2:
                return self._forward_prefill_FC2(
                    hidden_states,
                    cos,
                    sin,
                    attn_metadata,
                    mome_metadata,
                )
            else:
                hidden_states[num_decode_tokens:num_actual_tokens] = self._forward_prefill(
                    hidden_states[num_decode_tokens:num_actual_tokens],
                    cos[num_decode_tokens:num_actual_tokens],
                    sin[num_decode_tokens:num_actual_tokens],
                    attn_metadata,
                    mome_metadata,
                )
        else:
            if hidden_states.shape[0] == num_decode_tokens:
                hidden_states = self._forward_decode(
                    hidden_states,
                    cos,
                    sin,
                    attn_metadata,
                    mome_metadata,
                )
            else:
                hidden_states[:num_decode_tokens] = self._forward_decode(
                    hidden_states[:num_decode_tokens],
                    cos[:num_decode_tokens],
                    sin[:num_decode_tokens],
                    attn_metadata,
                    mome_metadata,
                )
        if self.tp_size > 1:
            need_reduce = self.o_proj.tp_size > 1
            need_scatter = self.moe_comm_strategy != "allreduce"
            if need_reduce and need_scatter:
                hidden_states = get_tp_group().reduce_scatter(hidden_states, dim=0)
            elif need_reduce:
                hidden_states = get_tp_group().all_reduce(hidden_states)
            elif need_scatter:
                tp_rank = get_tp_group().rank_in_group
                chunk_size = hidden_states.shape[0] // self.tp_size
                hidden_states = hidden_states[
                    chunk_size * tp_rank : chunk_size * (tp_rank + 1)
                ]
        return hidden_states


def npu_pangu_forward_fake(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.zeros_like(hidden_states)


direct_register_custom_op(
    op_name="npu_pangu_forward",
    op_func=npu_pangu_forward,
    mutates_args=[],
    fake_impl=npu_pangu_forward_fake,
    dispatch_key="PrivateUse1",
)


# Opt-in custom-op wrappers for NPUPanguSparseAttention sub-paths. Importing
# this module registers torch.ops.vllm.<op_name>(...) entry points that
# callers can substitute for the corresponding direct method calls when
# they need to hide in-place kv_cache / conv_states mutations from
# torch.compile / AOT autograd. See npu_pangu_custom_ops.py for the list of
# available ops and how to use / extend them.
from omni_npu.v1.layers.attention import npu_pangu_custom_ops  # noqa: F401, E402
