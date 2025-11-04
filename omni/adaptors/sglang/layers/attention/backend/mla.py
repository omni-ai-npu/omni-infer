from __future__ import annotations

"""
Support attention backend for NpuMLA.

#TODO
Enable speculative sampling in NpuMLA
"""

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo


PAGE_SIZE = 128
MAX_SEQ_LEN = 4096

@dataclass
class AscendMLAMetadata:
    npumla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    block_kv_indices: Optional[torch.Tensor] = None
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None

    def __init__(self,
        layer,
        forward_batch: ForwardBatch = None,
        block_kv_indices: Optional[torch.Tensor] = None,
        actual_seq_lengths: Dict = None,
        norm_res: Dict = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        self.page_size = PAGE_SIZE
        self.block_kv_indices = block_kv_indices
        bs = forward_batch.input_ids.size(0)

        # decode when block_kv_indices is not None
        if block_kv_indices is not None:
            self.norm_res = norm_res
            self.actual_seq_lengths = actual_seq_lengths         # Q:TND
            self.actual_seq_lengths_kv = (forward_batch.seq_lens # KV:NTD
                if forward_batch.seq_lens.size(0) == bs
                else forward_batch.positions.to(torch.int32) + 1)  # seq_lens without padding
            self.mc2_mask = forward_batch.positions.to(torch.bool) # reuse padding
        else:
            self.seq_lens_list = forward_batch.extend_seq_lens_cpu
            if self.seq_lens_list is not None:
                self.seq_lens_list_cumsum = np.cumsum(self.seq_lens_list).tolist()
            else:
                self.seq_lens_list = [1]
                self.seq_lens_list_cumsum = [1]

            if (
                forward_batch.is_extend_in_batch
                or forward_batch.global_num_tokens_cpu is None
            ):
                self.seq_lens_list_cumsum[-1] = bs
        self.attn_mask = attn_mask
        self.cos, self.sin = layer.self_attn.rotary_emb.get_cos_sin(forward_batch.positions)



class KVBlockTable:

    def __init__(self,
        req_to_token_pool, # ReqToTokenPool | DecodeReqToTokenPool
        max_seqlen_pad,
        device = "npu",
        page_size = PAGE_SIZE,
    ):
        self.device = device
        self.page_size = page_size
        self.ceil_div = lambda a, b: (a + (b - 1)) // b

        shape = req_to_token_pool.req_to_token.shape
        self.block_table = torch.zeros(
            shape[0],
            min(4096, max_seqlen_pad, self.ceil_div(shape[1], self.page_size)),
            dtype=torch.int32,
            device=device,
        )

        # monkey patch for write()
        old_fn = req_to_token_pool.write
        def new_fn(obj, indices, values):
            old_fn(indices, values)
            self._write(indices, values)
        req_to_token_pool.write = types.MethodType(new_fn, req_to_token_pool)

    def _write(self, indices, values):
        req_ptr, idx = indices
        pg = self.page_size # 128

        if type(idx) is torch.Tensor: # indices is (Tensor, Tensor)
            val = values // pg
            idx = idx // pg
        elif type(idx) is slice: # indices is (int, slice)
            cs, ce = self.ceil_div(idx.start, pg), self.ceil_div(idx.stop, pg)
            val = values[cs * pg - idx.start : idx.stop - idx.start : pg] // pg
            idx = torch.arange(cs, ce, device=self.device)

        self.block_table[req_ptr, idx] = val.to(self.device)

    def index(self, bs: int, ptr: torch.Tensor):
        if bs != ptr.size(0):
            assert bs > ptr.size(0)
            ptr = torch.nn.functional.pad(ptr, (0, bs - ptr.size(0)))
        return self.block_table[ptr, :]


class AscendMLABackend(TorchNativeAttnBackend):
    """npumla attention kernels."""

    def __init__(self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
    ):
        super().__init__(model_runner)
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.skip_prefill = skip_prefill
        self.forward_metadata: Optional[AscendMLAMetadata] = None

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.norm_res = None
        self.actual_seq_lengths = None
        if "deepseek" in model_runner.model_config.hf_config.architectures[0].lower():
            self.kv_lora_rank = model_runner.model_config.kv_lora_rank
            self.q_lora_rank = model_runner.model_config.hf_config.q_lora_rank
            self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
            self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
            self.v_head_dim = model_runner.model_config.v_head_dim
            self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
            self.scaling = model_runner.model_config.scaling
            if model_runner.dp_size > 1:
                bs = model_runner.server_args.torch_compile_max_bs
                self.norm_res = torch.zeros([bs, self.q_lora_rank], dtype=torch.bfloat16, device=self.device)
                self.actual_seq_lengths = torch.arange(1, bs + 1, dtype=torch.int64, device=self.device) # cumsum for TND layout
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.mask_length = 2048
        self.attn_mask = ~torch.tril(
            torch.ones(
                (self.mask_length, self.mask_length),
                dtype=torch.bool,
                device=model_runner.device,
            )
        )

        max_bs = model_runner.req_to_token_pool.size
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        self.q_indptr_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )
        max_total_tokens = model_runner.server_args.max_total_tokens or MAX_SEQ_LEN
        self.max_seqlen_pad = max_total_tokens // model_runner.server_args.page_size
        self.model_runner = model_runner

        pool = model_runner.req_to_token_pool
        if not hasattr(pool, "kv_block_table"):
            pool.kv_block_table = KVBlockTable(pool, self.max_seqlen_pad)
        self.kv_block_table = pool.kv_block_table

    def init_forward_metadata(self, forward_batch: ForwardBatch):

        model = self.model_runner.model.model
        layer = model.layers[0] if hasattr(model, "layers") else model.decoder

        if forward_batch.forward_mode.is_decode_or_idle():

            bs = forward_batch.input_ids.size(0)
            if not forward_batch.can_run_graph:
                self.norm_res = None
                self.actual_seq_lengths = torch.tensor(list(range(1, bs + 1)), dtype=torch.int64, device=self.device)

            self.forward_metadata = AscendMLAMetadata(
                layer=layer,
                forward_batch=forward_batch,
                block_kv_indices=self.kv_block_table.index(bs, forward_batch.req_pool_indices),
                actual_seq_lengths=self.actual_seq_lengths,
                norm_res=self.norm_res,
            )
        else:
            self.forward_metadata = AscendMLAMetadata(
                layer=layer,
                forward_batch=forward_batch,
                attn_mask=self.attn_mask,
            )

    def init_cuda_graph_state(self,
        max_bs: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        pass

    def init_forward_metadata_capture_cuda_graph(self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        forward_batch: ForwardBatch,
    ):
        self.forward_metadata = AscendMLAMetadata(
            None,
            torch.full(
                (bs, self.max_seqlen_pad),
                0,
                dtype=torch.int32,
                device=seq_lens.device,
            ),
            [1] * bs,
            forward_batch,
        )

    def init_forward_metadata_replay_cuda_graph(self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            if save_kv_cache:
                if k_rope is not None:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                    )
        padding_bs = forward_batch.input_ids.size(0)
        if q_rope is not None:
            q_nope = q.view(padding_bs, -1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                padding_bs, -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            reshape_q = q.view(padding_bs, -1, layer.tp_q_head_num, layer.head_dim)
            q_nope = reshape_q[..., : layer.v_head_dim]
            q_rope = reshape_q[..., layer.v_head_dim :]
            if q_rope.numel() == 0:
                q_rope = None

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
            q.dtype
        )
        if q_rope is None:
            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
                layer.layer_id
            ).to(q.dtype)
        else:
            v_cache = k_cache
        o = self._run_npu_forward_decode(
            (q_nope, q_rope), k_cache, v_cache, layer, forward_batch
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if forward_batch.is_extend_or_draft_extend:
            if k_rope is not None:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    k_rope,
                )
                bs = forward_batch.batch_size
                q_nope = q.view(bs, -1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    bs, -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
                k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).to(q.dtype)

                o = self._run_npu_forward_decode(
                    (q_nope, q_rope), k_cache, k_cache, layer, forward_batch
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            else:
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                    )
            use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
            return self._run_npu_forward_extend(q, k, v, layer, forward_batch, use_gqa)

    def _run_npu_forward_extend(self, q, k, v, layer, forward_batch, use_gqa=False):
        """
        q: (b*s, N, q_dim=192)
        k: (b*s, N, k_dim=192)
        v: (b*s, N, v_dim=128)
        """
        if q.ndim == 2:
            q = q.view(q.shape[0], self.num_local_heads, -1)
        bs_qlen, q_heads, q_dim = q.size()
        _, k_heads, k_dim = k.size()
        _, v_heads, v_dim = v.size()

        if use_gqa:
            attn_output = torch.empty(
                bs_qlen, q_heads, v_dim, device=q.device, dtype=q.dtype
            )
            q_len_offset = 0
            for q_len in forward_batch.seq_len:
                attn_output[q_len_offset : q_len_offset + q_len] = (
                    torch.ops.npu.npu_fused_infer_attention_score(
                        q[None, q_len_offset : q_len_offset + q_len],
                        k[None, q_len_offset : q_len_offset + q_len],
                        v[None, q_len_offset : q_len_offset + q_len],
                        num_heads=q_heads,
                        num_key_value_heads=k_heads,
                        input_layout="BSND",  # todo, TND not supports q_heads!=k_heads
                        atten_mask=self.attn_mask.unsqueeze(0),
                        sparse_mode=3,
                        scale=layer.scaling,
                        next_tokens=0,
                    )[0]
                )
                q_len_offset += q_len
        else:  # MHA
            if q_dim != v_dim:
                q_nope, q_rope = q.split(
                    [self.v_head_dim, self.qk_rope_head_dim], dim=-1
                )
                k_nope, k_rope = k.split(
                    [self.v_head_dim, self.qk_rope_head_dim], dim=-1
                )

                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope,
                    k_nope,
                    v,
                    query_rope=q_rope,
                    key_rope=k_rope,
                    num_heads=q_heads,
                    input_layout="TND",
                    atten_mask=self.attn_mask,
                    sparse_mode=3,
                    actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_list_cumsum,
                    scale=layer.scaling,
                    next_tokens=0,
                )
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q,
                    k,
                    v,
                    num_heads=q_heads,
                    input_layout="TND",
                    atten_mask=self.attn_mask,
                    sparse_mode=3,
                    actual_seq_lengths=self.forward_metadata.seq_lens_list_cumsum,
                    actual_seq_lengths_kv=self.forward_metadata.seq_lens_list_cumsum,
                    scale=layer.scaling,
                    next_tokens=0,
                )
            attn_output = attn_output[..., : layer.v_head_dim]

        return attn_output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _run_npu_forward_decode(self, q, k_cache, v_cache, layer, forward_batch):
        """
        q: (b, s, N, q_dim=576)
        k_cache: (tokens_capticy, 1, k_dim=576)
        """
        if not isinstance(q, torch.Tensor):
            q_nope, q_rope = q
        else:
            q_nope, q_rope = q.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        b, s, n, _ = q_nope.size()
        q_nope_dim = q_nope.shape[-1]
        _, k_heads, k_dim = k_cache.size()

        if q_rope is not None:  # MLA
            k_cache = k_cache.view(-1, PAGE_SIZE, k_dim)
            k_nope = k_cache[..., :q_nope_dim]
            k_rope = k_cache[..., q_nope_dim:]

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                k_nope,
                query_rope=q_rope,
                key_rope=k_rope,
                num_heads=n,
                num_key_value_heads=1,
                input_layout="BSND",
                atten_mask=None,
                sparse_mode=0,
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=self.forward_metadata.block_kv_indices,
                block_size=PAGE_SIZE,
                actual_seq_lengths_kv=self.forward_metadata.seq_lens_list,
            )
        else:  # MHA
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_cache.view(-1, PAGE_SIZE, k_heads * k_dim),
                v_cache.view(-1, PAGE_SIZE, k_heads * k_dim),
                num_heads=n,
                num_key_value_heads=k_heads,
                input_layout="BSND",
                atten_mask=None,
                block_size=PAGE_SIZE,
                block_table=self.forward_metadata.block_kv_indices,
                actual_seq_lengths_kv=self.forward_metadata.seq_lens_list,
                scale=layer.scaling,
            )
        attn_output = attn_output.view(b * s, layer.tp_q_head_num, layer.v_head_dim)
        return attn_output
