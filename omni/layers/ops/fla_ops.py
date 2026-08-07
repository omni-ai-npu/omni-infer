# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from math import log
from typing import Optional, Union

import omni_custom_ops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from einops import rearrange
from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("qwen3_next_rms_norm_gated")
class Qwen3NextRMSNormGated(CustomOp):
    """Qwen3 Next RMS Normalization with optional gating, optimized for Ascend NPUs.

    This implementation supports:
    - Standard RMS normalization (Accelerated via NPU kernel)
    - Group RMS normalization (Native fallback)
    - Optional gating with SiLU activation
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: Optional[int] = None,
        norm_before_gate: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize Qwen3NextRMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
            norm_before_gate: Flag to control gating order
            device: Device to create parameters on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        # Weights initialized to 1.0, so we do NOT use the weight + 1.0 Gemma trick
        torch.nn.init.ones_(self.weight)

    def forward(
        self, x: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Optimized implementation utilizing Ascend NPU kernels for standard RMSNorm.
        Falls back to native math for Grouped RMSNorm due to dimension requirements.
        """
        if torch_npu is None:
            return self.forward_native(x, z)

        orig_dtype = x.dtype

        # 1. Apply gating before normalization if needed
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z)

        # 2. RMS Normalization
        if self.group_size is None:
            x_normed = torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]
            out = x_normed.to(orig_dtype)
        else:
            x_group = rearrange(x, "... (g d) -> ... g d", d=self.group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + self.eps)
            out = rearrange(x_normed, "... g d -> ... (g d)") * self.weight

        # 3. Apply gating after normalization if needed
        if z is not None and self.norm_before_gate:
            out = out * F.silu(z)

        return out


@CustomOp.register("omni_gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    
    @staticmethod
    def forward_static(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """npu implementation of gemma rms norm"""
        orig_dtype = x.dtype
        if residual is not None:
            if orig_dtype == torch.float16:
                x = x + residual.float()
            else:
                x = x + residual
            residual = x
        x = torch_npu.npu_rms_norm(x, weight + 1.0, variance_epsilon)[0]
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)
    
    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_static(
            self.weight,
            self.variance_epsilon,
            x,
            residual,
        )


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_state = initial_state.contiguous()
    assert inplace_final_state == True
    eq_len = cu_seqlens is None
    contiguous_states = ssm_state_indices is None
    bs = q.shape[0] if eq_len else len(cu_seqlens) - 1
    T = q.shape[1] // bs
    hv_over_h = v.shape[-2] // q.shape[-2]
    if scale is None:
        scale = k.shape[-1]**-0.5
    if use_qk_l2norm_in_kernel:
        q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
        k = k / (torch.linalg.norm(k, dim=-1, keepdim=True) + 1e-6)
    q = q * scale
    g = g.exp()
    q = q.repeat(1, 1, 1, hv_over_h).reshape(v.shape)
    k = k.repeat(1, 1, 1, hv_over_h).reshape(v.shape)
    indices_0 = torch.arange(bs, device=q.device) * T
    if not contiguous_states:
        if num_accepted_tokens is None:
            indices = ssm_state_indices.view(-1)[indices_0]
        else:
            indices = ssm_state_indices.view(-1)[indices_0 + num_accepted_tokens - 1]
    else:
        indices = indices_0 if eq_len else cu_seqlens[:-1]
    S = initial_state[indices].to(torch.float32).transpose(-1, -2)
    A, tbs, C, D = q.shape
    E = v.shape[-1]
    q = q.reshape(A, bs, T, C, D)
    k = k.reshape(A, bs, T, C, D)
    v = v.reshape(A, bs, T, C, E)
    g = g.reshape(A, bs, T, C)
    beta = beta.reshape(A, bs, T, C)
    o = []
    for t in range(T):
        q_t = q[0, :, t].to(torch.float32)
        k_t = k[0, :, t].to(torch.float32)
        v_t = v[0, :, t].to(torch.float32)
        g_t = g[0, :, t].to(torch.float32)
        beta_t = beta[0, :, t].to(torch.float32)
        S = g_t.view(bs, C, 1, 1) * S
        x = torch.einsum('abc,abcd->abd', k_t, S)
        y = beta_t.unsqueeze(-1) * (v_t - x)
        S_ = k_t.unsqueeze(-1) * y.unsqueeze(-2)
        S = S + S_
        o_t = torch.einsum('abc,abcd->abd', q_t, S)
        if not contiguous_states:
            indices = ssm_state_indices.view(-1)[indices_0 + t]
        else:
            indices = cu_seqlens[:-1] + t if eq_len else indices_0 + t
        torch_npu.npu_scatter_nd_update_(
            initial_state,
            indices.unsqueeze(1),
            S.to(torch.bfloat16).transpose(-1, -2),
        )
        o.append(o_t.to(torch.bfloat16))
    o = torch.cat(o, dim=1)
    o = o.contiguous().reshape(A, tbs, C, E)
    return o, initial_state.contiguous()


def _fused_recurrent_gated_delta_rule_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    actual_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_qk_l2norm_in_kernel:
        q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
        k = k / (torch.linalg.norm(k, dim=-1, keepdim=True) + 1e-6)
    _, total_tokens, key_heads, key_head_dim = q.shape
    _, _, value_heads, value_head_dim = v.shape
    query = q.reshape(total_tokens, key_heads, key_head_dim)
    key = k.reshape(total_tokens, key_heads, key_head_dim)
    value = v.reshape(total_tokens, value_heads, value_head_dim)
    beta = beta.reshape(total_tokens, value_heads)
    gate = g.reshape(total_tokens, value_heads)
    output = torch.ops.custom.npu_ai_infra_recurrent_gated_delta_rule(
        query=query.to(torch.bfloat16),
        key=key.to(torch.bfloat16),
        value=value.to(torch.bfloat16),
        state=initial_state,
        beta=beta.to(torch.bfloat16),
        scale=scale,
        actual_seq_lengths=actual_seqlens.to(torch.int32),
        ssm_state_indices=ssm_state_indices.reshape(-1).to(torch.int32),
        num_accepted_tokens=(
            None
            if num_accepted_tokens is None
            else num_accepted_tokens.to(torch.int32)
        ),
        g=None if gate is None else gate.to(torch.float32),
        gk=None,
    )
    return output, initial_state


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    actual_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = _fused_recurrent_gated_delta_rule_npu(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            actual_seqlens=actual_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return o, final_state


def chunk_gated_delta_rule_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size=128,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    initial_dtype: torch.dtype = None,
):
    if initial_dtype is None:
        initial_dtype = query.dtype
    query = query.contiguous()
    key = key.contiguous()
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
    transposed_tensors = []
    for tensor in (query, key, value, beta, g):
        transposed_tensors.append(
            tensor.transpose(1, 2).contiguous().to(torch.float32)
        )
    query, key, value, beta, g = transposed_tensors
    batch_size, num_qk_heads, sequence_length, k_head_dim = key.shape
    num_v_heads = value.shape[1]
    v_head_dim = value.shape[-1]
    scale = 1 / (k_head_dim**0.5)
    query.mul_(scale)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size)).repeat_interleave(num_v_heads // num_qk_heads, dim=1)
    key = F.pad(key, (0, 0, 0, pad_size)).repeat_interleave(num_v_heads // num_qk_heads, dim=1)
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size)).unsqueeze(-1)
    g = F.pad(g, (0, pad_size))
    sequence_length_padded = sequence_length + pad_size
    v_beta = value * beta
    k_beta = key * beta
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) -
                   g.unsqueeze(-2)).exp().float()
    mask = torch.triu(torch.ones(chunk_size,
                                    chunk_size,
                                    dtype=torch.bool,
                                    device=query.device),
                        diagonal=0)
    attn = -(
        (k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    attn_inv = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device).repeat((tuple(attn.shape)[:-2] + (1, 1)))
    attn = attn_inv - attn
    attn = torch.ops.custom.npu_lower_triangular_inverse(attn)

    value = attn @ v_beta
    gexp = g.exp()
    k_cumdecay = attn @ (k_beta * gexp.unsqueeze(-1))

    query_view = query.reshape(query.shape[0], query.shape[1], -1, chunk_size, query.shape[-1])
    key_trans = key.reshape(key.shape[0], key.shape[1], -1, chunk_size, key.shape[-1]).transpose(-1, -2)
    qk = query_view @ key_trans
    mask = torch.triu(torch.ones(chunk_size,
                                    chunk_size,
                                    dtype=torch.bool,
                                    device=query.device),
                        diagonal=1)
    attn_score = qk * decay_mask.masked_fill_(mask, 0)

    gexp = gexp.unsqueeze(-1)
    qgexp = query * gexp

    kgexp = (g[:, :, :, -1, None] - g[:, :, :]).exp()[..., None]
    kgexp = key * kgexp

    if initial_state is None:
        initial_state_ = value.new_empty(batch_size, num_v_heads, v_head_dim, k_head_dim)
        initial_state_[...] = 0
    else:
        initial_state_ = initial_state.to(value)
    state_fp32 = initial_state_.to(torch.float32)
    attn_inter_out, v_new_out = torch.ops.custom.npu_chunk_gated_delta_rule_recurrence(
        state_fp32,
        kgexp.squeeze(0),
        value.squeeze(0),
        k_cumdecay.squeeze(0),
        qgexp.squeeze(0),
        gexp.squeeze(0),
        torch.ones(1, dtype=torch.int32, device=query.device) * sequence_length_padded,
    )
    core_attn_out = attn_inter_out + attn_score @ v_new_out
    initial_state_ = state_fp32.to(initial_state_.dtype)

    if not output_final_state:
        initial_state_ = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                          core_attn_out.shape[1], -1,
                                          core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1,
                                            2).contiguous().to(initial_dtype)
    return core_attn_out, initial_state_


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: list[int] | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    batch_size = initial_state.shape[0]
    core_attn_outs = []
    last_recurrent_states = []
    initial_dtype = q.dtype
    chunk_size = 128
    for b_idx in range(batch_size):
        start, end = cu_seqlens_cpu[b_idx], cu_seqlens_cpu[b_idx + 1]
        cur_q = q[:, start:end, ...].contiguous()
        cur_k = k[:, start:end, ...].contiguous()
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_b = beta[:, start:end, ...]
        cur_state = initial_state[b_idx].unsqueeze(0)

        core_attn_out, initial_state_ = chunk_gated_delta_rule_npu(
                                            query=cur_q,
                                            key=cur_k,
                                            value=cur_v,
                                            g=cur_g,
                                            beta=cur_b,
                                            chunk_size=chunk_size,
                                            initial_state=cur_state,
                                            output_final_state=True,
                                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                                            initial_dtype=initial_dtype,
                                        )
        core_attn_outs.append(core_attn_out)
        last_recurrent_states.append(initial_state_)

    core_attn_out_non_spec = torch.cat(core_attn_outs, dim=1)
    last_recurrent_states = torch.cat(last_recurrent_states, dim=0)
    return core_attn_out_non_spec, last_recurrent_states


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    batch_size = initial_state.shape[0]
    core_attn_outs = []
    last_recurrent_states = []
    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q[:, start:end, ...].contiguous()
        cur_k = k[:, start:end, ...].contiguous()
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_b = beta[:, start:end, ...]
        cur_state = initial_state[b_idx].unsqueeze(0)

        chunk_size = 128
        initial_dtype = cur_q.dtype
        if use_qk_l2norm_in_kernel:
            cur_q = F.normalize(cur_q, p=2, dim=-1)
            cur_k = F.normalize(cur_k, p=2, dim=-1)
        transposed_tensors = []
        for tensor in (cur_q, cur_k, cur_v, cur_b, cur_g):
            transposed_tensors.append(
                tensor.transpose(1, 2).contiguous().to(torch.float32)
            )
        cur_q, cur_k, cur_v, cur_b, cur_g = transposed_tensors

        bs, num_qk_heads, sequence_length, k_head_dim = cur_k.shape
        num_v_heads = cur_v.shape[1]
        v_head_dim = cur_v.shape[-1]
        scale = 1 / (k_head_dim**0.5)
        cur_q.mul_(scale)
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        cur_q = F.pad(cur_q, (0, 0, 0, pad_size)).repeat_interleave(num_v_heads // num_qk_heads, dim=1)
        cur_k = F.pad(cur_k, (0, 0, 0, pad_size)).repeat_interleave(num_v_heads // num_qk_heads, dim=1)
        cur_v = F.pad(cur_v, (0, 0, 0, pad_size))
        cur_b = F.pad(cur_b, (0, pad_size)).unsqueeze(-1)
        cur_g = F.pad(cur_g, (0, pad_size))
        sequence_length_padded = sequence_length + pad_size
        v_beta = cur_v * cur_b
        k_beta = cur_k * cur_b
        # reshape to chunks
        cur_q, cur_k, cur_v, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (cur_q, cur_k, cur_v, k_beta, v_beta)
        ]
        cur_g = cur_g.reshape(cur_g.shape[0], cur_g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size,
                                    chunk_size,
                                    dtype=torch.bool,
                                    device=cur_q.device),
                        diagonal=0)

        # chunk decay
        cur_g = cur_g.cumsum(dim=-1)
        decay_mask = (cur_g.unsqueeze(-1) -
                    cur_g.unsqueeze(-2)).exp().float()
        attn = -(
            (k_beta @ cur_k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        lg = int(log(chunk_size, 2))

        block_size = 1
        attn_inv = torch.eye(
            chunk_size,
            dtype=attn.dtype,
            device=attn.device,
        ).repeat(tuple(attn.shape)[:-2] + (1, 1))
        attn = attn_inv - attn
        for _ in range(lg):
            block_num = chunk_size // block_size
            prod = attn @ attn_inv
            block_shape = tuple(attn.shape)[:-2] + (
                block_num,
                block_size,
                block_num,
                block_size,
            )
            attn_inv_block = attn_inv.view(block_shape).transpose(-2, -3)
            prod_block = prod.view(block_shape).transpose(-2, -3)
            r0 = torch.arange(block_num // 2, device=attn.device) * 2
            r1 = r0 + 1
            attn_inv_block[:, :, :, r1, r0, :, :] = (
                -attn_inv_block[..., r1, r1, :, :]
                @ prod_block[..., r1, r0, :, :]
            )
            attn_inv = attn_inv_block.transpose(-2, -3).view(
                tuple(attn_inv_block.shape)[:-4] + (chunk_size, chunk_size)
            )
            block_size *= 2
        attn = attn_inv

        cur_v = attn @ v_beta
        gexp = cur_g.exp()
        k_cumdecay = attn @ (k_beta * gexp.unsqueeze(-1))

        last_recurrent_state = (torch.zeros(bs, num_v_heads,
                                            k_head_dim, v_head_dim).to(cur_v) if
                                cur_state is None else cur_state.to(cur_v))

        core_attn_out = torch.zeros_like(cur_v)
        mask = torch.triu(torch.ones(chunk_size,
                                    chunk_size,
                                    dtype=torch.bool,
                                    device=cur_q.device),
                        diagonal=1)
        query_view = cur_q.reshape(cur_q.shape[0], cur_q.shape[1], -1, chunk_size, cur_q.shape[-1])
        key_trans = cur_k.reshape(cur_k.shape[0], cur_k.shape[1], -1, chunk_size, cur_k.shape[-1]).transpose(-1, -2)
        qk = query_view @ key_trans
        attn_score = qk * decay_mask.masked_fill_(mask, 0)

        gexp = cur_g[:, :, :, :, None].exp()
        qgexp = cur_q * gexp

        kgexp = (cur_g[:, :, :, -1, None] - cur_g[:, :, :]).exp()[..., None]
        kgexp = cur_k * kgexp

        k_cumdecay_qgexp = torch.cat([k_cumdecay, qgexp], dim=3)
        v_new_out = torch.zeros_like(cur_v)
        attn_inter_out = torch.zeros_like(cur_v)

        for i in range(sequence_length_padded // chunk_size):
            v_i = cur_v[:, :, i]
            attn = attn_score[:, :, i]
            v_prime_attn_inter = (k_cumdecay_qgexp[:, :, i]) @ last_recurrent_state
            v_prime = v_prime_attn_inter[:, :, :chunk_size]
            attn_inter = v_prime_attn_inter[:, :, chunk_size:]
            v_new = v_i - v_prime
            v_new_out[:, :, i] = v_new
            attn_inter_out[:, :, i] = attn_inter
            last_recurrent_state *= gexp[:, :, i, -1, :, None]
            last_recurrent_state += (kgexp[:, :, i]).transpose(-1, -2) @ v_new
        core_attn_out = attn_inter_out + attn_score @ v_new_out

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                            core_attn_out.shape[1], -1,
                                            core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1,
                                                2).contiguous().to(initial_dtype)
        core_attn_outs.append(core_attn_out)
        last_recurrent_states.append(last_recurrent_state.transpose(-1, -2))

    tar_dtype = core_attn_outs[0].dtype
    tar_device = core_attn_outs[0].device
    tar_shape = list(core_attn_outs[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    core_attn_out_non_spec = torch.empty(tar_shape,
                                            dtype=tar_dtype,
                                            device=tar_device)
    for b_idx in range(batch_size):
        cur_core_attn_out = core_attn_outs[b_idx]
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        core_attn_out_non_spec[:, start:end, ...] = cur_core_attn_out
    last_recurrent_states = torch.cat(last_recurrent_states, dim=0)
    return core_attn_out_non_spec, last_recurrent_states
