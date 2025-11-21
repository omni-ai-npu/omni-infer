# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import os
from typing import Optional

import torch
import torch_npu, torchair
torch._dynamo.config.capture_scalar_outputs = True
from vllm.triton_utils import tl, tldevice, triton

PAD_SLOT_ID = -1

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    div = tldevice.fast_dividef
    exp = tldevice.fast_expf
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:

    @triton.jit
    def div_normal(x, y):
        return x / y

    div = div_normal
    # exp = tl.exp
    # log = tl.log
    # log2 = tl.log2


@triton.heuristics({
    'USE_INITIAL_STATE':
    lambda args: args['h0'] is not None,
    'IS_VARLEN':
    lambda args: args['cu_seqlens'] is not None,
    "IS_CONTINUOUS_BATCHING":
    lambda args: args['ssm_state_indices'] is not None,
    "IS_SPEC_DECODING":
    lambda args: args['num_accepted_tokens'] is not None,
})
@triton.jit(do_not_specialize=['N', 'T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.constexpr,  # num of sequences
    T: tl.constexpr,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.
    constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    # p_q = q + (bos * H + i_h) * K + o_k
    # p_k = k + (bos * H + i_h) * K + o_k
    # p_v = v + (bos * HV + i_hv) * V + o_v
    # if IS_BETA_HEADWISE:
    #     p_beta = beta + (bos * HV + i_hv) * V + o_v
    # else:
    #     p_beta = beta + bos * HV + i_hv
    # p_g = g + bos * HV + i_hv
    # p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = h0 + tl.load(ssm_state_indices + i_n * stride_indices_seq +
                                i_t).to(tl.int64) * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * K * V
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        p_q = q + (bos * H + i_h) * K + o_k + H * K * i_t
        p_k = k + (bos * H + i_h) * K + o_k + H * K * i_t
        p_v = v + (bos * HV + i_hv) * V + o_v + HV * V * i_t
        if IS_BETA_HEADWISE:
            p_beta = beta + (bos * HV + i_hv) * V + o_v + HV * V * i_t
        else:
            p_beta = beta + bos * HV + i_hv + HV * i_t
        p_g = g + bos * HV + i_hv + HV * i_t
        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v + HV * V * i_t

        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        # b_h *= tl.exp(b_g)
        b_h *= exp(b_g)
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            p_ht = ht + tl.load(ssm_state_indices + i_n * stride_indices_seq +
                                i_t).to(tl.int64) * stride_final_state_token
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # p_q += H * K
        # p_k += H * K
        # p_o += HV * V
        # p_v += HV * V
        # p_g += HV
        # p_beta += HV * (V if IS_BETA_HEADWISE else 1)

# --- self-defined torch implementation of fused_recurrent_gated_delta_rule_fwd FUNCTION ---
def _fused_recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool,
    cu_seqlens: torch.LongTensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert inplace_final_state == True
    eq_len = cu_seqlens is None
    contiguous_states = ssm_state_indices is None
    bs = q.shape[0] if eq_len else len(cu_seqlens) - 1
    T = q.shape[1] // bs
    hv_over_h = v.shape[-2] // q.shape[-2]
    if scale is None:
        scale = k.shape[-1]**-0.5
    #Apply QK L2 norm if enabled teeheehee
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
            indices = ssm_state_indices[indices_0]
        else:
            indices = ssm_state_indices[indices_0 + num_accepted_tokens - 1]
    else:
        indices = indices_0 if eq_len else cu_seqlens[:-1]
    S = initial_state[indices].to(torch.float32) #[bs, C, D, E]
    #reshape for tensor operation
    A, tbs, C, D = q.shape
    E = v.shape[-1]
    q = q.reshape(A, bs, T, C, D).transpose(1, 2) #[A, T, bs, C, D]
    k = k.reshape(A, bs, T, C, D).transpose(1, 2) #[A, T, bs, C, D]
    v = v.reshape(A, bs, T, C, E).transpose(1, 2) #[A, T, bs, C, E]
    g = g.reshape(A, bs, T, C).transpose(1, 2) #[A, T, bs, C]
    beta = beta.reshape(A, bs, T, C).transpose(1, 2) #[A, T, bs, C]
    o = []
    for t in range(T):
        q_t = q[0][t].to(torch.float32) #[bs, C, D]
        k_t = k[0][t].to(torch.float32) #[bs, C, D]
        v_t = v[0][t].to(torch.float32) #[bs, C, E]
        g_t = g[0][t].to(torch.float32) #[bs, C]
        beta_t = beta[0][t].to(torch.float32) #[bs, C]
        S = g_t.view(bs, C, 1, 1) * S
        x = torch.einsum('abc,abcd->abd', k_t, S) #[bs, C, E]
        y = beta_t.unsqueeze(-1) * (v_t - x) #[bs, C, E]
        S_ = k_t.unsqueeze(-1) * y.unsqueeze(-2) #[bs, C, D, E]
        S = S + S_ #[bs, C, D, E]
        o_t = torch.einsum('abc,abcd->abd', q_t, S) #[bs, C, E]
        if not contiguous_states:
            indices = ssm_state_indices[indices_0 + t]
        else:
            indices = cu_seqlens[:-1] + t if eq_len else indices_0 + t
        torch_npu.npu_scatter_nd_update_(initial_state, indices.unsqueeze(1), S.to(torch.bfloat16))
        o.append(o_t.to(torch.bfloat16))
    o = torch.cat(o, dim = 1)
    o = o.contiguous().reshape(A, tbs, C, E)
    return o, initial_state

# --- npu implementation of fused_recurrent_gated_delta_rule_fwd FUNCTION ---
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
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    #Apply QK L2 norm if enabled teeheehee
    if use_qk_l2norm_in_kernel:
        q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
        k = k / (torch.linalg.norm(k, dim=-1, keepdim=True) + 1e-6)
    _, T, Nk, Dk = q.shape
    _, _, Nv, Dv = v.shape
    q = q.reshape(T, Nk, Dk)
    k = k.reshape(T, Nk, Dk)
    v = v.reshape(T, Nv, Dv)
    beta = beta.reshape(T, Nv)
    g = g.reshape(T, Nv)
    actual_seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1] if cu_seqlens is not None else None
    # Call the NPU reference implementation
    o = torch_npu.npu_recurrent_gated_delta_rule(
        query=q.to(torch.bfloat16),
        key=k.to(torch.bfloat16),
        value=v.to(torch.bfloat16),
        state=initial_state.to(torch.bfloat16),
        beta=beta.to(torch.bfloat16),
        scale=scale,
        actual_seq_lengths=actual_seq_lengths.to(torch.int32),
        ssm_state_indices=ssm_state_indices.to(torch.int32),
        num_accepted_tokens= None if num_accepted_tokens is None else num_accepted_tokens.to(torch.int32),
        g=None if g is None else g.to(torch.float32),
        gk=None,
    )
    return o, initial_state  # Placeholder for final_state

def get_gdn_based_on_env():
    env_value = os.getenv("TORCH_NPU_USE_PARALLEL_TCPSTORE", "False").strip().lower()
    if env_value in ("true", "1", "yes", "on"):
        return _fused_recurrent_gated_delta_rule_npu  # Package is available
    else:
        return _fused_recurrent_gated_delta_rule_ref  # Use self-defined reference implementation


# --- MODIFIED fused_recurrent_gated_delta_rule_fwd FUNCTION ---
def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    gdn = get_gdn_based_on_env()
    o, final_state = gdn(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                scale: float,
                initial_state: torch.Tensor,
                inplace_final_state: bool = True,
                cu_seqlens: Optional[torch.LongTensor] = None,
                ssm_state_indices: Optional[torch.Tensor] = None,
                num_accepted_tokens: Optional[torch.Tensor] = None,
                use_qk_l2norm_in_kernel: bool = False):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

        return o, final_state


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing.")
    if scale is None:
        scale = k.shape[-1]**-0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
