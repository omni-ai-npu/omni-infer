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


# --- NEW PURE PYTORCH REFERENCE IMPLEMENTATION ---
def _fused_recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor],
    ssm_state_indices: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of the recurrent gated delta rule.
    This function replaces the Triton kernel for functional correctness.
    NOTE: This implementation uses Python loops over sequences and heads,
    which will be significantly slower than the highly optimized Triton kernel
    for large inputs.
    """
    # Determine input dimensions
    if cu_seqlens is not None:
        # For variable length, B=1 and T is the total number of tokens
        _, _, H, K = q.shape
        N_sequences = len(cu_seqlens) - 1
    else:
        # For fixed length, B is the batch size and T is the sequence length
        B_input, T_input, H, K = q.shape
        N_sequences = B_input

    HV = v.shape[2]
    V_dim = v.shape[-1]

    # Initialize output tensor with the same shape as v
    o = torch.empty_like(v)

    # Initialize final state tensor
    if inplace_final_state:
        final_state = initial_state
    else:
        # If not in-place, create a new tensor for the final state.
        # The shape should match initial_state.
        final_state = torch.empty_like(initial_state)

    # Check if beta is applied per-head or per-value
    is_beta_headwise = beta.ndim == v.ndim

    # Loop over each sequence in the batch
    for n_idx in range(N_sequences):
        # Determine the start and end token indices for the current sequence
        if cu_seqlens is not None:
            seq_start_idx = cu_seqlens[n_idx]
            seq_end_idx = cu_seqlens[n_idx + 1]
            current_seq_len = seq_end_idx - seq_start_idx
        else:
            seq_start_idx = 0  # Not used for fixed-length
            current_seq_len = T_input

        if current_seq_len == 0:
            continue

        # Determine the state slot index for the current sequence
        if ssm_state_indices is not None:
            state_slot_idx = ssm_state_indices[n_idx]
            if state_slot_idx == PAD_SLOT_ID:
                continue
        else:
            state_slot_idx = n_idx

        # Load initial hidden state for the current sequence and clone it
        # h_state shape: [HV, K, V]
        h_state = initial_state[state_slot_idx].clone()

        # Loop over each token in the current sequence
        for t_idx in range(current_seq_len):
            # --- Correctly slice input tensors based on input format ---
            if cu_seqlens is not None:
                global_t_idx = seq_start_idx + t_idx
                # Slicing from [1, total_tokens, ...]
                q_t = q[0, global_t_idx]
                k_t = k[0, global_t_idx]
                v_t = v[0, global_t_idx]
                g_t = g[0, global_t_idx]
                beta_t = beta[0, global_t_idx]
            else:
                # Slicing from [B, T, ...]
                q_t = q[n_idx, t_idx]
                k_t = k[n_idx, t_idx]
                v_t = v[n_idx, t_idx]
                g_t = g[n_idx, t_idx]
                beta_t = beta[n_idx, t_idx]

            # Apply QK L2 norm if enabled
            if use_qk_l2norm_in_kernel:
                q_t = q_t / (torch.linalg.norm(q_t, dim=-1, keepdim=True) + 1e-6)
                k_t = k_t / (torch.linalg.norm(k_t, dim=-1, keepdim=True) + 1e-6)

            q_t = q_t * scale

            current_o_t_per_hv = []
            # Loop over each head-value group (HV)
            for hv_idx in range(HV):
                # State for current head-value group: [K, V]
                h_state_hv = h_state[hv_idx]

                # Map the head-value group index (hv_idx) to the query/key head index (i_h)
                i_h_for_hv = hv_idx // (HV // H)
                k_t_for_hv = k_t[i_h_for_hv]  # Shape: [K]
                q_t_for_hv = q_t[i_h_for_hv]  # Shape: [K]

                # 1. Decay previous state: h_state_hv *= exp(g_t[hv_idx])
                h_state_hv = h_state_hv * torch.exp(g_t[hv_idx])

                # 2. Calculate v' (part 1): v_prime = v - sum(h * k)
                v_prime_part = torch.sum(h_state_hv * k_t_for_hv.unsqueeze(-1), dim=0)
                v_prime = v_t[hv_idx] - v_prime_part  # Shape: [V]

                # 3. Apply beta to v': v_prime *= beta_t
                if is_beta_headwise:
                    v_prime = v_prime * beta_t[hv_idx]
                else:
                    # beta_t is [HV], so beta_t[hv_idx] is a scalar
                    v_prime = v_prime * beta_t[hv_idx]

                # 4. Update state with new information: h += k @ v'
                h_state_hv = h_state_hv + torch.outer(k_t_for_hv, v_prime)

                # 5. Compute output: o = sum(h * q)
                o_t_hv = torch.sum(h_state_hv * q_t_for_hv.unsqueeze(-1), dim=0)

                # Store the updated state for the current head-value group
                h_state[hv_idx] = h_state_hv
                current_o_t_per_hv.append(o_t_hv)

            # Stack outputs from all head-value groups for the current token
            output_for_token = torch.stack(current_o_t_per_hv, dim=0)  # Shape: [HV, V]

            # --- Store the output in the correct position ---
            if cu_seqlens is not None:
                global_t_idx = seq_start_idx + t_idx
                o[0, global_t_idx] = output_for_token
            else:
                o[n_idx, t_idx] = output_for_token

        # After processing all tokens in a sequence, store the final state
        final_state[state_slot_idx] = h_state

    return o, final_state


# --- MODIFIED fused_recurrent_gated_delta_rule_fwd FUNCTION ---
# This function now calls the pure PyTorch reference implementation.
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
    # Call the pure PyTorch reference implementation
    return _fused_recurrent_gated_delta_rule_ref(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


# def fused_recurrent_gated_delta_rule_fwd(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     g: torch.Tensor,
#     beta: torch.Tensor,
#     scale: float,
#     initial_state: torch.Tensor,
#     inplace_final_state: bool = True,
#     cu_seqlens: Optional[torch.LongTensor] = None,
#     ssm_state_indices: Optional[torch.Tensor] = None,
#     num_accepted_tokens: Optional[torch.Tensor] = None,
#     use_qk_l2norm_in_kernel: bool = False,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     B, T, H, K, V = *k.shape, v.shape[-1]
#     HV = v.shape[2]
#     N = B if cu_seqlens is None else len(cu_seqlens) - 1
#     BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
#     NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
#     assert NK == 1, "NK > 1 is not supported yet"
#     num_stages = 3
#     num_warps = 1

#     o = q.new_empty(NK, *v.shape)
#     if inplace_final_state:
#         final_state = initial_state
#     else:
#         final_state = q.new_empty(T, HV, K, V, dtype=initial_state.dtype)

#     stride_init_state_token = initial_state.stride(0)
#     stride_final_state_token = final_state.stride(0)

#     if ssm_state_indices is None:
#         stride_indices_seq, stride_indices_tok = 1, 1
#     elif ssm_state_indices.ndim == 1:
#         stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
#     else:
#         stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

#     # print("N: ", N)
#     # print("T: ", T)
#     # print("B: ", B)
#     # print("H: ", H)
#     # print("HV: ", HV)
#     # print("K: ", K)
#     # print("V: ", V)
#     # print("BK: ", BK)
#     # print("BV: ", BV)

#     grid = (NK, NV, N * HV)
#     fused_recurrent_gated_delta_rule_fwd_kernel[grid](
#         q=q,
#         k=k,
#         v=v,
#         g=g,
#         beta=beta,
#         o=o,
#         h0=initial_state,
#         ht=final_state,
#         cu_seqlens=cu_seqlens,
#         ssm_state_indices=ssm_state_indices,
#         num_accepted_tokens=num_accepted_tokens,
#         scale=scale,
#         N=N,
#         T=T,
#         B=B,
#         H=H,
#         HV=HV,
#         K=K,
#         V=V,
#         BK=BK,
#         BV=BV,
#         stride_init_state_token=stride_init_state_token,
#         stride_final_state_token=stride_final_state_token,
#         stride_indices_seq=stride_indices_seq,
#         stride_indices_tok=stride_indices_tok,
#         IS_BETA_HEADWISE=beta.ndim == v.ndim,
#         USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
#         INPLACE_FINAL_STATE=inplace_final_state,
#         num_warps=num_warps,
#         num_stages=num_stages,
#     )
#     o = o.squeeze(0)
#     return o, final_state


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