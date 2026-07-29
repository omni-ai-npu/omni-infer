import torch

from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    split_decodes_and_prefills,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def _mask_padded_state_indices(
    state_indices: torch.Tensor | None,
    query_start_loc_cpu: torch.Tensor,
) -> torch.Tensor | None:
    if state_indices is None:
        return None
    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    padded_mask_cpu = query_lens_cpu == 0
    if not bool(padded_mask_cpu.any()):
        return state_indices

    padded_mask = padded_mask_cpu.to(state_indices.device, non_blocking=True)
    if padded_mask.shape[0] != state_indices.shape[0]:
        padded_mask = padded_mask[: state_indices.shape[0]]
    state_indices = state_indices.clone()
    if state_indices.ndim == 1:
        state_indices[padded_mask] = PAD_SLOT_ID
    else:
        state_indices[padded_mask, :] = PAD_SLOT_ID
    return state_indices


@register_patch("GDNAttentionMetadataBuilderPatch", GDNAttentionMetadataBuilder)
class GDNAttentionMetadataBuilderPatch(VLLMPatch):
    _attr_names_to_apply = [
        'build',
    ]

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None

        spec_sequence_masks_cpu = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or int(num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0].sum()) == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = int(spec_sequence_masks_cpu.sum())
            spec_sequence_masks = spec_sequence_masks_cpu.to(
                query_start_loc.device, non_blocking=True
            )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_padded = 0
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = _mask_padded_state_indices(
                m.block_table_tensor[:, 0],
                m.query_start_loc_cpu[: m.num_reqs + 1],
            )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = []
            for loc in m.query_start_loc_cpu[: m.num_reqs + 1]:
                non_spec_query_start_loc_cpu.append(int(loc))
            non_spec_seqlens_list = []
            for start, end in zip(
                non_spec_query_start_loc_cpu,
                non_spec_query_start_loc_cpu[1:],
            ):
                non_spec_seqlens_list.append(end - start)
            num_accepted_tokens = None
        else:
            if spec_sequence_masks_cpu is None:
                raise ValueError("spec_sequence_masks_cpu must be provided for speculative batches.")
            query_lens_cpu = (
                m.query_start_loc_cpu[1:] - m.query_start_loc_cpu[:-1]
            )
            query_lens = query_start_loc[1:] - query_start_loc[:-1]

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            non_spec_seqlens_list = []
            for seqlen in non_spec_query_lens_cpu:
                non_spec_seqlens_list.append(int(seqlen))
            non_spec_query_start_loc_cpu = [0]
            for seqlen in non_spec_seqlens_list:
                non_spec_query_start_loc_cpu.append(
                    non_spec_query_start_loc_cpu[-1] + seqlen
                )

            num_padded = int((non_spec_query_lens_cpu == 0).sum())
            num_decodes = int((non_spec_query_lens_cpu == 1).sum())
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_padded
            num_decode_tokens = num_decodes
            num_prefill_tokens = int(non_spec_query_lens_cpu.sum()) - num_decode_tokens
            num_spec_decode_tokens = (
                int(query_lens_cpu.sum()) - num_prefill_tokens - num_decode_tokens
            )
            num_spec_decode_tokens += num_padded * (self.num_spec + 1)

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = num_spec_decode_tokens
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                spec_state_indices_tensor = _mask_padded_state_indices(
                    m.block_table_tensor[:, : self.num_spec + 1],
                    m.query_start_loc_cpu[: m.num_reqs + 1],
                )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = m.block_table_tensor[
                    spec_sequence_masks, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = m.block_table_tensor[
                    ~spec_sequence_masks, 0
                ]
                query_lens_cpu = (
                    m.query_start_loc_cpu[1: m.num_reqs + 1]
                    - m.query_start_loc_cpu[: m.num_reqs]
                )
                spec_state_indices_tensor = _mask_padded_state_indices(
                    spec_state_indices_tensor,
                    torch.cat(
                        [
                            query_lens_cpu[spec_sequence_masks_cpu].new_zeros(1),
                            query_lens_cpu[spec_sequence_masks_cpu].cumsum(0),
                        ]
                    ),
                )
                non_spec_state_indices_tensor = _mask_padded_state_indices(
                    non_spec_state_indices_tensor,
                    torch.cat(
                        [
                            query_lens_cpu[~spec_sequence_masks_cpu].new_zeros(1),
                            query_lens_cpu[~spec_sequence_masks_cpu].cumsum(0),
                        ]
                    ),
                )

                spec_query_start_loc = torch.empty(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                spec_query_start_loc[0] = 0
                torch.cumsum(
                    query_lens[spec_sequence_masks], dim=0, out=spec_query_start_loc[1:]
                )
                non_spec_query_start_loc = torch.empty(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_query_start_loc[0] = 0
                torch.cumsum(
                    query_lens[~spec_sequence_masks],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )

            if num_accepted_tokens is None:
                raise ValueError("num_accepted_tokens must be provided for speculative batches.")
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(non_spec_query_start_loc)
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        spec_cudagraph_can_pad = (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )
        if spec_cudagraph_can_pad:
            spec_batch_size = num_spec_decodes + num_padded
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor[:num_spec_decodes], non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[
                :spec_batch_size
            ]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:spec_batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            if non_spec_token_indx is None or spec_token_indx is None:
                raise ValueError("token indices must be available before cudagraph padding.")
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc[: num_spec_decodes + 1], non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[:batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1:].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens[:num_spec_decodes], non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:spec_batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        decode_cudagraph_can_pad = (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        )
        if decode_cudagraph_can_pad:
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1:].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes + num_padded,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        if num_prefills > 0:
            attn_metadata.non_spec_query_start_loc_cpu = non_spec_query_start_loc_cpu
            attn_metadata.non_spec_seqlens_list = non_spec_seqlens_list
        return attn_metadata