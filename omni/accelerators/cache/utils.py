import numpy as np
from dataclasses import dataclass
import torch
from vllm.v1.worker.block_table import BlockTable
from .kv_cache_interface import OmniAttentionSpec
from omni.models.config_loader.loader import model_extra_config


@dataclass
class OmniAttnDecodeInput:
    block_table: torch.Tensor = None
    block_table_cpu: torch.Tensor = None
    block_table_np: np.ndarray = None

    slots: torch.Tensor = None
    slots_cpu: torch.Tensor = None
    slots_np: np.ndarray = None

    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_np: torch.Tensor = None


_omni_input: OmniAttnDecodeInput = None


def _initialize_tensors(graph_block_table_shape: tuple[int, ...], device: torch.device) -> OmniAttnDecodeInput:
    graph_size = graph_block_table_shape[0]

    block_table = torch.zeros(graph_block_table_shape, device=device, dtype=torch.int32)
    block_table_cpu = torch.zeros(graph_block_table_shape, device='cpu', dtype=torch.int32, pin_memory=True)
    block_table_np = block_table_cpu.numpy()

    slots = -torch.ones(graph_size, dtype=torch.int64, device=device)
    slots_cpu = -torch.ones(graph_size, dtype=torch.int64, device='cpu', pin_memory=True)
    slots_np = slots_cpu.numpy()

    seq_lens = torch.zeros(graph_size, dtype=torch.int64, device=device)
    seq_lens_cpu = torch.zeros(graph_size, dtype=torch.int64, device='cpu', pin_memory=True)
    seq_lens_np = seq_lens_cpu.numpy()

    return OmniAttnDecodeInput(
        block_table=block_table,
        block_table_cpu=block_table_cpu,
        block_table_np=block_table_np,
        slots=slots,
        slots_cpu=slots_cpu,
        slots_np=slots_np,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_np=seq_lens_np,
    )


def compute_omni_attn_metadata(
    kv_cache_spec: OmniAttentionSpec,
    block_table: BlockTable,
    num_decodes: int,
    num_decode_tokens: int,
    num_prefills: int,
    prompt_lens: np.ndarray,
    seq_lens: np.ndarray,
    device: torch.device,
    graph_block_table_shape: tuple[int, ...],
):
    if model_extra_config.operator_opt_config.enable_omni_attn_optimization:
        global _omni_input
        if _omni_input is None:
            _omni_input = _initialize_tensors(graph_block_table_shape, device)

        if num_prefills > 0:
            raise RuntimeError("Only support PD disaggregation and only enabled in D now.")
        if num_decode_tokens % num_decodes != 0:
            raise RuntimeError(f"num_decode_tokens is {num_decode_tokens} while num_decodes is {num_decodes}")

        num_reqs = num_decodes + num_prefills
        tokens_per_req = num_decode_tokens // num_decodes
        # TODO: should change to numpy array for faster performance
        block_table_np = block_table.get_numpy_array()[:num_reqs]  # (batch_size, max_blocks)
        sink, recent, block_size = (
            kv_cache_spec.sink,
            kv_cache_spec.recent,
            kv_cache_spec.block_size
        )

        # repeat prompt_lens and seq_lens for each speculative token
        if tokens_per_req > 1:
            prompt_lens = np.repeat(prompt_lens, tokens_per_req)
            seq_len_per_token = np.repeat(seq_lens, tokens_per_req)
        else:
            seq_len_per_token = seq_lens
        bases = np.maximum(sink+recent, prompt_lens) + 1

        # [0,1,2,3,4,5,...]
        token_ids = np.arange(num_decode_tokens)
        # [0,1,0,1,0,1,...]
        spec_ids = token_ids % tokens_per_req
        # [0,0,1,1,2,2,...]
        req_ids = token_ids // tokens_per_req

        # [L1,L1,L2,L2,L3,L3,...] -> [L1-1,L1,L2-1,L2,L3-1,L3,...]
        seq_len_per_token = seq_len_per_token + spec_ids + 1 - tokens_per_req
        mask = (seq_len_per_token > sink + recent)
        window_pos = (seq_len_per_token - bases) % recent + sink
        window_pos = np.where(mask, window_pos, seq_len_per_token-1)
        block_num_per_token, block_offset_per_token = window_pos // block_size, window_pos % block_size
        # slot_mapping = block_table_np[req_ids, block_num_per_token] * block_size + block_offset_per_token
        _omni_input.slots_np.fill(-1)
        np.add(block_table_np[req_ids, block_num_per_token] * block_size, block_offset_per_token, out=_omni_input.slots_np[:num_decode_tokens])
        _omni_input.slots.copy_(_omni_input.slots_cpu, non_blocking=True)

        if tokens_per_req > 1:
            # [1,3,5,...]
            last_token_idx = np.arange(tokens_per_req-1, num_decode_tokens, tokens_per_req)
            block_num_last_token = block_num_per_token[last_token_idx]
            mask = (seq_lens > sink + recent)
            block_shifts = np.where(mask, kv_cache_spec.max_num_blocks-block_num_last_token-1, 0)

            # shift block table rows
            m, n = block_table_np.shape
            rows, cols = np.indices((m, n))
            cols = (cols - block_shifts[:, None]) % n
            # block_table_np = block_table_np[rows, cols]

            # cpu->npu
            _omni_input.block_table_np.fill(0)
            _omni_input.block_table_np[:m*tokens_per_req, :n] = np.repeat(block_table_np[rows, cols], repeats=tokens_per_req, axis=0)
            _omni_input.block_table.copy_(_omni_input.block_table_cpu, non_blocking=True)
            graph_block_table = _omni_input.block_table

            # compute seq_lens
            last_token_pos = window_pos[last_token_idx]
            seq_len_last_token = block_shifts * block_size + last_token_pos + 1
            # omni_attn_seq_lens = (seq_len_last_token[:, None] + np.arange(-tokens_per_req+1, 1)).flatten()
            _omni_input.seq_lens_np.fill(0)
            np.add(seq_len_last_token[:, None], np.arange(-tokens_per_req+1, 1),
                out=_omni_input.seq_lens_np[:num_decode_tokens].reshape(-1, tokens_per_req))
            _omni_input.seq_lens.copy_(_omni_input.seq_lens_cpu, non_blocking=True)
            omni_attn_seq_lens = _omni_input.seq_lens
        else:
            graph_block_table = block_table.get_device_tensor()[:graph_block_table_shape[0]]
            omni_attn_seq_lens = None

        return (
            graph_block_table,
            _omni_input.slots,
            omni_attn_seq_lens,
        )
    else:
        if num_prefills > 0:
            raise RuntimeError("Only support PD disaggregation and only enabled in D now.")
        if num_decode_tokens % num_decodes != 0:
            raise RuntimeError(f"num_decode_tokens is {num_decode_tokens} while num_decodes is {num_decodes}")

        num_reqs = num_decodes + num_prefills
        tokens_per_req = num_decode_tokens // num_decodes
        # TODO: should change to numpy array for faster performance
        block_table_np = block_table.get_numpy_array()[:num_reqs]  # (batch_size, max_blocks)
        sink, recent, block_size = (
            kv_cache_spec.sink,
            kv_cache_spec.recent,
            kv_cache_spec.block_size
        )

        # repeat prompt_lens and seq_lens for each speculative token
        if tokens_per_req > 1:
            prompt_lens = np.repeat(prompt_lens, tokens_per_req)
            seq_len_per_token = np.repeat(seq_lens, tokens_per_req)
        else:
            seq_len_per_token = seq_lens
        bases = np.maximum(sink+recent, prompt_lens) + 1

        # [0,1,2,3,4,5,...]
        token_ids = np.arange(num_decode_tokens)
        # [0,1,0,1,0,1,...]
        spec_ids = token_ids % tokens_per_req
        # [0,0,1,1,2,2,...]
        req_ids = token_ids // tokens_per_req

        # [L1,L1,L2,L2,L3,L3,...] -> [L1-1,L1,L2-1,L2,L3-1,L3,...]
        seq_len_per_token = seq_len_per_token + spec_ids + 1 - tokens_per_req
        mask = (seq_len_per_token > sink + recent)
        window_pos = (seq_len_per_token - bases) % recent + sink
        window_pos = np.where(mask, window_pos, seq_len_per_token-1)
        block_num_per_token, block_offset_per_token = window_pos // block_size, window_pos % block_size
        slot_mapping = block_table_np[req_ids, block_num_per_token] * block_size + block_offset_per_token

        if tokens_per_req > 1:
            # [1,3,5,...]
            last_token_idx = np.arange(tokens_per_req-1, num_decode_tokens, tokens_per_req)
            block_num_last_token = block_num_per_token[last_token_idx]
            mask = (seq_lens > sink + recent)
            block_shifts = np.where(mask, kv_cache_spec.max_num_blocks-block_num_last_token-1, 0)

            # shift block table rows
            m, n = block_table_np.shape
            rows, cols = np.indices((m, n))
            cols = (cols - block_shifts[:, None]) % n
            block_table_np = block_table_np[rows, cols]

            # compute seq_lens
            last_token_pos = window_pos[last_token_idx]
            seq_len_last_token = block_shifts * block_size + last_token_pos + 1
            omni_attn_seq_lens = (seq_len_last_token[:, None] + np.arange(-tokens_per_req+1, 1)).flatten()
        else:
            omni_attn_seq_lens = None

        return (
            torch.from_numpy(block_table_np).to(torch.int32),
            torch.from_numpy(slot_mapping).to(torch.int64),
            torch.from_numpy(omni_attn_seq_lens).to(torch.int64) if omni_attn_seq_lens is not None else None,
        )


def to_bool_or_raise(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val == 1
    if isinstance(val, str):
        return val.lower() in ["1", "true"]
    raise ValueError(f"Cannot convert variable to bool. Type {type(val)}. Value {val}.")
