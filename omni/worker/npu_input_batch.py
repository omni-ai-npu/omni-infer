# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch
from vllm.v1.worker.gpu_input_batch import InputBatch

class NPUInputBatch(InputBatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_penalty_cache = True
        self.disable_multi_mtp_cache = True

    def init_penalty_cache(self, vocab_size: int, device):
        """Called dynamically after vLLM initializes the base class"""
        self.disable_penalty_cache = False

        self.vocab_size = vocab_size
        self.npu_device = device
        
        self.prompt_mask = torch.zeros((self.max_num_reqs, vocab_size), dtype=torch.bool, device=self.npu_device)
        self.output_mask = torch.zeros((self.max_num_reqs, vocab_size), dtype=torch.bool, device=self.npu_device)
        self.output_bin_counts = torch.zeros((self.max_num_reqs, vocab_size), dtype=torch.int32, device=self.npu_device)

        # Instantly attach to the existing metadata created by super().__init__
        self.sampling_metadata.npu_input_batch = self
    
    def init_target_model_hidden_states_cache(
            self, n_predict, hidden_size, dtype, device,
        ):
        self.disable_multi_mtp_cache = False
        self.target_model_hidden_states_cache = torch.zeros(
            (self.max_num_reqs, n_predict + 1, hidden_size),
            dtype=dtype, device=device,
        )
        self.target_token_ids_cache = torch.zeros(
            (self.max_num_reqs, n_predict + 1),
            dtype=torch.int32, device=device,
        )

    def _make_sampling_metadata(self):
        # Override so V1's internal metadata refreshes retain our reference
        metadata = super()._make_sampling_metadata()
        metadata.npu_input_batch = self
        return metadata

    def add_request(self, request):
        # Call super first, which returns the index where the request was placed
        req_index = super().add_request(request)
        self._add_request_for_penalty_cache(request, req_index)
        self._add_request_for_multi_mtp(req_index)
        return req_index

    def _add_request_for_multi_mtp(self, req_index):
        if self.disable_multi_mtp_cache:
            return
        self.target_model_hidden_states_cache[req_index].zero_()
        self.target_token_ids_cache[req_index].zero_()

    def _add_request_for_penalty_cache(self, request, req_index):
        if self.disable_penalty_cache:
            return
        
        # Zero out old penalty states for this reused slot
        self.output_bin_counts[req_index].zero_()
        self.output_mask[req_index].zero_()

        # Scatter the prompt token ids
        prompt_mask_cpu = torch.zeros(self.vocab_size, dtype=torch.bool, device='cpu')
        if request.prompt_token_ids is not None:
            prompt_token_ids_tensor = torch.tensor(request.prompt_token_ids, dtype=torch.int64)
            prompt_mask_cpu[prompt_token_ids_tensor] = True
        self.prompt_mask[req_index].copy_(prompt_mask_cpu, non_blocking=True)

        # If the request is resuming and already has output tokens, restore them!
        if request.output_token_ids:
            # Filter out -1 padding tokens
            valid_outs = [t for t in request.output_token_ids if t >= 0]
            if valid_outs:
                outs_tensor = torch.tensor(valid_outs, dtype=torch.int64)
                # Count frequencies on CPU and copy to NPU
                counts_cpu = torch.bincount(outs_tensor, minlength=self.vocab_size).to(torch.int32)
                mask_cpu = counts_cpu > 0
                self.output_bin_counts[req_index].copy_(counts_cpu, non_blocking=True)
                self.output_mask[req_index].copy_(mask_cpu, non_blocking=True)

    def condense(self) -> None:
        # Track the number of moves *before* super() executes
        num_prior_moves = len(self.batch_update_builder.moved)
        super().condense()
        self._condense_penalty_cache(num_prior_moves)
        self._condense_target_model_hidden_states_cache(num_prior_moves)

    def _condense_target_model_hidden_states_cache(self, num_prior_moves):
        if self.disable_multi_mtp_cache:
            return
        for last_req_index, empty_index, _ in self.batch_update_builder.moved[num_prior_moves:]:
            self.target_model_hidden_states_cache[empty_index] = self.target_model_hidden_states_cache[last_req_index]
            self.target_model_hidden_states_cache[last_req_index].zero_()
            self.target_token_ids_cache[empty_index] = self.target_token_ids_cache[last_req_index]
            self.target_token_ids_cache[last_req_index].zero_()

    def _condense_penalty_cache(self, num_prior_moves):
        if self.disable_penalty_cache:
            return
        # V1 records moves sequentially. Shift masks for any new slot moves
        for last_req_index, empty_index, _ in self.batch_update_builder.moved[num_prior_moves:]:
            self.prompt_mask[empty_index] = self.prompt_mask[last_req_index]
            self.output_mask[empty_index] = self.output_mask[last_req_index]
            self.output_bin_counts[empty_index] = self.output_bin_counts[last_req_index]

            # Clear the old vacated slot
            self.prompt_mask[last_req_index].zero_()
            self.output_mask[last_req_index].zero_()
            self.output_bin_counts[last_req_index].zero_()

    def swap_states(self, i1: int, i2: int) -> None:
        super().swap_states(i1, i2)
        self._swap_states_for_penalty_cache(i1, i2)
        self._swap_states_for_target_model_hidden_states_cache(i1, i2)

    def _swap_states_for_target_model_hidden_states_cache(self, i1: int, i2: int):
        if self.disable_multi_mtp_cache:
            return
        tmp_hidden = self.target_model_hidden_states_cache[i1].clone()
        self.target_model_hidden_states_cache[i1] = self.target_model_hidden_states_cache[i2]
        self.target_model_hidden_states_cache[i2] = tmp_hidden
        tmp_token = self.target_token_ids_cache[i1].clone()
        self.target_token_ids_cache[i1] = self.target_token_ids_cache[i2]
        self.target_token_ids_cache[i2] = tmp_token

    def _swap_states_for_penalty_cache(self, i1: int, i2: int):
        if self.disable_penalty_cache:
            return

        # V1 sometimes swaps active requests. Swap our NPU penalty tensors to match.
        tmp_prompt = self.prompt_mask[i1].clone()
        self.prompt_mask[i1] = self.prompt_mask[i2]
        self.prompt_mask[i2] = tmp_prompt

        tmp_output = self.output_mask[i1].clone()
        self.output_mask[i1] = self.output_mask[i2]
        self.output_mask[i2] = tmp_output

        tmp_counts = self.output_bin_counts[i1].clone()
        self.output_bin_counts[i1] = self.output_bin_counts[i2]
        self.output_bin_counts[i2] = tmp_counts

    def update_sampled_tokens(self, sampled_token_ids: torch.Tensor):
        if sampled_token_ids is None or sampled_token_ids.numel() == 0:
            return
            
        flat_tokens = sampled_token_ids.view(-1).to(self.npu_device)
        valid_mask = (flat_tokens >= 0).to(torch.int32)
        valid_mask_bool = valid_mask.to(torch.bool)

        safe_tokens = flat_tokens.clamp(min=0)
        batch_indices = torch.arange(flat_tokens.shape[0], device=self.npu_device)

        self.output_bin_counts[batch_indices, safe_tokens] += valid_mask
        self.output_mask[batch_indices, safe_tokens] = self.output_mask[batch_indices, safe_tokens] | valid_mask_bool
