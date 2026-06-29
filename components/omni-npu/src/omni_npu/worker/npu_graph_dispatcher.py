# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from itertools import product

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher


class NPUGraphDispatcher(CudagraphDispatcher):

    def _create_padded_batch_descriptor(
        self, num_tokens: int, uniform_decode: bool, has_lora: bool
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self.vllm_config.pad_for_cudagraph(num_tokens)

        if (uniform_decode 
            and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
            and num_tokens_padded % uniform_decode_query_len == 0
        ):
            num_reqs = num_tokens_padded // uniform_decode_query_len
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
        )

    def _relax_batch_descriptor_for_mixed_batch_cudagraphs(self, batch_descriptor):
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs

        return BatchDescriptor(
            batch_descriptor.num_tokens,
            num_reqs=min(max_num_seqs, batch_descriptor.num_tokens),
            uniform=False,
            has_lora=batch_descriptor.has_lora
        )
    

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int
    ):
        # This should be called only after attention backend is initialized. So we can
        # get the correct cudagraph mode after backend support is resolved.
        self.cudagraph_mode = cudagraph_mode

        # LoRA activation cases to specialize the cuda graphs on
        if self.vllm_config.lora_config:
            if self.compilation_config.cudagraph_specialize_lora:
                lora_cases = [True, False]
            else:
                lora_cases = [True]
        else:
            lora_cases = [False]

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            for bs, has_lora in product(
                self.compilation_config.cudagraph_capture_sizes, lora_cases
            ):
                # Adapt start
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    self._relax_batch_descriptor_for_mixed_batch_cudagraphs(
                        self._create_padded_batch_descriptor(
                        bs, False, has_lora
                    )),
                )
                # Adapt end

        # if decode cudagraph mode is FULL, and we don't already have mixed
        # mode full cudagraphs then add them here.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, has_lora in product(cudagraph_capture_sizes_for_decode, lora_cases):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(bs, True, has_lora),
                )

        self.keys_initialized = True

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        disable_full: bool = False,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """
        Given conditions(e.g.,batch descriptor and if using cascade attention),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).
        """
        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or num_tokens > self.compilation_config.max_cudagraph_capture_size
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, uniform_decode, has_lora
        )
        # Adapt start
        relaxed_batch_desc = self._relax_batch_descriptor_for_mixed_batch_cudagraphs(batch_desc)
        # Adapt end
        if not disable_full:
            # check if key exists for full cudagraph
            if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc

            # otherwise, check if the relaxed key exists
            if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, relaxed_batch_desc

        # also check if the relaxed key exists for more "general"
        # piecewise cudagraph
        if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, relaxed_batch_desc

        # finally, just return no cudagraphs and a trivial batch descriptor
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)